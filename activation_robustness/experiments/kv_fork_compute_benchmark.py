#!/usr/bin/env python3
"""
KV-fork compute overhead benchmark with prompt-length bucketing.

Measures absolute per-request latency overhead Δms = (KV-fork) - (baseline)
across length buckets to expose how the +30-token suffix scales with
baseline prompt length. Attention is O(n^2), so adding 30 tokens to a
baseline of n is expected to scale as O(60n + 900) — linear in n.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

SUFFIX_NEUTRAL = (
    "Before responding, take a moment to carefully reflect on the message above. "
    "Make sure that your answer is complete, accurate, and clearly expressed throughout."
)

GEN_PROMPT = "<|start_header_id|>assistant<|end_header_id|>\n\n"

# Target buckets (post chat-template tokens, approximately)
BUCKETS = {
    "short": 60,
    "medium": 160,
    "long": 500,
    "very_long": 1500,
}

# A realistic core prompt + filler text we can tile to hit target lengths.
CORE_PROMPT = (
    "Can you help me write a Python function that takes a list of integers "
    "and returns the second-largest element? I want it to handle edge cases "
    "like duplicates and lists with fewer than two elements."
)

FILLER = (
    "I'd also like to understand the time and space complexity. Please "
    "include type hints, a clear docstring with examples, and brief commentary "
    "comparing different algorithmic approaches. If there are tradeoffs "
    "between readability and performance, I want to know which approach you "
    "would recommend for production code review and why. Consider whether "
    "using built-in sorting versus a single-pass two-tracker approach makes "
    "more sense in different contexts. "
)


def make_user_msg(target_tokens: int, tokenizer) -> tuple[str, int]:
    """Pad CORE_PROMPT with FILLER until tokenized chat template reaches target."""
    body = CORE_PROMPT
    while True:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
        n = len(tokenizer(text)["input_ids"])
        if n >= target_tokens or len(body) > 50000:
            return body, n
        body += " " + FILLER


def build_text(tokenizer, body: str, with_fork: bool) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if with_fork:
        if text.endswith(GEN_PROMPT):
            text = text[: -len(GEN_PROMPT)]
        text += (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SUFFIX_NEUTRAL}<|eot_id|>"
        )
    return text


def time_forward(model, input_ids, attention_mask, n_warmup=3, n_reps=10):
    """Median (ms) and std (ms) of forward-pass time."""
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.std(times))


def main():
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()

    bucket_bodies = {}
    print("\n=== Bucket prep ===", flush=True)
    for name, target in BUCKETS.items():
        body, n = make_user_msg(target, tokenizer)
        bucket_bodies[name] = (body, n)
        print(f"  {name:>10}: target {target:>4} → got {n} tokens", flush=True)

    # Compute analytic suffix-only token count
    suffix_tokens = (
        len(tokenizer(build_text(tokenizer, "x", True))["input_ids"])
        - len(tokenizer(build_text(tokenizer, "x", False))["input_ids"])
    )
    print(f"  suffix delta: +{suffix_tokens} tokens (independent of bucket)", flush=True)

    print("\n=== Latency / throughput ===", flush=True)
    print(
        f"{'bucket':>10} {'n_base':>7} {'batch':>5} "
        f"{'base_ms':>9} {'fork_ms':>9} {'Δms':>8} {'Δms_std':>8} "
        f"{'rps_base':>9} {'rps_fork':>9}",
        flush=True,
    )

    results = {
        "model": MODEL_NAME,
        "suffix_tokens": int(suffix_tokens),
        "buckets": {},
        "latency": [],
    }

    for bucket_name, (body, n_base) in bucket_bodies.items():
        results["buckets"][bucket_name] = int(n_base)
        for batch_size in [1, 8, 32]:
            row = {
                "bucket": bucket_name,
                "n_base": int(n_base),
                "batch": batch_size,
            }
            # Baseline
            text_base = build_text(tokenizer, body, with_fork=False)
            enc = tokenizer(
                [text_base] * batch_size, return_tensors="pt", padding=True
            ).to("cuda")
            base_med, base_sd = time_forward(model, enc.input_ids, enc.attention_mask)
            # KV-fork
            text_fork = build_text(tokenizer, body, with_fork=True)
            enc = tokenizer(
                [text_fork] * batch_size, return_tensors="pt", padding=True
            ).to("cuda")
            fork_med, fork_sd = time_forward(model, enc.input_ids, enc.attention_mask)

            delta_med = fork_med - base_med
            # Conservative std for the delta: sqrt(s_fork^2 + s_base^2)
            delta_sd = float(np.sqrt(base_sd ** 2 + fork_sd ** 2))
            rps_base = batch_size / (base_med / 1000.0)
            rps_fork = batch_size / (fork_med / 1000.0)

            row.update(
                {
                    "baseline_ms_median": base_med,
                    "baseline_ms_std": base_sd,
                    "fork_ms_median": fork_med,
                    "fork_ms_std": fork_sd,
                    "delta_ms_median": delta_med,
                    "delta_ms_std": delta_sd,
                    "rps_baseline": rps_base,
                    "rps_fork": rps_fork,
                }
            )
            results["latency"].append(row)

            print(
                f"{bucket_name:>10} {n_base:>7} {batch_size:>5} "
                f"{base_med:>9.2f} {fork_med:>9.2f} "
                f"{delta_med:>8.2f} {delta_sd:>8.2f} "
                f"{rps_base:>9.1f} {rps_fork:>9.1f}",
                flush=True,
            )

    # Analytic KV memory delta (constant — depends only on suffix tokens, model)
    bytes_per_token_kv = 32 * 8 * 128 * 2 * 2  # 32 layers x 8 KV heads (GQA) x 128 dim x 2 (K,V) x 2 bytes (bf16)
    analytic_mb = suffix_tokens * bytes_per_token_kv / (1024 * 1024)
    results["kv_memory_mb_per_request_analytic"] = analytic_mb
    print(f"\nAnalytic KV memory delta per request: {analytic_mb:.2f} MB (suffix-only)")

    out_path = (
        Path(__file__).parent.parent / "results" / "kv_fork" / "compute_benchmark_buckets.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
