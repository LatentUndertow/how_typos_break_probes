#!/usr/bin/env python3
"""Run all §5 (activation-level) experiments for a chosen model.

Loads the model **once** in-process and dispatches each experiment via its
``run(model, tokenizer, device, ...)`` entry point. Falls back to a subprocess
for two scripts that still wrap the model in ``ActivationModelHF``
(``typo_decay_preamble_control`` and ``punct_decay_control``); those scripts
load their own model — refactoring them into the in-process pattern requires
adding a ``from_existing`` factory to ``ActivationModelHF`` (deferred).

Usage
-----
    python run_section_5.py --model llama --n-prompts 100
    python run_section_5.py --model qwen3
    python run_section_5.py --model gemma --n-prompts 100

Notes
-----
- ``--n-prompts`` is honored for the Llama scripts.
- Qwen3 / Gemma orchestrators (``run_all_qwen3.py``, ``run_all_gemma4.py``)
  use hardcoded N inside each experiment function (paper N).
- The four refactored Llama scripts share a single model load:
  ``per_layer_same_pos``, ``multi_typo_vanilla``, ``multi_perturbation_eot``,
  ``eot_baseline_variance``. The remaining two pay their own load.
"""
import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
EXP = REPO / "activation_robustness" / "experiments"


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def _subproc(cmd) -> None:
    cmd = [str(c) for c in cmd]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    print(f"  → exit {rc} ({time.time() - t0:.1f}s)", flush=True)
    if rc != 0:
        sys.exit(f"\nFAILED: {' '.join(cmd)}")


def run_llama(n: int) -> None:
    """One model load for the four refactored scripts; subprocess the
    two ``ActivationModelHF``-using scripts."""
    import torch
    from activation_robustness.analysis.extraction import load_hf_model
    from activation_robustness.experiments import (
        per_layer_same_pos,
        multi_typo_vanilla,
        multi_perturbation_eot,
        eot_baseline_variance,
    )

    _section(f"Loading meta-llama/Llama-3.1-8B-Instruct (one load for 4 experiments)")
    t0 = time.time()
    model, tokenizer = load_hf_model("meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16")
    device = next(model.parameters()).device
    print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)

    _section("per_layer_same_pos (in-process)")
    per_layer_same_pos.run(model, tokenizer, device, n_prompts=n)

    _section("multi_typo_vanilla (in-process)")
    multi_typo_vanilla.run(model, tokenizer, device, n_prompts=n)

    _section("multi_perturbation_eot (in-process)")
    multi_perturbation_eot.run(model, tokenizer, device, n_per_bucket=n)

    _section("eot_baseline_variance (in-process)")
    eot_baseline_variance.run(model, tokenizer, device)

    # Free model before subprocessing the ActivationModelHF scripts.
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("\n  In-process model freed before subprocess phase", flush=True)

    _section("typo_decay_preamble_control (subprocess — uses ActivationModelHF)")
    _subproc([sys.executable, EXP / "typo_decay_preamble_control.py", "--n-prompts", n])

    _section("punct_decay_control (subprocess — uses ActivationModelHF)")
    # punct_decay_control needs ≥ 20 to populate the question-prompt filter.
    _subproc([sys.executable, EXP / "punct_decay_control.py", "--n-prompts", max(n, 20)])


def run_qwen3() -> None:
    _section("Qwen3-8B per_layer_same_pos")
    _subproc([sys.executable, EXP / "run_all_qwen3.py", "--only", "per_layer_same_pos"])
    _section("Qwen3-8B long_decay")
    _subproc([sys.executable, EXP / "run_all_qwen3.py", "--only", "long_decay"])


def run_gemma() -> None:
    _section("Gemma-4-E4B-it per_layer_same_pos")
    _subproc([sys.executable, EXP / "run_all_gemma4.py", "--only", "per_layer_same_pos"])
    _section("Gemma-4-E4B-it long_decay")
    _subproc([sys.executable, EXP / "run_all_gemma4.py", "--only", "long_decay"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--model", choices=["llama", "qwen3", "gemma"], required=True)
    ap.add_argument("--n-prompts", type=int, default=100,
                    help="prompts per condition (Llama only). Default 100.")
    args = ap.parse_args()

    print(f"=== §5 reproduction: model={args.model}  N={args.n_prompts} ===")
    t0 = time.time()
    if args.model == "llama":
        run_llama(args.n_prompts)
    elif args.model == "qwen3":
        run_qwen3()
    elif args.model == "gemma":
        run_gemma()
    print(f"\n✓ All §5 experiments complete for {args.model} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
