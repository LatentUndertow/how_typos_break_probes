#!/usr/bin/env python3
"""
Control: EOT activation baseline variance.

Establishes reference values for how different EOT activations are
across prompts, so we can contextualize the typo perturbation magnitude.

Computes:
1. Between-prompt: cos(h_A, h_B) at EOT for random prompt pairs
2. Activation norms at EOT by prompt length
3. Within-length-bucket variance vs cross-bucket variance

Uses the same length buckets as E-033 for direct comparison.
"""
import sys, os, json, time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from activation_robustness.analysis.extraction import load_hf_model, format_prompt
from activation_robustness.data.external import sample_openorca

os.environ.setdefault("PYTHONUNBUFFERED", "1")

OUTPUT_DIR = _REPO / "activation_robustness" / "results" / "eot_baseline_variance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [8, 16, 24, 31]
SEED = 51

LENGTH_BUCKETS = {
    "short_50-100": (30, 70),
    "medium_200-400": (150, 300),
    "long_800+": (600, 1500),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(model, tokenizer, device, *, n_per_bucket: int = 200):
    """Run the experiment using a pre-loaded model.

    Bit-exact body of the original ``main()`` with model loading factored out.
    Same RNG seed, same ``sample_openorca`` call, same loops.
    """
    rng = np.random.default_rng(SEED)

    log("Sampling prompts...")
    orca = sample_openorca(
        n_per_bucket=n_per_bucket,
        buckets=LENGTH_BUCKETS,
        seed=SEED,
        scout_size=200000,
    )
    log(f"  {len(orca)} total prompts")

    prompts_by_bucket = defaultdict(list)
    for p in orca:
        prompts_by_bucket[p["length_bucket"]].append(p["text"])
    for b, ps in prompts_by_bucket.items():
        log(f"  {b}: {len(ps)} prompts")

    # Extract EOT activations for all prompts at all layers
    eot_acts = {}  # bucket -> list of {layer: vector, seq_len}
    for lb, bucket_prompts in sorted(prompts_by_bucket.items()):
        eot_acts[lb] = []
        log(f"\n  Extracting {lb}...")
        for i, text in enumerate(bucket_prompts):
            fmt = format_prompt(tokenizer, text, steering_prompt=None, add_generation_prompt=False)
            ids = tokenizer(fmt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            per_layer = {L: out.hidden_states[L + 1][0, -1, :].float().cpu().numpy().astype(np.float64)
                         for L in LAYERS}
            seq_len = ids.shape[1]
            eot_acts[lb].append({"h": per_layer, "seq_len": int(seq_len)})
            if (i + 1) % 50 == 0:
                log(f"    [{i+1}/{len(bucket_prompts)}]")

    del model
    torch.cuda.empty_cache()

    # =================================================================
    # ANALYSIS
    # =================================================================
    log("\n=== ANALYSIS ===")

    summary = {}

    # 1. Within-bucket pairwise cosines per layer
    max_pairs = 500
    for L in LAYERS:
        log(f"\nLayer {L} — Within-bucket pairwise cosines:")
        for lb, entries in sorted(eot_acts.items()):
            vecs = [e["h"][L] for e in entries]
            norms = [float(np.linalg.norm(v)) for v in vecs]
            seq_lens = [e["seq_len"] for e in entries]

            n = len(vecs)
            pair_cosines = []
            pair_angles = []
            pair_l2s = []
            n_pairs = min(max_pairs, n * (n - 1) // 2)
            for _ in range(n_pairs):
                i, j = rng.choice(n, size=2, replace=False)
                ni, nj = np.linalg.norm(vecs[i]), np.linalg.norm(vecs[j])
                if ni > 1e-10 and nj > 1e-10:
                    cos = float(np.dot(vecs[i], vecs[j]) / (ni * nj))
                    cos = max(-1.0, min(1.0, cos))
                    pair_cosines.append(cos)
                    pair_angles.append(float(np.degrees(np.arccos(cos))))
                    pair_l2s.append(float(np.linalg.norm(vecs[i] - vecs[j])))

            if lb not in summary:
                summary[lb] = {"n": n, "mean_seq_len": float(np.mean(seq_lens))}
            summary[lb][f"L{L}"] = {
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "within_angle_mean": float(np.mean(pair_angles)),
                "within_angle_std": float(np.std(pair_angles)),
                "within_cos_mean": float(np.mean(pair_cosines)),
                "within_cos_std": float(np.std(pair_cosines)),
                "within_l2_mean": float(np.mean(pair_l2s)),
            }

            log(f"  {lb}: norm={np.mean(norms):.1f}, "
                f"angle={np.mean(pair_angles):.1f}°±{np.std(pair_angles):.1f}, "
                f"cos={np.mean(pair_cosines):.4f}")

    # 2. Cross-bucket pairwise cosines (layer 31 only for simplicity)
    log("\nCross-bucket pairwise cosines (Layer 31):")
    buckets = sorted(eot_acts.keys())
    cross_results = {}
    for bi in range(len(buckets)):
        for bj in range(bi + 1, len(buckets)):
            lb1, lb2 = buckets[bi], buckets[bj]
            vecs1 = [e["h"][31] for e in eot_acts[lb1]]
            vecs2 = [e["h"][31] for e in eot_acts[lb2]]
            pair_cosines = []
            pair_angles = []
            for _ in range(max_pairs):
                i = rng.integers(len(vecs1))
                j = rng.integers(len(vecs2))
                ni = np.linalg.norm(vecs1[i])
                nj = np.linalg.norm(vecs2[j])
                if ni > 1e-10 and nj > 1e-10:
                    cos = float(np.dot(vecs1[i], vecs2[j]) / (ni * nj))
                    cos = max(-1.0, min(1.0, cos))
                    pair_cosines.append(cos)
                    pair_angles.append(float(np.degrees(np.arccos(cos))))

            key = f"{lb1}_vs_{lb2}"
            cross_results[key] = {
                "cos_mean": float(np.mean(pair_cosines)),
                "cos_std": float(np.std(pair_cosines)),
                "angle_mean": float(np.mean(pair_angles)),
                "angle_std": float(np.std(pair_angles)),
            }
            log(f"  {lb1} vs {lb2}: cos={np.mean(pair_cosines):.4f}, angle={np.mean(pair_angles):.1f}°")

    summary["cross_bucket"] = cross_results

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved to {OUTPUT_DIR}")

    # =================================================================
    # PLOTS
    # =================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    len_colors = {
        "short_50-100": "#C44E52",
        "medium_200-400": "#55A868",
        "long_800+": "#4C72B0",
    }

    # Figure 1: Per-layer baseline variance
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, L in enumerate(LAYERS):
        ax = axes[idx // 2, idx % 2]
        for lb, entries in sorted(eot_acts.items()):
            vecs = [e["h"][L] for e in entries]
            n = len(vecs)
            angles = []
            for _ in range(500):
                i, j = rng.choice(n, size=2, replace=False)
                ni, nj = np.linalg.norm(vecs[i]), np.linalg.norm(vecs[j])
                if ni > 1e-10 and nj > 1e-10:
                    cos = float(np.dot(vecs[i], vecs[j]) / (ni * nj))
                    cos = max(-1.0, min(1.0, cos))
                    angles.append(float(np.degrees(np.arccos(cos))))
            ax.hist(angles, bins=30, alpha=0.5, color=len_colors[lb], label=lb, density=True)
        ax.set_xlabel("Angular distance (°)")
        ax.set_ylabel("Density")
        ax.set_title(f"Layer {L}")
        ax.legend(fontsize=7)
    plt.suptitle("Between-Prompt Baseline Variance at EOT — Per Layer", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {OUTPUT_DIR / 'baseline_per_layer.png'}")

    # Figure 2: Cross-layer comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A: Mean angular distance by layer and length
    ax = axes[0]
    for lb in sorted(eot_acts.keys()):
        if lb == "cross_bucket":
            continue
        layer_angles = [summary[lb].get(f"L{L}", {}).get("within_angle_mean", 0) for L in LAYERS]
        ax.plot(LAYERS, layer_angles, "o-", color=len_colors.get(lb, "gray"), markersize=7, lw=2, label=lb)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean between-prompt angle (°)")
    ax.set_title("A. Baseline variance across layers")
    ax.legend(fontsize=8)

    # B: Activation norms by layer and length
    ax = axes[1]
    for lb in sorted(eot_acts.keys()):
        if lb == "cross_bucket":
            continue
        layer_norms = [summary[lb].get(f"L{L}", {}).get("mean_norm", 0) for L in LAYERS]
        ax.plot(LAYERS, layer_norms, "s-", color=len_colors.get(lb, "gray"), markersize=7, lw=2, label=lb)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean activation norm at EOT")
    ax.set_title("B. Activation norms across layers")
    ax.legend(fontsize=8)

    # C: Summary
    ax = axes[2]
    ax.axis("off")
    lines = ["EOT Baseline Variance (per layer)", "─" * 45, ""]
    for L in LAYERS:
        lines.append(f"Layer {L}:")
        for lb in sorted(summary.keys()):
            if lb == "cross_bucket":
                continue
            s = summary[lb].get(f"L{L}", {})
            if s:
                lines.append(f"  {lb}: {s['within_angle_mean']:.1f}° ± {s['within_angle_std']:.1f}")
        lines.append("")
    for i, line in enumerate(lines):
        w = "bold" if i == 0 else "normal"
        sz = 9 if i == 0 else 7.5
        ax.text(0.02, 0.98 - i * 0.04, line, transform=ax.transAxes,
                fontsize=sz, fontweight=w, fontfamily="monospace", va="top")

    plt.suptitle("Control: EOT Baseline Variance — Per Layer", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_variance.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {OUTPUT_DIR / 'baseline_variance.png'}")

    log("Done.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-bucket", type=int, default=200)
    args, _ = ap.parse_known_args()

    log("Loading model...")
    model, tokenizer = load_hf_model("meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16")
    device = next(model.parameters()).device
    log("Model loaded")
    run(model, tokenizer, device, n_per_bucket=args.n_per_bucket)


if __name__ == "__main__":
    main()
