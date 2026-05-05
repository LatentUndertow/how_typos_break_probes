#!/usr/bin/env python3
"""
Quick check: same-position, different-replacement cosines per layer.

At each typo position, extract 3 different adjacent-key replacements.
Compute pairwise cosines of per-layer deltas.
Does any layer show cos → 1.0 for same-position variants?

Small: 30 prompts, 4 positions per prompt, 3 replacements per position.
"""
import sys, os, json, time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from activation_robustness.analysis.extraction import load_hf_model, format_prompt
from activation_robustness.perturbations.typo import AdjacentKey
from activation_robustness.data.external import sample_openorca, sample_malicious

os.environ.setdefault("PYTHONUNBUFFERED", "1")

OUTPUT_DIR = _REPO / "activation_robustness" / "results" / "per_layer_same_pos_100"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = _REPO / "activation_robustness" / "results" / "llama_comparison_summary.json"

SEED = 49
N_LAYERS = 32


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_all_layers(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return [out.hidden_states[i][0].float().cpu().numpy().astype(np.float64)
            for i in range(len(out.hidden_states))]


def run(model, tokenizer, device, *, n_prompts: int = 100, n_malicious: int = 0):
    """Run the experiment using a pre-loaded model.

    Bit-exact body of the original ``main()`` with model loading factored out.
    Same RNG seed, same ``sample_openorca`` / ``sample_malicious`` calls.
    """
    rng = np.random.default_rng(SEED)

    log("Sampling prompts...")
    orca = sample_openorca(n_per_bucket=n_prompts, buckets={"100-300": (100, 300)}, seed=SEED)
    prompts = [{"text": p["text"], "safety": "benign", "source": p.get("source", "openorca")}
               for p in orca[:n_prompts]]

    if n_malicious > 0:
        log(f"Sampling {n_malicious} malicious prompts...")
        mal = sample_malicious(n=n_malicious, seed=SEED + 1)
        prompts.extend({"text": p["text"], "safety": "malicious", "source": p.get("source", "mal")}
                       for p in mal)

    rng.shuffle(prompts)
    log(f"  {len(prompts)} prompts total "
        f"(benign={sum(1 for p in prompts if p['safety']=='benign')}, "
        f"malicious={sum(1 for p in prompts if p['safety']=='malicious')})")

    adj_key = AdjacentKey()

    # Collect: per layer, list of same-position cosines
    same_pos_cosines = {L: [] for L in range(N_LAYERS)}
    # Also collect cross-position for comparison (reuse from same data)
    cross_pos_cosines = {L: [] for L in range(N_LAYERS)}
    # Residual stream comparison
    same_pos_resid = {L: [] for L in range(N_LAYERS)}

    # Per-class buckets (parallel to global lists; same cosines, partitioned)
    SAFETY_CLASSES = ["benign", "malicious"]
    same_pos_by_class = {c: {L: [] for L in range(N_LAYERS)} for c in SAFETY_CLASSES}
    cross_pos_by_class = {c: {L: [] for L in range(N_LAYERS)} for c in SAFETY_CLASSES}
    n_groups_by_class = {c: 0 for c in SAFETY_CLASSES}

    n_groups = 0

    for pi, prompt_obj in enumerate(prompts):
        prompt_text = prompt_obj["text"]
        safety = prompt_obj["safety"]
        orig_fmt = format_prompt(tokenizer, prompt_text, steering_prompt=None, add_generation_prompt=False)
        orig_hs = extract_all_layers(model, tokenizer, orig_fmt, device)
        seq_len = orig_hs[0].shape[0]
        orig_ids = tokenizer.encode(orig_fmt)

        # Get all variants grouped by character position
        variants = adj_key.enumerate_all(prompt_text)
        if not variants:
            continue

        by_char_pos = defaultdict(list)
        for pert_text, meta in variants:
            by_char_pos[meta["position"]].append((pert_text, meta))

        # Find positions with 3+ valid replacements that change exactly 1 token
        valid_groups = []  # list of (token_pos, [(pert_text, meta), ...])

        for char_pos in sorted(by_char_pos.keys()):
            candidates = by_char_pos[char_pos]
            valid_at_pos = []
            token_pos = None

            for pert_text, meta in candidates:
                pf = format_prompt(tokenizer, pert_text, steering_prompt=None, add_generation_prompt=False)
                pi_ids = tokenizer.encode(pf)
                if len(orig_ids) != len(pi_ids):
                    continue
                diffs = [i for i in range(len(orig_ids)) if orig_ids[i] != pi_ids[i]]
                if len(diffs) != 1:
                    continue
                tp = diffs[0]
                if token_pos is None:
                    token_pos = tp
                elif tp != token_pos:
                    continue  # different replacements map to different token positions
                valid_at_pos.append((pert_text, meta))

            if len(valid_at_pos) >= 3 and token_pos is not None:
                valid_groups.append((token_pos, valid_at_pos[:3]))

        # Pick up to 4 evenly spaced positions
        if len(valid_groups) > 4:
            step = len(valid_groups) // 4
            valid_groups = valid_groups[::step][:4]

        if len(valid_groups) < 1:
            continue

        # Extract all variants and compute per-layer deltas
        # group_deltas[group_idx][layer] = list of 3 delta vectors
        group_deltas = []

        for token_pos, replacements in valid_groups:
            layer_deltas = {L: [] for L in range(N_LAYERS)}

            for pert_text, meta in replacements:
                pf = format_prompt(tokenizer, pert_text, steering_prompt=None, add_generation_prompt=False)
                pert_hs = extract_all_layers(model, tokenizer, pf, device)

                if pert_hs[0].shape[0] != seq_len:
                    continue

                for L in range(N_LAYERS):
                    orig_contrib = orig_hs[L + 1][token_pos] - orig_hs[L][token_pos]
                    pert_contrib = pert_hs[L + 1][token_pos] - pert_hs[L][token_pos]
                    layer_delta = pert_contrib - orig_contrib
                    layer_deltas[L].append(layer_delta)

            # Only keep if we got all 3
            if all(len(layer_deltas[L]) == 3 for L in range(N_LAYERS)):
                group_deltas.append(layer_deltas)
                n_groups += 1
                n_groups_by_class[safety] += 1

        # Compute same-position cosines (within each group)
        for gd in group_deltas:
            for L in range(N_LAYERS):
                deltas = gd[L]  # 3 vectors
                for i in range(len(deltas)):
                    for j in range(i + 1, len(deltas)):
                        n1 = np.linalg.norm(deltas[i])
                        n2 = np.linalg.norm(deltas[j])
                        if n1 > 1e-10 and n2 > 1e-10:
                            cos = float(np.dot(deltas[i], deltas[j]) / (n1 * n2))
                            same_pos_cosines[L].append(cos)
                            same_pos_by_class[safety][L].append(cos)

                # Residual stream same-pos cosine
                resid_deltas = []
                # Need to recompute from stored hs... actually just use layer contribution sum
                # Skip this, use the layer contribution cosines

        # Cross-position cosines (between groups)
        if len(group_deltas) >= 2:
            for L in range(N_LAYERS):
                for gi in range(len(group_deltas)):
                    for gj in range(gi + 1, len(group_deltas)):
                        # Pick first replacement from each group
                        d1 = group_deltas[gi][L][0]
                        d2 = group_deltas[gj][L][0]
                        n1 = np.linalg.norm(d1)
                        n2 = np.linalg.norm(d2)
                        if n1 > 1e-10 and n2 > 1e-10:
                            cos = float(np.dot(d1, d2) / (n1 * n2))
                            cross_pos_cosines[L].append(cos)
                            cross_pos_by_class[safety][L].append(cos)

        if (pi + 1) % 10 == 0:
            log(f"  [{pi+1}/{len(prompts)}] groups: {n_groups}")

    del model
    torch.cuda.empty_cache()

    log(f"\nTotal position groups: {n_groups}")

    # =================================================================
    # ANALYSIS
    # =================================================================
    log("\n=== RESULTS ===")
    log(f"{'Layer':>6s} {'same_pos':>10s} {'±std':>8s} {'cross_pos':>10s} {'±std':>8s} {'N_same':>7s} {'N_cross':>8s}")
    log("-" * 65)

    results = {"n_groups": n_groups, "per_layer": {}}

    for L in range(N_LAYERS):
        sp = same_pos_cosines[L]
        cp = cross_pos_cosines[L]
        sp_mean = float(np.mean(sp)) if sp else 0
        sp_std = float(np.std(sp)) if sp else 0
        cp_mean = float(np.mean(cp)) if cp else 0
        cp_std = float(np.std(cp)) if cp else 0

        results["per_layer"][L] = {
            "same_pos_mean": sp_mean, "same_pos_std": sp_std, "n_same": len(sp),
            "cross_pos_mean": cp_mean, "cross_pos_std": cp_std, "n_cross": len(cp),
        }

        log(f"{L:>6d} {sp_mean:>10.4f} {sp_std:>8.4f} {cp_mean:>10.4f} {cp_std:>8.4f} {len(sp):>7d} {len(cp):>8d}")

    # Per-class breakdown (only meaningful if we sampled malicious prompts)
    if any(n_groups_by_class[c] > 0 for c in SAFETY_CLASSES):
        results["n_groups_by_class"] = dict(n_groups_by_class)
        results["per_layer_by_class"] = {c: {} for c in SAFETY_CLASSES}
        results["sample_counts"] = {
            c: sum(1 for p in prompts if p["safety"] == c) for c in SAFETY_CLASSES
        }
        results["sample_counts"]["total"] = len(prompts)
        for c in SAFETY_CLASSES:
            for L in range(N_LAYERS):
                sp = same_pos_by_class[c][L]
                cp = cross_pos_by_class[c][L]
                results["per_layer_by_class"][c][L] = {
                    "same_pos_mean": float(np.mean(sp)) if sp else 0,
                    "same_pos_std": float(np.std(sp)) if sp else 0,
                    "n_same": len(sp),
                    "cross_pos_mean": float(np.mean(cp)) if cp else 0,
                    "cross_pos_std": float(np.std(cp)) if cp else 0,
                    "n_cross": len(cp),
                }
        log("\n=== PER-CLASS SUMMARY (layer 31) ===")
        for c in SAFETY_CLASSES:
            d = results["per_layer_by_class"][c][31]
            log(f"  {c:>10s}: same_pos={d['same_pos_mean']:.4f}±{d['same_pos_std']:.4f} "
                f"(n={d['n_same']}), cross_pos={d['cross_pos_mean']:.4f}±{d['cross_pos_std']:.4f} "
                f"(n={d['n_cross']})")

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Patch comparison summary so the figures notebook picks up new data
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH) as f:
            summary = json.load(f)
        summary["per_layer_same_pos"] = results
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"Updated: {SUMMARY_PATH}")

    # =================================================================
    # PLOT
    # =================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    layers = list(range(N_LAYERS))
    sp_means = [results["per_layer"][L]["same_pos_mean"] for L in layers]
    sp_stds = [results["per_layer"][L]["same_pos_std"] for L in layers]
    cp_means = [results["per_layer"][L]["cross_pos_mean"] for L in layers]
    cp_stds = [results["per_layer"][L]["cross_pos_std"] for L in layers]

    ax.errorbar(layers, sp_means, yerr=sp_stds, fmt="o-", color="#4C72B0",
                markersize=5, capsize=2, lw=1.5, label="Same position, diff replacement")
    ax.errorbar(layers, cp_means, yerr=cp_stds, fmt="s-", color="#DD8452",
                markersize=5, capsize=2, lw=1.5, label="Different position")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity (per-layer contribution delta)")
    ax.set_title(f"Per-Layer Typo Direction: Same-Position vs Cross-Position (N={n_groups} groups)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "same_vs_cross_position.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {OUTPUT_DIR / 'same_vs_cross_position.png'}")
    log("Done.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=100,
                    help="Benign prompts from OpenOrca.")
    ap.add_argument("--n-malicious", type=int, default=0,
                    help="If >0, also sample N malicious prompts from "
                         "HarmBench+AdvBench and tag per-prompt safety. "
                         "Per-layer cosines are then split by safety class.")
    args, _ = ap.parse_known_args()

    log("Loading model...")
    model, tokenizer = load_hf_model("meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16")
    device = next(model.parameters()).device
    log("Model loaded")
    run(model, tokenizer, device, n_prompts=args.n_prompts, n_malicious=args.n_malicious)


if __name__ == "__main__":
    main()
