#!/usr/bin/env python3
"""
E-041: Multi-Perturbation EOT Comparison.

Group A — Mid-sequence perturbations at controlled distances from EOT:
  adjacent_key, missing_space, extra_space
  Uses enumerate_all() to find all valid sites, picks nearest to target distance.
  Fixed distances: 5, 10, 20 tokens from EOT.
  Length buckets: short (50-100 tok), medium (200-400 tok), long (800+ tok).

Group B — Terminal perturbations (distance ≈ 0):
  question_to_period, question_to_slash
  Question prompts across length buckets.

All layers: 8, 16, 24, 31.
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
from activation_robustness.perturbations.omission import MissingSpace, ExtraSpace
from activation_robustness.perturbations import get_by_name
from activation_robustness.data.external import sample_openorca

os.environ.setdefault("PYTHONUNBUFFERED", "1")

OUTPUT_DIR = _REPO / "activation_robustness" / "results" / "multi_perturbation_eot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [8, 16, 24, 31]
SEED = 52

FIXED_DISTANCES = [5, 10, 20]

LENGTH_BUCKETS = {
    "short_50-100": (30, 70),
    "medium_200-400": (150, 300),
    "long_800+": (600, 1500),
}

# Group A perturbations with enumerate_all support
GROUP_A_PERTS = {
    "adjacent_key": AdjacentKey(),
    "missing_space": MissingSpace(),
    "extra_space": ExtraSpace(),
}

# Group B: terminal perturbations
GROUP_B_NAMES = ["punctuation.question_to_period", "typo.question_to_slash"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_eot(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return {L: out.hidden_states[L + 1][0, -1, :].float().cpu().numpy().astype(np.float64)
            for L in LAYERS}, ids.shape[1]


def eot_metrics(orig_eot, pert_eot):
    result = {}
    for L in LAYERS:
        h_o, h_p = orig_eot[L], pert_eot[L]
        d = h_p - h_o
        d_norm = float(np.linalg.norm(d))
        h_o_norm = float(np.linalg.norm(h_o))
        h_p_norm = float(np.linalg.norm(h_p))
        if h_o_norm > 1e-10 and h_p_norm > 1e-10:
            cos = float(np.dot(h_o, h_p) / (h_o_norm * h_p_norm))
            cos = max(-1.0, min(1.0, cos))
            angle = float(np.degrees(np.arccos(cos)))
        else:
            cos, angle = 1.0, 0.0
        result[L] = {"angle": angle, "cos": cos, "delta_norm": d_norm,
                      "rel_delta": d_norm / (h_o_norm + 1e-10)}
    return result


def try_perturbation_near_target(tokenizer, orig_text, orig_formatted, pert_obj, target_tp, rng):
    """Find and apply a perturbation near target_tp. Returns (token_pos, pert_formatted) or (None, None).

    Strategy: sort all variants by char position distance from target,
    tokenize top candidates, check for single-token change near target.
    """
    orig_ids = tokenizer.encode(orig_formatted)

    all_variants = pert_obj.enumerate_all(orig_text)
    if not all_variants:
        return None, None

    # Estimate: char position in raw text maps roughly to token position
    # by the ratio (target_tp / len(orig_ids)) * len(orig_text)
    target_char = int(target_tp / len(orig_ids) * len(orig_text))

    # Sort variants by distance of their char position from target_char
    scored = [(abs(meta["position"] - target_char), pert_text, meta)
              for pert_text, meta in all_variants]
    scored.sort(key=lambda x: x[0])

    # Try top 10 closest candidates
    for _, pert_text, meta in scored[:10]:
        pert_formatted = format_prompt(tokenizer, pert_text, steering_prompt=None, add_generation_prompt=False)
        ids_p = tokenizer.encode(pert_formatted)
        if len(orig_ids) != len(ids_p):
            continue
        diffs = [i for i in range(len(orig_ids)) if orig_ids[i] != ids_p[i]]
        if len(diffs) == 0:
            continue
        tp = diffs[0]
        if abs(tp - target_tp) <= 5:
            return tp, pert_formatted

    return None, None


def run(model, tokenizer, device, *, n_per_bucket: int = 150):
    """Run the experiment using a pre-loaded model.

    Bit-exact body of the original ``main()`` with model loading factored out.
    """
    rng = np.random.default_rng(SEED)

    log("Sampling prompts...")
    orca_general = sample_openorca(
        n_per_bucket=n_per_bucket,
        buckets=LENGTH_BUCKETS,
        seed=SEED,
        scout_size=200000,
    )
    log(f"  General: {len(orca_general)} prompts")

    # Question prompts for Group B
    orca_q_raw = sample_openorca(
        n_per_bucket=n_per_bucket * 5,
        buckets=LENGTH_BUCKETS,
        seed=SEED + 1,
        scout_size=200000,
    )
    orca_questions = defaultdict(list)
    for p in orca_q_raw:
        if p["text"].strip().endswith("?"):
            orca_questions[p["length_bucket"]].append(p["text"])
    for lb, ps in orca_questions.items():
        log(f"  Questions {lb}: {len(ps)}")

    general_by_bucket = defaultdict(list)
    for p in orca_general:
        general_by_bucket[p["length_bucket"]].append(p["text"])

    # ==============================
    # GROUP A
    # ==============================
    log("\n=== GROUP A: Mid-sequence perturbations (enumerate_all) ===")

    results_a = {pn: {lb: {d: [] for d in FIXED_DISTANCES} for lb in LENGTH_BUCKETS}
                 for pn in GROUP_A_PERTS}

    total_a = 0
    for lb, bucket_prompts in sorted(general_by_bucket.items()):
        log(f"\n  Bucket {lb}: {len(bucket_prompts)} prompts")

        for pi, prompt_text in enumerate(bucket_prompts[:n_per_bucket]):
            orig_fmt = format_prompt(tokenizer, prompt_text, steering_prompt=None, add_generation_prompt=False)
            orig_eot, seq_len = extract_eot(model, tokenizer, orig_fmt, device)
            eot_pos = seq_len - 1

            for pname, pert_obj in GROUP_A_PERTS.items():
                for fixed_dist in FIXED_DISTANCES:
                    target_tp = eot_pos - fixed_dist
                    if target_tp < 5:
                        continue

                    tp, pert_fmt = try_perturbation_near_target(
                        tokenizer, prompt_text, orig_fmt, pert_obj, target_tp, rng)

                    if tp is None:
                        continue

                    pert_eot, pert_len = extract_eot(model, tokenizer, pert_fmt, device)
                    if pert_len != seq_len:
                        continue

                    em = eot_metrics(orig_eot, pert_eot)
                    results_a[pname][lb][fixed_dist].append({
                        "prompt_idx": total_a,
                        "seq_len": int(seq_len),
                        "actual_dist": int(eot_pos - tp),
                        "per_layer": em,
                    })

            total_a += 1
            if total_a % 30 == 0:
                counts = {pn: sum(len(results_a[pn][lb2][d])
                          for lb2 in LENGTH_BUCKETS for d in FIXED_DISTANCES)
                          for pn in GROUP_A_PERTS}
                log(f"    A [{total_a}] {counts}")

    # ==============================
    # GROUP B
    # ==============================
    log("\n=== GROUP B: Terminal perturbations ===")

    results_b = {pn: {lb: [] for lb in LENGTH_BUCKETS} for pn in GROUP_B_NAMES}

    total_b = 0
    for lb in sorted(orca_questions.keys()):
        prompts_q = orca_questions[lb][:n_per_bucket]
        log(f"\n  Bucket {lb}: {len(prompts_q)} question prompts")

        for pi, prompt_text in enumerate(prompts_q):
            orig_fmt = format_prompt(tokenizer, prompt_text, steering_prompt=None, add_generation_prompt=False)
            orig_eot, seq_len = extract_eot(model, tokenizer, orig_fmt, device)

            for pname in GROUP_B_NAMES:
                pert_fn = get_by_name(pname)
                pert_text = pert_fn.apply(prompt_text, rng)
                if pert_text is None or pert_text == prompt_text:
                    continue

                pert_fmt = format_prompt(tokenizer, pert_text, steering_prompt=None, add_generation_prompt=False)
                pert_eot, pert_len = extract_eot(model, tokenizer, pert_fmt, device)

                if pert_len != seq_len:
                    continue

                em = eot_metrics(orig_eot, pert_eot)
                results_b[pname][lb].append({
                    "prompt_idx": total_b,
                    "seq_len": int(seq_len),
                    "per_layer": em,
                })

            total_b += 1
            if total_b % 30 == 0:
                counts = {pn: sum(len(results_b[pn][lb2]) for lb2 in LENGTH_BUCKETS)
                          for pn in GROUP_B_NAMES}
                log(f"    B [{total_b}] {counts}")

    del model
    torch.cuda.empty_cache()

    # =================================================================
    # AGGREGATE
    # =================================================================
    log("\n=== AGGREGATE ===")

    summary = {"group_a": {}, "group_b": {}}

    for pname in GROUP_A_PERTS:
        summary["group_a"][pname] = {}
        log(f"\n  {pname}:")
        for lb in LENGTH_BUCKETS:
            summary["group_a"][pname][lb] = {}
            for d in FIXED_DISTANCES:
                entries = results_a[pname][lb][d]
                if not entries:
                    continue
                per_layer = {}
                for L in LAYERS:
                    angles = [e["per_layer"][L]["angle"] for e in entries]
                    per_layer[f"L{L}"] = {
                        "angle_mean": float(np.mean(angles)),
                        "angle_std": float(np.std(angles)),
                    }
                summary["group_a"][pname][lb][d] = {"n": len(entries), **per_layer}
                log(f"    {lb} d={d}: N={len(entries)}, L31={per_layer['L31']['angle_mean']:.1f}°")

    for pname in GROUP_B_NAMES:
        summary["group_b"][pname] = {}
        log(f"\n  {pname}:")
        for lb in LENGTH_BUCKETS:
            entries = results_b[pname][lb]
            if not entries:
                continue
            per_layer = {}
            for L in LAYERS:
                angles = [e["per_layer"][L]["angle"] for e in entries]
                per_layer[f"L{L}"] = {
                    "angle_mean": float(np.mean(angles)),
                    "angle_std": float(np.std(angles)),
                }
            summary["group_b"][pname][lb] = {"n": len(entries), **per_layer}
            log(f"    {lb}: N={len(entries)}, L31={per_layer['L31']['angle_mean']:.1f}°")

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved to {OUTPUT_DIR}")

    # =================================================================
    # PLOTS
    # =================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pert_colors = {
        "adjacent_key": "#4C72B0",
        "missing_space": "#DD8452",
        "extra_space": "#55A868",
        "punctuation.question_to_period": "#8172B2",
        "typo.question_to_slash": "#C44E52",
    }
    short_names = {
        "adjacent_key": "adj_key",
        "missing_space": "miss_space",
        "extra_space": "extra_space",
        "punctuation.question_to_period": "?→.",
        "typo.question_to_slash": "?→/",
    }

    # Figure 1: Group A by distance (L31)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for di, d in enumerate(FIXED_DISTANCES):
        ax = axes[di]
        for pname in GROUP_A_PERTS:
            lengths, angles = [], []
            for lb in ["short_50-100", "medium_200-400", "long_800+"]:
                data = summary["group_a"].get(pname, {}).get(lb, {}).get(d)
                if data and "L31" in data:
                    lengths.append(lb.split("_")[0])
                    angles.append(data["L31"]["angle_mean"])
            if lengths:
                ax.plot(range(len(lengths)), angles, "o-", color=pert_colors[pname],
                        markersize=7, lw=1.5, label=short_names[pname])
        ax.set_xticks(range(3))
        ax.set_xticklabels(["short", "medium", "long"])
        ax.set_ylabel("Angular shift at EOT (°)")
        ax.set_title(f"d={d} from EOT")
        ax.legend(fontsize=8)

    plt.suptitle("Group A: Mid-Sequence Perturbations (Layer 31)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "group_a_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Group B terminal
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for pname in GROUP_B_NAMES:
        lengths, angles = [], []
        for lb in ["short_50-100", "medium_200-400", "long_800+"]:
            data = summary["group_b"].get(pname, {}).get(lb)
            if data and "L31" in data:
                lengths.append(lb.split("_")[0])
                angles.append(data["L31"]["angle_mean"])
        if lengths:
            ax.plot(range(len(lengths)), angles, "o-", color=pert_colors[pname],
                    markersize=8, lw=2, label=short_names[pname])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["short", "medium", "long"])
    ax.set_ylabel("Angular shift at EOT (°)")
    ax.set_title("Group B: Terminal Perturbations at EOT (Layer 31)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "group_b_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 3: All compared at d=10 L31
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    all_perts, short_vals, long_vals = [], [], []
    for pname in GROUP_A_PERTS:
        s = summary["group_a"].get(pname, {}).get("short_50-100", {}).get(10, {})
        l = summary["group_a"].get(pname, {}).get("long_800+", {}).get(10, {})
        if s and l and "L31" in s and "L31" in l:
            all_perts.append(short_names[pname])
            short_vals.append(s["L31"]["angle_mean"])
            long_vals.append(l["L31"]["angle_mean"])
    for pname in GROUP_B_NAMES:
        s = summary["group_b"].get(pname, {}).get("short_50-100")
        l = summary["group_b"].get(pname, {}).get("long_800+")
        if s and l and "L31" in s and "L31" in l:
            all_perts.append(short_names[pname] + "\n(terminal)")
            short_vals.append(s["L31"]["angle_mean"])
            long_vals.append(l["L31"]["angle_mean"])
    if all_perts:
        x = np.arange(len(all_perts))
        w = 0.35
        ax.bar(x - w/2, short_vals, w, color="#C44E52", label="Short")
        ax.bar(x + w/2, long_vals, w, color="#4C72B0", label="Long")
        ax.set_xticks(x)
        ax.set_xticklabels(all_perts, fontsize=9)
        ax.set_ylabel("Angular shift at EOT (°)")
        ax.set_title("All Perturbations Compared (Layer 31)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_perturbations_compared.png", dpi=150, bbox_inches="tight")
    plt.close()

    log("Done.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-bucket", type=int, default=150)
    args = parser.parse_args()

    log("Loading model...")
    model, tokenizer = load_hf_model("meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16")
    device = next(model.parameters()).device
    log("Model loaded")
    run(model, tokenizer, device, n_per_bucket=args.n_per_bucket)


if __name__ == "__main__":
    main()
