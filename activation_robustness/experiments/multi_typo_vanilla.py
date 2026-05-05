#!/usr/bin/env python3
"""
E-031: Multi-Typo Interaction — Vanilla (offset-aligned from site1).

No anchor point. All curves aligned from site1. Measures how two typos
interact as seen from the perturbation dynamics themselves.

For each prompt and distance bucket:
  Extract 4 sequences: original, typo1-only, typo2-only, combined.
  site2 is BEFORE site1 (further from end).

Curves offset-aligned from site2 (the earlier typo):
  offset 0 = site2, offset = inter_dist → site1

Measurements:
  - Norm, angular shift, cos(h,h'), relative delta — all as offset curves
  - Superposition: cos(Δ_comb, Δ_1+Δ_2) and norm ratio at each offset post-site2
  - Single-typo baseline curves for comparison

Distance buckets: very_close(1-4), near(5-10), medium(10-20), mid_far(20-30), far(30+)
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
from activation_robustness.data.external import sample_openorca

os.environ.setdefault("PYTHONUNBUFFERED", "1")

OUTPUT_DIR = _REPO / "activation_robustness" / "results" / "multi_typo_vanilla"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [8, 16, 24, 31]
SEED = 47
MAX_OFFSET = 200  # offsets to track from site2

DISTANCE_BUCKETS = {
    "very_close_1-4": (1, 4),
    "near_5-10": (5, 10),
    "medium_10-20": (10, 20),
    "mid_far_20-30": (20, 30),
    "far_30+": (30, 999),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_full_sequence(model, tokenizer, text, device, layer=31):
    """Extract full hidden state sequence at one layer. Returns [seq_len, hidden] array."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[layer + 1][0].float().cpu().numpy().astype(np.float64)


def compute_metrics_from(orig_seq, pert_seq, start_pos, common_len, max_off=MAX_OFFSET):
    """Offset-aligned metrics from start_pos."""
    end = min(common_len, start_pos + max_off)
    norms, angles, cosines, rels = [], [], [], []
    for pos in range(start_pos, end):
        h_o, h_p = orig_seq[pos], pert_seq[pos]
        d = h_p - h_o
        dn = float(np.linalg.norm(d))
        hon = float(np.linalg.norm(h_o))
        hpn = float(np.linalg.norm(h_p))
        norms.append(dn)
        rels.append(dn / (hon + 1e-10))
        if hon > 1e-10 and hpn > 1e-10:
            c = float(np.dot(h_o, h_p) / (hon * hpn))
            c = max(-1.0, min(1.0, c))
            cosines.append(c)
            angles.append(float(np.degrees(np.arccos(c))))
        else:
            cosines.append(1.0)
            angles.append(0.0)
    return {"norms": norms, "angles": angles, "cosines": cosines, "rels": rels}


def superposition_from(orig_seq, comb_seq, pert1_seq, pert2_seq, start_pos, common_len, max_off=MAX_OFFSET):
    """Superposition metrics offset-aligned from start_pos.

    Only meaningful at positions where both typos have taken effect.
    """
    end = min(common_len, start_pos + max_off)
    lin_cos, norm_ratio = [], []
    for pos in range(start_pos, end):
        actual = comb_seq[pos] - orig_seq[pos]
        d1 = pert1_seq[pos] - orig_seq[pos]
        d2 = pert2_seq[pos] - orig_seq[pos]
        predicted = d1 + d2
        an = float(np.linalg.norm(actual))
        sn = float(np.linalg.norm(predicted))
        if an > 1e-10 and sn > 1e-10:
            c = float(np.dot(actual, predicted) / (an * sn))
            lin_cos.append(max(-1.0, min(1.0, c)))
        else:
            lin_cos.append(1.0)
        norm_ratio.append(an / (sn + 1e-10))
    return {"linearity_cos": lin_cos, "norm_ratio": norm_ratio}


def get_typo_sites(tokenizer, orig_formatted, text, adj_key, rng):
    variants = adj_key.enumerate_all(text)
    if not variants:
        return []
    by_cp = defaultdict(list)
    for pt, meta in variants:
        by_cp[meta["position"]].append((pt, meta))
    orig_ids = tokenizer.encode(orig_formatted)
    sites, seen = [], set()
    for cp in sorted(by_cp.keys()):
        cands = by_cp[cp]
        pt, meta = cands[rng.integers(len(cands))]
        pf = format_prompt(tokenizer, pt, steering_prompt=None, add_generation_prompt=False)
        pi = tokenizer.encode(pf)
        if len(orig_ids) != len(pi):
            continue
        diffs = [i for i in range(len(orig_ids)) if orig_ids[i] != pi[i]]
        if len(diffs) != 1:
            continue
        tp = diffs[0]
        if tp not in seen:
            seen.add(tp)
            sites.append({"token_pos": tp, "char_pos": cp, "rep": meta["replacement_char"]})
    return sorted(sites, key=lambda x: x["token_pos"])


def apply_typo(text, char_pos, rep):
    r = list(text)
    r[char_pos] = rep
    return "".join(r)


def run(model, tokenizer, device, *, n_prompts: int = 200):
    """Run the experiment using a pre-loaded model.

    Bit-exact body of the original ``main()`` with model loading factored out.
    Same RNG seed, same ``sample_openorca`` call, same loops.
    """
    rng = np.random.default_rng(SEED)

    log("Sampling prompts...")
    orca = sample_openorca(n_per_bucket=n_prompts, buckets={"400-1000": (400, 1000)}, seed=SEED, scout_size=100000)
    prompts = [p["text"] for p in orca[:n_prompts]]
    log(f"  {len(prompts)} prompts")

    adj_key = AdjacentKey()

    # Storage
    single_results = []  # single-typo baseline (one per prompt)
    two_typo_results = {k: [] for k in DISTANCE_BUCKETS}

    for pi, prompt_text in enumerate(prompts):
        orig_fmt = format_prompt(tokenizer, prompt_text, steering_prompt=None, add_generation_prompt=False)
        orig_seq = extract_full_sequence(model, tokenizer, orig_fmt, device)
        seq_len = orig_seq.shape[0]

        sites = get_typo_sites(tokenizer, orig_fmt, prompt_text, adj_key, rng)
        if len(sites) < 2:
            continue

        # Single-typo baseline: random site
        s1 = sites[rng.integers(len(sites))]
        tp1 = s1["token_pos"]
        pert1_text = apply_typo(prompt_text, s1["char_pos"], s1["rep"])
        pert1_fmt = format_prompt(tokenizer, pert1_text, steering_prompt=None, add_generation_prompt=False)
        pert1_seq = extract_full_sequence(model, tokenizer, pert1_fmt, device)

        if pert1_seq.shape[0] == seq_len:
            m = compute_metrics_from(orig_seq, pert1_seq, tp1, seq_len)
            single_results.append({
                "prompt_idx": pi, "site_pos": int(tp1), "seq_len": int(seq_len), **m,
            })

        # Two-typo: for each distance bucket
        for bucket_name, (min_d, max_d) in DISTANCE_BUCKETS.items():
            # site1 = later in sequence, site2 = earlier (site2 before site1)
            # inter-typo distance in tokens
            valid_pairs = []
            for i in range(len(sites)):
                for j in range(i + 1, len(sites)):
                    dist = sites[j]["token_pos"] - sites[i]["token_pos"]
                    if min_d <= dist <= max_d:
                        # site2=sites[i] (earlier), site1=sites[j] (later)
                        # need MAX_OFFSET room after site2
                        if sites[i]["token_pos"] + MAX_OFFSET <= seq_len:
                            valid_pairs.append((sites[i], sites[j]))

            if not valid_pairs:
                continue

            s2, s1 = valid_pairs[rng.integers(len(valid_pairs))]
            tp2, tp1 = s2["token_pos"], s1["token_pos"]
            inter_dist = tp1 - tp2

            # Extract all 4 conditions
            t1_text = apply_typo(prompt_text, s1["char_pos"], s1["rep"])
            t2_text = apply_typo(prompt_text, s2["char_pos"], s2["rep"])
            comb_text = apply_typo(apply_typo(prompt_text, s1["char_pos"], s1["rep"]),
                                   s2["char_pos"], s2["rep"])

            t1_fmt = format_prompt(tokenizer, t1_text, steering_prompt=None, add_generation_prompt=False)
            t2_fmt = format_prompt(tokenizer, t2_text, steering_prompt=None, add_generation_prompt=False)
            comb_fmt = format_prompt(tokenizer, comb_text, steering_prompt=None, add_generation_prompt=False)

            t1_seq = extract_full_sequence(model, tokenizer, t1_fmt, device)
            t2_seq = extract_full_sequence(model, tokenizer, t2_fmt, device)
            comb_seq = extract_full_sequence(model, tokenizer, comb_fmt, device)

            if t1_seq.shape[0] != seq_len or t2_seq.shape[0] != seq_len or comb_seq.shape[0] != seq_len:
                continue

            # Offset-aligned from site2 (earlier typo)
            m_comb = compute_metrics_from(orig_seq, comb_seq, tp2, seq_len)
            m_t1 = compute_metrics_from(orig_seq, t1_seq, tp2, seq_len)  # typo1-only from site2's perspective
            m_t2 = compute_metrics_from(orig_seq, t2_seq, tp2, seq_len)  # typo2-only from site2's perspective

            # Superposition from site1 onward (where both effects are active)
            sup = superposition_from(orig_seq, comb_seq, t1_seq, t2_seq, tp1, seq_len)

            two_typo_results[bucket_name].append({
                "prompt_idx": pi,
                "site1_pos": int(tp1),
                "site2_pos": int(tp2),
                "inter_dist": int(inter_dist),
                "seq_len": int(seq_len),
                "combined": m_comb,
                "typo1_only": m_t1,
                "typo2_only": m_t2,
                "superposition": sup,  # from site1 onward
            })

        if (pi + 1) % 20 == 0:
            counts = {k: len(v) for k, v in two_typo_results.items()}
            log(f"  [{pi+1}/{len(prompts)}] single={len(single_results)}, 2-typo={counts}")

    del model
    torch.cuda.empty_cache()

    # =================================================================
    # AGGREGATE
    # =================================================================
    log("\n=== AGGREGATE ===")

    MIN_CURVE = 80
    summary = {"n_prompts": len(prompts), "single": {}, "two_typo": {}}

    # Single baseline
    ang_curves = [e["angles"][:MIN_CURVE] for e in single_results if len(e["angles"]) >= MIN_CURVE]
    nrm_curves = [e["norms"][:MIN_CURVE] for e in single_results if len(e["norms"]) >= MIN_CURVE]
    if ang_curves:
        summary["single"] = {
            "n": len(single_results), "n_curves": len(ang_curves),
            "avg_angular": np.mean(ang_curves, axis=0).tolist(),
            "std_angular": np.std(ang_curves, axis=0).tolist(),
            "avg_norm": np.mean(nrm_curves, axis=0).tolist(),
        }
    log(f"Single baseline: {len(single_results)} entries, {len(ang_curves)} curves")

    # Two-typo per bucket
    for bname, entries in sorted(two_typo_results.items()):
        if not entries:
            continue
        mean_dist = float(np.mean([e["inter_dist"] for e in entries]))

        # Average curves (combined, typo1, typo2)
        comb_ang = [e["combined"]["angles"][:MIN_CURVE] for e in entries if len(e["combined"]["angles"]) >= MIN_CURVE]
        t1_ang = [e["typo1_only"]["angles"][:MIN_CURVE] for e in entries if len(e["typo1_only"]["angles"]) >= MIN_CURVE]
        t2_ang = [e["typo2_only"]["angles"][:MIN_CURVE] for e in entries if len(e["typo2_only"]["angles"]) >= MIN_CURVE]

        comb_nrm = [e["combined"]["norms"][:MIN_CURVE] for e in entries if len(e["combined"]["norms"]) >= MIN_CURVE]
        t1_nrm = [e["typo1_only"]["norms"][:MIN_CURVE] for e in entries if len(e["typo1_only"]["norms"]) >= MIN_CURVE]
        t2_nrm = [e["typo2_only"]["norms"][:MIN_CURVE] for e in entries if len(e["typo2_only"]["norms"]) >= MIN_CURVE]

        # Superposition curves (from site1, shorter)
        sup_min = 50
        sup_cos = [e["superposition"]["linearity_cos"][:sup_min] for e in entries
                    if len(e["superposition"]["linearity_cos"]) >= sup_min]
        sup_nr = [e["superposition"]["norm_ratio"][:sup_min] for e in entries
                   if len(e["superposition"]["norm_ratio"]) >= sup_min]

        bucket_data = {
            "n": len(entries), "mean_dist": mean_dist,
            "n_curves": len(comb_ang),
        }
        if comb_ang:
            bucket_data["avg_comb_angular"] = np.mean(comb_ang, axis=0).tolist()
            bucket_data["std_comb_angular"] = np.std(comb_ang, axis=0).tolist()
            bucket_data["avg_t1_angular"] = np.mean(t1_ang, axis=0).tolist()
            bucket_data["avg_t2_angular"] = np.mean(t2_ang, axis=0).tolist()
            bucket_data["avg_comb_norm"] = np.mean(comb_nrm, axis=0).tolist()
            bucket_data["avg_t1_norm"] = np.mean(t1_nrm, axis=0).tolist()
            bucket_data["avg_t2_norm"] = np.mean(t2_nrm, axis=0).tolist()
        if sup_cos:
            bucket_data["avg_sup_cos"] = np.mean(sup_cos, axis=0).tolist()
            bucket_data["avg_sup_nr"] = np.mean(sup_nr, axis=0).tolist()
            bucket_data["n_sup_curves"] = len(sup_cos)

        summary["two_typo"][bname] = bucket_data
        log(f"  {bname}: N={len(entries)}, curves={len(comb_ang)}, dist={mean_dist:.0f}")

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved to {OUTPUT_DIR}")

    # =================================================================
    # PLOTS
    # =================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist_colors = {
        "very_close_1-4": "#C44E52", "near_5-10": "#DD8452",
        "medium_10-20": "#CCB974", "mid_far_20-30": "#55A868", "far_30+": "#4C72B0",
    }

    # ===== Figure 1: Angular shift curves (combined vs singles) =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # A: Combined angular curves by distance
    ax = axes[0, 0]
    if "avg_angular" in summary["single"]:
        c = np.array(summary["single"]["avg_angular"])
        ax.plot(np.arange(len(c)), c, "k-", lw=2, alpha=0.5, label="1-typo baseline")
    for bname in DISTANCE_BUCKETS:
        data = summary["two_typo"].get(bname)
        if data and "avg_comb_angular" in data:
            c = np.array(data["avg_comb_angular"])
            ax.plot(np.arange(len(c)), c, color=dist_colors[bname], lw=1.5,
                    label=f"2t d={data['mean_dist']:.0f}")
    ax.set_xlabel("Offset from site2 (earlier typo)")
    ax.set_ylabel("Angular shift (°)")
    ax.set_title("A. Combined: angular decay from site2")
    ax.legend(fontsize=7)

    # B: Combined norm curves
    ax = axes[0, 1]
    if "avg_norm" in summary["single"]:
        c = np.array(summary["single"]["avg_norm"])
        ax.plot(np.arange(len(c)), c, "k-", lw=2, alpha=0.5, label="1-typo baseline")
    for bname in DISTANCE_BUCKETS:
        data = summary["two_typo"].get(bname)
        if data and "avg_comb_norm" in data:
            c = np.array(data["avg_comb_norm"])
            ax.plot(np.arange(len(c)), c, color=dist_colors[bname], lw=1.5,
                    label=f"2t d={data['mean_dist']:.0f}")
    ax.set_xlabel("Offset from site2")
    ax.set_ylabel("Delta L2 norm")
    ax.set_title("B. Combined: norm decay from site2")
    ax.legend(fontsize=7)

    # C: Overlay combined vs typo1-only vs typo2-only (pick one bucket)
    ax = axes[0, 2]
    example_bucket = "medium_10-20"
    data = summary["two_typo"].get(example_bucket, {})
    if "avg_comb_angular" in data:
        c_comb = np.array(data["avg_comb_angular"])
        c_t1 = np.array(data["avg_t1_angular"])
        c_t2 = np.array(data["avg_t2_angular"])
        offs = np.arange(len(c_comb))
        ax.plot(offs, c_comb, "r-", lw=2, label="Combined")
        ax.plot(offs, c_t1, "b--", lw=1.5, label="Typo1 only (later)")
        ax.plot(offs, c_t2, "g--", lw=1.5, label="Typo2 only (earlier)")
        # Mark where site1 is
        d = data["mean_dist"]
        ax.axvline(d, color="gray", linestyle=":", alpha=0.5, label=f"Site1 at offset {d:.0f}")
    ax.set_xlabel("Offset from site2")
    ax.set_ylabel("Angular shift (°)")
    ax.set_title(f"C. Decomposition ({example_bucket})")
    ax.legend(fontsize=7)

    # D: Superposition direction (from site1)
    ax = axes[1, 0]
    for bname in DISTANCE_BUCKETS:
        data = summary["two_typo"].get(bname)
        if data and "avg_sup_cos" in data:
            c = np.array(data["avg_sup_cos"])
            ax.plot(np.arange(len(c)), c, color=dist_colors[bname], lw=1.5,
                    label=f"d={data['mean_dist']:.0f}")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Offset from site1 (both active)")
    ax.set_ylabel("cos(Δ_comb, Δ_1+Δ_2)")
    ax.set_title("D. Superposition direction (post-site1)")
    ax.legend(fontsize=7)

    # E: Superposition norm ratio (from site1)
    ax = axes[1, 1]
    for bname in DISTANCE_BUCKETS:
        data = summary["two_typo"].get(bname)
        if data and "avg_sup_nr" in data:
            c = np.array(data["avg_sup_nr"])
            ax.plot(np.arange(len(c)), c, color=dist_colors[bname], lw=1.5,
                    label=f"d={data['mean_dist']:.0f}")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3, label="Perfect superposition")
    ax.set_xlabel("Offset from site1")
    ax.set_ylabel("||Δ_comb|| / ||Δ_1+Δ_2||")
    ax.set_title("E. Superposition magnitude (post-site1)")
    ax.legend(fontsize=7)

    # F: Per-bucket summary
    ax = axes[1, 2]
    ax.axis("off")
    lines = ["Vanilla Multi-Typo Summary", "─" * 45, ""]
    lines.append(f"Single baseline: N={summary['single'].get('n', 0)}")
    lines.append("")
    for bname in DISTANCE_BUCKETS:
        data = summary["two_typo"].get(bname)
        if data:
            sup_cos_mean = np.mean(data.get("avg_sup_cos", [0]))
            sup_nr_mean = np.mean(data.get("avg_sup_nr", [0]))
            lines.append(f"  {bname} (d={data['mean_dist']:.0f}, N={data['n']}):")
            lines.append(f"    sup_cos={sup_cos_mean:.3f}, norm_ratio={sup_nr_mean:.3f}")
    for i, line in enumerate(lines):
        w = "bold" if i == 0 else "normal"
        sz = 10 if i == 0 else 8
        ax.text(0.02, 0.95 - i * 0.05, line, transform=ax.transAxes,
                fontsize=sz, fontweight=w, fontfamily="monospace", va="top")

    plt.suptitle(f"Multi-Typo Vanilla (E-031) — N={len(prompts)}, Layer {LAYERS[-1]}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multi_typo_vanilla.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {OUTPUT_DIR / 'multi_typo_vanilla.png'}")

    log("Done.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=200)
    args = parser.parse_args()

    log("Loading model...")
    model, tokenizer = load_hf_model("meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16")
    device = next(model.parameters()).device
    log("Model loaded")
    run(model, tokenizer, device, n_prompts=args.n_prompts)


if __name__ == "__main__":
    main()
