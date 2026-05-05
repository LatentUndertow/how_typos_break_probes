#!/usr/bin/env python3
"""
Punctuation decay control: measure spatial decay for ?->. and ?->/ perturbations
to validate that the decay shape characterized for adjacent-key typos generalizes
to a structurally different perturbation family (terminal-punctuation swap).

Sample OpenOrca prompts containing a `?` with at least MIN_SUFFIX_TOKENS
downstream tokens. For each, measure activation decay over the suffix window
under two perturbations:
    cond_period:  ? -> .     (Table 1, d=1: 7.9 deg, 23% of between-prompt baseline)
    cond_slash:   ? -> /     (Table 1, d=1: 26.4 deg, 76% of between-prompt baseline)

Output:
    results/punct_decay_control/
        results.json     - aggregated curves + checkpoints
        decay_curves.png - period and slash decay curves on Llama-3.1-8B layer 31

Usage:
    python punct_decay_control.py [--n-prompts 50] [--main-layer 31]
"""
import sys, os, json, time, gc, argparse, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from activation_robustness.analysis.extraction import format_prompt
from activation_robustness.data.external import sample_openorca
from activation_robustness.models.hf_model import ActivationModelHF, ModelConfig

OUTPUT_DIR = _REPO / "activation_robustness" / "results" / "punct_decay_control"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LAYERS = [31]
DEFAULT_MAIN_LAYER = 31
SEED = 77

MIN_SUFFIX_TOKENS = 32     # require >=32 tokens after ?
DECAY_WINDOW      = 30     # measure 30 positions of decay
MIN_PREFIX_TOKENS = 50     # also require some preamble before ?, sanity bound

CONDITION_COLORS = {
    "period": "#DD8452",
    "slash":  "#C44E52",
}

PERTURBATIONS = {
    "period": (".", "?->."),
    "slash":  ("/", "?->/"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_q_positions_in_text(text):
    """Return list of character positions of '?' in text."""
    return [i for i, c in enumerate(text) if c == "?"]


def extract_layer_seqs(hf_model, text, layers):
    """Extract full-sequence activations at given layers for one text.
    Returns {L: ndarray(seq_len, d_model)}."""
    cache = hf_model.extract_activations(text, layers=layers, hook_point="hook_resid_post")
    return {
        L: cache[f"blocks.{L}.hook_resid_post"][0].float().numpy().astype(np.float64)
        for L in layers
    }


def find_changed_token(tokenizer, orig_text, pert_text):
    """First diverging token index, or None. Allows token count to differ."""
    ids_o = tokenizer.encode(orig_text)
    ids_p = tokenizer.encode(pert_text)
    n = min(len(ids_o), len(ids_p))
    for i in range(n):
        if ids_o[i] != ids_p[i]:
            return i
    if len(ids_o) != len(ids_p):
        return n
    return None


def compute_decay_from_site(orig_seqs, pert_seqs, site_pos, decay_window, layers):
    """Compute angular shift, relative norm, etc. over decay_window positions
    starting at site_pos. Returns dict[layer] -> dict of curves."""
    out = {}
    for L in layers:
        h_orig = orig_seqs[L]
        h_pert = pert_seqs[L]
        common_len = min(h_orig.shape[0], h_pert.shape[0])
        if site_pos >= common_len:
            out[L] = None
            continue
        n_steps = min(common_len - site_pos, decay_window)
        if n_steps < decay_window:
            out[L] = None
            continue

        delta_site = h_pert[site_pos] - h_orig[site_pos]
        site_norm = float(np.linalg.norm(delta_site))
        if site_norm < 1e-10:
            out[L] = None
            continue

        rel_norms, angular_shift, rel_to_act = [], [], []
        for offset in range(n_steps):
            pos = site_pos + offset
            d = h_pert[pos] - h_orig[pos]
            h_o = h_orig[pos]
            h_p = h_pert[pos]
            d_norm = float(np.linalg.norm(d))
            h_o_norm = float(np.linalg.norm(h_o))
            h_p_norm = float(np.linalg.norm(h_p))

            rel_norms.append(d_norm / site_norm)
            rel_to_act.append(d_norm / (h_o_norm + 1e-10))

            if h_o_norm > 1e-10 and h_p_norm > 1e-10:
                cos_act = float(np.dot(h_o, h_p) / (h_o_norm * h_p_norm))
                angular_shift.append(float(np.degrees(np.arccos(np.clip(cos_act, -1.0, 1.0)))))
            else:
                angular_shift.append(0.0)

        out[L] = {
            "site_norm":     site_norm,
            "rel_norms":     rel_norms,
            "angular_shift": angular_shift,
            "rel_to_act":    rel_to_act,
        }
    return out


def aggregate(entries, decay_window, layer):
    rel_curves, ang_curves = [], []
    for e in entries:
        ld = e["per_layer"].get(layer)
        if ld is None:
            continue
        if len(ld["rel_norms"]) < decay_window:
            continue
        rel_curves.append(ld["rel_norms"][:decay_window])
        ang_curves.append(ld["angular_shift"][:decay_window])
    if not rel_curves:
        return None

    avg_rel = np.mean(rel_curves, axis=0)
    std_rel = np.std(rel_curves, axis=0)
    avg_ang = np.mean(ang_curves, axis=0)
    std_ang = np.std(ang_curves, axis=0)

    checkpoints = {}
    for cp in [0, 1, 5, 10, 20, decay_window - 1]:
        if cp < decay_window:
            checkpoints[cp] = {
                "rel_norm": float(avg_rel[cp]),
                "angular_deg": float(avg_ang[cp]),
            }

    return {
        "n_curves": len(rel_curves),
        "avg_rel_curve": avg_rel.tolist(),
        "std_rel_curve": std_rel.tolist(),
        "avg_ang_curve": avg_ang.tolist(),
        "std_ang_curve": std_ang.tolist(),
        "checkpoints":   checkpoints,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument("--main-layer", type=int, default=DEFAULT_MAIN_LAYER)
    parser.add_argument("--scout-size", type=int, default=10000,
                        help="how many OpenOrca prompts to scout for ? candidates")
    args = parser.parse_args()

    main_layer = args.main_layer
    layers = [main_layer]

    log(f"Output: {OUTPUT_DIR}")
    log(f"Layers: {layers}  Main: {main_layer}")
    log(f"Window: {DECAY_WINDOW} tokens downstream of ?")

    # Sample any-length OpenOrca prompts; we filter for ? presence below.
    log("Scouting OpenOrca for prompts containing '?'...")
    candidates = sample_openorca(
        n_per_bucket=args.scout_size,
        buckets={"any": (1, 99999)},
        seed=SEED,
        scout_size=args.scout_size * 2,
    )
    log(f"  scouted {len(candidates)}")

    # Filter: contains '?' and chat-templated form has enough tokens around it
    cfg = ModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        dtype="bfloat16",
    )
    log(f"Loading {cfg.model_name}...")
    model = ActivationModelHF(cfg)
    model.load()
    tokenizer = model.tokenizer
    log("Model loaded.")

    selected = []
    for prompt in candidates:
        if len(selected) >= args.n_prompts:
            break
        raw = prompt["text"]
        q_positions = find_q_positions_in_text(raw)
        if not q_positions:
            continue
        # Pick the FIRST '?' that has plenty of suffix and prefix in the raw text.
        # We let chat templating wrap and then verify token-level constraints.
        for q_char_pos in q_positions:
            # Need raw text to have substantive content after ?
            if len(raw) - q_char_pos - 1 < 80:  # ~30+ tokens roughly
                continue
            # Construct clean and perturbed raw texts
            clean_raw  = raw
            period_raw = raw[:q_char_pos] + "." + raw[q_char_pos + 1:]
            slash_raw  = raw[:q_char_pos] + "/" + raw[q_char_pos + 1:]

            # Format with chat template
            clean_fmt  = format_prompt(tokenizer, clean_raw, add_generation_prompt=False)
            period_fmt = format_prompt(tokenizer, period_raw, add_generation_prompt=False)
            slash_fmt  = format_prompt(tokenizer, slash_raw, add_generation_prompt=False)

            # Find token position where the change starts (vs clean)
            tp_period = find_changed_token(tokenizer, clean_fmt, period_fmt)
            tp_slash  = find_changed_token(tokenizer, clean_fmt, slash_fmt)
            if tp_period is None or tp_slash is None:
                continue
            # Use the PERIOD-condition site as our reference (most likely to align
            # with the '?' token cleanly). Slash may shift by one token in some
            # cases; we measure decay separately from each perturbation's site.
            tp = tp_period

            # Need enough downstream tokens
            seq_len_clean = len(tokenizer.encode(clean_fmt))
            if seq_len_clean - tp < DECAY_WINDOW + 2:
                continue
            if tp < MIN_PREFIX_TOKENS:
                continue

            selected.append({
                "raw":        clean_raw,
                "q_char_pos": q_char_pos,
                "clean_fmt":  clean_fmt,
                "period_fmt": period_fmt,
                "slash_fmt":  slash_fmt,
                "tp_period":  tp_period,
                "tp_slash":   tp_slash,
                "seq_len":    seq_len_clean,
            })
            break  # one ? per prompt

    log(f"Selected {len(selected)} prompts after filtering.")
    if len(selected) < 10:
        log("ERROR: too few qualifying prompts; increase --scout-size")
        sys.exit(1)

    # Run forward passes and compute decay
    per_condition_entries = defaultdict(list)
    for pi, item in enumerate(selected):
        # Extract activations
        clean_seqs  = extract_layer_seqs(model, item["clean_fmt"], layers)
        period_seqs = extract_layer_seqs(model, item["period_fmt"], layers)
        slash_seqs  = extract_layer_seqs(model, item["slash_fmt"], layers)

        decay_period = compute_decay_from_site(
            clean_seqs, period_seqs, item["tp_period"], DECAY_WINDOW, layers)
        decay_slash = compute_decay_from_site(
            clean_seqs, slash_seqs, item["tp_slash"], DECAY_WINDOW, layers)

        if decay_period.get(main_layer) is not None:
            per_condition_entries["period"].append({
                "prompt_idx": pi,
                "site_token": item["tp_period"],
                "seq_len":    item["seq_len"],
                "per_layer":  {L: decay_period[L] for L in layers if decay_period[L] is not None},
            })
        if decay_slash.get(main_layer) is not None:
            per_condition_entries["slash"].append({
                "prompt_idx": pi,
                "site_token": item["tp_slash"],
                "seq_len":    item["seq_len"],
                "per_layer":  {L: decay_slash[L] for L in layers if decay_slash[L] is not None},
            })

        if (pi + 1) % 10 == 0:
            log(f"  processed {pi + 1}/{len(selected)}; "
                f"period={len(per_condition_entries['period'])}, "
                f"slash={len(per_condition_entries['slash'])}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Aggregate
    log("\nAggregating...")
    summary = {
        "model":        cfg.model_name,
        "main_layer":   main_layer,
        "decay_window": DECAY_WINDOW,
        "n_selected":   len(selected),
        "per_condition": {},
    }
    for cond_name in ["period", "slash"]:
        entries = per_condition_entries.get(cond_name, [])
        if not entries:
            log(f"  {cond_name}: no entries")
            continue
        agg = aggregate(entries, DECAY_WINDOW, main_layer)
        if agg is None:
            log(f"  {cond_name}: aggregation failed")
            continue
        summary["per_condition"][cond_name] = {
            "n_entries": len(entries),
            "per_layer": {main_layer: agg},
        }
        log(f"  {cond_name}: n={agg['n_curves']}, "
            f"site_ang={agg['checkpoints'][0]['angular_deg']:.2f}°, "
            f"+10_ang={agg['checkpoints'][10]['angular_deg']:.2f}°, "
            f"+29_ang={agg['checkpoints'][DECAY_WINDOW - 1]['angular_deg']:.2f}°")

    out_json = OUTPUT_DIR / "results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved: {out_json}")

    # Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    offsets = np.arange(DECAY_WINDOW)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Try to load AdjacentKey decay (mid condition, layer 31) for overlay context
    typo_path = _REPO / "activation_robustness" / "results" / "typo_decay_preamble_control" / "results.json"
    typo_curve_rel = None
    typo_curve_ang = None
    if typo_path.exists():
        try:
            with open(typo_path) as f:
                typo_data = json.load(f)
            mid = typo_data.get("per_condition", {}).get("mid", {}).get("per_layer", {}).get(str(main_layer))
            if mid is None:
                mid = typo_data.get("per_condition", {}).get("mid", {}).get("per_layer", {}).get(main_layer)
            if mid is not None:
                # Take only the first DECAY_WINDOW points of typo decay
                typo_curve_rel = mid["avg_rel_curve"][:DECAY_WINDOW]
                typo_curve_ang = mid["avg_ang_curve"][:DECAY_WINDOW]
        except Exception as e:
            log(f"  WARN: could not load typo decay overlay: {e}")

    if typo_curve_rel is not None:
        axes[0].plot(offsets, np.array(typo_curve_rel) * 100,
                     color="#4C72B0", lw=1.5, ls="--",
                     label=f"AdjacentKey (mid), L{main_layer}")
        axes[1].plot(offsets, typo_curve_ang,
                     color="#4C72B0", lw=1.5, ls="--",
                     label=f"AdjacentKey (mid), L{main_layer}")

    for cond_name, color in CONDITION_COLORS.items():
        cond = summary["per_condition"].get(cond_name)
        if cond is None:
            continue
        agg = cond["per_layer"][main_layer]
        n = agg["n_curves"]
        label = f"?→{'.' if cond_name == 'period' else '/'} (N={n})"
        rel = np.array(agg["avg_rel_curve"]) * 100
        ang = np.array(agg["avg_ang_curve"])
        axes[0].plot(offsets, rel, color=color, lw=1.8, label=label)
        axes[1].plot(offsets, ang, color=color, lw=1.8, label=label)

    axes[0].set_xlabel("Offset from perturbation site (tokens)")
    axes[0].set_ylabel("% of on-site delta norm")
    axes[0].set_title(f"Relative norm decay — Layer {main_layer}")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, DECAY_WINDOW - 1)

    axes[1].set_xlabel("Offset from perturbation site (tokens)")
    axes[1].set_ylabel("Angular shift (°)")
    axes[1].set_title(f"Activation rotation — Layer {main_layer}")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, DECAY_WINDOW - 1)

    plt.suptitle(f"Punctuation perturbation decay — {cfg.model_name}", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "decay_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {OUTPUT_DIR / 'decay_curves.png'}")
    log("Done.")


if __name__ == "__main__":
    main()
