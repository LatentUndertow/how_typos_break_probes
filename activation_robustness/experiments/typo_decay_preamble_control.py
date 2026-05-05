#!/usr/bin/env python3
"""
E-029: Typo Decay — Preamble Control Experiment.

Isolates the effect of upstream context (tokens BEFORE the typo) from the
effect of downstream tokens on the perturbation decay profile.

Design:
    For each long prompt (1500+ words / ~2000+ tokens), place the same-style
    typo at THREE controlled preamble sizes within the same text:

        early:  50–120  preamble tokens before typo
        mid:    350–550 preamble tokens before typo
        late:   750–1100 preamble tokens before typo

    All conditions use the same underlying prompt text, so any difference in
    decay shape is attributable purely to preamble size.

    All conditions share a COMMON_WINDOW = 300 token downstream comparison.

Key question:
    Does the upstream context size change the RATE at which the typo
    perturbation decays at each intermediate offset, or only the asymptote?

Theoretical prior (causal attention):
    Tokens after offset k cannot affect the activation at k — so downstream
    tokens beyond the current measurement offset should not alter the decay
    curve shape. Only preamble (upstream) tokens can.

Output:
    results/typo_decay_preamble_control/
        results.json      — per-condition aggregated curves + checkpoint stats
        decay_curves.png  — main comparison plot (all 3 conditions)
        per_layer.png     — same comparison repeated across layers
"""
import sys, os, json, time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from activation_robustness.analysis.extraction import format_prompt
from activation_robustness.perturbations.typo import AdjacentKey
from activation_robustness.data.external import sample_openorca
from activation_robustness.models.hf_model import ActivationModelHF, ModelConfig

os.environ.setdefault("PYTHONUNBUFFERED", "1")

OUTPUT_DIR = _REPO / "activation_robustness" / "results" / "typo_decay_preamble_control"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [8, 16, 24, 31]
SEED = 77
MAX_DOWNSTREAM = 400
COMMON_WINDOW = 300   # downstream tokens shared by all conditions for comparison

# Preamble token ranges — how many tokens come BEFORE the typo site
PREAMBLE_TARGETS = {
    "early": (50,  120),
    "mid":   (350, 550),
    "late":  (750, 1100),
}

CONDITION_COLORS = {
    "early": "#4C72B0",
    "mid":   "#DD8452",
    "late":  "#55A868",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_changed_token(tokenizer, orig_text, pert_text):
    """Return (token_pos, same_length). token_pos=None if no change or ambiguous."""
    ids_o = tokenizer.encode(orig_text)
    ids_p = tokenizer.encode(pert_text)
    if len(ids_o) != len(ids_p):
        return None, False
    diffs = [i for i in range(len(ids_o)) if ids_o[i] != ids_p[i]]
    if len(diffs) == 0:
        return None, True
    return diffs[0], True


def get_offset_mapping(tokenizer, text):
    """Return list of (char_start, char_end) per token position."""
    enc = tokenizer(text, return_offsets_mapping=True)
    return enc["offset_mapping"]


def find_raw_text_offset(formatted_text, raw_text):
    """Return char offset where raw_text begins in formatted_text.

    Searches for a 40-char anchor from the start of raw_text. Falls back to
    a rough estimate based on string lengths.
    """
    anchor_len = min(40, len(raw_text))
    anchor = raw_text[:anchor_len]
    idx = formatted_text.rfind(anchor)   # rfind avoids false matches in template header
    if idx >= 0:
        return idx
    # Fallback: template header is ~40-60 chars, user text follows
    return max(0, len(formatted_text) - len(raw_text) - 30)


def find_typo_for_preamble(tokenizer, raw_text, formatted_text, adj_key,
                            tgt_lo, tgt_hi, rng, max_tries=60):
    """Find a typo variant whose changed token falls in [tgt_lo, tgt_hi].

    Returns (pert_raw_text, pert_formatted_text, token_pos) or (None, None, None).
    """
    offsets = get_offset_mapping(tokenizer, formatted_text)
    seq_len = len(offsets)
    if tgt_hi >= seq_len:
        return None, None, None

    # Character range in formatted_text for the target token window
    char_lo_fmt = offsets[tgt_lo][0]
    char_hi_fmt = offsets[tgt_hi][1]

    # Convert to character range in raw_text
    raw_start = find_raw_text_offset(formatted_text, raw_text)
    raw_char_lo = max(0, char_lo_fmt - raw_start - 10)
    raw_char_hi = char_hi_fmt - raw_start + 10

    # Filter all variants to those in the estimated char range
    all_variants = adj_key.enumerate_all(raw_text)
    filtered = [(t, m) for t, m in all_variants
                if raw_char_lo <= m["position"] <= raw_char_hi]

    if not filtered:
        return None, None, None

    indices = rng.permutation(len(filtered))[:max_tries]
    for idx in indices:
        pert_text, meta = filtered[idx]
        pert_fmt = format_prompt(tokenizer, pert_text,
                                  steering_prompt=None, add_generation_prompt=False)
        tp, same_len = find_changed_token(tokenizer, formatted_text, pert_fmt)
        if same_len and tp is not None and tgt_lo <= tp <= tgt_hi:
            return pert_text, pert_fmt, tp

    return None, None, None


def extract_layer_seqs(hf_model, text):
    """Extract full-sequence activations at LAYERS for one text.

    Uses ActivationModelHF.extract_activations (hook_resid_post = after each
    transformer block). Returns {L: array(seq_len, d_model)}.

    ActivationModelHF frees all 32-layer hidden states immediately after
    selecting only the requested layers — same pattern as activation_classifier.
    """
    cache = hf_model.extract_activations(text, layers=LAYERS, hook_point="hook_resid_post")
    return {
        L: cache[f"blocks.{L}.hook_resid_post"][0].float().numpy().astype(np.float64)
        for L in LAYERS
    }


def compute_decay_curve(orig_seqs, pert_seqs, token_pos, max_downstream):
    """Compute per-layer decay statistics from token_pos up to max_downstream offsets.

    Returns dict:
        {layer: {
            "norms":         list[float],  # ||delta|| at each offset
            "rel_norms":     list[float],  # ||delta|| / ||delta_at_site||
            "cosines":       list[float],  # cos(delta, delta_at_site)
            "angular_shift": list[float],  # angle(h_orig, h_pert) in degrees
            "rel_to_act":    list[float],  # ||delta|| / ||h_orig||
            "site_norm":     float,
        }}
    """
    result = {}
    for L in LAYERS:
        h_orig = orig_seqs[L]  # (seq_len, d)
        h_pert = pert_seqs[L]
        common_len = min(h_orig.shape[0], h_pert.shape[0])
        if token_pos >= common_len:
            result[L] = None
            continue

        delta_site = h_pert[token_pos] - h_orig[token_pos]
        site_norm = float(np.linalg.norm(delta_site))
        if site_norm < 1e-10:
            result[L] = None
            continue

        norms, rel_norms, cosines, angular_shift, rel_to_act = [], [], [], [], []
        n_steps = min(common_len - token_pos, max_downstream)

        for offset in range(n_steps):
            pos = token_pos + offset
            d = h_pert[pos] - h_orig[pos]
            h_o = h_orig[pos]

            d_norm = float(np.linalg.norm(d))
            h_o_norm = float(np.linalg.norm(h_o))
            h_p_norm = float(np.linalg.norm(h_pert[pos]))

            norms.append(d_norm)
            rel_norms.append(d_norm / site_norm)
            rel_to_act.append(d_norm / (h_o_norm + 1e-10))

            # Direction consistency with site delta
            if d_norm > 1e-10:
                cos_dir = float(np.dot(d, delta_site) / (d_norm * site_norm))
                cosines.append(np.clip(cos_dir, -1.0, 1.0))
            else:
                cosines.append(0.0)

            # Activation rotation h_orig → h_pert
            if h_o_norm > 1e-10 and h_p_norm > 1e-10:
                cos_act = float(np.dot(h_o, h_pert[pos]) / (h_o_norm * h_p_norm))
                cos_act = float(np.clip(cos_act, -1.0, 1.0))
                angular_shift.append(float(np.degrees(np.arccos(cos_act))))
            else:
                angular_shift.append(0.0)

        result[L] = {
            "site_norm":     site_norm,
            "norms":         norms,
            "rel_norms":     rel_norms,
            "cosines":       cosines,
            "angular_shift": angular_shift,
            "rel_to_act":    rel_to_act,
            "downstream_steps": n_steps,
        }
    return result


def aggregate_curves(entries, common_window, layer):
    """Average curves across entries, aligned to common_window offsets."""
    rel_curves, cos_curves, ang_curves, rel_act_curves = [], [], [], []

    for e in entries:
        ld = e["per_layer"].get(layer)
        if ld is None:
            continue
        rn = ld["rel_norms"]
        co = ld["cosines"]
        ang = ld["angular_shift"]
        ra = ld["rel_to_act"]
        if len(rn) < common_window:
            continue
        rel_curves.append(rn[:common_window])
        cos_curves.append(co[:common_window])
        ang_curves.append(ang[:common_window])
        rel_act_curves.append(ra[:common_window])

    if not rel_curves:
        return None

    offsets = np.arange(common_window)
    avg_rel  = np.mean(rel_curves, axis=0)
    std_rel  = np.std(rel_curves, axis=0)
    avg_cos  = np.mean(cos_curves, axis=0)
    avg_ang  = np.mean(ang_curves, axis=0)
    avg_ra   = np.mean(rel_act_curves, axis=0)

    checkpoints = {}
    for cp in [1, 5, 10, 20, 50, 100, 200, common_window - 1]:
        if cp < common_window:
            checkpoints[cp] = {
                "rel_norm":    float(avg_rel[cp]),
                "cosine":      float(avg_cos[cp]),
                "angular_deg": float(avg_ang[cp]),
                "rel_to_act":  float(avg_ra[cp]),
            }

    tail_start = int(common_window * 0.8)
    return {
        "n_curves":       len(rel_curves),
        "avg_rel_curve":  avg_rel.tolist(),
        "std_rel_curve":  std_rel.tolist(),
        "avg_cos_curve":  avg_cos.tolist(),
        "avg_ang_curve":  avg_ang.tolist(),
        "avg_ra_curve":   avg_ra.tolist(),
        "checkpoints":    checkpoints,
        "tail_rel_norm":  float(avg_rel[tail_start:].mean()),
        "tail_cosine":    float(avg_cos[tail_start:].mean()),
        "tail_angular":   float(avg_ang[tail_start:].mean()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=100,
                        help="Long prompts to attempt (1500+ words each)")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    log("Sampling long prompts (1500+ words)...")
    orca = sample_openorca(
        n_per_bucket=args.n_prompts,
        buckets={"1500+": (1500, 99999)},
        seed=SEED,
        scout_size=20000,
    )
    log(f"  {len(orca)} prompts sampled")

    log("Loading model...")
    hf_model = ActivationModelHF(ModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        dtype="bfloat16",
    ))
    hf_model.load()
    tokenizer = hf_model.tokenizer
    log("Model loaded")

    adj_key = AdjacentKey()

    # per_condition_entries[cond] = list of per-prompt dicts
    per_condition_entries = defaultdict(list)
    n_attempted = 0

    for pi, prompt in enumerate(orca):
        raw_text = prompt["text"]
        orig_fmt = format_prompt(tokenizer, raw_text,
                                  steering_prompt=None, add_generation_prompt=False)

        # Check sequence is long enough for the late condition
        seq_len = len(tokenizer.encode(orig_fmt))
        if seq_len < PREAMBLE_TARGETS["late"][1] + COMMON_WINDOW + 20:
            continue

        n_attempted += 1

        # Find a valid typo for each preamble condition
        condition_typos = {}
        for cond_name, (tgt_lo, tgt_hi) in PREAMBLE_TARGETS.items():
            pert_text, pert_fmt, token_pos = find_typo_for_preamble(
                tokenizer, raw_text, orig_fmt, adj_key,
                tgt_lo, tgt_hi, rng,
            )
            if pert_text is not None:
                condition_typos[cond_name] = (pert_fmt, token_pos)

        if not condition_typos:
            continue

        # Batch original + all perturbed variants into a single forward pass.
        # find_typo_for_preamble guarantees same token length as orig_fmt, so
        # they can always be stacked into one batch.
        valid_conditions = {
            c: (pf, tp) for c, (pf, tp) in condition_typos.items()
            if seq_len - tp >= COMMON_WINDOW
        }
        if not valid_conditions:
            continue

        orig_seqs = extract_layer_seqs(hf_model, orig_fmt)
        pert_seqs_by_cond = {
            cond_name: extract_layer_seqs(hf_model, pf)
            for cond_name, (pf, _) in valid_conditions.items()
        }

        for cond_name, (pert_fmt, token_pos) in valid_conditions.items():
            downstream_room = seq_len - token_pos
            pert_seqs = pert_seqs_by_cond[cond_name]

            decay = compute_decay_curve(orig_seqs, pert_seqs, token_pos, MAX_DOWNSTREAM)

            # Check L31 is valid before storing
            if decay.get(31) is None:
                continue

            entry = {
                "prompt_idx":      pi,
                "condition":       cond_name,
                "preamble_tokens": token_pos,
                "seq_len":         seq_len,
                "downstream_room": downstream_room,
                "per_layer":       {L: decay[L] for L in LAYERS if decay[L] is not None},
            }
            per_condition_entries[cond_name].append(entry)

        if (n_attempted % 10) == 0:
            counts = {c: len(v) for c, v in per_condition_entries.items()}
            log(f"[{n_attempted}/{len(orca)}] found so far: {counts}")

    del hf_model
    torch.cuda.empty_cache()

    # =========================================================================
    # AGGREGATE
    # =========================================================================
    log("\n=== AGGREGATE ===")

    summary = {
        "n_attempted": n_attempted,
        "common_window": COMMON_WINDOW,
        "per_condition": {},
    }

    MAIN_LAYER = 31
    for cond_name in ["early", "mid", "late"]:
        entries = per_condition_entries.get(cond_name, [])
        if not entries:
            log(f"  {cond_name}: no entries")
            continue

        preamble_vals = [e["preamble_tokens"] for e in entries]
        log(f"\n  {cond_name}: N={len(entries)}, "
            f"preamble_tokens: mean={np.mean(preamble_vals):.0f} "
            f"[{min(preamble_vals)}–{max(preamble_vals)}]")

        cond_data = {
            "n_entries": len(entries),
            "preamble_mean": float(np.mean(preamble_vals)),
            "preamble_range": [int(min(preamble_vals)), int(max(preamble_vals))],
            "per_layer": {},
        }

        for layer in LAYERS:
            agg = aggregate_curves(entries, COMMON_WINDOW, layer)
            if agg is None:
                continue
            cond_data["per_layer"][layer] = agg

            if layer == MAIN_LAYER:
                log(f"    L{layer}: n_curves={agg['n_curves']}, "
                    f"tail_rel={agg['tail_rel_norm']*100:.2f}%, "
                    f"tail_ang={agg['tail_angular']:.2f}°")
                for cp, stats in sorted(agg["checkpoints"].items(), key=lambda x: int(x[0])):
                    log(f"      +{int(cp):>3d}: rel={stats['rel_norm']*100:.2f}%, "
                        f"cos={stats['cosine']:.4f}, ang={stats['angular_deg']:.2f}°")

        summary["per_condition"][cond_name] = cond_data

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved: {OUTPUT_DIR / 'results.json'}")

    # =========================================================================
    # PLOTS
    # =========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Main comparison plot (L31) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    offsets = np.arange(COMMON_WINDOW)

    metrics = [
        ("avg_rel_curve",  "std_rel_curve", "% of site norm",       "A. Relative norm decay by preamble size",       True),
        ("avg_cos_curve",  None,            "Cosine w/ site delta",  "B. Direction consistency by preamble size",     False),
        ("avg_ang_curve",  None,            "Angular shift (°)",     "C. Activation rotation — probe vulnerability",  False),
        ("avg_ra_curve",   None,            "||Δh|| / ||h|| (%)",    "D. Relative to activation norm",                True),
    ]

    for ax, (curve_key, std_key, ylabel, title, pct) in zip(axes.flat, metrics):
        for cond_name in ["early", "mid", "late"]:
            cond_data = summary["per_condition"].get(cond_name)
            if cond_data is None:
                continue
            layer_data = cond_data["per_layer"].get(MAIN_LAYER)
            if layer_data is None:
                continue
            curve = np.array(layer_data[curve_key])
            n = layer_data["n_curves"]
            color = CONDITION_COLORS[cond_name]
            preamble_mean = cond_data["preamble_mean"]
            label = f"{cond_name} (~{preamble_mean:.0f} tok preamble, N={n})"
            y = curve * 100 if pct else curve
            ax.plot(offsets, y, color=color, linewidth=1.5, label=label)
            if std_key and pct:
                std = np.array(layer_data[std_key])
                ax.fill_between(offsets, (curve - std) * 100, (curve + std) * 100,
                                color=color, alpha=0.12)
        ax.set_xlabel("Offset from typo site (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        if curve_key == "avg_rel_curve":
            ax.set_ylim(bottom=0)
        if curve_key == "avg_cos_curve":
            ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
            ax.set_ylim(-0.15, 1.05)

    plt.suptitle(
        f"Typo Decay: Preamble Control (E-029) — Layer {MAIN_LAYER}, "
        f"Common window = {COMMON_WINDOW} tokens",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "decay_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {OUTPUT_DIR / 'decay_curves.png'}")

    # --- Per-layer grid ---
    fig, axes = plt.subplots(len(LAYERS), 2, figsize=(13, 4 * len(LAYERS)))
    for row, layer in enumerate(LAYERS):
        for col, (curve_key, pct, title_sfx) in enumerate([
            ("avg_rel_curve", True, "Relative norm (% of site)"),
            ("avg_ang_curve", False, "Angular shift (°)"),
        ]):
            ax = axes[row, col]
            for cond_name in ["early", "mid", "late"]:
                cond_data = summary["per_condition"].get(cond_name)
                if cond_data is None:
                    continue
                ld = cond_data["per_layer"].get(layer)
                if ld is None:
                    continue
                curve = np.array(ld[curve_key])
                n = ld["n_curves"]
                color = CONDITION_COLORS[cond_name]
                y = curve * 100 if pct else curve
                ax.plot(offsets, y, color=color, linewidth=1.3,
                        label=f"{cond_name} N={n}")
            ax.set_title(f"Layer {layer} — {title_sfx}")
            ax.set_xlabel("Offset (tokens)")
            if pct:
                ax.set_ylim(bottom=0)
            ax.legend(fontsize=7)

    plt.suptitle("Preamble Control — All Layers", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {OUTPUT_DIR / 'per_layer.png'}")

    log("Done.")


if __name__ == "__main__":
    main()
