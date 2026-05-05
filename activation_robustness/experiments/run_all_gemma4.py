#!/usr/bin/env python3
"""
Overnight runner: Repeat all key experiments on google/gemma-4-E4B-it.

Fully self-contained. No user interaction needed. Saves all results
to activation_robustness/results/gemma4_e4b/<experiment_name>/.

Experiments (in order):
1. E-033: Length-controlled EOT (multi-layer)
2. Baseline variance (multi-layer)
3. E-028: Position sensitivity
4. E-041: Multi-perturbation EOT
5. Per-layer typo direction probe
6. Per-layer same-pos check
7. E-025: Typo direction (with probe impact + per-layer)
8. E-027: Long-prompt decay
"""
import sys, os, json, time, gc
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# =====================================================================
# CONFIG
# =====================================================================
MODEL_NAME = "google/gemma-4-E4B-it"
N_LAYERS_TOTAL = 42
# Sliding/full pairs at mid-late and last depth
# Full attention at 29, 41; sliding neighbors at 28, 40
LAYERS = [28, 29, 40, 41]
LAYER_TYPES = {28: "sliding", 29: "full", 40: "sliding", 41: "full"}
D_MODEL = 2560
SEED_BASE = 200  # offset seeds from Llama/Qwen runs
RESULTS_BASE = _REPO / "activation_robustness" / "results" / "gemma4_e4b"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model():
    from activation_robustness.analysis.extraction import load_hf_model
    log(f"Loading {MODEL_NAME}...")
    model, tokenizer = load_hf_model(MODEL_NAME, dtype="bfloat16")
    device = next(model.parameters()).device
    log("Model loaded")
    return model, tokenizer, device


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("Model freed")


def format_prompt(tokenizer, text, steering_prompt=None, add_generation_prompt=False):
    messages = []
    if steering_prompt:
        messages.append({"role": "system", "content": steering_prompt})
    messages.append({"role": "user", "content": text})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def extract_eot(model, tokenizer, text, device):
    """Extract EOT activation at all LAYERS."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return {L: out.hidden_states[L + 1][0, -1, :].float().cpu().numpy().astype(np.float64)
            for L in LAYERS}, ids.shape[1]


def extract_full_seq(model, tokenizer, text, device):
    """Extract full sequence at all LAYERS."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return {L: out.hidden_states[L + 1][0].float().cpu().numpy().astype(np.float64)
            for L in LAYERS}, ids.shape[1]


def extract_all_layers(model, tokenizer, text, device):
    """Extract hidden states at ALL layers (0..N_LAYERS_TOTAL-1)."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return [out.hidden_states[i][0].float().cpu().numpy().astype(np.float64)
            for i in range(len(out.hidden_states))], ids.shape[1]


def eot_metrics(orig_eot, pert_eot):
    result = {}
    for L in LAYERS:
        h_o, h_p = orig_eot[L], pert_eot[L]
        d = h_p - h_o
        dn = float(np.linalg.norm(d))
        hon = float(np.linalg.norm(h_o))
        hpn = float(np.linalg.norm(h_p))
        if hon > 1e-10 and hpn > 1e-10:
            cos = float(np.dot(h_o, h_p) / (hon * hpn))
            cos = max(-1.0, min(1.0, cos))
            angle = float(np.degrees(np.arccos(cos)))
        else:
            cos, angle = 1.0, 0.0
        result[L] = {"angle": angle, "cos": cos, "delta_norm": dn,
                      "rel_delta": dn / (hon + 1e-10), "act_norm": hon}
    return result


def get_typo_sites(tokenizer, orig_formatted, text, adj_key, rng, max_sites=None):
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
        pf = format_prompt(tokenizer, pt)
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
    sites = sorted(sites, key=lambda x: x["token_pos"])
    if max_sites and len(sites) > max_sites:
        step = len(sites) // max_sites
        sites = sites[::step][:max_sites]
    return sites


def apply_typo(text, char_pos, rep):
    r = list(text)
    r[char_pos] = rep
    return "".join(r)


# =====================================================================
# EXPERIMENT 1: E-033 Length-controlled EOT (multi-layer)
# =====================================================================
def run_e033(model, tokenizer, device):
    from activation_robustness.perturbations.typo import AdjacentKey
    from activation_robustness.data.external import sample_openorca

    log("\n" + "=" * 60)
    log("EXPERIMENT 1: E-033 Length-controlled EOT")
    log("=" * 60)

    OUT = RESULTS_BASE / "e033_length_controlled"
    OUT.mkdir(parents=True, exist_ok=True)

    adj_key = AdjacentKey()
    rng = np.random.default_rng(SEED_BASE + 1)
    FIXED_DISTANCES = [5, 10, 20]
    LENGTH_BUCKETS = {"short_50-100": (30, 70), "medium_200-400": (150, 300), "long_800+": (600, 1500)}

    orca = sample_openorca(n_per_bucket=200, buckets=LENGTH_BUCKETS, seed=SEED_BASE, scout_size=200000)
    by_bucket = defaultdict(list)
    for p in orca:
        by_bucket[p["length_bucket"]].append(p["text"])

    results = {lb: {d: [] for d in FIXED_DISTANCES} for lb in LENGTH_BUCKETS}
    total = 0

    for lb, prompts in sorted(by_bucket.items()):
        log(f"  Bucket {lb}: {len(prompts)} prompts")
        for prompt_text in prompts[:200]:
            orig_fmt = format_prompt(tokenizer, prompt_text)
            orig_eot, seq_len = extract_eot(model, tokenizer, orig_fmt, device)
            eot_pos = seq_len - 1
            sites = get_typo_sites(tokenizer, orig_fmt, prompt_text, adj_key, rng)
            if not sites:
                continue
            for d in FIXED_DISTANCES:
                target = eot_pos - d
                if target < 5:
                    continue
                best, best_diff = None, float("inf")
                for s in sites:
                    diff = abs(s["token_pos"] - target)
                    if diff < best_diff and diff <= 3:
                        best_diff = diff
                        best = s
                if best is None:
                    continue
                pt = apply_typo(prompt_text, best["char_pos"], best["rep"])
                pf = format_prompt(tokenizer, pt)
                pe, pl = extract_eot(model, tokenizer, pf, device)
                if pl != seq_len:
                    continue
                em = eot_metrics(orig_eot, pe)
                results[lb][d].append({"seq_len": int(seq_len), "per_layer": em})
            total += 1
            if total % 40 == 0:
                log(f"    [{total}]")

    # Aggregate
    summary = {}
    for lb in LENGTH_BUCKETS:
        summary[lb] = {}
        for d in FIXED_DISTANCES:
            entries = results[lb][d]
            if not entries:
                continue
            summary[lb][d] = {"n": len(entries), "seq_len_mean": float(np.mean([e["seq_len"] for e in entries]))}
            for L in LAYERS:
                angles = [e["per_layer"][L]["angle"] for e in entries]
                summary[lb][d][f"L{L}"] = {"angle_mean": float(np.mean(angles)), "angle_std": float(np.std(angles)),
                                            "act_norm_mean": float(np.mean([e["per_layer"][L]["act_norm"] for e in entries]))}

    with open(OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    for L in LAYERS:
        log(f"\n  Layer {L}:")
        for lb in LENGTH_BUCKETS:
            for d in FIXED_DISTANCES:
                if d in summary.get(lb, {}) and f"L{L}" in summary[lb][d]:
                    s = summary[lb][d][f"L{L}"]
                    log(f"    {lb} d={d}: N={summary[lb][d]['n']}, angle={s['angle_mean']:.1f}°")

    log(f"  Saved to {OUT}")


# =====================================================================
# EXPERIMENT 2: Baseline variance (multi-layer)
# =====================================================================
def run_baseline(model, tokenizer, device):
    from activation_robustness.data.external import sample_openorca

    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Baseline variance")
    log("=" * 60)

    OUT = RESULTS_BASE / "baseline_variance"
    OUT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED_BASE + 2)
    LENGTH_BUCKETS = {"short_50-100": (30, 70), "medium_200-400": (150, 300), "long_800+": (600, 1500)}

    orca = sample_openorca(n_per_bucket=200, buckets=LENGTH_BUCKETS, seed=SEED_BASE + 2, scout_size=200000)
    by_bucket = defaultdict(list)
    for p in orca:
        by_bucket[p["length_bucket"]].append(p["text"])

    eot_acts = {}
    for lb, prompts in sorted(by_bucket.items()):
        eot_acts[lb] = []
        log(f"  Extracting {lb}...")
        for i, text in enumerate(prompts[:200]):
            fmt = format_prompt(tokenizer, text)
            ids = tokenizer(fmt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            per_layer = {L: out.hidden_states[L + 1][0, -1, :].float().cpu().numpy().astype(np.float64)
                         for L in LAYERS}
            eot_acts[lb].append({"h": per_layer, "seq_len": int(ids.shape[1])})
            if (i + 1) % 50 == 0:
                log(f"    [{i+1}]")

    summary = {}
    for L in LAYERS:
        log(f"\n  Layer {L}:")
        for lb, entries in sorted(eot_acts.items()):
            vecs = [e["h"][L] for e in entries]
            n = len(vecs)
            norms = [float(np.linalg.norm(v)) for v in vecs]
            pair_angles = []
            for _ in range(500):
                i, j = rng.choice(n, size=2, replace=False)
                ni, nj = np.linalg.norm(vecs[i]), np.linalg.norm(vecs[j])
                if ni > 1e-10 and nj > 1e-10:
                    cos = float(np.dot(vecs[i], vecs[j]) / (ni * nj))
                    cos = max(-1.0, min(1.0, cos))
                    pair_angles.append(float(np.degrees(np.arccos(cos))))
            if lb not in summary:
                summary[lb] = {"n": n, "mean_seq_len": float(np.mean([e["seq_len"] for e in entries]))}
            summary[lb][f"L{L}"] = {
                "mean_norm": float(np.mean(norms)),
                "within_angle_mean": float(np.mean(pair_angles)),
                "within_angle_std": float(np.std(pair_angles)),
            }
            log(f"    {lb}: norm={np.mean(norms):.1f}, angle={np.mean(pair_angles):.1f}°±{np.std(pair_angles):.1f}")

    with open(OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Saved to {OUT}")


# =====================================================================
# EXPERIMENT 3: E-028 Position sensitivity
# =====================================================================
def run_e028(model, tokenizer, device):
    from activation_robustness.perturbations.typo import AdjacentKey
    from activation_robustness.data.external import sample_openorca

    log("\n" + "=" * 60)
    log("EXPERIMENT 3: E-028 Position sensitivity")
    log("=" * 60)

    OUT = RESULTS_BASE / "e028_position_sensitivity"
    OUT.mkdir(parents=True, exist_ok=True)

    adj_key = AdjacentKey()
    rng = np.random.default_rng(SEED_BASE + 3)
    PREFIX_BUCKETS = {"5-10": (5, 10), "15-25": (15, 25), "30-50": (30, 50),
                      "60-90": (60, 90), "100-150": (100, 150), "200+": (200, 9999)}
    MAX_DOWNSTREAM = 100

    orca = sample_openorca(n_per_bucket=200, buckets={"300-1000": (300, 1000)},
                           seed=SEED_BASE + 3, scout_size=100000)
    prompts = [p["text"] for p in orca[:200]]
    log(f"  {len(prompts)} prompts")

    # Find user content start
    empty_fmt = format_prompt(tokenizer, "")
    user_start = len(tokenizer.encode(empty_fmt)) - 1
    log(f"  User content starts at token {user_start}")

    results_by_bucket = {k: [] for k in PREFIX_BUCKETS}

    for pi, prompt_text in enumerate(prompts):
        orig_fmt = format_prompt(tokenizer, prompt_text)
        orig_seq, seq_len = extract_full_seq(model, tokenizer, orig_fmt, device)
        sites = get_typo_sites(tokenizer, orig_fmt, prompt_text, adj_key, rng)
        if not sites:
            continue

        for bname, (min_p, max_p) in PREFIX_BUCKETS.items():
            valid = [s for s in sites
                     if min_p <= (s["token_pos"] - user_start) < max_p
                     and s["token_pos"] + 10 < seq_len]
            if not valid:
                continue
            site = valid[rng.integers(len(valid))]
            tp = site["token_pos"]
            pt = apply_typo(prompt_text, site["char_pos"], site["rep"])
            pf = format_prompt(tokenizer, pt)
            pert_seq, pl = extract_full_seq(model, tokenizer, pf, device)
            if pl != seq_len:
                continue

            # Site metrics per layer
            per_layer_site = {}
            per_layer_last = {}
            for L in LAYERS:
                h_o, h_p = orig_seq[L][tp], pert_seq[L][tp]
                d = h_p - h_o
                hon, hpn = np.linalg.norm(h_o), np.linalg.norm(h_p)
                if hon > 1e-10 and hpn > 1e-10:
                    cos = float(np.dot(h_o, h_p) / (hon * hpn))
                    cos = max(-1.0, min(1.0, cos))
                    per_layer_site[L] = {"angle": float(np.degrees(np.arccos(cos))),
                                          "rel_delta": float(np.linalg.norm(d)) / (hon + 1e-10)}
                # Last token
                h_o_l, h_p_l = orig_seq[L][-1], pert_seq[L][-1]
                d_l = h_p_l - h_o_l
                hon_l, hpn_l = np.linalg.norm(h_o_l), np.linalg.norm(h_p_l)
                if hon_l > 1e-10 and hpn_l > 1e-10:
                    cos_l = float(np.dot(h_o_l, h_p_l) / (hon_l * hpn_l))
                    cos_l = max(-1.0, min(1.0, cos_l))
                    per_layer_last[L] = {"angle": float(np.degrees(np.arccos(cos_l)))}

            results_by_bucket[bname].append({
                "prefix_len": int(tp - user_start),
                "seq_len": int(seq_len),
                "site": per_layer_site,
                "last_tok": per_layer_last,
            })

        if (pi + 1) % 40 == 0:
            counts = {k: len(v) for k, v in results_by_bucket.items()}
            log(f"    [{pi+1}] {counts}")

    # Aggregate
    summary = {}
    for bname, entries in sorted(results_by_bucket.items()):
        if not entries:
            continue
        summary[bname] = {"n": len(entries), "mean_prefix": float(np.mean([e["prefix_len"] for e in entries]))}
        for L in LAYERS:
            site_angles = [e["site"][L]["angle"] for e in entries if L in e["site"]]
            last_angles = [e["last_tok"][L]["angle"] for e in entries if L in e["last_tok"]]
            summary[bname][f"L{L}"] = {
                "site_angle_mean": float(np.mean(site_angles)) if site_angles else 0,
                "site_angle_std": float(np.std(site_angles)) if site_angles else 0,
                "last_angle_mean": float(np.mean(last_angles)) if last_angles else 0,
            }

    with open(OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    for L in LAYERS:
        log(f"\n  Layer {L}:")
        for bname in sorted(summary.keys()):
            s = summary[bname].get(f"L{L}", {})
            log(f"    {bname}: N={summary[bname]['n']}, site={s.get('site_angle_mean',0):.1f}°, "
                f"last={s.get('last_angle_mean',0):.2f}°")

    log(f"  Saved to {OUT}")


# =====================================================================
# EXPERIMENT 4: E-041 Multi-perturbation EOT
# =====================================================================
def run_e041(model, tokenizer, device):
    from activation_robustness.perturbations.typo import AdjacentKey
    from activation_robustness.perturbations.omission import MissingSpace, ExtraSpace
    from activation_robustness.perturbations import get_by_name
    from activation_robustness.data.external import sample_openorca

    log("\n" + "=" * 60)
    log("EXPERIMENT 4: E-041 Multi-perturbation EOT")
    log("=" * 60)

    OUT = RESULTS_BASE / "e041_multi_perturbation"
    OUT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED_BASE + 4)
    FIXED_DISTANCES = [5, 10, 20]
    LENGTH_BUCKETS = {"short_50-100": (30, 70), "medium_200-400": (150, 300), "long_800+": (600, 1500)}
    GROUP_A = {"adjacent_key": AdjacentKey(), "missing_space": MissingSpace()}
    GROUP_B = ["punctuation.question_to_period", "typo.question_to_slash"]

    orca_gen = sample_openorca(n_per_bucket=150, buckets=LENGTH_BUCKETS, seed=SEED_BASE + 4, scout_size=200000)
    orca_q_raw = sample_openorca(n_per_bucket=450, buckets=LENGTH_BUCKETS, seed=SEED_BASE + 5, scout_size=200000)

    gen_by_bucket = defaultdict(list)
    for p in orca_gen:
        gen_by_bucket[p["length_bucket"]].append(p["text"])

    q_by_bucket = defaultdict(list)
    for p in orca_q_raw:
        if p["text"].strip().endswith("?"):
            q_by_bucket[p["length_bucket"]].append(p["text"])

    results_a = {pn: {lb: {d: [] for d in FIXED_DISTANCES} for lb in LENGTH_BUCKETS} for pn in GROUP_A}
    results_b = {pn: {lb: [] for lb in LENGTH_BUCKETS} for pn in GROUP_B}

    # Group A
    total = 0
    for lb, prompts in sorted(gen_by_bucket.items()):
        log(f"  A: {lb}: {len(prompts)} prompts")
        for prompt_text in prompts[:150]:
            orig_fmt = format_prompt(tokenizer, prompt_text)
            orig_ids = tokenizer.encode(orig_fmt)
            orig_eot, seq_len = extract_eot(model, tokenizer, orig_fmt, device)
            eot_pos = seq_len - 1

            for pname, pert_obj in GROUP_A.items():
                all_variants = pert_obj.enumerate_all(prompt_text)
                if not all_variants:
                    continue
                target_char_ratio = lambda tp: int(tp / len(orig_ids) * len(prompt_text))
                for d in FIXED_DISTANCES:
                    target_tp = eot_pos - d
                    if target_tp < 5:
                        continue
                    tc = target_char_ratio(target_tp)
                    scored = sorted(all_variants, key=lambda x: abs(x[1]["position"] - tc))
                    found = False
                    for pt, meta in scored[:10]:
                        pf = format_prompt(tokenizer, pt)
                        ids_p = tokenizer.encode(pf)
                        if len(orig_ids) != len(ids_p):
                            continue
                        diffs = [i for i in range(len(orig_ids)) if orig_ids[i] != ids_p[i]]
                        if not diffs:
                            continue
                        tp = diffs[0]
                        if abs(tp - target_tp) <= 5:
                            pe, pl = extract_eot(model, tokenizer, pf, device)
                            if pl != seq_len:
                                continue
                            em = eot_metrics(orig_eot, pe)
                            results_a[pname][lb][d].append({"per_layer": em})
                            found = True
                            break

            total += 1
            if total % 30 == 0:
                counts = {pn: sum(len(results_a[pn][lb2][d2]) for lb2 in LENGTH_BUCKETS for d2 in FIXED_DISTANCES) for pn in GROUP_A}
                log(f"    A [{total}] {counts}")

    # Group B
    total_b = 0
    for lb in sorted(q_by_bucket.keys()):
        prompts_q = q_by_bucket[lb][:150]
        log(f"  B: {lb}: {len(prompts_q)} question prompts")
        for prompt_text in prompts_q:
            orig_fmt = format_prompt(tokenizer, prompt_text)
            orig_eot, seq_len = extract_eot(model, tokenizer, orig_fmt, device)
            for pname in GROUP_B:
                pert_fn = get_by_name(pname)
                pt = pert_fn.apply(prompt_text, rng)
                if pt is None or pt == prompt_text:
                    continue
                pf = format_prompt(tokenizer, pt)
                pe, pl = extract_eot(model, tokenizer, pf, device)
                if pl != seq_len:
                    continue
                em = eot_metrics(orig_eot, pe)
                results_b[pname][lb].append({"per_layer": em})
            total_b += 1
            if total_b % 30 == 0:
                log(f"    B [{total_b}]")

    # Aggregate
    summary = {"group_a": {}, "group_b": {}}
    for pname in GROUP_A:
        summary["group_a"][pname] = {}
        for lb in LENGTH_BUCKETS:
            summary["group_a"][pname][lb] = {}
            for d in FIXED_DISTANCES:
                entries = results_a[pname][lb][d]
                if not entries:
                    continue
                per_layer = {}
                for L in LAYERS:
                    angles = [e["per_layer"][L]["angle"] for e in entries]
                    per_layer[f"L{L}"] = {"angle_mean": float(np.mean(angles)), "angle_std": float(np.std(angles))}
                summary["group_a"][pname][lb][d] = {"n": len(entries), **per_layer}
                log(f"  {pname} {lb} d={d}: N={len(entries)}, L{LAYERS[-1]}={per_layer[f'L{LAYERS[-1]}']['angle_mean']:.1f}°")

    for pname in GROUP_B:
        summary["group_b"][pname] = {}
        for lb in LENGTH_BUCKETS:
            entries = results_b[pname][lb]
            if not entries:
                continue
            per_layer = {}
            for L in LAYERS:
                angles = [e["per_layer"][L]["angle"] for e in entries]
                per_layer[f"L{L}"] = {"angle_mean": float(np.mean(angles)), "angle_std": float(np.std(angles))}
            summary["group_b"][pname][lb] = {"n": len(entries), **per_layer}
            log(f"  {pname} {lb}: N={len(entries)}, L{LAYERS[-1]}={per_layer[f'L{LAYERS[-1]}']['angle_mean']:.1f}°")

    with open(OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Saved to {OUT}")


# =====================================================================
# EXPERIMENT 5: Per-layer typo direction probe
# =====================================================================
def run_per_layer_direction(model, tokenizer, device):
    from activation_robustness.perturbations.typo import AdjacentKey
    from activation_robustness.data.external import sample_openorca

    log("\n" + "=" * 60)
    log("EXPERIMENT 5: Per-layer typo direction probe")
    log("=" * 60)

    OUT = RESULTS_BASE / "per_layer_direction"
    OUT.mkdir(parents=True, exist_ok=True)

    adj_key = AdjacentKey()
    rng = np.random.default_rng(SEED_BASE + 5)

    orca = sample_openorca(n_per_bucket=30, buckets={"100-300": (100, 300)}, seed=SEED_BASE + 5)
    prompts = [p["text"] for p in orca[:30]]
    log(f"  {len(prompts)} prompts")

    per_layer_all = {L: [] for L in range(N_LAYERS_TOTAL)}
    per_prompt = {L: defaultdict(list) for L in range(N_LAYERS_TOTAL)}
    n_variants = 0

    for pi, prompt_text in enumerate(prompts):
        orig_fmt = format_prompt(tokenizer, prompt_text)
        orig_hs, seq_len = extract_all_layers(model, tokenizer, orig_fmt, device)
        sites = get_typo_sites(tokenizer, orig_fmt, prompt_text, adj_key, rng, max_sites=6)
        if len(sites) < 2:
            continue
        for site in sites:
            tp = site["token_pos"]
            pt = apply_typo(prompt_text, site["char_pos"], site["rep"])
            pf = format_prompt(tokenizer, pt)
            pert_hs, pl = extract_all_layers(model, tokenizer, pf, device)
            if pl != seq_len:
                continue
            for L in range(N_LAYERS_TOTAL):
                orig_c = orig_hs[L + 1][tp] - orig_hs[L][tp]
                pert_c = pert_hs[L + 1][tp] - pert_hs[L][tp]
                delta = pert_c - orig_c
                per_layer_all[L].append(delta)
                per_prompt[L][pi].append(delta)
            n_variants += 1
        if (pi + 1) % 10 == 0:
            log(f"    [{pi+1}] variants: {n_variants}")

    log(f"  Total variants: {n_variants}")

    results = {"n_variants": n_variants, "per_layer": {}}
    for L in range(N_LAYERS_TOTAL):
        deltas = per_layer_all[L]
        if len(deltas) < 5:
            continue
        arr = np.array(deltas)
        norms = np.linalg.norm(arr, axis=1)
        within = []
        for prompt_idx, pd in per_prompt[L].items():
            if len(pd) < 2:
                continue
            for i in range(len(pd)):
                for j in range(i + 1, len(pd)):
                    n1, n2 = np.linalg.norm(pd[i]), np.linalg.norm(pd[j])
                    if n1 > 1e-10 and n2 > 1e-10:
                        within.append(float(np.dot(pd[i], pd[j]) / (n1 * n2)))
        mu = arr.mean(axis=0)
        mu_ratio = np.linalg.norm(mu) / (norms.mean() + 1e-10)
        results["per_layer"][L] = {
            "cross_pos_cos": float(np.mean(within)) if within else 0,
            "mu_ratio": float(mu_ratio),
            "mean_norm": float(norms.mean()),
        }

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log(f"\n  {'Layer':>5s} {'cross_cos':>10s} {'mu_ratio':>9s} {'norm':>8s}")
    for L in range(N_LAYERS_TOTAL):
        if L in results["per_layer"]:
            r = results["per_layer"][L]
            log(f"  {L:>5d} {r['cross_pos_cos']:>10.4f} {r['mu_ratio']:>9.4f} {r['mean_norm']:>8.2f}")

    log(f"  Saved to {OUT}")


# =====================================================================
# EXPERIMENT 6: Per-layer same-pos check
# =====================================================================
def run_per_layer_same_pos(model, tokenizer, device):
    from activation_robustness.perturbations.typo import AdjacentKey
    from activation_robustness.data.external import sample_openorca

    log("\n" + "=" * 60)
    log("EXPERIMENT 6: Per-layer same-pos check")
    log("=" * 60)

    OUT = RESULTS_BASE / "per_layer_same_pos"
    OUT.mkdir(parents=True, exist_ok=True)

    adj_key = AdjacentKey()
    rng = np.random.default_rng(SEED_BASE + 6)

    orca = sample_openorca(n_per_bucket=100, buckets={"100-300": (100, 300)}, seed=SEED_BASE + 6)
    prompts = [p["text"] for p in orca[:100]]

    same_pos_cos = {L: [] for L in range(N_LAYERS_TOTAL)}
    cross_pos_cos = {L: [] for L in range(N_LAYERS_TOTAL)}
    n_groups = 0

    for pi, prompt_text in enumerate(prompts):
        orig_fmt = format_prompt(tokenizer, prompt_text)
        orig_hs, seq_len = extract_all_layers(model, tokenizer, orig_fmt, device)
        orig_ids = tokenizer.encode(orig_fmt)

        variants = adj_key.enumerate_all(prompt_text)
        if not variants:
            continue
        by_cp = defaultdict(list)
        for pt, meta in variants:
            by_cp[meta["position"]].append((pt, meta))

        # Find positions with 3+ valid same-token replacements
        groups = []
        for cp in sorted(by_cp.keys()):
            cands = by_cp[cp]
            valid = []
            token_pos = None
            for pt, meta in cands:
                pf = format_prompt(tokenizer, pt)
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
                    continue
                valid.append((pt, meta))
            if len(valid) >= 3 and token_pos is not None:
                groups.append((token_pos, valid[:3]))

        if len(groups) > 4:
            step = len(groups) // 4
            groups = groups[::step][:4]
        if not groups:
            continue

        group_deltas = []
        for token_pos, replacements in groups:
            layer_deltas = {L: [] for L in range(N_LAYERS_TOTAL)}
            for pt, meta in replacements:
                pf = format_prompt(tokenizer, pt)
                pert_hs, pl = extract_all_layers(model, tokenizer, pf, device)
                if pl != seq_len:
                    continue
                for L in range(N_LAYERS_TOTAL):
                    oc = orig_hs[L + 1][token_pos] - orig_hs[L][token_pos]
                    pc = pert_hs[L + 1][token_pos] - pert_hs[L][token_pos]
                    layer_deltas[L].append(pc - oc)
            if all(len(layer_deltas[L]) == 3 for L in range(N_LAYERS_TOTAL)):
                group_deltas.append(layer_deltas)
                n_groups += 1

        for gd in group_deltas:
            for L in range(N_LAYERS_TOTAL):
                ds = gd[L]
                for i in range(len(ds)):
                    for j in range(i + 1, len(ds)):
                        n1, n2 = np.linalg.norm(ds[i]), np.linalg.norm(ds[j])
                        if n1 > 1e-10 and n2 > 1e-10:
                            same_pos_cos[L].append(float(np.dot(ds[i], ds[j]) / (n1 * n2)))

        if len(group_deltas) >= 2:
            for L in range(N_LAYERS_TOTAL):
                for gi in range(len(group_deltas)):
                    for gj in range(gi + 1, len(group_deltas)):
                        d1, d2 = group_deltas[gi][L][0], group_deltas[gj][L][0]
                        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                        if n1 > 1e-10 and n2 > 1e-10:
                            cross_pos_cos[L].append(float(np.dot(d1, d2) / (n1 * n2)))

        if (pi + 1) % 10 == 0:
            log(f"    [{pi+1}] groups: {n_groups}")

    results = {"n_groups": n_groups, "per_layer": {}}
    log(f"\n  {'Layer':>5s} {'same_pos':>10s} {'cross_pos':>10s}")
    for L in range(N_LAYERS_TOTAL):
        sp = same_pos_cos[L]
        cp = cross_pos_cos[L]
        results["per_layer"][L] = {
            "same_pos_mean": float(np.mean(sp)) if sp else 0,
            "same_pos_std": float(np.std(sp)) if sp else 0,
            "cross_pos_mean": float(np.mean(cp)) if cp else 0,
            "n_same": len(sp), "n_cross": len(cp),
        }
        log(f"  {L:>5d} {np.mean(sp) if sp else 0:>10.4f} {np.mean(cp) if cp else 0:>10.4f}")

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"  Saved to {OUT}")


# =====================================================================
# EXPERIMENT 7: Typo direction (probe impact + per-layer)
# =====================================================================
def run_typo_direction(model, tokenizer, device):
    from activation_robustness.perturbations.typo import AdjacentKey
    from activation_robustness.data.external import sample_openorca

    log("\n" + "=" * 60)
    log("EXPERIMENT 7: Typo direction (probe impact)")
    log("=" * 60)

    OUT = RESULTS_BASE / "typo_direction"
    OUT.mkdir(parents=True, exist_ok=True)

    adj_key = AdjacentKey()
    rng = np.random.default_rng(SEED_BASE + 7)
    MAX_DOWNSTREAM = 30

    orca = sample_openorca(n_per_bucket=50, buckets={"50-100": (50, 100)}, seed=SEED_BASE + 7)
    prompts = [p["text"] for p in orca[:50]]
    log(f"  {len(prompts)} prompts")

    all_site_deltas = []
    all_decay = []

    for pi, prompt_text in enumerate(prompts):
        orig_fmt = format_prompt(tokenizer, prompt_text)
        orig_seq, seq_len = extract_full_seq(model, tokenizer, orig_fmt, device)
        sites = get_typo_sites(tokenizer, orig_fmt, prompt_text, adj_key, rng, max_sites=8)
        if not sites:
            continue

        for site in sites:
            tp = site["token_pos"]
            pt = apply_typo(prompt_text, site["char_pos"], site["rep"])
            pf = format_prompt(tokenizer, pt)
            pert_seq, pl = extract_full_seq(model, tokenizer, pf, device)
            if pl != seq_len:
                continue

            L = LAYERS[-1]  # last layer for direction analysis
            if tp < orig_seq[L].shape[0]:
                delta = pert_seq[L][tp] - orig_seq[L][tp]
                all_site_deltas.append(delta)

            # Decay at last layer
            common = min(orig_seq[L].shape[0], pert_seq[L].shape[0])
            if tp < common and common - tp >= MAX_DOWNSTREAM:
                norms, act_cos, angles = [], [], []
                for off in range(MAX_DOWNSTREAM):
                    pos = tp + off
                    h_o, h_p = orig_seq[L][pos], pert_seq[L][pos]
                    d = h_p - h_o
                    norms.append(float(np.linalg.norm(d)))
                    hon, hpn = np.linalg.norm(h_o), np.linalg.norm(h_p)
                    if hon > 1e-10 and hpn > 1e-10:
                        c = float(np.dot(h_o, h_p) / (hon * hpn))
                        c = max(-1.0, min(1.0, c))
                        act_cos.append(c)
                        angles.append(float(np.degrees(np.arccos(c))))
                    else:
                        act_cos.append(1.0)
                        angles.append(0.0)
                all_decay.append({"norms": norms, "act_cosines": act_cos, "angular_shifts": angles})

        if (pi + 1) % 10 == 0:
            log(f"    [{pi+1}] deltas: {len(all_site_deltas)}, decay: {len(all_decay)}")

    # Analysis
    results = {"n_deltas": len(all_site_deltas), "n_decay": len(all_decay)}

    if all_site_deltas:
        arr = np.array(all_site_deltas)
        mu = arr.mean(axis=0)
        mu_hat = mu / (np.linalg.norm(mu) + 1e-10)
        per_cos = [abs(float(np.dot(arr[i], mu_hat) / (np.linalg.norm(arr[i]) + 1e-10))) for i in range(len(arr))]
        results["universal_direction"] = {
            "mean_abs_cos": float(np.mean(per_cos)),
            "mu_ratio": float(np.linalg.norm(mu) / (np.linalg.norm(arr, axis=1).mean() + 1e-10)),
        }
        log(f"  Universal direction: |cos|={np.mean(per_cos):.4f}")

    if all_decay:
        avg_ang = np.mean([d["angular_shifts"] for d in all_decay], axis=0)
        avg_cos = np.mean([d["act_cosines"] for d in all_decay], axis=0)
        results["decay"] = {
            "avg_angular": avg_ang.tolist(),
            "avg_act_cosine": avg_cos.tolist(),
        }
        log(f"  Decay: site={avg_ang[0]:.1f}°, +1={avg_ang[1]:.1f}°, +5={avg_ang[min(5,len(avg_ang)-1)]:.1f}°, "
            f"+20={avg_ang[min(20,len(avg_ang)-1)]:.1f}°")

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"  Saved to {OUT}")


# =====================================================================
# EXPERIMENT 8: Long-prompt decay
# =====================================================================
def run_long_decay(model, tokenizer, device):
    from activation_robustness.perturbations.typo import AdjacentKey
    from activation_robustness.data.external import sample_openorca

    log("\n" + "=" * 60)
    log("EXPERIMENT 8: Long-prompt typo decay")
    log("=" * 60)

    OUT = RESULTS_BASE / "long_decay"
    OUT.mkdir(parents=True, exist_ok=True)

    adj_key = AdjacentKey()
    rng = np.random.default_rng(SEED_BASE + 8)
    MAX_DOWNSTREAM = 200
    L = LAYERS[-1]

    buckets = {"200-500": (200, 500), "500-1000": (500, 1000)}
    orca = sample_openorca(n_per_bucket=100, buckets=buckets, seed=SEED_BASE + 8, scout_size=100000)
    by_bucket = defaultdict(list)
    for p in orca:
        by_bucket[p["length_bucket"]].append(p["text"])

    all_decay = defaultdict(list)

    for bname, prompts in sorted(by_bucket.items()):
        log(f"  Bucket {bname}: {len(prompts)} prompts")
        for pi, prompt_text in enumerate(prompts[:100]):
            orig_fmt = format_prompt(tokenizer, prompt_text)
            orig_seq, seq_len = extract_full_seq(model, tokenizer, orig_fmt, device)

            # Find early typo site
            sites = get_typo_sites(tokenizer, orig_fmt, prompt_text, adj_key, rng)
            early = [s for s in sites if s["token_pos"] < seq_len * 0.3]
            if not early:
                continue
            site = early[rng.integers(len(early))]
            tp = site["token_pos"]
            pt = apply_typo(prompt_text, site["char_pos"], site["rep"])
            pf = format_prompt(tokenizer, pt)
            pert_seq, pl = extract_full_seq(model, tokenizer, pf, device)
            if pl != seq_len:
                continue

            common = min(orig_seq[L].shape[0], pert_seq[L].shape[0])
            if tp >= common:
                continue

            norms, angles = [], []
            site_norm = float(np.linalg.norm(pert_seq[L][tp] - orig_seq[L][tp]))
            if site_norm < 1e-10:
                continue

            for off in range(min(common - tp, MAX_DOWNSTREAM)):
                pos = tp + off
                h_o, h_p = orig_seq[L][pos], pert_seq[L][pos]
                d = h_p - h_o
                norms.append(float(np.linalg.norm(d)))
                hon, hpn = np.linalg.norm(h_o), np.linalg.norm(h_p)
                if hon > 1e-10 and hpn > 1e-10:
                    c = float(np.dot(h_o, h_p) / (hon * hpn))
                    c = max(-1.0, min(1.0, c))
                    angles.append(float(np.degrees(np.arccos(c))))
                else:
                    angles.append(0.0)

            all_decay[bname].append({"norms": norms, "angles": angles, "site_norm": site_norm,
                                      "downstream_room": len(norms)})

            if (pi + 1) % 20 == 0:
                log(f"    [{pi+1}]")

    summary = {}
    for bname, entries in sorted(all_decay.items()):
        min_len = min(e["downstream_room"] for e in entries) if entries else 0
        min_len = min(min_len, MAX_DOWNSTREAM)
        curves = [e["angles"][:min_len] for e in entries if len(e["angles"]) >= min_len]
        if not curves:
            continue
        avg = np.mean(curves, axis=0)
        summary[bname] = {
            "n": len(entries), "curve_len": min_len,
            "avg_angular": avg.tolist(),
        }
        log(f"  {bname}: N={len(entries)}, site={avg[0]:.1f}°, +5={avg[min(5,len(avg)-1)]:.1f}°, "
            f"+50={avg[min(50,len(avg)-1)]:.1f}°, +100={avg[min(100,len(avg)-1)]:.1f}°")

    with open(OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Saved to {OUT}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", metavar="EXP", default=None,
                        help="Run only this experiment (e.g. per_layer_same_pos). "
                             "Choices: e033, baseline, e028, e041, "
                             "per_layer_direction, per_layer_same_pos, "
                             "typo_direction, long_decay")
    args = parser.parse_args()

    def should_run(name):
        return args.only is None or args.only == name

    log(f"{'=' * 60}")
    log(f"OVERNIGHT RUN: All experiments on {MODEL_NAME}")
    log(f"Results: {RESULTS_BASE}")
    log(f"Layers: {LAYERS} (of {N_LAYERS_TOTAL})")
    if args.only:
        log(f"Running only: {args.only}")
    log(f"{'=' * 60}")

    model, tokenizer, device = load_model()

    try:
        if should_run("e033"):              run_e033(model, tokenizer, device)
        if should_run("baseline"):          run_baseline(model, tokenizer, device)
        if should_run("e028"):              run_e028(model, tokenizer, device)
        if should_run("e041"):              run_e041(model, tokenizer, device)
        if should_run("per_layer_direction"): run_per_layer_direction(model, tokenizer, device)
        if should_run("per_layer_same_pos"):  run_per_layer_same_pos(model, tokenizer, device)
        if should_run("typo_direction"):    run_typo_direction(model, tokenizer, device)
        if should_run("long_decay"):        run_long_decay(model, tokenizer, device)
    finally:
        free_model(model)

    # Write comparison summary
    log("\nWriting comparison summary...")
    comparison = {
        "model": MODEL_NAME,
        "n_layers": N_LAYERS_TOTAL,
        "d_model": D_MODEL,
        "layers_sampled": LAYERS,
    }
    for exp_name in ["e033_length_controlled", "baseline_variance", "e028_position_sensitivity",
                      "e041_multi_perturbation", "per_layer_direction", "per_layer_same_pos",
                      "typo_direction", "long_decay"]:
        rpath = RESULTS_BASE / exp_name / "results.json"
        if rpath.exists():
            with open(rpath) as f:
                comparison[exp_name] = json.load(f)

    with open(RESULTS_BASE / "comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)
    log(f"Saved comparison summary to {RESULTS_BASE / 'comparison_summary.json'}")

    log(f"\n{'=' * 60}")
    log(f"ALL EXPERIMENTS COMPLETE")
    log(f"Results in: {RESULTS_BASE}")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
