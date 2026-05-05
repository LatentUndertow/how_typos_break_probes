#!/usr/bin/env python3
"""
Compute KV-fork LODO metrics from `kv_fork/lodo/fold_*/scores.npz`.

Compares two suffix conditions (`neutral`, `intent`) to the no-fork
linear-pos-5 baseline already computed in
`perturbation_lodo_results.json`. Reports:
  - Clean TPR@FPR=1% (recalibrated per fold/condition; mixed folds)
  - Bundle perturbation Δ TPR@FPR=1%
  - Cross-fold variance of Δ TPR (KV-fork claim is variance-tightening)
  - Pooled and mean AUC across mixed folds

Inputs:
  - `kv_fork/lodo/fold_<NAME>/scores.npz` keys:
      neutral_clean_scores, neutral_full_bundle_scores,
      intent_clean_scores,  intent_full_bundle_scores,
      *_valid (per-sample valid mask), y_test, test_pids
  - validation_report.json for fold class composition

The KV-fork eval script trains a fresh sklearn LogisticRegression
linear probe at pos=-5; the no-fork baseline uses the LODO sweep's
linear pos-5 probe (different training pipeline, same architecture
and readout position). The §7.3 in-distribution comparison treats
these as comparable, and we follow that convention OOD.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

EXPORT_ROOT = Path("./lodo_data/extracted/lodo_results_export")
KVF_ROOT = EXPORT_ROOT / "kv_fork" / "lodo"
VALIDATION_REPORT = Path(__file__).parent / "validation_report.json"
NO_FORK_RESULTS = Path(__file__).parent / "perturbation_lodo_results.json"
OUT_PATH = Path(__file__).parent / "kvfork_lodo_results.json"

SUFFIXES = ["neutral", "intent"]
CONDITIONS = ["clean", "full_bundle"]
TARGET_FPR = 0.01


def tpr_at_fpr(scores, labels, target_fpr):
    fpr, tpr, thr = roc_curve(labels, scores)
    idx = max(int(np.searchsorted(fpr, target_fpr, side="right") - 1), 0)
    return float(tpr[idx]), float(fpr[idx]), float(thr[idx])


def stat(arr):
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(a.mean()), "std": float(a.std()),
        "min": float(a.min()), "max": float(a.max()),
        "n": int(len(a)),
    }


def main() -> int:
    if not VALIDATION_REPORT.exists():
        print("ERROR: run 01_validate_integrity.py first.", file=sys.stderr)
        return 1
    if not NO_FORK_RESULTS.exists():
        print("ERROR: run 03_compute_perturbation_lodo.py first.", file=sys.stderr)
        return 1

    report = json.loads(VALIDATION_REPORT.read_text())

    # Audit KV-fork data against README
    audit = []
    for fold, e in sorted(report["folds"].items()):
        path = KVF_ROOT / f"fold_{fold}" / "scores.npz"
        if not path.exists():
            audit.append({"fold": fold, "status": "missing"})
            continue
        d = np.load(path)
        n = int(len(d["y_test"]))
        n_mal = int(d["y_test"].sum())
        exp = e["expected"]
        match = (n == exp["n"] and n_mal == exp["n_mal"])
        audit.append({
            "fold": fold,
            "status": "ok" if match else "mismatch",
            "expected_n": exp["n"], "expected_mal": exp["n_mal"],
            "actual_n": n, "actual_mal": n_mal,
        })

    # Mixed folds with valid kv_fork data
    mixed_folds = sorted(
        a["fold"] for a in audit
        if a["status"] == "ok"
        and report["folds"][a["fold"]]["expected"]["n_mal"] > 0
        and report["folds"][a["fold"]]["expected"]["n_ben"] > 0
    )
    print(f"[04_kvfork] Using {len(mixed_folds)} mixed folds with KV-fork data: {mixed_folds}")

    # Per-suffix per-condition aggregates
    fix_a: dict = {sfx: {c: [] for c in CONDITIONS} for sfx in SUFFIXES}
    fix_b_tpr: dict = {sfx: {c: [] for c in CONDITIONS} for sfx in SUFFIXES}
    fix_b_fpr: dict = {sfx: {c: [] for c in CONDITIONS} for sfx in SUFFIXES}
    aucs: dict = {sfx: {c: [] for c in CONDITIONS} for sfx in SUFFIXES}
    pooled_scores: dict = {sfx: {c: [] for c in CONDITIONS} for sfx in SUFFIXES}
    pooled_labels: dict = {sfx: {c: [] for c in CONDITIONS} for sfx in SUFFIXES}
    per_fold = {f: {} for f in mixed_folds}

    for fold in mixed_folds:
        d = np.load(KVF_ROOT / f"fold_{fold}" / "scores.npz")
        labels = d["y_test"].astype(int)

        for sfx in SUFFIXES:
            cond_scores = {}
            for c in CONDITIONS:
                key = f"{sfx}_{c}_scores"
                vkey = f"{sfx}_{c}_valid"
                sc = d[key]
                vmask = d[vkey].astype(bool) if vkey in d else np.ones_like(sc, dtype=bool)
                vmask = vmask & ~np.isnan(sc)
                cond_scores[c] = (sc, vmask)

                # Fix A
                tpr_, _, _ = tpr_at_fpr(sc[vmask], labels[vmask], TARGET_FPR)
                fix_a[sfx][c].append(tpr_)
                # AUC + pooled
                au = float(roc_auc_score(labels[vmask], sc[vmask]))
                aucs[sfx][c].append(au)
                pooled_scores[sfx][c].append(sc[vmask])
                pooled_labels[sfx][c].append(labels[vmask])

            # Fix B: clean threshold transferred
            cln_sc, cln_v = cond_scores["clean"]
            _, _, T = tpr_at_fpr(cln_sc[cln_v], labels[cln_v], TARGET_FPR)
            for c in CONDITIONS:
                sc, v = cond_scores[c]
                preds = sc[v] >= T
                lbv = labels[v]
                tpr_t = float(preds[lbv == 1].mean()) if (lbv == 1).any() else float("nan")
                fpr_t = float(preds[lbv == 0].mean()) if (lbv == 0).any() else float("nan")
                fix_b_tpr[sfx][c].append(tpr_t)
                fix_b_fpr[sfx][c].append(fpr_t)

            per_fold[fold][sfx] = {
                "tpr_clean": fix_a[sfx]["clean"][-1],
                "tpr_bundle": fix_a[sfx]["full_bundle"][-1],
                "delta_bundle": fix_a[sfx]["full_bundle"][-1] - fix_a[sfx]["clean"][-1],
                "deploy_threshold": T,
                "deploy_tpr_bundle": fix_b_tpr[sfx]["full_bundle"][-1],
                "deploy_fpr_bundle": fix_b_fpr[sfx]["full_bundle"][-1],
            }

    out: dict = {
        "target_fpr": TARGET_FPR,
        "audit": audit,
        "mixed_folds_used": mixed_folds,
        "fix_a_recalibrated_tpr_at_fpr1pct": {},
        "fix_b_deployment_view": {},
        "auc_summary": {},
        "per_fold": per_fold,
    }

    for sfx in SUFFIXES:
        cln_a = np.array(fix_a[sfx]["clean"])
        fb_a = np.array(fix_a[sfx]["full_bundle"])
        cln_t = np.array(fix_b_tpr[sfx]["clean"])
        fb_t = np.array(fix_b_tpr[sfx]["full_bundle"])
        cln_f = np.array(fix_b_fpr[sfx]["clean"])
        fb_f = np.array(fix_b_fpr[sfx]["full_bundle"])
        cln_au = np.array(aucs[sfx]["clean"])
        fb_au = np.array(aucs[sfx]["full_bundle"])
        ps_cln = np.concatenate(pooled_scores[sfx]["clean"])
        pl_cln = np.concatenate(pooled_labels[sfx]["clean"])
        ps_fb = np.concatenate(pooled_scores[sfx]["full_bundle"])
        pl_fb = np.concatenate(pooled_labels[sfx]["full_bundle"])

        out["fix_a_recalibrated_tpr_at_fpr1pct"][sfx] = {
            "tpr_clean": stat(cln_a),
            "tpr_bundle": stat(fb_a),
            "delta_bundle": stat(fb_a - cln_a),
        }
        out["fix_b_deployment_view"][sfx] = {
            "tpr_bundle": stat(fb_t),
            "fpr_bundle": stat(fb_f),
            "delta_tpr_bundle": stat(fb_t - cln_t),
            "delta_fpr_bundle": stat(fb_f - cln_f),
        }
        out["auc_summary"][sfx] = {
            "mean_auc_clean": stat(cln_au),
            "mean_auc_bundle": stat(fb_au),
            "pooled_auc_clean": float(roc_auc_score(pl_cln, ps_cln)),
            "pooled_auc_bundle": float(roc_auc_score(pl_fb, ps_fb)),
        }

    # Pull no-fork comparison from script 03's output (linear pos-5)
    no_fork = json.loads(NO_FORK_RESULTS.read_text())
    nf_a = no_fork["fix_a_recalibrated_tpr_at_fpr1pct"]["positional_linear_pos-5"]
    nf_b = no_fork["fix_b_deployment_view"]["positional_linear_pos-5"]
    out["no_fork_baseline"] = {
        "fix_a": {
            "tpr_clean": nf_a["tpr_clean"],
            "tpr_bundle": nf_a["tpr_bundle"],
            "delta_bundle": nf_a["delta_bundle"],
        },
        "fix_b": {
            "tpr_bundle": nf_b["tpr_bundle"],
            "fpr_bundle": nf_b["fpr_bundle"],
            "delta_tpr_bundle": nf_b["delta_tpr_bundle"],
            "delta_fpr_bundle": nf_b["delta_fpr_bundle"],
        },
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))

    # Console summary
    print(f"\n=== Fix A: TPR@FPR={TARGET_FPR:.0%} clean / bundle / Δ ===")
    print(f"{'Condition':<22} {'clean':>10} {'bundle':>10} {'Δ bundle (mean±std)':>26} {'Δ range':>22}")
    print(f"{'No fork (linear pos-5)':<22} "
          f"{100*nf_a['tpr_clean']['mean']:>8.2f}%  "
          f"{100*nf_a['tpr_bundle']['mean']:>8.2f}%  "
          f"{100*nf_a['delta_bundle']['mean']:>+9.2f} ± {100*nf_a['delta_bundle']['std']:>5.2f}pp  "
          f"[{100*nf_a['delta_bundle']['min']:+.1f},{100*nf_a['delta_bundle']['max']:+.1f}]pp")
    for sfx in SUFFIXES:
        r = out["fix_a_recalibrated_tpr_at_fpr1pct"][sfx]
        print(f"{'KV-fork ' + sfx:<22} "
              f"{100*r['tpr_clean']['mean']:>8.2f}%  "
              f"{100*r['tpr_bundle']['mean']:>8.2f}%  "
              f"{100*r['delta_bundle']['mean']:>+9.2f} ± {100*r['delta_bundle']['std']:>5.2f}pp  "
              f"[{100*r['delta_bundle']['min']:+.1f},{100*r['delta_bundle']['max']:+.1f}]pp")

    print(f"\n=== Fix B: deployment view (clean-calibrated T) ===")
    print(f"{'Condition':<22} {'TPR bundle':>11} {'FPR bundle':>11} {'ΔTPR':>9} {'ΔFPR':>9}")
    print(f"{'No fork (linear pos-5)':<22} "
          f"{100*nf_b['tpr_bundle']['mean']:>9.2f}%  "
          f"{100*nf_b['fpr_bundle']['mean']:>9.2f}%  "
          f"{100*nf_b['delta_tpr_bundle']['mean']:>+7.2f}pp  "
          f"{100*nf_b['delta_fpr_bundle']['mean']:>+7.2f}pp")
    for sfx in SUFFIXES:
        r = out["fix_b_deployment_view"][sfx]
        print(f"{'KV-fork ' + sfx:<22} "
              f"{100*r['tpr_bundle']['mean']:>9.2f}%  "
              f"{100*r['fpr_bundle']['mean']:>9.2f}%  "
              f"{100*r['delta_tpr_bundle']['mean']:>+7.2f}pp  "
              f"{100*r['delta_fpr_bundle']['mean']:>+7.2f}pp")

    print(f"\n=== AUC summary (mean ± std across mixed folds) ===")
    nf_pooled_auc = (
        no_fork["per_fold"]  # not directly stored; recompute below
    )
    # AUC was stored in fix_a but separately; recompute simple summary
    # from script 03's clean_lodo equivalents would require its JSON;
    # here we just print KV-fork.
    for sfx in SUFFIXES:
        a = out["auc_summary"][sfx]
        print(f"  KV-fork {sfx:<10s} clean: pooled={a['pooled_auc_clean']:.3f} "
              f"mean={a['mean_auc_clean']['mean']:.3f}±{a['mean_auc_clean']['std']:.3f}  "
              f"bundle: pooled={a['pooled_auc_bundle']:.3f} "
              f"mean={a['mean_auc_bundle']['mean']:.3f}±{a['mean_auc_bundle']['std']:.3f}")

    print(f"\n  Output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
