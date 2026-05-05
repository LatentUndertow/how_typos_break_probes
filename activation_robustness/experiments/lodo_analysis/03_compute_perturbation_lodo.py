#!/usr/bin/env python3
"""
Compute LODO perturbation metrics from the validated perteval directory.

Outputs the numbers feeding paper App G Tables G.3 (TPR@FPR=1%
recalibrated per fold/condition; apples-to-apples with §6) and
G.4 (deployment view: clean threshold transferred to perturbed).

The perturbation analysis runs only on mixed-class folds (where AUC
and TPR@FPR=1% are well-defined). We use the README class distribution
to identify which folds are mixed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

VALIDATED_DIR = Path("./lodo_data/validated_perteval")
VALIDATION_REPORT = Path(__file__).parent / "validation_report.json"
OUT_PATH = Path(__file__).parent / "perturbation_lodo_results.json"

PROBES = [
    "positional_linear_pos-5",
    "mean_linear_last16",
    "mlp_all",
    "attention_all",
    "multimax_all",
]
LABELS = {
    "positional_linear_pos-5": "Linear (EOT)",
    "mean_linear_last16": "Mean Linear (last 16)",
    "mlp_all": "MLP (all)",
    "attention_all": "Attention (all)",
    "multimax_all": "MultiMax (all)",
}
CONDITIONS = ["clean", "full_bundle", "every_second_word"]
TARGET_FPR = 0.01


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> tuple[float, float, float]:
    """TPR, actual FPR, threshold at the largest sweep point with FPR <= target."""
    fpr, tpr, thr = roc_curve(labels, scores)
    idx = int(np.searchsorted(fpr, target_fpr, side="right") - 1)
    idx = max(idx, 0)
    return float(tpr[idx]), float(fpr[idx]), float(thr[idx])


def main() -> int:
    if not VALIDATION_REPORT.exists():
        print("ERROR: run 01_validate_integrity.py first.", file=sys.stderr)
        return 1
    report = json.loads(VALIDATION_REPORT.read_text())

    # Use only mixed-class folds with validated perteval data.
    valid_perteval = {
        f: e for f, e in report["folds"].items()
        if e.get("decision", {}).get("source") in ("perteval_clean", "legacy_override")
    }
    mixed_folds = sorted(
        f for f, e in valid_perteval.items()
        if e["expected"]["n_mal"] > 0 and e["expected"]["n_ben"] > 0
    )
    print(f"[03_pert] Using {len(mixed_folds)} mixed-class folds "
          f"with validated perteval data: {mixed_folds}")

    results: dict = {
        "target_fpr": TARGET_FPR,
        "n_folds_used": len(mixed_folds),
        "folds_used": mixed_folds,
        "fix_a_recalibrated_tpr_at_fpr1pct": {},
        "fix_b_deployment_view": {},
        "per_fold": {f: {} for f in mixed_folds},
    }

    for probe in PROBES:
        # Fix A: per-condition recalibrated TPR@FPR=1%
        cond_tprs: dict[str, list[float]] = {c: [] for c in CONDITIONS}
        # Fix B: clean threshold applied to all conditions
        deploy_tpr: dict[str, list[float]] = {c: [] for c in CONDITIONS}
        deploy_fpr: dict[str, list[float]] = {c: [] for c in CONDITIONS}

        for fold in mixed_folds:
            path = VALIDATED_DIR / f"fold_{fold}_scores.npz"
            d = np.load(path)
            labels = d["labels"]
            results["per_fold"][fold].setdefault("class", "mixed")
            results["per_fold"][fold].setdefault("n", int(len(labels)))
            results["per_fold"][fold].setdefault("n_mal", int(labels.sum()))

            # Per-fold scores per condition.
            cond_scores = {c: d[f"{probe}_{c}"] for c in CONDITIONS}

            # Fix A: recalibrate per condition.
            for cond in CONDITIONS:
                sc = cond_scores[cond]
                v = ~np.isnan(sc)
                tpr, _, _ = tpr_at_fpr(sc[v], labels[v], TARGET_FPR)
                cond_tprs[cond].append(tpr)

            # Fix B: threshold T calibrated on clean fold, apply to all conditions.
            cln = cond_scores["clean"]
            vc = ~np.isnan(cln)
            _, _, T = tpr_at_fpr(cln[vc], labels[vc], TARGET_FPR)

            for cond in CONDITIONS:
                sc = cond_scores[cond]
                v = ~np.isnan(sc)
                preds = sc[v] >= T
                lbv = labels[v]
                if (lbv == 1).any():
                    deploy_tpr[cond].append(float(preds[lbv == 1].mean()))
                else:
                    deploy_tpr[cond].append(float("nan"))
                if (lbv == 0).any():
                    deploy_fpr[cond].append(float(preds[lbv == 0].mean()))
                else:
                    deploy_fpr[cond].append(float("nan"))

            # Stash per-fold details
            results["per_fold"][fold][probe] = {
                "fix_a_tpr_at_fpr1": {
                    c: cond_tprs[c][-1] for c in CONDITIONS
                },
                "fix_b_threshold_clean_calibrated": T,
                "fix_b_tpr_per_condition": {
                    c: deploy_tpr[c][-1] for c in CONDITIONS
                },
                "fix_b_fpr_per_condition": {
                    c: deploy_fpr[c][-1] for c in CONDITIONS
                },
            }

        def stat(arr):
            a = np.array(arr)
            return {"mean": float(a.mean()), "std": float(a.std()),
                    "min": float(a.min()), "max": float(a.max())}

        # Fix A summary
        cln_arr = np.array(cond_tprs["clean"])
        fb_arr = np.array(cond_tprs["full_bundle"])
        esw_arr = np.array(cond_tprs["every_second_word"])
        results["fix_a_recalibrated_tpr_at_fpr1pct"][probe] = {
            "label": LABELS[probe],
            "tpr_clean": stat(cln_arr),
            "tpr_bundle": stat(fb_arr),
            "tpr_esw": stat(esw_arr),
            "delta_bundle": stat(fb_arr - cln_arr),
            "delta_esw": stat(esw_arr - cln_arr),
        }

        # Fix B summary
        cln_t = np.array(deploy_tpr["clean"])
        fb_t = np.array(deploy_tpr["full_bundle"])
        esw_t = np.array(deploy_tpr["every_second_word"])
        cln_f = np.array(deploy_fpr["clean"])
        fb_f = np.array(deploy_fpr["full_bundle"])
        esw_f = np.array(deploy_fpr["every_second_word"])
        results["fix_b_deployment_view"][probe] = {
            "label": LABELS[probe],
            "tpr_clean": stat(cln_t),
            "tpr_bundle": stat(fb_t),
            "tpr_esw": stat(esw_t),
            "fpr_clean": stat(cln_f),
            "fpr_bundle": stat(fb_f),
            "fpr_esw": stat(esw_f),
            "delta_tpr_bundle": stat(fb_t - cln_t),
            "delta_tpr_esw": stat(esw_t - cln_t),
            "delta_fpr_bundle": stat(fb_f - cln_f),
            "delta_fpr_esw": stat(esw_f - cln_f),
        }

    OUT_PATH.write_text(json.dumps(results, indent=2))

    # Console: Fix A
    print(f"\n=== Fix A: TPR@FPR={TARGET_FPR:.0%} recalibrated per condition ===")
    print(f"{'Architecture':<22} {'TPR clean':>11} {'TPR bundle':>12} {'TPR ESW':>10} "
          f"{'Δ bundle':>11} {'Δ ESW':>11}")
    for probe in PROBES:
        r = results["fix_a_recalibrated_tpr_at_fpr1pct"][probe]
        print(f"{r['label']:<22} "
              f"{100*r['tpr_clean']['mean']:>9.2f}%  "
              f"{100*r['tpr_bundle']['mean']:>10.2f}%  "
              f"{100*r['tpr_esw']['mean']:>8.2f}%  "
              f"{100*r['delta_bundle']['mean']:>+9.2f}pp  "
              f"{100*r['delta_esw']['mean']:>+9.2f}pp")

    # Console: Fix B
    print(f"\n=== Fix B: deployment view (clean-calibrated T, transferred) ===")
    print(f"{'Architecture':<22} {'ESW TPR':>9} {'ESW FPR':>9} "
          f"{'ΔTPR ESW':>11} {'ΔFPR ESW':>11}")
    for probe in PROBES:
        r = results["fix_b_deployment_view"][probe]
        print(f"{r['label']:<22} "
              f"{100*r['tpr_esw']['mean']:>7.2f}%  "
              f"{100*r['fpr_esw']['mean']:>7.2f}%  "
              f"{100*r['delta_tpr_esw']['mean']:>+9.2f}pp  "
              f"{100*r['delta_fpr_esw']['mean']:>+9.2f}pp")

    print(f"\n  Output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
