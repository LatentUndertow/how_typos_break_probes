#!/usr/bin/env python3
"""
Compute clean LODO metrics from canonical per-fold test_scores.npz files.

Outputs the numbers feeding paper App G Tables G.1 (per-architecture
summary) and G.2 (per-dataset breakdown). Uses canonical
`fold_<NAME>/<probe>/test_scores.npz` -- NOT perteval_clean -- so the
perteval consolidation issues do not affect these numbers.

Metrics (following Fomin et al. 2026 for ACC at threshold=0.5):
  - Per-fold accuracy at threshold 0.5
  - Per-fold AUC (mixed-class folds only)
  - Cross-fold mean +- std accuracy (n=29)
  - Cross-fold mean +- std AUC (n=7 mixed)
  - Sample-weighted accuracy across all folds
  - Pooled AUC across the mixed-fold predictions

The README dataset metadata is the ground truth for class composition.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

EXPORT_ROOT = Path("./lodo_data/extracted/lodo_results_export")
VALIDATION_REPORT = Path(__file__).parent / "validation_report.json"
OUT_PATH = Path(__file__).parent / "clean_lodo_results.json"

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
THRESHOLD = 0.5


def main() -> int:
    if not VALIDATION_REPORT.exists():
        print(f"ERROR: run 01_validate_integrity.py first.", file=sys.stderr)
        return 1
    report = json.loads(VALIDATION_REPORT.read_text())

    # Use only canonically-validated folds.
    folds = sorted(
        f for f, e in report["folds"].items()
        if e.get("canonical_check", {}).get("matches_readme")
    )
    print(f"[02_clean] Using {len(folds)} canonically-validated folds")

    results: dict = {
        "threshold": THRESHOLD,
        "n_folds_used": len(folds),
        "folds_used": folds,
        "per_probe": {},
        "per_fold": {f: {} for f in folds},
    }

    for probe in PROBES:
        per_fold_acc: list[float] = []
        per_fold_auc: list[float] = []
        weighted_num = 0.0
        weighted_den = 0
        pooled_scores: list[np.ndarray] = []
        pooled_labels: list[np.ndarray] = []

        for fold in folds:
            path = EXPORT_ROOT / f"fold_{fold}" / probe / "test_scores.npz"
            if not path.exists():
                continue
            d = np.load(path)
            scores = d["scores"]
            labels = d["labels"]
            n = len(labels)
            n_mal = int(labels.sum())
            n_ben = n - n_mal

            acc = float(((scores >= THRESHOLD) == labels).mean())
            per_fold_acc.append(acc)
            weighted_num += acc * n
            weighted_den += n

            row = {
                "n": int(n),
                "n_mal": n_mal,
                "n_ben": n_ben,
                "class_type": (
                    "mixed" if n_mal > 0 and n_ben > 0
                    else ("malicious" if n_mal > 0 else "benign")
                ),
                "accuracy": acc,
            }

            if n_mal > 0 and n_ben > 0:
                auc = float(roc_auc_score(labels, scores))
                per_fold_auc.append(auc)
                pooled_scores.append(scores)
                pooled_labels.append(labels)
                row["auc"] = auc
            results["per_fold"][fold][probe] = row

        accs = np.array(per_fold_acc)
        aucs = np.array(per_fold_auc)
        pooled_scores_arr = np.concatenate(pooled_scores) if pooled_scores else np.array([])
        pooled_labels_arr = np.concatenate(pooled_labels) if pooled_labels else np.array([])
        pooled_auc = (
            float(roc_auc_score(pooled_labels_arr, pooled_scores_arr))
            if len(pooled_scores_arr) else None
        )

        summary = {
            "label": LABELS[probe],
            "n_folds": int(len(accs)),
            "n_mixed_folds": int(len(aucs)),
            "accuracy_mean": float(accs.mean()) if len(accs) else None,
            "accuracy_std": float(accs.std()) if len(accs) else None,
            "weighted_accuracy": (
                float(weighted_num / weighted_den) if weighted_den else None
            ),
            "auc_mean": float(aucs.mean()) if len(aucs) else None,
            "auc_std": float(aucs.std()) if len(aucs) else None,
            "auc_pooled": pooled_auc,
        }
        results["per_probe"][probe] = summary

    OUT_PATH.write_text(json.dumps(results, indent=2))

    # Console summary
    print(
        f"\n{'Architecture':<24} {'Pooled AUC':>11} {'Mean AUC (±std)':>18} "
        f"{'Weighted ACC':>13} {'Mean ACC (±std)':>18}"
    )
    print("-" * 86)
    for probe in PROBES:
        s = results["per_probe"][probe]
        print(
            f"{s['label']:<24} {s['auc_pooled']:>11.3f} "
            f"{s['auc_mean']:>9.3f} ±{s['auc_std']:.3f}     "
            f"{s['weighted_accuracy']:>13.3f}  "
            f"{s['accuracy_mean']:>9.3f} ±{s['accuracy_std']:.3f}"
        )
    print(f"\n  AUC over n={results['per_probe'][PROBES[0]]['n_mixed_folds']} mixed folds; "
          f"ACC over n={results['per_probe'][PROBES[0]]['n_folds']} folds.")
    print(f"  Output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
