#!/usr/bin/env python3
"""
Fomin-style pooled AUC across all held-out predictions.

Difference from `02_compute_clean_lodo.py`:
  - That script pools only across the 7 mixed-class folds (where AUC is
    individually defined per fold), so its "pooled AUC" is restricted.
  - This script pools across ALL 29 held-out folds. Single-class folds
    contribute their held-out predictions to the global pool, which
    has both classes once everything is concatenated. This matches
    Fomin et al. (2026), Table 3: "pooled AUC across all held-out
    predictions (N=105,034)".
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

EXPORT_ROOT = Path("./lodo_data/extracted/lodo_results_export")
VALIDATION_REPORT = Path(__file__).parent / "validation_report.json"
OUT_PATH = Path(__file__).parent / "fomin_style_pooled_auc.json"

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


def main() -> int:
    if not VALIDATION_REPORT.exists():
        print("ERROR: run 01_validate_integrity.py first.", file=sys.stderr)
        return 1
    report = json.loads(VALIDATION_REPORT.read_text())
    folds = sorted(
        f for f, e in report["folds"].items()
        if e.get("canonical_check", {}).get("matches_readme")
    )
    print(f"[07_fomin_pooled] Pooling held-out predictions across {len(folds)} folds")

    results: dict = {
        "n_folds_pooled": len(folds),
        "folds_pooled": folds,
        "per_probe": {},
    }

    for probe in PROBES:
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        per_fold_n: dict[str, int] = {}

        for fold in folds:
            path = EXPORT_ROOT / f"fold_{fold}" / probe / "test_scores.npz"
            if not path.exists():
                continue
            d = np.load(path)
            all_scores.append(d["scores"])
            all_labels.append(d["labels"])
            per_fold_n[fold] = int(len(d["labels"]))

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        n_total = len(labels)
        n_mal = int(labels.sum())
        n_ben = n_total - n_mal

        pooled_auc = float(roc_auc_score(labels, scores))

        results["per_probe"][probe] = {
            "label": LABELS[probe],
            "n_folds_contributing": len(per_fold_n),
            "n_total_predictions": int(n_total),
            "n_malicious": n_mal,
            "n_benign": n_ben,
            "pooled_auc_all_folds": pooled_auc,
        }

    OUT_PATH.write_text(json.dumps(results, indent=2))

    print()
    print(f"{'Architecture':<24} {'Pooled AUC (all 29 folds)':>28} {'N total':>10}")
    print("-" * 64)
    for probe in PROBES:
        s = results["per_probe"][probe]
        print(f"{s['label']:<24} {s['pooled_auc_all_folds']:>28.4f} {s['n_total_predictions']:>10}")
    print()

    # Comparison vs the 7-mixed-fold pooled AUC from clean_lodo_results.json
    clean_path = Path(__file__).parent / "clean_lodo_results.json"
    if clean_path.exists():
        clean = json.loads(clean_path.read_text())
        print(f"{'Architecture':<24} {'Pooled (29 folds)':>20} {'Pooled (7 mixed)':>20} {'Δ':>10}")
        print("-" * 76)
        for probe in PROBES:
            new_auc = results["per_probe"][probe]["pooled_auc_all_folds"]
            old_auc = clean["per_probe"][probe]["auc_pooled"]
            print(f"{LABELS[probe]:<24} {new_auc:>20.4f} {old_auc:>20.4f} {new_auc-old_auc:>+10.4f}")
        print()

    print(f"  Output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
