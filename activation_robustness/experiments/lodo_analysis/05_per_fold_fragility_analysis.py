#!/usr/bin/env python3
"""
Per-fold fragility analysis: does perturbation Δ correlate with clean
baseline TPR? Compare no-fork vs KV-fork (neutral suffix).

The hypothesis: under LODO, the OOD clean baseline is heterogeneous
across folds; perturbation Δ may scale with where the clean baseline
sits, rather than being constant. If so, the KV-fork's apparent
protection level depends on whether it raises the operating point or
truly attenuates the perturbation.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

NO_FORK = Path(__file__).parent / "perturbation_lodo_results.json"
KV_FORK = Path(__file__).parent / "kvfork_lodo_results.json"
OUT_PATH = Path(__file__).parent / "fragility_per_fold.json"


def main() -> int:
    if not NO_FORK.exists() or not KV_FORK.exists():
        print("ERROR: run 03_compute_perturbation_lodo.py and 04_compute_kvfork_lodo.py first.", file=sys.stderr)
        return 1
    nf = json.loads(NO_FORK.read_text())
    kvf = json.loads(KV_FORK.read_text())

    # Mixed folds shared by both
    nf_folds = set(nf["folds_used"])
    kvf_folds = set(kvf["mixed_folds_used"])
    folds = sorted(nf_folds & kvf_folds)
    print(f"Per-fold analysis on {len(folds)} mixed folds shared by both: {folds}")

    rows = []
    for fold in folds:
        # No-fork linear pos-5
        nf_row = nf["per_fold"][fold]["positional_linear_pos-5"]
        nf_clean = nf_row["fix_a_tpr_at_fpr1"]["clean"]
        nf_bundle = nf_row["fix_a_tpr_at_fpr1"]["full_bundle"]

        # KV-fork neutral
        kvf_row = kvf["per_fold"][fold]["neutral"]
        kvf_clean = kvf_row["tpr_clean"]
        kvf_bundle = kvf_row["tpr_bundle"]

        rows.append({
            "fold": fold,
            "no_fork_clean": nf_clean,
            "no_fork_bundle": nf_bundle,
            "no_fork_delta": nf_bundle - nf_clean,
            "kvfork_neutral_clean": kvf_clean,
            "kvfork_neutral_bundle": kvf_bundle,
            "kvfork_neutral_delta": kvf_bundle - kvf_clean,
            "baseline_lift_clean": kvf_clean - nf_clean,
            "baseline_lift_bundle": kvf_bundle - nf_bundle,
        })

    # Correlations
    nfc = np.array([r["no_fork_clean"] for r in rows])
    nfd = np.array([r["no_fork_delta"] for r in rows])
    kvc = np.array([r["kvfork_neutral_clean"] for r in rows])
    kvd = np.array([r["kvfork_neutral_delta"] for r in rows])

    def corr(x, y):
        c = np.corrcoef(x, y)[0, 1]
        return float(c)

    summary = {
        "n_folds": len(folds),
        "no_fork_corr_clean_vs_delta": corr(nfc, nfd),
        "kvfork_neutral_corr_clean_vs_delta": corr(kvc, kvd),
        "no_fork_clean_mean": float(nfc.mean()),
        "no_fork_clean_std": float(nfc.std()),
        "no_fork_delta_mean": float(nfd.mean()),
        "no_fork_delta_std": float(nfd.std()),
        "kvfork_neutral_clean_mean": float(kvc.mean()),
        "kvfork_neutral_clean_std": float(kvc.std()),
        "kvfork_neutral_delta_mean": float(kvd.mean()),
        "kvfork_neutral_delta_std": float(kvd.std()),
        "mean_baseline_lift_clean": float((kvc - nfc).mean()),
        "mean_baseline_lift_bundle": float(np.array([r["baseline_lift_bundle"] for r in rows]).mean()),
    }

    OUT_PATH.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    # Console table
    print(f"\n{'Fold':<32s} {'NF clean':>10} {'NF Δ':>10} {'KVF clean':>10} {'KVF Δ':>10} {'Lift':>9}")
    print("-" * 88)
    for r in rows:
        print(f"{r['fold']:<32s} "
              f"{100*r['no_fork_clean']:>8.2f}%  "
              f"{100*r['no_fork_delta']:>+8.2f}pp  "
              f"{100*r['kvfork_neutral_clean']:>8.2f}%  "
              f"{100*r['kvfork_neutral_delta']:>+8.2f}pp  "
              f"{100*r['baseline_lift_clean']:>+7.2f}pp")

    print(f"\n=== Correlations (per-fold clean baseline vs perturbation Δ) ===")
    print(f"  No-fork:        corr(clean, Δ) = {summary['no_fork_corr_clean_vs_delta']:+.3f}")
    print(f"  KV-fork (neutral): corr(clean, Δ) = {summary['kvfork_neutral_corr_clean_vs_delta']:+.3f}")
    print(f"\n  Mean baseline lift from KV-fork (clean):  {100*summary['mean_baseline_lift_clean']:+.2f}pp")
    print(f"  Mean baseline lift from KV-fork (bundle): {100*summary['mean_baseline_lift_bundle']:+.2f}pp")
    print(f"  Mean Δ (no-fork):     {100*summary['no_fork_delta_mean']:+.2f} ± {100*summary['no_fork_delta_std']:.2f}pp")
    print(f"  Mean Δ (KV-fork):     {100*summary['kvfork_neutral_delta_mean']:+.2f} ± {100*summary['kvfork_neutral_delta_std']:.2f}pp")

    print(f"\n  Output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
