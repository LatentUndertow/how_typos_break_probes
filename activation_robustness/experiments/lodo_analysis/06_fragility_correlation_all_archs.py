#!/usr/bin/env python3
"""
Cross-architecture: does per-fold perturbation Δ correlate with the
per-fold clean baseline TPR@FPR=1%? Computes correlation per
architecture across the 7 mixed folds, separately for bundle and ESW
perturbations.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

NO_FORK = Path(__file__).parent / "perturbation_lodo_results.json"
OUT_PATH = Path(__file__).parent / "fragility_correlation_all_archs.json"

PROBES = [
    "positional_linear_pos-5",
    "mean_linear_last16",
    "mlp_all",
    "attention_all",
    "multimax_all",
]
LABELS = {
    "positional_linear_pos-5": "Linear (EOT)",
    "mean_linear_last16": "Mean Linear",
    "mlp_all": "MLP",
    "attention_all": "Attention",
    "multimax_all": "MultiMax",
}


def main() -> int:
    nf = json.loads(NO_FORK.read_text())
    folds = nf["folds_used"]

    summary = {}
    print(f"{'Arch':<14} {'corr clean→Δbundle':>22} {'corr clean→ΔESW':>22}  "
          f"{'clean range':>22} {'Δbundle range':>22}")
    for probe in PROBES:
        clean = np.array([
            nf["per_fold"][f][probe]["fix_a_tpr_at_fpr1"]["clean"] for f in folds
        ])
        bundle = np.array([
            nf["per_fold"][f][probe]["fix_a_tpr_at_fpr1"]["full_bundle"] for f in folds
        ])
        esw = np.array([
            nf["per_fold"][f][probe]["fix_a_tpr_at_fpr1"]["every_second_word"] for f in folds
        ])
        d_bundle = bundle - clean
        d_esw = esw - clean
        c_b = float(np.corrcoef(clean, d_bundle)[0, 1])
        c_e = float(np.corrcoef(clean, d_esw)[0, 1])
        summary[probe] = {
            "label": LABELS[probe],
            "corr_clean_vs_delta_bundle": c_b,
            "corr_clean_vs_delta_esw": c_e,
            "clean_min": float(clean.min()), "clean_max": float(clean.max()),
            "delta_bundle_min": float(d_bundle.min()), "delta_bundle_max": float(d_bundle.max()),
            "delta_esw_min": float(d_esw.min()), "delta_esw_max": float(d_esw.max()),
            "per_fold": {
                folds[i]: {
                    "clean": float(clean[i]),
                    "delta_bundle": float(d_bundle[i]),
                    "delta_esw": float(d_esw[i]),
                } for i in range(len(folds))
            }
        }
        print(f"{LABELS[probe]:<14} {c_b:>+22.3f} {c_e:>+22.3f}  "
              f"[{100*clean.min():>5.1f},{100*clean.max():>5.1f}]%   "
              f"[{100*d_bundle.min():>+5.1f},{100*d_bundle.max():>+5.1f}]pp")

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n  n folds = {len(folds)}; output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
