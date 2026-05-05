#!/usr/bin/env python3
"""
LODO sweep results summary — reads raw all_scores.npz files directly.

Computes correct TPR / FPR / ACC at threshold=0.5 for each fold × probe.
Single-class test sets (all-benign or all-malicious) show only the
relevant metric (FPR or TPR respectively).

Usage:
    python lodo_results_summary.py
    python lodo_results_summary.py --lodo-dir /path/to/lodo_sweep
    python lodo_results_summary.py --thr 0.3
"""
import sys
import argparse
import numpy as np
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_LODO_DIR = _REPO / "activation_robustness" / "results" / "lodo_sweep"

PROBES = [
    "attention_all",
    "mlp_all",
    "multimax_all",
    "positional_linear_pos-5",
]
SHORT = {
    "attention_all":            "attn ",
    "mlp_all":                  "mlp  ",
    "multimax_all":             "mmax ",
    "positional_linear_pos-5":  "pos-5",
}


def metrics_single_class(scores, label, thr):
    preds = (scores >= thr).astype(int)
    if label == 0:  # all benign → report FPR
        fp = preds.sum(); tn = len(preds) - fp
        fpr = fp / max(fp + tn, 1)
        return "FPR", fpr
    else:           # all malicious → report TPR
        tp = preds.sum(); fn = len(preds) - tp
        tpr = tp / max(tp + fn, 1)
        return "TPR", tpr


def metrics_mixed(scores, labels, thr):
    preds = (scores >= thr).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    acc = (tp + tn) / max(len(labels), 1)
    return tpr, fpr, acc


CONDITIONS = ["full_bundle", "every_second_word"]
COND_SHORT = {"full_bundle": "FB ", "every_second_word": "ESW"}


def pert_delta(clean_scores, pert_scores, labels, thr):
    """Return (ΔTPR, ΔFPR) or (Δmetric, None) for single-class."""
    valid = ~np.isnan(clean_scores) & ~np.isnan(pert_scores)
    s_c, s_p, l = clean_scores[valid], pert_scores[valid], labels[valid]
    unique = np.unique(l)
    if len(unique) == 1:
        _, val_c = metrics_single_class(s_c, unique[0], thr)
        _, val_p = metrics_single_class(s_p, unique[0], thr)
        return val_p - val_c, None
    tpr_c, fpr_c, _ = metrics_mixed(s_c, l, thr)
    tpr_p, fpr_p, _ = metrics_mixed(s_p, l, thr)
    return tpr_p - tpr_c, fpr_p - fpr_c


def fmt_delta(v):
    if v is None or np.isnan(v): return "  --- "
    return f"{v:+.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lodo-dir", default=str(DEFAULT_LODO_DIR))
    parser.add_argument("--thr", type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    parser.add_argument("--pert", action="store_true",
                        help="Show perturbation deltas (ΔTPR / ΔFPR per condition)")
    args = parser.parse_args()

    lodo_dir = Path(args.lodo_dir)
    thr = args.thr

    fold_dirs = sorted([
        d for d in lodo_dir.iterdir()
        if d.is_dir() and d.name.startswith("fold_")
        and (d / "perturbation_eval" / "all_scores.npz").exists()
    ])

    if not fold_dirs:
        print(f"No fold results found in {lodo_dir}")
        sys.exit(1)

    print(f"LODO sweep results  (threshold={thr})")
    print(f"Directory: {lodo_dir}")
    print(f"Folds found: {len(fold_dirs)}")
    print()

    col = "  ".join(f"{SHORT[p]:5s}" for p in PROBES)
    print(f"{'Fold':42s} {'Type':3s}  {'n':6s}  {col}")
    print("-" * 90)

    for fold_dir in fold_dirs:
        fold = fold_dir.name[len("fold_"):]
        npz_path = fold_dir / "perturbation_eval" / "all_scores.npz"
        d = np.load(npz_path)
        labels = d["labels"]
        n = len(labels)
        unique = np.unique(labels)

        if len(unique) == 1:
            metric_name, vals = [], []
            for p in PROBES:
                key = f"{p}_clean"
                s = d[key] if key in d else np.full(n, np.nan)
                name, val = metrics_single_class(s, unique[0], thr)
                metric_name.append(name)
                vals.append(val)
            typ = "BEN" if unique[0] == 0 else "MAL"
            row = "  ".join(f"{v:.3f}" for v in vals)
            print(f"{fold:42s} {typ}  n={n:<6d}  {row}  ← {metric_name[0]}")
            if args.pert:
                for cond in CONDITIONS:
                    dvals = []
                    for p in PROBES:
                        ck = f"{p}_clean"; pk = f"{p}_{cond}"
                        sc = d[ck] if ck in d else np.full(n, np.nan)
                        sp = d[pk] if pk in d else np.full(n, np.nan)
                        dv, _ = pert_delta(sc, sp, labels, thr)
                        dvals.append(fmt_delta(dv))
                    print(f"  {COND_SHORT[cond]} Δ{'FPR' if unique[0]==0 else 'TPR'}:  {'  '.join(dvals)}")
        else:
            n_mal = int((labels == 1).sum()); n_ben = int((labels == 0).sum())
            tpr_row, fpr_row = [], []
            for p in PROBES:
                key = f"{p}_clean"
                s = d[key] if key in d else np.full(n, np.nan)
                tpr, fpr, _ = metrics_mixed(s, labels, thr)
                tpr_row.append(tpr); fpr_row.append(fpr)
            tpr_str = "  ".join(f"{v:.3f}" for v in tpr_row)
            fpr_str = "  ".join(f"{v:.3f}" for v in fpr_row)
            print(f"{fold:42s} MIX  n={n:<6d}  {tpr_str}  ← TPR  (mal={n_mal} ben={n_ben})")
            print(f"{'':42s}      {'':6s}  {fpr_str}  ← FPR")
            if args.pert:
                for cond in CONDITIONS:
                    dtpr_row, dfpr_row = [], []
                    for p in PROBES:
                        ck = f"{p}_clean"; pk = f"{p}_{cond}"
                        sc = d[ck] if ck in d else np.full(n, np.nan)
                        sp = d[pk] if pk in d else np.full(n, np.nan)
                        dt, df = pert_delta(sc, sp, labels, thr)
                        dtpr_row.append(fmt_delta(dt)); dfpr_row.append(fmt_delta(df))
                    print(f"  {COND_SHORT[cond]} ΔTPR: {'  '.join(dtpr_row)}")
                    print(f"  {COND_SHORT[cond]} ΔFPR: {'  '.join(dfpr_row)}")

    print()
    print(f"Probes (columns): {' | '.join(SHORT[p]+': '+p for p in PROBES)}")
    if args.pert:
        print(f"Conditions: FB=full_bundle, ESW=every_second_word")
        print(f"Δ = perturbed − clean  (negative = worse, positive = better)")


if __name__ == "__main__":
    main()
