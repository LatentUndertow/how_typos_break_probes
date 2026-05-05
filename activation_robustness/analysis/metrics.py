"""Probe evaluation metrics for activation robustness analysis.

All functions operate on numpy arrays and return plain dicts or scalars.
No hidden state — every function is independently callable.

Primary use: macro-averaged ROC curves and TPR@FPR operating-point summaries
for cross-validated probe evaluations stored in perteval .npz score files.
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc as sklearn_auc


# Common FPR grid for macro-averaging across folds.
# Each fold's ROC is interpolated onto this grid before averaging,
# so mean ± std are well-defined at every FPR value.
FPR_GRID = np.linspace(0, 1, 2000)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_perteval_scores(perteval_dir, archs, conditions, n_folds=5):
    """Load per-fold (labels, scores) for each architecture × condition.

    Args:
        perteval_dir: Path to directory containing fold_X_scores.npz files.
        archs: Iterable of architecture key strings (must match .npz array names).
        conditions: Iterable of perturbation condition strings.
        n_folds: Number of CV folds (default 5).

    Returns:
        dict: arch -> {'clean': [(labels, scores), ...], cond: [...], ...}
              One (labels, scores) tuple per fold for each key.
    """
    all_data = {}
    perteval_dir = Path(perteval_dir)
    for fold in range(n_folds):
        d = np.load(perteval_dir / f"fold_{fold}_scores.npz")
        labels = d["labels"]
        for arch in archs:
            if arch not in all_data:
                all_data[arch] = {"clean": [], **{c: [] for c in conditions}}
            all_data[arch]["clean"].append((labels, d[f"{arch}_clean"]))
            for cond in conditions:
                all_data[arch][cond].append((labels, d[f"{arch}_{cond}"]))
    return all_data


# ── ROC utilities ─────────────────────────────────────────────────────────────

def fold_roc(labels, scores):
    """ROC curve for one fold, interpolated onto FPR_GRID.

    Non-finite scores are dropped before computation.

    Args:
        labels: 1-D int array of ground-truth labels.
        scores: 1-D float array of probe scores.

    Returns:
        tpr_interp: TPR values on FPR_GRID (shape: len(FPR_GRID),).
        auc_val: Scalar AUC for this fold.
    """
    mask = np.isfinite(scores)
    fpr, tpr, _ = roc_curve(labels[mask], scores[mask])
    tpr_interp = np.interp(FPR_GRID, fpr, tpr)
    return tpr_interp, float(sklearn_auc(fpr, tpr))


def macro_roc(fold_list):
    """Macro-average ROC across folds.

    Computes per-fold ROC on FPR_GRID, then averages.

    Args:
        fold_list: List of (labels, scores) tuples, one per fold.

    Returns:
        mean_tpr: Mean TPR on FPR_GRID.
        std_tpr:  Std TPR on FPR_GRID.
        aucs:     Per-fold AUC array.
    """
    tprs, aucs = [], []
    for labels, scores in fold_list:
        t, a = fold_roc(labels, scores)
        tprs.append(t)
        aucs.append(a)
    tprs = np.array(tprs)
    return tprs.mean(axis=0), tprs.std(axis=0), np.array(aucs)


def tpr_at_fpr(mean_tpr, target_fpr):
    """Interpolated TPR from a macro-averaged curve at a given FPR.

    Args:
        mean_tpr: Mean TPR array on FPR_GRID (from macro_roc).
        target_fpr: Scalar target FPR (e.g. 0.01 for 1%).

    Returns:
        Scalar TPR value.
    """
    return float(np.interp(target_fpr, FPR_GRID, mean_tpr))


def delta_tpr_per_fold(clean_fold_list, pert_fold_list, target_fpr):
    """Per-fold ΔTPR at a target FPR operating point.

    Args:
        clean_fold_list: List of (labels, scores) for clean inputs, one per fold.
        pert_fold_list:  List of (labels, scores) for perturbed inputs, one per fold.
        target_fpr: Target FPR (e.g. 0.01).

    Returns:
        mean_delta_pp: Mean ΔTPR in percentage points across folds.
        std_delta_pp:  Std ΔTPR in percentage points across folds.
    """
    deltas = []
    for (lc, sc), (lp, sp) in zip(clean_fold_list, pert_fold_list):
        mc, _ = fold_roc(lc, sc)
        mp, _ = fold_roc(lp, sp)
        deltas.append(tpr_at_fpr(mp, target_fpr) - tpr_at_fpr(mc, target_fpr))
    return float(np.mean(deltas)) * 100, float(np.std(deltas)) * 100
