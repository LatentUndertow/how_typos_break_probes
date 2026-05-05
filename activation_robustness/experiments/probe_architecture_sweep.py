#!/usr/bin/env python3
"""
Probe architecture sweep: train and evaluate multiple probe architectures
on a stratified 80/20 split of the full activation cache.

Uses the multi-architecture probe implementations from activation_robustness.probes
and the CachedActivationDataset/CachedBatchProvider for data loading.

Usage:
    python probe_architecture_sweep.py
    python probe_architecture_sweep.py --probe-types positional_linear mean_linear
    python probe_architecture_sweep.py --datasets-file datasets_17.txt
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from activation_robustness.data.activation_cache import ActivationCache, CachedActivationDataset
from activation_robustness.data.batch_provider import CachedBatchProvider
from activation_robustness.probes.architectures import MultiArchProbe, ProbeConfig
from activation_robustness.probes.trainer import train_probes_parallel

CACHE_DIR = os.environ.get("ACTIVATION_CACHE_DIR", "./cache_data/activations")
OUTPUT_BASE = _REPO / "activation_robustness" / "results" / "probe_architecture_sweep"

# Original 17 datasets (kept for backwards compatibility)
DATASETS_17 = [
    "EnronDataset", "Dolly15kDataset", "OpenOrcaDataset", "PromptsRanked10kDataset",
    "AlpacaDataset", "SafeGuardDataset", "QualifireDataset", "SoftAgeDataset",
    "BIPIADataset", "InjecAgentDataset", "LLMailDataset", "MosscapDataset",
    "WildJailbreakDataset", "JayavibhavDataset", "DeepsetDataset",
    "YanismiraouiDataset", "AdvBenchDataset",
]

# Full 29-dataset list for LODO sweep (all datasets in local_activation_cache).
# AgentDojoDataset and APIGenMTDataset are 50%-subsampled copies from blobfuse.
# GandalfSummarizationDataset (114 samples) and ScamDataset (25 samples) are
# too small to be standalone LODO folds; they are merged with similar datasets
# (Mosscap and AdvBench respectively) via LODO_MERGE_MAP below.
DATASETS_ALL = DATASETS_17 + [
    "HarmBenchDataset",
    "AgentDojoDataset", "APIGenMTDataset",
    "BitextCustomerSupportDataset", "CodeExerciseDataset",
    "GandalfSummarizationDataset", "JailbreakClassificationDataset",
    "PythonCodeAlpacaDataset", "PythonCodes25kDataset",
    "ScamDataset", "WritingPromptsDataset", "XlamFunctionCallingDataset",
]

# Small datasets merged into larger ones for LODO fold stability.
# Key = dataset to merge, value = dataset it folds into (same train/test split).
LODO_MERGE_MAP = {
    "GandalfSummarizationDataset": "MosscapDataset",   # 114 samples, indirect PI
    "ScamDataset": "AdvBenchDataset",                   # 25 samples, harmful requests
}

# Probe configurations to sweep
# _max_tokens: how many trailing tokens to load per sample (None = all)
# _base_params: which param set to use
PROBE_CONFIGS = {
    "positional_linear_pos-5": {
        "probe_type": "positional_linear",
        "token_position": -5,
        "_max_tokens": 16,
        "_base_params": "linear",
    },
    "positional_linear_pos-1": {
        "probe_type": "positional_linear",
        "token_position": -1,
        "_max_tokens": 16,
        "_base_params": "linear",
    },
    "mean_linear_all": {
        "probe_type": "mean_linear",
        "_max_tokens": None,
        "_base_params": "linear",
    },
    "mean_linear_last16": {
        "probe_type": "mean_linear",
        "_max_tokens": 16,
        "_base_params": "linear",
    },
    "mean_linear_last32": {
        "probe_type": "mean_linear",
        "_max_tokens": 32,
        "_base_params": "linear",
    },
    "ema_all": {
        "probe_type": "ema",
        "ema_alpha": 0.5,
        "_max_tokens": None,
        "_base_params": "linear",
    },
    "ema_last16": {
        "probe_type": "ema",
        "ema_alpha": 0.5,
        "_max_tokens": 16,
        "_base_params": "linear",
    },
    "ema_last32": {
        "probe_type": "ema",
        "ema_alpha": 0.5,
        "_max_tokens": 32,
        "_base_params": "linear",
    },
    "mlp_all": {
        "probe_type": "mlp",
        "mlp_hidden_dim": 100,
        "mlp_layers": 2,
        "_max_tokens": None,
        "_base_params": "mlp",
    },
    "mlp_pos-5": {
        "probe_type": "mlp",
        "token_position": -5,
        "mlp_hidden_dim": 100,
        "mlp_layers": 2,
        "_max_tokens": 16,
        "_base_params": "mlp",
    },
    "attention_all": {
        "probe_type": "attention",
        "n_heads": 10,
        "mlp_hidden_dim": 100,
        "mlp_layers": 2,
        "_max_tokens": None,
        "_base_params": "mlp",
    },
    "multimax_all": {
        "probe_type": "multimax",
        "n_heads": 10,
        "mlp_hidden_dim": 100,
        "mlp_layers": 2,
        "_max_tokens": None,
        "_base_params": "mlp",
    },
}

# Training params: linear probes get lr=1e-3 + standard norm,
# MLP-based probes get lr=1e-4 + no norm (Gemini paper defaults)
_LINEAR_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 3e-3,
    "max_epochs": 1000,
    "early_stopping_patience": 50,
    "val_split": 0.0,
    "normalize": "standard",
    "batch_size": 0,  # 0 = full-batch (all training samples per step)
    "use_class_weight": True,
    "random_state": 42,
}
_MLP_PARAMS = {
    "lr": 1e-4,
    "weight_decay": 3e-3,
    "max_epochs": 5,
    "early_stopping_patience": 50,
    "val_split": 0.0,
    "normalize": "none",
    "batch_size": 64,
    "use_class_weight": True,
    "random_state": 42,
}
SHARED_PARAMS = _LINEAR_PARAMS  # default, overridden per probe below

SPLIT_SEED = 42
TEST_RATIO = 0.2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_metrics(labels, scores, threshold=0.5):
    """Compute classification metrics at a given threshold.

    AUC is NaN when only one class is present (single-class test set, e.g. LODO
    fold that is all-malicious or all-benign).  Callers should display NaN as
    'N/A' rather than treating it as a score of 0.
    """
    preds = (scores >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    f1 = f1_score(labels, preds, zero_division=0)

    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tpr       = tp / max(tp + fn, 1)          # recall
    precision = tp / max(tp + fp, 1)          # NaN-free: if no predictions, precision=1 by convention
    fpr       = fp / max(fp + tn, 1)
    tnr       = tn / max(tn + fp, 1)

    return {
        "accuracy":  float(acc),
        "auc":       float(auc),
        "f1":        float(f1),
        "tpr":       float(tpr),       # recall
        "precision": float(precision),
        "fpr":       float(fpr),
        "tnr":       float(tnr),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": float(threshold),
        "n_samples":   int(len(labels)),
        "n_malicious": int(labels.sum()),
        "n_benign":    int(len(labels) - labels.sum()),
        "mean_score_malicious": float(scores[labels == 1].mean()) if labels.sum() > 0 else float("nan"),
        "mean_score_benign":    float(scores[labels == 0].mean()) if (labels == 0).sum() > 0 else float("nan"),
    }


def tpr_at_fpr(labels, scores, target_fpr):
    """Compute TPR at a given FPR operating point.

    Returns (nan, nan, nan) when only one class is present — roc_curve requires
    both classes.  Callers should treat nan TPR as 'N/A'.
    """
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    from sklearn.metrics import roc_curve
    fprs, tprs, thresholds = roc_curve(labels, scores)
    # Find threshold closest to target FPR
    idx = np.argmin(np.abs(fprs - target_fpr))
    return float(tprs[idx]), float(thresholds[idx]), float(fprs[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-types", nargs="+", default=None,
                        help="Subset of probe configs to run (keys from PROBE_CONFIGS)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names to use (default: DATASETS_17)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--lodo-fold", type=str, default=None,
                        help="LODO mode: name of dataset to hold out as test set. "
                             "All other datasets are used for training. "
                             "Overrides --test-ratio; the held-out dataset is the entire test set.")
    parser.add_argument("--kfold", type=int, default=None,
                        help="K-fold CV mode: number of folds (e.g. 5). "
                             "Requires --fold-idx. Overrides --test-ratio.")
    parser.add_argument("--fold-idx", type=int, default=None,
                        help="Which fold to use as test set (0-indexed, 0 to --kfold-1).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_BASE
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or DATASETS_17
    probe_keys = args.probe_types or list(PROBE_CONFIGS.keys())

    # Validate probe keys
    for k in probe_keys:
        if k not in PROBE_CONFIGS:
            log(f"Unknown probe config: {k}. Available: {list(PROBE_CONFIGS.keys())}")
            sys.exit(1)

    # ── LODO mode: reorder so held-out is last, record its position ──
    lodo_held_out = args.lodo_fold
    if lodo_held_out is not None:
        if lodo_held_out not in datasets:
            log(f"ERROR: --lodo-fold '{lodo_held_out}' not in dataset list")
            sys.exit(1)
        # Put held-out last so its _dataset_indices value is predictable
        datasets = [d for d in datasets if d != lodo_held_out] + [lodo_held_out]
        lodo_held_out_pos = len(datasets) - 1
        log(f"LODO mode: held-out={lodo_held_out} (position {lodo_held_out_pos})")

    # ── K-fold mode validation ──
    if args.kfold is not None:
        if args.fold_idx is None:
            log("ERROR: --kfold requires --fold-idx")
            sys.exit(1)
        if not (0 <= args.fold_idx < args.kfold):
            log(f"ERROR: --fold-idx must be in [0, {args.kfold-1}]")
            sys.exit(1)

    log(f"Output: {output_dir}")
    log(f"Datasets: {len(datasets)}")
    log(f"Probes: {probe_keys}")
    if lodo_held_out is None and args.kfold is None:
        log(f"Split: {1 - args.test_ratio:.0%} train / {args.test_ratio:.0%} test, seed={args.split_seed}")
    elif args.kfold is not None:
        log(f"K-fold CV: {args.kfold} folds, test fold={args.fold_idx}, seed={args.split_seed}")

    # ── Load data ──
    log("\nLoading activation cache...")
    cache = ActivationCache(cache_dir=CACHE_DIR)

    # Check all datasets are cached
    missing = [d for d in datasets if not cache.is_cached(d)]
    if missing:
        log(f"ERROR: Missing from cache: {missing}")
        sys.exit(1)

    ds = CachedActivationDataset(cache, datasets)
    log(f"Loaded {len(ds)} samples, d_model={ds.d_model}")

    all_labels = ds.labels.numpy().astype(np.float32)
    n_mal = int(all_labels.sum())
    n_ben = len(all_labels) - n_mal
    log(f"Labels: {n_mal} malicious, {n_ben} benign ({n_mal/len(all_labels)*100:.1f}% positive)")

    # ── Build train/test split ──
    all_indices = np.arange(len(ds))

    if lodo_held_out is not None:
        # LODO split: held-out dataset is the entire test set
        ds_idx_arr = ds._dataset_indices.numpy()
        test_mask  = (ds_idx_arr == lodo_held_out_pos)
        train_idx  = all_indices[~test_mask]
        test_idx   = all_indices[test_mask]
        log(f"LODO split: {len(train_idx)} train, {len(test_idx)} test ({lodo_held_out})")
    elif args.kfold is not None:
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.split_seed)
        folds = list(skf.split(all_indices, all_labels))
        train_idx, test_idx = folds[args.fold_idx]
        log(f"K-fold split: fold {args.fold_idx}/{args.kfold}, "
            f"{len(train_idx)} train, {len(test_idx)} test")
    else:
        train_idx, test_idx = train_test_split(
            all_indices, test_size=args.test_ratio,
            stratify=all_labels, random_state=args.split_seed,
        )
        log(f"Split: {len(train_idx)} train, {len(test_idx)} test")

    # Save split for reproducibility
    split_path = output_dir / "split_indices.npz"
    np.savez(split_path, train=train_idx, test=test_idx, seed=args.split_seed,
             lodo_held_out=lodo_held_out or "",
             kfold=args.kfold or 0, fold_idx=args.fold_idx if args.fold_idx is not None else -1)
    log(f"Split saved to {split_path}")

    train_labels = all_labels[train_idx]
    test_labels = all_labels[test_idx]
    log(f"Train: {int(train_labels.sum())} mal, {int(len(train_labels) - train_labels.sum())} ben")
    log(f"Test:  {int(test_labels.sum())} mal, {int(len(test_labels) - test_labels.sum())} ben")

    # ── Helper: extract positions from cache into numpy ──
    def extract_from_cache(ds, indices, max_tokens):
        """Load last max_tokens positions for each sample into a numpy array.

        Uses CachedActivationDataset.load_batch with our max_tokens support
        to batch-load in chunks, avoiding per-sample Python loop overhead.
        """
        if max_tokens is None:
            return None

        n = len(indices)
        d = ds.d_model
        X = np.zeros((n, max_tokens, d), dtype=np.float32)

        chunk_size = 512
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_indices = indices[start:end]
            hidden, mask, _, _ = ds.load_batch(chunk_indices, max_tokens=max_tokens)
            # hidden is (chunk, max_tokens, d) bf16 left-padded
            X[start:end] = hidden.float().numpy()
            if start % 5000 < chunk_size:
                log(f"    preload: {end}/{n}")

        return X

    # ── Preload once at the largest max_tokens needed ──
    # Covers linear probes AND positional-MLP probes (single token, numpy path).
    def _uses_preload(cfg):
        return (
            cfg.get("_base_params", "linear") == "linear" and cfg.get("_max_tokens") is not None
        ) or (
            "token_position" in cfg and cfg.get("_max_tokens") is not None
        )

    max_needed = max(
        (PROBE_CONFIGS[k].get("_max_tokens") or 0) for k in probe_keys
        if _uses_preload(PROBE_CONFIGS[k])
    ) if any(_uses_preload(PROBE_CONFIGS[k]) for k in probe_keys) else 0
    if max_needed > 0:
        log(f"\nPreloading last {max_needed} tokens for all samples...")
        t_pre = time.time()
        X_preloaded = extract_from_cache(ds, np.arange(len(ds)), max_needed)
        log(f"Preloaded in {time.time()-t_pre:.1f}s, shape={X_preloaded.shape}, "
            f"{X_preloaded.nbytes/1e9:.1f} GB")
    else:
        X_preloaded = None

    # ── Sweep ──
    all_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # MLP provider-path probes are collected here and trained in one parallel
    # pass after the sequential loop so each activation batch is read once.
    _parallel_pending: dict = {}

    for probe_key in probe_keys:
        probe_dir_check = output_dir / probe_key
        if (probe_dir_check / "results.json").exists():
            log(f"\n  [SKIP] {probe_key} — results.json already exists")
            # Still load result into all_results so the summary is complete
            try:
                with open(probe_dir_check / "results.json") as f:
                    all_results[probe_key] = json.load(f)
            except Exception:
                pass
            continue

        log(f"\n{'='*60}")
        log(f"  Probe: {probe_key}")
        log(f"{'='*60}")

        probe_cfg = dict(PROBE_CONFIGS[probe_key])  # copy to avoid mutating
        max_tokens = probe_cfg.pop("_max_tokens", None)
        base = probe_cfg.pop("_base_params", "linear")
        base_params = _LINEAR_PARAMS if base == "linear" else _MLP_PARAMS
        probe_params = {**base_params, **probe_cfg}
        config = ProbeConfig(**probe_params)
        is_linear = (base == "linear")
        # Single-position MLP: extract one token then train MLP on that vector.
        # Routes through the numpy preload path (like positional_linear) so the
        # IO cost matches pos-5 linear, not mlp_all.
        is_positional_mlp = (base == "mlp" and "token_position" in probe_params
                             and max_tokens is not None)

        log(f"  max_tokens: {max_tokens} ({'full sequence' if max_tokens is None else f'last {max_tokens}'})")
        log(f"  training: {'full-batch numpy' if (is_linear or is_positional_mlp) and max_tokens is not None else 'mini-batch provider'}")

        # TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        probe_dir = output_dir / probe_key
        probe_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = probe_dir / "tb"
        tb_writer = SummaryWriter(log_dir=str(tb_dir))

        clf = MultiArchProbe(config=config, device=device)

        # Compute eval-time input slicing semantics for this probe. Persisted
        # in config.json; eval scripts apply the same slice to online hidden.
        if max_tokens is not None:
            eval_slice = [-max_tokens, None]  # default: last `max_tokens` tokens
            tp = probe_params.get("token_position")
            if tp is not None:
                # Positional probes (linear or MLP): slice to single token.
                # Single-position slice so the scaler fits only on the target
                # position's distribution (the library flattens B*T then fits,
                # so including other positions pollutes the stats).
                eval_slice = [-1, None] if tp == -1 else [tp, tp + 1]
        else:
            eval_slice = None
        probe_params["_eval_slice"] = eval_slice

        if (is_linear or is_positional_mlp) and max_tokens is not None and X_preloaded is not None:
            # ── Full-batch numpy path (convex, exact gradient) ──
            # Slice last max_tokens from the preloaded array
            if max_tokens < max_needed:
                X_all = X_preloaded[:, max_needed - max_tokens:, :]
            else:
                X_all = X_preloaded

            # Apply per-probe input slice (positional gets single token)
            if eval_slice is not None:
                s, e = eval_slice
                X_all = X_all[:, s:e, :]
                log(f"  Input slice {eval_slice}: X_all shape={X_all.shape}")

            X_train_np = X_all[train_idx]
            X_test_np = X_all[test_idx]
            log(f"  Using preloaded data: train {X_train_np.shape}, test {X_test_np.shape}")

            log("  Training (full-batch)...")
            t0 = time.time()
            clf.fit(X_train_np, train_labels, verbose=True)
            train_time = time.time() - t0
            log(f"  Training done in {train_time:.1f}s")

            # Evaluate
            log("  Evaluating...")
            test_scores = clf.predict_scores(X_test_np)

        else:
            # ── Mini-batch provider path (non-convex or full-sequence) ──

            if not is_linear and not is_positional_mlp:
                # MLP-based probes (mlp_all, attention_all, multimax_all, …):
                # reading each activation batch N times is the bottleneck (~4h/probe).
                # Defer to the IO-amortised parallel block that runs after this loop.
                # (mlp_pos-5 is excluded — it uses the preloaded numpy path above.)
                _parallel_pending[probe_key] = {
                    "clf": clf, "probe_dir": probe_dir, "probe_params": probe_params,
                    "max_tokens": max_tokens, "tb_writer": tb_writer,
                }
                continue

            train_prov = CachedBatchProvider(ds, indices=train_idx, max_tokens=max_tokens)
            test_prov = CachedBatchProvider(ds, indices=test_idx, max_tokens=max_tokens)

            epoch_history = []
            ckpt_dir = probe_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            def _on_epoch(_clf, epoch, metrics):
                avg_loss = metrics.get("avg_loss", 0)
                epoch_history.append({"epoch": epoch, "avg_loss": avg_loss})
                tb_writer.add_scalar(f"loss/{probe_key}", avg_loss, epoch)
                if epoch <= 5 or epoch % 50 == 0:
                    log(f"    epoch {epoch:>4d}: loss={avg_loss:.6f}")
                # Save checkpoint at key epochs
                if epoch in (1, 3, 5, 7, 10):
                    model_obj = getattr(_clf, '_model', None)
                    if model_obj is not None:
                        torch.save(model_obj.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

            log("  Training (mini-batch)...")
            t0 = time.time()
            clf.fit(batch_provider=train_prov, on_epoch_end=_on_epoch)
            train_time = time.time() - t0
            log(f"  Training done in {train_time:.1f}s")

            # Evaluate
            log("  Evaluating...")
            test_scores = clf.predict_scores(batch_provider=test_prov)

        tb_writer.close()

        # Save probe weights
        probe_obj = getattr(clf, '_model', None)
        if probe_obj is not None:
            torch.save(probe_obj.state_dict(), probe_dir / "probe.pt")

        # Save normalization state (if any) so eval scripts don't have to
        # reconstruct it. This covers the correctness gap described in
        # the probe architectures README.
        if probe_params.get("normalize") == "standard":
            if is_linear and max_tokens is not None:
                # Numpy path: sklearn StandardScaler stored on clf._scaler
                if clf._scaler is not None:
                    torch.save({
                        "mean": torch.from_numpy(clf._scaler.mean_.astype(np.float32)),
                        "std":  torch.from_numpy(clf._scaler.scale_.astype(np.float32)),
                        "source": "sklearn_StandardScaler",
                        "n_samples_seen": int(getattr(clf._scaler, "n_samples_seen_", 0)),
                        "max_tokens": int(max_tokens),
                    }, probe_dir / "scaler.pt")
                    log(f"  Saved scaler.pt (sklearn, n_seen={clf._scaler.n_samples_seen_})")
            else:
                # Provider path: Welford stats on clf._running_mean / _running_var
                if clf._running_mean is not None:
                    torch.save({
                        "mean": clf._running_mean.detach().cpu().float(),
                        "std":  clf._running_var.detach().cpu().float().sqrt().add_(1e-5),
                        "source": "welford",
                        "n_samples_seen": int(clf._n_seen),
                        "max_tokens": int(max_tokens) if max_tokens is not None else -1,
                    }, probe_dir / "scaler.pt")
                    log(f"  Saved scaler.pt (welford, n_seen={clf._n_seen})")

        # Save config and training history
        # Include _max_tokens so eval scripts can slice the right window for
        # windowed probes (mean_linear_last16, attention_all, etc.). Without
        # this, the probe gets the full sequence at eval and window-pooling
        # diverges from training. Tag with leading underscore to match the
        # PROBE_CONFIGS convention; eval scripts key on this at load time.
        config_out = dict(probe_params)
        config_out["_max_tokens"] = max_tokens  # None or int
        with open(probe_dir / "config.json", "w") as f:
            json.dump(config_out, f, indent=2)
        if 'epoch_history' in locals():
            with open(probe_dir / "train_history.json", "w") as f:
                json.dump(epoch_history, f, indent=2)

        metrics_05 = compute_metrics(test_labels, test_scores, threshold=0.5)

        # TPR at fixed FPR points
        fpr_targets = [0.001, 0.01, 0.05]
        fpr_results = {}
        for target_fpr in fpr_targets:
            tpr, thr, actual_fpr = tpr_at_fpr(test_labels, test_scores, target_fpr)
            fpr_results[f"fpr_{target_fpr}"] = {
                "tpr": tpr, "threshold": thr, "actual_fpr": actual_fpr,
            }

        result = {
            "probe_key": probe_key,
            "config": probe_params,
            "train_time_s": train_time,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "metrics_at_0.5": metrics_05,
            "fixed_fpr": fpr_results,
        }

        def _fmt(v):
            return "  N/A " if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.4f}"

        log(f"  Accuracy:     {_fmt(metrics_05['accuracy'])}")
        log(f"  AUC:          {_fmt(metrics_05['auc'])}")
        log(f"  Recall@0.5:   {_fmt(metrics_05['tpr'])}")
        log(f"  Precision@0.5:{_fmt(metrics_05['precision'])}")
        log(f"  FPR@0.5:      {_fmt(metrics_05['fpr'])}")
        for target_fpr in fpr_targets:
            fr = fpr_results[f"fpr_{target_fpr}"]
            tpr_s = _fmt(fr['tpr'])
            thr_s = _fmt(fr['threshold'])
            log(f"  TPR@FPR={target_fpr}: {tpr_s} (thr={thr_s})")

        # Save per-probe results (allow_nan so float('nan') → NaN in JSON)
        with open(probe_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2, allow_nan=True)

        # Save test scores for later perturbation analysis
        np.savez_compressed(
            probe_dir / "test_scores.npz",
            scores=test_scores,
            labels=test_labels,
            indices=test_idx,
        )

        all_results[probe_key] = result

        # Free GPU memory before next probe — large full-batch allocations
        # (e.g. mean_linear_last16 = 30GB) must be released or subsequent
        # probes (mlp_all etc.) can hit OOM or CUDA errors.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Parallel training for MLP provider-path probes ──────────────────────
    # Each activation batch is streamed once per epoch regardless of how many
    # MLP probes are in the group.  All MLP probes currently have max_tokens=None
    # (full sequence); grouped by max_tokens in case that changes in the future.
    if _parallel_pending:
        from collections import defaultdict
        _by_mt: dict = defaultdict(list)
        for pk in probe_keys:           # respect probe_keys ordering
            if pk in _parallel_pending:
                _by_mt[_parallel_pending[pk]["max_tokens"]].append(pk)

        for mt, group_keys in _by_mt.items():
            log(f"\n{'='*60}")
            log(f"  Parallel MLP training: {group_keys}  (max_tokens={mt})")
            log(f"{'='*60}")

            train_prov = CachedBatchProvider(ds, indices=train_idx, max_tokens=mt)
            _epoch_histories: dict = {pk: [] for pk in group_keys}

            def _make_cb(pk, pdir, tbw, hist, clf_ref):
                ckpt_dir = pdir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                def _cb(key, epoch, metrics):
                    avg_loss = metrics.get("avg_loss", 0)
                    hist.append({"epoch": epoch, "avg_loss": avg_loss})
                    tbw.add_scalar(f"loss/{key}", avg_loss, epoch)
                    if epoch <= 5 or epoch % 50 == 0:
                        log(f"    [{key}] epoch {epoch:>4d}: loss={avg_loss:.6f}")
                    if epoch in (1, 3, 5, 7, 10):
                        model_obj = getattr(clf_ref, '_model', None)
                        if model_obj is not None:
                            torch.save(model_obj.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")
                return _cb

            probes_dict = {}
            on_epoch_end_cbs = {}
            for pk in group_keys:
                info = _parallel_pending[pk]
                probes_dict[pk] = info["clf"]
                on_epoch_end_cbs[pk] = _make_cb(
                    pk, info["probe_dir"], info["tb_writer"],
                    _epoch_histories[pk], info["clf"],
                )

            t0 = time.time()
            train_probes_parallel(
                probes=probes_dict,
                batch_provider=train_prov,
                device=device,
                on_epoch_end=on_epoch_end_cbs,
                verbose=True,
            )
            train_time_group = time.time() - t0
            train_time_per = train_time_group / max(len(group_keys), 1)
            log(f"  Parallel training done in {train_time_group:.1f}s total "
                f"(~{train_time_per:.1f}s amortised per probe)")

            # ── Evaluate and save each probe in the group ────────────────────
            for pk in group_keys:
                info         = _parallel_pending[pk]
                clf          = info["clf"]
                probe_dir    = info["probe_dir"]
                probe_params = info["probe_params"]
                max_tokens   = info["max_tokens"]
                tb_writer    = info["tb_writer"]
                epoch_history = _epoch_histories[pk]

                log(f"\n  Evaluating {pk}...")
                test_prov = CachedBatchProvider(ds, indices=test_idx, max_tokens=max_tokens)
                test_scores = clf.predict_scores(batch_provider=test_prov)

                tb_writer.close()

                probe_obj = getattr(clf, '_model', None)
                if probe_obj is not None:
                    torch.save(probe_obj.state_dict(), probe_dir / "probe.pt")
                    log(f"  Saved probe.pt")

                # MLP probes use normalize="none" — no scaler.pt required.
                # If a future MLP config uses normalize="standard", add Welford
                # scaler saving here (same pattern as the sequential else branch).

                config_out = dict(probe_params)
                config_out["_max_tokens"] = max_tokens
                with open(probe_dir / "config.json", "w") as f:
                    json.dump(config_out, f, indent=2)
                with open(probe_dir / "train_history.json", "w") as f:
                    json.dump(epoch_history, f, indent=2)

                metrics_05 = compute_metrics(test_labels, test_scores, threshold=0.5)
                fpr_targets = [0.001, 0.01, 0.05]
                fpr_results = {}
                for target_fpr in fpr_targets:
                    tpr_v, thr_v, actual_fpr_v = tpr_at_fpr(test_labels, test_scores, target_fpr)
                    fpr_results[f"fpr_{target_fpr}"] = {
                        "tpr": tpr_v, "threshold": thr_v, "actual_fpr": actual_fpr_v,
                    }

                result = {
                    "probe_key": pk,
                    "config": probe_params,
                    "train_time_s": train_time_per,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "metrics_at_0.5": metrics_05,
                    "fixed_fpr": fpr_results,
                }

                def _fmt(v):
                    return "  N/A " if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.4f}"

                log(f"  [{pk}] Accuracy:     {_fmt(metrics_05['accuracy'])}")
                log(f"  [{pk}] AUC:          {_fmt(metrics_05['auc'])}")
                log(f"  [{pk}] Recall@0.5:   {_fmt(metrics_05['tpr'])}")
                log(f"  [{pk}] Precision@0.5:{_fmt(metrics_05['precision'])}")
                log(f"  [{pk}] FPR@0.5:      {_fmt(metrics_05['fpr'])}")
                for target_fpr in fpr_targets:
                    fr = fpr_results[f"fpr_{target_fpr}"]
                    log(f"  [{pk}] TPR@FPR={target_fpr}: {_fmt(fr['tpr'])} (thr={_fmt(fr['threshold'])})")

                with open(probe_dir / "results.json", "w") as f:
                    json.dump(result, f, indent=2, allow_nan=True)
                np.savez_compressed(
                    probe_dir / "test_scores.npz",
                    scores=test_scores,
                    labels=test_labels,
                    indices=test_idx,
                )
                all_results[pk] = result

                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ── Summary ──
    def _sfmt(v):
        """Format a metric value for the summary table (7-char wide)."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "    N/A"
        return f"{v:>7.4f}"

    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    log(f"{'Probe':>30s} {'Acc':>7s} {'AUC':>7s} {'Rec@.5':>7s} {'Prec@.5':>8s} {'FPR@.5':>7s} {'Rec@1%':>7s} {'Time':>6s}")
    log("-" * 83)
    for pk in probe_keys:
        r = all_results.get(pk)
        if r:
            m = r["metrics_at_0.5"]
            tpr1 = r["fixed_fpr"].get("fpr_0.01", {}).get("tpr", float("nan"))
            log(f"{pk:>30s}{_sfmt(m['accuracy'])}{_sfmt(m['auc'])}{_sfmt(m['tpr'])}"
                f" {_sfmt(m['precision'])}{_sfmt(m['fpr'])}{_sfmt(tpr1)} {r['train_time_s']:>5.0f}s")

    # Save combined results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump({
            "datasets": datasets,
            "split_seed": args.split_seed,
            "test_ratio": args.test_ratio,
            "n_total": len(ds),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "shared_params": SHARED_PARAMS,
            "results": all_results,
        }, f, indent=2, allow_nan=True)
    log(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
