"""Train fast linear probes (positional_linear_pos-5, mean_linear_last16) on
all LODO folds with a SINGLE preload pass.

Why this exists: probe_architecture_sweep.py preloads the activation tensor
per invocation. For LODO that means 29 redundant preloads (~30min each on
cold disk cache). Since the underlying tensor is identical across folds and
only the train/test split changes, we can preload once and iterate.

Saves probes in the same format as probe_architecture_sweep.py so existing
perteval scripts (perteval_5fold.py, lodo_perteval_combined.py) can score
them without modification.

Usage:
    python train_lodo_fast_probes.py                       # all 29 folds, both probes
    python train_lodo_fast_probes.py --folds EnronDataset  # smoke test on 1 fold
    python train_lodo_fast_probes.py --probes positional_linear_pos-5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score



os.environ.setdefault("PYTHONUNBUFFERED", "1")

from activation_robustness.data.activation_cache import ActivationCache, CachedActivationDataset
from activation_robustness.probes.architectures import MultiArchProbe, ProbeConfig

CACHE_DIR = os.environ.get("ACTIVATION_CACHE_DIR", "./cache_data/activations")
LODO_DIR  = Path("./interpretability-research/activation_robustness/results/lodo_sweep_fixed")

DATASETS_ALL = [
    "EnronDataset", "Dolly15kDataset", "OpenOrcaDataset", "PromptsRanked10kDataset",
    "AlpacaDataset", "SafeGuardDataset", "QualifireDataset", "SoftAgeDataset",
    "BIPIADataset", "InjecAgentDataset", "LLMailDataset", "MosscapDataset",
    "WildJailbreakDataset", "JayavibhavDataset", "DeepsetDataset",
    "YanismiraouiDataset", "AdvBenchDataset", "HarmBenchDataset", "AgentDojoDataset",
    "APIGenMTDataset", "BitextCustomerSupportDataset", "CodeExerciseDataset",
    "GandalfSummarizationDataset", "JailbreakClassificationDataset",
    "PythonCodeAlpacaDataset", "PythonCodes25kDataset", "ScamDataset",
    "WritingPromptsDataset", "XlamFunctionCallingDataset",
]

# Mirrors probe_architecture_sweep.py PROBE_CONFIGS for the fast probes
PROBE_CONFIGS = {
    "positional_linear_pos-5": {
        "probe_type": "positional_linear",
        "token_position": -5,
        "_max_tokens": 16,
        "_eval_slice": [-5, -4],  # single token at pos -5
    },
    "mean_linear_last16": {
        "probe_type": "mean_linear",
        "_max_tokens": 16,
        "_eval_slice": [-16, None],
    },
}

LINEAR_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 3e-3,
    "max_epochs": 1000,
    "early_stopping_patience": 50,
    "val_split": 0.0,
    "normalize": "standard",
    "batch_size": 0,
    "use_class_weight": True,
    "random_state": 42,
}

MAX_TOKENS = 16  # all fast probes use last 16


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def preload_last_n_tokens(ds: CachedActivationDataset, max_tokens: int) -> np.ndarray:
    """Single pass over the cache to load last `max_tokens` of all samples."""
    n = len(ds)
    d = ds.d_model
    X = np.zeros((n, max_tokens, d), dtype=np.float32)
    chunk = 512
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        idxs = np.arange(start, end)
        hidden, _, _, _ = ds.load_batch(idxs, max_tokens=max_tokens)
        X[start:end] = hidden.float().numpy()
        if start % 5000 < chunk:
            log(f"  preload: {end}/{n}")
    log(f"Preloaded in {time.time()-t0:.1f}s, shape={X.shape}, {X.nbytes/1e9:.1f} GB")
    return X


def tpr_at_fpr(y_true, y_score, target_fpr):
    """Compute TPR at a target FPR, plus the actual FPR achieved."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # Find largest threshold whose FPR <= target
    valid = fpr <= target_fpr
    if not valid.any():
        return float("nan"), float("nan"), float("nan")
    idx = np.where(valid)[0][-1]
    return float(tpr[idx]), float(thr[idx]), float(fpr[idx])


def train_one_probe(
    probe_key: str,
    fold_dir: Path,
    X_preloaded: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
):
    cfg_extras = PROBE_CONFIGS[probe_key]
    eval_slice = cfg_extras["_eval_slice"]
    max_tokens = cfg_extras["_max_tokens"]

    probe_dir = fold_dir / probe_key
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Skip if results.json already present (resume support)
    if (probe_dir / "results.json").exists():
        log(f"    [SKIP] {probe_key} (results.json exists)")
        return

    # Build classifier
    config_dict = {
        **LINEAR_PARAMS,
        "probe_type": cfg_extras["probe_type"],
    }
    if "token_position" in cfg_extras:
        config_dict["token_position"] = cfg_extras["token_position"]
    config = ProbeConfig(**config_dict)

    clf = MultiArchProbe(config=config, device=device)

    # Slice preloaded data
    s, e = eval_slice
    X_all = X_preloaded[:, s:e, :]  # (N, slice_len, D)
    X_train = X_all[train_idx]
    X_test = X_all[test_idx]

    log(f"    [{probe_key}] X_train={X_train.shape}  X_test={X_test.shape}")

    t0 = time.time()
    clf.fit(X_train, y_train, verbose=False)
    fit_t = time.time() - t0
    log(f"    [{probe_key}] fit in {fit_t:.1f}s")

    test_scores = clf.predict_scores(X_test)
    test_pred = (test_scores >= 0.5).astype(int)

    # ── Metrics ──
    acc = float(accuracy_score(y_test, test_pred))
    if len(np.unique(y_test)) == 2:
        auc = float(roc_auc_score(y_test, test_scores))
    else:
        auc = float("nan")
    metrics_05 = {
        "accuracy": acc,
        "auc": auc,
        "tpr": float(((test_pred == 1) & (y_test == 1)).sum() / max((y_test == 1).sum(), 1)),
        "precision": float(((test_pred == 1) & (y_test == 1)).sum() / max((test_pred == 1).sum(), 1)),
        "fpr": float(((test_pred == 1) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1)),
        "f1": float(2 * ((test_pred == 1) & (y_test == 1)).sum() /
                    max(((test_pred == 1).sum() + (y_test == 1).sum()), 1)),
    }
    fpr_targets = [0.001, 0.01, 0.05]
    fpr_results = {}
    for tf in fpr_targets:
        tpr, thr, actual = tpr_at_fpr(y_test, test_scores, tf)
        fpr_results[f"fpr_{tf}"] = {"tpr": tpr, "threshold": thr, "actual_fpr": actual}

    # ── Save artifacts (matching probe_architecture_sweep.py format) ──
    probe_obj = clf._model
    torch.save(probe_obj.state_dict(), probe_dir / "probe.pt")

    if clf._scaler is not None:
        torch.save({
            "mean": torch.from_numpy(clf._scaler.mean_.astype(np.float32)),
            "std":  torch.from_numpy(clf._scaler.scale_.astype(np.float32)),
            "source": "sklearn_StandardScaler",
            "n_samples_seen": int(getattr(clf._scaler, "n_samples_seen_", 0)),
            "max_tokens": int(max_tokens),
        }, probe_dir / "scaler.pt")

    config_out = {**LINEAR_PARAMS, **{k: v for k, v in cfg_extras.items()
                                       if k != "_eval_slice" and k != "_max_tokens"}}
    config_out["_max_tokens"] = max_tokens
    config_out["_eval_slice"] = eval_slice
    with open(probe_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    result = {
        "probe_key": probe_key,
        "config": {**config_dict, "_max_tokens": max_tokens, "_eval_slice": eval_slice},
        "train_time_s": fit_t,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "metrics_at_0.5": metrics_05,
        "fixed_fpr": fpr_results,
    }
    with open(probe_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    np.savez_compressed(probe_dir / "test_scores.npz",
                        scores=test_scores, labels=y_test)

    log(f"    [{probe_key}] acc={acc:.4f}  AUC={auc:.4f}  "
        f"TPR@1%={fpr_results['fpr_0.01']['tpr']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=None,
                        help="LODO fold names to train (default: all 29)")
    parser.add_argument("--probes", nargs="+",
                        default=list(PROBE_CONFIGS.keys()),
                        choices=list(PROBE_CONFIGS.keys()))
    parser.add_argument("--lodo-dir", default=str(LODO_DIR),
                        help="Dir with fold_*/split_indices.npz")
    args = parser.parse_args()

    lodo_dir = Path(args.lodo_dir)
    fold_names = args.folds or DATASETS_ALL

    log(f"Probes: {args.probes}")
    log(f"Folds:  {len(fold_names)}")
    log(f"LODO dir: {lodo_dir}")

    log("Loading cache + labels (canonical 29-dataset ordering)...")
    cache = ActivationCache(cache_dir=CACHE_DIR)
    ds = CachedActivationDataset(cache, DATASETS_ALL)
    labels = ds.labels.numpy().astype(np.int32)
    pids   = np.array(ds.prompt_ids)
    n_total = len(ds)
    log(f"  {n_total} samples, d_model={ds.d_model}")

    log(f"Preloading last {MAX_TOKENS} tokens once...")
    X_preloaded = preload_last_n_tokens(ds, MAX_TOKENS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    for fold_name in fold_names:
        fold_dir = lodo_dir / f"fold_{fold_name}"
        split_path = fold_dir / "split_indices.npz"
        if not split_path.exists():
            log(f"\n[fold_{fold_name}] SKIP — no split file at {split_path}")
            continue

        split = np.load(split_path, allow_pickle=True)
        train_idx = split["train"]
        test_idx  = split["test"]

        # Validate split (catch broken files)
        held_out = str(split.get("lodo_held_out", "")) if "lodo_held_out" in split.files else ""
        if held_out and held_out != fold_name:
            log(f"  WARN: split lodo_held_out='{held_out}' but fold_name='{fold_name}'")
        test_pfx = set(p.split(":")[0] for p in pids[test_idx])
        if len(test_pfx) != 1:
            log(f"  ERROR: fold_{fold_name} test contains multiple prefixes: {test_pfx}")
            continue

        y_train = labels[train_idx]
        y_test  = labels[test_idx]
        log(f"\n[fold_{fold_name}] train={len(train_idx)} ({y_train.sum()} mal)  "
            f"test={len(test_idx)} ({y_test.sum()} mal)")

        for probe_key in args.probes:
            train_one_probe(
                probe_key=probe_key,
                fold_dir=fold_dir,
                X_preloaded=X_preloaded,
                train_idx=train_idx,
                test_idx=test_idx,
                y_train=y_train,
                y_test=y_test,
                device=device,
            )

    log("\nALL DONE")


if __name__ == "__main__":
    main()
