#!/usr/bin/env python3
"""
5-fold CV training — single preload, all folds in one run.

Key optimisations vs calling probe_architecture_sweep.py 5 times:
  1. 16-token activation window preloaded once (~44 GB) and reused across
     all 5 folds for positional/windowed probes.
  2. Full-sequence MLP probes (mlp_all, multimax_all, attention_all) trained
     with a SINGLE NVMe pass per epoch: one CachedBatchProvider streams every
     batch once; per-batch fold masks route each sample to the 4 folds that
     treat it as train, skipping the 1 fold that holds it out as test.
     Result: 5× less NVMe IO for the bottleneck probes.

Output layout is identical to probe_architecture_sweep.py so perteval_5fold.py
consumes results unchanged.

Usage:
    python train_5fold_shared.py --output-dir results/5fold_cv
    python train_5fold_shared.py --output-dir results/5fold_cv --probe-types positional_linear_pos-5 mlp_pos-5
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from activation_robustness.data.activation_cache import ActivationCache, CachedActivationDataset
from activation_robustness.data.batch_provider import CachedBatchProvider
from activation_robustness.probes.architectures import MultiArchProbe, ProbeConfig

CACHE_DIR  = os.environ.get("ACTIVATION_CACHE_DIR", "./cache_data/activations")
SPLIT_SEED = 42
N_FOLDS    = 5

# 9 datasets shipped in this release (subset of the paper's 29-dataset corpus).
# Pass --datasets to override.
DATASETS_ALL = [
    "OpenOrcaDataset", "AlpacaDataset", "Dolly15kDataset", "BitextCustomerSupportDataset",
    "DeepsetDataset", "BIPIADataset", "InjecAgentDataset", "HarmBenchDataset", "AdvBenchDataset",
]

PROBE_CONFIGS = {
    "positional_linear_pos-5": {
        "probe_type": "positional_linear",
        "token_position": -5,
        "_max_tokens": 16,
        "_base_params": "linear",
    },
    "mlp_pos-5": {
        "probe_type": "mlp",
        "token_position": -5,
        "mlp_hidden_dim": 100,
        "mlp_layers": 2,
        "_max_tokens": 16,
        "_base_params": "mlp",
    },
    "mlp_all": {
        "probe_type": "mlp",
        "mlp_hidden_dim": 100,
        "mlp_layers": 2,
        "_max_tokens": None,
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
    # Windowed probes — last-N token aggregation. eval_slice=[-N, None].
    # Used to test whether limiting the context window cushions ESW fragility.
    # last32 variants OOM on 80GB H100 with full-batch numpy training
    # (134k × 32 × 4096 × 4 = 69GB just for the train tensor); kept to last16.
    "mean_linear_last16": {
        "probe_type": "mean_linear",
        "_max_tokens": 16,
        "_base_params": "linear",
    },
    "ema_last16": {
        "probe_type": "ema",
        "ema_alpha": 0.5,
        "_max_tokens": 16,
        "_base_params": "linear",
    },
}

_LINEAR_PARAMS = {
    "lr": 1e-3, "weight_decay": 3e-3, "max_epochs": 1000,
    "early_stopping_patience": 50, "val_split": 0.0,
    "normalize": "standard", "batch_size": 0,
    "use_class_weight": True, "random_state": 42,
}
_MLP_PARAMS = {
    "lr": 1e-4, "weight_decay": 3e-3, "max_epochs": 5,
    "early_stopping_patience": 50, "val_split": 0.0,
    "normalize": "none", "batch_size": 64,
    "use_class_weight": True, "random_state": 42,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_metrics(labels, scores, threshold=0.5):
    preds = (scores >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    f1  = f1_score(labels, preds, zero_division=0)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {
        "accuracy": float(acc), "auc": float(auc), "f1": float(f1),
        "tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1), "tnr": tn / max(tn + fp, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": float(threshold), "n_samples": int(len(labels)),
        "n_malicious": int(labels.sum()), "n_benign": int(len(labels) - labels.sum()),
        "mean_score_malicious": float(scores[labels==1].mean()) if labels.sum()>0 else float("nan"),
        "mean_score_benign":    float(scores[labels==0].mean()) if (labels==0).sum()>0 else float("nan"),
    }


def tpr_at_fpr(labels, scores, target_fpr):
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    from sklearn.metrics import roc_curve
    fprs, tprs, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fprs - target_fpr))
    return float(tprs[idx]), float(thresholds[idx]), float(fprs[idx])


def save_probe_results(probe_dir, clf_or_model, probe_params, max_tokens,
                       train_time, train_idx, test_idx, test_scores, test_labels):
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    model_obj = getattr(clf_or_model, '_model', clf_or_model)
    if isinstance(model_obj, nn.Module):
        torch.save(model_obj.state_dict(), probe_dir / "probe.pt")

    # Save scaler if present
    if probe_params.get("normalize") == "standard":
        clf = clf_or_model if hasattr(clf_or_model, '_scaler') else None
        if clf is not None and getattr(clf, '_scaler', None) is not None:
            torch.save({
                "mean": torch.from_numpy(clf._scaler.mean_.astype(np.float32)),
                "std":  torch.from_numpy(clf._scaler.scale_.astype(np.float32)),
                "source": "sklearn_StandardScaler",
                "n_samples_seen": int(getattr(clf._scaler, "n_samples_seen_", 0)),
                "max_tokens": int(max_tokens) if max_tokens else -1,
            }, probe_dir / "scaler.pt")

    config_out = dict(probe_params)
    config_out["_max_tokens"] = max_tokens
    with open(probe_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    metrics_05 = compute_metrics(test_labels, test_scores)
    fpr_results = {}
    for tfpr in [0.001, 0.01, 0.05]:
        tpr_v, thr_v, afpr_v = tpr_at_fpr(test_labels, test_scores, tfpr)
        fpr_results[f"fpr_{tfpr}"] = {"tpr": tpr_v, "threshold": thr_v, "actual_fpr": afpr_v}

    result = {
        "probe_key": probe_dir.name,
        "config": probe_params,
        "train_time_s": train_time,
        "n_train": len(train_idx),
        "n_test":  len(test_idx),
        "metrics_at_0.5": metrics_05,
        "fixed_fpr": fpr_results,
    }
    with open(probe_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    np.savez_compressed(
        probe_dir / "test_scores.npz",
        scores=test_scores, labels=test_labels, indices=test_idx,
    )
    return metrics_05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--probe-types", nargs="+", default=None)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    out_base   = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    datasets   = args.datasets or DATASETS_ALL
    probe_keys = args.probe_types or list(PROBE_CONFIGS.keys())
    n_folds    = args.n_folds
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for k in probe_keys:
        if k not in PROBE_CONFIGS:
            log(f"Unknown probe: {k}. Available: {list(PROBE_CONFIGS.keys())}")
            sys.exit(1)

    log(f"Output: {out_base}")
    log(f"Datasets: {len(datasets)}  Probes: {probe_keys}  Folds: {n_folds}")

    # ── Load data ──────────────────────────────────────────────────────────────
    log("\nLoading activation cache...")
    cache = ActivationCache(cache_dir=CACHE_DIR)
    missing = [d for d in datasets if not cache.is_cached(d)]
    if missing:
        log(f"ERROR: Missing from cache: {missing}"); sys.exit(1)
    ds = CachedActivationDataset(cache, datasets)
    log(f"Loaded {len(ds)} samples, d_model={ds.d_model}")

    all_labels  = ds.labels.numpy().astype(np.float32)
    all_indices = np.arange(len(ds))
    log(f"Labels: {int(all_labels.sum())} mal, {int((all_labels==0).sum())} ben")

    # ── Compute all fold splits ────────────────────────────────────────────────
    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.split_seed)
    folds = list(skf.split(all_indices, all_labels))
    log(f"\n{n_folds}-fold CV splits (seed={args.split_seed}):")

    fold_dirs = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        fold_dir = out_base / f"fold_{fold_i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_dirs.append(fold_dir)
        split_path = fold_dir / "split_indices.npz"
        if not split_path.exists():
            np.savez(split_path, train=train_idx, test=test_idx,
                     seed=args.split_seed, kfold=n_folds, fold_idx=fold_i)
        n_mal_tr = int(all_labels[train_idx].sum())
        n_mal_te = int(all_labels[test_idx].sum())
        log(f"  fold_{fold_i}: train={len(train_idx)} ({n_mal_tr} mal), "
            f"test={len(test_idx)} ({n_mal_te} mal)")

    # fold_membership[i] = which fold uses sample i as TEST (-1 if none, shouldn't happen)
    fold_membership = np.full(len(ds), -1, dtype=np.int8)
    for fold_i, (_, test_idx) in enumerate(folds):
        fold_membership[test_idx] = fold_i

    # ── Separate probe keys by path ────────────────────────────────────────────
    # Windowed probes (single-position OR last-N window): numpy preload path.
    #   - single-position: token_position present, _max_tokens set, eval_slice=[tp, tp+1]
    #   - windowed:        no token_position, _max_tokens set, eval_slice=[-max_tokens, None]
    # Full-sequence MLP probes: single-pass fold-masked provider path.
    windowed_keys = [k for k in probe_keys
                     if PROBE_CONFIGS[k].get("_max_tokens") is not None]
    fullseq_keys  = [k for k in probe_keys
                     if PROBE_CONFIGS[k].get("_max_tokens") is None]

    log(f"\nWindowed probes (shared preload): {windowed_keys}")
    log(f"Full-seq probes (single-pass):    {fullseq_keys}")

    # ── Preload max-needed-window once ────────────────────────────────────────
    windowed_todo = [
        (fi, pk) for fi in range(n_folds) for pk in windowed_keys
        if not (fold_dirs[fi] / pk / "results.json").exists()
    ]
    X_preloaded = None
    if windowed_todo:
        max_tokens_pre = max(PROBE_CONFIGS[k]["_max_tokens"] for k in windowed_keys)
        log(f"\nPreloading last {max_tokens_pre} tokens for all {len(ds)} samples...")
        t_pre = time.time()
        n, d = len(ds), ds.d_model
        X_preloaded = np.zeros((n, max_tokens_pre, d), dtype=np.float32)
        chunk = 512
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            hidden, mask, _, _ = ds.load_batch(np.arange(start, end), max_tokens=max_tokens_pre)
            X_preloaded[start:end] = hidden.float().numpy()
            if start % 10000 < chunk:
                log(f"    preload: {end}/{n}")
        log(f"Preloaded in {time.time()-t_pre:.1f}s, shape={X_preloaded.shape}, "
            f"{X_preloaded.nbytes/1e9:.1f} GB")
    else:
        log("All windowed probes already done — skipping preload.")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Windowed probes (single-position + last-N): shared preload, loop folds
    # ══════════════════════════════════════════════════════════════════════════
    for probe_key in windowed_keys:
        cfg        = PROBE_CONFIGS[probe_key]
        max_tokens = cfg["_max_tokens"]
        base       = cfg["_base_params"]
        tp         = cfg.get("token_position")
        if tp is not None:
            eval_slice = [-1, None] if tp == -1 else [tp, tp + 1]
        else:
            # Last-N window — slice the full N tokens
            eval_slice = [-max_tokens, None]

        probe_cfg = {k: v for k, v in cfg.items()
                     if not k.startswith("_")}
        base_params = _LINEAR_PARAMS if base == "linear" else _MLP_PARAMS
        probe_params = {**base_params, **probe_cfg, "_eval_slice": eval_slice}

        log(f"\n{'='*60}")
        log(f"  Positional probe: {probe_key}  (all {n_folds} folds)")
        log(f"{'='*60}")

        # Check if any fold needs training for this probe
        folds_needed = [fi for fi in range(n_folds)
                        if not (fold_dirs[fi] / probe_key / "results.json").exists()]
        if not folds_needed:
            log(f"  all folds already done, skipping")
            continue

        # Slice preloaded data to single position — shared across all folds
        s, e = eval_slice
        X_pos = X_preloaded[:, s:e, :]   # (n, 1, d_model) — read-only view
        log(f"  Slice {eval_slice}: X_pos shape={X_pos.shape}")

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            probe_dir  = fold_dirs[fold_i] / probe_key
            result_path = probe_dir / "results.json"
            if result_path.exists():
                log(f"  fold_{fold_i}: already done, skipping")
                continue

            config   = ProbeConfig(**{k: v for k, v in probe_params.items() if not k.startswith("_")})
            clf      = MultiArchProbe(config=config, device=device)
            X_train  = X_pos[train_idx]
            X_test   = X_pos[test_idx]
            y_train  = all_labels[train_idx]
            y_test   = all_labels[test_idx]

            log(f"  fold_{fold_i}: training on {len(train_idx)} samples...")
            t0 = time.time()
            clf.fit(X_train, y_train, verbose=False)
            train_time = time.time() - t0

            test_scores = clf.predict_scores(X_test)
            m = save_probe_results(probe_dir, clf, probe_params, max_tokens,
                                   train_time, train_idx, test_idx, test_scores, y_test)
            log(f"  fold_{fold_i}: done {train_time:.1f}s  "
                f"AUC={m['auc']:.4f}  TPR={m['tpr']:.4f}  FPR={m['fpr']:.4f}")

        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Full-sequence MLP probes: single NVMe pass per epoch
    # ══════════════════════════════════════════════════════════════════════════
    if not fullseq_keys:
        log("\nNo full-sequence probes to train.")
    else:
        log(f"\n{'='*60}")
        log(f"  Full-seq MLP probes: {fullseq_keys}")
        log(f"  Single NVMe pass per epoch, all {n_folds} folds simultaneously")
        log(f"{'='*60}")

        # Determine which (fold, probe) pairs still need training
        todo = [
            (fold_i, pk)
            for fold_i in range(n_folds)
            for pk in fullseq_keys
            if not (fold_dirs[fold_i] / pk / "results.json").exists()
        ]
        if not todo:
            log("  All full-seq probes already done, skipping.")
        else:
            log(f"  {len(todo)} (fold, probe) pairs to train")

            # Build per-fold-probe models and optimizers
            models_map    = {}   # (fold_i, pk) -> nn.Module
            optimizers_map = {}
            criteria_map  = {}

            # We need d_model to build models — already known
            d_model = ds.d_model

            for fold_i, pk in todo:
                cfg        = PROBE_CONFIGS[pk]
                probe_cfg  = {k: v for k, v in cfg.items() if not k.startswith("_")}
                probe_params = {**_MLP_PARAMS, **probe_cfg,
                                "_eval_slice": None, "_max_tokens": None}

                config = ProbeConfig(**{k: v for k, v in probe_params.items() if not k.startswith("_")})
                clf    = MultiArchProbe(config=config, device=device)
                clf._d_model = d_model
                model  = clf._build_model(d_model).to(device)

                # Per-fold class weight
                train_idx = folds[fold_i][0]
                n_mal = int(all_labels[train_idx].sum())
                n_ben = len(train_idx) - n_mal
                pos_w = torch.tensor([n_ben / max(n_mal, 1)], device=device, dtype=torch.float32)

                models_map[(fold_i, pk)]     = model
                optimizers_map[(fold_i, pk)] = torch.optim.AdamW(
                    model.parameters(), lr=_MLP_PARAMS["lr"],
                    weight_decay=_MLP_PARAMS["weight_decay"],
                )
                criteria_map[(fold_i, pk)] = nn.BCEWithLogitsLoss(pos_weight=pos_w,
                                                                   reduction='none')

            max_epochs = _MLP_PARAMS["max_epochs"]
            batch_size = _MLP_PARAMS["batch_size"]
            log(f"  Training {len(todo)} models × {max_epochs} epochs, batch_size={batch_size}")

            # Single-pass training loop
            t0_total = time.time()
            for epoch in range(max_epochs):
                t_ep = time.time()

                # One CachedBatchProvider over ALL samples, single ordered pass
                prov    = CachedBatchProvider(ds, indices=None, max_tokens=None)
                batches = prov._make_batches(batch_size=batch_size, shuffle=True,
                                             seed=epoch * 1000 + args.split_seed)

                epoch_losses = {key: 0.0 for key in todo}
                n_batches_seen = 0

                for batch_idx_arr, (hidden, mask, batch_labels, _) in zip(
                    batches, prov.iter_train_batches(batch_size, epoch_seed=epoch * 1000 + args.split_seed,
                                                     device=device)
                ):
                    # hidden is bf16 on GPU (worker threads prefetch + transfer in background).
                    # .float() is a fast CUDA cast, not a CPU op + large PCIe transfer.
                    hidden_gpu = hidden.float()  # (B, T, d_model) fp32 on GPU

                    for fold_i, pk in todo:
                        # Samples where this fold trains (not the held-out test fold)
                        train_mask = fold_membership[batch_idx_arr] != fold_i
                        if not train_mask.any():
                            continue

                        model     = models_map[(fold_i, pk)]
                        optimizer = optimizers_map[(fold_i, pk)]
                        criterion = criteria_map[(fold_i, pk)]

                        # Full-batch forward; zero out test-fold samples in the loss.
                        # Avoids 3D fancy indexing which hits CUDA kernel size limits.
                        labels_all = torch.from_numpy(
                            all_labels[batch_idx_arr]
                        ).float().to(device)
                        sample_w = torch.from_numpy(
                            train_mask.astype(np.float32)
                        ).to(device)
                        n_train = sample_w.sum()

                        model.train()
                        optimizer.zero_grad()
                        logits   = model(hidden_gpu, mask).squeeze(-1)    # (B,)
                        loss_per = criterion(logits, labels_all)           # (B,) reduction='none'
                        loss     = (loss_per * sample_w).sum() / n_train  # scalar
                        loss.backward()
                        optimizer.step()

                        epoch_losses[(fold_i, pk)] += loss.item()

                    n_batches_seen += 1
                    if n_batches_seen % 500 == 0:
                        elapsed = time.time() - t_ep
                        rate    = n_batches_seen / elapsed
                        log(f"  epoch {epoch+1} [{n_batches_seen}/{len(batches)}]  "
                            f"{rate:.1f} it/s")

                ep_time = time.time() - t_ep
                avg_losses = {
                    f"f{fi}_{pk}": f"{epoch_losses[(fi,pk)]/max(n_batches_seen,1):.4f}"
                    for fi, pk in todo[:3]   # log first 3 to keep line short
                }
                log(f"  epoch {epoch+1}/{max_epochs} done  {ep_time:.0f}s  "
                    f"sample losses: {avg_losses}")

            log(f"\n  All epochs done in {time.time()-t0_total:.0f}s")

            # ── Evaluate and save each (fold, probe) ──────────────────────────
            log("  Evaluating...")
            for fold_i, pk in todo:
                cfg        = PROBE_CONFIGS[pk]
                probe_cfg  = {k: v for k, v in cfg.items() if not k.startswith("_")}
                probe_params = {**_MLP_PARAMS, **probe_cfg,
                                "_eval_slice": None, "_max_tokens": None}

                _, test_idx = folds[fold_i]
                model       = models_map[(fold_i, pk)]
                model.eval()

                test_prov = CachedBatchProvider(ds, indices=test_idx, max_tokens=None)
                test_scores_list = []
                with torch.no_grad():
                    for hidden, mask in test_prov.iter_predict_batches(batch_size=64, device=device):
                        logits = model(hidden.float(), mask).squeeze(-1)
                        test_scores_list.append(torch.sigmoid(logits).cpu().numpy())
                test_scores = np.concatenate(test_scores_list)
                y_test      = all_labels[test_idx]

                probe_dir = fold_dirs[fold_i] / pk
                m = save_probe_results(probe_dir, model, probe_params, None,
                                       (time.time() - t0_total) / len(todo),
                                       folds[fold_i][0], test_idx, test_scores, y_test)
                log(f"  fold_{fold_i} {pk}: AUC={m['auc']:.4f}  TPR={m['tpr']:.4f}  FPR={m['fpr']:.4f}")

            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("SUMMARY  (mean ± std across folds)")
    log(f"{'='*70}")
    log(f"{'Probe':30s}  {'ACC':14s}  {'AUC':14s}  {'TPR':14s}  {'FPR':14s}")
    log("-" * 80)

    for pk in probe_keys:
        accs, aucs, tprs, fprs = [], [], [], []
        for fold_i in range(n_folds):
            rpath = fold_dirs[fold_i] / pk / "results.json"
            if not rpath.exists(): continue
            with open(rpath) as f:
                r = json.load(f)
            m = r["metrics_at_0.5"]
            for lst, key in [(accs,"accuracy"),(aucs,"auc"),(tprs,"tpr"),(fprs,"fpr")]:
                v = m.get(key, float("nan"))
                if not (v != v): lst.append(v)

        def _fmt(lst):
            return f"{np.mean(lst):.4f}±{np.std(lst):.4f}" if lst else "     N/A      "
        log(f"{pk:30s}  {_fmt(accs)}  {_fmt(aucs)}  {_fmt(tprs)}  {_fmt(fprs)}")

    log(f"\nAll results in: {out_base}")
    log("Done. Run perteval_5fold.py next.")


if __name__ == "__main__":
    main()
