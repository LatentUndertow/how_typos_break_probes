#!/usr/bin/env python3
"""End-to-end smoke test of the §6/§7 probe pipeline.

Exercises every major component (extractor, cache, data loader, probe,
training, evaluation) on a tiny corpus (~50 benign + ~50 malicious) so a
fresh checkout can be sanity-checked in under ~10 min.

Usage:
    python activation_robustness/experiments/smoke_e2e.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Point the data loader at our symlinked BIPIA / InjecAgent (only used if
# those datasets are listed; harmless when not).
os.environ.setdefault("BIPIA_ROOT", str(_REPO / "data" / "BIPIA"))
os.environ.setdefault("INJECAGENT_ROOT", str(_REPO / "data" / "InjecAgent"))

from activation_robustness.data.activation_extractor import ActivationExtractor
from activation_robustness.data.activation_cache import ActivationCache, CachedActivationDataset
from activation_robustness.data.data_loader import DataLoader
from activation_robustness.probes.architectures import MultiArchProbe, ProbeConfig

CACHE_DIR = str(_REPO / "cache_data" / "smoke_test")
N_PER_DATASET = 50

DATASETS = [
    {"class": "AlpacaDataset",    "max_samples": N_PER_DATASET},  # benign
    {"class": "HarmBenchDataset", "max_samples": N_PER_DATASET},  # malicious
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t_start = time.time()

    # ── 1. Load model + extractor ────────────────────────────────────────
    log("Step 1/5 — Loading Llama-3.1-8B-Instruct (slow first time)...")
    extractor = ActivationExtractor(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        layer=31,
        max_seq_len=2048,
    )
    log(f"  Model loaded.  d_model={extractor.d_model}  device={extractor.device}")

    # ── 2. DataLoader ────────────────────────────────────────────────────
    log("Step 2/5 — Setting up DataLoader (chat template + dataset registry)")
    loader = DataLoader(tokenizer=extractor.tokenizer, add_generation_prompt=True)

    # ── 3. Build the activation cache ────────────────────────────────────
    log(f"Step 3/5 — Building activation cache at {CACHE_DIR}")
    cache = ActivationCache(cache_dir=CACHE_DIR, extractor=extractor, loader=loader, batch_size=4)
    for spec in DATASETS:
        log(f"  precompute_dataset: {spec['class']}  (max_samples={spec.get('max_samples')})")
        stats = cache.precompute_dataset(spec["class"], spec)
        log(f"    → {stats.get('n_samples', '?')} samples, "
            f"{stats.get('total_tokens', '?')} tokens, "
            f"{stats.get('size_bytes', 0) / 1e6:.1f} MB, "
            f"{stats.get('time_s', 0):.1f}s")

    # ── 4. Load X/y from the cache and fit a Linear probe ────────────────
    log("Step 4/5 — Training a Linear probe at user-EOT (positional_linear, token=-1)")
    cached_ds = CachedActivationDataset(cache, [spec["class"] for spec in DATASETS])
    log(f"  cached samples: {len(cached_ds)}  d_model={cached_ds.d_model}")

    labels = cached_ds.labels.numpy().astype(np.float32)
    n_mal = int(labels.sum()); n_ben = int((labels == 0).sum())
    log(f"  labels: {n_mal} mal, {n_ben} ben")
    if n_mal == 0 or n_ben == 0:
        sys.exit("FAIL: smoke needs both classes; aborting")

    # 80/20 stratified split (small N → simple deterministic shuffle)
    rng = np.random.default_rng(0)
    n = len(cached_ds)
    perm = rng.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = perm[:split], perm[split:]
    log(f"  split: train N={len(train_idx)}, test N={len(test_idx)}")

    # Load just the last token per sample → shape (n, 1, D)
    X_train, _, y_train, _ = cached_ds.load_batch(train_idx, max_tokens=1)
    X_test,  _, y_test,  _ = cached_ds.load_batch(test_idx,  max_tokens=1)
    X_train_np = X_train.float().numpy()
    X_test_np  = X_test.float().numpy()
    y_train_np = y_train.numpy().astype(np.int64)
    y_test_np  = y_test.numpy().astype(np.int64)
    log(f"  X_train shape: {X_train_np.shape}  y_train: {y_train_np.shape}")

    config = ProbeConfig(
        probe_type="positional_linear",
        token_position=-1,        # user-EOT readout
        max_epochs=50,            # small for smoke
        early_stopping_patience=10,
        lr=1e-3,
        batch_size=16,
        val_split=0.15,           # internal split for early stopping
        random_state=0,
    )
    probe = MultiArchProbe(config=config, device=extractor.device)

    log("  fitting probe...")
    t_train = time.time()
    probe.fit(X_train_np, y_train_np, verbose=False)
    log(f"  fit complete in {time.time() - t_train:.1f}s")

    # ── 5. Predict on test set + report ─────────────────────────────────
    log("Step 5/5 — Predicting on held-out 20 % + computing AUC + TPR")
    scores = probe.predict_scores(X_test_np)
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(y_test_np, scores)
    fpr, tpr, _ = roc_curve(y_test_np, scores)
    tpr_at_10 = float(np.interp(0.10, fpr, tpr))
    tpr_at_20 = float(np.interp(0.20, fpr, tpr))
    log(f"  test AUC = {auc:.3f}")
    log(f"  TPR@FPR=10%: {tpr_at_10:.3f}")
    log(f"  TPR@FPR=20%: {tpr_at_20:.3f}")

    # On a 100-sample smoke we expect AUC ~ 1.0 if the probe learned anything
    # (Alpaca vs HarmBench is structurally easy at the last-token level).
    if auc < 0.7:
        sys.exit(f"FAIL: probe didn't learn (AUC={auc:.3f}); pipeline issue")

    # ── 6. Perturbation evaluation (apply typos, re-extract, predict) ────
    log("Step 6/7 — Perturbation eval: apply AdjacentKey typo, re-extract, predict")
    from activation_robustness.perturbations.typo import AdjacentKey
    import torch

    # Re-load the same prompts in the same order via DataLoader so we can
    # access the formatted texts and align with the cached order.
    all_samples = []
    for spec in DATASETS:
        all_samples.extend(loader.load(spec, verbose=False))
    test_texts = [all_samples[i]["text"] for i in test_idx]

    adj_key = AdjacentKey()
    rng_p = np.random.default_rng(123)
    perturbed_texts = []
    for txt in test_texts:
        out = adj_key.apply(txt, rng_p)
        perturbed_texts.append(out if out else txt)

    log(f"  re-extracting {len(perturbed_texts)} perturbed test prompts...")
    t_pert = time.time()
    hidden, mask = extractor.extract_all_positions(perturbed_texts)
    last_idx = mask.sum(dim=1) - 1
    X_pert = torch.stack([hidden[i, last_idx[i].item()] for i in range(len(perturbed_texts))])
    X_pert_np = X_pert.unsqueeze(1).float().cpu().numpy()
    log(f"  extracted in {time.time() - t_pert:.1f}s, shape={X_pert_np.shape}")

    scores_pert = probe.predict_scores(X_pert_np)
    auc_pert = roc_auc_score(y_test_np, scores_pert)
    fpr_p, tpr_p, _ = roc_curve(y_test_np, scores_pert)
    tpr_pert_at_10 = float(np.interp(0.10, fpr_p, tpr_p))
    log(f"  CLEAN     AUC={auc:.3f}      TPR@FPR=10%={tpr_at_10:.3f}")
    log(f"  PERTURBED AUC={auc_pert:.3f}  TPR@FPR=10%={tpr_pert_at_10:.3f}")
    log(f"  Δ AUC = {auc - auc_pert:+.3f}     Δ TPR = {tpr_at_10 - tpr_pert_at_10:+.3f}")

    # ── 7. KV-fork: train + eval probe at end-of-suffix ──────────────────
    log("Step 7/7 — KV-fork: append generic suffix + retrain + evaluate")
    SUFFIX_NEUTRAL = (
        " Before responding, take a moment to carefully reflect on the message "
        "above. Make sure that your answer is complete, accurate, and clearly "
        "expressed throughout."
    )
    fork_train_texts = [all_samples[i]["text"] + SUFFIX_NEUTRAL for i in train_idx]
    fork_test_texts  = [all_samples[i]["text"] + SUFFIX_NEUTRAL for i in test_idx]

    log(f"  extracting {len(fork_train_texts)} fork-train prompts...")
    t_fork = time.time()
    h_tr, m_tr = extractor.extract_all_positions(fork_train_texts)
    last_tr = m_tr.sum(dim=1) - 1
    X_fork_train = torch.stack([h_tr[i, last_tr[i].item()] for i in range(len(fork_train_texts))])
    X_fork_train_np = X_fork_train.unsqueeze(1).float().cpu().numpy()

    h_te, m_te = extractor.extract_all_positions(fork_test_texts)
    last_te = m_te.sum(dim=1) - 1
    X_fork_test = torch.stack([h_te[i, last_te[i].item()] for i in range(len(fork_test_texts))])
    X_fork_test_np = X_fork_test.unsqueeze(1).float().cpu().numpy()
    log(f"  fork extraction done in {time.time() - t_fork:.1f}s")

    fork_probe = MultiArchProbe(config=config, device=extractor.device)
    fork_probe.fit(X_fork_train_np, y_train_np, verbose=False)
    scores_fork_clean = fork_probe.predict_scores(X_fork_test_np)
    auc_fork_clean = roc_auc_score(y_test_np, scores_fork_clean)

    # Perturbed + suffix
    fork_pert_texts = [perturbed_texts[i] + SUFFIX_NEUTRAL for i in range(len(perturbed_texts))]
    h_fp, m_fp = extractor.extract_all_positions(fork_pert_texts)
    last_fp = m_fp.sum(dim=1) - 1
    X_fork_pert = torch.stack([h_fp[i, last_fp[i].item()] for i in range(len(fork_pert_texts))])
    X_fork_pert_np = X_fork_pert.unsqueeze(1).float().cpu().numpy()
    scores_fork_pert = fork_probe.predict_scores(X_fork_pert_np)
    auc_fork_pert = roc_auc_score(y_test_np, scores_fork_pert)
    fpr_fp, tpr_fp, _ = roc_curve(y_test_np, scores_fork_pert)
    tpr_fork_pert_at_10 = float(np.interp(0.10, fpr_fp, tpr_fp))

    log(f"  KV-FORK CLEAN     AUC={auc_fork_clean:.3f}")
    log(f"  KV-FORK PERTURBED AUC={auc_fork_pert:.3f}  TPR@FPR=10%={tpr_fork_pert_at_10:.3f}")
    log(f"  Δ AUC vs clean baseline   = {auc_fork_clean - auc:+.3f}")
    log(f"  Δ AUC vs perturbed-base   = {auc_fork_pert - auc_pert:+.3f} (positive = KV-fork recovers)")

    log(f"\n✓ E2E smoke complete — total wall {time.time() - t_start:.1f}s")
    log(f"  Baseline:  CLEAN AUC={auc:.3f}, PERTURBED AUC={auc_pert:.3f}")
    log(f"  KV-fork:   CLEAN AUC={auc_fork_clean:.3f}, PERTURBED AUC={auc_fork_pert:.3f}")


if __name__ == "__main__":
    main()
