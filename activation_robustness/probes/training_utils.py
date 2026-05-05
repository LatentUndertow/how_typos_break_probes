"""Shared training utilities for online probe classifiers.

Extracts common patterns from detection_probe.py, gemini_probe.py,
and run_gemini_ad_loso.py into standalone, reusable functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from tqdm import tqdm

from activation_robustness.data.activation_extractor import PrefetchExtractor


# ---------------------------------------------------------------------------
# Running statistics
# ---------------------------------------------------------------------------

def welford_update(
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    n_seen: int,
    batch_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Welford's online algorithm for running mean/variance.

    Args:
        running_mean: Current running mean tensor (d,) or None for first call.
        running_var: Current running variance tensor (d,) or None.
        n_seen: Number of samples seen so far (int).
        batch_values: New batch of values (n_batch, d) tensor.

    Returns:
        (updated_mean, updated_var, new_n_seen)
    """
    if batch_values.shape[0] == 0:
        if running_mean is None:
            raise ValueError("Cannot update with empty batch on first call")
        return running_mean, running_var, n_seen

    batch_n = batch_values.shape[0]
    batch_mean = batch_values.mean(dim=0)
    batch_var = batch_values.var(dim=0)

    new_n = n_seen + batch_n

    if n_seen == 0 or running_mean is None:
        return batch_mean, batch_var, new_n

    delta = batch_mean - running_mean
    updated_mean = running_mean + delta * (batch_n / new_n)
    m_old = running_var * n_seen
    m_new = batch_var * batch_n
    updated_var = (m_old + m_new + delta.pow(2) * n_seen * batch_n / new_n) / new_n

    return updated_mean, updated_var, new_n


# ---------------------------------------------------------------------------
# Online training config and loop
# ---------------------------------------------------------------------------

@dataclass
class OnlineTrainingConfig:
    """Configuration for train_probe_online."""
    epochs: int = 3
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 3e-3
    checkpoint_dir: Optional[Path] = None
    checkpoint_prefix: str = ""
    monitor_n: int = 200
    log_every: int = 100
    ae_lambda_l1: float = 1e-4
    ae_lambda_ortho: float = 1e-3
    dtype: str = "bfloat16"

    @property
    def torch_dtype(self) -> "torch.dtype":
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            self.dtype, torch.bfloat16
        )


# ---------------------------------------------------------------------------
# Online training loop
# ---------------------------------------------------------------------------

def train_probe_online(
    module: nn.Module,
    extractor,
    prompts: List[str],
    labels,
    config: OnlineTrainingConfig,
    callbacks: Optional[List[Callable]] = None,
) -> List[Tuple[int, int, float]]:
    """Online training loop for Gemini probe architectures with on-the-fly extraction.

    This is the List[str]-prompts training path for MultiArchProbe.
    It trains arbitrary nn.Module probes (MLP, attention, AlphaEvolve, EMA, etc.)
    by extracting all-position hidden states via PrefetchExtractor in
    mini-batches, then forwarding through the probe to produce (B,) logits.

    For batch-provider training (cached/live), see
    MultiArchProbe._fit_from_provider instead.

    Uses PrefetchExtractor to overlap CPU tokenization with GPU work.
    All computation is in bf16.

    Handles special module types via duck typing:
      - EMA modules (``hasattr(module, 'set_eval_ema')``): toggles EMA
        mode for monitoring vs training.
      - AlphaEvolve modules (``hasattr(module, 'l1_penalty') and
        hasattr(module, 'ortho_penalty')``): adds L1 and orthogonality
        regularization terms to the loss.

    Args:
        module: Probe nn.Module. Takes (B, T, D) bf16 input, outputs (B,) logits.
        extractor: ActivationExtractor with ``extract_all_positions``,
            ``tokenize_batch_async``, and ``forward_from_encoded`` methods.
        prompts: Training prompt strings.
        labels: Binary labels (list or array, length n).
        config: Training hyperparameters.
        callbacks: Optional list of callables ``fn(epoch, batch_idx, loss_val)``
            invoked after each optimizer step.

    Returns:
        loss_curve: list of (epoch, batch_idx, loss_value) tuples.
            An epoch summary entry with batch_idx=-1 is appended after each epoch.
    """
    device = next(module.parameters()).device
    n = len(prompts)
    batch_size = config.batch_size

    y_arr = np.asarray(labels, dtype=np.float32)
    _dtype = config.torch_dtype
    y_tensor = torch.tensor(y_arr, dtype=_dtype, device=device)

    # Pre-compute per-class weight scalars (allocated once, reused every batch)
    n_pos = float(y_arr.sum())
    n_neg = float(n - n_pos)
    w_pos_t = torch.tensor(n / (2.0 * max(n_pos, 1)), dtype=_dtype, device=device)
    w_neg_t = torch.tensor(n / (2.0 * max(n_neg, 1)), dtype=_dtype, device=device)

    is_ema = hasattr(module, 'set_eval_ema')
    is_alphaevolve = hasattr(module, 'l1_penalty') and hasattr(module, 'ortho_penalty')

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    loss_curve: List[Tuple[int, int, float]] = []

    # Fixed subsample for end-of-epoch monitoring
    monitor_n = min(config.monitor_n, n)
    rng = np.random.RandomState(42)
    monitor_idx = rng.choice(n, monitor_n, replace=False)

    for epoch in range(config.epochs):
        module.train()
        if is_ema:
            module.set_eval_ema(False)

        perm = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        n_total_batches = (n + batch_size - 1) // batch_size

        # Build batch index arrays
        batch_indices = [
            perm[start: start + batch_size]
            for start in range(0, n, batch_size)
        ]

        # Prefetch iterator: tokenize next batch on CPU while GPU processes current
        prefetch = PrefetchExtractor(
            extractor, prompts, batch_indices, return_offsets=False,
        )

        pbar = tqdm(prefetch, desc=f"Epoch {epoch + 1}/{config.epochs}", total=n_total_batches)

        for idx, (hidden, attn_mask) in pbar:
            yb = y_tensor[torch.as_tensor(idx, device=device, dtype=torch.long)]

            # Forward through probe (bf16)
            logits = module(hidden)  # (B,)

            # Weighted BCE loss with pre-allocated scalars
            weights = torch.where(yb > 0.5, w_pos_t, w_neg_t)
            loss = F.binary_cross_entropy_with_logits(logits, yb, weight=weights)

            # AlphaEvolve extra regularization
            if is_alphaevolve:
                loss = loss + config.ae_lambda_l1 * module.l1_penalty()
                loss = loss + config.ae_lambda_ortho * module.ortho_penalty()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            n_batches += 1
            loss_curve.append((epoch, n_batches, loss_val))
            pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

            if callbacks:
                for cb in callbacks:
                    cb(epoch, n_batches, loss_val)

            # Periodic plain-text logging for tail -f
            if config.log_every > 0 and n_batches % config.log_every == 0:
                recent = [l for _, _, l in loss_curve[-config.log_every:]]
                print(
                    f"  [batch {n_batches}/{n_total_batches}] "
                    f"running_avg={epoch_loss / n_batches:.4f}  "
                    f"last{config.log_every}_avg={np.mean(recent):.4f}  "
                    f"last{config.log_every}_min={np.min(recent):.4f}  "
                    f"last{config.log_every}_max={np.max(recent):.4f}",
                    flush=True,
                )

            del hidden, attn_mask

        avg_loss = epoch_loss / max(n_batches, 1)

        # --- End-of-epoch monitoring on subsample ---
        module.eval()
        if is_ema:
            module.set_eval_ema(True)

        mon_scores = []
        mon_labels = []
        # Build monitor batch indices
        mon_batch_indices = [
            monitor_idx[ms: ms + batch_size]
            for ms in range(0, monitor_n, batch_size)
        ]
        mon_prefetch = PrefetchExtractor(
            extractor, prompts, mon_batch_indices, return_offsets=False,
        )

        with torch.no_grad():
            for mi, (hid, _) in mon_prefetch:
                mb_labels = y_arr[mi]
                logits_m = module(hid)
                mon_scores.extend(torch.sigmoid(logits_m).float().cpu().numpy().tolist())
                mon_labels.extend(mb_labels.tolist())
                del hid

        mon_scores_arr = np.array(mon_scores)
        mon_labels_arr = np.array(mon_labels)
        mon_preds = (mon_scores_arr >= 0.5).astype(int)
        mon_acc = (mon_preds == mon_labels_arr).mean()
        mon_recall = (
            (mon_preds[mon_labels_arr == 1] == 1).mean()
            if (mon_labels_arr == 1).any()
            else 0.0
        )
        mon_fpr = (
            (mon_preds[mon_labels_arr == 0] == 1).mean()
            if (mon_labels_arr == 0).any()
            else 0.0
        )
        print(
            f"  Epoch {epoch + 1}: avg_loss={avg_loss:.4f}  "
            f"train_monitor(n={monitor_n}): acc={mon_acc:.3f} "
            f"recall={mon_recall:.3f} fpr={mon_fpr:.3f}"
        )
        loss_curve.append((epoch, -1, avg_loss))  # sentinel: batch_idx=-1 = epoch summary

        # --- Per-epoch checkpoint ---
        if config.checkpoint_dir and config.checkpoint_prefix:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = (
                config.checkpoint_dir
                / f"{config.checkpoint_prefix}_epoch{epoch + 1}.pt"
            )
            torch.save(
                {
                    "state_dict": {
                        k: v.cpu() for k, v in module.state_dict().items()
                    },
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "monitor": {
                        "acc": float(mon_acc),
                        "recall": float(mon_recall),
                        "fpr": float(mon_fpr),
                    },
                },
                ckpt_path,
            )
            print(f"    Checkpoint saved: {ckpt_path}")

        # Back to train mode for next epoch
        module.train()
        if is_ema:
            module.set_eval_ema(False)

    return loss_curve
