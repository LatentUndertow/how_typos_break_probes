"""
multi_probe_trainer.py — IO-amortized parallel training for multiple MultiArchProbes.

Problem
-------
Training N provider-path probes sequentially reads every activation batch N times
from the cache.  For full-sequence probes (mlp_all, attention_all, multimax_all) on a
large corpus this is the dominant bottleneck: ~4 h/probe on 168K samples, ~12 h total.

Solution
--------
Read each activation batch ONCE, then run forward + backward for every probe within
the same step.  IO cost drops from O(N) to O(1); only the GPU work scales with N.
Since all probe models are small (<1 MB each), fitting many on a 79 GB GPU is trivial.

Usage
-----
    from activation_robustness.probes.trainer import train_probes_parallel
    from activation_robustness.probes.architectures import MultiArchProbe

    probes = {
        "mlp_all":      MultiArchProbe(config=mlp_config,      device=device),
        "attention_all":MultiArchProbe(config=attn_config,     device=device),
        "multimax_all": MultiArchProbe(config=mmax_config,     device=device),
    }
    histories = train_probes_parallel(probes, batch_provider, device)
    # Each clf is now fitted: clf._model, clf._fitted, clf._running_mean, etc.

Constraints
-----------
- All probes must share the same batch_size (used to iterate the provider).
- Supports normalize in {'none', 'standard', 'l2'} per probe independently.
- Does NOT support alphaevolve extra penalties (L1/ortho) — use standard fit() for those.
- Each probe runs for its own config.max_epochs; training stops per-probe when done.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from activation_robustness.probes.training_utils import welford_update


# ---------------------------------------------------------------------------
# Internal per-probe state
# ---------------------------------------------------------------------------

@dataclass
class _ProbeState:
    key: str
    clf: "MultiArchProbe"  # noqa: F821
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    max_epochs: int
    # Welford running stats (only used when normalize='standard')
    running_mean: Optional[torch.Tensor] = None
    running_var: Optional[torch.Tensor] = None
    n_seen: int = 0
    # Tracking
    current_epoch: int = 0
    done: bool = False
    epoch_history: list = field(default_factory=list)


def _build_state(key: str, clf, batch_provider, device: torch.device) -> _ProbeState:
    """Set up training state for one probe."""
    d_model = batch_provider.d_model
    clf._d_model = d_model

    model = clf._build_model(d_model).to(device=device, dtype=clf.config.torch_dtype)
    model.train()
    clf._model = model   # expose immediately so epoch callbacks can checkpoint it

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=clf.config.lr,
        weight_decay=clf.config.weight_decay,
        betas=clf.config.adamw_betas,
    )

    if clf.config.use_class_weight:
        n_mal = int(batch_provider.labels.sum())
        n_ben = len(batch_provider.labels) - n_mal
        pos_weight = torch.tensor(
            [n_ben / max(n_mal, 1)], device=device, dtype=clf.config.torch_dtype,
        )
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Welford state for standard normalization
    running_mean = running_var = None
    if clf.config.normalize == "standard":
        running_mean = torch.zeros(d_model, dtype=clf.config.torch_dtype, device=device)
        running_var  = torch.ones( d_model, dtype=clf.config.torch_dtype, device=device)

    return _ProbeState(
        key=key,
        clf=clf,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        max_epochs=clf.config.max_epochs,
        running_mean=running_mean,
        running_var=running_var,
    )


def _normalize(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    state: _ProbeState,
    epoch: int,
) -> torch.Tensor:
    """Apply per-probe normalization, updating Welford stats on epoch 0.

    Mirrors the normalization block in MultiArchProbe._fit_from_provider
    exactly — same Welford update using mask, same dtype handling — so that
    multi-probe and single-probe training produce identical dynamics.
    """
    cfg = state.clf.config
    if cfg.normalize == "standard":
        if epoch == 0:
            valid = hidden[mask]
            if valid.shape[0] > 0:
                state.running_mean, state.running_var, state.n_seen = welford_update(
                    state.running_mean, state.running_var, state.n_seen, valid,
                )
        X = (hidden - state.running_mean) / (state.running_var.sqrt() + 1e-5)
    elif cfg.normalize == "l2":
        import torch.nn.functional as F
        X = F.normalize(hidden.float(), p=2, dim=-1, eps=1e-12).to(hidden.dtype)
    else:  # "none"
        X = hidden
    return X


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_probes_parallel(
    probes: "dict[str, MultiArchProbe]",  # noqa: F821
    batch_provider,
    device: torch.device,
    batch_size: Optional[int] = None,
    verbose: bool = True,
    on_epoch_end: Optional[dict] = None,  # key -> callable(key, epoch, metrics)
) -> "dict[str, list[dict]]":
    """Train multiple MultiArchProbes sharing a single batch provider.

    Parameters
    ----------
    probes:
        Mapping of probe_key -> MultiArchProbe (not yet fitted).
        All probes must use the same batch_size in their config, or batch_size
        must be supplied explicitly.
    batch_provider:
        A BatchProvider (CachedBatchProvider or similar).  Iterated ONCE per
        epoch regardless of how many probes there are.
    device:
        Torch device to train on.
    batch_size:
        Override batch size.  If None, uses probes[first].config.batch_size.
    verbose:
        Print per-epoch loss for each probe.
    on_epoch_end:
        Optional dict of per-probe callbacks: on_epoch_end[key](key, epoch, metrics).

    Returns
    -------
    histories : dict[str, list[dict]]
        Per-probe list of epoch metrics dicts.
    """
    if not probes:
        return {}

    first_clf = next(iter(probes.values()))
    bs = batch_size or (first_clf.config.batch_size if first_clf.config.batch_size > 0
                        else batch_provider.n_samples)

    # Validate consistent batch_size across probes
    for key, clf in probes.items():
        cfg_bs = clf.config.batch_size if clf.config.batch_size > 0 else batch_provider.n_samples
        if batch_size is None and cfg_bs != bs:
            raise ValueError(
                f"Probe '{key}' has batch_size={cfg_bs} but first probe has {bs}. "
                "Pass batch_size= explicitly to override."
            )

    torch.manual_seed(first_clf.config.random_state)
    torch.cuda.manual_seed(first_clf.config.random_state)

    # Build per-probe state
    states = {
        key: _build_state(key, clf, batch_provider, device)
        for key, clf in probes.items()
    }

    max_epochs_global = max(s.max_epochs for s in states.values())
    n_samples = batch_provider.n_samples
    total_batches_est = (n_samples + bs - 1) // bs
    log_every = max(1, total_batches_est // 5)

    if verbose:
        keys = list(probes.keys())
        print(f"[MultiProbeTrainer] {len(keys)} probes: {keys}", flush=True)
        print(f"[MultiProbeTrainer] {n_samples} samples, batch_size={bs}, "
              f"up to {max_epochs_global} epochs", flush=True)

    for epoch in range(max_epochs_global):
        active = [s for s in states.values() if not s.done]
        if not active:
            break

        for s in active:
            s.model.train()

        epoch_losses = {s.key: 0.0 for s in active}
        n_batches = 0
        epoch_start = time.time()

        # ── Single pass through the provider ─────────────────────────────
        epoch_seed = first_clf.config.random_state + epoch
        for hidden, mask, batch_labels, _spans in batch_provider.iter_train_batches(
            bs,
            epoch_seed=epoch_seed,
            device=device,
            return_offsets=False,
        ):
            y_batch_raw = batch_labels.to(device=device)

            with torch.no_grad():
                # Per-probe normalization (Welford update only on epoch 0)
                X_per_probe = {
                    s.key: _normalize(hidden, mask, s, epoch)
                    for s in active
                }

            for s in active:
                y_batch = y_batch_raw.to(dtype=s.clf.config.torch_dtype)
                s.optimizer.zero_grad()
                logits = s.model(X_per_probe[s.key], mask)
                loss = s.criterion(logits, y_batch)
                loss.backward()
                s.optimizer.step()
                epoch_losses[s.key] += loss.item()

            n_batches += 1
            if verbose and n_batches % log_every == 0:
                elapsed = time.time() - epoch_start
                it_s = n_batches / elapsed if elapsed > 0 else 0
                loss_str = "  ".join(
                    f"{s.key}={epoch_losses[s.key]/n_batches:.4f}" for s in active
                )
                print(f"  Epoch {epoch+1} [{n_batches}/{total_batches_est}]  "
                      f"{loss_str}  {it_s:.1f} it/s", flush=True)

        # ── Per-probe epoch bookkeeping ───────────────────────────────────
        epoch_elapsed = time.time() - epoch_start
        for s in active:
            avg_loss = epoch_losses[s.key] / max(n_batches, 1)
            metrics = {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "n_batches": n_batches,
                "epoch_time_s": epoch_elapsed,
            }
            s.epoch_history.append(metrics)
            s.current_epoch += 1

            if verbose and (epoch + 1) % max(1, max_epochs_global // 10) == 0:
                print(f"  [{s.key}] Epoch {epoch+1}: loss={avg_loss:.4f} "
                      f"({epoch_elapsed:.1f}s)", flush=True)

            if on_epoch_end and s.key in on_epoch_end:
                on_epoch_end[s.key](s.key, epoch + 1, metrics)

            if s.current_epoch >= s.max_epochs:
                s.done = True

    # ── Finalise each probe ───────────────────────────────────────────────
    for s in states.values():
        clf = s.clf
        clf._model = s.model
        clf._model.eval()
        clf._fitted = True
        # Attach Welford stats so predict_scores can normalize consistently
        if s.running_mean is not None:
            clf._running_mean = s.running_mean
            clf._running_var  = s.running_var
            clf._n_seen       = s.n_seen

    if verbose:
        print("[MultiProbeTrainer] Done.", flush=True)

    return {s.key: s.epoch_history for s in states.values()}
