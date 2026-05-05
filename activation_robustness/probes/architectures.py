"""
Gemini probe architectures implementing ClassifierProtocol.

Based on: "Building Production-Ready Probes For Gemini"
(Kramár et al., 2026)

7 probe architectures for classifying LLM activations across multiple
token positions. All probes take (B, T, D) input and output (B,) logits.

Probe types:
- mean_linear: Linear probe per token, mean-pooled (Eq. 3)
- positional_linear: Linear probe on a configurable single token position
- ema: Linear probe + EMA at inference, max over positions (Eq. 4)
- mlp: MLP per token, mean-pooled (Eq. 5-6)
- attention: MLP + multi-head softmax attention pooling (Eq. 7-8)
- multimax: MLP + max per head, no softmax (Eq. 9)
- rolling_attention: MLP + windowed attention, max over windows (Eq. 10)
- alphaevolve: LayerNorm + MLP + gated projection + bipolar pool (Algorithm 1)

Reference sections:
- Equations 3-10: Probe definitions
- Algorithm 1: AlphaEvolve probe
- Appendix C: Hyperparameters (width 100, 2 layers, lr=1e-4, wd=3e-3)
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, normalize
from sklearn.metrics import f1_score
from typing import Optional, Literal, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
import copy
from activation_robustness.probes.training_utils import welford_update
from activation_robustness.data.batch_provider import BatchProvider


# =============================================================================
# Shared Infrastructure
# =============================================================================

def build_mlp(
    d_in: int,
    hidden_dim: int,
    n_layers: int,
    activation: str = 'relu',
    out_dim: int = 1,
) -> nn.Sequential:
    """
    Build MLP as described in Eq. 5.

    Args:
        d_in: Input dimension
        hidden_dim: Hidden layer width
        n_layers: Number of hidden layers
        activation: Activation function ('relu' or 'gelu')

    Returns:
        nn.Sequential MLP mapping d_in → out_dim
    """
    act_fn = nn.GELU if activation == 'gelu' else nn.ReLU
    layers = []
    prev_dim = d_in
    for _ in range(n_layers):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act_fn())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, out_dim))
    return nn.Sequential(*layers)


# =============================================================================
# Probe Modules (all take (B, T, D) → (B,) logits)
# =============================================================================

class MeanLinearModule(nn.Module):
    """
    Mean-pooled linear probe (Eq. 3).

    Per-token linear: s_t = w^T x_t + b
    Output: mean_t(s_t)
    """

    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) bool, True = valid
        s = self.linear(x).squeeze(-1)  # (B, T)
        if mask is not None:
            s = s * mask.float()
            return s.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
        return s.mean(dim=1)  # (B,)


class PositionalLinearModule(nn.Module):
    """Linear probe on a single token position (negative = from end)."""

    def __init__(self, d_model: int, token_position: int = -5, **kwargs):
        super().__init__()
        self.token_position = token_position
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) bool, True = valid
        offset = abs(self.token_position)
        if mask is None:
            pos = max(x.shape[1] - offset, 0)
            token_hidden = x[:, pos, :]
        else:
            seq_lens = mask.sum(dim=1).to(torch.long)
            pos = torch.clamp(seq_lens - offset, min=0)
            batch_idx = torch.arange(x.shape[0], device=x.device)
            token_hidden = x[batch_idx, pos, :]
        return self.linear(token_hidden).squeeze(-1)  # (B,)


class EMAModule(nn.Module):
    """
    EMA probe (Eq. 4).

    Training: same as MeanLinearModule (mean pool for gradient flow).
    Inference: EMA_j = alpha * s_j + (1-alpha) * EMA_{j-1}, output max_j EMA_j.
    """

    def __init__(self, d_model: int, ema_alpha: float = 0.5, **kwargs):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        self.ema_alpha = ema_alpha
        self._eval_mode_ema = False

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) bool, True = valid
        s = self.linear(x).squeeze(-1)  # (B, T)
        if self._eval_mode_ema:
            return self._forward_ema(s, mask)
        if mask is not None:
            s = s * mask.float()
            return s.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
        return s.mean(dim=1)  # (B,) - train mode: mean pool

    def _forward_ema(self, s: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """EMA inference: max_j EMA_j, respecting padding mask."""
        # s: (B, T), mask: (B, T) bool, True = valid
        _, T = s.shape
        alpha = self.ema_alpha
        # Eq. 4 uses EMA_0 = 0.
        ema = torch.zeros_like(s[:, 0])
        max_ema = torch.full_like(ema, float('-inf'))
        for t in range(T):
            if mask is not None:
                valid = mask[:, t].float()
                # Only update EMA for valid (non-padded) positions
                ema = valid * (alpha * s[:, t] + (1 - alpha) * ema) + (1 - valid) * ema
                max_ema = torch.where(mask[:, t], torch.max(max_ema, ema), max_ema)
            else:
                ema = alpha * s[:, t] + (1 - alpha) * ema
                max_ema = torch.max(max_ema, ema)
        return max_ema  # (B,)

    def set_eval_ema(self, enabled: bool):
        """Toggle EMA inference mode."""
        self._eval_mode_ema = enabled


class MLPModule(nn.Module):
    """
    MLP probe with mean pooling (Eq. 5-6).

    Per-token MLP: s_t = MLP(x_t)
    Output: mean_t(s_t)
    """

    def __init__(self, d_model: int, hidden_dim: int = 100, n_layers: int = 2,
                 activation: str = 'relu', **kwargs):
        super().__init__()
        self.mlp = build_mlp(d_model, hidden_dim, n_layers, activation)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) bool, True = valid
        s = self.mlp(x).squeeze(-1)  # (B, T)
        if mask is not None:
            s = s * mask.float()
            return s.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
        return s.mean(dim=1)  # (B,)


class AttentionModule(nn.Module):
    """
    MLP + multi-head attention pooling (Eq. 7-8).

    Per-token transformed feature: y_t = MLP(x_tad) ∈ R^{d'}
    Per-head attention: alpha_{h,t} ∝ exp(q_h^T y_t)
    Per-head value: v_{h,t} = v_h^T y_t
    Per-head output: s_h = sum_t alpha_{h,t} * v_{h,t}
    Final: mean_h(s_h)
    """

    def __init__(self, d_model: int, n_heads: int = 10, hidden_dim: int = 100,
                 n_layers: int = 2, activation: str = 'relu', **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.mlp_out_dim = hidden_dim
        self.mlp = build_mlp(
            d_model, hidden_dim, n_layers, activation, out_dim=self.mlp_out_dim
        )
        self.query = nn.Parameter(torch.randn(n_heads, self.mlp_out_dim) * 0.02)
        self.value = nn.Parameter(torch.randn(n_heads, self.mlp_out_dim) * 0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) bool, True = valid
        y = self.mlp(x)  # (B, T, d')
        attn_logits = torch.einsum('btd,hd->bth', y, self.query)  # (B, T, H)
        values = torch.einsum('btd,hd->bth', y, self.value)  # (B, T, H)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, H) softmax over T
        # Weighted sum per head
        s = (attn_weights * values).sum(dim=1)  # (B, H)
        return s.mean(dim=1)  # (B,)


class MultiMaxModule(nn.Module):
    """
    MLP + max per head, no softmax (Eq. 9).

    Per-token transformed feature: y_t = MLP(x_t) ∈ R^{d'}
    Per-head value: v_{h,t} = v_h^T y_t
    Per-head: s_h = max_t(v_{h,t})
    Final: mean_h(s_h)
    """

    def __init__(self, d_model: int, n_heads: int = 10, hidden_dim: int = 100,
                 n_layers: int = 2, activation: str = 'relu', **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.mlp_out_dim = hidden_dim
        self.mlp = build_mlp(
            d_model, hidden_dim, n_layers, activation, out_dim=self.mlp_out_dim
        )
        self.value = nn.Parameter(torch.randn(n_heads, self.mlp_out_dim) * 0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) bool, True = valid
        y = self.mlp(x)  # (B, T, d')
        values = torch.einsum('btd,hd->bth', y, self.value)  # (B, T, H)
        if mask is not None:
            values = values.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        s = values.max(dim=1).values  # (B, H)
        return s.mean(dim=1)  # (B,)


class RollingAttentionModule(nn.Module):
    """
    MLP + windowed attention, max over windows (Eq. 10).

    Per-token transformed feature: y_t = MLP(x_t) ∈ R^{d'}
    Per window of size W: attention-pool within a sliding window → s_w ∈ R^H
    Per-head: max over windows
    Final: mean_h(max_w s_{h,w})
    """

    def __init__(self, d_model: int, n_heads: int = 10, window_size: int = 10,
                 hidden_dim: int = 100, n_layers: int = 2, activation: str = 'relu',
                 **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.window_size = window_size
        self.mlp_out_dim = hidden_dim
        self.mlp = build_mlp(
            d_model, hidden_dim, n_layers, activation, out_dim=self.mlp_out_dim
        )
        self.query = nn.Parameter(torch.randn(n_heads, self.mlp_out_dim) * 0.02)
        self.value = nn.Parameter(torch.randn(n_heads, self.mlp_out_dim) * 0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) bool, True = valid
        _, T, _ = x.shape
        y = self.mlp(x)  # (B, T, d')
        # Zero out padded positions before computing logits/values
        if mask is not None:
            y = y * mask.unsqueeze(-1).float()
        attn_logits = torch.einsum('btd,hd->bth', y, self.query)  # (B, T, H)
        values = torch.einsum('btd,hd->bth', y, self.value)  # (B, T, H)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        W = self.window_size
        if T <= W:
            # For short sequences, treat the full sequence as the only window.
            attn_weights = F.softmax(attn_logits, dim=1)
            s = (attn_weights * values).sum(dim=1)  # (B, H)
            return s.mean(dim=1)

        # Sliding windows over all contiguous windows of width W.
        # Shapes: (B, n_windows, H, W) where n_windows = T - W + 1.
        attn_win = attn_logits.unfold(dimension=1, size=W, step=1)
        val_win = values.unfold(dimension=1, size=W, step=1)

        attn_weights = F.softmax(attn_win, dim=-1)  # softmax over window positions
        s_win = (attn_weights * val_win).sum(dim=-1)  # (B, n_windows, H)

        # Max over windows per head
        s_max = s_win.max(dim=1).values  # (B, H)
        return s_max.mean(dim=1)  # (B,)


class AlphaEvolveModule(nn.Module):
    """
    AlphaEvolve probe (Algorithm 1).

    LayerNorm → MLP → projected gated features → bipolar pooling (max, -min).

    Architecture:
    1. LayerNorm(x_t)
    2. h_t = MLP(x_t)
    3. v_t = (W_proj h_t) * softplus(W_gate h_t)
    4. pool_pos = max_t(v_t), pool_neg = -min_t(v_t) (per dimension)
    6. s = w^T [pool_pos; pool_neg] + b

    Regularization: L1 on model weights + orthogonality penalty on W_proj.
    """

    def __init__(self, d_model: int, ae_proj_dim: int = 64, hidden_dim: int = 100,
                 n_layers: int = 2, activation: str = 'relu', **kwargs):
        super().__init__()
        self.proj_dim = ae_proj_dim
        self.layer_norm = nn.LayerNorm(d_model)

        # Shared feature MLP producing H in Algorithm 1.
        self.mlp = build_mlp(
            d_model, hidden_dim, n_layers, activation, out_dim=hidden_dim
        )

        # V = (W_proj H) ⊙ softplus(W_gate H)
        self.proj = nn.Linear(hidden_dim, ae_proj_dim)
        self.gate = nn.Linear(hidden_dim, ae_proj_dim)

        # Final classifier on bipolar pool: [max; -min] → 2*proj_dim → 1
        self.classifier = nn.Linear(2 * ae_proj_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) bool, True = valid
        x_norm = self.layer_norm(x)  # (B, T, D)
        h = self.mlp(x_norm)  # (B, T, hidden_dim)
        v = self.proj(h) * F.softplus(self.gate(h))  # (B, T, proj_dim)

        # Bipolar pooling — mask padded positions out of max/min
        if mask is not None:
            m = mask.unsqueeze(-1)  # (B, T, 1)
            v_max = v.masked_fill(~m, float('-inf'))
            v_min = v.masked_fill(~m, float('inf'))
        else:
            v_max = v_min = v
        pool_pos = v_max.max(dim=1).values  # (B, proj_dim)
        pool_neg = -v_min.min(dim=1).values  # (B, proj_dim)
        pooled = torch.cat([pool_pos, pool_neg], dim=1)  # (B, 2*proj_dim)

        return self.classifier(pooled).squeeze(-1)  # (B,)

    def l1_penalty(self) -> torch.Tensor:
        """L1 penalty over all weight matrices in the module."""
        penalties = [
            p.abs().mean() for p in self.parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        if not penalties:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.stack(penalties).sum()

    def ortho_penalty(self) -> torch.Tensor:
        """Orthogonality penalty on W_proj^T W_proj."""
        W = self.proj.weight  # (proj_dim, hidden_dim)
        WTW = W.T @ W
        I = torch.eye(WTW.shape[0], device=W.device, dtype=W.dtype)
        return (WTW - I).pow(2).mean()


# =============================================================================
# Module Registry
# =============================================================================

_MODULE_REGISTRY: Dict[str, type] = {
    'mean_linear': MeanLinearModule,
    'positional_linear': PositionalLinearModule,
    'ema': EMAModule,
    'mlp': MLPModule,
    'attention': AttentionModule,
    'multimax': MultiMaxModule,
    'rolling_attention': RollingAttentionModule,
    'alphaevolve': AlphaEvolveModule,
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProbeConfig:
    """
    Unified configuration for all Gemini probe types.

    Defaults from Appendix C: 2 hidden layers of width 100, lr=1e-4, wd=3e-3.
    Irrelevant fields are ignored per probe_type.

    Attributes:
        probe_type: Which probe architecture to use
        lr: Learning rate (AdamW)
        weight_decay: L2 regularization (AdamW)
        max_epochs: Maximum training epochs (full-batch steps)
        early_stopping_patience: Stop if no improvement for N epochs
        val_split: Fraction of training data for early-stopping validation
        normalize: Input normalization ('none', 'standard', 'maxabs', 'l2')
        mlp_hidden_dim: MLP hidden layer width
        mlp_layers: Number of MLP hidden layers
        mlp_activation: Activation function ('relu' or 'gelu')
        n_heads: Number of attention/multimax heads
        ema_alpha: EMA smoothing factor
        window_size: Window size for rolling attention
        ae_proj_dim: AlphaEvolve projection dimension
        ae_lambda_l1: AlphaEvolve L1 penalty weight
        ae_lambda_ortho: AlphaEvolve orthogonality penalty weight
        batch_size: Batch size (0 = full-batch, paper default)
        adamw_betas: AdamW beta parameters
        random_state: Random seed
    """
    probe_type: Literal[
        'mean_linear', 'positional_linear', 'ema', 'mlp', 'attention',
        'multimax', 'rolling_attention', 'alphaevolve',
    ] = 'mean_linear'
    lr: float = 1e-4
    weight_decay: float = 3e-3
    max_epochs: int = 1000
    early_stopping_patience: int = 50
    val_split: float = 0.15
    normalize: Literal['none', 'standard', 'maxabs', 'l2'] = 'none'
    mlp_hidden_dim: int = 100
    mlp_layers: int = 2
    mlp_activation: str = 'relu'
    n_heads: int = 10
    token_position: int = -5  # token position for positional_linear probe
    ema_alpha: float = 0.5
    window_size: int = 10
    ae_proj_dim: int = 64
    ae_lambda_l1: float = 1e-4
    ae_lambda_ortho: float = 1e-3
    batch_size: int = 0
    adamw_betas: Tuple[float, float] = (0.9, 0.999)
    random_state: int = 42
    use_class_weight: bool = True  # auto pos_weight = n_ben/n_mal for BCEWithLogitsLoss
    dtype: str = "bfloat16"  # torch dtype for probe weights and training

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            self.dtype, torch.bfloat16
        )


# =============================================================================
# Classifier (implements ClassifierProtocol)
# =============================================================================

class MultiArchProbe:
    """
    Gemini probe classifier implementing ClassifierProtocol.

    Single classifier class dispatching to 7 probe architectures via
    probe_type config. Handles 2D (single position) and 3D (multi-position)
    inputs uniformly.

    Supports two modes:
    - Numpy mode: fit(X_array, y) with pre-extracted activations
    - Online mode: fit(prompts_list, y) with on-the-fly extraction via ActivationExtractor

    Example (numpy):
        >>> clf = MultiArchProbe(ProbeConfig(probe_type='attention'))
        >>> clf.fit(X_train, y_train)  # X: (n, T, D) or (n, D)
        >>> scores = clf.predict_scores(X_test)

    Example (online):
        >>> from activation_robustness.data.activation_extractor import ActivationExtractor
        >>> ext = ActivationExtractor("meta-llama/Llama-3.1-8B-Instruct", layer=31)
        >>> clf = MultiArchProbe(ProbeConfig(probe_type='mlp'), extractor=ext)
        >>> clf.fit(prompts, y)  # prompts: List[str]
        >>> scores = clf.predict_scores(test_prompts)
    """

    def __init__(
        self,
        config: Optional[ProbeConfig] = None,
        device: Optional[torch.device] = None,
        extractor=None,
    ):
        self.config = config or ProbeConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._extractor = extractor
        self._model: Optional[nn.Module] = None
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False
        self._d_model: Optional[int] = None
        self._running_mean: Optional[torch.Tensor] = None
        self._running_var: Optional[torch.Tensor] = None
        self._n_seen: int = 0

    def _build_model(self, d_model: int) -> nn.Module:
        """Dispatch to the appropriate probe module via registry."""
        probe_type = self.config.probe_type
        if probe_type not in _MODULE_REGISTRY:
            raise ValueError(
                f"Unknown probe_type: {probe_type}. "
                f"Choose from: {list(_MODULE_REGISTRY.keys())}"
            )
        module_cls = _MODULE_REGISTRY[probe_type]
        return module_cls(
            d_model=d_model,
            hidden_dim=self.config.mlp_hidden_dim,
            n_layers=self.config.mlp_layers,
            activation=self.config.mlp_activation,
            n_heads=self.config.n_heads,
            token_position=self.config.token_position,
            ema_alpha=self.config.ema_alpha,
            window_size=self.config.window_size,
            ae_proj_dim=self.config.ae_proj_dim,
        )

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Apply normalization to input features.

        For 3D input (B, T, D), reshape to (B*T, D) for scaler fitting,
        then reshape back.
        """
        if self.config.normalize == 'none':
            # Return as-is — batch-level code handles dtype conversion.
            # This avoids a full-array copy when input is float16.
            return X

        original_shape = X.shape
        is_3d = X.ndim == 3

        if is_3d:
            B, T, D = X.shape
            X_flat = X.reshape(B * T, D)
        else:
            X_flat = X

        if self.config.normalize == 'l2':
            X_flat = normalize(X_flat, norm='l2').astype(np.float32)
        elif self.config.normalize == 'standard':
            if fit:
                self._scaler = StandardScaler()
                X_flat = self._scaler.fit_transform(X_flat).astype(np.float32)
            else:
                X_flat = self._scaler.transform(X_flat).astype(np.float32)
        elif self.config.normalize == 'maxabs':
            if fit:
                self._scaler = MaxAbsScaler()
                X_flat = self._scaler.fit_transform(X_flat).astype(np.float32)
            else:
                X_flat = self._scaler.transform(X_flat).astype(np.float32)

        if is_3d:
            return X_flat.reshape(original_shape).astype(np.float32)
        return X_flat.astype(np.float32)

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Ensure input is 3D (B, T, D). Expand 2D to (B, 1, D)."""
        if X.ndim == 2:
            return X[:, np.newaxis, :]
        return X

    def _init_running_normalization(self) -> None:
        """Initialize running normalization state for provider mode."""
        if self.config.normalize == 'standard' and self._running_mean is None:
            self._running_mean = torch.zeros(
                self._d_model, dtype=self.config.torch_dtype, device=self.device,
            )
            self._running_var = torch.ones(
                self._d_model, dtype=self.config.torch_dtype, device=self.device,
            )
            self._n_seen = 0

    def fit(
        self,
        X=None,
        y: np.ndarray = None,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = True,
        batch_provider: Optional[BatchProvider] = None,
        on_batch_end: Optional[callable] = None,
        on_epoch_end: Optional[callable] = None,
        **kwargs,
    ) -> "MultiArchProbe":
        """
        Train the Gemini probe classifier.

        Args:
            X: Feature matrix (n, D) or (n, T, D) numpy array, OR List[str]
                prompts for online mode (requires extractor).
                Ignored when batch_provider is set.
            y: Labels (n_samples,). Ignored when batch_provider is set.
            sample_weight: Not used (interface compatibility)
            verbose: Print progress
            batch_provider: BatchProvider instance. When provided, X/y are
                ignored — all data comes from the provider.
            **kwargs: Ignored (accepts 'datasets' etc. for interface compat)

        Returns:
            self
        """
        if batch_provider is not None:
            return self._fit_from_provider(
                batch_provider,
                verbose=verbose,
                on_batch_end=on_batch_end,
                on_epoch_end=on_epoch_end,
            )
        if isinstance(X, list):
            return self._fit_online(X, y, verbose=verbose, **kwargs)

        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        torch.cuda.manual_seed(self.config.random_state)

        X = self._prepare_input(X)
        self._d_model = X.shape[2]

        # Normalize (returns same array if normalize='none' and already float32)
        X_norm = self._normalize(X, fit=True)

        # Build model
        self._model = self._build_model(self._d_model).to(self.device)

        # EMA: train with mean pooling
        if self.config.probe_type == 'ema' and isinstance(self._model, EMAModule):
            self._model.set_eval_ema(False)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.adamw_betas,
        )
        if self.config.use_class_weight:
            n_mal = int(y.sum())
            n_ben = len(y) - n_mal
            pos_weight = torch.tensor(
                [n_ben / max(n_mal, 1)], device=self.device, dtype=torch.float32,
            )
        else:
            pos_weight = None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Train/val split
        n_samples = len(X_norm)
        use_val = self.config.val_split > 0
        if use_val:
            n_val = int(n_samples * self.config.val_split)
            perm = np.random.permutation(n_samples)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
        else:
            train_idx = np.arange(n_samples)

        n_train = len(train_idx)
        batch_size = self.config.batch_size if self.config.batch_size > 0 else n_train

        # Prepare y as float32 (small, fine to copy)
        y_train_np = y[train_idx].astype(np.float32)

        # Move training data to GPU as float16 to leverage GPU memory bandwidth
        # (~3.35 TB/s on H100 vs ~200 GB/s CPU DRAM). This eliminates per-batch
        # CPU fancy indexing, float16→float32 conversion, and CPU→GPU transfer.
        # Training fold is ~46GB float16, fits in 95GB H100 with room to spare.
        if use_val:
            X_gpu = torch.from_numpy(np.ascontiguousarray(X_norm[train_idx])).to(self.device)
            X_val_gpu = torch.from_numpy(np.ascontiguousarray(X_norm[val_idx])).to(self.device)
            y_val = y[val_idx].astype(np.float32)
        else:
            X_gpu = torch.from_numpy(np.ascontiguousarray(X_norm)).to(self.device)
        y_gpu = torch.from_numpy(y_train_np).to(self.device)

        if verbose:
            gpu_mb = X_gpu.element_size() * X_gpu.nelement() / 1e6
            print(f"  Training data on GPU: {X_gpu.shape}, {X_gpu.dtype}, {gpu_mb:.0f}MB")

        # Training loop
        best_f1 = 0.0
        best_state = None
        patience_counter = 0
        global_step = 0

        for epoch in range(self.config.max_epochs):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle indices only (tiny: n_train int64 values ≈ 0.7MB)
            perm = torch.randperm(n_train, device=self.device)

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                batch_idx = perm[start:end]
                # All ops on GPU: index → float32 convert → forward.
                # For full-batch, fancy indexing (X_gpu[batch_idx]) creates a
                # full copy even though we want all rows — skip it to halve
                # peak GPU memory (critical for large multi-position probes).
                if end - start == n_train:
                    X_batch = X_gpu.float()  # no fancy-index copy (saves peak GPU mem)
                    y_batch = y_gpu           # full batch — order matches X_gpu
                else:
                    X_batch = X_gpu[batch_idx].float()
                    y_batch = y_gpu[batch_idx]

                optimizer.zero_grad()
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                if self.config.probe_type == 'alphaevolve' and isinstance(self._model, AlphaEvolveModule):
                    loss = loss + self.config.ae_lambda_l1 * self._model.l1_penalty()
                    loss = loss + self.config.ae_lambda_ortho * self._model.ortho_penalty()
                loss.backward()
                optimizer.step()

                global_step += 1
                n_batches += 1
                batch_loss = loss.item()
                epoch_loss += batch_loss
                if on_batch_end is not None:
                    on_batch_end(self, {
                        'epoch': epoch + 1,
                        'batch': n_batches,
                        'global_step': global_step,
                        'loss': batch_loss,
                    })

            if not use_val:
                if on_epoch_end is not None:
                    on_epoch_end(self, epoch + 1, {
                        'epoch': epoch + 1,
                        'avg_loss': epoch_loss / max(n_batches, 1),
                        'n_batches': n_batches,
                        'global_step': global_step,
                    })
                continue

            # Validation for early stopping
            self._model.eval()
            with torch.no_grad():
                if self.config.probe_type == 'ema' and isinstance(self._model, EMAModule):
                    self._model.set_eval_ema(True)

                n_val = len(X_val_gpu)
                val_bs = min(4096, n_val)
                val_logits_list = []
                for i in range(0, n_val, val_bs):
                    xv = X_val_gpu[i:i + val_bs].float()
                    val_logits_list.append(self._model(xv).cpu())
                val_logits = torch.cat(val_logits_list)

                val_preds = (torch.sigmoid(val_logits) > 0.5).numpy().astype(int)
                val_f1 = f1_score(y_val.astype(int), val_preds, average='macro', zero_division=0)

                if self.config.probe_type == 'ema' and isinstance(self._model, EMAModule):
                    self._model.set_eval_ema(False)

            if on_epoch_end is not None:
                on_epoch_end(self, epoch + 1, {
                    'epoch': epoch + 1,
                    'avg_loss': epoch_loss / max(n_batches, 1),
                    'n_batches': n_batches,
                    'global_step': global_step,
                    'val_macro_f1': float(val_f1),
                    'best_val_macro_f1': float(best_f1),
                })

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}, best macro-F1={best_f1:.4f}")
                break

        # Free GPU training data before restoring best model
        del X_gpu, y_gpu
        if use_val:
            del X_val_gpu

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        if verbose and use_val and patience_counter < self.config.early_stopping_patience:
            print(f"Completed {self.config.max_epochs} epochs, best macro-F1={best_f1:.4f}")

        self._fitted = True
        return self

    def _fit_online(
        self,
        prompts: List[str],
        y: np.ndarray,
        verbose: bool = True,
        **kwargs,
    ) -> "MultiArchProbe":
        """Train via on-the-fly activation extraction using train_probe_online."""
        from activation_robustness.probes.training_utils import (
            train_probe_online, OnlineTrainingConfig,
        )

        if self._extractor is None:
            raise ValueError(
                "Online mode requires an ActivationExtractor. "
                "Pass extractor= to the constructor."
            )

        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        torch.cuda.manual_seed(self.config.random_state)

        self._d_model = self._extractor.d_model
        self._model = self._build_model(self._d_model).to(self.device)

        train_config = OnlineTrainingConfig(
            epochs=self.config.max_epochs,
            batch_size=self.config.batch_size if self.config.batch_size > 0 else 4,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            dtype=self.config.dtype,
        )

        train_probe_online(
            module=self._model,
            extractor=self._extractor,
            prompts=prompts,
            labels=y,
            config=train_config,
        )

        self._fitted = True
        return self

    def predict_scores(self, X=None, batch_provider: Optional[BatchProvider] = None) -> np.ndarray:
        """
        Return prediction scores (sigmoid probabilities).

        Args:
            X: Feature matrix (n, D) or (n, T, D) numpy array, OR List[str]
                prompts for online mode. Ignored when batch_provider is set.
            batch_provider: BatchProvider instance. When provided, X is ignored.

        Returns:
            Scores array (n_samples,) in [0, 1]
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        if batch_provider is not None:
            return self._predict_from_provider(batch_provider)
        if isinstance(X, list):
            return self._predict_online(X)

        X = self._prepare_input(X)
        X_norm = self._normalize(X, fit=False)

        self._model.eval()
        if self.config.probe_type == 'ema' and isinstance(self._model, EMAModule):
            self._model.set_eval_ema(True)

        with torch.no_grad():
            X_gpu = torch.from_numpy(np.ascontiguousarray(X_norm)).to(self.device)
            batch_size = 4096
            all_scores = []
            for i in range(0, len(X_gpu), batch_size):
                X_batch = X_gpu[i:i + batch_size].float()
                logits = self._model(X_batch)
                all_scores.append(torch.sigmoid(logits).cpu().numpy())
            scores = np.concatenate(all_scores)
            del X_gpu

        return scores

    def _predict_online(self, prompts: List[str]) -> np.ndarray:
        """Predict via on-the-fly extraction."""
        if self._extractor is None:
            raise ValueError("Online prediction requires an ActivationExtractor.")

        self._model.eval()
        if self.config.probe_type == 'ema' and isinstance(self._model, EMAModule):
            self._model.set_eval_ema(True)

        batch_size = self.config.batch_size if self.config.batch_size > 0 else 4
        all_scores = []

        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start: start + batch_size]
                hidden, _ = self._extractor.extract_all_positions(batch)
                logits = self._model(hidden.float())
                all_scores.append(torch.sigmoid(logits).cpu().numpy())
                del hidden

        return np.concatenate(all_scores)

    # ----- Batch provider mode (cache or live, probe doesn't care) -----

    def _fit_from_provider(
        self,
        batch_provider: BatchProvider,
        verbose: bool = True,
        on_batch_end: Optional[callable] = None,
        on_epoch_end: Optional[callable] = None,
    ) -> "MultiArchProbe":
        """Train via a BatchProvider (cached or live activations).

        Uses the same training loop as the numpy path but reads batches from
        the provider instead of an in-memory array. Supports early stopping
        via a held-out validation split of the provider indices.
        """
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        torch.cuda.manual_seed(self.config.random_state)

        self._d_model = batch_provider.d_model
        self._init_running_normalization()
        self._model = self._build_model(self._d_model).to(device=self.device, dtype=self.config.torch_dtype)

        if self.config.probe_type == 'ema' and isinstance(self._model, EMAModule):
            self._model.set_eval_ema(False)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.adamw_betas,
        )

        # Class weighting: pos_weight = n_ben / n_mal to counter class imbalance
        if self.config.use_class_weight:
            n_mal = int(batch_provider.labels.sum())
            n_ben = len(batch_provider.labels) - n_mal
            pos_weight = torch.tensor([n_ben / max(n_mal, 1)], device=self.device, dtype=self.config.torch_dtype)
        else:
            pos_weight = None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        is_alphaevolve = self.config.probe_type == 'alphaevolve' and isinstance(
            self._model, AlphaEvolveModule,
        )

        n = batch_provider.n_samples
        batch_size = self.config.batch_size if self.config.batch_size > 0 else n
        total_batches_est = (n + batch_size - 1) // batch_size
        log_every = max(1, total_batches_est // 5)  # ~5 logs per epoch
        global_step = 0

        for epoch in range(self.config.max_epochs):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0
            epoch_start = time.time()

            for hidden, mask, batch_labels, _spans in batch_provider.iter_train_batches(
                batch_size,
                epoch_seed=self.config.random_state + epoch,
                device=self.device,
                return_offsets=False,
            ):
                with torch.no_grad():
                    if epoch == 0 and self.config.normalize == 'standard':
                        valid = hidden[mask]
                        if valid.shape[0] > 0:
                            self._running_mean, self._running_var, self._n_seen = welford_update(
                                self._running_mean, self._running_var, self._n_seen, valid,
                            )

                    if self.config.normalize == 'standard':
                        X_batch = (hidden - self._running_mean) / (
                            self._running_var.sqrt() + 1e-5
                        )
                    elif self.config.normalize == 'l2':
                        X_batch = F.normalize(hidden.float(), p=2, dim=-1, eps=1e-12).to(hidden.dtype)
                    elif self.config.normalize == 'none':
                        X_batch = hidden
                    else:
                        raise NotImplementedError(
                            "Provider-mode Gemini normalization currently supports "
                            "'none', 'standard', and 'l2'."
                        )
                y_batch = batch_labels.to(device=self.device, dtype=self.config.torch_dtype)

                optimizer.zero_grad()
                logits = self._model(X_batch, mask)
                loss = criterion(logits, y_batch)
                if is_alphaevolve:
                    loss = loss + self.config.ae_lambda_l1 * self._model.l1_penalty()
                    loss = loss + self.config.ae_lambda_ortho * self._model.ortho_penalty()
                loss.backward()
                optimizer.step()

                global_step += 1
                epoch_loss += loss.item()
                n_batches += 1

                if on_batch_end is not None:
                    on_batch_end(self, {
                        'epoch': epoch + 1,
                        'batch': n_batches,
                        'global_step': global_step,
                        'loss': loss.item(),
                    })

                if verbose and n_batches % log_every == 0:
                    elapsed = time.time() - epoch_start
                    it_s = n_batches / elapsed if elapsed > 0 else 0
                    avg_loss = epoch_loss / n_batches
                    print(f"  Epoch {epoch+1} [{n_batches}/{total_batches_est}] "
                          f"loss={avg_loss:.4f} {it_s:.1f} it/s", flush=True)

            epoch_elapsed = time.time() - epoch_start
            if verbose and ((epoch + 1) % 100 == 0 or self.config.max_epochs <= 100):
                print(f"  Epoch {epoch+1}: loss={epoch_loss / max(n_batches, 1):.4f} "
                      f"({epoch_elapsed:.1f}s, {n_batches/epoch_elapsed:.1f} it/s)", flush=True)
            if on_epoch_end is not None:
                on_epoch_end(self, epoch + 1, {
                    'epoch': epoch + 1,
                    'avg_loss': epoch_loss / max(n_batches, 1),
                    'n_batches': n_batches,
                    'epoch_time_s': epoch_elapsed,
                    'global_step': global_step,
                })

        self._fitted = True
        return self

    def _predict_from_provider(self, batch_provider: BatchProvider) -> np.ndarray:
        """Predict scores from a BatchProvider."""
        self._model.eval()
        if self.config.probe_type == 'ema' and isinstance(self._model, EMAModule):
            self._model.set_eval_ema(True)

        batch_size = self.config.batch_size if self.config.batch_size > 0 else 4096
        all_scores = []

        with torch.no_grad():
            for hidden, mask in batch_provider.iter_predict_batches(
                batch_size, device=self.device,
            ):
                if self.config.normalize == 'standard' and self._running_mean is not None:
                    hidden = (hidden - self._running_mean) / (
                        self._running_var.sqrt() + 1e-5
                    )
                elif self.config.normalize == 'l2':
                    hidden = F.normalize(hidden.float(), p=2, dim=-1, eps=1e-12).to(hidden.dtype)
                elif self.config.normalize != 'none':
                    raise NotImplementedError(
                        "Provider-mode Gemini normalization currently supports "
                        "'none', 'standard', and 'l2'."
                    )
                logits = self._model(hidden, mask)  # bf16 in, bf16 out
                all_scores.append(torch.sigmoid(logits).float().cpu().numpy())
                del hidden

        return np.concatenate(all_scores)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        scores = self.predict_scores(X)
        return (scores >= threshold).astype(int)

    def evaluate(self, batch_provider: BatchProvider) -> dict:
        """Evaluate on a test set via the unified eval interface.

        Returns a dict with:
            aggregations: {"score_t{thresh}": {recall, fpr, ...}} for 3 thresholds
            token_scores: None (sequence-level classifier)
            seq_scores: {"score": (n,) array}
        """
        from activation_robustness.data.eval_utils import OnlineEvaluator
        scores = self.predict_scores(batch_provider=batch_provider)
        labels = batch_provider.labels
        return OnlineEvaluator().evaluate_sequence_scores(scores, labels)

    def clone(self) -> "MultiArchProbe":
        """Create unfitted copy with same configuration."""
        return MultiArchProbe(
            config=copy.deepcopy(self.config),
            device=self.device,
            extractor=self._extractor,
        )

    def get_params(self) -> Dict[str, Any]:
        """Return configuration as dict for reproducibility."""
        return asdict(self.config)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"MultiArchProbe(probe_type={self.config.probe_type}, "
            f"normalize={self.config.normalize}, {status})"
        )
