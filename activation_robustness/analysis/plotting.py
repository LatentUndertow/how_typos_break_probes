"""Standardized visualization for activation robustness analysis.

All plot functions optionally save to a file and return the matplotlib Figure
for further customization. Uses a consistent style across all plots.

Dependencies: matplotlib, numpy.
"""

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np


# -- Shared style defaults --------------------------------------------------

_STYLE = {
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
}


def _apply_style():
    """Apply project-wide matplotlib style."""
    plt.rcParams.update(_STYLE)


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
    """Save figure to file if path provided, otherwise just return it."""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# -- Plot functions ----------------------------------------------------------


def plot_eigenspectrum(
    eigenvalues: np.ndarray,
    mp_edge: float,
    null_eigenvalues: Optional[np.ndarray] = None,
    title: str = "Noise Covariance Eigenspectrum",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Eigenvalue decay plot with Marchenko-Pastur edge overlay.

    Args:
        eigenvalues: 1-D array of eigenvalues in descending order.
        mp_edge: Marchenko-Pastur upper edge value (horizontal line).
        null_eigenvalues: Optional 1-D array of null-model eigenvalues
            (e.g., from background_null) for comparison.
        title: Plot title.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    _apply_style()
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

    fig, ax = plt.subplots()

    ax.semilogy(
        np.arange(1, len(eigenvalues) + 1),
        eigenvalues,
        "o-",
        markersize=3,
        linewidth=1.2,
        label="Observed",
        color="#2563eb",
    )

    ax.axhline(
        mp_edge,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label=f"MP edge = {mp_edge:.3f}",
    )

    if null_eigenvalues is not None:
        null_eigenvalues = np.asarray(null_eigenvalues, dtype=np.float64)
        ax.semilogy(
            np.arange(1, len(null_eigenvalues) + 1),
            null_eigenvalues,
            "s-",
            markersize=2,
            linewidth=1.0,
            alpha=0.7,
            label="Background null (mean)",
            color="#9333ea",
        )

    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_cumulative_variance(
    eigenvalues: np.ndarray,
    title: str = "Cumulative Variance Explained",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Cumulative variance explained curve.

    Args:
        eigenvalues: 1-D array of eigenvalues in descending order.
        title: Plot title.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    _apply_style()
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

    total = eigenvalues.sum()
    if total > 0:
        cumvar = np.cumsum(eigenvalues) / total
    else:
        cumvar = np.zeros_like(eigenvalues)

    fig, ax = plt.subplots()

    ranks = np.arange(1, len(cumvar) + 1)
    ax.plot(ranks, cumvar, "-", linewidth=1.5, color="#2563eb")
    ax.fill_between(ranks, cumvar, alpha=0.1, color="#2563eb")

    # Mark key thresholds
    for thresh in [0.5, 0.8, 0.9, 0.95]:
        idx = np.searchsorted(cumvar, thresh)
        if idx < len(cumvar):
            ax.axhline(thresh, color="#6b7280", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.axvline(idx + 1, color="#6b7280", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.annotate(
                f"{thresh:.0%} at k={idx + 1}",
                (idx + 1, thresh),
                textcoords="offset points",
                xytext=(8, -5),
                fontsize=9,
                color="#374151",
            )

    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_radial_tangential(
    radial_fracs: np.ndarray,
    tangential_fracs: np.ndarray,
    title: str = "Radial vs. Tangential Noise Fractions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of radial vs tangential fractions across prompts.

    Args:
        radial_fracs: 1-D array (n,) -- radial fraction per prompt.
        tangential_fracs: 1-D array (n,) -- tangential fraction per prompt.
        title: Plot title.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    _apply_style()
    radial_fracs = np.asarray(radial_fracs, dtype=np.float64)
    tangential_fracs = np.asarray(tangential_fracs, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(
        radial_fracs, bins=30, color="#f59e0b", alpha=0.8, edgecolor="white"
    )
    axes[0].set_xlabel("Radial fraction (||delta_r|| / ||delta||)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Radial (norm-changing)")
    axes[0].axvline(
        radial_fracs.mean(),
        color="#b45309",
        linestyle="--",
        label=f"mean = {radial_fracs.mean():.4f}",
    )
    axes[0].legend()

    axes[1].hist(
        tangential_fracs, bins=30, color="#3b82f6", alpha=0.8, edgecolor="white"
    )
    axes[1].set_xlabel("Tangential fraction (||delta_t|| / ||delta||)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Tangential (direction-changing)")
    axes[1].axvline(
        tangential_fracs.mean(),
        color="#1d4ed8",
        linestyle="--",
        label=f"mean = {tangential_fracs.mean():.4f}",
    )
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_layerwise_heatmap(
    stats_by_layer: dict[int, dict],
    metric_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of a scalar statistic across layers.

    If the metric is a scalar per layer, plots a 1-D bar chart.
    Designed for use with the output of statistics.layerwise_summary().

    Args:
        stats_by_layer: Dict mapping layer index to stats dict.
        metric_name: Key to extract from each layer's stats dict.
            Must map to a scalar value.
        title: Plot title. Defaults to the metric name.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    layers = sorted(stats_by_layer.keys())
    values = []
    for layer in layers:
        val = stats_by_layer[layer].get(metric_name)
        if val is None:
            raise KeyError(f"Metric '{metric_name}' not found in layer {layer} stats")
        values.append(float(val))

    values = np.array(values)

    if title is None:
        title = metric_name.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.4), 5))

    bars = ax.bar(
        range(len(layers)),
        values,
        color="#2563eb",
        alpha=0.8,
        edgecolor="white",
    )
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers], rotation=45 if len(layers) > 20 else 0)
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric_name)
    ax.set_title(title)

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_background_null_comparison(
    delta_eigenvalues: np.ndarray,
    null_eigenvalues: np.ndarray,
    null_q95: Optional[np.ndarray] = None,
    title: str = "Noise vs. Background Null Eigenspectrum",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side eigenspectrum comparison: observed noise vs. background null.

    Args:
        delta_eigenvalues: 1-D array -- eigenvalues of the observed noise
            covariance (descending).
        null_eigenvalues: 1-D array -- mean eigenvalues from the background
            null model (descending).
        null_q95: Optional 1-D array -- 95th percentile of null eigenvalues.
            If provided, shown as a shaded upper bound.
        title: Plot title.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    _apply_style()
    delta_eigenvalues = np.asarray(delta_eigenvalues, dtype=np.float64)
    null_eigenvalues = np.asarray(null_eigenvalues, dtype=np.float64)

    fig, ax = plt.subplots()

    max_rank = max(len(delta_eigenvalues), len(null_eigenvalues))

    ranks_d = np.arange(1, len(delta_eigenvalues) + 1)
    ax.semilogy(
        ranks_d,
        delta_eigenvalues,
        "o-",
        markersize=3,
        linewidth=1.2,
        label="Observed noise",
        color="#2563eb",
    )

    ranks_n = np.arange(1, len(null_eigenvalues) + 1)
    ax.semilogy(
        ranks_n,
        np.maximum(null_eigenvalues, 1e-15),  # avoid log(0)
        "s-",
        markersize=2,
        linewidth=1.0,
        label="Background null (mean)",
        color="#9333ea",
        alpha=0.8,
    )

    if null_q95 is not None:
        null_q95 = np.asarray(null_q95, dtype=np.float64)
        ax.fill_between(
            ranks_n[:len(null_q95)],
            np.maximum(null_eigenvalues[:len(null_q95)], 1e-15),
            np.maximum(null_q95, 1e-15),
            alpha=0.15,
            color="#9333ea",
            label="Null 95th percentile",
        )

    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return _save_or_show(fig, save_path)


# -- Sequence-level plots ----------------------------------------------------


def plot_spatial_delta_profile(
    profile: dict,
    title: str = "Spatial Delta Profile",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Delta norm across token positions with std band.

    Args:
        profile: Output of sequence.spatial_delta_profile().
        title: Plot title.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    norms = np.asarray(profile["position_norms"], dtype=np.float64)
    stds = np.asarray(profile["position_norms_std"], dtype=np.float64)
    positions = np.arange(len(norms))

    fig, ax = plt.subplots()

    ax.plot(positions, norms, "-", linewidth=1.5, color="#2563eb", label="Mean norm")
    ax.fill_between(
        positions,
        norms - stds,
        norms + stds,
        alpha=0.15,
        color="#2563eb",
        label="\u00b11 std",
    )

    peak = profile["peak_position"]
    ax.axvline(
        peak,
        color="#dc2626",
        linestyle="--",
        linewidth=1.0,
        label=f"Peak at pos {peak}",
    )

    ax.set_xlabel("Token position")
    ax.set_ylabel("Delta L2 norm")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_cross_position_heatmap(
    covariance: dict,
    title: str = "Cross-Position Correlation",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of cross-position correlation matrix.

    Args:
        covariance: Output of sequence.cross_position_covariance().
        title: Plot title.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    corr = np.asarray(covariance["correlation_matrix"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, label="Correlation")

    ax.set_xlabel("Token position")
    ax.set_ylabel("Token position")
    ax.set_title(
        f"{title}  (mean off-diag = {covariance['mean_off_diagonal']:.3f})"
    )

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_autocorrelation(
    autocorr: dict,
    title: str = "Delta Norm Autocorrelation",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Autocorrelation function with half-life marker.

    Args:
        autocorr: Output of sequence.autocorrelation().
        title: Plot title.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    lags = np.asarray(autocorr["lags"], dtype=np.float64)
    ac = np.asarray(autocorr["autocorr"], dtype=np.float64)
    half_life = autocorr["half_life"]

    fig, ax = plt.subplots()

    ax.plot(lags, ac, "o-", markersize=3, linewidth=1.2, color="#2563eb")
    ax.axhline(0, color="#6b7280", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.axhline(0.5, color="#9ca3af", linestyle=":", linewidth=0.8, alpha=0.5)

    if half_life is not None:
        ax.axvline(
            half_life,
            color="#dc2626",
            linestyle="--",
            linewidth=1.2,
            label=f"Half-life = {half_life}",
        )
        ax.legend()

    ax.set_xlabel("Lag (positions)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_decay_profile(
    decay: dict,
    title: str = "Delta Decay from Perturbation Site",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Decay curve with fitted model overlay.

    Args:
        decay: Output of sequence.decay_profile().
        title: Plot title.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    distances = np.asarray(decay["distances"], dtype=np.float64)
    mean_norms = np.asarray(decay["mean_norms"], dtype=np.float64)
    decay_type = decay["decay_type"]
    decay_rate = decay["decay_rate"]
    r_squared = decay["r_squared"]

    # Sort by distance for clean plotting
    sort_idx = np.argsort(distances)
    d_sorted = distances[sort_idx]
    n_sorted = mean_norms[sort_idx]

    fig, ax = plt.subplots()

    ax.plot(
        d_sorted, n_sorted, "o", markersize=4, color="#2563eb",
        alpha=0.7, label="Observed",
    )

    # Overlay fitted model
    if decay_type != "flat" and decay_rate > 0:
        d_fit = d_sorted[d_sorted > 0]
        amp = n_sorted[0]

        if decay_type == "exponential":
            fitted = amp * np.exp(-decay_rate * d_fit)
            label = f"Exp fit (rate={decay_rate:.3f}, R2={r_squared:.3f})"
        elif decay_type == "power_law":
            fitted = amp * np.power(d_fit, -decay_rate)
            label = f"Power law (exp={decay_rate:.3f}, R2={r_squared:.3f})"
        else:
            fitted = None
            label = None

        if fitted is not None:
            ax.plot(d_fit, fitted, "--", linewidth=1.5, color="#dc2626", label=label)

    ax.set_xlabel("Distance from perturbation site")
    ax.set_ylabel("Mean delta norm")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_pooling_reduction(
    pooling: dict,
    title: str = "Pooling Noise Reduction",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Noise reduction vs k, compared to sqrt(k) theoretical.

    Args:
        pooling: Output of sequence.pooling_noise_reduction().
        title: Plot title.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    k_vals = np.asarray(pooling["k_values"], dtype=np.float64)
    reduction = np.asarray(pooling["reduction_factors"], dtype=np.float64)
    theoretical = np.asarray(pooling["theoretical_sqrt_k"], dtype=np.float64)

    fig, ax = plt.subplots()

    ax.plot(
        k_vals, reduction, "o-", markersize=5, linewidth=1.5,
        color="#2563eb", label="Observed reduction",
    )
    ax.plot(
        k_vals, theoretical, "s--", markersize=4, linewidth=1.2,
        color="#9333ea", alpha=0.7, label="Theoretical sqrt(k)",
    )

    ax.set_xlabel("k (number of pooled positions)")
    ax.set_ylabel("Noise reduction factor")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.legend()

    fig.tight_layout()
    return _save_or_show(fig, save_path)
