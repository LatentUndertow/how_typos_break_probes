"""Sequence-level signal analysis for activation deltas across token positions.

When we perturb the last token (e.g., ? -> .), the effect propagates through
attention to all positions. This module characterizes that spatial pattern.

All functions operate on float64 numpy arrays and return plain dicts.
Input arrays have shape (n_prompts, seq_len, d_model).
"""

import numpy as np
from scipy.optimize import curve_fit


def _ensure_3d_float64(arr: np.ndarray) -> np.ndarray:
    """Validate and cast input to float64 3-D array."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3-D array (n_prompts, seq_len, d_model), got shape {arr.shape}"
        )
    return arr


def spatial_delta_profile(
    deltas_by_position: np.ndarray,
    k_tail: int = 4,
) -> dict:
    """Compute delta norm at each token position.

    Args:
        deltas_by_position: Array of shape (n_prompts, seq_len, d_model).
        k_tail: Number of trailing positions used for concentration_ratio.
            Default 4.

    Returns:
        Dict with keys:
            position_norms: (seq_len,) mean delta norm per position.
            position_norms_std: (seq_len,) std of delta norms across prompts.
            position_cosines: (seq_len,) mean cosine similarity of each
                position's delta to the mean delta direction at that position.
            peak_position: int -- position with the largest mean delta norm.
            concentration_ratio: float -- fraction of total mean-norm mass
                concentrated in the last k_tail positions.
    """
    deltas = _ensure_3d_float64(deltas_by_position)
    n_prompts, seq_len, d_model = deltas.shape

    # Per-sample, per-position norms: (n_prompts, seq_len)
    norms = np.linalg.norm(deltas, axis=2)

    position_norms = norms.mean(axis=0)       # (seq_len,)
    position_norms_std = norms.std(axis=0)    # (seq_len,)

    # Cosine similarity to position-mean direction
    mean_delta = deltas.mean(axis=0)  # (seq_len, d_model)
    mean_delta_norm = np.linalg.norm(mean_delta, axis=1, keepdims=True)  # (seq_len, 1)
    safe_mean_norm = np.maximum(mean_delta_norm, 1e-12)
    mean_hat = mean_delta / safe_mean_norm  # (seq_len, d_model)

    # Per-sample cosine to mean direction at each position
    sample_norms = np.maximum(norms, 1e-12)  # (n_prompts, seq_len)
    # dot product: (n_prompts, seq_len)
    dots = np.sum(deltas * mean_hat[np.newaxis, :, :], axis=2)
    cosines = dots / sample_norms  # (n_prompts, seq_len)
    position_cosines = cosines.mean(axis=0)  # (seq_len,)

    peak_position = int(np.argmax(position_norms))

    total_norm = position_norms.sum()
    if total_norm > 0:
        tail_norm = position_norms[-k_tail:].sum()
        concentration_ratio = float(tail_norm / total_norm)
    else:
        concentration_ratio = 0.0

    return {
        "position_norms": position_norms,
        "position_norms_std": position_norms_std,
        "position_cosines": position_cosines,
        "peak_position": peak_position,
        "concentration_ratio": concentration_ratio,
    }


def cross_position_covariance(
    deltas_by_position: np.ndarray,
    method: str = "scalar",
) -> dict:
    """Correlation of deltas between pairs of token positions.

    Args:
        deltas_by_position: Array of shape (n_prompts, seq_len, d_model).
        method: "scalar" projects deltas onto position-mean direction first,
                then computes Pearson correlation of the scalar projections.
                "cosine" computes mean cosine similarity between delta vectors
                at position i and position j across prompts.

    Returns:
        Dict with keys:
            correlation_matrix: (seq_len, seq_len) cross-position correlation.
            mean_off_diagonal: float -- mean of off-diagonal entries.
    """
    deltas = _ensure_3d_float64(deltas_by_position)
    n_prompts, seq_len, d_model = deltas.shape

    if method == "scalar":
        # Project each position's deltas onto position-mean direction
        mean_delta = deltas.mean(axis=0)  # (seq_len, d_model)
        mean_norms = np.linalg.norm(mean_delta, axis=1, keepdims=True)
        safe_norms = np.maximum(mean_norms, 1e-12)
        mean_hat = mean_delta / safe_norms  # (seq_len, d_model)

        # Scalar projections: (n_prompts, seq_len)
        projections = np.sum(deltas * mean_hat[np.newaxis, :, :], axis=2)

        # Pearson correlation across positions
        # Center each position
        proj_centered = projections - projections.mean(axis=0, keepdims=True)
        stds = proj_centered.std(axis=0, keepdims=True)
        safe_stds = np.maximum(stds, 1e-12)
        proj_normed = proj_centered / safe_stds  # (n_prompts, seq_len)

        corr = proj_normed.T @ proj_normed / n_prompts  # (seq_len, seq_len)

    elif method == "cosine":
        corr = np.zeros((seq_len, seq_len), dtype=np.float64)
        # Normalize per-sample per-position
        norms = np.linalg.norm(deltas, axis=2, keepdims=True)
        safe_norms = np.maximum(norms, 1e-12)
        normed = deltas / safe_norms  # (n_prompts, seq_len, d_model)

        # Mean cosine between all position pairs
        # normed[:, i, :] . normed[:, j, :] averaged over prompts
        # Efficient: (n_prompts, seq_len, d_model) -> batch matmul
        # Reshape to (n_prompts, seq_len, d_model) and compute batch outer
        for i in range(seq_len):
            # (n_prompts, d_model) @ (n_prompts, d_model, 1) broadcasted
            cos_ij = np.sum(
                normed[:, i, :, np.newaxis] * normed.transpose(0, 2, 1),
                axis=1,
            )  # (n_prompts, seq_len)
            corr[i, :] = cos_ij.mean(axis=0)
    else:
        raise ValueError(f"method must be 'scalar' or 'cosine', got '{method}'")

    # Mean off-diagonal
    mask = ~np.eye(seq_len, dtype=bool)
    mean_off_diag = float(corr[mask].mean()) if seq_len > 1 else 0.0

    return {
        "correlation_matrix": corr,
        "mean_off_diagonal": mean_off_diag,
    }


def autocorrelation(
    deltas_by_position: np.ndarray,
    max_lag: int = None,
) -> dict:
    """Autocorrelation of delta norms across token positions.

    Measures whether noise magnitude at position t predicts noise magnitude
    at position t+lag.

    Args:
        deltas_by_position: Array of shape (n_prompts, seq_len, d_model).
        max_lag: Maximum lag to compute. Default: seq_len // 2.

    Returns:
        Dict with keys:
            lags: 1-D array of lag values (0, 1, ..., max_lag).
            autocorr: 1-D array of autocorrelation at each lag.
            half_life: int or None -- lag at which autocorrelation first
                drops below 0.5. None if it never drops below 0.5.
    """
    deltas = _ensure_3d_float64(deltas_by_position)
    n_prompts, seq_len, d_model = deltas.shape

    if max_lag is None:
        max_lag = seq_len // 2

    max_lag = min(max_lag, seq_len - 1)

    # Delta norms per sample per position: (n_prompts, seq_len)
    norms = np.linalg.norm(deltas, axis=2)

    # Center each prompt's norm sequence
    norms_centered = norms - norms.mean(axis=1, keepdims=True)

    # Variance per prompt
    var_per_prompt = np.sum(norms_centered ** 2, axis=1)  # (n_prompts,)
    safe_var = np.maximum(var_per_prompt, 1e-12)

    lags = np.arange(0, max_lag + 1, dtype=np.int64)
    autocorr = np.zeros(len(lags), dtype=np.float64)

    for idx, lag in enumerate(lags):
        if lag == 0:
            autocorr[idx] = 1.0
        else:
            # Cross-product at this lag, averaged over prompts
            cross = np.sum(
                norms_centered[:, :seq_len - lag] * norms_centered[:, lag:],
                axis=1,
            )  # (n_prompts,)
            autocorr[idx] = float(np.mean(cross / safe_var))

    # Half-life: first lag where autocorr drops below 0.5
    half_life = None
    below_mask = autocorr[1:] < 0.5  # skip lag=0
    if np.any(below_mask):
        half_life = int(lags[1:][below_mask][0])

    return {
        "lags": lags,
        "autocorr": autocorr,
        "half_life": half_life,
    }


def decay_profile(
    deltas_by_position: np.ndarray,
    perturbation_position: int = -1,
) -> dict:
    """Characterize how delta magnitude decays with distance from perturbation site.

    Fits exponential and power-law models and picks the best fit.

    Args:
        deltas_by_position: Array of shape (n_prompts, seq_len, d_model).
        perturbation_position: Which position was perturbed. Default -1
            (last position).

    Returns:
        Dict with keys:
            distances: 1-D array -- distance from perturbation site per position.
            mean_norms: 1-D array -- mean delta norm at each distance.
            decay_type: str -- "exponential", "power_law", or "flat".
            decay_rate: float -- fitted decay parameter (rate for exponential,
                exponent for power law).
            r_squared: float -- goodness of fit for the selected model.
    """
    deltas = _ensure_3d_float64(deltas_by_position)
    n_prompts, seq_len, d_model = deltas.shape

    # Resolve negative index
    if perturbation_position < 0:
        perturbation_position = seq_len + perturbation_position

    # Distance from perturbation site for each position
    positions = np.arange(seq_len, dtype=np.float64)
    distances = np.abs(positions - perturbation_position)

    # Mean delta norm at each position (already ordered by position)
    norms = np.linalg.norm(deltas, axis=2)  # (n_prompts, seq_len)
    mean_norms = norms.mean(axis=0)  # (seq_len,)

    # Sort by distance for fitting
    sort_idx = np.argsort(distances)
    dist_sorted = distances[sort_idx]
    norm_sorted = mean_norms[sort_idx]

    # Amplitude at perturbation site
    amp0 = norm_sorted[0] if norm_sorted[0] > 1e-12 else 1.0

    # -- Fit candidates --
    # Only fit on points with distance > 0
    fit_mask = dist_sorted > 0
    d_fit = dist_sorted[fit_mask]
    n_fit = norm_sorted[fit_mask]

    ss_total = np.sum((n_fit - n_fit.mean()) ** 2)
    if ss_total < 1e-20 or len(d_fit) < 2:
        # All norms are essentially equal -> flat
        return {
            "distances": distances,
            "mean_norms": mean_norms,
            "decay_type": "flat",
            "decay_rate": 0.0,
            "r_squared": 1.0 if ss_total < 1e-20 else 0.0,
        }

    results = {}

    # Exponential: a * exp(-rate * d)
    def _exp_model(d, a, rate):
        return a * np.exp(-rate * d)

    try:
        popt, _ = curve_fit(
            _exp_model, d_fit, n_fit,
            p0=[amp0, 0.1],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=5000,
        )
        pred = _exp_model(d_fit, *popt)
        ss_res = np.sum((n_fit - pred) ** 2)
        r2 = 1.0 - ss_res / ss_total
        results["exponential"] = {"rate": float(popt[1]), "r_squared": float(r2)}
    except (RuntimeError, ValueError):
        results["exponential"] = {"rate": 0.0, "r_squared": -np.inf}

    # Power law: a * d^(-alpha)
    def _pow_model(d, a, alpha):
        return a * np.power(d, -alpha)

    try:
        popt, _ = curve_fit(
            _pow_model, d_fit, n_fit,
            p0=[amp0, 1.0],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=5000,
        )
        pred = _pow_model(d_fit, *popt)
        ss_res = np.sum((n_fit - pred) ** 2)
        r2 = 1.0 - ss_res / ss_total
        results["power_law"] = {"rate": float(popt[1]), "r_squared": float(r2)}
    except (RuntimeError, ValueError):
        results["power_law"] = {"rate": 0.0, "r_squared": -np.inf}

    # Pick the best fit (or flat if both are poor)
    best_type = max(results, key=lambda k: results[k]["r_squared"])
    best = results[best_type]

    if best["r_squared"] < 0.1:
        best_type = "flat"
        best = {"rate": 0.0, "r_squared": 0.0}

    return {
        "distances": distances,
        "mean_norms": mean_norms,
        "decay_type": best_type,
        "decay_rate": best["rate"],
        "r_squared": best["r_squared"],
    }


def pooling_noise_reduction(
    deltas_by_position: np.ndarray,
    k_values: list = None,
) -> dict:
    """Measure noise reduction from averaging over k contiguous positions.

    For each k, average the last k positions' deltas and measure the
    resulting noise norm. Compare to theoretical sqrt(k) reduction for
    independent noise.

    Args:
        deltas_by_position: Array of shape (n_prompts, seq_len, d_model).
        k_values: List of k values to test. Default [1, 2, 4, 8, 16, 32].
            Values larger than seq_len are silently skipped.

    Returns:
        Dict with keys:
            k_values: 1-D array of tested k values.
            noise_norms: 1-D array -- mean noise norm after pooling k positions.
            reduction_factors: 1-D array -- noise_norm[k=1] / noise_norm[k].
            theoretical_sqrt_k: 1-D array -- sqrt(k) for comparison.
            cosine_to_unpooled: 1-D array -- mean cosine similarity between
                the pooled delta and the k=1 (last position only) delta.
    """
    deltas = _ensure_3d_float64(deltas_by_position)
    n_prompts, seq_len, d_model = deltas.shape

    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32]

    # Filter to valid k values
    k_values = [k for k in k_values if k <= seq_len and k >= 1]

    k_arr = np.array(k_values, dtype=np.int64)
    noise_norms = np.zeros(len(k_values), dtype=np.float64)
    cosine_to_unpooled = np.zeros(len(k_values), dtype=np.float64)

    # Reference: k=1 (last position only)
    ref_delta = deltas[:, -1, :]  # (n_prompts, d_model)
    ref_norms = np.linalg.norm(ref_delta, axis=1, keepdims=True)
    safe_ref_norms = np.maximum(ref_norms, 1e-12)

    for idx, k in enumerate(k_values):
        # Average the last k positions
        pooled = deltas[:, -k:, :].mean(axis=1)  # (n_prompts, d_model)
        pool_norms = np.linalg.norm(pooled, axis=1)  # (n_prompts,)
        noise_norms[idx] = float(pool_norms.mean())

        # Cosine to unpooled (k=1) signal
        safe_pool_norms = np.maximum(pool_norms[:, np.newaxis], 1e-12)
        cos_vals = np.sum(pooled * ref_delta, axis=1, keepdims=True) / (
            safe_pool_norms * safe_ref_norms
        )
        cosine_to_unpooled[idx] = float(cos_vals.mean())

    # Reduction factors relative to k=1
    norm_k1 = noise_norms[0] if k_values[0] == 1 else float(
        np.linalg.norm(ref_delta, axis=1).mean()
    )
    safe_norms = np.maximum(noise_norms, 1e-12)
    reduction_factors = norm_k1 / safe_norms

    theoretical_sqrt_k = np.sqrt(k_arr.astype(np.float64))

    return {
        "k_values": k_arr,
        "noise_norms": noise_norms,
        "reduction_factors": reduction_factors,
        "theoretical_sqrt_k": theoretical_sqrt_k,
        "cosine_to_unpooled": cosine_to_unpooled,
    }


def spectral_energy_by_position(
    deltas_by_position: np.ndarray,
    n_components: int = 10,
) -> dict:
    """PCA of deltas at each position, track how eigenspectrum changes along sequence.

    At each token position, computes the covariance of deltas across prompts,
    extracts the top eigenvalues, and computes the effective rank.

    Args:
        deltas_by_position: Array of shape (n_prompts, seq_len, d_model).
        n_components: Number of top eigenvalues to track per position.

    Returns:
        Dict with keys:
            positions: (seq_len,) position indices.
            top_eigenvalues: (seq_len, n_components) top eigenvalues at
                each position. Padded with zeros if fewer than n_components
                positive eigenvalues exist.
            effective_ranks: (seq_len,) effective rank at each position.
            total_variance: (seq_len,) total noise variance at each position.
    """
    deltas = _ensure_3d_float64(deltas_by_position)
    n_prompts, seq_len, d_model = deltas.shape

    positions = np.arange(seq_len, dtype=np.int64)
    top_eigenvalues = np.zeros((seq_len, n_components), dtype=np.float64)
    effective_ranks = np.zeros(seq_len, dtype=np.float64)
    total_variance = np.zeros(seq_len, dtype=np.float64)

    for pos in range(seq_len):
        # Deltas at this position across prompts: (n_prompts, d_model)
        pos_deltas = deltas[:, pos, :]

        # Center
        pos_deltas = pos_deltas - pos_deltas.mean(axis=0, keepdims=True)

        n, d = pos_deltas.shape

        if n <= d:
            gram = pos_deltas @ pos_deltas.T / n
            eigvals = np.linalg.eigvalsh(gram)
        else:
            cov = pos_deltas.T @ pos_deltas / n
            eigvals = np.linalg.eigvalsh(cov)

        # Descending order, keep positive
        eigvals = eigvals[::-1]
        eigvals = eigvals[eigvals > 0]

        total_var = eigvals.sum()
        total_variance[pos] = float(total_var)

        # Top eigenvalues
        k = min(n_components, len(eigvals))
        top_eigenvalues[pos, :k] = eigvals[:k]

        # Effective rank (Shannon entropy)
        if len(eigvals) > 0 and total_var > 0:
            p = eigvals / total_var
            entropy = -np.sum(p * np.log(p))
            effective_ranks[pos] = float(np.exp(entropy))

    return {
        "positions": positions,
        "top_eigenvalues": top_eigenvalues,
        "effective_ranks": effective_ranks,
        "total_variance": total_variance,
    }
