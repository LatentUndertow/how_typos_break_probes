"""Core statistical computations for activation robustness analysis.

All functions operate on float64 numpy arrays and return plain dicts.
No hidden state -- every function is independently callable.

Implements the Phase 1 priority analyses from the methods report:
  1. Noise covariance eigenspectrum with Marchenko-Pastur baseline
  2. Effective rank and participation ratio
  3. Radial/tangential decomposition
  4. Mean displacement and centered covariance
  5. Directional SNR
  6. Layerwise summaries
"""

import numpy as np


def compute_deltas(
    original: np.ndarray,
    perturbed: np.ndarray,
) -> np.ndarray:
    """Compute delta (difference) vectors between original and perturbed activations.

    Args:
        original: Array of shape (n, d) -- original activations.
        perturbed: Array of shape (n, d) -- perturbed activations.

    Returns:
        Delta array of shape (n, d), where delta[i] = perturbed[i] - original[i].
        Dtype is float64.
    """
    original = np.asarray(original, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    if original.shape != perturbed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs perturbed {perturbed.shape}"
        )
    return perturbed - original


def eigenspectrum(
    deltas: np.ndarray,
    center: bool = True,
) -> dict:
    """Eigendecomposition of the noise covariance matrix.

    When n < d (common case: fewer prompts than dimensions), computes
    the n x n Gram matrix for efficiency. Eigenvalues are scaled to
    match the d x d covariance convention.

    Args:
        deltas: Array of shape (n, d) -- delta vectors.
        center: If True, subtract the mean delta before computing the
            covariance. This separates the mean displacement from the
            spread. Default True.

    Returns:
        Dict with keys:
            eigenvalues: 1-D array of eigenvalues in descending order.
                Length is min(n, d). Includes only non-negative values.
            eigenvectors: 2-D array of shape (d, k) where k = len(eigenvalues).
                Columns are eigenvectors corresponding to eigenvalues.
            cumulative_variance: 1-D array -- cumulative fraction of total
                variance explained by top-k components.
            total_variance: Scalar -- sum of all eigenvalues (trace of cov).
            n: Number of samples.
            d: Dimensionality.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n, d = deltas.shape

    if center:
        deltas = deltas - deltas.mean(axis=0, keepdims=True)

    if n <= d:
        # Gram matrix approach: (1/n) * deltas @ deltas.T is n x n
        gram = deltas @ deltas.T / n  # (n, n)
        eigvals, gram_vecs = np.linalg.eigh(gram)  # ascending order

        # Reverse to descending
        eigvals = eigvals[::-1]
        gram_vecs = gram_vecs[:, ::-1]

        # Keep only positive eigenvalues
        mask = eigvals > 0
        eigvals = eigvals[mask]
        gram_vecs = gram_vecs[:, mask]

        # Recover eigenvectors in d-space: v_j = (1 / sqrt(n * lambda_j)) * X^T @ u_j
        eigvecs = deltas.T @ gram_vecs / np.sqrt(n * eigvals[np.newaxis, :])
    else:
        cov = deltas.T @ deltas / n  # (d, d)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Reverse to descending, keep positive
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        mask = eigvals > 0
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]

    total_var = eigvals.sum()
    cumvar = np.cumsum(eigvals) / total_var if total_var > 0 else np.zeros_like(eigvals)

    return {
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "cumulative_variance": cumvar,
        "total_variance": float(total_var),
        "n": n,
        "d": d,
    }


def marchenko_pastur_edge(
    eigenvalues: np.ndarray,
    n: int,
    d: int,
) -> dict:
    """Compute the Marchenko-Pastur upper edge and count super-MP eigenvalues.

    The MP law describes the eigenvalue distribution of a sample covariance
    matrix from i.i.d. Gaussian vectors. Eigenvalues exceeding the MP upper
    edge indicate structured (non-random) noise components.

    Args:
        eigenvalues: 1-D array of eigenvalues in descending order.
        n: Number of samples used to estimate the covariance.
        d: Dimensionality of the data.

    Returns:
        Dict with keys:
            mp_upper_edge: The MP upper edge lambda_+.
            mp_lower_edge: The MP lower edge lambda_-.
            sigma_sq: Estimated bulk variance (trace / min(n, d)).
            gamma: Aspect ratio d / n.
            n_super_mp: Number of eigenvalues exceeding the upper edge.
            super_mp_eigenvalues: Array of eigenvalues above the edge.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    gamma = d / n

    # Estimate sigma^2 from the bulk: total variance / effective rank
    # Using trace / min(n, d) as in the methods report
    sigma_sq = eigenvalues.sum() / min(n, d)

    sqrt_gamma = np.sqrt(gamma)
    upper_edge = sigma_sq * (1.0 + sqrt_gamma) ** 2
    lower_edge = sigma_sq * (1.0 - sqrt_gamma) ** 2

    super_mp_mask = eigenvalues > upper_edge
    n_super_mp = int(super_mp_mask.sum())

    return {
        "mp_upper_edge": float(upper_edge),
        "mp_lower_edge": float(lower_edge),
        "sigma_sq": float(sigma_sq),
        "gamma": float(gamma),
        "n_super_mp": n_super_mp,
        "super_mp_eigenvalues": eigenvalues[super_mp_mask],
    }


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Shannon-entropy effective rank (Roy and Vetterli, 2007).

    Defined as exp(H(p)) where p_i = lambda_i / sum(lambda_j) and
    H(p) = -sum(p_i * log(p_i)).

    For isotropic noise in d dimensions, r_eff = d.
    For rank-1 noise, r_eff = 1.

    Args:
        eigenvalues: 1-D array of non-negative eigenvalues.

    Returns:
        Effective rank as a float.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) == 0:
        return 0.0

    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Participation ratio: (sum lambda_i)^2 / sum(lambda_i^2).

    Simpler and more robust to small-eigenvalue estimation noise than
    effective rank. Widely used in physics.

    Args:
        eigenvalues: 1-D array of non-negative eigenvalues.

    Returns:
        Participation ratio as a float.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) == 0:
        return 0.0

    trace = eigenvalues.sum()
    trace_sq = np.sum(eigenvalues ** 2)

    if trace_sq == 0:
        return 0.0

    return float(trace ** 2 / trace_sq)


def radial_tangential(
    original: np.ndarray,
    perturbed: np.ndarray,
) -> dict:
    """Decompose deltas into radial (norm-changing) and tangential (direction-changing) components.

    The radial component is the part of delta along the original activation
    direction (changes norm, preserves direction). The tangential component
    is orthogonal to the original direction (changes direction, preserves norm
    to first order).

    Args:
        original: Array of shape (n, d) -- original activations.
        perturbed: Array of shape (n, d) -- perturbed activations.

    Returns:
        Dict with keys:
            radial_norms: 1-D array (n,) -- L2 norm of radial component per sample.
            tangential_norms: 1-D array (n,) -- L2 norm of tangential component per sample.
            delta_norms: 1-D array (n,) -- L2 norm of full delta per sample.
            radial_fractions: 1-D array (n,) -- ||delta_r|| / ||delta|| per sample.
            tangential_fractions: 1-D array (n,) -- ||delta_t|| / ||delta|| per sample.
            radial_frac_mean: Scalar -- mean radial fraction.
            radial_frac_std: Scalar -- std of radial fraction.
            tangential_frac_mean: Scalar -- mean tangential fraction.
            tangential_frac_std: Scalar -- std of tangential fraction.
    """
    original = np.asarray(original, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)

    deltas = perturbed - original  # (n, d)

    # Original unit vectors
    orig_norms = np.linalg.norm(original, axis=1, keepdims=True)  # (n, 1)
    # Avoid division by zero
    safe_norms = np.maximum(orig_norms, 1e-12)
    h_hat = original / safe_norms  # (n, d)

    # Radial component: projection of delta onto h_hat
    # delta_r = (delta . h_hat) * h_hat
    radial_proj = np.sum(deltas * h_hat, axis=1, keepdims=True)  # (n, 1)
    delta_r = radial_proj * h_hat  # (n, d)
    delta_t = deltas - delta_r  # (n, d)

    radial_norms = np.linalg.norm(delta_r, axis=1)
    tangential_norms = np.linalg.norm(delta_t, axis=1)
    delta_norms = np.linalg.norm(deltas, axis=1)

    # Fractions (avoid div by zero)
    safe_delta_norms = np.maximum(delta_norms, 1e-12)
    radial_fracs = radial_norms / safe_delta_norms
    tangential_fracs = tangential_norms / safe_delta_norms

    return {
        "radial_norms": radial_norms,
        "tangential_norms": tangential_norms,
        "delta_norms": delta_norms,
        "radial_fractions": radial_fracs,
        "tangential_fractions": tangential_fracs,
        "radial_frac_mean": float(radial_fracs.mean()),
        "radial_frac_std": float(radial_fracs.std()),
        "tangential_frac_mean": float(tangential_fracs.mean()),
        "tangential_frac_std": float(tangential_fracs.std()),
    }


def mean_displacement(deltas: np.ndarray) -> dict:
    """Compute mean displacement vector and centered covariance decomposition.

    Separates the noise into a rank-1 mean shift (systematic bias) and a
    centered spread component (variable noise).

    Args:
        deltas: Array of shape (n, d) -- delta vectors.

    Returns:
        Dict with keys:
            mu: 1-D array (d,) -- mean displacement vector.
            mu_norm: Scalar -- L2 norm of the mean displacement.
            mean_delta_norm: Scalar -- mean of per-sample delta norms.
            ratio: Scalar -- mu_norm / mean_delta_norm. A high ratio
                indicates a systematic bias direction.
            mu_hat: 1-D array (d,) -- unit vector in the mean displacement
                direction.
            centered_eigenvalues: 1-D array -- eigenvalues of the centered
                covariance (mean removed), descending order.
            centered_total_variance: Scalar -- total variance after centering.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n, d = deltas.shape

    mu = deltas.mean(axis=0)  # (d,)
    mu_norm = float(np.linalg.norm(mu))

    delta_norms = np.linalg.norm(deltas, axis=1)
    mean_delta_norm = float(delta_norms.mean())

    # Unit direction
    if mu_norm > 1e-12:
        mu_hat = mu / mu_norm
    else:
        mu_hat = np.zeros(d, dtype=np.float64)

    # Centered covariance eigenvalues
    centered = deltas - mu[np.newaxis, :]
    spec = eigenspectrum(centered, center=False)

    return {
        "mu": mu,
        "mu_norm": mu_norm,
        "mean_delta_norm": mean_delta_norm,
        "ratio": mu_norm / max(mean_delta_norm, 1e-12),
        "mu_hat": mu_hat,
        "centered_eigenvalues": spec["eigenvalues"],
        "centered_total_variance": spec["total_variance"],
    }


def directional_snr(
    activations: np.ndarray,
    deltas: np.ndarray,
    directions: np.ndarray,
) -> dict:
    """Signal-to-noise ratio along specific directions.

    For each direction w, computes:
        SNR_w = Var_prompts(w . h) / mean(Var_perturbations(w . h))

    Here the "signal" variance is across prompts (between-prompt variance)
    and the "noise" variance is the perturbation-induced variance projected
    onto w.

    Note: when each prompt has only one perturbation, the per-prompt
    perturbation variance is estimated from (w . delta)^2.

    Args:
        activations: Array of shape (n, d) -- original activations.
        deltas: Array of shape (n, d) -- delta vectors (perturbed - original).
        directions: Array of shape (k, d) -- directions to project onto.
            Each row is a direction vector (need not be unit-length;
            will be normalized internally).

    Returns:
        Dict with keys:
            snr: 1-D array (k,) -- SNR for each direction.
            signal_variance: 1-D array (k,) -- between-prompt variance
                projected onto each direction.
            noise_variance: 1-D array (k,) -- mean perturbation variance
                projected onto each direction.
            directions_used: 2-D array (k, d) -- the normalized directions.
    """
    activations = np.asarray(activations, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)

    if directions.ndim == 1:
        directions = directions[np.newaxis, :]

    # Normalize directions
    dir_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    safe_norms = np.maximum(dir_norms, 1e-12)
    w = directions / safe_norms  # (k, d)

    # Project activations and deltas onto each direction
    proj_act = activations @ w.T  # (n, k)
    proj_delta = deltas @ w.T  # (n, k)

    # Between-prompt signal variance
    signal_var = proj_act.var(axis=0)  # (k,)

    # Perturbation noise variance: mean of (w . delta_i)^2
    # When one perturbation per prompt, Var_pert(w.h') ~ (w.delta)^2
    noise_var = np.mean(proj_delta ** 2, axis=0)  # (k,)

    # SNR
    safe_noise = np.maximum(noise_var, 1e-12)
    snr = signal_var / safe_noise

    return {
        "snr": snr,
        "signal_variance": signal_var,
        "noise_variance": noise_var,
        "directions_used": w,
    }


def layerwise_summary(
    activations_by_layer: dict[int, np.ndarray],
    deltas_by_layer: dict[int, np.ndarray],
) -> dict[int, dict]:
    """Compute key statistics at each layer.

    Runs eigenspectrum analysis, effective rank, participation ratio,
    radial/tangential decomposition, mean displacement, and MP edge
    for each layer.

    Args:
        activations_by_layer: Dict mapping layer index to activations
            array of shape (n, d).
        deltas_by_layer: Dict mapping layer index to deltas array
            of shape (n, d).

    Returns:
        Dict mapping layer index to a stats dict containing:
            eigenvalues: top eigenvalues of noise covariance
            n_super_mp: number of eigenvalues above MP edge
            mp_upper_edge: MP upper edge
            effective_rank: Shannon-entropy effective rank of noise
            participation_ratio: participation ratio of noise
            r_eff_signal: effective rank of the activations themselves
            pr_signal: participation ratio of activations
            total_variance: total noise variance (trace of cov)
            mean_delta_norm: mean L2 norm of deltas
            mean_act_norm: mean L2 norm of activations
            radial_frac_mean: mean radial fraction
            tangential_frac_mean: mean tangential fraction
            mu_norm: norm of mean displacement
            mu_ratio: mu_norm / mean_delta_norm
            cumvar_top1: variance explained by top eigenvalue
            cumvar_top5: variance explained by top 5
            cumvar_top10: variance explained by top 10
    """
    layers = sorted(set(activations_by_layer.keys()) & set(deltas_by_layer.keys()))
    results = {}

    for layer in layers:
        acts = np.asarray(activations_by_layer[layer], dtype=np.float64)
        delt = np.asarray(deltas_by_layer[layer], dtype=np.float64)
        n, d = delt.shape

        # Eigenspectrum of noise
        spec = eigenspectrum(delt, center=True)
        ev = spec["eigenvalues"]

        # MP edge
        mp = marchenko_pastur_edge(ev, spec["n"], spec["d"])

        # Effective rank and PR of noise
        r_eff = effective_rank(ev)
        pr = participation_ratio(ev)

        # Eigenspectrum of signal (activations)
        sig_spec = eigenspectrum(acts, center=True)
        sig_ev = sig_spec["eigenvalues"]
        r_eff_sig = effective_rank(sig_ev)
        pr_sig = participation_ratio(sig_ev)

        # Radial/tangential
        perturbed = acts + delt
        rt = radial_tangential(acts, perturbed)

        # Mean displacement
        md = mean_displacement(delt)

        # Cumulative variance at key thresholds
        cv = spec["cumulative_variance"]

        stats = {
            "eigenvalues": ev,
            "n_super_mp": mp["n_super_mp"],
            "mp_upper_edge": mp["mp_upper_edge"],
            "effective_rank": r_eff,
            "participation_ratio": pr,
            "r_eff_signal": r_eff_sig,
            "pr_signal": pr_sig,
            "total_variance": spec["total_variance"],
            "mean_delta_norm": float(np.linalg.norm(delt, axis=1).mean()),
            "mean_act_norm": float(np.linalg.norm(acts, axis=1).mean()),
            "radial_frac_mean": rt["radial_frac_mean"],
            "tangential_frac_mean": rt["tangential_frac_mean"],
            "mu_norm": md["mu_norm"],
            "mu_ratio": md["ratio"],
            "cumvar_top1": float(cv[0]) if len(cv) > 0 else 0.0,
            "cumvar_top5": float(cv[4]) if len(cv) > 4 else float(cv[-1]) if len(cv) > 0 else 0.0,
            "cumvar_top10": float(cv[9]) if len(cv) > 9 else float(cv[-1]) if len(cv) > 0 else 0.0,
        }
        results[layer] = stats

    return results


# ---------------------------------------------------------------------------
# SIP-style analysis (uncentered second moment + eigengap)
# ---------------------------------------------------------------------------

def second_moment_spectrum(activations: np.ndarray) -> dict:
    """Uncentered second moment (Fisher operator) eigenspectrum.

    SIP (Huang 2025) uses Gamma = E[h h^T] — the uncentered second moment
    of activations, NOT the mean-centered covariance. This captures both
    the dominant representation directions AND the mean.

    Args:
        activations: Array of shape (n, d).

    Returns:
        Dict with eigenvalues, eigenvectors, cumulative_variance, total_variance.
        Same format as eigenspectrum() for compatibility.
    """
    return eigenspectrum(activations, center=False)


def eigengap_analysis(eigenvalues: np.ndarray, k: int = None) -> dict:
    """SIP-style eigengap diagnostic.

    The eigengap at position k is lambda_k - lambda_{k+1}. SIP shows
    that probe reliability depends on this gap exceeding the estimation
    error. Large gap = stable probe subspace. Small gap = fragile.

    Args:
        eigenvalues: Sorted descending eigenvalues.
        k: Position to measure gap. If None, finds the position with
           the largest relative gap (gap / lambda_k).

    Returns:
        Dict with:
            gaps: all consecutive gaps (lambda_i - lambda_{i+1})
            relative_gaps: gaps normalized by lambda_i
            k: the position of largest relative gap (natural subspace cut)
            gap_at_k: the gap value at position k
            ratio_at_k: gap / lambda_k at position k
            eigenvalues_above_k: eigenvalues[0:k]
            eigenvalues_below_k: eigenvalues[k:]
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eigenvalues = eigenvalues[eigenvalues > 0]  # only positive

    if len(eigenvalues) < 2:
        return {"gaps": np.array([]), "k": 0, "gap_at_k": 0.0}

    gaps = eigenvalues[:-1] - eigenvalues[1:]
    relative_gaps = gaps / (eigenvalues[:-1] + 1e-10)

    if k is None:
        # Find largest relative gap (natural subspace boundary)
        k = int(np.argmax(relative_gaps)) + 1  # +1 because gap at index i is between i and i+1

    k = min(k, len(eigenvalues) - 1)
    gap_idx = k - 1 if k > 0 else 0

    return {
        "gaps": gaps,
        "relative_gaps": relative_gaps,
        "k": k,
        "gap_at_k": float(gaps[gap_idx]),
        "ratio_at_k": float(relative_gaps[gap_idx]),
        "eigenvalues_above_k": eigenvalues[:k],
        "eigenvalues_below_k": eigenvalues[k:],
    }


def noise_subspace_alignment(
    noise_eigenvectors: np.ndarray,
    activation_eigenvectors: np.ndarray,
    k_noise: int = 5,
    k_activation: int = 10,
) -> dict:
    """Measure alignment between noise principal directions and activation principal directions.

    Projects noise eigenvectors onto the top activation eigenvectors to determine
    whether perturbation noise aligns with the dominant representation directions.

    Args:
        noise_eigenvectors: (d, k) from eigenspectrum of noise covariance.
        activation_eigenvectors: (d, k) from eigenspectrum/second_moment of activations.
        k_noise: Number of top noise directions to use.
        k_activation: Number of top activation directions to use.

    Returns:
        Dict with:
            cosines: (k_noise, k_activation) matrix of absolute cosine similarities.
            max_alignment_per_noise_dir: for each noise direction, its max alignment with any activation dir.
            mean_alignment: overall mean of the cosine matrix.
            principal_angles: canonical angles between the two subspaces.
    """
    U_noise = noise_eigenvectors[:, :k_noise]  # (d, k_noise)
    U_act = activation_eigenvectors[:, :k_activation]  # (d, k_activation)

    # Cosine matrix
    cosines = np.abs(U_noise.T @ U_act)  # (k_noise, k_activation)

    # Principal angles
    svd_vals = np.linalg.svd(U_noise.T @ U_act, compute_uv=False)
    angles = np.arccos(np.clip(svd_vals, 0, 1))

    return {
        "cosines": cosines,
        "max_alignment_per_noise_dir": cosines.max(axis=1),
        "mean_alignment": float(cosines.mean()),
        "principal_angles_deg": np.degrees(angles),
        "mean_principal_angle_deg": float(np.degrees(angles.mean())),
    }
