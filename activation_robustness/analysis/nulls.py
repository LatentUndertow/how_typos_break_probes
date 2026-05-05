"""Null models and reproducibility checks for activation robustness analysis.

Provides three core checks:
  1. Background null -- compare noise eigenspectrum against draws from
     the activation covariance to separate perturbation-specific structure
     from inherited representation anisotropy.
  2. Bootstrap eigenspectrum -- confidence intervals on eigenvalues.
  3. Split-half stability -- verify that eigenspectrum and mean displacement
     are reproducible across prompt subsets.

All randomness is controlled via np.random.Generator with explicit seeds.
"""

import numpy as np

from . import statistics as stats


def background_null(
    activations: np.ndarray,
    deltas: np.ndarray,
    n_samples: int = 100,
    seed: int = 42,
) -> dict:
    """Compare noise eigenspectrum against background-anisotropy-matched null.

    Generates synthetic noise by sampling from N(0, C_bg) where C_bg is the
    covariance of the unperturbed activations, scaled to match the observed
    noise magnitude. Any structure in the real noise that exceeds the null
    is perturbation-specific (not inherited from representation anisotropy).

    Args:
        activations: Array of shape (n, d) -- original (unperturbed) activations.
        deltas: Array of shape (n, d) -- observed delta vectors.
        n_samples: Number of null resamples. Each resample draws n synthetic
            noise vectors and computes their eigenspectrum. Default 100.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            delta_eigenvalues: 1-D array -- eigenvalues of the real noise
                covariance (descending).
            null_eigenvalues_mean: 1-D array -- mean eigenvalues across
                null resamples (descending, length min(n, d)).
            null_eigenvalues_std: 1-D array -- std of eigenvalues across
                null resamples.
            null_eigenvalues_q95: 1-D array -- 95th percentile of each
                eigenvalue across null resamples.
            n_super_mp_delta: Number of real eigenvalues above the MP edge
                of the real noise.
            n_super_mp_null_mean: Mean number of super-MP eigenvalues in
                null resamples.
            mp_upper_edge: MP upper edge of the real noise.
            delta_effective_rank: Effective rank of the real noise.
            null_effective_rank_mean: Mean effective rank of null noise.
    """
    activations = np.asarray(activations, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    n, d = deltas.shape
    rng = np.random.default_rng(seed)

    # Eigenspectrum of real noise
    delta_spec = stats.eigenspectrum(deltas, center=True)
    delta_ev = delta_spec["eigenvalues"]
    mp = stats.marchenko_pastur_edge(delta_ev, n, d)

    # Background covariance from activations
    acts_centered = activations - activations.mean(axis=0, keepdims=True)
    # Compute top-k eigenvectors of activation covariance for efficient sampling
    # Using Gram matrix since n < d typically
    if n <= d:
        gram = acts_centered @ acts_centered.T / n
        bg_eigvals, bg_gram_vecs = np.linalg.eigh(gram)
        bg_eigvals = bg_eigvals[::-1]
        bg_gram_vecs = bg_gram_vecs[:, ::-1]
        # Keep positive eigenvalues
        pos_mask = bg_eigvals > 1e-10
        bg_eigvals = bg_eigvals[pos_mask]
        bg_gram_vecs = bg_gram_vecs[:, pos_mask]
        # Recover eigenvectors in d-space
        bg_eigvecs = acts_centered.T @ bg_gram_vecs / np.sqrt(n * bg_eigvals[np.newaxis, :])
    else:
        cov_bg = acts_centered.T @ acts_centered / n
        bg_eigvals, bg_eigvecs = np.linalg.eigh(cov_bg)
        bg_eigvals = bg_eigvals[::-1]
        bg_eigvecs = bg_eigvecs[:, ::-1]
        pos_mask = bg_eigvals > 1e-10
        bg_eigvals = bg_eigvals[pos_mask]
        bg_eigvecs = bg_eigvecs[:, pos_mask]

    # Scale background to match noise magnitude
    # Target: total variance of null should match total variance of noise
    bg_total_var = bg_eigvals.sum()
    delta_total_var = delta_ev.sum()
    if bg_total_var > 0:
        scale_factor = np.sqrt(delta_total_var / bg_total_var)
    else:
        scale_factor = 1.0

    scaled_sqrt_eigvals = np.sqrt(bg_eigvals) * scale_factor

    # Generate null resamples
    k_bg = len(bg_eigvals)
    max_ev_len = min(n, d)

    null_eigenvalues_all = []
    null_super_mp_counts = []
    null_effrk = []

    for _ in range(n_samples):
        # Draw n random vectors in the background eigenbasis, then project to d-space
        z = rng.standard_normal((n, k_bg))  # (n, k_bg)
        z *= scaled_sqrt_eigvals[np.newaxis, :]  # scale each component
        null_deltas = z @ bg_eigvecs.T  # (n, d)

        null_spec = stats.eigenspectrum(null_deltas, center=True)
        null_ev = null_spec["eigenvalues"]

        # Pad or truncate to consistent length
        padded = np.zeros(max_ev_len, dtype=np.float64)
        padded[: len(null_ev)] = null_ev[:max_ev_len]
        null_eigenvalues_all.append(padded)

        null_mp = stats.marchenko_pastur_edge(null_ev, n, d)
        null_super_mp_counts.append(null_mp["n_super_mp"])
        null_effrk.append(stats.effective_rank(null_ev))

    null_eigenvalues_all = np.array(null_eigenvalues_all)  # (n_samples, max_ev_len)

    return {
        "delta_eigenvalues": delta_ev,
        "null_eigenvalues_mean": null_eigenvalues_all.mean(axis=0),
        "null_eigenvalues_std": null_eigenvalues_all.std(axis=0),
        "null_eigenvalues_q95": np.percentile(null_eigenvalues_all, 95, axis=0),
        "n_super_mp_delta": mp["n_super_mp"],
        "n_super_mp_null_mean": float(np.mean(null_super_mp_counts)),
        "mp_upper_edge": mp["mp_upper_edge"],
        "delta_effective_rank": stats.effective_rank(delta_ev),
        "null_effective_rank_mean": float(np.mean(null_effrk)),
    }


def bootstrap_eigenspectrum(
    deltas: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence intervals on eigenvalues of the noise covariance.

    Resamples the prompt dimension (rows of deltas) with replacement and
    recomputes the eigenspectrum each time.

    Args:
        deltas: Array of shape (n, d) -- delta vectors.
        n_bootstrap: Number of bootstrap resamples. Default 100.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            eigenvalues_median: 1-D array -- median eigenvalue at each rank.
            eigenvalues_ci_low: 1-D array -- 2.5th percentile (lower CI bound).
            eigenvalues_ci_high: 1-D array -- 97.5th percentile (upper CI bound).
            eigenvalues_mean: 1-D array -- mean eigenvalue at each rank.
            eigenvalues_std: 1-D array -- std at each rank.
            effective_rank_ci: Tuple (low, median, high) for effective rank.
            n_bootstrap: Number of resamples performed.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n, d = deltas.shape
    rng = np.random.default_rng(seed)

    max_ev_len = min(n, d)
    all_eigenvalues = []
    all_effrk = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_deltas = deltas[indices]

        spec = stats.eigenspectrum(boot_deltas, center=True)
        ev = spec["eigenvalues"]

        padded = np.zeros(max_ev_len, dtype=np.float64)
        padded[: len(ev)] = ev[:max_ev_len]
        all_eigenvalues.append(padded)

        all_effrk.append(stats.effective_rank(ev))

    all_eigenvalues = np.array(all_eigenvalues)  # (n_bootstrap, max_ev_len)
    all_effrk = np.array(all_effrk)

    return {
        "eigenvalues_median": np.median(all_eigenvalues, axis=0),
        "eigenvalues_ci_low": np.percentile(all_eigenvalues, 2.5, axis=0),
        "eigenvalues_ci_high": np.percentile(all_eigenvalues, 97.5, axis=0),
        "eigenvalues_mean": all_eigenvalues.mean(axis=0),
        "eigenvalues_std": all_eigenvalues.std(axis=0),
        "effective_rank_ci": (
            float(np.percentile(all_effrk, 2.5)),
            float(np.median(all_effrk)),
            float(np.percentile(all_effrk, 97.5)),
        ),
        "n_bootstrap": n_bootstrap,
    }


def split_half_stability(
    deltas: np.ndarray,
    n_splits: int = 10,
    seed: int = 42,
) -> dict:
    """Split prompts in half and compare eigenspectrum and mean displacement across halves.

    Tests whether the measured noise structure is a stable property of the
    perturbation or an artifact of the particular prompt sample.

    Args:
        deltas: Array of shape (n, d) -- delta vectors.
        n_splits: Number of random splits to perform. Default 10.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            eigenvalue_cosine_sims: 1-D array (n_splits,) -- cosine similarity
                between eigenvalue vectors from the two halves.
            mu_cosine_sims: 1-D array (n_splits,) -- cosine similarity between
                mean displacement vectors from the two halves.
            top_k_subspace_overlaps: 1-D array (n_splits,) -- mean squared
                cosine of principal angles between top-5 eigenvector subspaces
                of the two halves.
            eigenvalue_cosine_mean: Scalar mean of eigenvalue cosine similarities.
            mu_cosine_mean: Scalar mean of mu cosine similarities.
            subspace_overlap_mean: Scalar mean of subspace overlaps.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n, d = deltas.shape
    rng = np.random.default_rng(seed)

    ev_cosines = []
    mu_cosines = []
    subspace_overlaps = []

    k = 5  # number of top eigenvectors for subspace overlap

    for _ in range(n_splits):
        indices = rng.permutation(n)
        half = n // 2
        idx_a = indices[:half]
        idx_b = indices[half : 2 * half]

        deltas_a = deltas[idx_a]
        deltas_b = deltas[idx_b]

        # Eigenspectrum comparison
        spec_a = stats.eigenspectrum(deltas_a, center=True)
        spec_b = stats.eigenspectrum(deltas_b, center=True)

        # Align eigenvalue vectors to same length
        min_len = min(len(spec_a["eigenvalues"]), len(spec_b["eigenvalues"]))
        ev_a = spec_a["eigenvalues"][:min_len]
        ev_b = spec_b["eigenvalues"][:min_len]

        # Cosine similarity of eigenvalue profiles
        norm_a = np.linalg.norm(ev_a)
        norm_b = np.linalg.norm(ev_b)
        if norm_a > 0 and norm_b > 0:
            ev_cos = float(np.dot(ev_a, ev_b) / (norm_a * norm_b))
        else:
            ev_cos = 0.0
        ev_cosines.append(ev_cos)

        # Mean displacement comparison
        mu_a = deltas_a.mean(axis=0)
        mu_b = deltas_b.mean(axis=0)
        mu_norm_a = np.linalg.norm(mu_a)
        mu_norm_b = np.linalg.norm(mu_b)
        if mu_norm_a > 0 and mu_norm_b > 0:
            mu_cos = float(np.dot(mu_a, mu_b) / (mu_norm_a * mu_norm_b))
        else:
            mu_cos = 0.0
        mu_cosines.append(mu_cos)

        # Top-k subspace overlap via principal angles
        actual_k = min(k, min_len)
        if actual_k > 0:
            vecs_a = spec_a["eigenvectors"][:, :actual_k]  # (d, k)
            vecs_b = spec_b["eigenvectors"][:, :actual_k]  # (d, k)
            # Singular values of V_a^T @ V_b give cosines of principal angles
            cross = vecs_a.T @ vecs_b  # (k, k)
            svs = np.linalg.svd(cross, compute_uv=False)
            # Mean squared cosine: higher = more overlap
            overlap = float(np.mean(svs ** 2))
        else:
            overlap = 0.0
        subspace_overlaps.append(overlap)

    ev_cosines = np.array(ev_cosines)
    mu_cosines = np.array(mu_cosines)
    subspace_overlaps = np.array(subspace_overlaps)

    return {
        "eigenvalue_cosine_sims": ev_cosines,
        "mu_cosine_sims": mu_cosines,
        "top_k_subspace_overlaps": subspace_overlaps,
        "eigenvalue_cosine_mean": float(ev_cosines.mean()),
        "mu_cosine_mean": float(mu_cosines.mean()),
        "subspace_overlap_mean": float(subspace_overlaps.mean()),
    }
