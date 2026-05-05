"""Reusable statistical analysis toolkit for activation robustness research.

Usage:
    from activation_robustness.analysis import extraction, statistics, nulls, plotting

    # Load model and extract activations
    model, tokenizer = extraction.load_hf_model("meta-llama/Llama-3.1-8B-Instruct")
    acts = extraction.extract_activations_hf(model, tokenizer, texts, layers=[31])

    # Compute statistics on delta vectors
    deltas = statistics.compute_deltas(original, perturbed)
    spec = statistics.eigenspectrum(deltas)
    r_eff = statistics.effective_rank(spec["eigenvalues"])

    # Null model comparisons
    null_result = nulls.background_null(activations, deltas)

    # Plotting
    plotting.plot_eigenspectrum(spec["eigenvalues"], mp_edge, save_path="eigen.png")
"""

from . import extraction
from . import statistics
from . import nulls
from . import plotting
from . import sequence
from . import metrics

__all__ = ["extraction", "statistics", "nulls", "plotting", "sequence", "metrics"]
