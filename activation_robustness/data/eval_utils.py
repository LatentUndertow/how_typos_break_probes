"""
Shared evaluation utilities for online-training classifiers.

Used by run_experiment.py and the evaluate() method on DetectionProbe
and MultiArchProbe. These classifiers train
on activations via BatchProviders (cached or live extraction) and evaluate
at fixed thresholds — unlike the sklearn/LODO classifiers in
base_classifier.py which use threshold-strategy selection on pre-extracted
features.

Usage::

    evaluator = OnlineEvaluator()

    # Token-level probe (DetectionProbe):
    result = evaluator.evaluate_token_scores(token_scores, labels)

    # Sequence-level probe (OnlineProbe, MultiArchProbe):
    result = evaluator.evaluate_sequence_scores(scores, labels)

Both return the same schema::

    {
        'aggregations': {"{agg}_t{thresh}": {recall, fpr, ...}, ...},
        'token_scores': List[np.ndarray] | None,
        'seq_scores': {name: (n,) array, ...},
    }
"""

from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class SeqAggregation(str, Enum):
    """Token-to-sequence score aggregation strategy."""
    MAX = 'max'
    RUNNING_MEAN_MAX = 'running_mean_max'
    TOP_K_MEAN = 'top_k_mean'


class EvalThreshold(float, Enum):
    """Fixed thresholds for binary classification evaluation."""
    LOW = 0.3
    MID = 0.5
    HIGH = 0.7


# Preferred aggregation key for summary printing, in priority order.
SUMMARY_AGG_PREFERENCE = [
    f'{SeqAggregation.RUNNING_MEAN_MAX.value}_t{EvalThreshold.MID.value}',
    f'score_t{EvalThreshold.MID.value}',
]


class OnlineEvaluator:
    """Fixed-threshold evaluator for online-training classifiers.

    Computes recall / FPR / precision at fixed thresholds. For token-level
    probes, first aggregates per-token scores to sequence level via three
    strategies (max, running_mean_max, top_k_mean).

    Args:
        thresholds: Score thresholds to evaluate at.
    """

    def __init__(self, thresholds: Optional[List[EvalThreshold]] = None):
        self.thresholds = thresholds or list(EvalThreshold)

    # ------------------------------------------------------------------
    # Public API — two entry points, same output schema
    # ------------------------------------------------------------------

    def evaluate_token_scores(
        self,
        token_scores: List[np.ndarray],
        labels: np.ndarray,
    ) -> dict:
        """Evaluate a token-level probe (e.g. DetectionProbe).

        Aggregates per-token scores to sequence level via three strategies,
        then evaluates each at every threshold.

        Returns:
            {aggregations, token_scores, seq_scores}
        """
        seq_scores = self._aggregate_token_scores(token_scores)
        return {
            'aggregations': self._eval_at_thresholds(seq_scores, labels),
            'token_scores': token_scores,
            'seq_scores': seq_scores,
        }

    def evaluate_sequence_scores(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """Evaluate a sequence-level probe (e.g. OnlineProbe, MultiArchProbe).

        Wraps the single score array and evaluates at every threshold.

        Returns:
            {aggregations, token_scores: None, seq_scores}
        """
        seq_scores = {'score': scores}
        return {
            'aggregations': self._eval_at_thresholds(seq_scores, labels),
            'token_scores': None,
            'seq_scores': seq_scores,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_token_scores(
        token_scores: List[np.ndarray],
        running_mean_window: int = 10,
        top_k: int = 20,
    ) -> Dict[str, np.ndarray]:
        """Compute 3 sequence-level aggregations from per-token scores."""
        seq_max = np.array([ts.max() if len(ts) > 0 else 0.0
                            for ts in token_scores])

        w = running_mean_window

        def _running_mean_max(ts):
            if len(ts) < w:
                return ts.mean() if len(ts) > 0 else 0.0
            kernel = np.ones(w) / w
            return np.convolve(ts, kernel, mode='valid').max()

        seq_rm = np.array([_running_mean_max(ts) for ts in token_scores])

        k = top_k

        def _top_k_mean(ts):
            if len(ts) == 0:
                return 0.0
            kk = min(k, len(ts))
            return np.sort(ts)[-kk:].mean()

        seq_topk = np.array([_top_k_mean(ts) for ts in token_scores])
        return {
            SeqAggregation.MAX.value: seq_max,
            SeqAggregation.RUNNING_MEAN_MAX.value: seq_rm,
            SeqAggregation.TOP_K_MEAN.value: seq_topk,
        }

    def _eval_at_thresholds(
        self,
        seq_scores_dict: Dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate named score arrays at fixed thresholds.

        Convention (matches cross_dataset_detection.py):
        fpr is nan when n_ben==0, recall uses n_mal as denominator.
        """
        n_mal = int((labels == 1).sum())
        n_ben = int((labels == 0).sum())
        results = {}
        for agg_name, scores in seq_scores_dict.items():
            for thresh in self.thresholds:
                preds = (scores > thresh.value).astype(int)
                tp = int(((preds == 1) & (labels == 1)).sum())
                fp = int(((preds == 1) & (labels == 0)).sum())
                fn = int(((preds == 0) & (labels == 1)).sum())
                tn = int(((preds == 0) & (labels == 0)).sum())
                recall = tp / max(n_mal, 1)
                fpr = fp / max(n_ben, 1) if n_ben > 0 else float('nan')
                precision = tp / max(tp + fp, 1)
                results[f"{agg_name}_t{thresh.value}"] = {
                    'recall': float(recall),
                    'fpr': float(fpr),
                    'precision': float(precision),
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                }
        return results
