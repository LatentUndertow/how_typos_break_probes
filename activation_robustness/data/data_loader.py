"""
Generic data loader for online classifier training.

Resolves dataset classes from ``activation_robustness.datasets`` by class name,
instantiates them with the params dict from the experiment config, applies
the chat template, and optionally finds injection spans.

The experiment YAML lists datasets the same way the ingestion config does::

    datasets:
      - class: EnronDataset
        params:
          split: train
      - class: BIPIADataset
        params:
          bipia_root: /path/to/BIPIA
          task_names: [email, code, table]

DataLoader.load() accepts either a dict ``{class, params, ...}`` or a plain
class-name string (resolved with default params from the class's DATASET_META).

Example:
    >>> loader = DataLoader(tokenizer)
    >>> samples = loader.load({'class': 'EnronDataset', 'params': {'split': 'train'}})
    >>> samples = loader.load('EnronDataset')  # uses defaults
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

from activation_robustness.data.prompt_spec import PromptSpec


# ---------------------------------------------------------------------------
# Default data paths (can be overridden via DataLoader constructor or env vars)
# ---------------------------------------------------------------------------

_DEFAULT_PATHS = {
    'bipia_root': os.environ.get('BIPIA_ROOT', ''),
    'injecagent_root': os.environ.get('INJECAGENT_ROOT', ''),
    'scam_root': os.environ.get('SCAM_ROOT', ''),
}


# ---------------------------------------------------------------------------
# Per-class metadata
# ---------------------------------------------------------------------------

@dataclass
class DatasetMeta:
    """Metadata for a dataset class.

    Each dataset class defines this as a class attribute ``DATASET_META``.
    """
    default_params: Dict[str, Any] = field(default_factory=dict)
    has_spans: bool = False
    category: str = 'benign'
    max_samples: int = 10000


# --- Default train/test splits (by class name) ---
DEFAULT_TRAIN = [
    'AgentDojoDataset', 'BIPIADataset',
    'EnronDataset', 'Dolly15kDataset', 'OpenOrcaDataset',
    'PromptsRanked10kDataset', 'AlpacaDataset', 'APIGenMTDataset',
    'PythonCodeAlpacaDataset', 'PythonCodes25kDataset', 'XlamFunctionCallingDataset',
]

TEST_INSCOPE = ['LLMailDataset', 'InjecAgentDataset', 'GandalfSummarizationDataset', 'ScamDataset']
TEST_BENIGN_FPR = [
    'BitextCustomerSupportDataset', 'WritingPromptsDataset',
    'CodeExerciseDataset', 'SoftAgeDataset',
]
TEST_EXPLORATORY = [
    'JayavibhavDataset', 'DeepsetDataset', 'MosscapDataset',
    'YanismiraouiDataset', 'SafeGuardDataset', 'QualifireDataset',
    'JailbreakClassificationDataset', 'WildJailbreakDataset',
    'AdvBenchDataset', 'HarmBenchDataset',
]
DEFAULT_TEST = TEST_INSCOPE + TEST_BENIGN_FPR + TEST_EXPLORATORY


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def _resolve_class(class_name: str):
    """Resolve a dataset class name from activation_robustness.datasets."""
    import activation_robustness.datasets as ds_module
    cls = getattr(ds_module, class_name, None)
    if cls is None:
        raise KeyError(
            f"Unknown dataset class: {class_name}. "
            f"Available: {[k for k in dir(ds_module) if k.endswith('Dataset')]}"
        )
    return cls


class DataLoader:
    """Load datasets by class name via Dataset classes.

    Args:
        tokenizer: HuggingFace tokenizer for chat template application.
        data_paths: Override default data paths (keys: agentdojo_components,
            bipia_root, injecagent_root, scam_root).
    """

    def __init__(
        self,
        tokenizer,
        data_paths: Optional[Dict[str, str]] = None,
        add_generation_prompt: bool = True,
    ):
        self.tokenizer = tokenizer
        self._span_contexts: Dict[str, Any] = {}
        self._data_paths = {**_DEFAULT_PATHS, **(data_paths or {})}
        self._add_generation_prompt = add_generation_prompt

    def load(
        self,
        spec: Union[str, Dict[str, Any]],
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Load a dataset.

        Args:
            spec: Either a class name string (uses class DATASET_META) or a dict
                with keys ``class`` and optionally ``params``, ``max_samples``.
            max_samples: Override max samples cap.
            verbose: Print loading summary.

        Returns:
            List of dicts with keys: text, injection_span, labels, prompt_id, dataset_id.
        """
        if isinstance(spec, str):
            class_name = spec
            params = {}
            spec_max = None
        else:
            class_name = spec['class']
            params = dict(spec.get('params', {}))
            spec_max = spec.get('max_samples')

        # Resolve class
        cls = _resolve_class(class_name)

        # Merge defaults with overrides (explicit params win)
        meta = getattr(cls, 'DATASET_META', DatasetMeta())
        merged_params = {**meta.default_params, **params}
        cap = max_samples or spec_max or meta.max_samples

        # Resolve data paths from env vars (override empty defaults)
        for key in ('bipia_root', 'injecagent_root', 'scam_root'):
            env_val = self._data_paths.get(key)
            if env_val and key in merged_params:
                merged_params[key] = env_val

        # Instantiate
        dataset = cls(**merged_params)

        # Build span function if applicable
        span_fn = self._get_span_fn(cls, dataset) if meta.has_spans else None

        samples = []
        for i, ps in enumerate(dataset):
            if i >= cap:
                break

            text = self._apply_chat_template(ps)
            is_mal = ps.labels.get('malicious', False)

            span = None
            if span_fn and is_mal:
                span = span_fn(text, ps.labels)
                if span is None:
                    continue

            samples.append({
                'text': text,
                'injection_span': span,
                'labels': ps.labels,
                'prompt_id': ps.prompt_id,
                'dataset_id': ps.dataset_id,
            })

        if verbose:
            n_mal = sum(1 for s in samples if s['labels'].get('malicious', False))
            n_spans = sum(1 for s in samples if s['injection_span'] is not None)
            print(f"  {class_name}: {len(samples)} samples "
                  f"(mal={n_mal}, spans={n_spans})")

        return samples

    def _apply_chat_template(self, ps: PromptSpec) -> str:
        """Apply chat template to a PromptSpec."""
        return self.tokenizer.apply_chat_template(
            ps.messages,
            tools=ps.tools,
            tokenize=False,
            add_generation_prompt=self._add_generation_prompt,
        )

    def _get_span_fn(self, cls, dataset):
        """Return a (text, labels) -> span callable, or None."""
        if not hasattr(cls, 'get_injection_span'):
            return None

        key = cls.__name__
        if key not in self._span_contexts:
            self._span_contexts[key] = self._build_span_context(cls, dataset)
        ctx = self._span_contexts[key]
        return lambda text, labels: cls.get_injection_span(text, labels, **ctx)

    def _build_span_context(self, cls, dataset) -> dict:
        """Build auxiliary kwargs for get_injection_span.

        Delegates to dataset.get_span_context() if available, otherwise
        returns empty dict (for datasets whose get_injection_span needs no
        extra context).
        """
        if hasattr(dataset, 'get_span_context'):
            return dataset.get_span_context()
        return {}

    @staticmethod
    def list_datasets() -> List[str]:
        """Return all dataset class names that have DATASET_META defined."""
        import activation_robustness.datasets as ds_module
        return [
            name for name in dir(ds_module)
            if name.endswith('Dataset')
            and name != 'Dataset'
            and hasattr(getattr(ds_module, name, None), 'DATASET_META')
        ]

    @staticmethod
    def get_meta(class_name: str) -> DatasetMeta:
        """Return metadata for a dataset class from its DATASET_META attribute."""
        cls = _resolve_class(class_name)
        return getattr(cls, 'DATASET_META', DatasetMeta())
