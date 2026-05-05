"""
YAML-driven experiment configuration for online classifier training.

Dataset entries are class names, with optional params as a nested dict::

    data:
      train:
        - EnronDataset
        - BIPIADataset:
            task_names: [email, code, table]
      test:
        - LLMailDataset

Example:
    >>> config = ExperimentConfig.from_yaml('configs/detection_cross_dataset.yaml')
    >>> clf = config.build_classifier(extractor=extractor)
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path


# A dataset entry from YAML is either:
#   "EnronDataset"                          -> str
#   {"BIPIADataset": {"task_names": [...]}} -> dict with one key
DatasetEntry = Union[str, Dict[str, Any]]


def _entry_class(entry: DatasetEntry) -> str:
    """Extract class name from a dataset entry."""
    if isinstance(entry, str):
        return entry
    # dict with single key: {ClassName: {params}}
    return next(iter(entry))


def _entry_params(entry: DatasetEntry) -> Dict[str, Any]:
    """Extract params dict from a dataset entry."""
    if isinstance(entry, str):
        return {}
    return next(iter(entry.values())) or {}


def entry_to_loader_spec(entry: DatasetEntry) -> Union[str, Dict[str, Any]]:
    """Convert a config entry to what DataLoader.load() expects."""
    if isinstance(entry, str):
        return entry
    class_name = _entry_class(entry)
    params = _entry_params(entry)
    if not params:
        return class_name
    return {'class': class_name, 'params': params}


@dataclass
class DataConfig:
    train: List[DatasetEntry] = field(default_factory=list)
    test: List[DatasetEntry] = field(default_factory=list)
    lodo_hold_out: Optional[str] = None  # class name to hold out
    seed: int = 42


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    layer: int = 31
    max_seq_len: int = 16384
    attn_implementation: Optional[str] = "sdpa"


@dataclass
class OutputConfig:
    output_dir: str = "experiment_output"
    save_scores: bool = True
    save_probes: bool = True
    resume: bool = False
    cache_dir: Optional[str] = None
    mlflow_enabled: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: Optional[str] = None
    mlflow_artifact_location: Optional[str] = None
    mlflow_run_name: Optional[str] = None
    mlflow_log_every: int = 20


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    classifier_type: str  # 'detection_probe', 'gemini_probe'
    classifier_params: Dict[str, Any] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        data = DataConfig(**raw.get('data', {}))
        model = ModelConfig(**raw.get('model', {}))
        output = OutputConfig(**raw.get('output', {}))

        return cls(
            classifier_type=raw['classifier_type'],
            classifier_params=raw.get('classifier_params', {}),
            data=data,
            model=model,
            output=output,
        )

    def resolve_splits(self) -> Tuple[List[DatasetEntry], List[DatasetEntry]]:
        """Apply LODO hold-out and return (train_entries, test_entries).

        If lodo_hold_out is set, removes matching class from train and
        prepends it to test.
        """
        train = list(self.data.train)
        test = list(self.data.test)

        if self.data.lodo_hold_out:
            lodo = self.data.lodo_hold_out
            train = [e for e in train if _entry_class(e) != lodo]
            # In LODO mode, only evaluate on the held-out dataset
            test = [lodo]

        return train, test

    def build_classifier(self, extractor=None):
        """Instantiate the classifier from config."""
        params = dict(self.classifier_params)

        if self.classifier_type == 'detection_probe':
            raise NotImplementedError(
                "The 'detection_probe' classifier is not included in this "
                "release; only the multi-architecture probe used for paper "
                "§6/§7 is shipped."
            )

        elif self.classifier_type == 'gemini_probe':
            from activation_robustness.probes.architectures import (
                MultiArchProbe, ProbeConfig,
            )
            config = ProbeConfig(**params)
            return MultiArchProbe(config, extractor=extractor)

        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        raw = {
            'classifier_type': self.classifier_type,
            'classifier_params': self.classifier_params,
            'data': {
                'train': self.data.train,
                'test': self.data.test,
                'lodo_hold_out': self.data.lodo_hold_out,
                'seed': self.data.seed,
            },
            'model': {
                'model_name': self.model.model_name,
                'layer': self.model.layer,
                'max_seq_len': self.model.max_seq_len,
                'attn_implementation': self.model.attn_implementation,
            },
            'output': {
                'output_dir': self.output.output_dir,
                'save_scores': self.output.save_scores,
                'save_probes': self.output.save_probes,
                'resume': self.output.resume,
                'cache_dir': self.output.cache_dir,
                'mlflow_enabled': self.output.mlflow_enabled,
                'mlflow_tracking_uri': self.output.mlflow_tracking_uri,
                'mlflow_experiment': self.output.mlflow_experiment,
                'mlflow_artifact_location': self.output.mlflow_artifact_location,
                'mlflow_run_name': self.output.mlflow_run_name,
                'mlflow_log_every': self.output.mlflow_log_every,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
