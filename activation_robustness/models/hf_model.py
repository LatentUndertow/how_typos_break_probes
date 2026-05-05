"""
ActivationModel — load HuggingFace model and extract activations.

Merges ModelWrapper + HuggingFaceAdapter + RawActivationExtractor into a single class.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_model import ActivationModelBase
from .types import (
    DEFAULT_DTYPE,
    DEFAULT_HOOK_POINT,
    DEFAULT_POSITION,
    SUPPORTED_HOOK_POINTS,
    PositionStrategy,
)


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    model_name: str
    dtype: str = DEFAULT_DTYPE
    device_map: Optional[Union[str, Dict[str, Any]]] = "auto"
    max_memory: Optional[Dict[int, str]] = None
    trust_remote_code: bool = False


class ActivationModelHF(ActivationModelBase):
    """
    Load a HuggingFace causal LM and extract intermediate activations.

    Usage:
        model = ActivationModel(ModelConfig(model_name="meta-llama/..."))
        model.load()
        features = model.extract_features(text, layer=16, position="last")
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.n_layers: Optional[int] = None
        self.d_model: Optional[int] = None

    def load(self) -> None:
        """Load model and tokenizer via AutoModelForCausalLM."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float32)

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.config.trust_remote_code,
        }

        if self.config.device_map is not None:
            load_kwargs["device_map"] = self.config.device_map
        else:
            load_kwargs["device_map"] = {"": "cpu"}

        if self.config.max_memory is not None:
            load_kwargs["max_memory"] = self.config.max_memory

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs,
        )

        model_config = self.model.config
        self.n_layers = model_config.num_hidden_layers
        self.d_model = model_config.hidden_size

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

    def extract_activations(
        self,
        text: str,
        layers: List[int],
        hook_point: str = DEFAULT_HOOK_POINT,
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and return cached activations for specified layers.

        Args:
            text: Input text to process.
            layers: Layer indices to extract activations from.
            hook_point: "hook_resid_pre" or "hook_resid_post".

        Returns:
            Dict mapping "blocks.{layer}.{hook_point}" to tensors of shape
            (batch, seq_len, d_model).
        """
        self._ensure_loaded()

        if hook_point not in SUPPORTED_HOOK_POINTS:
            raise ValueError(
                f"Unsupported hook point: {hook_point}. "
                f"Supported: {SUPPORTED_HOOK_POINTS}"
            )

        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= self.n_layers:
                raise ValueError(
                    f"Layer {layer_idx} out of range. "
                    f"Model has {self.n_layers} layers (0-{self.n_layers - 1})."
                )

        tokens = self.tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True, use_cache=False)

        hidden_states = outputs.hidden_states

        # Extract only the layers we need and move to CPU immediately
        cache = {}
        for layer_idx in layers:
            if hook_point == "hook_resid_pre":
                cache[f"blocks.{layer_idx}.{hook_point}"] = hidden_states[layer_idx].cpu()
            elif hook_point == "hook_resid_post":
                cache[f"blocks.{layer_idx}.{hook_point}"] = hidden_states[
                    layer_idx + 1
                ].cpu()

        # Free GPU memory — outputs holds all 32 layers of hidden states
        del outputs, hidden_states, input_ids, tokens
        return cache

    def extract_features(
        self,
        text: str,
        layer: int,
        position: PositionStrategy = DEFAULT_POSITION,
        hook_point: str = DEFAULT_HOOK_POINT,
    ) -> np.ndarray:
        """
        Extract a 1D feature vector from a specific layer and position.

        Args:
            text: Input text.
            layer: Layer index.
            position: Position aggregation strategy.
            hook_point: Hook point name.

        Returns:
            1D numpy array of shape (d_model,) or (N * d_model,) for multi-position.
        """
        cache = self.extract_activations(text, layers=[layer], hook_point=hook_point)
        key = f"blocks.{layer}.{hook_point}"
        activations = cache[key]
        return self.aggregate_positions(activations, position)

    @staticmethod
    def aggregate_positions(
        activations: torch.Tensor, position: PositionStrategy
    ) -> np.ndarray:
        """
        Aggregate activations across sequence positions.

        Args:
            activations: (batch, seq_len, d_model) tensor.
            position: Aggregation strategy — 'last', 'mean', 'max', int index,
                or list of int indices (concatenated).

        Returns:
            1D numpy array.
        """
        acts = activations.squeeze(0)  # (seq_len, d_model)

        if position == "last":
            result = acts[-1]
        elif position == "mean":
            result = acts.mean(dim=0)
        elif position == "max":
            result = acts.max(dim=0).values
        elif isinstance(position, int):
            result = acts[position]
        elif isinstance(position, list):
            result = torch.cat([acts[p] for p in position], dim=0)
        else:
            raise ValueError(f"Unknown position type: {position}")

        return result.float().detach().cpu().numpy()
