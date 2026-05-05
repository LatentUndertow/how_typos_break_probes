"""
ActivationModelBase — abstract interface for activation extraction backends.

Subclasses:
    - ActivationModelHF: HuggingFace in-process (hf_model.py)
    - ActivationModelVLLM: vLLM server over HTTP (vllm_server_model.py)
    - ActivationModelVLLMOffline: vLLM in-process (vllm_offline_model.py)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from .types import DEFAULT_HOOK_POINT, DEFAULT_POSITION, PositionStrategy


class ActivationModelBase(ABC):
    """
    Abstract interface for activation extraction.

    Any backend that can produce a 1D feature vector from text
    can implement this interface and plug into ActivationPipeline.
    """

    #: Number of transformer layers (set by subclass after load)
    n_layers: Optional[int] = None

    #: Hidden dimension size (set by subclass after load)
    d_model: Optional[int] = None

    #: Tokenizer instance (set by subclass after load)
    tokenizer: Any = None

    @abstractmethod
    def load(self) -> None:
        """Initialize the backend (load model, connect to server, etc.)."""
        ...

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """Format messages into a prompt string using the model's chat template."""
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=kwargs.pop("add_generation_prompt", True),
            enable_thinking=False,
            **kwargs,
        )

    @abstractmethod
    def extract_features(
        self,
        text: str,
        layer: int,
        position: PositionStrategy = DEFAULT_POSITION,
        hook_point: str = DEFAULT_HOOK_POINT,
    ) -> np.ndarray:
        """
        Extract a 1D feature vector from text.

        Args:
            text: Pre-formatted input text.
            layer: Layer index to extract from.
            position: Position strategy (backend may ignore if not applicable).
            hook_point: Hook point name (backend may ignore if not applicable).

        Returns:
            1D numpy array of shape (d_model,).
        """
        ...
