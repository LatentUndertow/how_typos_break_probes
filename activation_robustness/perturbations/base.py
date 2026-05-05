"""Abstract base classes for the perturbation augmentation kit."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Perturbation(ABC):
    """Single intent-preserving text perturbation.

    All perturbations must preserve the user's intent — they model
    realistic typos, formatting mistakes, and keyboard errors, NOT
    semantic changes.
    """

    name: str
    meta_class: str
    description: str
    preserves_intent: bool = True
    changes_tokenization: bool = True  # most perturbations change tokenization

    @abstractmethod
    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        """Apply perturbation to text.

        Args:
            text: Input text to perturb.
            rng: Numpy random generator for deterministic behavior.

        Returns:
            Perturbed text, or None if not applicable to this input.
        """

    def enumerate_all(self, text: str) -> list[tuple[str, dict]]:
        """Enumerate ALL possible single-application variants of this perturbation.

        Returns:
            List of (perturbed_text, metadata_dict) where metadata contains
            details about the specific variant (e.g., which character was changed,
            at which position). Default implementation returns a single random variant.
            Override in subclasses that support full enumeration.
        """
        result = self.apply(text, np.random.default_rng(0))
        if result is None or result == text:
            return []
        return [(result, {"variant": "default"})]

    def is_applicable(self, text: str) -> bool:
        """Check if this perturbation can be applied to the text."""
        return self.apply(text, np.random.default_rng(0)) is not None

    def validate(self, original: str, perturbed: str) -> bool:
        """Check that perturbation actually changed the text."""
        return perturbed != original

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', meta_class='{self.meta_class}')"


class PerturbationFamily:
    """A named set of perturbations from one or more meta-classes."""

    def __init__(self, name: str, perturbations: list[Perturbation]):
        self.name = name
        self.perturbations = perturbations

    def apply_random(self, text: str, rng: np.random.Generator) -> Optional[tuple[str, str]]:
        """Apply one random applicable perturbation.

        Returns:
            (perturbed_text, perturbation_name) or None if no perturbation is applicable.
        """
        applicable = [p for p in self.perturbations if p.is_applicable(text)]
        if not applicable:
            return None
        p = applicable[rng.integers(len(applicable))]
        result = p.apply(text, rng)
        if result is None:
            return None
        return result, p.name

    def apply_all(self, text: str, rng: np.random.Generator) -> list[tuple[str, str]]:
        """Apply all applicable perturbations.

        Returns:
            List of (perturbed_text, perturbation_name) for each applicable perturbation.
        """
        results = []
        for p in self.perturbations:
            result = p.apply(text, rng)
            if result is not None and result != text:
                results.append((result, p.name))
        return results

    def __repr__(self) -> str:
        return f"PerturbationFamily(name='{self.name}', n={len(self.perturbations)})"

    def __len__(self) -> int:
        return len(self.perturbations)
