"""Omission perturbations — user forgot to type something."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Perturbation
from .registry import register


@dataclass
class RemoveTrailingPunctuation(Perturbation):
    """Remove trailing punctuation entirely.

    Example: "How do I cook?" → "How do I cook"
    Models: user forgot to type punctuation at end
    """
    name: str = "omission.remove_trailing_punctuation"
    meta_class: str = "omission"
    description: str = "Remove trailing punctuation (forgot to type it)"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        stripped = text.rstrip()
        if not stripped or stripped[-1] not in ".?!,;:":
            return None
        return stripped[:-1]


@dataclass
class MissingSpace(Perturbation):
    """Remove one space from the text.

    Example: "How do I cook pasta?" → "How doI cook pasta?"
    Models: spacebar not pressed between words
    """
    name: str = "omission.missing_space"
    meta_class: str = "omission"
    description: str = "Remove one space (forgot spacebar)"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        space_indices = [i for i, c in enumerate(text) if c == " "]
        if not space_indices:
            return None
        idx = space_indices[rng.integers(len(space_indices))]
        return text[:idx] + text[idx + 1:]

    def enumerate_all(self, text: str) -> list[tuple[str, dict]]:
        results = []
        for i, c in enumerate(text):
            if c == " ":
                perturbed = text[:i] + text[i + 1:]
                results.append((perturbed, {"position": i, "type": "missing_space"}))
        return results


@dataclass
class ExtraSpace(Perturbation):
    """Add one extra space somewhere in the text.

    Example: "How do I cook pasta?" → "How do  I cook pasta?"
    Models: accidental double spacebar press
    """
    name: str = "omission.extra_space"
    meta_class: str = "omission"
    description: str = "Add extra space (double spacebar)"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        space_indices = [i for i, c in enumerate(text) if c == " "]
        if not space_indices:
            return None
        idx = space_indices[rng.integers(len(space_indices))]
        return text[:idx] + " " + text[idx:]

    def enumerate_all(self, text: str) -> list[tuple[str, dict]]:
        results = []
        for i, c in enumerate(text):
            if c == " ":
                perturbed = text[:i] + " " + text[i:]
                results.append((perturbed, {"position": i, "type": "extra_space"}))
        return results


# Register all
register(RemoveTrailingPunctuation())
register(MissingSpace())
register(ExtraSpace())
