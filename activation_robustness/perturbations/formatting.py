"""Formatting perturbations — intentional text style differences."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Perturbation
from .registry import register


@dataclass
class LowercaseAll(Perturbation):
    """Convert entire text to lowercase.

    Example: "How Do I Cook Pasta?" → "how do i cook pasta?"
    Models: user typed without shift / caps lock off
    """
    name: str = "formatting.lowercase_all"
    meta_class: str = "formatting"
    description: str = "Convert all text to lowercase"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        lowered = text.lower()
        if lowered == text:
            return None
        return lowered


@dataclass
class NoCapitalizeFirst(Perturbation):
    """Lowercase the first character.

    Example: "How do I cook pasta?" → "how do I cook pasta?"
    Models: forgot shift at start of sentence
    """
    name: str = "formatting.no_capitalize_first"
    meta_class: str = "formatting"
    description: str = "Lowercase the first character"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        if not text or not text[0].isupper():
            return None
        return text[0].lower() + text[1:]


@dataclass
class RandomCapitalize(Perturbation):
    """Capitalize one random word entirely.

    Example: "How do I cook pasta?" → "How do I COOK pasta?"
    Models: caps lock accidentally on for one word
    """
    name: str = "formatting.random_capitalize"
    meta_class: str = "formatting"
    description: str = "Capitalize one random word (accidental caps lock)"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        words = text.split(" ")
        candidates = [i for i, w in enumerate(words) if w != w.upper() and any(c.isalpha() for c in w)]
        if not candidates:
            return None
        idx = candidates[rng.integers(len(candidates))]
        words[idx] = words[idx].upper()
        return " ".join(words)


# Register all
register(LowercaseAll())
register(NoCapitalizeFirst())
register(RandomCapitalize())
