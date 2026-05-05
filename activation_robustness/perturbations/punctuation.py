"""Punctuation perturbations — intentional punctuation that differs from original."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Perturbation
from .registry import register


@dataclass
class QuestionToPeriod(Perturbation):
    """Replace trailing ? with .

    Example: "How do I cook?" → "How do I cook."
    Models: user typed a statement ending instead of a question ending
    """
    name: str = "punctuation.question_to_period"
    meta_class: str = "punctuation"
    description: str = "Replace trailing ? with ."

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        stripped = text.rstrip()
        if not stripped.endswith("?"):
            return None
        return stripped[:-1] + "."


# Register all
register(QuestionToPeriod())
