"""Typo perturbations — realistic keyboard errors."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import Perturbation
from .registry import register

# QWERTY adjacency map (lowercase)
QWERTY_ADJACENT: dict[str, str] = {
    "q": "wa", "w": "qeas", "e": "wrds", "r": "etf", "t": "ryg",
    "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "o",
    "a": "qwsz", "s": "wedxza", "d": "erfcxs", "f": "rtgvcd",
    "g": "tyhbvf", "h": "yujnbg", "j": "uikmnh", "k": "iolmj",
    "l": "opk", "z": "asx", "x": "zsdc", "c": "xdfv",
    "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}

# Shift map for wrong_shift
SHIFT_MAP: dict[str, str] = {
    "1": "!", "2": "@", "3": "#", "4": "$", "5": "%",
    "6": "^", "7": "&", "8": "*", "9": "(", "0": ")",
    "-": "_", "=": "+", "[": "{", "]": "}", "\\": "|",
    ";": ":", "'": '"', ",": "<", ".": ">", "/": "?",
}
SHIFT_MAP_INV = {v: k for k, v in SHIFT_MAP.items()}


def _find_word_chars(text: str) -> list[int]:
    """Find indices of alphabetic characters in the text (skip spaces/punct)."""
    return [i for i, c in enumerate(text) if c.isalpha()]


@dataclass
class AdjacentKey(Perturbation):
    """Replace one character with an adjacent QWERTY key.

    Example: "How do I cook pasta?" → "How do I cook oasta?"
    Models: finger hit the wrong key
    """
    name: str = "typo.adjacent_key"
    meta_class: str = "typo"
    description: str = "Replace one char with adjacent QWERTY key"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        candidates = []
        for i, c in enumerate(text):
            cl = c.lower()
            if cl in QWERTY_ADJACENT:
                candidates.append(i)
        if not candidates:
            return None
        idx = candidates[rng.integers(len(candidates))]
        char = text[idx]
        adj = QWERTY_ADJACENT[char.lower()]
        replacement = adj[rng.integers(len(adj))]
        if char.isupper():
            replacement = replacement.upper()
        return text[:idx] + replacement + text[idx + 1:]

    def enumerate_all(self, text: str) -> list[tuple[str, dict]]:
        """Enumerate ALL single-character adjacent key replacements.

        Returns one variant per (position, adjacent_key) combination.
        For a 20-word prompt this produces ~250-350 variants.
        """
        results = []
        for i, c in enumerate(text):
            cl = c.lower()
            if cl not in QWERTY_ADJACENT:
                continue
            for adj_char in QWERTY_ADJACENT[cl]:
                replacement = adj_char.upper() if c.isupper() else adj_char
                perturbed = text[:i] + replacement + text[i + 1:]
                results.append((perturbed, {
                    "position": i,
                    "original_char": c,
                    "replacement_char": replacement,
                }))
        return results


@dataclass
class DoubleLetter(Perturbation):
    """Duplicate one character.

    Example: "How do I cook pasta?" → "How do I coook pasta?"
    Models: key held too long or double-tap
    """
    name: str = "typo.double_letter"
    meta_class: str = "typo"
    description: str = "Duplicate one character"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        chars = _find_word_chars(text)
        if not chars:
            return None
        idx = chars[rng.integers(len(chars))]
        return text[:idx] + text[idx] + text[idx:]


@dataclass
class MissingLetter(Perturbation):
    """Remove one character.

    Example: "How do I cook pasta?" → "How do I cok pasta?"
    Models: finger missed the key
    """
    name: str = "typo.missing_letter"
    meta_class: str = "typo"
    description: str = "Remove one character"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        chars = _find_word_chars(text)
        if len(chars) < 3:  # don't destroy very short words
            return None
        idx = chars[rng.integers(len(chars))]
        return text[:idx] + text[idx + 1:]


@dataclass
class SwappedLetters(Perturbation):
    """Swap two adjacent characters.

    Example: "How do I cook pasta?" → "How do I cook patsa?"
    Models: fingers hit keys in wrong order
    """
    name: str = "typo.swapped_letters"
    meta_class: str = "typo"
    description: str = "Swap two adjacent characters"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        # Find pairs of adjacent alpha chars
        pairs = []
        for i in range(len(text) - 1):
            if text[i].isalpha() and text[i + 1].isalpha():
                pairs.append(i)
        if not pairs:
            return None
        idx = pairs[rng.integers(len(pairs))]
        return text[:idx] + text[idx + 1] + text[idx] + text[idx + 2:]


@dataclass
class WrongShift(Perturbation):
    """Press or release shift at the wrong time.

    Example: "How do I cook pasta?" → "How do I cook pasta/" (? without shift = /)
    Example: "type 123" → "type !23" (1 with shift = !)
    """
    name: str = "typo.wrong_shift"
    meta_class: str = "typo"
    description: str = "Wrong shift key state (pressed or not pressed)"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        candidates = []
        for i, c in enumerate(text):
            if c in SHIFT_MAP or c in SHIFT_MAP_INV or (c.isalpha() and len(text) > 1):
                candidates.append(i)
        if not candidates:
            return None
        idx = candidates[rng.integers(len(candidates))]
        char = text[idx]
        if char in SHIFT_MAP:
            replacement = SHIFT_MAP[char]
        elif char in SHIFT_MAP_INV:
            replacement = SHIFT_MAP_INV[char]
        elif char.islower():
            replacement = char.upper()
        elif char.isupper():
            replacement = char.lower()
        else:
            return None
        return text[:idx] + replacement + text[idx + 1:]


@dataclass
class QuestionToSlash(Perturbation):
    """Replace trailing ? with /

    Example: "How do I cook?" → "How do I cook/"
    Models: user didn't press shift, / is the unshifted ? key
    """
    name: str = "typo.question_to_slash"
    meta_class: str = "typo"
    description: str = "Replace trailing ? with / (no shift pressed)"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        stripped = text.rstrip()
        if not stripped.endswith("?"):
            return None
        return stripped[:-1] + "/"


@dataclass
class PeriodToComma(Perturbation):
    """Replace trailing . with ,

    Example: "Tell me how." → "Tell me how,"
    Models: adjacent key error (. and , are next to each other)
    """
    name: str = "typo.period_to_comma"
    meta_class: str = "typo"
    description: str = "Replace trailing . with , (adjacent key)"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        stripped = text.rstrip()
        if not stripped.endswith("."):
            return None
        return stripped[:-1] + ","


@dataclass
class DoublePunctuation(Perturbation):
    """Duplicate trailing punctuation.

    Example: "How do I cook?" → "How do I cook??"
    Models: accidental double-tap of punctuation key
    """
    name: str = "typo.double_punctuation"
    meta_class: str = "typo"
    description: str = "Duplicate trailing punctuation (double-tap)"

    def apply(self, text: str, rng: np.random.Generator) -> Optional[str]:
        stripped = text.rstrip()
        if not stripped or stripped[-1] not in ".?!":
            return None
        return stripped + stripped[-1]


# Register all
register(AdjacentKey())
register(DoubleLetter())
register(MissingLetter())
register(SwappedLetters())
register(WrongShift())
register(QuestionToSlash())
register(PeriodToComma())
register(DoublePunctuation())
