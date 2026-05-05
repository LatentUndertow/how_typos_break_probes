"""Global perturbation registry with discovery and filtering."""
import fnmatch
from typing import Optional

from .base import Perturbation, PerturbationFamily

_REGISTRY: dict[str, Perturbation] = {}


def register(perturbation: Perturbation) -> Perturbation:
    """Register a perturbation instance. Returns it for decorator-style use."""
    _REGISTRY[perturbation.name] = perturbation
    return perturbation


def get_all() -> list[Perturbation]:
    """All registered perturbations."""
    return list(_REGISTRY.values())


def get_by_meta_class(meta_class: str) -> list[Perturbation]:
    """All perturbations belonging to a meta-class."""
    return [p for p in _REGISTRY.values() if p.meta_class == meta_class]


def get_by_name(name: str) -> Optional[Perturbation]:
    """Get a single perturbation by exact name."""
    return _REGISTRY.get(name)


def get_family(names: list[str]) -> PerturbationFamily:
    """Build a family from a list of names or glob patterns.

    Example:
        get_family(["typo.*", "punctuation.question_to_period"])
    """
    matched = set()
    for pattern in names:
        for name in _REGISTRY:
            if fnmatch.fnmatch(name, pattern):
                matched.add(name)
    perturbations = [_REGISTRY[n] for n in sorted(matched)]
    family_name = "+".join(names) if len(names) <= 3 else f"{len(names)}_patterns"
    return PerturbationFamily(name=family_name, perturbations=perturbations)


def from_config(config: dict) -> PerturbationFamily:
    """Build a family from a config dict.

    Config format:
        {
            "name": "keyboard_typos",
            "include": ["typo.*", "punctuation.question_to_slash"],
            "exclude": ["typo.wrong_shift"]
        }
    """
    include = config.get("include", ["*"])
    exclude = config.get("exclude", [])
    name = config.get("name", "custom")

    # Resolve includes
    matched = set()
    for pattern in include:
        for n in _REGISTRY:
            if fnmatch.fnmatch(n, pattern):
                matched.add(n)

    # Remove excludes
    for pattern in exclude:
        for n in list(matched):
            if fnmatch.fnmatch(n, pattern):
                matched.discard(n)

    perturbations = [_REGISTRY[n] for n in sorted(matched)]
    return PerturbationFamily(name=name, perturbations=perturbations)
