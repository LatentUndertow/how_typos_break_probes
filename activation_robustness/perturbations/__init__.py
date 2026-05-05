"""Perturbation augmentation kit for activation robustness research.

Usage:
    from activation_robustness.perturbations import get_all, get_by_meta_class, get_family, from_config

    # All perturbations
    all_perturbs = get_all()

    # By meta-class
    typos = get_by_meta_class("typo")

    # By glob pattern
    family = get_family(["typo.*", "punctuation.question_to_*"])

    # From config dict
    family = from_config({
        "name": "keyboard_errors",
        "include": ["typo.*", "punctuation.*"],
        "exclude": ["typo.wrong_shift"],
    })

    # Apply
    rng = np.random.default_rng(42)
    for p in family.perturbations:
        result = p.apply("How do I cook pasta?", rng)
        if result:
            print(f"{p.name}: {result}")
"""

# Import modules to trigger registration
from . import punctuation  # noqa: F401
from . import typo  # noqa: F401
from . import formatting  # noqa: F401
from . import omission  # noqa: F401

# Re-export registry API
from .registry import get_all, get_by_meta_class, get_by_name, get_family, from_config
from .base import Perturbation, PerturbationFamily

__all__ = [
    "Perturbation",
    "PerturbationFamily",
    "get_all",
    "get_by_meta_class",
    "get_by_name",
    "get_family",
    "from_config",
]
