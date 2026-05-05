"""
Type aliases and constants for the activation classifier package.
"""
from typing import List, Literal, Union

# Position strategy for aggregating activations across sequence positions
PositionStrategy = Union[Literal["last", "mean", "max"], int, List[int]]

# Supported hook points for activation extraction
SUPPORTED_HOOK_POINTS = ("hook_resid_pre", "hook_resid_post")

# Default values
DEFAULT_HOOK_POINT = "hook_resid_post"
DEFAULT_POSITION: PositionStrategy = -5
DEFAULT_DTYPE = "bfloat16"
DEFAULT_THRESHOLD = 0.5
