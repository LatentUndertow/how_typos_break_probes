"""Probe architectures + parallel trainer for §6/§7 experiments."""
from .architectures import (
    MultiArchProbe,
    ProbeConfig,
)
from .trainer import train_probes_parallel
