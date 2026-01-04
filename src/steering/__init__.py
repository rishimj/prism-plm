"""Activation Steering module for ESM-2 models.

This module provides tools for deriving steering vectors from protein sequences
and injecting them into the ESM-2 residual stream to causally test features
identified by PRISM.
"""

from .vector import SteeringVector
from .hooks import SteeringHook
from .analysis import (
    compare_hidden_states,
    compare_logits,
    generate_steering_report,
)

__all__ = [
    "SteeringVector",
    "SteeringHook",
    "compare_hidden_states",
    "compare_logits",
    "generate_steering_report",
]

