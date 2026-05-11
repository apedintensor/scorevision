"""SN44 / Football Event local scoring harness.

Mirrors the validator scoring logic in
`turbovision/scorevision/validator/central/private_track/scoring.py` exactly,
without the pydantic/settings/logger overhead. Use this as your training
eval metric, threshold optimizer, and offline simulator.

A `verify_against_official.py` script asserts numeric parity against the
official implementation on randomized inputs.
"""

from .actions import (
    ACTION_CONFIGS,
    ACTION_CLASS_INDEX,
    ACTIONS,
    NUM_ACTION_CLASSES,
    ActionConfig,
)
from .schemas import FramePrediction
from .scoring import (
    PRIVATE_FRAME_RATE,
    calculate_time_decay,
    score,
    score_with_breakdown,
    expected_value_per_class,
    optimize_class_thresholds,
)

__all__ = [
    "ACTION_CONFIGS",
    "ACTION_CLASS_INDEX",
    "ACTIONS",
    "NUM_ACTION_CLASSES",
    "ActionConfig",
    "FramePrediction",
    "PRIVATE_FRAME_RATE",
    "calculate_time_decay",
    "score",
    "score_with_breakdown",
    "expected_value_per_class",
    "optimize_class_thresholds",
]
