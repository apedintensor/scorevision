"""Frozen copy of action weights / tolerances / min_scores.

Mirrors `turbovision/scorevision/utils/actions.py`. If Score updates the
weights upstream, regenerate this file by re-reading the source of truth.
Last synced: 2026-05-11 against turbovision @ 05d5ef3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionConfig:
    weight: float
    min_score: float
    tolerance_seconds: float


ACTIONS: tuple[str, ...] = (
    "pass",
    "pass_received",
    "recovery",
    "tackle",
    "interception",
    "ball_out_of_play",
    "clearance",
    "take_on",
    "substitution",
    "block",
    "aerial_duel",
    "shot",
    "save",
    "foul",
    "goal",
)

ACTION_CONFIGS: dict[str, ActionConfig] = {
    "pass":             ActionConfig(weight=1.0,  min_score=0.0, tolerance_seconds=1.0),
    "pass_received":    ActionConfig(weight=1.4,  min_score=0.0, tolerance_seconds=1.0),
    "recovery":         ActionConfig(weight=1.5,  min_score=0.0, tolerance_seconds=1.5),
    "tackle":           ActionConfig(weight=2.5,  min_score=0.1, tolerance_seconds=1.5),
    "interception":     ActionConfig(weight=2.8,  min_score=0.5, tolerance_seconds=2.0),
    "ball_out_of_play": ActionConfig(weight=2.9,  min_score=0.5, tolerance_seconds=2.0),
    "clearance":        ActionConfig(weight=3.1,  min_score=0.5, tolerance_seconds=2.0),
    "take_on":          ActionConfig(weight=3.2,  min_score=0.5, tolerance_seconds=2.0),
    "substitution":     ActionConfig(weight=4.2,  min_score=0.5, tolerance_seconds=2.0),
    "block":            ActionConfig(weight=4.2,  min_score=0.5, tolerance_seconds=2.0),
    "aerial_duel":      ActionConfig(weight=4.3,  min_score=0.5, tolerance_seconds=2.0),
    "shot":             ActionConfig(weight=4.7,  min_score=0.5, tolerance_seconds=2.0),
    "save":             ActionConfig(weight=7.3,  min_score=0.5, tolerance_seconds=2.0),
    "foul":             ActionConfig(weight=7.7,  min_score=0.5, tolerance_seconds=2.5),
    "goal":             ActionConfig(weight=10.9, min_score=0.5, tolerance_seconds=3.0),
}

assert tuple(ACTION_CONFIGS.keys()) == ACTIONS, "ACTIONS / ACTION_CONFIGS drift"

ACTION_CLASS_INDEX: dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
NUM_ACTION_CLASSES: int = len(ACTIONS)
