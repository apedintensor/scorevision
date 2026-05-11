"""Pretrained-taxonomy -> Score 15-class action mapping.

Most video models are pretrained on Kinetics-400 / 600 / 700 / 710 — none
of which has class definitions that match Score's football-action set. We
use approximate keyword mappings here: a model's K400 class "shooting goal
(soccer)" maps to "shot", "playing soccer" is dropped (too ambiguous), etc.

Zero-shot performance will be poor with these maps — that's expected. The
purpose is to verify the full pipeline runs end-to-end. Real performance
needs fine-tuning on football-specific data with the proper 15-class head.

Keys are case-insensitive substring matches against the source taxonomy
label, in order; first match wins.
"""

from __future__ import annotations

from eval.actions import ACTIONS

# Tuple of (substring, our_action). Order matters — more specific first.
KINETICS_TO_SCORE: list[tuple[str, str]] = [
    # high-weight specific events first
    ("shooting goal", "shot"),
    ("kicking field goal", "shot"),
    ("scoring goal", "goal"),
    ("celebrating goal", "goal"),
    ("goal celebrate", "goal"),
    ("saving (soccer)", "save"),
    ("goalkeeping", "save"),
    ("goalie save", "save"),
    ("blocking shot", "block"),
    ("heading ball", "aerial_duel"),
    ("header (soccer)", "aerial_duel"),
    ("tackling (rugby)", "tackle"),
    ("tackling (american football)", "tackle"),
    ("tackling (soccer)", "tackle"),
    ("sliding (soccer)", "tackle"),
    ("dribbling soccer", "take_on"),
    ("dribbling basketball", "take_on"),
    ("juggling soccer ball", "pass"),
    ("passing american football", "pass"),
    ("kicking soccer ball", "pass"),
    ("kicking ball", "pass"),
    # Substitution / off-ball events are rare in K-* taxonomies; leave unmapped.
    # Common ambient soccer classes — explicitly drop:
    ("playing soccer", None),
    ("watching soccer", None),
]


def map_kinetics_label(label: str) -> str | None:
    """Project a K400/K600/K700/K710 label onto Score's 15 actions or None."""
    s = label.lower().strip()
    for needle, target in KINETICS_TO_SCORE:
        if needle in s:
            return target
    return None


def k_label_relevance_mask(class_names: list[str]) -> tuple[list[int], list[str]]:
    """Identify which logit indices in a pretrained head correspond to
    SN44-mappable classes.

    Returns:
        keep_indices: ordered list of logit indices that map to a Score action
        mapped_actions: parallel list of Score actions (same length)

    Useful for "argmax over only mappable classes" zero-shot baselines.
    """
    keep_indices: list[int] = []
    mapped_actions: list[str] = []
    for i, name in enumerate(class_names):
        target = map_kinetics_label(name)
        if target is None:
            continue
        if target not in ACTIONS:
            continue  # safety: target must be one of our 15
        keep_indices.append(i)
        mapped_actions.append(target)
    return keep_indices, mapped_actions
