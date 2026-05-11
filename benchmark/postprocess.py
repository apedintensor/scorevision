"""Generic logits -> FramePrediction list.

Strategy: sliding-window inference over a video. For each clip, take the
softmax over the model's pretrained taxonomy, restrict to logits that map
to one of Score's 15 actions (via `zoo.class_map`), and emit a
FramePrediction at the clip's temporal center if the per-action softmax
exceeds a (per-action) threshold.

This is the simplest baseline. A better version would do temporal NMS,
peak-finding within softmax trajectories, or learn a proper localization
head. We start with this to verify the chain.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from eval.actions import ACTIONS
from eval.schemas import FramePrediction
from zoo.class_map import k_label_relevance_mask


def softmax_to_predictions(
    *,
    logits: torch.Tensor,            # (n_clips, num_classes)
    clip_centers_frames: list[int],  # absolute frame number at each clip center, len == n_clips
    class_names: list[str],          # pretrained taxonomy names (len == num_classes)
    confidence_threshold: float | dict[str, float] = 0.05,
) -> list[FramePrediction]:
    """Project model logits onto Score's 15-class action set."""
    assert logits.shape[0] == len(clip_centers_frames)
    assert logits.shape[1] == len(class_names)

    keep_idx, mapped_actions = k_label_relevance_mask(class_names)
    if not keep_idx:
        return []

    # Per-class threshold dict.
    if isinstance(confidence_threshold, (int, float)):
        thr = {a: float(confidence_threshold) for a in ACTIONS}
    else:
        thr = {a: float(confidence_threshold.get(a, 0.05)) for a in ACTIONS}

    probs = torch.softmax(logits, dim=-1)  # (n_clips, num_classes)
    # Restrict to mappable classes: (n_clips, K)
    sub = probs[:, keep_idx]

    # For each clip, emit one prediction per mappable action class that exceeds threshold.
    # (Could also enforce "one action per clip" via argmax — left as a parameter elsewhere.)
    out: list[FramePrediction] = []
    for clip_idx, center_frame in enumerate(clip_centers_frames):
        for k, action in enumerate(mapped_actions):
            p = float(sub[clip_idx, k])
            if p >= thr.get(action, 1.0):
                out.append(FramePrediction(frame=int(center_frame), action=action, confidence=p))
    return out


def dedup_predictions(
    preds: list[FramePrediction],
    *,
    same_action_window_frames: int = 25,
) -> list[FramePrediction]:
    """Greedy temporal NMS: when multiple predictions of the same action fall
    within `same_action_window_frames`, keep only the highest-confidence one.
    Default window = 1s @ 25fps.
    """
    by_action: dict[str, list[FramePrediction]] = defaultdict(list)
    for p in sorted(preds, key=lambda p: -p.confidence):
        kept = by_action[p.action]
        if any(abs(p.frame - k.frame) <= same_action_window_frames for k in kept):
            continue
        kept.append(p)
    out: list[FramePrediction] = []
    for plist in by_action.values():
        out.extend(plist)
    out.sort(key=lambda p: p.frame)
    return out
