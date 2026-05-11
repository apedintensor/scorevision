"""Football Event scoring — local clean-room reimplementation.

Faithful port of:
  turbovision/scorevision/validator/central/private_track/scoring.py
  :_legacy_score_predictions

Differences from the upstream version:
  - No pydantic / settings / logger dependency (uses plain dataclasses).
  - Adds `score_with_breakdown` returning per-action diagnostics.
  - Adds `expected_value_per_class` and `optimize_class_thresholds` helpers
    used during training / inference threshold tuning.

The numeric output of `score` is asserted bit-equal to the upstream impl
in `verify_against_official.py` on randomized inputs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .actions import ACTION_CONFIGS, ActionConfig
from .schemas import FramePrediction

# Validator's default — see turbovision/scorevision/utils/settings.py:PRIVATE_FRAME_RATE
PRIVATE_FRAME_RATE: int = 25


def _frame_to_seconds(frame: int, fps: int = PRIVATE_FRAME_RATE) -> float:
    return frame / fps


def calculate_time_decay(time_diff: float, tolerance: float, min_score: float) -> float:
    """Linear decay from 1.0 at time_diff=0 to min_score at time_diff=tolerance.

    Mirrors upstream exactly:
        if time_diff > tolerance: return 0.0
        else:                     return 1 - (time_diff/tolerance) * (1 - min_score)
    """
    if time_diff > tolerance:
        return 0.0
    return 1.0 - (time_diff / tolerance) * (1.0 - min_score)


def _find_best_match(
    pred: FramePrediction,
    ground_truth: Sequence[FramePrediction],
    used_indices: set[int],
    fps: int,
) -> tuple[int | None, float]:
    cfg = ACTION_CONFIGS.get(pred.action)
    if cfg is None:
        return None, 0.0

    pred_time = _frame_to_seconds(pred.frame, fps)
    best_idx: int | None = None
    best_decay = 0.0

    for i, gt in enumerate(ground_truth):
        if i in used_indices or gt.action != pred.action:
            continue
        time_diff = abs(pred_time - _frame_to_seconds(gt.frame, fps))
        if time_diff > cfg.tolerance_seconds:
            continue
        decay = calculate_time_decay(time_diff, cfg.tolerance_seconds, cfg.min_score)
        if decay > best_decay:
            best_decay = decay
            best_idx = i

    return best_idx, best_decay


def score(
    predictions: Sequence[FramePrediction],
    ground_truth: Sequence[FramePrediction],
    *,
    fps: int = PRIVATE_FRAME_RATE,
) -> float:
    """Return the validator's scalar score in [0, 1].

    Matches `_legacy_score_predictions` in the official repo. Greedy match:
    predictions are processed in ascending frame order; each prediction
    claims the highest-decay unused GT of the same action class within
    tolerance. Unmatched predictions cost the full action weight (no
    decay); unmatched GTs cost nothing directly (they just reduce the
    achievable numerator).
    """
    if not ground_truth:
        return 0.0

    gt_total_weight = 0.0
    for gt in ground_truth:
        cfg = ACTION_CONFIGS.get(gt.action)
        if cfg is not None:
            gt_total_weight += cfg.weight

    if gt_total_weight == 0.0:
        return 0.0

    sorted_preds = sorted(predictions, key=lambda p: p.frame)

    matched_score = 0.0
    unmatched_penalty = 0.0
    used: set[int] = set()

    for pred in sorted_preds:
        cfg = ACTION_CONFIGS.get(pred.action)
        if cfg is None:
            continue  # unknown action class: silently dropped (upstream behavior)
        match_idx, decay = _find_best_match(pred, ground_truth, used, fps)
        if match_idx is not None:
            used.add(match_idx)
            matched_score += cfg.weight * decay
        else:
            unmatched_penalty += cfg.weight

    return max(0.0, min(1.0, (matched_score - unmatched_penalty) / gt_total_weight))


def score_with_breakdown(
    predictions: Sequence[FramePrediction],
    ground_truth: Sequence[FramePrediction],
    *,
    fps: int = PRIVATE_FRAME_RATE,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Same score, plus a per-class diagnostic dict.

    Returns (final_score, breakdown) where breakdown[action] = {
        "tp": <count>, "fp": <count>, "fn": <count>,
        "gt_weight": <sum gt weight>,
        "matched_decayed_weight": <sum of weight*decay for matched preds>,
        "fp_penalty": <sum of weight for unmatched preds>,
        "contribution": <(matched_decayed - fp_penalty) / total_gt_weight>,
    }
    Useful for debugging which actions are dragging your score down.
    """
    if not ground_truth:
        return 0.0, {}

    gt_total_weight = 0.0
    gt_count_by_action: dict[str, int] = defaultdict(int)
    gt_weight_by_action: dict[str, float] = defaultdict(float)
    for gt in ground_truth:
        cfg = ACTION_CONFIGS.get(gt.action)
        if cfg is None:
            continue
        gt_total_weight += cfg.weight
        gt_count_by_action[gt.action] += 1
        gt_weight_by_action[gt.action] += cfg.weight

    if gt_total_weight == 0.0:
        return 0.0, {}

    sorted_preds = sorted(predictions, key=lambda p: p.frame)

    tp_count: dict[str, int] = defaultdict(int)
    fp_count: dict[str, int] = defaultdict(int)
    matched_decayed: dict[str, float] = defaultdict(float)
    fp_penalty: dict[str, float] = defaultdict(float)
    used: set[int] = set()

    for pred in sorted_preds:
        cfg = ACTION_CONFIGS.get(pred.action)
        if cfg is None:
            continue
        match_idx, decay = _find_best_match(pred, ground_truth, used, fps)
        if match_idx is not None:
            used.add(match_idx)
            tp_count[pred.action] += 1
            matched_decayed[pred.action] += cfg.weight * decay
        else:
            fp_count[pred.action] += 1
            fp_penalty[pred.action] += cfg.weight

    final = max(0.0, min(1.0, (sum(matched_decayed.values()) - sum(fp_penalty.values())) / gt_total_weight))

    breakdown: dict[str, dict[str, float]] = {}
    actions = set(gt_count_by_action) | set(tp_count) | set(fp_count)
    for a in actions:
        fn = max(0, gt_count_by_action[a] - tp_count[a])
        breakdown[a] = {
            "tp": tp_count[a],
            "fp": fp_count[a],
            "fn": fn,
            "gt_count": gt_count_by_action[a],
            "gt_weight": gt_weight_by_action[a],
            "matched_decayed_weight": matched_decayed[a],
            "fp_penalty": fp_penalty[a],
            "contribution": (matched_decayed[a] - fp_penalty[a]) / gt_total_weight,
        }

    return final, breakdown


# ---------------------------------------------------------------------------
# Training-side helpers — not part of the upstream impl.
# ---------------------------------------------------------------------------


def expected_value_per_class(
    action: str,
    precision: float,
    avg_time_decay_when_correct: float = 1.0,
) -> float:
    """Expected score *per emitted prediction* for `action`.

    Given the model's per-class precision (probability that an emitted
    prediction at this confidence threshold is matched to a GT) and the
    average time-decay among true positives, return the expected
    contribution to the *numerator* of `score` per emitted prediction:

        E[Δ] = weight * (precision * avg_decay - (1 - precision))
             = weight * (precision * (avg_decay + 1) - 1)

    Use this for threshold tuning: emit a prediction only if its
    estimated precision exceeds the breakeven point where E[Δ] = 0:

        precision_breakeven = 1 / (avg_decay + 1)

    For avg_decay=1.0: breakeven precision = 0.50
    For avg_decay=0.5: breakeven precision = 0.67
    """
    cfg = ACTION_CONFIGS.get(action)
    if cfg is None:
        return 0.0
    return cfg.weight * (precision * (avg_time_decay_when_correct + 1.0) - 1.0)


def optimize_class_thresholds(
    val_predictions_with_scores: Sequence[tuple[Sequence[FramePrediction], Sequence[FramePrediction]]],
    *,
    candidate_thresholds: Sequence[float] = tuple(i / 100.0 for i in range(0, 100, 2)),
    fps: int = PRIVATE_FRAME_RATE,
) -> dict[str, float]:
    """Per-class confidence threshold optimizer.

    Args:
        val_predictions_with_scores: list of (preds, gt) pairs over the val set.
            Each `pred` must carry a meaningful `.confidence`.
        candidate_thresholds: thresholds to grid-search per class.
        fps: frame rate.

    Returns:
        {action: best_threshold} maximizing mean validator score on the val set,
        optimized **independently per class** (greedy — assumes class scores are
        approximately independent, which holds for the official scorer since
        matching is per-class). For tightly-coupled actions you can iterate this
        a few times alternating classes.

    This is the single most useful training-side tool we have: the upstream
    scorer rewards being conservative on high-weight actions and aggressive on
    low-weight actions, and the optimal threshold per class falls out of this
    search directly.
    """
    best_per_class: dict[str, float] = {}
    base_keep = {a: 1.01 for a in ACTION_CONFIGS}  # > all confidences = drop everything for that class

    def _filtered(preds: Sequence[FramePrediction], thr_by_class: dict[str, float]) -> list[FramePrediction]:
        return [p for p in preds if p.confidence >= thr_by_class.get(p.action, 1.01)]

    for action in ACTION_CONFIGS:
        best_thr = 1.01
        best_mean = float("-inf")
        for thr in candidate_thresholds:
            thr_map = {**best_per_class, action: thr}
            # for classes not yet decided, drop them entirely (we'll re-add them in their own pass).
            for other in ACTION_CONFIGS:
                if other not in thr_map:
                    thr_map[other] = 1.01
            total = 0.0
            for preds, gt in val_predictions_with_scores:
                kept = _filtered(preds, thr_map)
                total += score(kept, gt, fps=fps)
            mean = total / max(1, len(val_predictions_with_scores))
            if mean > best_mean:
                best_mean = mean
                best_thr = thr
        best_per_class[action] = best_thr

    return best_per_class
