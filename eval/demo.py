"""Quick smoke test + usage examples.

Run:
    cd e:/sn44
    python -m eval.demo
"""

from __future__ import annotations

from eval import (
    FramePrediction,
    PRIVATE_FRAME_RATE,
    score,
    score_with_breakdown,
    expected_value_per_class,
    optimize_class_thresholds,
)


def example_perfect():
    gt = [
        FramePrediction(frame=25, action="pass"),       # t=1.0s
        FramePrediction(frame=125, action="shot"),      # t=5.0s
        FramePrediction(frame=200, action="goal"),      # t=8.0s
    ]
    preds = [FramePrediction(frame=g.frame, action=g.action) for g in gt]
    s = score(preds, gt)
    print(f"perfect match -> {s:.6f} (expected 1.0)")


def example_one_false_positive_goal():
    """One spurious goal prediction costs the FULL goal weight (10.9)."""
    gt = [
        FramePrediction(frame=25, action="pass"),
        FramePrediction(frame=125, action="shot"),
    ]
    preds = [
        FramePrediction(frame=25, action="pass"),
        FramePrediction(frame=125, action="shot"),
        FramePrediction(frame=500, action="goal"),  # spurious
    ]
    s, breakdown = score_with_breakdown(preds, gt)
    # GT weight = 1.0 + 4.7 = 5.7. Matched = 5.7. FP penalty = 10.9 (goal).
    # (5.7 - 10.9) / 5.7 = -0.912 → clamped to 0.0
    print(f"+1 false-pos goal -> {s:.6f} (expected 0.0 -- clamped)")
    print(f"  goal contribution: {breakdown.get('goal', {}).get('contribution', 0):+.3f}")


def example_late_prediction_time_decay():
    """A pass predicted at t=0.5s from the GT at t=1.0s loses 50% (min_score=0 for pass)."""
    gt = [FramePrediction(frame=25, action="pass")]
    pred = [FramePrediction(frame=int(25 + 0.5 * PRIVATE_FRAME_RATE), action="pass")]
    s = score(pred, gt)
    print(f"pass off by 0.5s -> {s:.6f} (expected 0.5)")


def example_goal_slow_decay():
    """A goal predicted at t=1.5s from GT (tolerance=3s, min_score=0.5) loses 25%."""
    gt = [FramePrediction(frame=200, action="goal")]
    pred = [FramePrediction(frame=int(200 + 1.5 * PRIVATE_FRAME_RATE), action="goal")]
    s = score(pred, gt)
    # decay = 1 - (1.5/3) * (1 - 0.5) = 1 - 0.5 * 0.5 = 0.75
    # numerator = 10.9 * 0.75 = 8.175, denom = 10.9 → 0.75
    print(f"goal off by 1.5s -> {s:.6f} (expected 0.75)")


def example_expected_value():
    print("\nBreakeven precision per action (avg_decay=1.0):")
    for action in ("pass", "shot", "goal"):
        ev_at_50 = expected_value_per_class(action, precision=0.50)
        ev_at_70 = expected_value_per_class(action, precision=0.70)
        print(f"  {action:5s}  prec=0.50 -> EV={ev_at_50:+.2f}   prec=0.70 -> EV={ev_at_70:+.2f}")
    # Reminder: EV>0 means worth emitting. precision=0.50, avg_decay=1.0 is exactly breakeven.


def example_threshold_optimizer():
    # Tiny synthetic val set: 1 video. Pretend the model emits 4 candidates with
    # different confidences. Optimizer should pick the cutoff that keeps the
    # TP and drops the FP, on a per-class basis.
    gt = [FramePrediction(frame=25, action="pass"), FramePrediction(frame=200, action="goal")]
    preds = [
        FramePrediction(frame=25, action="pass", confidence=0.8),   # TP
        FramePrediction(frame=100, action="pass", confidence=0.3),  # FP (low conf)
        FramePrediction(frame=200, action="goal", confidence=0.9),  # TP
        FramePrediction(frame=300, action="goal", confidence=0.4),  # FP (low conf)
    ]
    thresholds = optimize_class_thresholds([(preds, gt)])
    print("\nLearned thresholds (synthetic):")
    for action, thr in sorted(thresholds.items(), key=lambda kv: -ACTION_WEIGHT.get(kv[0], 0)):
        if thr <= 1.0:
            print(f"  {action:18s} >= {thr:.2f}")


from eval.actions import ACTION_CONFIGS
ACTION_WEIGHT = {a: c.weight for a, c in ACTION_CONFIGS.items()}


if __name__ == "__main__":
    example_perfect()
    example_late_prediction_time_decay()
    example_goal_slow_decay()
    example_one_false_positive_goal()
    example_expected_value()
    example_threshold_optimizer()
