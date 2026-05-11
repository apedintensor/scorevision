"""Zero-shot model comparison harness.

Loads each registered zoo model, runs inference on a fixed eval set,
converts outputs to FramePrediction lists, scores via `eval.score`,
and tabulates speed / VRAM / Score.
"""
