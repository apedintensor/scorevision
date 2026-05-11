"""Dataset adapters — synthetic + real, same interface for swap-in.

Each adapter yields:
    clip:     torch.Tensor (C, T, H, W) float32, range [0, 1] or normalized
    meta:     dict with keys:
        video_id:      int | str
        video_n_frames: int
        start_frame:   int (within the source video)
        end_frame:     int
        fps:           int
        clip_gt:       list[FramePrediction]  (events inside this clip, frames relative to clip start)
        video_gt:      list[FramePrediction]  (full-video GT, absolute frames)

Adapters present so far:
    synthetic   — random tensors, for pipeline smoke-testing on 4060
    score_samples — TODO: real Score-team sample videos
    soccernet   — TODO: SoccerNet-v2 / v3 (gated, requires NDA)
"""

from .label_map import SOCCERNET_TO_SCORE
from .synthetic import SyntheticFootballDataset

__all__ = ["SyntheticFootballDataset", "SOCCERNET_TO_SCORE"]
