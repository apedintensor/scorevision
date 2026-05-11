"""Synthetic football-clip dataset.

Generates random tensors of shape (C, T, H, W) and synthetic GT events
in the FramePrediction format. Use this to verify the pipeline runs
end-to-end without waiting on SoccerNet access.

Each "video" is logically `video_duration_seconds` long. The dataset
yields short clips sampled from random positions in the video; each
clip carries the GT events that fall within its temporal extent.

Usage:
    ds = SyntheticFootballDataset(num_videos=4, clip_frames=32, fps=25)
    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate)
    for batch in loader:
        clips, clip_meta, gt_events_per_video = batch
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch.utils.data import Dataset

from eval.actions import ACTIONS, ACTION_CONFIGS
from eval.schemas import FramePrediction


# How likely each class is to appear in a random video (very rough match
# to football statistics — pass dominates, goals are rare).
DEFAULT_CLASS_RATES_PER_MINUTE: dict[str, float] = {
    "pass": 60.0,
    "pass_received": 50.0,
    "recovery": 8.0,
    "tackle": 4.0,
    "interception": 3.0,
    "ball_out_of_play": 5.0,
    "clearance": 2.0,
    "take_on": 4.0,
    "substitution": 0.2,
    "block": 1.0,
    "aerial_duel": 3.0,
    "shot": 1.5,
    "save": 0.8,
    "foul": 1.5,
    "goal": 0.05,
}


@dataclass
class SyntheticVideo:
    video_id: int
    duration_seconds: float
    fps: int
    gt: list[FramePrediction] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.duration_seconds * self.fps)


def _sample_events(
    duration_s: float,
    fps: int,
    rng: random.Random,
    class_rates: dict[str, float] = DEFAULT_CLASS_RATES_PER_MINUTE,
) -> list[FramePrediction]:
    events: list[FramePrediction] = []
    minutes = duration_s / 60.0
    for action, rate_per_min in class_rates.items():
        n = max(0, int(round(rng.gauss(rate_per_min * minutes, max(1.0, rate_per_min * minutes * 0.3)))))
        for _ in range(n):
            t = rng.uniform(0, duration_s)
            events.append(FramePrediction(frame=int(t * fps), action=action))
    events.sort(key=lambda e: e.frame)
    return events


class SyntheticFootballDataset(Dataset):
    def __init__(
        self,
        *,
        num_videos: int = 16,
        video_duration_seconds: float = 90.0,
        clip_frames: int = 32,
        clips_per_video: int = 4,
        fps: int = 25,
        img_size: int = 112,
        seed: int = 0,
    ) -> None:
        self.fps = fps
        self.clip_frames = clip_frames
        self.img_size = img_size
        self.clips_per_video = clips_per_video

        rng = random.Random(seed)
        self.videos: list[SyntheticVideo] = []
        for i in range(num_videos):
            v = SyntheticVideo(video_id=i, duration_seconds=video_duration_seconds, fps=fps)
            v.gt = _sample_events(v.duration_seconds, fps, rng)
            self.videos.append(v)

        # Pre-sample clip positions per video so __getitem__ is deterministic.
        rng2 = random.Random(seed + 1)
        self.clip_index: list[tuple[int, int]] = []  # (video_idx, start_frame)
        for v_idx, v in enumerate(self.videos):
            max_start = v.n_frames - clip_frames
            for _ in range(clips_per_video):
                start = rng2.randint(0, max(0, max_start))
                self.clip_index.append((v_idx, start))

    def __len__(self) -> int:
        return len(self.clip_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        v_idx, start_frame = self.clip_index[idx]
        v = self.videos[v_idx]
        # Random tensor stands in for actual frames. Shape (C, T, H, W).
        clip = torch.randn(3, self.clip_frames, self.img_size, self.img_size, dtype=torch.float32)
        end_frame = start_frame + self.clip_frames
        clip_gt = [
            FramePrediction(frame=e.frame - start_frame, action=e.action)
            for e in v.gt
            if start_frame <= e.frame < end_frame
        ]
        meta = {
            "video_id": v.video_id,
            "video_n_frames": v.n_frames,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "fps": self.fps,
            "clip_gt": clip_gt,            # GT events INSIDE this clip (frames relative to clip start)
            "video_gt": v.gt,              # Full-video GT (absolute frames) — for video-level eval
        }
        return clip, meta

    @staticmethod
    def collate(batch: Sequence[tuple[torch.Tensor, dict]]):
        clips = torch.stack([b[0] for b in batch], dim=0)
        metas = [b[1] for b in batch]
        return clips, metas

    # Convenience: enumerate one clip per (video, position) for full-video eval.
    def iter_video_clips(self, video_idx: int, stride_frames: int) -> list[tuple[torch.Tensor, dict]]:
        v = self.videos[video_idx]
        out: list[tuple[torch.Tensor, dict]] = []
        for start in range(0, max(1, v.n_frames - self.clip_frames + 1), stride_frames):
            clip = torch.randn(3, self.clip_frames, self.img_size, self.img_size, dtype=torch.float32)
            end = start + self.clip_frames
            clip_gt = [
                FramePrediction(frame=e.frame - start, action=e.action)
                for e in v.gt
                if start <= e.frame < end
            ]
            meta = {
                "video_id": v.video_id,
                "video_n_frames": v.n_frames,
                "start_frame": start,
                "end_frame": end,
                "fps": self.fps,
                "clip_gt": clip_gt,
                "video_gt": v.gt,
            }
            out.append((clip, meta))
        return out
