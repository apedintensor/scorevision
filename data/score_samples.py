"""Real Score sample videos from scoredata.me CDN.

Score Technologies hosts unauthenticated sample clips at
`https://scoredata.me/<date>/<hash>/<segment>.mp4`. The MINER.md example
in turbovision points at `2025_03_14/35ae7a/h1_0f2ca0.mp4` as a known-good
URL. We download a small set of these on first use into `data/downloads/`
(gitignored) and decode lazily into (C, T, H, W) clip tensors.

No ground-truth labels — we use these for zero-shot inference smoke
tests, not for training. Empty `video_gt=[]` so `eval.score == 0.0`
unless a model emits 0 predictions (matching the empty GT case).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import requests
import torch
from torch.utils.data import Dataset

from eval.schemas import FramePrediction


# Known-good Score sample URLs (extend as we find more in Discord intel).
DEFAULT_SAMPLE_URLS: list[str] = [
    "https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4",
]


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    headers = {"User-Agent": "Mozilla/5.0 scorevision-zero-shot/0.1"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        os.replace(tmp, dest)
    return dest


def _decode_clip(
    video_path: Path,
    *,
    start_frame: int,
    n_frames: int,
    img_size: int,
) -> tuple[torch.Tensor, int]:
    """Read `n_frames` from a video starting at `start_frame`, resize to
    img_size x img_size. Returns (tensor (C, T, H, W) float in [0,1], native_fps)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 25))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path} at start={start_frame}")
    # Pad if short (e.g. last clip at video end).
    while len(frames) < n_frames:
        frames.append(frames[-1])
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)
    return tensor, fps


class ScoreSamplesDataset(Dataset):
    """Downloads-on-init dataset for Score's hosted sample clips."""

    def __init__(
        self,
        urls: Sequence[str] = DEFAULT_SAMPLE_URLS,
        *,
        cache_dir: Path | str = "data/downloads",
        clip_frames: int = 16,
        img_size: int = 224,
        clips_per_video: int = 4,
    ) -> None:
        self.clip_frames = clip_frames
        self.img_size = img_size
        self.clips_per_video = clips_per_video
        self.cache_dir = Path(cache_dir)

        self.videos: list[dict] = []
        for url in urls:
            fname = url.rsplit("/", 1)[-1]
            path = _download(url, self.cache_dir / fname)
            cap = cv2.VideoCapture(str(path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 25))
            cap.release()
            self.videos.append(
                {"url": url, "path": path, "n_frames": n_frames, "fps": fps}
            )

        # Pre-sample uniformly-spaced clip start frames.
        self.clip_index: list[tuple[int, int]] = []
        for v_idx, v in enumerate(self.videos):
            max_start = max(0, v["n_frames"] - clip_frames)
            if max_start == 0:
                self.clip_index.append((v_idx, 0))
                continue
            stride = max(1, max_start // max(1, clips_per_video - 1))
            for k in range(clips_per_video):
                start = min(max_start, k * stride)
                self.clip_index.append((v_idx, start))

    def __len__(self) -> int:
        return len(self.clip_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        v_idx, start = self.clip_index[idx]
        v = self.videos[v_idx]
        clip, fps = _decode_clip(
            v["path"], start_frame=start, n_frames=self.clip_frames, img_size=self.img_size
        )
        meta = {
            "video_id": v["url"],
            "video_n_frames": v["n_frames"],
            "start_frame": start,
            "end_frame": start + self.clip_frames,
            "fps": fps,
            "clip_gt": [],
            "video_gt": [],  # No ground-truth labels for Score sample clips
        }
        return clip, meta

    # Same interface as SyntheticFootballDataset for benchmark/ swap-in.
    def iter_video_clips(self, video_idx: int, stride_frames: int) -> list[tuple[torch.Tensor, dict]]:
        v = self.videos[video_idx]
        out: list[tuple[torch.Tensor, dict]] = []
        for start in range(0, max(1, v["n_frames"] - self.clip_frames + 1), stride_frames):
            clip, fps = _decode_clip(
                v["path"], start_frame=start, n_frames=self.clip_frames, img_size=self.img_size
            )
            meta = {
                "video_id": v["url"],
                "video_n_frames": v["n_frames"],
                "start_frame": start,
                "end_frame": start + self.clip_frames,
                "fps": fps,
                "clip_gt": [],
                "video_gt": [],
            }
            out.append((clip, meta))
        return out

    # For prediction-printing (no GT, so display the top action predictions).
    def describe(self) -> str:
        lines = ["ScoreSamplesDataset:"]
        for v in self.videos:
            duration_s = v["n_frames"] / v["fps"]
            lines.append(f"  {v['url']}  ({v['n_frames']} frames @ {v['fps']}fps ~ {duration_s:.1f}s)")
        return "\n".join(lines)
