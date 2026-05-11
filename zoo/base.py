"""Abstract interfaces shared by every model adapter.

We don't enforce these via runtime ABC checks — Python's structural typing
handles it. The protocols below document the contract and let static type
checkers catch mistakes.

Three roles a zoo entry can play:

  1. VideoEncoder        — clips -> features (for fine-tuning new heads on)
  2. ActionClassifier    — clips -> logits over the model's pretrained taxonomy
                           (K400, K710, SSv2). Use class_map.py to project onto
                           Score's 15-class action set.
  3. ZeroShotPrompter    — (video_path, prompt) -> text  (for VLMs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import torch


@dataclass(slots=True)
class ModelMeta:
    name: str               # short id used in REGISTRY ("videomae_v2_base")
    hf_repo: str
    params_millions: float
    n_frames: int           # how many frames the model expects
    img_size: int           # spatial input size (assumes square)
    fits_4060_8gb: bool
    note: str = ""


@runtime_checkable
class VideoEncoder(Protocol):
    meta: ModelMeta
    device: torch.device
    dtype: torch.dtype

    def encode(self, clips: torch.Tensor) -> torch.Tensor:
        """clips: (B, C, T, H, W) -> features (B, D) or (B, T', D)."""
        ...


@runtime_checkable
class ActionClassifier(Protocol):
    meta: ModelMeta
    device: torch.device
    dtype: torch.dtype
    class_names: list[str]   # the pretrained taxonomy (e.g. K400, K710)
    num_classes: int

    def predict(self, clips: torch.Tensor) -> torch.Tensor:
        """clips: (B, C, T, H, W) -> logits (B, num_classes)."""
        ...


@runtime_checkable
class ZeroShotPrompter(Protocol):
    meta: ModelMeta

    def query(self, video_path: str, prompt: str) -> str:
        """Run a VLM on the video with a text prompt; return raw text response."""
        ...
