"""Model zoo — unified inference interface across pretrained video backbones.

Each adapter (videomae_v2, internvideo2, ...) exposes one of three roles:

    VideoEncoder        — clips → feature tensor
    ActionClassifier    — clips → logits over a fixed taxonomy (K400/K710/SSv2)
    ZeroShotPrompter    — (video_path, prompt) → text response (for VLMs)

The `REGISTRY` below is the single source of truth for which models we
benchmark; add a model by adding one entry. Adapters lazy-load weights so
importing this package is cheap.
"""

from __future__ import annotations

from typing import Callable

REGISTRY: dict[str, Callable[..., object]] = {}
"""Populated by submodule imports. Keys are short names used by CLI/configs."""

__all__ = ["REGISTRY"]
