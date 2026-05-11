"""HuggingFace-native VideoMAE (v1) on Kinetics-400.

This is the safest "it just loads" baseline — published by VideoMAE
authors via MCG-NJU/videomae-base-finetuned-kinetics, supported natively
by transformers' VideoMAEForVideoClassification (no trust_remote_code).

400 classes, 16 frames @ 224x224. ~86M params, fits comfortably on a 4060.
Decent zero-shot signal for soccer-related actions if the validator
challenge happens to look like a Kinetics clip.

Use this to validate the model->postprocess->score chain. Real
performance comes from VideoMAEv2 / InternVideo2 (adapters below).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import ActionClassifier, ModelMeta

HF_REPO = "MCG-NJU/videomae-base-finetuned-kinetics"
META = ModelMeta(
    name="videomae_v1_k400",
    hf_repo=HF_REPO,
    params_millions=86.0,
    n_frames=16,
    img_size=224,
    fits_4060_8gb=True,
    note="HF-native VideoMAE-v1, K400 finetuned. Reliable load, modest accuracy.",
)


class VideoMAEv1Kinetics:
    """Wraps `VideoMAEForVideoClassification` in our ActionClassifier protocol."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        # Lazy imports so `from zoo import REGISTRY` is cheap.
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        self.meta = META
        self.device = torch.device(device)
        self.dtype = dtype

        self.processor = VideoMAEImageProcessor.from_pretrained(HF_REPO)
        self.model = (
            VideoMAEForVideoClassification.from_pretrained(HF_REPO, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        # id2label is a dict[int, str]; preserve index order.
        id2label = self.model.config.id2label
        self.num_classes = len(id2label)
        self.class_names = [id2label[i] for i in range(self.num_classes)]

    @torch.inference_mode()
    def predict(self, clips: torch.Tensor) -> torch.Tensor:
        """clips: (B, C, T, H, W) float in [0,1] -> logits (B, 400)."""
        # VideoMAE processor expects list-of-PIL or numpy uint8 frames per video.
        # Skip it for tensor input — model accepts pixel_values directly.
        # Expected shape: (B, T, C, H, W) for the model.
        if clips.dim() != 5:
            raise ValueError(f"clips must be (B, C, T, H, W); got {tuple(clips.shape)}")
        # Convert (B, C, T, H, W) -> (B, T, C, H, W)
        x = clips.permute(0, 2, 1, 3, 4).to(self.device, dtype=self.dtype)
        # Apply ImageNet-style normalization (mean/std from processor).
        mean = torch.tensor(self.processor.image_mean, device=self.device, dtype=self.dtype).view(1, 1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=self.device, dtype=self.dtype).view(1, 1, 3, 1, 1)
        x = (x - mean) / std
        out = self.model(pixel_values=x)
        return out.logits.float()


# Register lazily — actual class is constructed on demand.
def _build(**kw):
    return VideoMAEv1Kinetics(**kw)


from . import REGISTRY  # noqa: E402

REGISTRY[META.name] = _build
