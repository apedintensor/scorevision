"""OpenGVLab VideoMAE-v2 adapter (Base/Large/Huge/giant).

VideoMAE-v2 was retrained on the K710 super-taxonomy (= K400 ∪ K600 ∪
K700, deduped) and provides 4 sizes:

    Base   86M   ✅ 4060
    Large  305M  ✅ 4060 (bf16)
    Huge   633M  ❌ 4060
    giant  1B    ❌ 4060   ← engineering sweet spot per user's reference

All checkpoints use `trust_remote_code=True` (custom_code on HF).

K710 has 710 classes; use `class_map.k_label_relevance_mask` to project
onto Score's 15 actions for zero-shot eval.
"""

from __future__ import annotations

import torch

from .base import ActionClassifier, ModelMeta

# size -> (hf_repo, params_M, fits_4060)
_VARIANTS: dict[str, tuple[str, float, bool]] = {
    "base":  ("OpenGVLab/VideoMAEv2-Base",  86.0,  True),
    "large": ("OpenGVLab/VideoMAEv2-Large", 305.0, True),
    "huge":  ("OpenGVLab/VideoMAEv2-Huge",  633.0, False),
    "giant": ("OpenGVLab/VideoMAEv2-giant", 1000.0, False),
}


class VideoMAEv2Classifier:
    """Wraps an OpenGVLab/VideoMAEv2-* checkpoint."""

    def __init__(
        self,
        *,
        size: str = "base",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if size not in _VARIANTS:
            raise ValueError(f"size must be one of {list(_VARIANTS)}; got {size!r}")
        hf_repo, params_m, fits = _VARIANTS[size]

        self.meta = ModelMeta(
            name=f"videomae_v2_{size}",
            hf_repo=hf_repo,
            params_millions=params_m,
            n_frames=16,
            img_size=224,
            fits_4060_8gb=fits,
            note=f"VideoMAE-v2 {size}, K710 finetuned.",
        )

        from transformers import AutoModel, AutoImageProcessor

        self.device = torch.device(device)
        self.dtype = dtype

        self.processor = AutoImageProcessor.from_pretrained(hf_repo, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(hf_repo, trust_remote_code=True, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )

        # OpenGVLab heads expose id2label on config when finetuned for classification.
        cfg = getattr(self.model, "config", None)
        id2label = getattr(cfg, "id2label", None) if cfg is not None else None
        if id2label:
            self.num_classes = len(id2label)
            self.class_names = [id2label[i] for i in range(self.num_classes)]
        else:
            # Encoder-only — caller should not use predict().
            self.num_classes = 0
            self.class_names = []

    @torch.inference_mode()
    def predict(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.dim() != 5:
            raise ValueError(f"clips must be (B, C, T, H, W); got {tuple(clips.shape)}")
        if self.num_classes == 0:
            raise RuntimeError(
                f"{self.meta.name} is loaded as an encoder; no classification head available."
            )
        x = clips.permute(0, 2, 1, 3, 4).to(self.device, dtype=self.dtype)
        # Use processor's normalization stats.
        mean = torch.tensor(self.processor.image_mean, device=self.device, dtype=self.dtype).view(1, 1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=self.device, dtype=self.dtype).view(1, 1, 3, 1, 1)
        x = (x - mean) / std
        out = self.model(pixel_values=x)
        # Some heads return ImageClassifierOutput, others a tensor; handle both.
        if hasattr(out, "logits"):
            return out.logits.float()
        if isinstance(out, torch.Tensor):
            return out.float()
        raise RuntimeError(f"unexpected output type {type(out)}")


from . import REGISTRY  # noqa: E402

for _size in _VARIANTS:
    def _factory(size=_size):
        def _b(**kw):
            return VideoMAEv2Classifier(size=size, **kw)
        return _b
    REGISTRY[f"videomae_v2_{_size}"] = _factory()
