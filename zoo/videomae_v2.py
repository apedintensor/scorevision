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

⚠️  Known issue (2026-05): OpenGVLab's `modeling_videomaev2.py` was written
for transformers 4.x and is **not source-compatible with transformers 5.x**.
On 5.x you'll hit cascading errors:
    1. `torch.linspace().item()` fails (meta-device init)  — workaround below
    2. `'VideoMAEv2' has no attribute 'all_tied_weights_keys'` — workaround
    3. `'NoneType' object has no attribute 'keys'` — not yet patched

For Lium runs, pin `transformers==4.45` (or compatible 4.x). The shims
below cover (1) and (2) so the code is ready for the moment 4.x is
installed.
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
        # OpenGVLab's custom modeling_videomaev2.py calls `torch.linspace(...).item()`
        # during __init__. accelerate's init_empty_weights (entered transparently by
        # transformers >= 5.x) sets the default torch device to "meta", which makes
        # `.item()` on a fresh tensor raise. Two-pronged workaround:
        #  1. `low_cpu_mem_usage=False` to ask transformers not to enter meta init,
        #  2. monkey-patch `torch.linspace` to force a CPU device for the duration of
        #     the load — guards against any other accelerate-style context an upgrade
        #     might introduce.
        _orig_linspace = torch.linspace

        def _cpu_linspace(*args, **kwargs):
            kwargs.setdefault("device", "cpu")
            return _orig_linspace(*args, **kwargs)

        torch.linspace = _cpu_linspace
        try:
            # Pull the custom model class via AutoConfig (downloads + registers the
            # module), then monkey-patch the legacy `_tied_weights_keys` to the
            # transformers >= 5.x name `all_tied_weights_keys` before loading.
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)
            module_path = cfg.auto_map["AutoModel"]
            module_name, class_name = module_path.rsplit(".", 1)
            import importlib
            mod = importlib.import_module(
                f"transformers_modules.{hf_repo.replace('/', '.').replace('-', '_hyphen_')}."
                + module_name.replace(".", "_")
            ) if False else None  # noqa: SIM222 — handled by Auto* below
            # Easier: let AutoModel resolve the class but inject the attr after.
            target_cls = type(AutoModel.from_config(cfg, trust_remote_code=True))
            if not hasattr(target_cls, "all_tied_weights_keys"):
                target_cls.all_tied_weights_keys = getattr(target_cls, "_tied_weights_keys", [])
            model = AutoModel.from_pretrained(
                hf_repo,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            )
        finally:
            torch.linspace = _orig_linspace
        self.model = model.to(self.device, dtype=dtype).eval()

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
