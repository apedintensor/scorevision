"""Zero-shot benchmark runner.

Loads each registered zoo model, runs sliding-window inference over the
synthetic dataset (replace with `data.score_samples` when available),
converts logits -> FramePrediction lists, scores via eval.score, and
prints a comparison table.

Usage:
    cd e:/sn44
    python -m benchmark.run_zero_shot                          # all registered models
    python -m benchmark.run_zero_shot videomae_v1_k400         # subset
    python -m benchmark.run_zero_shot --device cpu --dtype f32 # CPU fallback
"""

from __future__ import annotations

import argparse
import time
import traceback
from typing import Any

import torch

import zoo  # populates REGISTRY via submodule imports below
import zoo.kinetics_classifier   # noqa: F401  (side effect: registers)
import zoo.videomae_v2           # noqa: F401

from data import SyntheticFootballDataset
from eval import score
from benchmark.postprocess import softmax_to_predictions, dedup_predictions


def _device_choices() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def run_model_on_dataset(
    model: Any,
    ds: SyntheticFootballDataset,
    *,
    max_videos: int = 4,
    clip_stride_frames: int = 8,
    confidence_threshold: float = 0.05,
) -> tuple[float, dict[str, Any]]:
    """Returns (mean_video_score, stats)."""
    per_video_scores: list[float] = []
    n_clips_total = 0
    t_inference = 0.0

    for v_idx in range(min(max_videos, len(ds.videos))):
        clips_with_meta = ds.iter_video_clips(video_idx=v_idx, stride_frames=clip_stride_frames)
        if not clips_with_meta:
            continue

        # Stack into a single batch — synthetic data is tiny.
        clips = torch.stack([c for c, _ in clips_with_meta], dim=0)
        centers = [int((m["start_frame"] + m["end_frame"]) / 2) for _, m in clips_with_meta]
        video_gt = clips_with_meta[0][1]["video_gt"]
        n_clips_total += len(clips_with_meta)

        t0 = time.perf_counter()
        logits = model.predict(clips)
        t_inference += time.perf_counter() - t0

        preds = softmax_to_predictions(
            logits=logits,
            clip_centers_frames=centers,
            class_names=model.class_names,
            confidence_threshold=confidence_threshold,
        )
        preds = dedup_predictions(preds, same_action_window_frames=25)
        s = score(preds, video_gt)
        per_video_scores.append(s)

    mean = sum(per_video_scores) / max(1, len(per_video_scores))
    return mean, {
        "n_videos": len(per_video_scores),
        "n_clips_total": n_clips_total,
        "inference_seconds": t_inference,
        "clips_per_second": n_clips_total / max(1e-6, t_inference),
        "per_video": per_video_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="Model names to run (default: all registered).")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--dtype", default=None, help="bf16 / f16 / f32 (default: bf16 on cuda, f32 on cpu)")
    parser.add_argument("--max-videos", type=int, default=4)
    parser.add_argument("--clip-stride", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    auto_device, auto_dtype = _device_choices()
    device = torch.device(args.device) if args.device else auto_device
    if args.dtype:
        dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[args.dtype]
    else:
        dtype = auto_dtype

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Registered models: {list(zoo.REGISTRY.keys())}")

    targets = args.models or list(zoo.REGISTRY.keys())
    ds = SyntheticFootballDataset(
        num_videos=args.max_videos,
        video_duration_seconds=60.0,
        clip_frames=16,
        clips_per_video=4,
        fps=25,
        img_size=224,
        seed=0,
    )

    rows: list[tuple[str, str, float, float, str]] = []
    for name in targets:
        if name not in zoo.REGISTRY:
            print(f"[skip] {name} not in REGISTRY")
            continue
        print(f"\n=== {name} ===")
        try:
            model = zoo.REGISTRY[name](device=device, dtype=dtype)
        except Exception as e:
            print(f"[load failed] {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            rows.append((name, "load_failed", 0.0, 0.0, str(e)[:60]))
            continue

        print(
            f"  loaded: {model.meta.params_millions:.0f}M params, "
            f"{model.num_classes} classes, fits_4060={model.meta.fits_4060_8gb}"
        )
        try:
            mean_score, stats = run_model_on_dataset(
                model, ds,
                max_videos=args.max_videos,
                clip_stride_frames=args.clip_stride,
                confidence_threshold=args.threshold,
            )
            print(f"  mean score: {mean_score:.4f}   clips/s: {stats['clips_per_second']:.1f}")
            rows.append((name, "ok", mean_score, stats["clips_per_second"], ""))
        except Exception as e:
            print(f"[run failed] {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            rows.append((name, "run_failed", 0.0, 0.0, str(e)[:60]))
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n=== Summary ===")
    print(f"{'model':<24} {'status':<12} {'score':>8} {'clips/s':>10}   note")
    print("-" * 78)
    for name, status, sc, cps, note in rows:
        print(f"{name:<24} {status:<12} {sc:>8.4f} {cps:>10.2f}   {note}")


if __name__ == "__main__":
    main()
