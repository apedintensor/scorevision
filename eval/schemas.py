from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FramePrediction:
    """One predicted (or ground-truth) action event.

    Mirrors `turbovision/scorevision/utils/schemas.py:FramePrediction` but
    without pydantic — same field names so JSON round-trips work.

    `frame` is the 0-indexed integer frame within the video, sampled at
    `PRIVATE_FRAME_RATE` fps (default 25). Time in seconds = frame / fps.
    `confidence` is unused by the official scorer (it's a greedy 0/1 match
    regardless of confidence) but kept for your training-side reranker /
    threshold optimization.
    """

    frame: int
    action: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.frame < 0:
            raise ValueError(f"frame must be >= 0, got {self.frame}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
