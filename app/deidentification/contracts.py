from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlateDetection:
    """A plate box and optional OCR text returned by the plate detector."""

    box: tuple[int, int, int, int]
    text: str = ""
    confidence: float | None = None
    class_name: str | None = None


@dataclass(frozen=True)
class DetectionBundle:
    """Outputs from all enabled detectors for one decoded image."""

    plates: tuple[PlateDetection, ...] = ()
    human_mask: Any = None
    watermark_regions: tuple[tuple[int, int, int, int], ...] = ()