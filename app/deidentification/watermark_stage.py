from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np

from app.watermark_removal.masking import redact_pixels


def merge_watermark_boxes(bboxes: Iterable[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    boxes = sorted(bboxes, key=lambda box: (box[1], box[0]))
    merged: list[list[int]] = []
    for box in boxes:
        if not merged:
            merged.append(list(box))
            continue
        last = merged[-1]
        x1, y1, x2, y2 = box
        overlap_y = min(last[3], y2) - max(last[1], y1)
        overlap_x = min(last[2], x2) - max(last[0], x1)
        close = x1 <= last[2] + 15 and last[0] <= x2 + 15 and y1 <= last[3] + 10 and last[1] <= y2 + 10
        if close and (overlap_y >= -10 or overlap_x >= -10):
            last[0], last[1] = min(last[0], x1), min(last[1], y1)
            last[2], last[3] = max(last[2], x2), max(last[3], y2)
        else:
            merged.append(list(box))
    return [tuple(box) for box in merged]


def detect_watermark_regions(
    image: np.ndarray,
    reader: Any,
    border_fraction: float = 0.15,
    padding_fraction: float = 0.03,
) -> list[tuple[int, int, int, int]]:
    """Detect border text using a caller-owned EasyOCR reader."""
    height, width = image.shape[:2]
    band_h = max(24, int(height * border_fraction))
    band_w = max(24, int(width * border_fraction))
    slices = [(0, 0, width, band_h), (0, height - band_h, width, height), (0, 0, band_w, height), (width - band_w, 0, width, height)]
    padding = max(8, int(min(height, width) * padding_fraction))
    regions = []
    for x1, y1, x2, y2 in slices:
        results = reader.readtext(image[y1:y2, x1:x2])
        for box, _, _ in results:
            bx1, by1 = int(box[0][0]), int(box[0][1])
            bx2, by2 = int(box[2][0]), int(box[2][1])
            regions.append((max(0, x1 + min(bx1, bx2) - padding), max(0, y1 + min(by1, by2) - padding), min(width, x1 + max(bx1, bx2) + padding), min(height, y1 + max(by1, by2) + padding)))
    return merge_watermark_boxes(regions)


def redact_watermark_regions(image: np.ndarray, regions: Iterable[tuple[int, int, int, int]]) -> np.ndarray:
    """Redact watermark regions without writing files."""
    phi_regions = [{"bbox": region, "zone": "anatomy"} for region in regions]
    return redact_pixels(image, phi_regions, ds=None)[0]