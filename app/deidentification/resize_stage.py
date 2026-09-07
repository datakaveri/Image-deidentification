from __future__ import annotations

import cv2
import numpy as np


def resize_for_quality(image: np.ndarray, high_thresh: float, low_thresh: float) -> tuple[np.ndarray, str, float]:
    """Resize an in-memory image using the existing quality thresholds."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if score > high_thresh:
        label, size = "HIGH", (1920, 1080)
    elif score >= low_thresh:
        label, size = "MEDIUM", (1280, 720)
    else:
        label, size = "LOW", (640, 480)
    height, width = image.shape[:2]
    interpolation = cv2.INTER_AREA if size[0] < width or size[1] < height else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interpolation), label, score