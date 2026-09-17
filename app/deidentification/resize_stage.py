from __future__ import annotations

import cv2
import numpy as np

from .contracts import PlateDetection


def prepare_detection_image(image: np.ndarray, max_side: int = 1600) -> tuple[np.ndarray, float, float]:
    """Create a downscaled image for detection while keeping the original image for final redaction."""
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return image.copy(), 1.0, 1.0

    scale = max_side / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale, scale


def map_plate_detections_to_original(
    detections: list[PlateDetection],
    scale_x: float,
    scale_y: float,
) -> list[PlateDetection]:
    """Scale plate detections from the resized detection image back to the original image space."""
    mapped: list[PlateDetection] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        mapped_box = (
            int(round(x1 / scale_x)),
            int(round(y1 / scale_y)),
            int(round(x2 / scale_x)),
            int(round(y2 / scale_y)),
        )
        mapped.append(
            PlateDetection(
                box=mapped_box,
                text=detection.text,
                confidence=detection.confidence,
                class_name=detection.class_name,
            )
        )
    return mapped


def map_human_mask_to_original(
    mask: np.ndarray,
    original_shape: tuple[int, int],
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Scale a human mask from resized detection space back to original image space."""
    if mask is None:
        return np.zeros(original_shape, dtype=np.uint8)
    resized_shape = mask.shape[:2]
    if resized_shape == original_shape:
        return mask.astype(np.uint8, copy=False)

    target_width = max(1, original_shape[1])
    target_height = max(1, original_shape[0])
    resized_to_original = cv2.resize(mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return resized_to_original


def map_watermark_regions_to_original(
    regions: list[tuple[int, int, int, int]],
    scale_x: float,
    scale_y: float,
) -> list[tuple[int, int, int, int]]:
    """Scale watermark boxes from the detection image back to the original image space."""
    mapped: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in regions:
        mapped.append(
            (
                int(round(x1 / scale_x)),
                int(round(y1 / scale_y)),
                int(round(x2 / scale_x)),
                int(round(y2 / scale_y)),
            )
        )
    return mapped


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