from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np

from .contracts import PlateDetection


def _ocr_plate_text(reader: Any, roi: np.ndarray, reader_lock: Any = None) -> str:
    if reader is None or roi.size == 0:
        return ""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if min(height, width) < 50:
        scale = max(1, 128 // min(height, width))
        gray = cv2.resize(gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    try:
        if reader_lock is None:
            texts = reader.readtext(gray, detail=0, paragraph=False)
        else:
            with reader_lock:
                texts = reader.readtext(gray, detail=0, paragraph=False)
    except Exception:
        return ""
    return " | ".join(text.strip() for text in texts if isinstance(text, str) and text.strip())


def detect_plate_boxes(
    model: Any,
    image: np.ndarray,
    conf: float,
    imgsz: int,
    device: str,
    ocr_reader: Any = None,
    allowed_classes: Iterable[str] | None = None,
    reader_lock: Any = None,
) -> list[PlateDetection]:
    """Detect plates without modifying or writing the supplied image."""
    allowed = {value.strip().lower().replace("_", " ") for value in allowed_classes} if allowed_classes else None
    results = model.predict(source=image, conf=conf, imgsz=imgsz, device=device, verbose=False)
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
    confidences = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
    classes = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else None
    names = getattr(result, "names", None)
    detections: list[PlateDetection] = []

    for index, box in enumerate(xyxy):
        class_name = None
        if classes is not None and names is not None:
            class_name = str(names[int(classes[index])])
            if allowed is not None and class_name.lower().replace("_", " ") not in allowed:
                continue
        coordinates = tuple(int(value) for value in box)
        x1, y1, x2, y2 = coordinates
        text = _ocr_plate_text(ocr_reader, image[y1:y2, x1:x2], reader_lock)
        confidence = float(confidences[index]) if confidences is not None else None
        detections.append(PlateDetection(coordinates, text, confidence, class_name))

    return detections


def redact_plate_boxes(image: np.ndarray, detections: Iterable[PlateDetection], mode: str = "black") -> np.ndarray:
    """Apply plate redactions to a copy of an image and return that copy."""
    redacted = image.copy()
    height, width = redacted.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if mode == "black":
            redacted[y1:y2, x1:x2] = 0
        elif mode == "blur":
            roi = redacted[y1:y2, x1:x2]
            kernel = max(3, (min(max(1, (x2 - x1) // 10), 99) // 2) * 2 + 1)
            redacted[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        elif mode == "pixelate":
            roi = redacted[y1:y2, x1:x2]
            small = cv2.resize(roi, (max(1, (x2 - x1) // 10), max(1, (y2 - y1) // 10)))
            redacted[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError(f"Unsupported plate masking mode: {mode}")
    return redacted