from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def detect_human_mask(model: Any, device: Any, image: np.ndarray, dilation_size: int = 5) -> np.ndarray:
    """Return a person mask without modifying the supplied BGR image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with __import__("torch").no_grad():
        output = model(preprocess(pil_image).unsqueeze(0).to(device))["out"][0]
    predictions = output.argmax(0).byte().cpu().numpy()
    mask = (predictions == 15).astype(np.uint8) * 255
    height, width = image.shape[:2]
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    filled = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    if dilation_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        filled = cv2.dilate(filled, kernel, iterations=1)
    return filled


def redact_human_mask(
    image: np.ndarray,
    mask: np.ndarray,
    mask_type: str = "blur",
    blur_kernel: int = 51,
    color_bgr: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Apply a DeepLab mask to a copy of an image."""
    redacted = image.copy()
    selected = mask > 0
    if not np.any(selected):
        return redacted
    if mask_type == "blur":
        if blur_kernel <= 0 or blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")
        blurred = cv2.GaussianBlur(redacted, (blur_kernel, blur_kernel), 0)
        redacted[selected] = blurred[selected]
    elif mask_type == "color":
        redacted[selected] = color_bgr
    else:
        raise ValueError(f"Unsupported human masking type: {mask_type}")
    return redacted