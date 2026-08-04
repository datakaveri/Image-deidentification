"""
image.py

Shared image utilities for MedDeID.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_images(
    image,
) -> list[np.ndarray]:
    """
    Normalize artifact image representation.

    Supports:

        ndarray
        frame stacks
        list[pages]
    """

    if image is None:

        return []

    if isinstance(
        image,
        list,
    ):

        return image

    if isinstance(image, np.ndarray):

        if image.ndim == 4:
            # (frames, H, W) or (H, W, frames, channels) — split on axis 0
            return [image[i] for i in range(image.shape[0])]

        if image.ndim == 3:
            c = image.shape[2]
            if c not in (1, 3, 4):
                # NIfTI volume (H, W, D) — return each depth slice as 2D
                return [image[:, :, i] for i in range(c)]

        return [image]

    return []


def to_uint8(
    image: np.ndarray,
) -> np.ndarray:
    """
    Safe uint8 conversion.

    Supports:

        uint16
        float
        medical volumes
    """

    if image.dtype == np.uint8:

        return image

    norm = cv2.normalize(

        image,

        None,

        0,

        255,

        cv2.NORM_MINMAX,
    )

    return norm.astype(
        np.uint8
    )


def to_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert RGB/BGR/grayscale → single-channel uint8 grayscale.

    Handles:
        2D (H, W)          — already grayscale, return as-is
        3D (H, W, 1)       — squeeze and return
        3D (H, W, 3/4)     — BGR or BGRA, convert with cv2
        3D (H, W, N) N>4   — NIfTI/volume slice, take mean across channels
    """

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        c = image.shape[2]
        if c == 1:
            return image[:, :, 0]
        if c in (3, 4):
            code = cv2.COLOR_BGR2GRAY if c == 3 else cv2.COLOR_BGRA2GRAY
            return cv2.cvtColor(image, code)
        # Unusual channel count (e.g. NIfTI slice with many frames)
        return image.mean(axis=2).astype(np.uint8)

    return image


def preprocess_for_ocr(
    image: np.ndarray,
) -> np.ndarray:
    """
    OCR preprocessing pipeline.
    """

    image = to_uint8(
        image
    )

    image = to_grayscale(
        image
    )

    image = cv2.equalizeHist(
        image
    )

    return image


def clip_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
):
    """
    Clamp bbox to image bounds.
    """

    return (

        max(0, x1),

        max(0, y1),

        min(width, x2),

        min(height, y2),
    )