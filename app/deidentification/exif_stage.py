from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags


def extract_geo_exif(source_path: Path) -> Image.Exif | None:
    """Read GPS metadata before image detection and return a GPS-only EXIF object."""
    with Image.open(source_path) as source:
        source_exif = source.getexif()
    gps_tag = next((key for key, name in ExifTags.TAGS.items() if name == "GPSInfo"), None)
    if gps_tag is None or gps_tag not in source_exif:
        return None
    exif = Image.Exif()
    exif[gps_tag] = source_exif[gps_tag]
    return exif


def save_with_geo_exif(
    image: np.ndarray,
    output_path: Path,
    strip_exif: bool = True,
    geo_exif: Image.Exif | None = None,
) -> None:
    """Save a BGR image once using GPS metadata captured before detection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    if strip_exif:
        pil_image.save(output_path, exif=geo_exif or Image.Exif(), quality=95, optimize=True)
    else:
        pil_image.save(output_path, quality=95, optimize=True)