from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags


def save_with_geo_exif(image: np.ndarray, source_path: Path, output_path: Path, strip_exif: bool = True) -> None:
    """Save a BGR image once, retaining only GPS metadata when requested."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    if strip_exif:
        with Image.open(source_path) as source:
            source_exif = source.getexif()
        exif = Image.Exif()
        gps_tag = next((key for key, name in ExifTags.TAGS.items() if name == "GPSInfo"), None)
        if gps_tag is not None and gps_tag in source_exif:
            exif[gps_tag] = source_exif[gps_tag]
        pil_image.save(output_path, exif=exif, quality=95, optimize=True)
    else:
        pil_image.save(output_path, quality=95, optimize=True)