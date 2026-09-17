"""Reusable in-memory detection, redaction, resizing, and save helpers."""

from .contracts import PlateDetection
from .deeplab_stage import detect_human_mask, redact_human_mask
from .exif_stage import extract_geo_exif, save_with_geo_exif
from .plate_stage import detect_plate_boxes, redact_plate_boxes
from .resize_stage import (
    map_human_mask_to_original,
    map_plate_detections_to_original,
    map_watermark_regions_to_original,
    prepare_detection_image,
    resize_for_quality,
)
from .watermark_stage import detect_watermark_regions, redact_watermark_regions

__all__ = [
    "PlateDetection",
    "detect_human_mask",
    "redact_human_mask",
    "save_with_geo_exif",
    "extract_geo_exif",
    "detect_plate_boxes",
    "redact_plate_boxes",
    "prepare_detection_image",
    "map_plate_detections_to_original",
    "map_human_mask_to_original",
    "map_watermark_regions_to_original",
    "resize_for_quality",
    "detect_watermark_regions",
    "redact_watermark_regions",
]