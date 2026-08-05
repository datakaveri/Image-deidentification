"""
core.py

Core schemas and shared data contracts for MedDeID.

These objects are intentionally modality-agnostic and are used across:

    loaders/
    metadata/
    detectors/
    redactors/
    validators/
    api/

Author: MedDeID
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ============================================================
# ENUMS
# ============================================================

class FileFormat(str, Enum):
    """Supported input file formats."""

    DICOM = "dicom"
    NIFTI = "nifti"

    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    TIFF = "tiff"
    BMP = "bmp"

    PDF = "pdf"

    DOCX = "docx"
    TXT = "txt"

    UNKNOWN = "unknown"


class Modality(str, Enum):
    """Medical imaging modalities."""

    MRI = "MRI"
    CT = "CT"

    OCT = "OCT"

    FUNDUS = "FUNDUS"

    ULTRASOUND = "ULTRASOUND"

    XRAY = "XRAY"

    PET = "PET"

    HISTOPATH = "HISTOPATH"

    DOCUMENT = "DOCUMENT"

    UNKNOWN = "UNKNOWN"


class PHISource(str, Enum):
    """Origin of detected PHI."""

    OCR = "ocr"

    DICOM_TAG = "dicom_tag"

    EXIF = "exif"

    NIFTI_HEADER = "nifti_header"

    RULE = "rule"

    MODEL = "model"

    MANUAL = "manual"


class RedactionMethod(str, Enum):
    """How PHI was removed."""

    MASK = "mask"

    CROP = "crop"

    INPAINT = "inpaint"

    METADATA_STRIP = "metadata_strip"

    UID_REGENERATION = "uid_regeneration"


# ============================================================
# OVERLAY / REGION DEFINITIONS
# ============================================================

@dataclass
class BoundingBox:
    """
    Pixel coordinate bounding box.

    Format:
        (x1,y1) ──────
          |           |
          |           |
          ─────── (x2,y2)
    """

    x1: int
    y1: int

    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class OverlayRegion:
    """
    UI overlay / burned-in annotation region.
    """

    bbox: BoundingBox

    text: str | None = None

    confidence: float = 1.0

    source: str = "unknown"


# ============================================================
# PHI ENTITY
# ============================================================

@dataclass
class PHIEntity:
    """
    Universal PHI object.

    Used by:

        OCR detectors
        metadata detectors
        validators
        API outputs
    """

    label: str

    confidence: float

    source: PHISource

    text: str | None = None

    bbox: BoundingBox | None = None

    metadata_key: str | None = None

    metadata_value: str | None = None

    page: int | None = None

    slice_index: int | None = None


# ============================================================
# CORE ARTIFACT OBJECT
# ============================================================

@dataclass
class MedicalArtifact:
    """
    Universal internal representation.

    Every loader returns this.
    """

    filepath: Path

    format: FileFormat

    modality: Modality

    image: np.ndarray | list[np.ndarray] | None

    metadata: dict[str, Any] = field(default_factory=dict)

    overlays: list[OverlayRegion] = field(default_factory=list)

    notes: dict[str, Any] = field(default_factory=dict)

    original_filename: str | None = None

    patient_removed: bool = False

    validated: bool = False


# ============================================================
# REDACTION OUTPUT
# ============================================================

@dataclass
class RedactionReport:
    """
    Output of anonymization step.
    """

    metadata_removed: int = 0

    burnedin_removed: int = 0

    overlays_removed: int = 0

    redaction_methods: list[RedactionMethod] = field(
        default_factory=list
    )

    detected_phi: list[PHIEntity] = field(
        default_factory=list
    )

    residual_phi: list[PHIEntity] = field(
        default_factory=list
    )


# ============================================================
# VALIDATION
# ============================================================

@dataclass
class ValidationResult:
    """
    Post-deidentification validation result.
    """

    passed: bool

    risk_score: float

    residual_phi: list[PHIEntity] = field(
        default_factory=list
    )

    validator_name: str = "default_validator"

    notes: str | None = None


# ============================================================
# PROCESSING CONFIG
# ============================================================

@dataclass
class ProcessingConfig:
    """
    Runtime processing configuration.
    """

    use_ocr: bool = True

    use_metadata_cleaner: bool = True

    use_overlay_detector: bool = True

    validation_enabled: bool = True

    ocr_backend: str = "paddleocr"

    redaction_method: RedactionMethod = (
        RedactionMethod.INPAINT
    )

    confidence_threshold: float = 0.50

    corner_scan_only: bool = False

    save_debug_images: bool = False


# ============================================================
# API RESPONSE OBJECT
# ============================================================

@dataclass
class DeidentificationResult:
    """
    Final pipeline result returned by API / CLI.
    """

    job_id: str

    modality: str

    format: str

    output_path: str

    success: bool

    report: RedactionReport

    validation: ValidationResult | None = None

    message: str | None = None