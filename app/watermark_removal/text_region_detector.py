"""
text_region_detector.py

Modality-aware Connected-Component Text Region Detector for MedDeID.

Finds burned-in patient overlay text by SHAPE, not by reading it.

Strategy — two modes, chosen automatically per image:
───────────────────────────────────────────────────
NARROW  (fundus / OCT / ophthalmic colour images)
    Scan only the first _FUNDUS_STRIP_MAX pixels of height (5 %, ≤ 100 px).
    This stays entirely within the pure-black circular aperture border where
    patient text is always placed, and never enters the orange/red retinal
    tissue whose bright blobs would produce false positives.
    Only the TOP strip is scanned; PHI text never appears at the bottom of
    ophthalmic images.

WIDE    (US / CT / MRI / XRAY / grayscale images)
    Scan full-width TOP and BOTTOM strips (_WIDE_STRIP_FRAC = 15 %).
    Ultrasound machines burn a patient-name header across the entire width
    of the image (not just corners), so only a full-width strip catches it.

Mode selection
──────────────
1. artifact.modality == ULTRASOUND / CT / MRI / XRAY / PET → WIDE
2. artifact.modality == FUNDUS / OCT / HISTOPATH             → NARROW
3. modality == UNKNOWN and the image is colour               → NARROW
   (fundus/ophthalmic images are always orange/red RGB;
    US/CT/MRI are always grayscale → mean |R−B| discriminates perfectly)
4. everything else                                           → WIDE
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

try:
    from .core import (
        BoundingBox,
        MedicalArtifact,
        Modality,
        PHIEntity,
        PHISource,
    )

    from .image import (
        normalize_images,
        to_grayscale,
        to_uint8,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from core import (
        BoundingBox,
        MedicalArtifact,
        Modality,
        PHIEntity,
        PHISource,
    )

    from image import (
        normalize_images,
        to_grayscale,
        to_uint8,
    )

logger = logging.getLogger(__name__)

# ── Upscale for component analysis ───────────────────────────────────────────
_SCALE = 2

# ── Character blob size limits (original-image pixels) ───────────────────────
_MIN_CHAR_H    = 2
_MAX_CHAR_H    = 80
_MIN_CHAR_W    = 1
_MAX_CHAR_W    = 200
_MIN_CHAR_AREA = 2

# ── Line-grouping tolerance (original pixels) ─────────────────────────────────
_LINE_Y_TOLERANCE = 8

# ── Minimum blobs per line ────────────────────────────────────────────────────
_MIN_BLOBS_PER_LINE = 1

# ── Padding around each returned line bbox ────────────────────────────────────
_PAD_X = 8
_PAD_Y = 3

# ── Strip-level minimum peak (whole band must have at least one bright pixel) ──
_MIN_PEAK = 120

# ── Per-line minimum peak ─────────────────────────────────────────────────────
# Fundus tissue blobs: max ≈ 80–101.  Real text overlays: max ≥ 120.
# This is the primary guard against tissue false-positives inside the top strip.
_MIN_LINE_PEAK = 120

# ── Background mean upper bound ───────────────────────────────────────────────
_MAX_BG_MEAN = 80

# ── Brightness threshold for tight x-bound detection ─────────────────────────
# Defines the horizontal extent of a text line: columns whose brightest pixel
# exceeds this are treated as containing a glyph.
# MUST be above fundus-tissue brightness (orange retina near the aperture edge
# is ~80–120) or the box stretches across the whole image as tissue bleeds in.
# Real overlay text glyphs are always ≥130 (white/near-white) on every
# modality — fundus (~200–255), US header (~180), US bottom labels (255) —
# so 130 cleanly separates true glyphs from tissue.  _PAD_X compensates for
# any anti-aliased edge trimming.
_BRIGHT_THRESHOLD = 130

# ── NARROW mode: fundus / ophthalmic colour images ───────────────────────────
_FUNDUS_STRIP_FRAC = 0.05   # 5 % of image height
_FUNDUS_STRIP_MAX  = 100    # absolute cap in pixels

# ── WIDE mode: US / CT / MRI / grayscale ─────────────────────────────────────
_WIDE_STRIP_FRAC = 0.15     # 15 % of image height

# ── Colour detection: mean |R−B| above this → colour → NARROW mode ───────────
_COLOR_DIFF_THRESHOLD = 20

# ── Modality sets ─────────────────────────────────────────────────────────────
_WIDE_MODALITIES = {
    Modality.ULTRASOUND,
    Modality.CT,
    Modality.MRI,
    Modality.XRAY,
    Modality.PET,
    Modality.DOCUMENT,
}

_NARROW_MODALITIES = {
    Modality.FUNDUS,
    Modality.OCT,
    Modality.HISTOPATH,
}


class TextRegionDetector:
    """
    Detects burned-in text regions along image edges.

    Returns PHIEntity objects with bboxes in full-image coordinates.
    The text field is always None — we detect WHERE text is, not WHAT it says.
    """

    def detect(
        self,
        artifact: MedicalArtifact,
    ) -> list[PHIEntity]:

        if artifact.image is None:
            return []

        images   = normalize_images(artifact.image)
        modality = getattr(artifact, "modality", Modality.UNKNOWN)
        entities: list[PHIEntity] = []

        for img in images:

            wide = self._use_wide_strategy(img, modality)

            for band, ox, oy in self._edge_bands(img, wide):

                for raw_bbox in self._detect_text_lines(band):

                    entities.append(
                        PHIEntity(
                            label="BURNED_IN_TEXT",
                            confidence=0.95,
                            source=PHISource.RULE,
                            text=None,
                            bbox=BoundingBox(
                                x1=int(max(0, raw_bbox.x1 + ox)),
                                y1=int(max(0, raw_bbox.y1 + oy)),
                                x2=int(raw_bbox.x2 + ox),
                                y2=int(raw_bbox.y2 + oy),
                            ),
                        )
                    )

        logger.debug(
            "[TextRegionDetector] %d text-line regions found (wide=%s)",
            len(entities), wide if images else "n/a",
        )

        return entities

    def detect_bottom_watermark(
        self,
        image: np.ndarray | None,
    ) -> list[PHIEntity]:
        """
        Detect burned-in text along the bottom edge of an image.
        This is useful for camera/device watermarks that appear near the base.
        """

        if image is None:
            return []

        images = normalize_images(image)
        entities: list[PHIEntity] = []

        for img in images:
            h, w = img.shape[:2]
            band_h = max(24, int(h * 0.15))
            band_h = min(band_h, max(24, h // 3))
            band = img[max(0, h - band_h):h, 0:w]
            offset_y = max(0, h - band_h)

            for raw_bbox in self._detect_text_lines(band, relaxed=True):
                entities.append(
                    PHIEntity(
                        label="BURNED_IN_TEXT",
                        confidence=0.95,
                        source=PHISource.RULE,
                        text=None,
                        bbox=BoundingBox(
                            x1=int(max(0, raw_bbox.x1)),
                            y1=int(max(0, raw_bbox.y1 + offset_y)),
                            x2=int(raw_bbox.x2),
                            y2=int(raw_bbox.y2 + offset_y),
                        ),
                    )
                )

        logger.debug(
            "[TextRegionDetector] %d bottom watermark regions found",
            len(entities),
        )
        return entities

    # ── Strategy selection ────────────────────────────────────────────────────

    def _use_wide_strategy(
        self,
        image: np.ndarray,
        modality: Modality,
    ) -> bool:
        """
        True  → WIDE  (US/CT/MRI: full-width 15 % top+bottom strips)
        False → NARROW (Fundus/OCT: narrow 5 % top strip only)
        """

        # Explicit modality takes precedence
        if modality in _WIDE_MODALITIES:
            return True
        if modality in _NARROW_MODALITIES:
            return False

        # UNKNOWN modality: detect from colour channel difference.
        # Fundus/ophthalmic images are always orange/red (high R−B diff).
        # US/CT/MRI images are always grayscale (R ≈ G ≈ B, diff ≈ 0).
        if image.ndim >= 3 and image.shape[2] >= 3:
            b = image[:, :, 0].astype(np.int32)
            r = image[:, :, 2].astype(np.int32)
            mean_rb_diff = float(np.mean(np.abs(r - b)))
            if mean_rb_diff > _COLOR_DIFF_THRESHOLD:
                logger.debug(
                    "Colour image detected (mean |R−B|=%.1f) → NARROW mode",
                    mean_rb_diff,
                )
                return False  # Colour → likely fundus → NARROW

        # Default: WIDE (safe for DICOM, NIfTI, grayscale scans)
        return True

    # ── Edge band extraction ──────────────────────────────────────────────────

    def _edge_bands(
        self,
        image: np.ndarray,
        wide: bool,
    ) -> list[tuple[np.ndarray, int, int]]:
        """
        Return [(band, offset_x, offset_y), ...].

        WIDE  → full-width TOP strip only (15 % height)
        NARROW → narrow TOP strip only (5 %, ≤ 100 px)

        Why TOP-only for both modes
        ───────────────────────────
        On ultrasound machines (GE Logiq, Philips, etc.) the patient identity
        strip — name, MRN, institution, study date/time — is ALWAYS burned
        across the very top of the frame.  The BOTTOM-LEFT region holds
        clinical MEASUREMENTS (D1/D2/D3, volume) and the RIGHT panel holds
        scanner parameters (Frq, Gn, DR, AO%); centre annotations like
        "BLADDER"/"POSTVOID" are findings.  None of those are patient PHI and
        all must be preserved.  Scanning only the top strip removes the
        identity header while leaving every piece of clinical data intact.

        Fundus/OCT likewise place patient text only at the top, inside the
        black aperture border.
        """

        h, w = image.shape[:2]

        if wide:
            ch = int(h * _WIDE_STRIP_FRAC)
            return [
                (image[0:ch, 0:w], 0, 0),         # TOP full width — identity header
            ]
        else:
            # Narrow: stays within the black circular aperture border.
            ch = min(int(h * _FUNDUS_STRIP_FRAC), _FUNDUS_STRIP_MAX)
            return [
                (image[0:ch, 0:w], 0, 0),         # TOP narrow strip only
            ]

    # ── Text-line detection ───────────────────────────────────────────────────

    def _detect_text_lines(
        self,
        crop: np.ndarray,
        relaxed: bool = False,
    ) -> list[BoundingBox]:
        """
        Find text-line bounding boxes inside a single edge band.
        Returns boxes in band-local coordinates (offset not yet applied).
        """

        crop_h, crop_w = crop.shape[:2]

        gray_check = to_grayscale(to_uint8(crop))
        if int(gray_check.max()) < _MIN_PEAK:
            # Bottom watermarks often consist of bright white text, so allow
            # borderline cases that still contain strong edge structure.
            if int(gray_check.max()) < 180:
                return []

        binary = self._binarize(crop)
        fg = cv2.bitwise_not(binary)   # text pixels → foreground (white)

        # ── Connected components ──────────────────────────────────────────
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

        char_boxes: list[tuple[int, int, int, int]] = []

        for i in range(1, n):
            x  = stats[i, cv2.CC_STAT_LEFT]
            y  = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            ba = stats[i, cv2.CC_STAT_AREA]

            x1_c   = x  // _SCALE
            y1_c   = y  // _SCALE
            x2_c   = (x + bw) // _SCALE
            y2_c   = (y + bh) // _SCALE
            h_c    = y2_c - y1_c
            w_c    = x2_c - x1_c
            area_c = ba // (_SCALE * _SCALE)

            if relaxed:
                min_h, min_w, min_area = 2, 1, 2
                max_h, max_w = 80, 200
            else:
                min_h, min_w, min_area = _MIN_CHAR_H, _MIN_CHAR_W, _MIN_CHAR_AREA
                max_h, max_w = _MAX_CHAR_H, _MAX_CHAR_W

            if (
                min_h <= h_c <= max_h and
                min_w <= w_c <= max_w and
                area_c >= min_area
            ):
                char_boxes.append((x1_c, y1_c, x2_c, y2_c))

        if not char_boxes:
            return []

        # ── Group blobs into text lines by vertical proximity ─────────────
        char_boxes.sort(key=lambda b: ((b[1] + b[3]) / 2, b[0]))

        lines: list[list[tuple]] = []
        current: list[tuple]     = [char_boxes[0]]

        for box in char_boxes[1:]:
            prev_cy = (current[-1][1] + current[-1][3]) / 2
            curr_cy = (box[1]         + box[3])          / 2
            if abs(curr_cy - prev_cy) <= _LINE_Y_TOLERANCE:
                current.append(box)
            else:
                lines.append(current)
                current = [box]
        lines.append(current)

        # ── Convert lines → tight bounding boxes ──────────────────────────
        gray_orig = to_grayscale(to_uint8(crop))
        results: list[BoundingBox] = []

        for line in lines:

            if len(line) < (_MIN_BLOBS_PER_LINE if not relaxed else 1):
                continue

            y1 = max(0,      min(b[1] for b in line) - _PAD_Y)
            y2 = min(crop_h, max(b[3] for b in line) + _PAD_Y)

            row_band   = gray_orig[y1:y2, :]

            if not relaxed:
                # Per-line peak: fundus tissue max ≈ 80–101 < 120 → skip.
                # Text overlays always have max ≥ 120 (white or near-white glyphs).
                line_peak = int(row_band.max()) if row_band.size > 0 else 0
                if line_peak < _MIN_LINE_PEAK:
                    logger.debug(
                        "Skipping line y=%d–%d: peak=%d < %d",
                        y1, y2, line_peak, _MIN_LINE_PEAK,
                    )
                    continue

                # Background mean: reject rows that are bright clinical content
                bg_mean = float(np.mean(row_band)) if row_band.size > 0 else 0.0
                if bg_mean > _MAX_BG_MEAN:
                    logger.debug(
                        "Skipping line y=%d–%d: bg_mean=%.1f > %d",
                        y1, y2, bg_mean, _MAX_BG_MEAN,
                    )
                    continue

            # X-bounds from actual bright text pixels
            bright_cols = np.where(
                row_band.max(axis=0) > _BRIGHT_THRESHOLD
            )[0]

            if len(bright_cols) >= 2:
                x1 = max(0,       int(bright_cols[0])  - _PAD_X)
                x2 = min(crop_w,  int(bright_cols[-1]) + _PAD_X)
            else:
                x1 = max(0,       min(b[0] for b in line) - _PAD_X)
                x2 = min(crop_w,  max(b[2] for b in line) + _PAD_X)

            results.append(BoundingBox(
                x1=int(x1), y1=int(y1),
                x2=int(x2), y2=int(y2),
            ))

        return results

    # ── Binarization ─────────────────────────────────────────────────────────

    def _binarize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Upscale 2×, OTSU binarise → black text on white background.
        Auto-flips polarity for dark-on-light images.
        """

        gray = to_grayscale(to_uint8(image))

        gray2x = cv2.resize(
            gray, None,
            fx=_SCALE, fy=_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )

        _, binary = cv2.threshold(
            gray2x, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        return binary
