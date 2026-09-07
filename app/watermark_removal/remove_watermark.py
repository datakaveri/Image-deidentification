from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import easyocr
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from masking import redact_pixels


def merge_bboxes(bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not bboxes:
        return []

    def boxes_should_merge(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        y_overlap = min(ay2, by2) - max(ay1, by1)
        x_overlap = min(ax2, bx2) - max(ax1, bx1)
        y_tol = 10
        x_tol = 15

        vertical_close = by1 <= ay2 + y_tol and ay1 <= by2 + y_tol
        horizontal_close = bx1 <= ax2 + x_tol and ax1 <= bx2 + x_tol

        return vertical_close and horizontal_close and (y_overlap >= -y_tol or x_overlap >= -x_tol)

    sorted_boxes = sorted(bboxes, key=lambda x: (x[1], x[0]))
    merged: list[list[int]] = [list(sorted_boxes[0])]

    for x1, y1, x2, y2 in sorted_boxes[1:]:
        last = merged[-1]
        if boxes_should_merge(tuple(last), (x1, y1, x2, y2)):
            last[0] = min(last[0], x1)
            last[1] = min(last[1], y1)
            last[2] = max(last[2], x2)
            last[3] = max(last[3], y2)
        else:
            merged.append([x1, y1, x2, y2])

    return [tuple(box) for box in merged]


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


def detect_watermark_regions(
    image: np.ndarray,
    reader: easyocr.Reader,
    border_fraction: float = 0.15,
) -> list[tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    band_h = max(24, int(h * border_fraction))
    band_w = max(24, int(w * border_fraction))

    regions = []
    edge_slices = [
        (0, 0, w, band_h),
        (0, h - band_h, w, h),
        (0, 0, band_w, h),
        (w - band_w, 0, w, h),
    ]

    for x1, y1, x2, y2 in edge_slices:
        slice_img = image[y1:y2, x1:x2]
        if slice_img.size == 0:
            continue

        results = reader.readtext(slice_img)
        for box, text, score in results:
            bx1, by1 = int(box[0][0]), int(box[0][1])
            bx2, by2 = int(box[2][0]), int(box[2][1])
            abs_x1 = x1 + min(bx1, bx2)
            abs_y1 = y1 + min(by1, by2)
            abs_x2 = x1 + max(bx1, bx2)
            abs_y2 = y1 + max(by1, by2)
            regions.append((abs_x1, abs_y1, abs_x2, abs_y2))

    return merge_bboxes(regions)


def remove_bottom_watermark(
    input_path: str,
    output_path: str,
    reader: easyocr.Reader,
    detection_output_path: str | None = None,
) -> dict:
    input_file = Path(input_path)
    output_file = Path(output_path)

    image = load_image(input_file)
    detection_started_at = time.perf_counter()
    regions = detect_watermark_regions(image, reader, border_fraction=0.15)
    detection_seconds = time.perf_counter() - detection_started_at

    if detection_output_path is not None:
        detection_image = image.copy()
        for x1, y1, x2, y2 in regions:
            cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        save_image(Path(detection_output_path), detection_image)

    phi_regions = [{"bbox": (x1, y1, x2, y2), "zone": "anatomy"} for x1, y1, x2, y2 in regions]
    redaction_started_at = time.perf_counter()
    cleaned, mask = redact_pixels(image, phi_regions, ds=None)
    save_image(output_file, cleaned)
    redaction_seconds = time.perf_counter() - redaction_started_at

    return {
        "input": str(input_file),
        "output": str(output_file),
        "detections": len(phi_regions),
        "mask_pixels": int(np.sum(mask > 0)),
        "detection_seconds": round(detection_seconds, 3),
        "redaction_seconds": round(redaction_seconds, 3),
        "total_seconds": round(detection_seconds + redaction_seconds, 3),
    }


def process_folder(input_dir: Path, output_dir: Path, reader: easyocr.Reader, extensions: tuple[str, ...]) -> None:
    image_paths = sorted(
        path
        for extension in extensions
        for path in input_dir.rglob(f"*.{extension}")
        if path.is_file()
    )
    if not image_paths:
        raise SystemExit(f"No supported images found in {input_dir}")

    detection_seconds_total = 0.0
    redaction_seconds_total = 0.0
    successful = 0
    for image_path in image_paths:
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        try:
            result = remove_bottom_watermark(str(image_path), str(output_path), reader)
            detection_seconds = result["detection_seconds"]
            redaction_seconds = result["redaction_seconds"]
            detection_seconds_total += detection_seconds
            redaction_seconds_total += redaction_seconds
            successful += 1
            print(
                f"{relative_path}: Detection {detection_seconds:.3f}s;"
                f" Redaction {redaction_seconds:.3f}s"
            )
        except Exception as exc:
            print(f"Warning: failed to process {relative_path}: {exc}")

    print(f"Processed {successful}/{len(image_paths)} images")
    print(f"Detection time: {detection_seconds_total:.3f} seconds")
    print(f"Redaction time: {redaction_seconds_total:.3f} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and remove watermark text from an image folder")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory to save cleaned images")
    parser.add_argument("--ocr-langs", default="en", help="Comma-separated EasyOCR language codes")
    parser.add_argument("--ext", default="jpg,jpeg,png,bmp,tif,tiff,webp", help="Comma-separated image extensions")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr_langs = [lang.strip() for lang in args.ocr_langs.split(",") if lang.strip()]
    model_started_at = time.perf_counter()
    reader = easyocr.Reader(ocr_langs, gpu=False)
    model_seconds = time.perf_counter() - model_started_at
    print(f"EasyOCR model loaded in {model_seconds:.3f} seconds")

    process_folder(
        input_dir,
        output_dir,
        reader,
        tuple(extension.strip().lower().lstrip(".") for extension in args.ext.split(",") if extension.strip()),
    )


if __name__ == "__main__":
    main()
