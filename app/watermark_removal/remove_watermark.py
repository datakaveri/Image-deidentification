from __future__ import annotations

import argparse
from pathlib import Path
import sys

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


def detect_watermark_regions(image: np.ndarray, border_fraction: float = 0.15) -> list[tuple[int, int, int, int]]:
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

    reader = easyocr.Reader(["en"], gpu=False)
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

            # Expand the detected box slightly to ensure edge pixels are included.
            # Use a fraction of the smaller image dimension with a sensible minimum.
            pad = max(8, int(min(h, w) * 0.03))
            exp_x1 = max(0, abs_x1 - pad)
            exp_y1 = max(0, abs_y1 - pad)
            exp_x2 = min(w, abs_x2 + pad)
            exp_y2 = min(h, abs_y2 + pad)

            regions.append((exp_x1, exp_y1, exp_x2, exp_y2))

    return merge_bboxes(regions)


def remove_bottom_watermark(input_path: str, output_path: str, detection_output_path: str | None = None) -> dict:
    input_file = Path(input_path)
    output_file = Path(output_path)

    image = load_image(input_file)
    regions = detect_watermark_regions(image, border_fraction=0.15)

    if detection_output_path is not None:
        detection_image = image.copy()
        for x1, y1, x2, y2 in regions:
            cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        save_image(Path(detection_output_path), detection_image)

    phi_regions = [{"bbox": (x1, y1, x2, y2), "zone": "anatomy"} for x1, y1, x2, y2 in regions]
    cleaned, mask = redact_pixels(image, phi_regions, ds=None)
    save_image(output_file, cleaned)

    return {
        "input": str(input_file),
        "output": str(output_file),
        "detections": len(phi_regions),
        "mask_pixels": int(np.sum(mask > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and remove bottom-edge watermark text from an image")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to save the cleaned image")
    parser.add_argument(
        "--detection-output",
        dest="detection_output",
        help="Optional path to save an image with the detected text boxes overlaid",
    )
    args = parser.parse_args()

    result = remove_bottom_watermark(args.input_image, args.output_image, args.detection_output)
    print(result)


if __name__ == "__main__":
    main()
