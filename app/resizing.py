import argparse
from pathlib import Path
from typing import Iterable

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_image_quality_score(image):
    """Compute a blur-based quality score using the variance of the Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def choose_target_size(score: float, high_thresh: float, low_thresh: float):
    """Return a quality label and target size for a given quality score."""
    sizes = {
        "HIGH": (1920, 1080),
        "MEDIUM": (1280, 720),
        "LOW": (640, 480),
    }
    if score > high_thresh:
        return "HIGH", sizes["HIGH"]
    if score >= low_thresh:
        return "MEDIUM", sizes["MEDIUM"]
    return "LOW", sizes["LOW"]


def resize_image(image, target_size):
    """Resize the image with the appropriate interpolation method."""
    height, width = image.shape[:2]
    target_width, target_height = target_size
    if target_width < width or target_height < height:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    return cv2.resize(image, target_size, interpolation=interp)


def iter_image_files(folder: Path, extensions: set[str]) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def process_folder(input_folder: Path, output_folder: Path, high_thresh: float, low_thresh: float, extensions: set[str]):
    output_folder.mkdir(parents=True, exist_ok=True)
    for src_path in iter_image_files(input_folder, extensions):
        image = cv2.imread(str(src_path))
        if image is None:
            print(f"Skipping invalid image: {src_path}")
            continue

        score = get_image_quality_score(image)
        quality_label, target_size = choose_target_size(score, high_thresh, low_thresh)
        resized = resize_image(image, target_size)

        relative_path = src_path.relative_to(input_folder)
        dest_path = output_folder / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest_path), resized)

        print(
            f"Processed {relative_path}: quality={quality_label}, score={score:.2f}, "
            f"size={target_size[0]}x{target_size[1]}"
        )

def main():
    parser = argparse.ArgumentParser(description="Resize images in a folder based on quality and save results to an output folder.")
    parser.add_argument("input_folder", help="Path to the input image folder")
    parser.add_argument("output_folder", help="Path to the output folder")
    parser.add_argument("--high-thresh", type=float, default=500.0, help="High quality threshold")
    parser.add_argument("--low-thresh", type=float, default=100.0, help="Low quality threshold")
    parser.add_argument(
        "--ext",
        default="jpg,png,jpeg,bmp,tif,tiff",
        help="Comma-separated list of image extensions to process",
    )
    args = parser.parse_args()

    input_folder = Path(args.input_folder).expanduser().resolve()
    output_folder = Path(args.output_folder).expanduser().resolve()
    extensions = {ext.strip().lower() if ext.startswith('.') else f".{ext.strip().lower()}" for ext in args.ext.split(",")}

    if not input_folder.exists() or not input_folder.is_dir():
        raise SystemExit(f"Input folder not found or not a directory: {input_folder}")

    process_folder(input_folder, output_folder, args.high_thresh, args.low_thresh, extensions)


if __name__ == "__main__":
    main()