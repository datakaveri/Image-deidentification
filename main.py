#!/usr/bin/env python3
"""End-to-end anonymization pipeline for road-defect image datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG: dict[str, Any] = {
    "input_dir": "train/images",
    "output_dir": "outputs/final",
    "temp_dir": "outputs/temp",
    "weights": "app/sensitive_data_masking/license_plate_detector.pt",
    "device": "auto",
    "mask_mode": "black",
    "ocr": False,
    "ocr_langs": "en",
    "conf": 0.1,
    "imgsz": 640,
    "high_thresh": 500.0,
    "low_thresh": 100.0,
    "ext": "jpg,jpeg,png,bmp,tif,tiff,webp",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full anonymization pipeline: EXIF cleanup -> watermark removal -> human masking -> plate masking -> resizing"
    )
    parser.add_argument("--config", default="pipeline_config.json", help="Path to the JSON config file")
    parser.add_argument("--input-dir", default=None, help="Directory containing the input image dataset")
    parser.add_argument("--output-dir", default=None, help="Directory where the final resized images will be written")
    parser.add_argument("--temp-dir", default=None, help="Directory for intermediate pipeline outputs")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to the YOLO plate detector weights file",
    )
    parser.add_argument("--device", default=None, help="Device to use for DeepLab and plate detection (cpu, cuda, auto)")
    parser.add_argument("--mask-mode", default=None, choices=["black", "blur", "pixelate"], help="Masking mode for plate masking")
    parser.add_argument("--ocr", action="store_true", default=None, help="Run EasyOCR for detected plates")
    parser.add_argument("--ocr-langs", default=None, help="Comma-separated EasyOCR language codes")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold for plate detection")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size")
    parser.add_argument("--high-thresh", type=float, default=None, help="High image quality threshold for resizing")
    parser.add_argument("--low-thresh", type=float, default=None, help="Low image quality threshold for resizing")
    parser.add_argument("--ext", default=None, help="Comma-separated list of image extensions to process")
    return parser


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")

    return data


def resolve_value(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def resolve_path(path_value: Any, fallback: str) -> Path:
    raw_value = str(resolve_value(path_value, None, fallback))
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_image_files(folder: Path, extensions: Iterable[str]) -> list[Path]:
    ext_set = {ext.lower().lstrip(".") for ext in extensions}
    files: list[Path] = []
    if not folder.exists():
        return files
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower().lstrip(".") in ext_set:
            files.append(path)
    return files


def run_command(cmd: Sequence[str], description: str) -> None:
    print(f"\n== {description} ==")
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(ROOT_DIR), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {completed.returncode}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (ROOT_DIR / config_path).resolve()

    config = {**DEFAULT_CONFIG, **load_config(config_path)}

    input_dir = resolve_path(args.input_dir, config["input_dir"])
    output_dir = resolve_path(args.output_dir, config["output_dir"])
    temp_dir = resolve_path(args.temp_dir, config["temp_dir"])
    weights = resolve_path(args.weights, config["weights"])
    device = str(resolve_value(args.device, config.get("device"), DEFAULT_CONFIG["device"]))
    mask_mode = str(resolve_value(args.mask_mode, config.get("mask_mode"), DEFAULT_CONFIG["mask_mode"]))
    ocr = bool(resolve_value(args.ocr, config.get("ocr"), DEFAULT_CONFIG["ocr"]))
    ocr_langs = str(resolve_value(args.ocr_langs, config.get("ocr_langs"), DEFAULT_CONFIG["ocr_langs"]))
    conf = float(resolve_value(args.conf, config.get("conf"), DEFAULT_CONFIG["conf"]))
    imgsz = int(resolve_value(args.imgsz, config.get("imgsz"), DEFAULT_CONFIG["imgsz"]))
    high_thresh = float(resolve_value(args.high_thresh, config.get("high_thresh"), DEFAULT_CONFIG["high_thresh"]))
    low_thresh = float(resolve_value(args.low_thresh, config.get("low_thresh"), DEFAULT_CONFIG["low_thresh"]))
    extensions = [item.strip().lower() for item in str(resolve_value(args.ext, config.get("ext"), DEFAULT_CONFIG["ext"])).split(",") if item.strip()]

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    image_files = iter_image_files(input_dir, extensions)
    if not image_files:
        raise SystemExit(f"No supported images found in {input_dir}")

    ensure_dir(output_dir)
    ensure_dir(temp_dir)

    exif_dir = temp_dir / "exif"
    watermark_dir = temp_dir / "watermark_removed"
    human_mask_dir = temp_dir / "human_masked"
    plate_mask_dir = temp_dir / "plate_masked"

    ensure_dir(exif_dir)
    ensure_dir(watermark_dir)
    ensure_dir(human_mask_dir)
    ensure_dir(plate_mask_dir)

    print(f"Discovered {len(image_files)} images from {input_dir}")

    run_command(
        [
            sys.executable,
            "app/exif_geo_tag/store_geo_tag_exif.py",
            str(input_dir),
            str(exif_dir),
            "--recursive",
        ],
        "Step 1/5: Strip EXIF metadata except GPS/geo tags",
    )

    for src_image in image_files:
        rel_path = src_image.relative_to(input_dir)
        exif_image = exif_dir / rel_path
        watermark_image = watermark_dir / rel_path
        watermark_image.parent.mkdir(parents=True, exist_ok=True)
        if not exif_image.exists():
            print(f"Skipping watermark step for {rel_path}: EXIF output missing")
            continue
        try:
            run_command(
                [
                    sys.executable,
                    "app/watermark_removal/remove_watermark.py",
                    str(exif_image),
                    str(watermark_image),
                ],
                f"Step 2/5: Remove watermark from {rel_path}",
            )
        except RuntimeError as exc:
            print(f"Warning: {exc}. Falling back to the EXIF-cleaned image.")
            shutil.copy2(exif_image, watermark_image)

    run_command(
        [
            sys.executable,
            "app/sensitive_data_masking/deeplab.py",
            "--input-dir",
            str(watermark_dir),
            "--output-dir",
            str(human_mask_dir),
            "--mask-type",
            "blur",
            "--device",
            device,
        ],
        "Step 3/5: Apply human masking with DeepLab",
    )

    plate_args = [
        sys.executable,
        "app/sensitive_data_masking/mask_plates.py",
        "--weights",
        str(weights),
        "--source",
        str(human_mask_dir),
        "--out",
        str(plate_mask_dir),
        "--mode",
        mask_mode,
        "--device",
        device,
        "--conf",
        str(conf),
        "--imgsz",
        str(imgsz),
        "--classes",
        "license plate,number plate,plate",
        "--ext",
        ",".join(extensions),
    ]
    if ocr:
        plate_args.extend(["--ocr", "--ocr-langs", ocr_langs])
    plate_args.extend(["--output-csv", str(temp_dir / "plate_results.csv")])

    run_command(plate_args, "Step 4/5: Mask license plates")

    run_command(
        [
            sys.executable,
            "app/resizing.py",
            str(plate_mask_dir),
            str(output_dir),
            "--high-thresh",
            str(high_thresh),
            "--low-thresh",
            str(low_thresh),
            "--ext",
            ",".join(extensions),
        ],
        "Step 5/5: Resize the final images",
    )

    print("\nPipeline complete.")
    print(f"Final images saved to: {output_dir}")
    print(f"Intermediate outputs are in: {temp_dir}")


if __name__ == "__main__":
    main()
