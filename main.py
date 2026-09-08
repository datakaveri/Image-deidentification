#!/usr/bin/env python3
#"""Configuration-driven in-memory road-defect anonymization pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable
import cv2
import numpy as np

from app.deidentification import (
    detect_human_mask,
    detect_plate_boxes,
    detect_watermark_regions,
    redact_human_mask,
    redact_plate_boxes,
    redact_watermark_regions,
    resize_for_quality,
    save_with_geo_exif,
)

ROOT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG: dict[str, Any] = {
    "input_dir": os.environ.get("SKALD_DATA_DIR", "train/images"),
    "output_dir": os.environ.get("SKALD_OUTPUT_DIR", "outputs/final"),
    "temp_dir": os.environ.get("SKALD_TEMP_DIR", "outputs/temp"),
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
    "exif_strip": True,
    "watermark_removal": True,
    "human_mask": True,
    "plate_mask": True,
    "resizing": True,
    "parallel_detections": True,
    "compare_sequential": False,
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EXIF-aware in-memory anonymization: plate -> human -> watermark -> resize -> save"
    )
    parser.add_argument("--config", default="pipeline_config.json")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mask-mode", default=None, choices=["black", "blur", "pixelate"])
    parser.add_argument("--ocr", action="store_true", default=None)
    parser.add_argument("--ocr-langs", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--high-thresh", type=float, default=None)
    parser.add_argument("--low-thresh", type=float, default=None)
    parser.add_argument("--ext", default=None)
    parser.add_argument("--parallel-detections", dest="parallel_detections", action="store_true", default=None)
    parser.add_argument("--no-parallel-detections", dest="parallel_detections", action="store_false")
    parser.add_argument("--compare-sequential", dest="compare_sequential", action="store_true", default=None)
    for name in ("exif_strip", "watermark_removal", "human_mask", "plate_mask", "resizing"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, action="store_true", default=None)
        parser.add_argument(f"--no-{name.replace('_', '-')}", dest=name, action="store_false")
    return parser

def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return data

def resolve(cli_value: Any, config: dict[str, Any], key: str) -> Any:
    if cli_value is not None:
        return cli_value
    return config.get(key, DEFAULT_CONFIG[key])

def resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()

def iter_image_files(folder: Path, extensions: Iterable[str]) -> list[Path]:
    allowed = {extension.lower().lstrip(".") for extension in extensions}
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in allowed
    )

def load_models(
    plate_mask: bool,
    human_mask: bool,
    watermark_removal: bool,
    ocr: bool,
    weights: Path,
    device_name: str,
    ocr_langs: str,
) -> tuple[Any, Any, Any, Any, dict[str, float]]:
    load_times: dict[str, float] = {}
    yolo_model = None
    deeplab_model = None
    deeplab_device = None
    ocr_reader = None

    import torch

    if device_name == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device_name
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        resolved_device = "cpu"

    if plate_mask:
        from ultralytics import YOLO

        started = time.perf_counter()
        yolo_model = YOLO(str(weights))
        load_times["yolo_model_load_seconds"] = time.perf_counter() - started

    if human_mask:
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

        started = time.perf_counter()
        deeplab_device = torch.device(resolved_device)
        deeplab_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        deeplab_model.to(deeplab_device)
        deeplab_model.eval()
        load_times["deeplab_model_load_seconds"] = time.perf_counter() - started

    if watermark_removal or (plate_mask and ocr):
        import easyocr

        started = time.perf_counter()
        languages = [lang.strip() for lang in ocr_langs.split(",") if lang.strip()]
        ocr_reader = easyocr.Reader(languages, gpu=False)
        load_times["easyocr_model_load_seconds"] = time.perf_counter() - started

    return yolo_model, deeplab_model, deeplab_device, ocr_reader, load_times


def run_detections(image: Any, models: tuple[Any, Any, Any, Any], settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float], str]:
    yolo_model, deeplab_model, deeplab_device, ocr_reader = models
    detections: dict[str, Any] = {}
    timings: dict[str, float] = {}
    started = time.perf_counter()
    reader_lock = threading.Lock()

    def detect_plates() -> list[Any]:
        item_started = time.perf_counter()
        result = detect_plate_boxes(yolo_model, image, settings["conf"], settings["imgsz"], settings["device"], ocr_reader if settings["ocr"] else None, {"license plate", "number plate", "plate"}, reader_lock)
        timings["plate_detection_seconds"] = time.perf_counter() - item_started
        return result

    def detect_humans() -> Any:
        item_started = time.perf_counter()
        result = detect_human_mask(deeplab_model, deeplab_device, image, dilation_size=5)
        timings["human_detection_seconds"] = time.perf_counter() - item_started
        return result

    def detect_watermarks() -> list[Any]:
        item_started = time.perf_counter()
        result = detect_watermark_regions(image, ocr_reader, reader_lock=reader_lock)
        timings["watermark_detection_seconds"] = time.perf_counter() - item_started
        return result

    tasks = {}
    if settings["plate_mask"]:
        tasks["plates"] = detect_plates
    if settings["human_mask"]:
        tasks["human_mask"] = detect_humans
    if settings["watermark_removal"]:
        tasks["watermarks"] = detect_watermarks

    cuda_mode = settings["device"].startswith("cuda")
    if not settings["parallel_detections"] or len(tasks) < 2:
        parallel_mode = "sequential"
        for name, task in tasks.items():
            detections[name] = task()
    elif cuda_mode:
        parallel_mode = "watermark_parallel_with_models"
        watermark_task = tasks.pop("watermarks", None)
        with ThreadPoolExecutor(max_workers=2 if watermark_task else 1) as executor:
            watermark_future = executor.submit(watermark_task) if watermark_task else None
            for name, task in tasks.items():
                detections[name] = task()
            if watermark_future is not None:
                detections["watermarks"] = watermark_future.result()
    else:
        parallel_mode = "all_detectors_parallel"
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {name: executor.submit(task) for name, task in tasks.items()}
            for name, future in futures.items():
                detections[name] = future.result()

    timings["detection_wall_seconds"] = time.perf_counter() - started
    return detections, timings, parallel_mode

def process_image(
    source_path: Path,
    output_path: Path,
    models: tuple[Any, Any, Any, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    yolo_model, deeplab_model, deeplab_device, ocr_reader = models
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {source_path}")

    detections, timings, parallel_mode = run_detections(image, models, settings)
    if settings["compare_sequential"] and parallel_mode != "sequential":
        sequential_settings = {**settings, "parallel_detections": False}
        sequential_detections, _, _ = run_detections(image, models, sequential_settings)
        timings["detection_equivalent"] = float(
            detections.get("plates", []) == sequential_detections.get("plates", [])
            and np.array_equal(detections.get("human_mask"), sequential_detections.get("human_mask"))
            and detections.get("watermarks", []) == sequential_detections.get("watermarks", [])
        )

    if settings["plate_mask"]:
        started = time.perf_counter()
        image = redact_plate_boxes(image, detections["plates"], settings["mask_mode"])
        timings["plate_redaction_seconds"] = time.perf_counter() - started

    if settings["human_mask"]:
        started = time.perf_counter()
        image = redact_human_mask(image, detections["human_mask"], "blur")
        timings["human_redaction_seconds"] = time.perf_counter() - started

    if settings["watermark_removal"]:
        started = time.perf_counter()
        image = redact_watermark_regions(image, detections["watermarks"])
        timings["watermark_redaction_seconds"] = time.perf_counter() - started

    if settings["resizing"]:
        started = time.perf_counter()
        image, quality, quality_score = resize_for_quality(
            image, settings["high_thresh"], settings["low_thresh"]
        )
        timings["resize_seconds"] = time.perf_counter() - started
    else:
        quality, quality_score = "UNCHANGED", None

    started = time.perf_counter()
    save_with_geo_exif(image, source_path, output_path, settings["exif_strip"])
    timings["save_seconds"] = time.perf_counter() - started
    with output_path.open("rb") as output_file:
        output_sha256 = hashlib.sha256(output_file.read()).hexdigest()

    plate_text = " | ".join(
        detection.text for detection in detections.get("plates", []) if detection.text
    )
    return {
        "image_name": str(source_path),
        "plate_text": plate_text,
        "quality": quality,
        "quality_score": quality_score,
        "parallel_mode": parallel_mode,
        "plate_detection_count": len(detections.get("plates", [])),
        "human_detection_count": int((detections.get("human_mask", 0) > 0).sum()) if "human_mask" in detections else 0,
        "watermark_detection_count": len(detections.get("watermarks", [])),
        "output_sha256": output_sha256,
        **timings,
    }

def main() -> None:
    started = time.perf_counter()
    args = build_parser().parse_args()
    config_path = resolve_path(args.config)
    config = {**DEFAULT_CONFIG, **load_config(config_path)}

    input_dir = resolve_path(resolve(args.input_dir, config, "input_dir"))
    output_dir = resolve_path(resolve(args.output_dir, config, "output_dir"))
    temp_dir = resolve_path(resolve(args.temp_dir, config, "temp_dir"))
    extensions = str(resolve(args.ext, config, "ext")).split(",")
    image_files = iter_image_files(input_dir, extensions)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not image_files:
        raise SystemExit(f"No supported images found in {input_dir}")

    settings = {
        "weights": resolve_path(resolve(args.weights, config, "weights")),
        "device": str(resolve(args.device, config, "device")),
        "mask_mode": str(resolve(args.mask_mode, config, "mask_mode")),
        "ocr": bool(resolve(args.ocr, config, "ocr")),
        "ocr_langs": str(resolve(args.ocr_langs, config, "ocr_langs")),
        "conf": float(resolve(args.conf, config, "conf")),
        "imgsz": int(resolve(args.imgsz, config, "imgsz")),
        "high_thresh": float(resolve(args.high_thresh, config, "high_thresh")),
        "low_thresh": float(resolve(args.low_thresh, config, "low_thresh")),
        "exif_strip": bool(resolve(args.exif_strip, config, "exif_strip")),
        "watermark_removal": bool(resolve(args.watermark_removal, config, "watermark_removal")),
        "human_mask": bool(resolve(args.human_mask, config, "human_mask")),
        "plate_mask": bool(resolve(args.plate_mask, config, "plate_mask")),
        "resizing": bool(resolve(args.resizing, config, "resizing")),
        "parallel_detections": bool(resolve(args.parallel_detections, config, "parallel_detections")),
        "compare_sequential": bool(resolve(args.compare_sequential, config, "compare_sequential")),
    }

    if settings["device"] == "auto":
        import torch

        settings["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    models_started = time.perf_counter()
    model_objects = load_models(
        settings["plate_mask"], settings["human_mask"], settings["watermark_removal"],
        settings["ocr"], settings["weights"], settings["device"], settings["ocr_langs"],
    )
    yolo_model, deeplab_model, deeplab_device, ocr_reader, model_times = model_objects
    print(f"Loaded enabled models in {time.perf_counter() - models_started:.3f} seconds")
    for name, seconds in model_times.items():
        print(f"{name}: {seconds:.3f} seconds")
    import torch

    if settings["device"].startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    processing_started = time.perf_counter()
    cpu_started = time.process_time()

    csv_path = temp_dir / "plate_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_name", "plate_text"])
        successful = 0
        errors = 0
        total_plate_detections = 0
        total_human_mask_pixels = 0
        total_watermark_detections = 0
        equivalent_images = 0
        for index, source_path in enumerate(image_files, 1):
            relative_path = source_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            print(f"[{index}/{len(image_files)}] Processing {relative_path}")
            try:
                result = process_image(
                    source_path,
                    output_path,
                    (yolo_model, deeplab_model, deeplab_device, ocr_reader),
                    settings,
                )
                writer.writerow([relative_path.as_posix(), result["plate_text"]])
                total_plate_detections += result["plate_detection_count"]
                total_human_mask_pixels += result["human_detection_count"]
                total_watermark_detections += result["watermark_detection_count"]
                equivalent_images += int(result.get("detection_equivalent", False))
                timing_text = "; ".join(
                    f"{key}={value:.3f}s" for key, value in result.items() if key.endswith("seconds")
                )
                print(f"  {timing_text}")
                successful += 1
            except Exception as exc:
                errors += 1
                writer.writerow([relative_path.as_posix(), ""])
                print(f"  Failed: {exc}")

    total_seconds = time.perf_counter() - started
    print(f"Processed: {len(image_files)} | Successful: {successful} | Errors: {errors}")
    processing_wall_seconds = time.perf_counter() - processing_started
    cpu_utilization = (time.process_time() - cpu_started) / max(processing_wall_seconds, 1e-9)
    cpu_utilization = cpu_utilization / max(os.cpu_count() or 1, 1) * 100.0
    peak_gpu_mb = 0.0
    if settings["device"].startswith("cuda") and torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"Detection counts: plates={total_plate_detections}; human_mask_pixels={total_human_mask_pixels}; watermarks={total_watermark_detections}")
    if settings["compare_sequential"]:
        print(f"Parallel/sequential detection equivalence: {equivalent_images}/{successful} images")
    print(f"CPU utilization during image processing: {cpu_utilization:.1f}%")
    print(f"Peak GPU memory allocated: {peak_gpu_mb:.1f} MB")
    print(f"Image processing wall time: {processing_wall_seconds:.3f} seconds")
    print(f"Pipeline complete in {total_seconds:.3f} seconds")
    print(f"Final images saved to: {output_dir}")
    print(f"Plate results saved to: {csv_path}")

if __name__ == "__main__":
    main()

"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parent

# The container sets these (SKALD_* names, matching skald-dicom) so the image
# works against the SPIDEr/TEE mount contract — /app/data, /app/config,
# /app/output — even when the mounted config directory is empty. Outside the
# container they are unset and the repo-relative defaults below apply.
DEFAULT_CONFIG: dict[str, Any] = {
    "input_dir": os.environ.get("SKALD_DATA_DIR", "train/images"),
    "output_dir": os.environ.get("SKALD_OUTPUT_DIR", "outputs/final"),
    "temp_dir": os.environ.get("SKALD_TEMP_DIR", "outputs/temp"),
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
    "exif_strip": True,
    "watermark_removal": True,
    "human_mask": True,
    "plate_mask": True,
    "resizing": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full anonymization pipeline: EXIF cleanup -> plate masking -> human masking -> watermark removal -> resizing"
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
    parser.add_argument("--exif-strip", dest="exif_strip", action="store_true", default=None, help="Enable EXIF stripping")
    parser.add_argument("--no-exif-strip", dest="exif_strip", action="store_false", help="Disable EXIF stripping")
    parser.add_argument("--watermark-removal", dest="watermark_removal", action="store_true", default=None, help="Enable watermark removal")
    parser.add_argument("--no-watermark-removal", dest="watermark_removal", action="store_false", help="Disable watermark removal")
    parser.add_argument("--human-mask", dest="human_mask", action="store_true", default=None, help="Enable human masking")
    parser.add_argument("--no-human-mask", dest="human_mask", action="store_false", help="Disable human masking")
    parser.add_argument("--plate-mask", dest="plate_mask", action="store_true", default=None, help="Enable plate masking")
    parser.add_argument("--no-plate-mask", dest="plate_mask", action="store_false", help="Disable plate masking")
    parser.add_argument("--resizing", dest="resizing", action="store_true", default=None, help="Enable resizing")
    parser.add_argument("--no-resizing", dest="resizing", action="store_false", help="Disable resizing")
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
    exif_strip = bool(resolve_value(args.exif_strip, config.get("exif_strip"), DEFAULT_CONFIG["exif_strip"]))
    watermark_removal = bool(resolve_value(args.watermark_removal, config.get("watermark_removal"), DEFAULT_CONFIG["watermark_removal"]))
    human_mask = bool(resolve_value(args.human_mask, config.get("human_mask"), DEFAULT_CONFIG["human_mask"]))
    plate_mask = bool(resolve_value(args.plate_mask, config.get("plate_mask"), DEFAULT_CONFIG["plate_mask"]))
    resizing = bool(resolve_value(args.resizing, config.get("resizing"), DEFAULT_CONFIG["resizing"]))

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

    prev_dir = input_dir
    # Step 1: EXIF cleaning (optional)
    if exif_strip:
        run_command(
            [
                sys.executable,
                "app/exif_geo_tag/store_geo_tag_exif.py",
                str(input_dir),
                str(exif_dir),
                "--recursive",
            ],
            "Step 1/4: Strip EXIF metadata except GPS/geo tags",
        )
        prev_dir = exif_dir
    else:
        print("Skipping EXIF stripping step")

    # Step 2: Plate masking (run early so plates are removed before other ops)
    if plate_mask:
        plate_args = [
            sys.executable,
            "app/sensitive_data_masking/mask_plates.py",
            "--weights",
            str(weights),
            "--source",
            str(prev_dir),
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

        run_command(plate_args, "Step 2/4: Mask license plates")
        prev_dir = plate_mask_dir
    else:
        print("Skipping plate masking step")

    # Step 3: Human masking
    if human_mask:
        run_command(
            [
                sys.executable,
                "app/sensitive_data_masking/deeplab.py",
                "--input-dir",
                str(prev_dir),
                "--output-dir",
                str(human_mask_dir),
                "--mask-type",
                "blur",
                "--device",
                device,
            ],
            "Step 3/4: Apply human masking with DeepLab",
        )
        prev_dir = human_mask_dir
    else:
        print("Skipping human masking step")

    # Step 4: Watermark removal
    if watermark_removal:
        for src_image in image_files:
            rel_path = src_image.relative_to(input_dir)
            source_image = prev_dir / rel_path if prev_dir != input_dir else src_image
            watermark_image = watermark_dir / rel_path
            watermark_image.parent.mkdir(parents=True, exist_ok=True)
            if not source_image.exists():
                print(f"Skipping watermark step for {rel_path}: source image missing")
                continue
            try:
                run_command(
                    [
                        sys.executable,
                        "app/watermark_removal/remove_watermark.py",
                        str(source_image),
                        str(watermark_image),
                    ],
                    f"Step 4/4: Remove watermark from {rel_path}",
                )
            except RuntimeError as exc:
                print(f"Warning: {exc}. Falling back to the prior image source.")
                shutil.copy2(source_image, watermark_image)
        prev_dir = watermark_dir
    else:
        print("Skipping watermark removal step")

    if resizing:
        run_command(
            [
                sys.executable,
                "app/resizing.py",
                str(prev_dir),
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
    else:
        print("Skipping resizing step")
        if prev_dir != output_dir:
            shutil.copytree(prev_dir, output_dir, dirs_exist_ok=True)

    print("\nPipeline complete.")
    print(f"Final images saved to: {output_dir}")
    print(f"Intermediate outputs are in: {temp_dir}")


if __name__ == "__main__":
    main()
"""
