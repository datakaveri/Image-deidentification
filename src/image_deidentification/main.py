#!/usr/bin/env python3
"""Configuration-driven in-memory road-survey image de-identification pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import psutil

from image_deidentification.deidentification import (
    detect_human_mask,
    detect_plate_boxes,
    detect_watermark_regions,
    extract_geo_exif,
    map_human_mask_to_original,
    map_plate_detections_to_original,
    map_watermark_regions_to_original,
    prepare_detection_image,
    redact_human_mask,
    redact_plate_boxes,
    redact_watermark_regions,
    resize_for_quality,
    save_with_geo_exif,
)

# Relative paths (config, weights, input/output dirs) resolve against the working
# directory: the repo root for local runs, WORKDIR /app in the container. The
# package itself may live in site-packages, so its own location is no anchor.
ROOT_DIR = Path.cwd()
_WORKER_MODELS: tuple[Any, Any, Any, Any] | None = None
_WORKER_SETTINGS: dict[str, Any] | None = None
_WORKER_LOGGER: logging.Logger | None = None
_WORKER_STATUS_QUEUE: Any = None
_WORKER_PEAK_RSS_BYTES = 0

DEFAULT_CONFIG: dict[str, Any] = {
    "input_dir": os.environ.get("SKALD_DATA_DIR", "train/images"),
    "output_dir": os.environ.get("SKALD_OUTPUT_DIR", "outputs/final"),
    "temp_dir": os.environ.get("SKALD_TEMP_DIR", "outputs/temp"),
    "weights": "models/license_plate_detector.pt",
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
    "human_mask_mode": "blur",
    "human_blur_kernel": 51,
    "human_dilation_size": 5,
    "plate_mask": True,
    "resizing": True,
    "workers": 0,
    "max_gpu_workers": 1,
    "worker_start_method": "spawn",
    "log_file": "",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-deidentification",
        description="Run EXIF-aware in-memory anonymization: plate -> human -> watermark -> resize -> save",
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
    parser.add_argument("--human-blur-kernel", type=int, default=None)
    parser.add_argument("--human-dilation-size", type=int, default=None)
    parser.add_argument("--human-mask-mode", choices=["blur", "black", "grey"], default=None)
    parser.add_argument("--ext", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-gpu-workers", type=int, default=None)
    parser.add_argument("--log-file", default=None)
    for name in ("exif_strip", "watermark_removal", "human_mask", "plate_mask", "resizing"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", dest=name, action="store_true", default=None
        )
        parser.add_argument(f"--no-{name.replace('_', '-')}", dest=name, action="store_false")
    return parser


def configure_worker_logging(log_queue: Any) -> logging.Logger:
    logger = logging.getLogger("image_deidentification")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.handlers.QueueHandler(log_queue))
    return logger


def configure_run_logging(
    log_path: Path,
    multiprocessing_context: Any,
) -> tuple[logging.Logger, Any, logging.handlers.QueueListener]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_queue = multiprocessing_context.Queue()
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(processName)s[%(process)d] %(levelname)s %(message)s")
    )
    listener = logging.handlers.QueueListener(log_queue, file_handler)
    listener.start()
    logger = configure_worker_logging(log_queue)
    return logger, log_queue, listener


def cuda_memory_snapshot(device_name: str) -> dict[str, int] | None:
    if not device_name.startswith("cuda"):
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


class MemoryMonitor:
    def __init__(self, process: psutil.Process, interval_seconds: float = 0.1) -> None:
        self.process = process
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="memory-monitor", daemon=True)
        self.parent_peak_rss_bytes = 0
        self.worker_peak_rss_bytes: dict[int, int] = {}
        self.system_total_bytes = 0
        self.system_peak_used_bytes = 0
        self.system_peak_cpu_percent = 0.0
        self.logical_cpu_count = psutil.cpu_count(logical=True) or 0
        self.physical_core_count = psutil.cpu_count(logical=False) or 0
        self.cpu_affinity = self._get_cpu_affinity()

    def start(self) -> None:
        self._sample()
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()
        self._sample()

    def _sample(self) -> None:
        try:
            self.parent_peak_rss_bytes = max(
                self.parent_peak_rss_bytes,
                self.process.memory_info().rss,
            )
            for child in self.process.children(recursive=True):
                try:
                    rss_bytes = child.memory_info().rss
                    self.worker_peak_rss_bytes[child.pid] = max(
                        self.worker_peak_rss_bytes.get(child.pid, 0),
                        rss_bytes,
                    )
                except psutil.Error:
                    continue
            system_memory = psutil.virtual_memory()
            self.system_total_bytes = system_memory.total
            self.system_peak_used_bytes = max(
                self.system_peak_used_bytes,
                system_memory.total - system_memory.available,
            )
            self.system_peak_cpu_percent = max(
                self.system_peak_cpu_percent,
                psutil.cpu_percent(interval=None),
            )
        except psutil.Error:
            return

    def _get_cpu_affinity(self) -> list[int]:
        try:
            return self.process.cpu_affinity()
        except (AttributeError, psutil.Error):
            return list(range(self.logical_cpu_count))

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()


def megabytes(byte_count: int) -> float:
    return byte_count / (1024 * 1024)


class QueueTracker:
    def __init__(self, logger: logging.Logger, status_queue: Any) -> None:
        self.logger = logger
        self.status_queue = status_queue
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._consume, name="queue-monitor", daemon=True)
        self.submitted = 0
        self.started = 0
        self.completed = 0
        self.failed = 0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.status_queue.put(None)
        self.thread.join()

    def submit(self, task_id: int, image_name: str) -> None:
        with self.lock:
            self.submitted += 1
            snapshot = self.snapshot()
        self.logger.info("QUEUE submit task=%s image=%s %s", task_id, image_name, snapshot)

    def complete(self, task_id: int, image_name: str) -> None:
        with self.lock:
            self.completed += 1
            snapshot = self.snapshot()
        self.logger.info("QUEUE complete task=%s image=%s %s", task_id, image_name, snapshot)

    def fail(self, task_id: int, image_name: str) -> None:
        with self.lock:
            self.failed += 1
            snapshot = self.snapshot()
        self.logger.info("QUEUE failed task=%s image=%s %s", task_id, image_name, snapshot)

    def snapshot(self) -> str:
        waiting = max(0, self.submitted - self.started)
        running = max(0, self.started - self.completed - self.failed)
        return (
            f"submitted={self.submitted} waiting={waiting} running={running} "
            f"completed={self.completed} failed={self.failed}"
        )

    def _consume(self) -> None:
        while True:
            try:
                event = self.status_queue.get(timeout=0.5)
            except Exception:
                continue
            if event is None:
                break
            task_id, image_name, worker_pid = event
            with self.lock:
                self.started += 1
                snapshot = self.snapshot()
            self.logger.info(
                "QUEUE start task=%s image=%s worker_pid=%s %s",
                task_id,
                image_name,
                worker_pid,
                snapshot,
            )


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
        path
        for path in folder.rglob("*")
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
        weights_path = Path(weights)
        if not weights_path.exists():
            # Configs written before the src/ move point at
            # app/sensitive_data_masking/<name>; weights now live in models/.
            alt_path = ROOT_DIR / "models" / weights_path.name
            if alt_path.exists():
                weights_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Plate detector weights not found at '{weights}'. "
                    f"Please download them by running: python scripts/download_models.py"
                )
        yolo_model = YOLO(str(weights_path))
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


def run_detections(
    image: Any, models: tuple[Any, Any, Any, Any], settings: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float], str]:
    yolo_model, deeplab_model, deeplab_device, ocr_reader = models
    detections: dict[str, Any] = {}
    timings: dict[str, float] = {}
    started = time.perf_counter()
    reader_lock = threading.Lock()

    def detect_plates() -> list[Any]:
        item_started = time.perf_counter()
        result = detect_plate_boxes(
            yolo_model,
            image,
            settings["conf"],
            settings["imgsz"],
            settings["device"],
            ocr_reader if settings["ocr"] else None,
            {"license plate", "number plate", "plate"},
            reader_lock,
        )
        timings["plate_detection_seconds"] = time.perf_counter() - item_started
        return result

    def detect_humans() -> Any:
        item_started = time.perf_counter()
        result = detect_human_mask(
            deeplab_model,
            deeplab_device,
            image,
            dilation_size=int(settings.get("human_dilation_size", 5)),
        )
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
    if len(tasks) < 2:
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
    processing_started = time.perf_counter()
    image_size_bytes = source_path.stat().st_size
    yolo_model, deeplab_model, deeplab_device, ocr_reader = models
    original_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if original_image is None:
        raise ValueError(f"Could not read image: {source_path}")

    exif_started = time.perf_counter()
    geo_exif = extract_geo_exif(source_path) if settings["exif_strip"] else None
    exif_extracted = time.perf_counter() - exif_started

    initial_height, initial_width = original_image.shape[:2]
    detection_image, scale_x, scale_y = prepare_detection_image(original_image, max_side=1600)
    detections, timings, parallel_mode = run_detections(detection_image, models, settings)

    if settings["plate_mask"]:
        detections["plates"] = map_plate_detections_to_original(
            detections["plates"], scale_x, scale_y
        )

    if settings["human_mask"]:
        detections["human_mask"] = map_human_mask_to_original(
            detections["human_mask"],
            (initial_height, initial_width),
            scale_x,
            scale_y,
        )

    if settings["watermark_removal"]:
        detections["watermarks"] = map_watermark_regions_to_original(
            detections["watermarks"], scale_x, scale_y
        )

    redacted_image = original_image.copy()
    if settings["plate_mask"]:
        started = time.perf_counter()
        redacted_image = redact_plate_boxes(
            redacted_image, detections["plates"], settings["mask_mode"]
        )
        timings["plate_redaction_seconds"] = time.perf_counter() - started

    if settings["human_mask"]:
        started = time.perf_counter()
        redacted_image = redact_human_mask(
            redacted_image,
            detections["human_mask"],
            settings.get("human_mask_mode", "blur"),
            blur_kernel=int(settings.get("human_blur_kernel", 51)),
        )
        timings["human_redaction_seconds"] = time.perf_counter() - started

    if settings["watermark_removal"]:
        started = time.perf_counter()
        redacted_image = redact_watermark_regions(redacted_image, detections["watermarks"])
        timings["watermark_redaction_seconds"] = time.perf_counter() - started

    if settings["resizing"]:
        started = time.perf_counter()
        redacted_image, quality, quality_score = resize_for_quality(
            redacted_image, settings["high_thresh"], settings["low_thresh"]
        )
        timings["resize_seconds"] = time.perf_counter() - started
    else:
        quality, quality_score = "UNCHANGED", None

    started = time.perf_counter()
    save_with_geo_exif(redacted_image, output_path, settings["exif_strip"], geo_exif)
    timings["save_seconds"] = time.perf_counter() - started
    timings["exif_extract_seconds"] = exif_extracted
    with output_path.open("rb") as output_file:
        output_sha256 = hashlib.sha256(output_file.read()).hexdigest()

    plate_text = " | ".join(
        detection.text for detection in detections.get("plates", []) if detection.text
    )
    return {
        "image_name": str(source_path),
        "image_width_px": initial_width,
        "image_height_px": initial_height,
        "image_size_bytes": image_size_bytes,
        "image_size_mb": image_size_bytes / (1024 * 1024),
        "processing_time_seconds": time.perf_counter() - processing_started,
        "plate_text": plate_text,
        "quality": quality,
        "quality_score": quality_score,
        "parallel_mode": parallel_mode,
        "plate_detection_count": len(detections.get("plates", [])),
        "human_detection_count": (
            int(np.count_nonzero(detections.get("human_mask", 0) > 0))
            if "human_mask" in detections
            else 0
        ),
        "watermark_detection_count": len(detections.get("watermarks", [])),
        "output_sha256": output_sha256,
        **timings,
    }


def initialize_worker(settings: dict[str, Any], log_queue: Any, status_queue: Any) -> None:
    """Load enabled models once in each worker process."""
    global _WORKER_MODELS, _WORKER_SETTINGS, _WORKER_LOGGER, _WORKER_STATUS_QUEUE, _WORKER_PEAK_RSS_BYTES
    _WORKER_LOGGER = configure_worker_logging(log_queue)
    _WORKER_STATUS_QUEUE = status_queue
    _WORKER_SETTINGS = dict(settings)
    _WORKER_LOGGER.info("WORKER start pid=%s", os.getpid())
    loaded = load_models(
        _WORKER_SETTINGS["plate_mask"],
        _WORKER_SETTINGS["human_mask"],
        _WORKER_SETTINGS["watermark_removal"],
        _WORKER_SETTINGS["ocr"],
        _WORKER_SETTINGS["weights"],
        _WORKER_SETTINGS["device"],
        _WORKER_SETTINGS["ocr_langs"],
    )
    _WORKER_MODELS = loaded[:4]
    _WORKER_PEAK_RSS_BYTES = psutil.Process().memory_info().rss
    model_timings = loaded[4]
    _WORKER_LOGGER.info(
        "WORKER models_loaded pid=%s %s",
        os.getpid(),
        "; ".join(f"{key}={value:.3f}s" for key, value in model_timings.items()) or "none",
    )
    print(
        f"Worker {os.getpid()} loaded models: "
        + ("; ".join(f"{key}={value:.3f}s" for key, value in model_timings.items()) or "none"),
        flush=True,
    )


def process_worker(task: tuple[int, str, str]) -> dict[str, Any]:
    """Process one image using the models owned by the current worker."""
    global _WORKER_PEAK_RSS_BYTES
    index, source_name, output_name = task
    if _WORKER_MODELS is None or _WORKER_SETTINGS is None:
        raise RuntimeError("Worker models were not initialized")
    source_path = Path(source_name)
    output_path = Path(output_name)
    if _WORKER_STATUS_QUEUE is not None:
        _WORKER_STATUS_QUEUE.put((index, source_name, os.getpid()))
    if _WORKER_LOGGER is not None:
        _WORKER_LOGGER.info("QUEUE worker_start task=%s image=%s", index, source_name)
    result = process_image(source_path, output_path, _WORKER_MODELS, _WORKER_SETTINGS)
    _WORKER_PEAK_RSS_BYTES = max(_WORKER_PEAK_RSS_BYTES, psutil.Process().memory_info().rss)
    gpu_memory = cuda_memory_snapshot(_WORKER_SETTINGS["device"])
    result["index"] = index
    result["image_name"] = source_name
    result["worker_pid"] = os.getpid()
    result["worker_rss_bytes"] = psutil.Process().memory_info().rss
    result["worker_peak_rss_bytes"] = _WORKER_PEAK_RSS_BYTES
    result["gpu_memory"] = gpu_memory
    if _WORKER_LOGGER is not None:
        _WORKER_LOGGER.info("QUEUE worker_complete task=%s image=%s", index, source_name)
    return result


def choose_worker_count(settings: dict[str, Any]) -> int:
    import torch

    configured = int(settings["workers"])
    if configured > 0:
        requested = configured
    else:
        requested = 2
    if settings["device"].startswith("cuda") and torch.cuda.is_available():
        return min(requested, max(1, int(settings["max_gpu_workers"])))
    return requested


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
        "human_mask_mode": str(resolve(args.human_mask_mode, config, "human_mask_mode")),
        "human_blur_kernel": int(resolve(args.human_blur_kernel, config, "human_blur_kernel")),
        "human_dilation_size": int(
            resolve(args.human_dilation_size, config, "human_dilation_size")
        ),
        "plate_mask": bool(resolve(args.plate_mask, config, "plate_mask")),
        "resizing": bool(resolve(args.resizing, config, "resizing")),
        "workers": int(resolve(args.workers, config, "workers")),
        "max_gpu_workers": int(resolve(args.max_gpu_workers, config, "max_gpu_workers")),
        "worker_start_method": str(resolve(None, config, "worker_start_method")),
        "log_file": resolve(args.log_file, config, "log_file"),
    }

    if settings["device"] == "auto":
        import torch

        settings["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    start_method = str(settings["worker_start_method"])
    context = mp.get_context(start_method)
    status_queue = context.Queue()
    log_file_value = str(settings["log_file"]).strip()
    log_path = resolve_path(log_file_value) if log_file_value else temp_dir / "pipeline.log"
    logger, log_queue, log_listener = configure_run_logging(log_path, context)
    worker_count = choose_worker_count(settings)
    memory_monitor = MemoryMonitor(psutil.Process())
    memory_monitor.start()
    queue_tracker = QueueTracker(logger, status_queue)
    queue_tracker.start()
    print(f"Starting {worker_count} worker process(es) with start method={start_method}")
    logger.info(
        "RUN start input=%s output=%s workers=%s start_method=%s",
        input_dir,
        output_dir,
        worker_count,
        start_method,
    )

    csv_path = temp_dir / "plate_results.csv"
    report_path = temp_dir / "processing_report.csv"
    with (
        csv_path.open("w", newline="", encoding="utf-8") as csv_file,
        report_path.open("w", newline="", encoding="utf-8") as report_file,
    ):
        writer = csv.writer(csv_file)
        report_writer = csv.DictWriter(
            report_file,
            fieldnames=[
                "image_name",
                "image_width_px",
                "image_height_px",
                "image_size_bytes",
                "image_size_mb",
                "processing_time_seconds",
                "status",
                "error",
            ],
        )
        writer.writerow(["image_name", "plate_text"])
        report_writer.writeheader()
        successful = 0
        errors = 0
        total_plate_detections = 0
        total_human_mask_pixels = 0
        total_watermark_detections = 0
        worker_peak_rss_by_pid: dict[int, int] = {}
        gpu_peak_allocated_by_pid: dict[int, int] = {}
        gpu_peak_reserved_by_pid: dict[int, int] = {}
        tasks = [
            (index, str(source_path), str(output_dir / source_path.relative_to(input_dir)))
            for index, source_path in enumerate(image_files, 1)
        ]
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(settings, log_queue, status_queue),
        ) as executor:
            futures = {}
            for task in tasks:
                futures[executor.submit(process_worker, task)] = task
                queue_tracker.submit(task[0], task[1])
            completed_results: dict[int, dict[str, Any]] = {}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    completed_results[result["index"]] = result
                    successful += 1
                    total_plate_detections += result["plate_detection_count"]
                    total_human_mask_pixels += result["human_detection_count"]
                    total_watermark_detections += result["watermark_detection_count"]
                    worker_pid = int(result["worker_pid"])
                    worker_peak_rss_by_pid[worker_pid] = max(
                        worker_peak_rss_by_pid.get(worker_pid, 0),
                        int(result["worker_peak_rss_bytes"]),
                    )
                    gpu_memory = result.get("gpu_memory")
                    if gpu_memory is not None:
                        gpu_peak_allocated_by_pid[worker_pid] = max(
                            gpu_peak_allocated_by_pid.get(worker_pid, 0),
                            int(gpu_memory["peak_allocated_bytes"]),
                        )
                        gpu_peak_reserved_by_pid[worker_pid] = max(
                            gpu_peak_reserved_by_pid.get(worker_pid, 0),
                            int(gpu_memory["peak_reserved_bytes"]),
                        )
                    print(
                        f"[{result['index']}/{len(image_files)}] Completed {Path(result['image_name']).name}"
                    )
                    queue_tracker.complete(result["index"], result["image_name"])
                except Exception as exc:
                    errors += 1
                    completed_results[task[0]] = {"plate_text": "", "error": str(exc)}
                    print(f"[{task[0]}/{len(image_files)}] Failed: {exc}")
                    queue_tracker.fail(task[0], task[1])
                    logger.exception("QUEUE exception task=%s image=%s", task[0], task[1])

        for index, source_path in enumerate(image_files, 1):
            relative_path = source_path.relative_to(input_dir)
            result = completed_results[index]
            writer.writerow([relative_path.as_posix(), result.get("plate_text", "")])
            image_size_bytes = result.get("image_size_bytes", source_path.stat().st_size)
            report_writer.writerow(
                {
                    "image_name": relative_path.as_posix(),
                    "image_width_px": result.get("image_width_px", ""),
                    "image_height_px": result.get("image_height_px", ""),
                    "image_size_bytes": image_size_bytes,
                    "image_size_mb": f"{image_size_bytes / (1024 * 1024):.6f}",
                    "processing_time_seconds": result.get("processing_time_seconds", ""),
                    "status": "failed" if result.get("error") else "success",
                    "error": result.get("error", ""),
                }
            )

    memory_monitor.stop()
    queue_tracker.stop()
    worker_peak_rss_by_pid.update(
        {
            pid: max(worker_peak_rss_by_pid.get(pid, 0), rss_bytes)
            for pid, rss_bytes in memory_monitor.worker_peak_rss_bytes.items()
            if pid in worker_peak_rss_by_pid
        }
    )
    total_worker_peak_rss_bytes = sum(worker_peak_rss_by_pid.values())
    total_seconds = time.perf_counter() - started
    logger.info("QUEUE snapshot final %s", queue_tracker.snapshot())
    virtual_memory = psutil.virtual_memory()
    logger.info(
        "HOST vm_total=%.2fGB vm_used=%.2fGB vm_available=%.2fGB vm_percent=%.1f "
        "cpu_logical=%s cpu_physical_cores=%s cpu_affinity=%s cpu_peak=%.1f%% configured_workers=%s",
        virtual_memory.total / (1024**3),
        virtual_memory.used / (1024**3),
        virtual_memory.available / (1024**3),
        virtual_memory.percent,
        memory_monitor.logical_cpu_count,
        memory_monitor.physical_core_count,
        ",".join(str(cpu) for cpu in memory_monitor.cpu_affinity),
        memory_monitor.system_peak_cpu_percent,
        worker_count,
    )
    gpu_available = bool(gpu_peak_allocated_by_pid or gpu_peak_reserved_by_pid)
    logger.info(
        "MEMORY parent_peak_rss=%.2fMB worker_peak_sum=%.2fMB system_total=%.2fGB system_peak_used=%.2fGB",
        megabytes(memory_monitor.parent_peak_rss_bytes),
        megabytes(total_worker_peak_rss_bytes),
        memory_monitor.system_total_bytes / (1024**3),
        memory_monitor.system_peak_used_bytes / (1024**3),
    )
    for worker_pid, worker_peak_rss_bytes in sorted(worker_peak_rss_by_pid.items()):
        logger.info(
            "MEMORY worker_pid=%s peak_rss=%.2fMB",
            worker_pid,
            megabytes(worker_peak_rss_bytes),
        )
    if gpu_available:
        logger.info(
            "MEMORY gpu_peak_allocated_sum=%.2fMB gpu_peak_reserved_sum=%.2fMB",
            megabytes(sum(gpu_peak_allocated_by_pid.values())),
            megabytes(sum(gpu_peak_reserved_by_pid.values())),
        )
    logger.info(
        "RUN complete processed=%s successful=%s errors=%s duration=%.3fs",
        len(image_files),
        successful,
        errors,
        total_seconds,
    )
    log_listener.stop()
    print(f"Processed: {len(image_files)} | Successful: {successful} | Errors: {errors}")
    print(
        f"Detection counts: plates={total_plate_detections}; human_mask_pixels={total_human_mask_pixels}; watermarks={total_watermark_detections}"
    )
    print(f"Pipeline complete in {total_seconds:.3f} seconds")
    print(f"Final images saved to: {output_dir}")
    print(f"Plate results saved to: {csv_path}")
    print(f"Processing report saved to: {report_path}")
    print(f"Run log saved to: {log_path}")
    print(f"Parent peak RAM: {megabytes(memory_monitor.parent_peak_rss_bytes):.2f} MB")
    for worker_pid, worker_peak_rss_bytes in sorted(worker_peak_rss_by_pid.items()):
        print(f"Worker {worker_pid} peak RAM: {megabytes(worker_peak_rss_bytes):.2f} MB")
    print(f"Worker peak RAM sum: {megabytes(total_worker_peak_rss_bytes):.2f} MB")
    print(
        f"System RAM: {memory_monitor.system_total_bytes / (1024 ** 3):.2f} GB total; peak used: {memory_monitor.system_peak_used_bytes / (1024 ** 3):.2f} GB"
    )
    print(
        f"VM: {virtual_memory.total / (1024 ** 3):.2f} GB total; {virtual_memory.used / (1024 ** 3):.2f} GB used; {virtual_memory.available / (1024 ** 3):.2f} GB available ({virtual_memory.percent:.1f}%)"
    )
    print(
        f"CPUs: {memory_monitor.logical_cpu_count} logical; {memory_monitor.physical_core_count} physical cores; affinity={memory_monitor.cpu_affinity}; peak system CPU={memory_monitor.system_peak_cpu_percent:.1f}%"
    )
    print(f"Configured workers: {worker_count}")
    print(f"Queue final snapshot: {queue_tracker.snapshot()}")
    if gpu_available:
        print(
            f"GPU peak memory: {megabytes(sum(gpu_peak_allocated_by_pid.values())):.2f} MB allocated; {megabytes(sum(gpu_peak_reserved_by_pid.values())):.2f} MB reserved"
        )
    else:
        print("GPU memory: not used")


if __name__ == "__main__":
    main()
