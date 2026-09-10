# Road Defect Anonymization

This project performs end-to-end anonymization for road-defect images using a reusable in-memory pipeline packaged under `app/deidentification`.

The current workflow is:

`decode image` -> `extract GPS-only EXIF metadata` -> `parallel plate / human / watermark detection` -> `sequential redaction` -> `resize in memory` -> `save final image once with preserved GPS metadata`

```mermaid
flowchart LR
    A[Input image directory] --> B[main.py
worker scheduler]
    B --> C[ProcessPoolExecutor]
    C --> D[Worker 1: load models once]
    C --> E[Worker 2: load models once]
    D --> F[Plate / Human / Watermark detection]
    E --> F
    F --> G[Sequential redaction]
    G --> H[Resize in memory]
    H --> I[Save final image + GPS EXIF]
    I --> J[CSV output + metrics]

    subgraph App[app/deidentification package]
        F
        G
        H
        I
    end
```

The main entrypoint is [main.py](main.py). It loads enabled models once per worker process, processes images in parallel, and avoids writing intermediate masked images to disk unless the caller explicitly uses a temporary debug directory structure.

## Requirements

- Python 3.10+
- OpenCV, PyTorch, torchvision, ultralytics, EasyOCR, and Pillow
- CUDA-capable GPU is optional; the pipeline falls back to CPU automatically
- The YOLO plate model is expected at [app/sensitive_data_masking/license_plate_detector.pt](app/sensitive_data_masking/license_plate_detector.pt)

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The pipeline reads settings from `pipeline_config.json` and supports command-line overrides.

Example config keys:

- `input_dir`, `output_dir`, `temp_dir`
- `weights`, `device`, `mask_mode`
- `ocr`, `ocr_langs`
- `conf`, `imgsz`, `high_thresh`, `low_thresh`
- `ext`
- `exif_strip`, `watermark_removal`, `human_mask`, `plate_mask`, `resizing`
- `workers`, `max_gpu_workers`, `worker_start_method`

A typical config is:

```json
{
  "input_dir": "train/images",
  "output_dir": "outputs/final",
  "temp_dir": "outputs/temp",
  "weights": "app/sensitive_data_masking/license_plate_detector.pt",
  "device": "auto",
  "mask_mode": "black",
  "ocr": false,
  "ocr_langs": "en",
  "conf": 0.1,
  "imgsz": 640,
  "high_thresh": 500.0,
  "low_thresh": 100.0,
  "ext": "jpg,jpeg,png,bmp,tif,tiff,webp",
  "exif_strip": true,
  "watermark_removal": true,
  "human_mask": true,
  "plate_mask": true,
  "resizing": true,
  "workers": 0,
  "max_gpu_workers": 1,
  "worker_start_method": "spawn"
}
```

## Current package-based pipeline

The reusable API lives in [app/deidentification/__init__.py](app/deidentification/__init__.py). The package exports the main detection, redaction, resize, and EXIF helpers:

```python
from app.deidentification import (
    detect_human_mask,
    detect_plate_boxes,
    detect_watermark_regions,
    redact_human_mask,
    redact_plate_boxes,
    redact_watermark_regions,
    resize_for_quality,
    extract_geo_exif,
    save_with_geo_exif,
)
```

The implementation is split into stage modules:

- [app/deidentification/plate_stage.py](app/deidentification/plate_stage.py): YOLO plate detection and redaction
- [app/deidentification/deeplab_stage.py](app/deidentification/deeplab_stage.py): DeepLab human segmentation and masking
- [app/deidentification/watermark_stage.py](app/deidentification/watermark_stage.py): EasyOCR-based watermark detection and redaction
- [app/deidentification/exif_stage.py](app/deidentification/exif_stage.py): GPS-only EXIF extraction and final save with metadata preservation
- [app/deidentification/resize_stage.py](app/deidentification/resize_stage.py): optional resize step before final save

Per image, the pipeline now follows this order:

1. Read the image
2. Extract only GPS/geo EXIF metadata if enabled
3. Run enabled detections in parallel
4. Apply redactions sequentially
5. Resize in memory if enabled
6. Save the final anonymized image once, preserving the captured EXIF metadata

## Parallel detection and worker processes

The orchestration in [main.py](main.py) runs one task per image over a ProcessPoolExecutor.

Behavior:

- `workers: 0` uses a default worker count of 2 unless GPU mode is enabled
- `max_gpu_workers` caps process count on CUDA
- `worker_start_method` is configurable and defaults to `spawn`
- each worker initializes the enabled models once and reuses them for all images in that process
- EasyOCR is shared per worker to avoid reloading it for every image
- detection timings are recorded per stage, as well as model-load timing and end-to-end detection wall time

The pipeline records:

- plate detection time
- human detection time
- watermark detection time
- total detection wall time
- plate redaction time
- human redaction time
- watermark redaction time
- EXIF extraction time
- save time
- model-load time for YOLO, DeepLab, and EasyOCR

## Running the full pipeline

```bash
python main.py --config pipeline_config.json
```

Optional CLI overrides:

```bash
python main.py \
  --config pipeline_config.json \
  --input-dir train/images \
  --output-dir outputs/final \
  --temp-dir outputs/temp \
  --device auto \
  --mask-mode black \
  --conf 0.1 \
  --imgsz 640 \
  --ocr \
  --exif-strip \
  --watermark-removal \
  --human-mask \
  --plate-mask \
  --resizing
```

## Output behavior

- Final anonymized images are written to `output_dir`
- A CSV of detected plate text is written to `temp_dir/plate_results.csv`
- EXIF metadata is preserved only for GPS/geo tags before the final save step
- Intermediate stage images are not normally written to disk as part of the main package pipeline

## Docker

Build the image:

```bash
docker build -t road-defect-anonymization .
```

Run the pipeline in the container:

```bash
docker run --rm -it \
  -v $(pwd)/train:/app/train \
  -v $(pwd)/outputs:/app/outputs \
  road-defect-anonymization \
  python main.py --config pipeline_config.json
```

## Docker Compose

Start the stack:

```bash
docker compose up --build
```

Run the pipeline inside the compose service:

```bash
docker compose run --rm road-defect-app \
  python main.py --config pipeline_config.json
```

## Project structure

```text
road-defect_anonymization/
├── app/
│   ├── deidentification/
│   │   ├── __init__.py
│   │   ├── contracts.py
│   │   ├── deeplab_stage.py
│   │   ├── exif_stage.py
│   │   ├── plate_stage.py
│   │   ├── resize_stage.py
│   │   └── watermark_stage.py
│   ├── sensitive_data_masking/
│   │   └── license_plate_detector.pt
│   └── ...
├── main.py
├── pipeline_config.json
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
├── outputs/
├── train/
└── app/images_test/
```

## Notes

- The main workflow is package-based and in-memory; it is not the older multi-step script chain.
- EasyOCR is the active OCR engine for watermark detection.
- Plate detection uses Ultralytics YOLO, and human masking uses torchvision DeepLabV3.
- The project preserves only the geo/GPS EXIF footprint before the final save so sensitive metadata is stripped while location data remains available when needed.
