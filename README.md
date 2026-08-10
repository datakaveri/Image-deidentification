# Road Defect Anonymization

This project builds an end-to-end image anonymization pipeline for road-defect datasets. The complete pipeline flow is:

`EXIF cleaning` -> `watermark removal (OCR-based)` -> `human masking` -> `license plate masking` -> `resizing`

The main entrypoint is [main.py](main.py), which runs the full sequence in one command.

## Requirements

- Python 3.9+
- CUDA-capable GPU is optional; the pipeline will fall back to CPU when unavailable.
- The repository expects the following model assets:
  - [app/sensitive_data_masking/license_plate_detector.pt](app/sensitive_data_masking/license_plate_detector.pt)

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

## Run the full pipeline

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
  --ocr
```

### Configuration

The pipeline supports configuration via `pipeline_config.json` and command-line overrides. Example config fields:

- `input_dir`, `output_dir`, `temp_dir`
- `weights`, `device`, `mask_mode`
- `ocr`, `ocr_langs`
- `conf`, `imgsz`, `high_thresh`, `low_thresh`
- `ext`
- `exif_strip`, `watermark_removal`, `human_mask`, `plate_mask`, `resizing`

Local run using config:

```bash
python main.py --config pipeline_config.json
```

Override a config value from the command line:

```bash
python main.py --config pipeline_config.json --ocr --mask-mode blur
```

### What the pipeline produces

- Intermediate outputs are written under [outputs](outputs)
- Final resized images are written under [outputs](outputs)
- Plate detection results are saved under [outputs](outputs)

## Run individual steps

### 2. Remove watermark (OCR-based)

```bash
python app/watermark_removal/remove_watermark.py \
  app/watermark_removal/IMG20260730125027.jpg \
  outputs/cleaned.jpg \
  --detection-output outputs/detection_boxes.jpg
```

This step uses OCR to detect text regions in the image border before redacting watermark text.

### 3. Mask humans with DeepLab

### 1. Preserve only GPS/geo EXIF tags

```bash
python app/exif_geo_tag/store_geo_tag_exif.py \
  input_dir \
  output_dir \
  --recursive
```

### 2. Remove watermark

```bash
python app/watermark_removal/remove_watermark.py \
  app/watermark_removal/IMG20260730125027.jpg \
  outputs/cleaned.jpg \
  --detection-output outputs/detection_boxes.jpg
```

### 3. Mask humans with DeepLab

```bash
python app/sensitive_data_masking/deeplab.py \
  --input-dir outputs/temp/watermark_removed \
  --output-dir outputs/temp/human_masked \
  --mask-type blur
```

### 4. Mask license plates

```bash
python app/sensitive_data_masking/mask_plates.py \
  --weights app/sensitive_data_masking/license_plate_detector.pt \
  --source outputs/temp/human_masked \
  --out outputs/temp/plate_masked \
  --mode black \
  --ocr \
  --ocr-langs en \
  --output-csv outputs/temp/plate_results.csv \
  --device auto \
  --conf 0.1 \
  --imgsz 640 \
  --classes "license plate,number plate,plate"
```

### 5. Resize images

```bash
python app/resizing.py \
  outputs/temp/plate_masked \
  outputs/final \
  --high-thresh 500.0 \
  --low-thresh 100.0
```

## Docker

Build the image:

```bash
docker build -t road-defect-anonymization .
```

Run the full pipeline inside the container:

```bash
docker run --rm -it \
  -v $(pwd)/train:/app/train \
  -v $(pwd)/outputs:/app/outputs \
  road-defect-anonymization \
  python main.py \
    --config pipeline_config.json \
    --input-dir train/images \
    --output-dir outputs/final \
    --temp-dir outputs/temp \
    --device auto \
    --mask-mode black \
    --conf 0.1 \
    --imgsz 640 \
    --ocr
```

## Docker Compose

Start the service with:

```bash
docker compose up --build
```

To run the pipeline from the compose service:

```bash
docker compose run --rm road-defect-app \
  python main.py \    --config pipeline_config.json \    --input-dir train/images \
    --output-dir outputs/final \
    --temp-dir outputs/temp \
    --device auto \
    --mask-mode black \
    --conf 0.1 \
    --imgsz 640 \
    --ocr
```

## Project structure

```text
road-defect_anonymization/
├── app/
│   ├── exif_geo_tag/
│   │   └── store_geo_tag_exif.py
│   ├── sensitive_data_masking/
│   │   ├── deeplab.py
│   │   ├── mask_plates.py
│   │   └── license_plate_detector.pt
│   └── watermark_removal/
│       ├── remove_watermark.py
│       └── ...
├── main.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── outputs/
└── train/
```

## Notes

- The pipeline is designed to work with image directories rather than single files.
- The DeepLab step can be slow on CPU, so a GPU is recommended for larger datasets.
- Do not commit model weights or large image datasets to the repository unless required.
