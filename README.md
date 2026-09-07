# Road Defect Anonymization

This project builds an end-to-end image anonymization pipeline for road-defect datasets. The complete pipeline flow is:

`EXIF cleaning` -> `license plate masking` -> `human masking` -> `watermark removal (EasyOCR-based)` -> `resizing`

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

- Intermediate outputs are written under temp-dir named folder for debugging
- Final resized images are written under output-dir named folder
- Each pipeline step reports its elapsed time. DeepLab human masking and YOLO plate masking report detection and redaction times separately, both per image and as totals. Watermark processing also reports EasyOCR detection and pixel redaction times separately.

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
  --ocr \
  --exif-strip \
  --watermark-removal \
  --human-mask \
  --plate-mask \
  --resizing
```

## Run individual steps

### 1. Preserve only GPS/geo EXIF tags

```bash
python app/exif_geo_tag/store_geo_tag_exif.py \
  input_dir \
  output_dir \
  --recursive
```

### 2. Mask license plates

```bash
python app/sensitive_data_masking/mask_plates.py \
  --weights app/sensitive_data_masking/license_plate_detector.pt \
  --source outputs/temp/exif \
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

### 3. Mask humans with DeepLab

```bash
python app/sensitive_data_masking/deeplab.py \
  --input-dir outputs/temp/plate_masked \
  --output-dir outputs/temp/human_masked \
  --mask-type blur
```

### 4. Remove watermark (EasyOCR-based)

```bash
python app/watermark_removal/remove_watermark.py \
  outputs/temp/human_masked \
  outputs/temp/watermark_removed \
  --ocr-langs en \
  --ext jpg,jpeg,png
```

The above step uses one EasyOCR model instance for the whole folder, detecting text regions in image borders before redacting watermark text. It reports model loading, detection, and redaction times.

### 5. Resize images

```bash
python app/resizing.py \
  outputs/temp/plate_masked \
  outputs/final \
  --high-thresh 500.0 \
  --low-thresh 100.0
```

## Docker

This container installs the EasyOCR-based watermark removal pipeline and does not use PaddleOCR.

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
    --config pipeline_config.json
```

## Docker Compose

Start the service with:

```bash
docker compose up --build
```

To run the pipeline from the compose service:

```bash
docker compose run --rm road-defect-app \
  python main.py \    --config pipeline_config.json
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
