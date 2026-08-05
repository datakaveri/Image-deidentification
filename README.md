# Road Defect Anonymization

This repository contains scripts for license plate detection and anonymization using YOLOv8, EasyOCR, and a watermark-removal pipeline.

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run plate masking and OCR

```bash
./venv/bin/python3 app/sensitive_data_masking/mask_plates.py \
  --weights app/sensitive_data_masking/license_plate_detector.pt \
  --source train/images \
  --out outputs/masked \
  --mode black \
  --ocr \
  --ocr-langs en \
  --output-csv results.csv \
  --device auto \
  --conf 0.1 \
  --imgsz 640 \
  --classes "license_plate,license plate,plate"
```

## Run watermark removal

```bash
./venv/bin/python3 app/watermark_removal/remove_watermark.py \
  app/watermark_removal/IMG20260730125027.jpg \
  outputs/cleaned.jpg \
  --detection-output outputs/detection_boxes.jpg
```

## Preserve only GPS/geo EXIF tags

```bash
./venv/bin/python3 app/exif_geo_tag/store_geo_tag_exif.py \
  input_image.jpg \
  output_image.jpg
```

Or process a whole directory:

```bash
./venv/bin/python3 app/exif_geo_tag/store_geo_tag_exif.py \
  input_dir \
  output_dir \
  --recursive
```

## Docker

Build the image:

```bash
docker build -t road-defect-anonymization .
```

Run the plate-masking script inside the container:

```bash
docker run --rm -it \
  -v $(pwd)/train:/app/train \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/app/sensitive_data_masking:/app/app/sensitive_data_masking \
  road-defect-anonymization \
  python app/sensitive_data_masking/mask_plates.py --weights app/sensitive_data_masking/license_plate_detector.pt --source train/images --out outputs/masked --mode black --ocr --ocr-langs en --output-csv results.csv --device auto --conf 0.1 --imgsz 640 --classes "license_plate,license plate,plate"
```

Run watermark removal inside the container:

```bash
docker run --rm -it \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/app/watermark_removal:/app/app/watermark_removal \
  road-defect-anonymization \
  python app/watermark_removal/remove_watermark.py app/watermark_removal/IMG20260730125027.jpg outputs/cleaned.jpg --detection-output outputs/detection_boxes.jpg
```

Run the EXIF geo-tag preservation utility inside the container:

```bash
docker run --rm -it \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  road-defect-anonymization \
  python app/exif_geo_tag/store_geo_tag_exif.py /app/input/input_image.jpg /app/output/output_image.jpg
```

## Docker Compose

Start the service with:

```bash
docker compose up --build
```

This will build the image and launch a shell-ready container with the repository mounted.

Run the plate-masking command inside the Compose service:

```bash
docker compose run --rm road-defect-app \
  python app/sensitive_data_masking/mask_plates.py --weights app/sensitive_data_masking/license_plate_detector.pt --source train/images --out outputs/masked --mode black --ocr --ocr-langs en --output-csv results.csv --device auto --conf 0.1 --imgsz 640 --classes "license_plate,license plate,plate"
```

Run watermark removal inside the Compose service:

```bash
docker compose run --rm road-defect-app \
  python app/watermark_removal/remove_watermark.py app/watermark_removal/IMG20260730125027.jpg outputs/cleaned.jpg --detection-output outputs/detection_boxes.jpg
```

Run EXIF geo-tag preservation inside the Compose service:

```bash
docker compose run --rm road-defect-app \
  python app/exif_geo_tag/store_geo_tag_exif.py /app/input/input_image.jpg /app/output/output_image.jpg
```


## Project structure

```text
road-defect_anonymization/
├── .dockerignore
├── Dockerfile
├── README.md
├── app/
│   ├── exif_geo_tag/
│   │   ├── __pycache__/
│   │   └── store_geo_tag_exif.py
│   ├── sensitive_data_masking/
│   │   ├── deeplab.py
│   │   ├── inspect_model.py
│   │   ├── license_plate_detector.pt
│   │   ├── mask_plates.py
│   │   ├── sample.py
│   │   └── yolov8n.pt
│   └── watermark_removal/
│       ├── IMG20260730125027.jpg
│       ├── IMG20260730154858.jpg
│       ├── IMG20260804122250.jpg
│       ├── __pycache__/
│       ├── cleaned.jpg
│       ├── cleaned2.jpg
│       ├── core.py
│       ├── detection_boxes.jpg
│       ├── image.py
│       ├── masking.py
│       ├── remove_watermark.py
│       ├── test_output.png
│       ├── test_watermark.png
│       └── text_region_detector.py
├── outputs/
├── requirements.txt
├── results.csv
├── train/
├── venv/
└── .venv/
```

## Notes

- Do not commit model weights or image data to the repository unless explicitly required.
- The Docker setup is designed to run without a local Python environment, mounting only the input/output folders needed at runtime.
- The repository contains three primary workflows:
  - `app/sensitive_data_masking/` for license plate detection, masking, and OCR
  - `app/watermark_removal/` for watermark detection and removal
  - `app/exif_geo_tag/` for preserving only GPS/geo EXIF tags while stripping other metadata
