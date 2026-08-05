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
docker run --rm -it -v $(pwd)/train:/app/train -v $(pwd)/outputs:/app/outputs road-defect-anonymization \
  python app/sensitive_data_masking/mask_plates.py --weights app/sensitive_data_masking/license_plate_detector.pt --source train/images --out outputs/masked --mode black --ocr --ocr-langs en --output-csv results.csv --device auto --conf 0.1 --imgsz 640 --classes "license_plate,license plate,plate"
```

## Project structure

```text
road-defect_anonymization/
├── .dockerignore
├── Dockerfile
├── README.md
├── app/
│   ├── exif_geo_tag/
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

- Do not commit model weights or image data to the repository.
- Use `train/images` only for source data; this directory is ignored by `.gitignore`.
- The repository currently keeps the large model weights and image data out of version control.
