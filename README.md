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
./venv/bin/python3 mask_plates.py \
  --weights license_plate_detector.pt \
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
./venv/bin/python3 watermark_removal/remove_watermark.py \
  watermark_removal/IMG20260730125027.jpg \
  outputs/cleaned.jpg \
  --detection-output outputs/detection_boxes.jpg
```

## Preserve only GPS/geo EXIF tags

```bash
./venv/bin/python3 exif_geo_tag/store_geo_tag_exif.py \
  input_image.jpg \
  output_image.jpg
```

Or process a whole directory:

```bash
./venv/bin/python3 exif_geo_tag/store_geo_tag_exif.py \
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
  python mask_plates.py --weights license_plate_detector.pt --source train/images --out outputs/masked --mode black --ocr --ocr-langs en --output-csv results.csv --device auto --conf 0.1 --imgsz 640 --classes "license_plate,license plate,plate"
```

## Files

- `mask_plates.py`: Detects license plates, masks them, and optionally performs OCR.
- `watermark_removal/remove_watermark.py`: Removes bottom-edge watermark text from images.
- `requirements.txt`: Python dependencies required by the scripts.
- `deeplab.py`, `sample.py`, `inspect_model.py`: supporting scripts.

## Notes

- Do not commit model weights or image data to the repository.
- Use `train/images` only for source data; this directory is ignored by `.gitignore`.
