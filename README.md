# Road Defect Anonymization

This repository contains scripts for license plate detection and anonymization using YOLOv8 and EasyOCR.

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

## Run masking and OCR

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

## Files

- `mask_plates.py`: Detects license plates, masks them, and optionally performs OCR.
- `requirements.txt`: Python dependencies required by the scripts.
- `deeplab.py`, `sample.py`, `inspect_model.py`: supporting scripts.

## Notes

- Do not commit model weights or image data to the repository.
- Use `train/images` only for source data; this directory is ignored by `.gitignore`.
