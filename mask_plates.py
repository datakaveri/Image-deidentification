#!/usr/bin/env python3
"""Mask vehicle number plates in a directory of images using a YOLOv8-nano model.

Usage examples:
  python mask_plates.py --weights yolov8n.pt --source train/images --out outputs/masked --mode black
  python mask_plates.py --weights path/to/best.pt --source train/images --out outputs/masked --mode blur --device 0
"""
import argparse
import csv
import os
import sys
from glob import glob
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as e:
    print("Missing dependency: ultralytics. Install from requirements.txt")
    raise

try:
    import easyocr
except ImportError:
    easyocr = None

def ocr_plate_text(reader, roi):
    if reader is None or roi is None or roi.size == 0:
        return ""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if min(h, w) < 50:
        scale = max(1, 128 // min(h, w))
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    try:
        texts = reader.readtext(gray, detail=0, paragraph=False)
    except Exception:
        return ""

    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    return " | ".join(texts)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def mask_box(img, box, mode="black"):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return img
    if mode == "black":
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    elif mode == "blur":
        roi = img[y1:y2, x1:x2]
        # ensure kernel size odd and proportional
        k = max(3, (min(max(1, (w // 10)), 99) // 2) * 2 + 1)
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        img[y1:y2, x1:x2] = blurred
    elif mode == "pixelate":
        roi = img[y1:y2, x1:x2]
        # downscale then upscale
        small = cv2.resize(roi, (max(1, w // 10), max(1, h // 10)), interpolation=cv2.INTER_LINEAR)
        pixel = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        img[y1:y2, x1:x2] = pixel
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return img


def process_image(model, img_path: Path, out_path: Path, conf: float, imgsz: int, mode: str, device: str, ocr_reader=None, allowed_classes=None):
    img = cv2.imread(str(img_path))
    if img is None:
        return ""

    # run inference
    results = model.predict(source=str(img_path), conf=conf, imgsz=imgsz, device=device, verbose=False)
    if not results:
        ensure_dir(str(out_path.parent))
        cv2.imwrite(str(out_path), img)
        return ""

    # take first (and only) result
    r = results[0]
    boxes = getattr(r, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        ensure_dir(str(out_path.parent))
        cv2.imwrite(str(out_path), img)
        return ""

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else []
    cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else None
    names = getattr(r, 'names', None)

    plate_texts = []
    for i, box in enumerate(xyxy):
        if allowed_classes is not None and cls is not None and names is not None:
            label = str(names[int(cls[i])]).lower()
            label_norm = label.replace('_', ' ')
            if label not in allowed_classes and label_norm not in allowed_classes:
                continue

        if ocr_reader is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            roi = img[y1:y2, x1:x2]
            text = ocr_plate_text(ocr_reader, roi)
            if text:
                plate_texts.append(text)

        mask_box(img, box, mode=mode)

    ensure_dir(str(out_path.parent))
    cv2.imwrite(str(out_path), img)
    return " | ".join(plate_texts)


def main():
    p = argparse.ArgumentParser(description="Mask number plates using YOLOv8-nano weights")
    p.add_argument("--weights", default="yolov8n.pt", help="Path or HF repo: YOLOv8 weights file (pt) or model id")
    p.add_argument("--source", default="train/images", help="Input image directory")
    p.add_argument("--out", default="outputs/masked", help="Output directory to save masked images")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="auto", help="Device to run on (e.g. 0 or cpu or auto)")
    p.add_argument("--mode", choices=["black", "blur", "pixelate"], default="black", help="Masking mode")
    p.add_argument("--classes", default="license plate,number plate,plate", help="Comma-separated class names to mask")
    p.add_argument("--ocr", action="store_true", help="Run EasyOCR on detected plate regions")
    p.add_argument("--ocr-langs", default="en", help="Comma-separated EasyOCR language codes")
    p.add_argument("--output-csv", default=None, help="CSV file path to store image name and plate text")
    p.add_argument("--ext", default="jpg,png", help="Image extensions to process (comma-separated)")
    args = p.parse_args()

    weights = args.weights
    src_dir = Path(args.source)
    out_dir = Path(args.out)
    ensure_dir(str(out_dir))

    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        sys.exit(1)

    # determine device
    requested = str(args.device).lower()
    if requested == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'
    else:
        device = requested
        if device != 'cpu' and not torch.cuda.is_available():
            print(f"CUDA not available; falling back to CPU (requested device={args.device})")
            device = 'cpu'

    print(f"Loading model on device={device}...")
    model = YOLO(weights)

    ocr_reader = None
    if args.ocr:
        if easyocr is None:
            print("Missing dependency: easyocr. Install it with 'pip install easyocr'")
            sys.exit(1)
        ocr_langs = [lang.strip() for lang in args.ocr_langs.split(',') if lang.strip()]
        ocr_reader = easyocr.Reader(ocr_langs, gpu=(device != 'cpu' and torch.cuda.is_available()))

    exts = [e.strip().lower() for e in args.ext.split(',') if e.strip()]
    files = []
    for ext in exts:
        files.extend(glob(str(src_dir / f"**/*.{ext}"), recursive=True))

    files = sorted(files)
    if not files:
        print("No images found in source directory.")
        sys.exit(1)

    allowed_classes = {c.strip().lower() for c in args.classes.split(',') if c.strip()}
    csv_writer = None
    csv_handle = None
    if args.output_csv:
        csv_path = Path(args.output_csv)
        ensure_dir(str(csv_path.parent))
        csv_handle = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(["image_name", "plate_text"])

    print(f"Processing {len(files)} images -> {out_dir} (mode={args.mode})")

    for f in tqdm(files, desc="images"):
        rel = Path(f).relative_to(src_dir)
        out_path = out_dir / rel
        plate_text = ""
        try:
            plate_text = process_image(
                model,
                Path(f),
                out_path,
                conf=args.conf,
                imgsz=args.imgsz,
                mode=args.mode,
                device=device,
                ocr_reader=ocr_reader,
                allowed_classes=allowed_classes,
            )
        except Exception as e:
            print(f"Warning: failed to process {f}: {e}")

        if csv_writer is not None:
            csv_writer.writerow([rel.as_posix(), plate_text])

    if csv_handle is not None:
        csv_handle.close()
        print("Done. CSV saved to:", csv_path)

    print("Done. Masked images saved to:", out_dir)


if __name__ == '__main__':
    main()
