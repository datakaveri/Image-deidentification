#!/usr/bin/env python3
"""Mask vehicle number plates in a directory of images using a YOLOv8-nano model.

Usage examples:
  python mask_plates.py --weights yolov8n.pt --source train/images --out outputs/masked --mode black
  python mask_plates.py --weights path/to/best.pt --source train/images --out outputs/masked --mode blur --device 0
"""
import argparse
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


def process_image(model, img_path: Path, out_path: Path, conf: float, imgsz: int, mode: str, device: str):
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    # run inference
    results = model.predict(source=str(img_path), conf=conf, imgsz=imgsz, device=device, verbose=False)
    if not results:
        ensure_dir(str(out_path.parent))
        cv2.imwrite(str(out_path), img)
        return True

    # take first (and only) result
    r = results[0]
    boxes = getattr(r, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        ensure_dir(str(out_path.parent))
        cv2.imwrite(str(out_path), img)
        return True

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else []

    for box in xyxy:
        mask_box(img, box, mode=mode)

    ensure_dir(str(out_path.parent))
    cv2.imwrite(str(out_path), img)
    return True


def main():
    p = argparse.ArgumentParser(description="Mask number plates using YOLOv8-nano weights")
    p.add_argument("--weights", default="yolov8n.pt", help="Path or HF repo: YOLOv8 weights file (pt) or model id")
    p.add_argument("--source", default="train/images", help="Input image directory")
    p.add_argument("--out", default="outputs/masked", help="Output directory to save masked images")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="auto", help="Device to run on (e.g. 0 or cpu or auto)")
    p.add_argument("--mode", choices=["black", "blur", "pixelate"], default="black", help="Masking mode")
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

    exts = [e.strip().lower() for e in args.ext.split(',') if e.strip()]
    files = []
    for ext in exts:
        files.extend(glob(str(src_dir / f"**/*.{ext}"), recursive=True))

    files = sorted(files)
    if not files:
        print("No images found in source directory.")
        sys.exit(1)

    print(f"Processing {len(files)} images -> {out_dir} (mode={args.mode})")

    for f in tqdm(files, desc="images"):
        rel = Path(f).relative_to(src_dir)
        out_path = out_dir / rel
        try:
            process_image(model, Path(f), out_path, conf=args.conf, imgsz=args.imgsz, mode=args.mode, device=device)
        except Exception as e:
            print(f"Warning: failed to process {f}: {e}")
            continue

    print("Done. Masked images saved to:", out_dir)


if __name__ == '__main__':
    main()
