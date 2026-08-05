#!/usr/bin/env python3
"""Strip all EXIF metadata from images except GPS/geo-location tags.

Usage:
    python exif_geo_tag/store_geo_tag_exif.py input.jpg output.jpg
    python exif_geo_tag/store_geo_tag_exif.py input_dir output_dir --recursive
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image
from PIL.ExifTags import TAGS


def iter_image_files(root: Path, recursive: bool = False) -> Iterable[Path]:
    if root.is_file():
        yield root
        return

    pattern = "**/*" if recursive else "*"
    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
            yield path


def preserve_only_geo_tags(input_path: Path, output_path: Path) -> None:
    with Image.open(input_path) as img:
        exif = img.getexif()
        if not exif:
            # No EXIF to preserve; just copy the image without metadata.
            img.save(output_path, quality=95, optimize=True)
            return

        geo_tags = {}
        for key, value in exif.items():
            tag_name = TAGS.get(key, str(key))
            if tag_name.lower() in {"gpsinfo", "gpslatitude", "gpslatituderef", "gpslongitude", "gpslongituderef", "gpsaltitude", "gpsaltituderef", "gpsdatetime", "gpsdatestamp"}:
                geo_tags[key] = value

        if geo_tags:
            new_exif = exif.copy()
            for key in list(new_exif.keys()):
                if key not in geo_tags:
                    del new_exif[key]
            img.save(output_path, exif=new_exif, quality=95, optimize=True)
        else:
            img.save(output_path, quality=95, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove all EXIF metadata except GPS/geo-location tags")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("output", help="Output image file or directory")
    parser.add_argument("--recursive", action="store_true", help="Process images recursively when input is a directory")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / input_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preserve_only_geo_tags(input_path, output_path)
        print(f"Processed: {input_path} -> {output_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    for src in iter_image_files(input_path, recursive=args.recursive):
        rel = src.relative_to(input_path)
        dst = output_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        preserve_only_geo_tags(src, dst)
        print(f"Processed: {src} -> {dst}")


if __name__ == "__main__":
    main()
