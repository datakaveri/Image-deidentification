#!/usr/bin/env python3
"""
Model download and verification script for Road Defect Anonymization.

Downloads model weights (e.g., license_plate_detector.pt) from a GitHub Release
or configurable URL, verifies their SHA-256 checksum, and ensures model weights
are kept outside the Git repository.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Default GitHub Release asset URL and checksums
DEFAULT_PLATE_MODEL_URL = os.environ.get(
    "PLATE_MODEL_URL",
    "https://github.com/datakaveri/Image-deidentification/releases/download/v2.1.0/license_plate_detector.pt",
)
DEFAULT_PLATE_MODEL_SHA256 = (
    "2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822"
)
KNOWN_VALID_PLATE_SHA256 = {
    "2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822",  # models/license_plate_detector.pt (6.25 MB)
    "8ec3b254a6c87610f037a90957462cafa11a9c03224e33a28c6a1d1ac2ac51b0",  # app/sensitive_data_masking/license_plate_detector.pt (6.24 MB)
}

DEFAULT_YOLOV8N_URL = os.environ.get(
    "YOLOV8N_URL",
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
)
DEFAULT_YOLOV8N_SHA256 = (
    "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
)


def compute_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest().lower()


def verify_file(
    file_path: Path,
    expected_sha256: str | None = None,
    allow_known: set[str] | None = None,
) -> bool:
    """Check if file exists and has an acceptable SHA-256 checksum."""
    if not file_path.is_file():
        return False

    actual_hash = compute_sha256(file_path)
    if expected_sha256 and actual_hash == expected_sha256.lower():
        return True
    if allow_known and actual_hash in allow_known:
        return True

    return False


def copy_or_link(source: Path, target: Path) -> None:
    """Copy or link source file to target path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() == source.resolve():
        return
    if target.is_symlink() or target.exists():
        target.unlink()
    try:
        # Create relative symlink if on Unix
        rel_target = os.path.relpath(source, target.parent)
        target.symlink_to(rel_target)
    except OSError:
        shutil.copy2(source, target)


def download_file(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    allow_known: set[str] | None = None,
    force: bool = False,
) -> bool:
    """Download a file with progress reporting and checksum verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not force and verify_file(dest, expected_sha256, allow_known):
        print(f"[OK] Valid model file already present at: {dest}")
        return True

    tmp_file = dest.with_suffix(dest.suffix + ".tmp")
    print(f"Downloading model from: {url}")
    print(f"Destination: {dest}")

    headers = {"User-Agent": "RoadDefectAnonymization-ModelDownloader/1.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as response, tmp_file.open("wb") as out:
            total_size_header = response.getheader("Content-Length")
            total_size = int(total_size_header) if total_size_header else None
            downloaded = 0
            chunk_size = 65536

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100
                    mb_cur = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    sys.stdout.write(f"\rDownloading: {mb_cur:.1f}/{mb_tot:.1f} MB ({pct:.1f}%)")
                else:
                    mb_cur = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\rDownloading: {mb_cur:.1f} MB")
                sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

    except urllib.error.HTTPError as e:
        if tmp_file.exists():
            tmp_file.unlink()
        print(f"\n[ERROR] HTTP Error {e.code} while downloading {url}: {e.reason}", file=sys.stderr)
        if e.code == 404:
            print(
                "\nHint: The release asset might not be published yet on GitHub.\n"
                "To publish the release asset, you can run:\n"
                f"  gh release create v2.1.0 {dest} --title 'v2.1.0' --notes 'Release v2.1.0'\n"
                "Or provide a custom download URL with --url <URL> or PLATE_MODEL_URL=<URL>.",
                file=sys.stderr,
            )
        return False
    except Exception as e:
        if tmp_file.exists():
            tmp_file.unlink()
        print(f"\n[ERROR] Failed to download {url}: {e}", file=sys.stderr)
        return False

    actual_hash = compute_sha256(tmp_file)
    is_valid = True
    if expected_sha256 and actual_hash != expected_sha256.lower():
        if allow_known and actual_hash in allow_known:
            is_valid = True
        else:
            is_valid = False

    if not is_valid:
        print(
            f"[ERROR] SHA-256 checksum mismatch for downloaded file!\n"
            f"  Expected: {expected_sha256}\n"
            f"  Actual:   {actual_hash}",
            file=sys.stderr,
        )
        tmp_file.unlink()
        return False

    # Atomically move temporary file to destination
    if dest.exists():
        dest.unlink()
    tmp_file.rename(dest)
    print(f"[OK] Downloaded and verified: {dest} (SHA-256: {actual_hash})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify model weights for road defect anonymization."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_PLATE_MODEL_URL,
        help=f"Download URL for license_plate_detector.pt (default: {DEFAULT_PLATE_MODEL_URL})",
    )
    parser.add_argument(
        "--sha256",
        default=DEFAULT_PLATE_MODEL_SHA256,
        help=f"Expected SHA-256 checksum (default: {DEFAULT_PLATE_MODEL_SHA256})",
    )
    parser.add_argument(
        "--dest",
        default="models/license_plate_detector.pt",
        help="Primary destination path (default: models/license_plate_detector.pt)",
    )
    parser.add_argument(
        "--link-dest",
        default="app/sensitive_data_masking/license_plate_detector.pt",
        help="Secondary path to sync/link (default: app/sensitive_data_masking/license_plate_detector.pt)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify that model files exist and match checksums; do not download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if file is already present.",
    )
    parser.add_argument(
        "--include-yolov8n",
        action="store_true",
        help="Also download yolov8n.pt if not present in app/sensitive_data_masking/",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dest_path = (repo_root / args.dest).resolve() if not Path(args.dest).is_absolute() else Path(args.dest)
    link_path = (
        (repo_root / args.link_dest).resolve()
        if args.link_dest and not Path(args.link_dest).is_absolute()
        else (Path(args.link_dest) if args.link_dest else None)
    )

    if args.check_only:
        valid_dest = verify_file(dest_path, args.sha256, KNOWN_VALID_PLATE_SHA256)
        valid_link = link_path and verify_file(link_path, args.sha256, KNOWN_VALID_PLATE_SHA256)
        if valid_dest or valid_link:
            print("[OK] License plate detector model file is present and verified.")
            return 0
        else:
            print("[FAIL] License plate detector model file missing or checksum invalid.", file=sys.stderr)
            return 1

    # Check if file exists at link_path already (e.g. from local repo checkout)
    if not args.force:
        if verify_file(dest_path, args.sha256, KNOWN_VALID_PLATE_SHA256):
            if link_path and not link_path.exists():
                copy_or_link(dest_path, link_path)
            print(f"[OK] Model weights verified at {dest_path}")
            return 0
        elif link_path and verify_file(link_path, args.sha256, KNOWN_VALID_PLATE_SHA256):
            copy_or_link(link_path, dest_path)
            print(f"[OK] Synced valid weights from {link_path} to {dest_path}")
            return 0

    success = download_file(
        url=args.url,
        dest=dest_path,
        expected_sha256=args.sha256,
        allow_known=KNOWN_VALID_PLATE_SHA256,
        force=args.force,
    )

    if not success:
        return 1

    if link_path:
        copy_or_link(dest_path, link_path)
        print(f"[OK] Synced weights to {link_path}")

    if args.include_yolov8n:
        yolo_dest = repo_root / "app/sensitive_data_masking/yolov8n.pt"
        if not verify_file(yolo_dest, DEFAULT_YOLOV8N_SHA256):
            print("Downloading yolov8n.pt...")
            download_file(
                url=DEFAULT_YOLOV8N_URL,
                dest=yolo_dest,
                expected_sha256=DEFAULT_YOLOV8N_SHA256,
                force=args.force,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
