"""CLI and config plumbing. Nothing here loads model weights."""

import json
import re
import subprocess
import sys
from importlib import metadata

import pytest

import image_deidentification
from image_deidentification import main as pipeline


def test_version_is_semver_and_matches_package_metadata():
    version = image_deidentification.__version__

    assert re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version)
    assert metadata.version("image-deidentification") == version


def test_parser_toggles_override_config():
    args = pipeline.build_parser().parse_args(["--no-human-mask", "--workers", "3"])
    config = {"human_mask": True, "workers": 1}

    assert pipeline.resolve(args.human_mask, config, "human_mask") is False
    assert pipeline.resolve(args.workers, config, "workers") == 3
    assert pipeline.resolve(args.plate_mask, config, "plate_mask") is True


def test_load_config_missing_file_is_empty(tmp_path):
    assert pipeline.load_config(tmp_path / "absent.json") == {}


def test_load_config_rejects_non_object(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError):
        pipeline.load_config(path)


def test_relative_paths_resolve_against_working_directory():
    assert pipeline.resolve_path("models/x.pt") == (pipeline.ROOT_DIR / "models/x.pt").resolve()


def test_missing_plate_weights_point_at_download_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        pipeline.load_models(
            plate_mask=True,
            human_mask=False,
            watermark_removal=False,
            ocr=False,
            weights=tmp_path / "missing.pt",
            device_name="cpu",
            ocr_langs="en",
        )


def test_module_entry_point_prints_help():
    result = subprocess.run(
        [sys.executable, "-m", "image_deidentification", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
