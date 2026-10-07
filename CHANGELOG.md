# Changelog

All notable changes to `Image-deidentification` will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `release.yml` creates the GitHub Release with the `gh` CLI instead of a third-party action the
  repo's Actions policy does not allow, which stopped the `v2.2.0` run from starting. It can also
  be run by hand (`workflow_dispatch`) to publish an existing tag.

## [2.2.0] - 2026-10-07

### Added
- CI publishes `ghcr.io/datakaveri/skald-image:main` and `:sha-<commit>` after the tests pass on
  every merge to `main`. `latest` is still retagged by hand at rollout.
- Apache-2.0 `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md` and this changelog.
- `pyproject.toml` with bounded dependency pins, an `image-deidentification` console script and
  `python -m image_deidentification`; `requirements.lock` and `requirements-dev.lock` generated
  with pip-compile.
- GitHub workflows: CI (ruff, black, pytest with coverage on CPU-only torch), CodeQL, and a
  release workflow that publishes `ghcr.io/datakaveri/skald-image:<tag>` on `vX.Y.Z` tags.
- Dependabot (pip, docker, github-actions), CODEOWNERS, pull request and issue templates.
- Docker `HEALTHCHECK`, `.env.example`, and pre-commit hooks (ruff, black, gitleaks).
- Tests for the CLI, config loading and missing-weights handling.

### Changed
- Package moved from `app/` to `src/image_deidentification/`; `main.py` is now
  `image_deidentification.main`. Relative paths resolve against the working directory.
- Plate weights default to `models/license_plate_detector.pt`; configs pointing at the old
  `app/sensitive_data_masking/` path fall back to `models/`.
- Dockerfile is multi-stage on a digest-pinned `python:3.12-slim`; the runtime image carries
  no pip or build tooling. Python 3.12 is now the minimum (numpy 2.4 requires 3.11+).
- Codebase formatted with black and linted with ruff.

### Removed
- `requirements.txt` (superseded by `pyproject.toml` and the lock files).
- The commented-out legacy subprocess orchestrator at the end of `main.py`.

[Unreleased]: https://github.com/datakaveri/Image-deidentification/compare/v2.2.0...HEAD
[2.2.0]: https://github.com/datakaveri/Image-deidentification/compare/v2.1.0...v2.2.0
