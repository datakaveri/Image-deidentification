# syntax=docker/dockerfile:1

# python:3.12-slim, pinned by multi-arch index digest. Dependabot (docker) bumps it.
ARG PYTHON_IMAGE=python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Installs the locked dependencies into a venv and fetches every model weight, so
# the final stage only copies results and needs no pip, scripts or network.
FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH

# Same cache locations as the final stage: weights are downloaded here and copied.
# YOLO_CONFIG_DIR must NOT be under /tmp: the settings file written at build time
# has to survive into the running container, otherwise ultralytics recreates it
# with telemetry (sync: true) re-enabled on every run.
ENV YOLO_CONFIG_DIR=/opt/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    EASYOCR_MODULE_PATH=/opt/easyocr \
    TORCH_HOME=/opt/torch

# OpenCV needs these shared libraries just to import, which the prefetch below does.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

# Same WORKDIR as the final stage: ultralytics records it in its settings file.
WORKDIR /app

# requirements.lock pins CPU-only torch/torchvision (+cpu builds from
# download.pytorch.org/whl/cpu), so the ~3 GB of CUDA wheels a slim, GPU-less
# image can never execute are not pulled.
COPY requirements.lock ./
RUN pip install -r requirements.lock

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-deps .

# Pre-download the model sets that are otherwise fetched on first run, so the
# container works with no egress. EasyOCR and DeepLab weights go straight into
# their cache directories.
RUN mkdir -p "$YOLO_CONFIG_DIR" && \
    python -c "import easyocr; easyocr.Reader(['en'], gpu=False)" && \
    python -c "from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights as W; deeplabv3_resnet50(weights=W.DEFAULT)" && \
    python -c "\
from ultralytics import settings; \
settings.update({'sync': False}); \
assert settings.get('sync') is False, 'telemetry still on'; \
assert settings.file.is_relative_to('/opt'), f'settings landed outside /opt: {settings.file}'; \
print('telemetry sync = False at', settings.file)"

# GitHub Release asset URL and SHA-256 of the plate detector weights. The weights
# are not tracked in Git; this downloads and verifies them.
ARG PLATE_MODEL_URL="https://github.com/datakaveri/Image-deidentification/releases/download/v2.1.0/license_plate_detector.pt"
ARG PLATE_MODEL_SHA256="2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822"

COPY scripts/download_models.py ./scripts/
RUN python scripts/download_models.py \
    --url "$PLATE_MODEL_URL" \
    --sha256 "$PLATE_MODEL_SHA256" \
    --dest /opt/models/license_plate_detector.pt && \
    chmod -R a+rX /opt/easyocr /opt/torch /opt/ultralytics /opt/models

# pip is build tooling: keep it out of the venv that ships.
RUN python -m pip uninstall -y pip

# ── Stage 2: final ────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    YOLO_CONFIG_DIR=/opt/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    EASYOCR_MODULE_PATH=/opt/easyocr \
    TORCH_HOME=/opt/torch

# SPIDEr/TEE volume contract, matching the SKALD_* names skald-dicom uses.
# The deployment compose lives in the datakaveri/Docker-Compose repo.
ENV SKALD_DATA_DIR=/app/data \
    SKALD_CONFIG_DIR=/app/config \
    SKALD_OUTPUT_DIR=/app/output

# Intermediates MUST stay inside the container. Stages 1-4 emit partially
# redacted images (and step 1 deliberately preserves GPS), so writing them under
# SKALD_OUTPUT_DIR would push un-redacted data onto the volume that leaves the
# enclave. Only the final images land in SKALD_OUTPUT_DIR.
ENV SKALD_TEMP_DIR=/tmp/skald-image

# Runtime shared libraries for OpenCV only; no compilers or package tooling. The
# base image's own pip is removed too, so the image carries no installer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/* && \
    python -m pip uninstall -y pip

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/easyocr /opt/easyocr
COPY --from=builder /opt/torch /opt/torch
COPY --from=builder /opt/ultralytics /opt/ultralytics

WORKDIR /app

COPY --from=builder /opt/models/license_plate_detector.pt ./models/license_plate_detector.pt
COPY config/ ./config/

# Create the mount points so a run with no volumes still fails cleanly rather
# than mkdir-ing into the image. Cache permissions were set in the builder, so
# nothing copied above is rewritten into another layer here.
RUN mkdir -p /app/data /app/config /app/output

# This is a batch job, so "healthy" means the package and its native deps import.
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "import cv2, image_deidentification" || exit 1

# Runs as root on purpose, matching skald-dicom: the TEE bind-mounts host
# directories and the container must be able to write /app/output regardless of
# host ownership.
ENTRYPOINT ["image-deidentification"]
CMD ["--config", "/app/config/pipeline_config.json"]
