FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ultralytics/matplotlib insist on a writable config dir. YOLO_CONFIG_DIR must
# NOT be under /tmp: the settings file written at build time has to survive into
# the running container, otherwise ultralytics recreates it with telemetry
# (sync: true) re-enabled on every run.
ENV YOLO_CONFIG_DIR=/opt/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# Model weight caches. Populated at build time so the container needs no network.
ENV EASYOCR_MODULE_PATH=/opt/easyocr \
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

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only PyTorch build FIRST so the resolver does not pull the
# ~3 GB of CUDA wheels that a slim, GPU-less image can never execute.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# GitHub Release asset URL and SHA-256 for downloading plate detector weights at build time.
ARG PLATE_MODEL_URL="https://github.com/datakaveri/Image-deidentification/releases/download/v2.1.0/license_plate_detector.pt"
ARG PLATE_MODEL_SHA256="2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822"

# Pre-download the model sets that are otherwise fetched on first run, so the
# container works with no egress. EasyOCR and DeepLab weights are downloaded
# directly into their system cache directories.
RUN mkdir -p "$YOLO_CONFIG_DIR" && \
    python -c "import easyocr; easyocr.Reader(['en'], gpu=False)" && \
    python -c "from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights as W; deeplabv3_resnet50(weights=W.DEFAULT)" && \
    python -c "\
from ultralytics import settings; \
settings.update({'sync': False}); \
assert settings.get('sync') is False, 'telemetry still on'; \
assert settings.file.is_relative_to('/opt'), f'settings landed outside /opt: {settings.file}'; \
print('telemetry sync = False at', settings.file)" && \
    chmod -R a+rX /opt/easyocr /opt/torch /opt/ultralytics

COPY main.py pipeline_config.json ./
COPY app/ ./app/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Download and verify the plate detector model weights at build time so the binary
# does not need to be tracked in Git. If weights are already present from local build
# context, it verifies SHA-256; otherwise it downloads from PLATE_MODEL_URL.
RUN python scripts/download_models.py \
    --url "$PLATE_MODEL_URL" \
    --sha256 "$PLATE_MODEL_SHA256" \
    --dest "models/license_plate_detector.pt" \
    --link-dest "app/sensitive_data_masking/license_plate_detector.pt"

# Create the mount points so a run with no volumes still fails cleanly rather
# than mkdir-ing into the image.
RUN mkdir -p /app/data /app/config /app/output

# Runs as root, matching skald-dicom: the TEE bind-mounts host directories and
# the container must be able to write /app/output regardless of host ownership.
CMD ["python", "main.py", "--config", "/app/config/pipeline_config.json"]
