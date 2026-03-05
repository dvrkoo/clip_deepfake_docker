# syntax=docker/dockerfile:1.7
FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY requirements-no-torch.txt .

RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir uv

RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system --no-cache \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system --no-cache -r requirements-no-torch.txt

COPY . .

ENV WATCH_FOLDER=/data/input
ENV OUTPUT_FOLDER=/data/output
ENV LOG_FILE=/data/logs/app.log
ENV MODEL_CKPT=/data/models/model_best.pth
ENV ARCH=CLIP:ViT-L/14
ENV THRESHOLD=0.5
ENV BATCH_SIZE=32
ENV FORCE_CPU=false
ENV CLIP_DOWNLOAD_ROOT=/data/models/clip_cache

RUN mkdir -p /data/input /data/output /data/logs /data/models /data/models/clip_cache

VOLUME ["/data/input", "/data/output", "/data/logs", "/data/models"]

CMD ["python", "docker_app.py"]
