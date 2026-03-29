# clip_deepfake_docker

Docker-first runtime for CLIP-based deepfake image scoring.

This repository is intentionally focused on operations (build, run, deploy), not paper/training documentation.

## What it does

- Watches `input/` for incoming images.
- Runs inference with the configured checkpoint.
- Writes outputs to `output/`:
  - `<image_stem>_result.txt`
  - `predictions.csv`

Checkpoint loading supports both formats:
- full model checkpoint (for example `model_best.pth` with full state)
- fc-only checkpoint (for example `pretrained_weights/fc_weights.pth` with `weight` and `bias`)

## Prebuilt images (GHCR)

Published by GitHub Actions on pushes to `main`.

Base image path:

```text
ghcr.io/dvrkoo/clip_deepfake_docker/clip-deepfake
```

Main tags:
- `latest` (CPU)
- `latest-cuda`
- `latest-mps`

Pull examples:

```bash
docker pull ghcr.io/dvrkoo/clip_deepfake_docker/clip-deepfake:latest
docker pull ghcr.io/dvrkoo/clip_deepfake_docker/clip-deepfake:latest-cuda
docker pull ghcr.io/dvrkoo/clip_deepfake_docker/clip-deepfake:latest-mps
```

## Quick start (docker compose)

Create folders:

```bash
mkdir -p input output logs
```

Mount your checkpoint directory so container path `/data/models/model_best.pth` exists.
Default compose mapping:

```text
./ckpt/clip_vitl14_mediaeval_ftval4k_randomfc--1 -> /data/models
```

Run CPU:

```bash
docker compose up clip-deepfake-cpu
```

Run CUDA:

```bash
docker compose --profile cuda up clip-deepfake-cuda
```

Run Apple Silicon profile image:

```bash
docker compose --profile mps up clip-deepfake-mps
```

## Fast local iteration (no rebuild per code edit)

Use the `dev` profile services, which bind-mount the repo into `/app`:

```bash
docker compose --profile dev up clip-deepfake-cpu-dev
```

CUDA dev service:

```bash
docker compose --profile dev --profile cuda up clip-deepfake-cuda-dev
```

After editing Python files, just restart the container. No image rebuild is needed unless dependencies or Dockerfiles changed.

## Run prebuilt image directly

```bash
docker run -d --name clip-deepfake-cpu \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -v $(pwd)/logs:/data/logs \
  -v $(pwd)/ckpt/clip_vitl14_mediaeval_ftval4k_randomfc--1:/data/models \
  -e WATCH_FOLDER=/data/input \
  -e OUTPUT_FOLDER=/data/output \
  -e MODEL_CKPT=/data/models/model_best.pth \
  -e FORCE_CPU=true \
  ghcr.io/dvrkoo/clip_deepfake_docker/clip-deepfake:latest
```

## Environment variables

- `WATCH_FOLDER` default `/data/input`
- `OUTPUT_FOLDER` default `/data/output`
- `MODEL_CKPT` default `/data/models/model_best.pth`
- `ARCH` default `CLIP:ViT-L/14`
- `THRESHOLD` default `0.5`
- `BATCH_SIZE` default `32` (`64` in CUDA compose service)
- `FORCE_CPU` default `false`
- `AUTO_FALLBACK_CPU_ON_UNSUPPORTED_CUDA` default `true`
- `CLIP_DOWNLOAD_ROOT` default `/data/models/clip_cache`

## CI/CD

On each push/PR:
- unit tests
- CPU docker smoke build

On push to `main`:
- publish CPU/CUDA/MPS images to GHCR
