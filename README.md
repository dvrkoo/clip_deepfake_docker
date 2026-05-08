# clip_deepfake_docker

Docker-first runtime for CLIP-based deepfake image scoring.

This repository is intentionally focused on operations (build, run, deploy), not paper/training documentation.

## What it does

- Watches `input/` for incoming images.
- Runs inference with the configured checkpoint.
- Writes outputs to `output/`:
  - `<image_stem>_result.txt`
  - `predictions.csv`

Operationally, the container behaves like a black box:

1. Put one or more images into the mounted `input/` folder.
2. The app detects supported image files automatically.
3. Each image is preprocessed and scored by the CLIP deepfake model.
4. A text result file is written to the mounted `output/` folder.
5. A cumulative `predictions.csv` file is also updated in `output/`.

Supported input image extensions:

```text
.jpg, .jpeg, .png, .bmp, .webp
```

For an input image named:

```text
input/example.png
```

The app writes:

```text
output/example_result.txt
output/predictions.csv
```

Example text output:

```text
file=example.png
fake_probability=0.873421
label=1
threshold=0.5
device=cpu
arch=CLIP:ViT-L/14
```

Output fields:

- `file`: original input filename
- `fake_probability`: model probability that the image is fake
- `label`: `1` if `fake_probability > threshold`, otherwise `0`
- `threshold`: decision threshold, default `0.5`
- `device`: runtime device used for inference (`cpu`, `cuda`, or `mps`)
- `arch`: configured model architecture

Checkpoint loading supports both formats:
- full model checkpoint (for example `model_best.pth` with full state)
- fc-only checkpoint (for example `pretrained_weights/fc_weights.pth` with `weight` and `bias`)

## Model checkpoint

The Docker image does not include the trained checkpoint because it is too large for GitHub.

Download the checkpoint from Google Drive:

```text
https://drive.google.com/drive/folders/1WEUbBaCr1fqoBB8E718yuUkAKjp6Pgwi?usp=sharing
```

Use this checkpoint file:

```text
model_best.pth
```

Expected local layout:

```text
ckpt/
└── clip_vitl14_mediaeval_ftval4k_randomfc--1/
    └── model_best.pth
```

The default Docker Compose services mount this folder to `/data/models`, so the checkpoint is available inside the container as:

```text
/data/models/model_best.pth
```

If you store the checkpoint somewhere else, update `MODEL_CKPT` to point at the mounted file path inside the container.

## Preprocessing

Inference preprocessing matches the original `to_nicco/submit.py` pipeline:

- resize to `256` with bilinear interpolation
- center crop to `224x224`
- convert to tensor
- normalize with official CLIP mean/std

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
mkdir -p input output logs ckpt/clip_vitl14_mediaeval_ftval4k_randomfc--1
```

Download `model_best.pth` from Google Drive and place it here:

```text
ckpt/clip_vitl14_mediaeval_ftval4k_randomfc--1/model_best.pth
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

Then copy or save images into `input/`. Results will appear in `output/`.

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
