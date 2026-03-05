# clip_deepfake_docker Runtime

This adds a clean Docker runtime for the **full finetuned checkpoint** model.

The container watches `input/` for images and writes predictions to `output/`.

## Model mode

- Default checkpoint mode is full finetuned.
- Expected model file in container: `/data/models/model_best.pth`
- Default bind mount maps local folder:
  - `./ckpt/clip_vitl14_mediaeval_ftval4k_randomfc--1` -> `/data/models`

## Output

For each input image, output includes:
- `<image_stem>_result.txt`
- `predictions.csv` (appended)

CSV columns:
- `file`, `fake_probability`, `label`, `threshold`, `device`, `arch`

## Quick start

Create folders:

```bash
mkdir -p input output logs
```

### CPU

```bash
docker compose up clip-deepfake-cpu
```

### CUDA

```bash
docker compose --profile cuda up clip-deepfake-cuda
```

### Apple Silicon profile image

```bash
docker compose --profile mps up clip-deepfake-mps
```

## Prebuilt images (GHCR)

Images are published from GitHub Actions on pushes to `main`.

Base image path:

```text
ghcr.io/<owner>/<repo>/clip-deepfake
```

Main tags:
- `latest` (CPU)
- `latest-cuda`
- `latest-mps`

Example:

```bash
docker pull ghcr.io/<owner>/<repo>/clip-deepfake:latest
```

## Environment variables

- `WATCH_FOLDER` default `/data/input`
- `OUTPUT_FOLDER` default `/data/output`
- `MODEL_CKPT` default `/data/models/model_best.pth`
- `ARCH` default `CLIP:ViT-L/14`
- `THRESHOLD` default `0.5`
- `BATCH_SIZE` default `32` (CUDA defaults to `64` in compose)
- `FORCE_CPU` default `false` (CPU service sets true)
- `AUTO_FALLBACK_CPU_ON_UNSUPPORTED_CUDA` default `true`
- `CLIP_DOWNLOAD_ROOT` default `/data/models/clip_cache`

## Notes

- First run may download CLIP backbone weights into `CLIP_DOWNLOAD_ROOT`.
- Dockerfiles use BuildKit cache mounts for faster rebuilds.
- Supported input extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
- CI runs unit tests on every push/PR and publishes docker images from `main`.
