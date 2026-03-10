import csv
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from models import get_model

logger = logging.getLogger(__name__)

WATCH_FOLDER = os.getenv("WATCH_FOLDER", "./input" if not os.path.exists("/data") else "/data/input")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "./output" if not os.path.exists("/data") else "/data/output")
LOG_FILE = os.getenv("LOG_FILE", "./logs/app.log")
MODEL_CKPT = os.getenv("MODEL_CKPT", "./ckpt/clip_vitl14_mediaeval_ftval4k_randomfc--1/model_best.pth")
ARCH = os.getenv("ARCH", "CLIP:ViT-L/14")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"
AUTO_FALLBACK_CPU_ON_UNSUPPORTED_CUDA = (
    os.getenv("AUTO_FALLBACK_CPU_ON_UNSUPPORTED_CUDA", "true").lower() == "true"
)
PROCESS_EXISTING_ON_START = os.getenv("PROCESS_EXISTING_ON_START", "true").lower() == "true"
CLIP_DOWNLOAD_ROOT = os.getenv("CLIP_DOWNLOAD_ROOT", "/data/models/clip_cache")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MEAN_CLIP = [0.48145466, 0.4578275, 0.40821073]
STD_CLIP = [0.26862954, 0.26130258, 0.27577711]

os.environ.setdefault("CLIP_DOWNLOAD_ROOT", CLIP_DOWNLOAD_ROOT)
os.environ.setdefault("LOG_FILE", LOG_FILE)

os.makedirs(WATCH_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
log_dir = os.path.dirname(LOG_FILE)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

import logger_config

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=Image.Resampling.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_CLIP, std=STD_CLIP),
    ]
)


def resolve_device() -> torch.device:
    if FORCE_CPU:
        return torch.device("cpu")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if device.type == "cuda":
        capability = torch.cuda.get_device_capability()
        gpu_arch = f"sm_{capability[0]}{capability[1]}"
        supported_arches = {arch for arch in torch.cuda.get_arch_list() if arch.startswith("sm_")}
        if gpu_arch not in supported_arches and AUTO_FALLBACK_CPU_ON_UNSUPPORTED_CUDA:
            logger.warning(
                "GPU arch %s not supported by this torch build (%s); falling back to CPU",
                gpu_arch,
                " ".join(sorted(supported_arches)),
            )
            return torch.device("cpu")

    return device


def load_model(device: torch.device):
    if not os.path.exists(MODEL_CKPT):
        raise FileNotFoundError(f"MODEL_CKPT not found: {MODEL_CKPT}")

    model = get_model(ARCH)
    ckpt_obj = torch.load(MODEL_CKPT, map_location="cpu")

    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
        logger.info("Loading full checkpoint state from key 'model'")
        model.load_state_dict(ckpt_obj["model"], strict=True)
    elif isinstance(ckpt_obj, dict):
        logger.info("Loading checkpoint as direct state_dict")
        model.load_state_dict(ckpt_obj, strict=True)
    else:
        raise ValueError("Unsupported checkpoint format")

    model = model.to(device)
    model.eval()
    return model


def is_supported_image(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def wait_for_file_stable(path: str, checks: int = 3, delay: float = 0.5) -> bool:
    stable_count = 0
    previous_size = -1
    for _ in range(30):
        if not os.path.exists(path):
            time.sleep(delay)
            continue

        current_size = os.path.getsize(path)
        if current_size > 0 and current_size == previous_size:
            stable_count += 1
            if stable_count >= checks:
                return True
        else:
            stable_count = 0
            previous_size = current_size
        time.sleep(delay)
    return False


def predict_batch(image_paths: List[str], model, device: torch.device):
    batch_tensors = []
    kept_paths = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            batch_tensors.append(transform(img))
            kept_paths.append(path)
        except Exception as exc:
            logger.error("Failed reading image %s: %s", path, exc)

    if not batch_tensors:
        return []

    input_tensor = torch.stack(batch_tensors, dim=0).to(device)
    with torch.inference_mode():
        logits = model(input_tensor).flatten()
        probs = torch.sigmoid(logits).detach().cpu().tolist()

    results = []
    for path, prob in zip(kept_paths, probs):
        label = 1 if prob > THRESHOLD else 0
        results.append((path, float(prob), label))
    return results


def write_result(path: str, prob: float, label: int, device: torch.device) -> None:
    stem = Path(path).stem
    txt_path = Path(OUTPUT_FOLDER) / f"{stem}_result.txt"
    txt_path.write_text(
        "\n".join(
            [
                f"file={Path(path).name}",
                f"fake_probability={prob:.6f}",
                f"label={label}",
                f"threshold={THRESHOLD}",
                f"device={device}",
                f"arch={ARCH}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = Path(OUTPUT_FOLDER) / "predictions.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["file", "fake_probability", "label", "threshold", "device", "arch"])
        writer.writerow([Path(path).name, f"{prob:.6f}", label, THRESHOLD, str(device), ARCH])


def cleanup_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


file_queue: queue.Queue[str] = queue.Queue()
seen_files = set()


class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = event.src_path
        if is_supported_image(path):
            file_queue.put(path)

    def on_moved(self, event):
        if event.is_directory:
            return
        path = event.dest_path
        if is_supported_image(path):
            file_queue.put(path)


def worker(model, device: torch.device):
    pending = []
    last_flush = time.time()
    while True:
        try:
            path = file_queue.get(timeout=0.5)
            if path in seen_files:
                file_queue.task_done()
                continue
            if not wait_for_file_stable(path):
                logger.error("File not stable, skipping: %s", path)
                file_queue.task_done()
                continue
            pending.append(path)
            seen_files.add(path)
            file_queue.task_done()
        except queue.Empty:
            pass

        should_flush = len(pending) >= BATCH_SIZE or (
            pending and (time.time() - last_flush) > 1.0
        )
        if should_flush:
            logger.info("Running inference for %d images", len(pending))
            results = predict_batch(pending, model, device)
            for path, prob, label in results:
                write_result(path, prob, label, device)
                logger.info("Processed %s prob=%.4f label=%d", path, prob, label)
            pending = []
            last_flush = time.time()
            cleanup_cache()


def queue_existing_files() -> None:
    if not PROCESS_EXISTING_ON_START:
        return
    for entry in sorted(Path(WATCH_FOLDER).iterdir()):
        if entry.is_file() and is_supported_image(str(entry)):
            file_queue.put(str(entry))


def main() -> None:
    device = resolve_device()
    logger.info("Starting clip_deepfake_docker app")
    logger.info("Watch folder: %s", WATCH_FOLDER)
    logger.info("Output folder: %s", OUTPUT_FOLDER)
    logger.info("Model ckpt: %s", MODEL_CKPT)
    logger.info("Arch: %s", ARCH)
    logger.info("Using device: %s", device)

    model = load_model(device)

    queue_existing_files()
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_FOLDER, recursive=False)
    observer.start()

    threading.Thread(target=worker, args=(model, device), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
