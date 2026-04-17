from pathlib import Path

import docker_app
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


def test_is_supported_image_accepts_expected_extensions():
    assert docker_app.is_supported_image("frame.jpg")
    assert docker_app.is_supported_image("frame.JPEG")
    assert docker_app.is_supported_image("frame.png")
    assert docker_app.is_supported_image("frame.webp")


def test_is_supported_image_rejects_non_images():
    assert not docker_app.is_supported_image("video.mp4")
    assert not docker_app.is_supported_image("document.txt")


def test_wait_for_file_stable_returns_true_for_static_file(tmp_path):
    target = tmp_path / "img.jpg"
    target.write_bytes(b"abc")
    assert docker_app.wait_for_file_stable(str(target), checks=2, delay=0.01)


def test_write_result_creates_txt_and_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_app, "OUTPUT_FOLDER", str(tmp_path))
    docker_app.write_result("/tmp/sample.jpg", 0.9, 1, "cpu")

    txt = tmp_path / "sample_result.txt"
    csv = tmp_path / "predictions.csv"

    assert txt.exists()
    assert csv.exists()
    assert "fake_probability=0.900000" in txt.read_text(encoding="utf-8")
    assert "sample.jpg" in csv.read_text(encoding="utf-8")


def test_queue_existing_files_only_supported(tmp_path, monkeypatch):
    (tmp_path / "a.jpg").write_bytes(b"1")
    (tmp_path / "b.png").write_bytes(b"1")
    (tmp_path / "c.txt").write_bytes(b"1")

    monkeypatch.setattr(docker_app, "WATCH_FOLDER", str(tmp_path))
    monkeypatch.setattr(docker_app, "PROCESS_EXISTING_ON_START", True)

    while not docker_app.file_queue.empty():
        docker_app.file_queue.get_nowait()

    docker_app.queue_existing_files()

    queued = []
    while not docker_app.file_queue.empty():
        queued.append(Path(docker_app.file_queue.get_nowait()).name)

    assert sorted(queued) == ["a.jpg", "b.png"]


def test_transform_matches_to_nicco_submit_preprocessing():
    img_array = np.zeros((300, 500, 3), dtype=np.uint8)
    img_array[..., 0] = 12
    img_array[..., 1] = 128
    img_array[..., 2] = 245
    img = Image.fromarray(img_array, mode="RGB")

    reference_transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda pil_img: TF.resize(pil_img, 256, interpolation=Image.BILINEAR)
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=docker_app.MEAN_CLIP, std=docker_app.STD_CLIP),
        ]
    )

    out_docker = docker_app.transform(img)
    out_reference = reference_transform(img)

    assert out_docker.shape == (3, 224, 224)
    assert torch.allclose(out_docker, out_reference, atol=0, rtol=0)
