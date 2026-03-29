import torch
from torch import nn

from checkpoint_utils import load_checkpoint_into_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 3)
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(self.backbone(x))


def test_load_fc_only_checkpoint(tmp_path):
    model = TinyModel()
    checkpoint_path = tmp_path / "fc_only.pth"
    torch.save({"weight": torch.randn(1, 3), "bias": torch.randn(1)}, checkpoint_path)

    mode, _ = load_checkpoint_into_model(model, str(checkpoint_path), strict=True)

    assert mode == "fc_only"


def test_load_full_checkpoint_from_model_key(tmp_path):
    model = TinyModel()
    checkpoint_path = tmp_path / "full_model.pth"
    torch.save({"model": model.state_dict(), "optimizer": {}}, checkpoint_path)

    mode, _ = load_checkpoint_into_model(model, str(checkpoint_path), strict=True)

    assert mode == "full"
