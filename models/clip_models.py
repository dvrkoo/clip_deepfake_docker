import os

import torch.nn as nn

from .clip import clip


CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
}


class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super().__init__()

        clip_cache_dir = os.getenv("CLIP_DOWNLOAD_ROOT") or os.path.expanduser("~/.cache/clip")
        self.model, self.preprocess = clip.load(
            name,
            device="cpu",
            download_root=clip_cache_dir,
        )
        self.fc = nn.Linear(CHANNELS[name], num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)
