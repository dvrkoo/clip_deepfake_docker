from .clip import clip 
from PIL import Image
import torch.nn as nn
import os


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        clip_cache_dir = os.getenv("CLIP_DOWNLOAD_ROOT")
        self.model, self.preprocess = clip.load(
            name,
            device="cpu",
            download_root=clip_cache_dir,
        ) # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )
 

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)
