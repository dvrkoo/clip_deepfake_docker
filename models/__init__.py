from .clip_models import CLIPModel


VALID_NAMES = [
    "CLIP:RN50",
    "CLIP:ViT-L/14",
]


def get_model(name):
    assert name in VALID_NAMES
    return CLIPModel(name[5:])
