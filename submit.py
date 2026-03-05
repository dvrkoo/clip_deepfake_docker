import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import random



import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import is_image_file, default_loader
import torchvision.transforms.functional as TF

from data.datasets import MEAN, STD
from models import get_model



def sort_key(p: Path):
    try:
        return int(p.stem)
    except:
        return p.stem   
    
class ImageFolderDataset(Dataset):
    def __init__(self, directory, transform):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        # self.image_paths = [directory / file for file in os.listdir(directory) if file.endswith(("jpg", "jpeg", "png"))]
        self.image_paths = [os.path.join(directory, file) for file in os.listdir(directory) if is_image_file(file)]
        self.image_paths = [directory / file for file in os.listdir(directory) if is_image_file(file)]
        self.image_paths.sort(key=sort_key)

        # self.image_paths = self.image_paths[:100]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        # txt_path = image_path.with_suffix(".txt")   # 用于替换路径的文件扩展名（后缀）。

        # filename = image_path.stem    # 获取文件路径中“去掉扩展名后的文件名”

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # return {"jpg": image, "caption": label}
        # return {"image": image, "caption": label}
        return {"image": image, "path": str(image_path)}



csv_file = "./submit/clipvitl14_ft4k_initfc.csv"
ckpt_path = "./ckpt/clip_vitl14_mediaeval_ftval4k_randomfc/model_best.pth"

data_dir = "./datasets/taska_test/"

arch_name = "CLIP:ViT-L/14"
model = get_model(arch_name)
print("Loading from %s" % ckpt_path)
state_dict = torch.load(ckpt_path, map_location='cpu')["model"]
model.load_state_dict(state_dict)

model = model.cuda()

stat_from = "clip"  # "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"

# blur_sig = "0.0,3.0"
# blur_sig = [float(s) for s in blur_sig.split(',')]

transform = transforms.Compose([
                transforms.Lambda(lambda img: TF.resize(img, 256, interpolation=Image.BILINEAR)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN["clip"], std=STD["clip"] ),
            ])

test_set = ImageFolderDataset(
                            directory=Path(data_dir),
                              transform=transform)
data_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=100,
                                              shuffle=False,
                                            #   sampler=sampler,
                                              num_workers=4)



data_pd = {} # for submission format (MediaEval)

y_pred = []
Hs, Ws = [], []
images_id = []
with torch.no_grad(): # example of standard evaluation 
    for data in tqdm(data_loader):
        #y_true.extend(label.flatten().tolist())
        image = data['image'].cuda()
        y_pred.extend(model(image).sigmoid().flatten().tolist())

        images_id.extend([os.path.basename(path) for path in data['path']])

# after the evaluation you have y_pred as a list of numbers
data_pd = {
    'images_id' : images_id,
    # 'images_id' : [str(path.name) for path in data['path']],
    'prob' : np.array(y_pred),
    'label' : np.where(np.array(y_pred) > 0.5, 1, 0).tolist(),
    'threshold' : [0.5] * len(images_id),
}

csv_fn = csv_file
df = pd.DataFrame(data_pd)
df.to_csv(csv_fn, index=False)
print('Results exported to {}\n.'.format(csv_fn))