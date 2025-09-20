import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import re

### Dataset Class for Supervised Learning
class SupervisedClassificationDS(Dataset):
    def __init__(self, root_path, dict_path, transforms = None):
        super(SupervisedClassificationDS, self).__init__()
        self.x_list = [os.path.join(root_path, file_name) for file_name in os.listdir(root_path)]
        self.dict = np.load(dict_path, allow_pickle=True).item()
        self.transforms = transforms
    
    def __len__(self):
        return(len(self.x_list))
    
    def __getitem__(self, index):
        x_sample = Image.open(self.x_list[index]).convert("RGB")
        filename = os.path.basename(self.x_list[index])
        x_idx = int(re.sub(r"\D", "", filename))
        if self.transforms is not None:
            x_sample = self.transforms(x_sample)
        y_sample = self.dict[x_idx]
        return x_sample, y_sample
    