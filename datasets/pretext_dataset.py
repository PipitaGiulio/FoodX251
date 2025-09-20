import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import re


### Dataset Class for Self Supervised Learning
class SelfSupervisedPretextDS(Dataset):
    def __init__(self, root_path, transforms = None):
        super(SelfSupervisedPretextDS, self).__init__()
        self.x_list = [os.path.join(root_path, file_name) for file_name in os.listdir(root_path)]
        self.transforms = transforms
    def __len__(self):
        return(len(self.x_list))
    def __getitem__(self, index):
        x_sample = Image.open(self.x_list[index]).convert("RGB")
        return self.transforms(x_sample), self.transforms(x_sample)