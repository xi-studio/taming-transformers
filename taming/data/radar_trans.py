import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.radar_gpt import ImagePaths


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class RadarHQTrain(FacesBase):
    def __init__(self, size, crop_size=None, keys=None):
        super().__init__()
        root = "data/rdhq"
        with open("data/radarhqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class RadarHQValidation(FacesBase):
    def __init__(self, size, crop_size=None, keys=None):
        super().__init__()
        root = "data/rdhq"
        with open("data/radarhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys
