import os
import numpy as np
import albumentations
import csv
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
        with open("data/radarhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        #radar_name = "Z_RADR_I_Z9010_%s_P_DOR_SA_R_10_230_15.010_clean.png"
        #rain_list = []
        #with open('data/rian_train.csv') as f:
        #    f_csv = csv.reader(f)
        #    for row in f_csv:
        #        rain_list.append(radar_name % row)
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        paths.sort()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class RadarHQValidation(FacesBase):
    def __init__(self, size, crop_size=None, keys=None):
        super().__init__()
        root = "data/rdhq"
        with open("data/radarhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        paths.sort()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys
