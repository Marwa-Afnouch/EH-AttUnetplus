# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:24:08 2022

@author: Marwa
"""

from torch.utils.data import Dataset
import os
import os.path
import torch
import numpy as np

##############

 
class DataBone(Dataset):
     'Characterizes a dataset for PyTorch'
     def __init__(self, list_IDs, path, transform=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.path = path
        self.transform = transform

     def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

     def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        pth = self.path

        image, mask_b,mask_l = torch.load( os.path.join(pth, ID + '.pt'))
 

        mask_b[mask_b > 0.0] = 1.0
        mask_l[mask_l > 0.0] = 1.0
        if self.transform is not None:
              augmentations = self.transform(image=image, mask_b=mask_b,mask_l=mask_l)
              image = augmentations["image"]
              mask_b = augmentations["mask_b"]
              mask_l = augmentations["mask_l"]

        return image,mask_b,mask_l
