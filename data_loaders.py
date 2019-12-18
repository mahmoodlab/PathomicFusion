### data_loaders.py
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets, transforms


################
# Dataset Loader
################
class PathgraphomicDatasetLoader(Dataset):
    def __init__(self, opt, data, split, mode='omic'):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.X_path = data[split]['x_path']
        self.X_grph = data[split]['x_grph']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        self.mode = mode
        
        self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomCrop(opt.input_size_path),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)

        if self.mode == "path" or self.mode == 'pathpath':
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            return (self.transforms(single_X_path), 0, 0, single_e, single_t, single_g)
        elif self.mode == "graph" or self.mode == 'graphgraph':
            single_X_grph = torch.load(self.X_grph[index])
            return (0, single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "omic" or self.mode == 'omicomic':
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathomic":
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.transforms(single_X_path), 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "graphomic":
            single_X_grph = torch.load(self.X_grph[index])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, single_X_grph, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathgraph":
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            single_X_grph = torch.load(self.X_grph[index])
            return (self.transforms(single_X_path), single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "pathgraphomic":
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            single_X_grph = torch.load(self.X_grph[index])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.transforms(single_X_path), single_X_grph, single_X_omic, single_e, single_t, single_g)

    def __len__(self):
        return len(self.X_path)


class PathgraphomicFastDatasetLoader(Dataset):
    def __init__(self, opt, data, split, mode='omic'):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.X_path = data[split]['x_path']
        self.X_grph = data[split]['x_grph']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        self.mode = mode

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)

        if self.mode == "path" or self.mode == 'pathpath':
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            return (single_X_path, 0, 0, single_e, single_t, single_g)
        elif self.mode == "graph" or self.mode == 'graphgraph':
            single_X_grph = torch.load(self.X_grph[index])
            return (0, single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "omic" or self.mode == 'omicomic':
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathomic":
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (single_X_path, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "graphomic":
            single_X_grph = torch.load(self.X_grph[index])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, single_X_grph, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathgraph":
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            single_X_grph = torch.load(self.X_grph[index])
            return (single_X_path, single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "pathgraphomic":
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            single_X_grph = torch.load(self.X_grph[index])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (single_X_path, single_X_grph, single_X_omic, single_e, single_t, single_g)

    def __len__(self):
        return len(self.X_path)
