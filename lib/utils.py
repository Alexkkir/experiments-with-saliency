import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torchvision.datasets import ImageFolder

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import floor, ceil
from sklearn.model_selection import train_test_split

import shutil
import requests
import functools
import pathlib
from pathlib import Path
import shutil
from tqdm.notebook import tqdm
import os
from collections import defaultdict

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def display_batch(loader, label='subj_mean', cols=8):
    if isinstance(loader, DataLoader):
        batch = next(iter(loader))
    else:
        batch = loader

    rows = ceil(len(batch['image']) / cols)
    fig_size = matplotlib.rcParams['figure.figsize'][0] / cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols,
                              figsize=(fig_size * cols, fig_size * rows))

    ax = ax.ravel()
    for i, image in enumerate(batch['image']):
        image = image.permute(1, 2, 0)
        image = image * STD + MEAN
        image = (image * 255).type(torch.uint8)

        label_val = round(int(batch[label][i]), 0)

        if 'saliency' in batch:
            sal = batch['saliency'][i].permute(1, 2, 0)
            sal = torch.dstack([sal] * 3)
            if isinstance(image, np.ndarray):
                red = np.zeros_like(image)
            elif isinstance(image, torch.Tensor):
                red = torch.zeros_like(image)
            red[..., 2] = 255
            n, m, _ = image.shape
            sal = cv2.resize(np.array(sal), (m, n))
            image = image * (1 - sal) + red * sal

        image = image.type(torch.uint8)

        ax[i].imshow(image)
        ax[i].set_title(label_val, fontsize=20)
        ax[i].set_axis_off()
    plt.tight_layout()

def display_sample(sample, label='subj_mean'):
    image = sample['image'].permute(1, 2, 0)
    image = image * STD + MEAN
    image = (image * 255).type(torch.uint8)

    label_val = round(int(sample[label]), 0)

    if 'saliency' in sample:
        sal = sample['saliency'].permute(1, 2, 0)
        if isinstance(image, np.ndarray):
            red = np.zeros_like(image)
        elif isinstance(image, torch.Tensor):
            red = torch.zeros_like(image)
        red[..., 2] = 255
        image = image * (1 - sal) + red * sal

    image = image.type(torch.uint8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.set_title(label_val, fontsize=20)
    ax.set_axis_off()