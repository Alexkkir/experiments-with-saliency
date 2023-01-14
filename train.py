# %%
# %load_ext autoreload
# %autoreload 2

# %%
import lib

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
from scipy import stats

import shutil
import requests
import functools
import pathlib
from pathlib import Path
import shutil
from tqdm.auto import tqdm
import os
from collections import defaultdict
import pickle

from IPython.display import clear_output


matplotlib.rcParams['figure.figsize'] = (30, 5)

# %%
IMAGE_SHAPE = (384, 512)

DATA_ROOT = Path('iqa')

# %%
class IQADataset(Dataset):
    def __init__(self, images_path, labels_path, mode, transforms=None):
        assert mode in ['train', 'valid', 'test', 'all']
        TRAIN_RATIO = 0.7
        TRAIN_VALID_RATIO = 0.8
        self.images_path = images_path
        self.files = os.listdir(images_path)
        self.labels_path = labels_path

        df = pd.read_csv(labels_path).astype('float32', errors='ignore')
        train_size = int(TRAIN_RATIO * len(df))
        train_valid_size = int(TRAIN_VALID_RATIO * len(df))

        if mode == 'train':
            self.df = df.iloc[:train_size]
        elif mode == 'valid':
            self.df = df.iloc[train_size:train_valid_size]
        elif mode == 'test':
            self.df = df.iloc[train_valid_size:]
        elif mode == 'all':
            self.df = df

        self.transforms = transforms
        self.default_size = (500, 500)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> tuple:
        name, subj_mean, subj_std = self.df.iloc[index]
        image = cv2.imread(str(self.images_path / name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.ndim == 2:
            image = np.expand_dims(image, 2)

        if image.shape[:2] != self.default_size:
            image = cv2.resize(image, self.default_size)

        if self.transforms:
            image = self.transforms(image=image)['image']
        return {'image': image, 'name': name, 'subj_mean': subj_mean, 'subj_std': subj_std}

# %%
transforms = A.Compose([
    A.Resize(*IMAGE_SHAPE),
    A.HorizontalFlip(),
    A.Normalize(),
    ToTensorV2(),
])
dataset_train = IQADataset(images_path=DATA_ROOT / 'koniq10k' / 'images', labels_path=DATA_ROOT / 'koniq_data.csv', mode='train', transforms=transforms)
dataset_valid = IQADataset(images_path=DATA_ROOT / 'koniq10k' / 'images', labels_path=DATA_ROOT / 'koniq_data.csv', mode='valid', transforms=transforms)
# dataset_test = IQADataset(images_path=DATA_ROOT / 'CLIVE' / 'images', labels_path=DATA_ROOT / 'clive_data.csv', mode='all', transforms=transforms)

# %%
loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=16)
batch = next(iter(loader_train))
lib.display_batch(batch, 'subj_mean')

# %%
loader_valid = DataLoader(dataset_valid, batch_size=16, shuffle=True, num_workers=16)
batch = next(iter(loader_valid))
lib.display_batch(batch, 'subj_mean')

# %%
class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, activation=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if activation:
            self.layers = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            )
        else:
            self.layers = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, out_dim),
            )

    def forward(self, x):
        return self.layers(x)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        backbone = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        backbone = list(backbone.children())[:-2]
        backbone = nn.Sequential(*backbone)

        self.backbone = backbone
        
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            LinearBlock(1408, 1024, 0.25),
            LinearBlock(1024, 256, 0.25),
            LinearBlock(256, 1, 0, activation=False)
        )

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlp(x)
        return x

    def training_step(self, batch, batch_idx):
        """the full training loop"""
        x, y = batch['image'], batch['subj_mean']
        pred = self(x).flatten()
        loss = self.loss(pred, y)
        true = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        return {'loss': loss, 'results': (true, pred)}
    
    # OPTIONAL
    def validation_step(self, batch, batch_idx):
        """the full validation loop"""
        x, y = batch['image'], batch['subj_mean']
        pred = self(x).flatten()
        loss = self.loss(pred, y)
        true = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        return {'loss': loss, 'results': (true, pred)}

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters(), 'lr': 3e-5},
            {'params': self.mlp.parameters(), 'lr': 3e-4}
        ], weight_decay=3e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.2, 
            patience=5, 
            verbose=True)
            
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_srocc"
        } 

        return [optimizer], [lr_dict]

    # OPTIONAL
    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        true = np.concatenate([x['results'][0] for x in outputs])
        predicted = np.concatenate([x['results'][1] for x in outputs])

        plcc = stats.pearsonr(predicted, true)[0]
        srocc = stats.spearmanr(predicted, true)[0]

        print(f"| TRAIN plcc: {plcc:.2f}, srocc: {srocc:.2f}, loss: {avg_loss:.2f}" )

        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_plcc', plcc, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_srocc', srocc, prog_bar=True, on_epoch=True, on_step=False)

    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        true = np.concatenate([x['results'][0] for x in outputs])
        predicted = np.concatenate([x['results'][1] for x in outputs])
        
        plcc = stats.pearsonr(predicted, true)[0]
        srocc = stats.spearmanr(predicted, true)[0]

        print(f"[Epoch {self.trainer.current_epoch:3}] VALID plcc: {plcc:.2f}, srocc: {srocc:.2f}, loss: {avg_loss:.2f}", end= " ")

        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_plcc', plcc, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_srocc', srocc, prog_bar=True, on_epoch=True, on_step=False)

# %%
MyModelCheckpoint = ModelCheckpoint(dirpath='checkpoints/',
                                    filename='{epoch}_{val_srocc:.3f}_{val_plcc:.3f}_{val_loss:.3f}',
                                    monitor='val_srocc', 
                                    mode='max', 
                                    save_top_k=1,
                                    save_weights_only=True,
                                    verbose=False)

MyEarlyStopping = EarlyStopping(monitor = "val_srocc",
                                mode = "max",
                                patience = 15,
                                verbose = True)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=[0],
    callbacks=[MyEarlyStopping, MyModelCheckpoint],
    log_every_n_steps=1,
)

model = Model()

# %%
trainer.fit(model, loader_train, loader_valid)