from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
import os, sys
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class IQADataset(Dataset):
    def __init__(self, images_path, labels_path, mode, saliency_path=None, augment=True, transforms=None):
        assert isinstance(images_path, str) or isinstance(images_path, Path)
        assert isinstance(labels_path, str) or isinstance(labels_path, Path)
        assert saliency_path is None or isinstance(saliency_path, str) or isinstance(saliency_path, Path)
        assert mode in ['train', 'valid', 'test', 'all']
        assert transforms is None or isinstance(transforms, dict) and np.all([isinstance(t, A.BaseCompose) for t in transforms.values()])

        TRAIN_RATIO = 0.7
        TRAIN_VALID_RATIO = 0.8

        self.images_path = Path(images_path)
        self.labels_path = Path(labels_path)
        self.saliency_path = Path(saliency_path)

        self.saliency = True if saliency_path is not None else False

        df = pd.read_csv(labels_path).astype('float32', errors='ignore')
        available = os.listdir(images_path)
        df = df[df.name.isin(available)]

        train_size = int(TRAIN_RATIO * len(df))
        train_valid_size = int(TRAIN_VALID_RATIO * len(df))

        self.mode = mode

        if mode == 'train':
            self.df = df.iloc[:train_size]
        elif mode == 'valid':
            self.df = df.iloc[train_size:train_valid_size]
        elif mode == 'test':
            self.df = df.iloc[train_valid_size:]
        elif mode == 'all':
            self.df = df

        self.transforms = transforms
        self.augment = augment
        self.file2suffix_saliency = {Path(file).with_suffix(''): Path(file).suffix for file in os.listdir(saliency_path)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> tuple:
        name, subj_mean, subj_std = self.df.iloc[index]
        image = cv2.imread(str(self.images_path / name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.saliency:
            name_clean= Path(name).with_suffix('')
            name_sal = str(name_clean) + self.file2suffix_saliency[name_clean]
            name_sal = str(self.saliency_path / name_sal)
            saliency = cv2.imread(name_sal)
            saliency = saliency[..., 0:1]

        if self.transforms:
            NUM_CHANNELS = 3

            # first stage: preprocessing (resize)
            image = self.transforms['1_both'](image=image)['image']
            if self.saliency:
                saliency = self.transforms['1_both'](image=saliency)['image']
            image = np.dstack([image, saliency])

            # second stage: augmentations (hflip, rotate)
            if self.augment:
                image = self.transforms['2_both'](image=image)['image']

            # third stage: postprocessing (normalization, to_tensor)
            image, saliency = image[..., :NUM_CHANNELS], image[..., NUM_CHANNELS:]
            image = self.transforms['3_images'](image=image)['image']
            if self.saliency:
                saliency = self.transforms['3_saliency'](image=saliency)['image']
        
        out = {'image': image, 'name': name, 'subj_mean': subj_mean, 'subj_std': subj_std}
        if self.saliency:
            out['saliency'] = saliency
        return out

def get_datasets(opts, fast=False):
    transforms = {
        '1_both': A.Compose([
            A.Resize(*opts['image_shape']),
        ]),
        '2_both': A.Compose([
            A.HorizontalFlip()
        ]),
        '3_images': A.Compose([
            A.Normalize(),
            ToTensorV2(),
        ]),
        '3_saliency': A.Compose([
            A.Resize(*opts['saliency_shape']),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ])
    }
    if not fast:
        dataset_train = IQADataset(
            images_path=opts['koniq10k']['images'],
            saliency_path=opts['koniq10k']['saliency_maps'],
            labels_path=opts['koniq10k']['data'],
            mode='train', 
            augment=True,
            transforms=transforms)

        dataset_valid = IQADataset(
            images_path=opts['koniq10k']['images'],
            saliency_path=opts['koniq10k']['saliency_maps'],
            labels_path=opts['koniq10k']['data'],
            mode='valid', 
            augment=False,
            transforms=transforms)

        dataset_test_koniq = IQADataset(
            images_path=opts['koniq10k']['images'],
            saliency_path=opts['koniq10k']['saliency_maps'],
            labels_path=opts['koniq10k']['data'],
            mode='test', 
            augment=False,
            transforms=transforms)

        dataset_test_clive = IQADataset(
            images_path=opts['clive']['images'],
            saliency_path=opts['clive']['saliency_maps'],
            labels_path=opts['clive']['data'],
            mode='all', 
            augment=False,
            transforms=transforms)
    else:
        dataset_train = IQADataset(
            images_path=opts['clive']['images'],
            saliency_path=opts['clive']['saliency_maps'],
            labels_path=opts['clive']['data'],
            mode='valid', 
            augment=True,
            transforms=transforms)

        dataset_valid = IQADataset(
            images_path=opts['clive']['images'],
            saliency_path=opts['clive']['saliency_maps'],
            labels_path=opts['clive']['data'],
            mode='test', 
            augment=False,
            transforms=transforms)

        dataset_test_koniq = IQADataset(
            images_path=opts['clive']['images'],
            saliency_path=opts['clive']['saliency_maps'],
            labels_path=opts['clive']['data'],
            mode='test', 
            augment=False,
            transforms=transforms)

        dataset_test_clive = IQADataset(
            images_path=opts['clive']['images'],
            saliency_path=opts['clive']['saliency_maps'],
            labels_path=opts['clive']['data'],
            mode='all', 
            augment=False,
            transforms=transforms)

    return dict(
        dataset_train=dataset_train,
        dataset_valid=dataset_valid,
        dataset_test_koniq=dataset_test_koniq,
        dataset_test_clive=dataset_test_clive
    )

def get_loaders(opts, fast=False):
    datasets = get_datasets(opts, fast)
    loader_train = DataLoader(datasets['dataset_train'], batch_size=opts['batch_size'], shuffle=True, num_workers=opts['num_workers'])
    loader_valid = DataLoader(datasets['dataset_valid'], batch_size=opts['batch_size'], shuffle=False, num_workers=opts['num_workers'])
    loader_test_koniq = DataLoader(datasets['dataset_test_koniq'], batch_size=opts['batch_size'], shuffle=False, num_workers=opts['num_workers'])
    loader_test_clive = DataLoader(datasets['dataset_test_clive'], batch_size=opts['batch_size'], shuffle=False, num_workers=opts['num_workers'])
    return loader_train, loader_valid, loader_test_koniq, loader_test_clive