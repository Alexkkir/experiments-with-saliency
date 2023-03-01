import yaml
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from math import ceil
import numpy as np
import torch
import cv2
import datetime
from argparse import ArgumentParser
import json

from .constants import MEAN, STD, CONFIG



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

def today():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")

def get_default_opts():
    return yaml.load(open(CONFIG), Loader=yaml.FullLoader)

def get_args():
    parser = ArgumentParser()
    
    parser.add_argument('--name', type=str, required=True, help='name of experiment')
    parser.add_argument('--device', type=int, required=True, help='index of cuda device')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=False)

    parser.add_argument('--vis-sal', dest='visualize_saliency', action='store_true')
    parser.set_defaults(visualize_saliency=False)
    
    args = parser.parse_args()
    opts = vars(args)
    print(json.dumps(opts, indent=4))
    
    opts.update(get_default_opts())

    return opts