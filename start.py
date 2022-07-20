import re
import os
import itertools
from typing import Optional
from collections import OrderedDict
from functools import partial

import git

import torch
from torch import nn, optim, linalg

import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

from src.models.mlp import SkipMLP, SimpleMLP
from src.data.mnist import MNISTDataModule

def main(config):
    mnist = MNISTDataModule(data_dir="./data", batch_size=config['batch_size'])
    model = config['model_type'](config)
    model_name = re.findall(r"[\w]+", str(type(model)))[-1]

    logger = TensorBoardLogger(
        'lightning_logs/',
        name=model_name,
        version="depth"+str(config['hl_depth'])
    )
    logger.log_hyperparams(config)

    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         num_processes=1,
                         #accelerator='gpu',
                         #devices=1,
                         logger=logger,
                         deterministic=True)
    trainer.fit(model, datamodule=mnist)


if __name__ == '__main__':
    pl.seed_everything(1234)
    # Git current git commit:
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    config = {
        # git revision
        'sha': sha,
        # dataset
        'batch_size': 32,
        # model config
        'model_type': SimpleMLP,
        #'num_parameters': (28**2 + 10) * w + (h-1)*w**2 # for single P 28**2 * 10
        #'hl_depth': 2,
        #'hl_width': 40,
        'negative_slope': 0.01,
        # training
        'max_epochs': 20,
    }
    # w * ((28*2 + 10) + (h-1)*w) aaprox h*w^2
    # TODO: Find way to split depth versus wide.
    # (h,w) = [(1,8), (4,4), (16,2), (64,1), () ]
    # TODO: Change loop to set width and weight dependent on num parameters
    # TODO: Perhaps around 10 layers of 40x40 size ish. And 1,2,4 to that.
    # (2**4 * 10**4)
    # OLD
    num_params = lambda w, h : (h-1)*w**2 + (28**2 + 10) * w
    base_width = 8 * 40 //2
    base_depth = 1
    for n in range(3):
        config['hl_width'] = base_width // (2**n)
        config['hl_depth'] = base_depth * 2**(2*n) + int(
            2**(-0.5) * 2**(2**0.75*(n-1)) * (n>0) * (28**2 + 10)/config['hl_width'] #* (1-2**(-n))
        )
        #print(config['hl_width'])
        #print(config['hl_depth'])
        #print(num_params(config['hl_width'], config['hl_depth']))
        main(config)
    # Add configuration for single wide network with all the parameters


    #main_manual_train()
    #print(cifar100.train_dataloader)
    #trainer = pl.Trainer(max_epochs=1, num_processes=1, gpus=0)
    #trainer.fit(model, datamodule=cifar100)
