import re
import os
import itertools
from typing import Optional
from collections import OrderedDict
from functools import partial

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
    import pdb; pdb.set_trace()
    model = config['model_type'](config)
    model_name = re.findall(r"[\w]+", str(type(model)))[-1]

    logger = TensorBoardLogger('lightning_logs/', name=model_name, version="depth"+str(config['hl_depth']))
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
    config = {
        # dataset
        'batch_size': 32,
        # model config
        'model_type': SimpleMLP,
        'hl_depth': 2,
        'hl_width': 40,
        'negative_slope': 0.01,
        # training
        'max_epochs': 40,
    }

    for model_type, hl_depth in itertools.product(
        [SimpleMLP, SkipMLP],
        range(2,11,2)
    ):
        config['model_type'] = model_type
        config['hl_depth'] = hl_depth
        main(config)
    #main_manual_train()
    #print(cifar100.train_dataloader)
    #trainer = pl.Trainer(max_epochs=1, num_processes=1, gpus=0)
    #trainer.fit(model, datamodule=cifar100)
