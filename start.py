import re
from typing import Optional
from collections import OrderedDict
from functools import partial
import pdb

import torch
from torch import nn, optim, linalg

import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

from models.simple_mlp import SimpleMLP
from models.skip_mlp import SkipMLP
from data.mnist import MNISTDataModule
from func_geometric_bounds import norm_grad_x, norm_grad_params, get_bounds

def main():
    mnist = MNISTDataModule(data_dir="./data", batch_size=32)


    #simple_mlp = SimpleMLP([40,40,10])  # this is our LightningModule
    model = SkipMLP([40,40,40,40,10], 0.01)
    model_name = re.findall(r"[\w]+", str(type(model)))[-1]
    logger = TensorBoardLogger('lightning_logs/', name=model_name)
    trainer = pl.Trainer(max_epochs=1,
                         num_processes=1,
                         #accelerator='gpu',
                         #devices=1,
                         logger=logger,
                         deterministic=True)
    trainer.fit(model, datamodule=mnist)


if __name__ == '__main__':
    pl.seed_everything(1234)
    main()
    #main_manual_train()
    #print(cifar100.train_dataloader)
    #trainer = pl.Trainer(max_epochs=1, num_processes=1, gpus=0)
    #trainer.fit(model, datamodule=cifar100)
