# labelwise to pytorch_lightning
# import os
# import random
# import shutil
# import time
# import warnings
# import functools
# import copy
#
# import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
#
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# import torch.multiprocessing as mp
# import torch.utils.data
# import torch.utils.data.distributed
#
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models

from typing import Optional
import torch
from torch import nn, optim

import torchmetrics

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms



class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)


# class simpler dataset

# class model_factory
class SimpleMLP(pl.LightningModule):
    def __init__(self, hidden_layer_config):
        super().__init__()
        layers = []
        next_layer_input = 784
        for hidden_layer in hidden_layer_config:
            # Create layer
            layers.append(
                nn.Linear(in_features=next_layer_input, out_features=hidden_layer)
            )
            layers.append(nn.LeakyReLU())
            # Update input size
            next_layer_input = hidden_layer

        self.layers = nn.Sequential(*layers)

        self.criterium = nn.CrossEntropyLoss(label_smoothing= 0.1)

        # metrics
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x_resized = x.view(batch_size, -1)
        return self.layers(x_resized)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)

        loss = self.criterium(preds, target)
        # Logging to TensorBoard by default
        self.train_acc(preds, target)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        data, target = valid_batch
        preds = self(data)
        _, max_pred = torch.max(preds, 1)
        loss = self.criterium(preds, target)
        self.log("valid_loss", loss)

        self.valid_acc(preds, target)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
        #self.criterion(self(inputs), labels)

        #return {"loss": loss, "pred": max_pred}


# class bound_evaluator

# class residual_bound_evaluator

from pytorch_lightning.loggers import TensorBoardLogger


def main():
    mnist = MNISTDataModule(data_dir="./data")
    model = SimpleMLP([40,40,10])  # this is our LightningModule
    logger = TensorBoardLogger('lightning_logs/', name='my_model')

    trainer = pl.Trainer(max_epochs=1, num_processes=1, logger=logger, deterministic=True)


    trainer.fit(model, datamodule=mnist)


if __name__ == '__main__':
    pl.seed_everything(1234)
    main()
    #print(cifar100.train_dataloader)
    #trainer = pl.Trainer(max_epochs=1, num_processes=1, gpus=0)
    #trainer.fit(model, datamodule=cifar100)
