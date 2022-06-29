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
from utils import get_all_layers
from func_geometric_bounds import norm_grad_x, norm_grad_params, get_bounds

def train_one_epoch(training_loader, model, loss_fn,optimizer, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # test bounds and gradients
        bound = get_bounds(model, inputs).sum()
        norm_grad_wrt_x = norm_grad_x(model, loss_fn, inputs, labels)
        norm_grad_wrt_params = norm_grad_params(model)
        print('Gradient with x:   ', norm_grad_wrt_x)
        print('Bound:             ', bound)
        print('Gradient with p:   ', norm_grad_wrt_params)
        print(f"Is {bound*norm_grad_wrt_x} < {norm_grad_wrt_params}")
        print(bound*norm_grad_wrt_x < norm_grad_wrt_params)
        pdb.set_trace()
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            #tb_x = epoch_index * len(training_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



def main_manual_train():
    mnist = MNISTDataModule(data_dir="./data", batch_size=1)
    weakly_relu_neg_slope = 0.01
    #simple_mlp = SimpleMLP([40,40,10], 0.01)
    #simple_mlp = SimpleMLP([40,40,40,40,10], 0.01)
    simple_mlp = SkipMLP([40,40,40,40,10], weakly_relu_neg_slope)
    logger = TensorBoardLogger('lightning_logs/', name='my_model')
    mnist.setup()
    optimizer = optim.Adam(simple_mlp.parameters(), lr=1e-3)

    train_one_epoch(mnist.train_dataloader(), simple_mlp, nn.CrossEntropyLoss(label_smoothing=0.1), optimizer, 1, None)

def main():
    mnist = MNISTDataModule(data_dir="./data", batch_size=32)


    #simple_mlp = SimpleMLP([40,40,10])  # this is our LightningModule
    simple_mlp = SkipMLP([40,40,40,40,10], 0.01)
    logger = TensorBoardLogger('lightning_logs/', name='my_model')
    trainer = pl.Trainer(max_epochs=1,
                         num_processes=1,
                         accelerator='gpu',
                         devices=1,
                         logger=logger,
                         deterministic=True)
    trainer.fit(simple_mlp, datamodule=mnist)


if __name__ == '__main__':
    pl.seed_everything(1234)
    #main()
    main_manual_train()
    #print(cifar100.train_dataloader)
    #trainer = pl.Trainer(max_epochs=1, num_processes=1, gpus=0)
    #trainer.fit(model, datamodule=cifar100)
