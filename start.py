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
#from collections.abc import Mapping
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

import pdb


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
        return DataLoader(self.mnist_train, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)


# class simpler dataset

# class model_factory
class SimpleMLP(pl.LightningModule):
    def __init__(self, hidden_layer_config, wealy_relu_neg_slope=0.01):
        super().__init__()
        next_layer_input = 784
        #layers = []
        layers = OrderedDict()
        for _i, hidden_layer in enumerate(hidden_layer_config):
            # Create layer
            # layers.append(nn.Linear(in_features=next_layer_input, out_features=hidden_layer))
            # layers.append(nn.LeakyReLU(negative_slope=wealy_relu_neg_slope))
            layers.update({'fc'+ str(_i): nn.Linear(in_features=next_layer_input, out_features=hidden_layer)})
            layers.update({'af'+ str(_i): nn.LeakyReLU(negative_slope=wealy_relu_neg_slope)})
            # Update input size
            next_layer_input = hidden_layer

        self.layers = nn.Sequential(layers)

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

# function for MLP bound
def mlp_term_bound(mlp_layer):
    weight_spectral_norm = linalg.matrix_norm(mlp_layer.weight, ord=2)
    h_n = linalg.vector_norm(1)
    jac_h_n =linalg.matrix_norm(1, ord='fro')
    return (1 + h_n**2) / (weight_spectral_norm**2 * jac_h_n**2)



visualisation = {}
def named_hook(m,i,o, name=None):
    #print(type(i))
    #for elem in i: print(type(elem))
    #print(i)
    visualisation[name] = i

def hook_fn(m, i, o):
    visualisation[m] = i

def get_all_layers(net):
    for name, layer in net._modules.items():
    #If it is a sequential, don't register a hook on it
    # but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer)
        else:
        # it's a non sequential. Register a hook
            hook = partial(named_hook, name=name)
            layer.register_forward_hook(hook)
            # layer.register_forward_hook(hook_fn)

# function for residual MLP bound

# class bound_evaluator

# class residual_bound_evaluator

from functorch import jacrev

def get_hn_and_jhn(model, x):
    h_0 = x.view(1,-1)
    jh_0 = torch.eye(h_0.shape[1]).unsqueeze(0)

    h_n = [h_0]
    jh_n = [jh_0]

    for n in range(1,3):
        partial_net =nn.Sequential(OrderedDict(list(model.layers._modules.items())[:(n*2)]))
        h_n.append(partial_net(h_0))
        jh_n.append(torch.squeeze(jacrev(partial_net)(h_0)))
    return h_n, jh_n


def mlp_bound(model, x):
    h_n, jh_n = get_hn_and_jhn(model, x)
    params = [param for (i,param) in enumerate(model.parameters()) if i%2==0]
    bound = 0
    for w, h, jh in zip(params,h_n,jh_n):
        bound += (
            (1 + linalg.vector_norm(h)**2) /
            (linalg.matrix_norm(w, ord=2)**2 *
            linalg.matrix_norm(jh, ord='fro')**2)
        )
    return bound
#a=nn.Sequential(OrderedDict(list(model.layers._modules.items())[:4]))

#ft_jacobian = jacrev(predict, argnums=2)(weight, bias, x)

#jacrev(a)(inputs.view(1,-1))

def train_one_epoch(training_loader, model, loss_fn,optimizer, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs.requires_grad_(True)
        inputs.retain_grad()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)


        loss.backward()


        norm_grad_wrt_input = (inputs.grad**2).sum()
        norm_grad_wrt_weights = torch.tensor([(param.grad**2).sum() for param in model.parameters()]).sum()
        #
        #test = torch.autograd.functional.jacobian(model, inputs)
        # def cfnew(f,g): return lambda x : f(g(x))
        #  torch.autograd.functional.jacobian(cfnew(partial(loss_fn,target=labels), model), inputs)
        # [torch.allclose(a,b) for (a,b) in zip(h_n, visualisation.values()[0:6:2])]
        inputs.requires_grad_(False)

        #assert [torch.allclose(a,b[0]) for (a,b) in zip(h_n, list(visualisation.values())[0:6:2])]

        h_n, jh_n = get_hn_and_jhn(model, inputs)

        bound = mlp_bound(model,inputs)
        print('f in', norm_grad_wrt_input)
        print('bound', bound)
        print('f in * bound', norm_grad_wrt_input*bound)
        print('f param', norm_grad_wrt_weights)
        #assert all([torch.allclose(a,b[0]) for (a,b) in zip(h_n, list(visualisation.values())[0:6:2])])


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
    mnist = MNISTDataModule(data_dir="./data")
    simple_mlp = SimpleMLP([40,40,10], 0.01)
    get_all_layers(simple_mlp)
    logger = TensorBoardLogger('lightning_logs/', name='my_model')
    mnist.setup()
    optimizer = optim.Adam(simple_mlp.parameters(), lr=1e-3)

    train_one_epoch(mnist.train_dataloader(), simple_mlp, nn.CrossEntropyLoss(label_smoothing=0.1), optimizer, 1, None)

def main():
    mnist = MNISTDataModule(data_dir="./data")
    simple_mlp = SimpleMLP([40,40,10])  # this is our LightningModule
    logger = TensorBoardLogger('lightning_logs/', name='my_model')
    trainer = pl.Trainer(max_epochs=1, num_processes=1, logger=logger, deterministic=True)
    trainer.fit(simple_mlp, datamodule=mnist)


if __name__ == '__main__':
    pl.seed_everything(1234)
    main_manual_train()
    #print(cifar100.train_dataloader)
    #trainer = pl.Trainer(max_epochs=1, num_processes=1, gpus=0)
    #trainer.fit(model, datamodule=cifar100)
