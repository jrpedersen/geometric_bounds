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
from data.mnist import MNISTDataModule


# class simpler dataset
        #self.criterion(self(inputs), labels)

        #return {"loss": loss, "pred": max_pred}

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

def tighter_mlp_bound(model, x):
    h_n, jh_n = get_hn_and_jhn(model, x)
    params = [param for (i,param) in enumerate(model.parameters()) if i%2==0]
    bound = 0
    for w, h, jh in zip(params,h_n,jh_n):
        bound += (
            (1 + linalg.vector_norm(h)**2) /
            (linalg.matrix_norm(torch.matmul(w, jh), ord='fro')**2)
        )
    return bound


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

        print('bound2', tighter_mlp_bound(model, inputs))

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
