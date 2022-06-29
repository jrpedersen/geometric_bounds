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
from models.skip_mlp import SkipMLP
from data.mnist import MNISTDataModule
from utils import get_all_layers

from functorch import jacrev, vmap

def calc_bound(w, h, jh):
    return (
        (1 + linalg.vector_norm(h, dim=1)**2) /
        (linalg.matrix_norm(w, ord=2)**2 *
        linalg.matrix_norm(jh, ord='fro')**2)
    )

def calc_tight_bound(w, h, jh):
    return (
        (1 + linalg.vector_norm(h, dim=1)**2) /
        (linalg.matrix_norm(torch.bmm(w, jh), ord='fro')**2)
    )

def mlp_bound(weight_layer, partial_net, x, bound_f = calc_tight_bound):
    h_0 = x.view(x.shape[0],-1)
    if len(partial_net) > 0:
        h_n = partial_net(h_0)
        jh_n = vmap(jacrev(partial_net))(h_0)
    else:
        h_n = h_0
        jh_n = torch.eye(h_0.shape[1]).unsqueeze(0).repeat(x.shape[0],1,1)

    w = weight_layer.weight.repeat(x.shape[0],1,1)
    return bound_f(w, h_n, jh_n), (w,h_n,jh_n)

def skip_mlp_bound(weight_layer, partial_net, x, bound_f = calc_tight_bound):
    negative_slope = weight_layer.activation.negative_slope

    h_0 = x.view(x.shape[0],-1)
    h_n = partial_net(h_0)
    jh_n = vmap(jacrev(partial_net))(h_0)

    w_base = weight_layer.fc.weight.repeat(x.shape[0],1,1)
    w_skip =torch.eye(*w_base.shape[1:]).unsqueeze(0).repeat(x.shape[0],1,1)
    w_skip = w_skip * (1 + (torch.gt(-weight_layer.fc(h_n), 0) * (negative_slope**(-1) - 1 ))).unsqueeze(1)
    w = w_base + w_skip
    return bound_f(w, h_n, jh_n), (w,h_n,jh_n)

def get_bounds(model, x):
    modules_list = list(model.layers._modules.items())
    bound_types = {
        'fc': mlp_bound,
        'sk': skip_mlp_bound
    }
    layer_wise_bound = []
    for i in range(len(modules_list)):
        name, layer = modules_list[i]
        if name[:2] in bound_types.keys():
            bound, _ = bound_types[name[:2]](
                layer,
                nn.Sequential(OrderedDict(modules_list[:i])),
                x
            )
            layer_wise_bound.append(bound)
    pdb.set_trace()
    return torch.stack(layer_wise_bound).T

def compose_fns(f,g): return lambda x : g(f(x))

def norm_grad_params(model):
    return torch.tensor([(param.grad**2).sum() for param in model.parameters()]).sum()

def norm_grad_x(model, loss_fn, x, labels):
    grad_x = jacrev(compose_fns(model, partial(loss_fn,target=labels)))(x)
    return (grad_x**2).sum()

def norm_grads(model, loss_fn, optimizer, x, labels):
    optimizer.zero_grad()
    inputs = x.clone()
    inputs.requires_grad_(True)
    inputs.retain_grad()

    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()

    norm_grad_wrt_input = (inputs.grad**2).sum()
    norm_grad_wrt_params = torch.tensor([(param.grad**2).sum() for param in model.parameters()]).sum()
    return (norm_grad_wrt_input,
        norm_grad_wrt_params
    )

def train_one_epoch(training_loader, model, loss_fn,optimizer, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()

        #no need for vmap since loss fn has sum over batch
        grad_wrt_input = jacrev(compose_fns(model, partial(loss_fn,target=labels)))(inputs)
        norm_grad_wrt_input = (grad_wrt_input**2).sum()
        print(torch.allclose(norm_grad_x(model, loss_fn, inputs, labels), norm_grad_wrt_input))
        norm_grad_wrt_weights = torch.tensor([(param.grad**2).sum() for param in model.parameters()]).sum()
        print(torch.allclose(norm_grad_wrt_weights, norm_grad_params(model)))
        #
        print(norm_grad_x(model, loss_fn, inputs, labels), norm_grad_params(model))
        print(norm_grads(model, loss_fn, optimizer, inputs, labels))
        #test = torch.autograd.functional.jacobian(model, inputs)
        # def cfnew(f,g): return lambda x : f(g(x))
        #  torch.autograd.functional.jacobian(cfnew(partial(loss_fn,target=labels), model), inputs)
        # [torch.allclose(a,b) for (a,b) in zip(h_n, visualisation.values()[0:6:2])]


        #assert [torch.allclose(a,b[0]) for (a,b) in zip(h_n, list(visualisation.values())[0:6:2])]
        if 0:
            h_n, jh_n = get_hn_and_jhn(model, inputs)
            bound = old_mlp_bound(model,inputs)
            tight_bound=tighter_mlp_bound(model,inputs)
            print('f in', norm_grad_wrt_input)
            print('bound', bound)
            print('bound2', tight_bound)
            print('f in * bound', norm_grad_wrt_input*bound)
            print('Tighter f in * bound', norm_grad_wrt_input*tight_bound)
            print('f param', norm_grad_wrt_weights)
            #assert all([torch.allclose(a,b[0]) for (a,b) in zip(h_n, list(visualisation.values())[0:6:2])])

        test = get_bounds(model,inputs)
        print('\n\n new way to calc bounds')
        print(test)
        print(test.sum(dim=1))
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
