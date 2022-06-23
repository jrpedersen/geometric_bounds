from collections import OrderedDict

import torch
from torch import nn, optim
import torchmetrics

import pytorch_lightning as pl

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
        return loss
