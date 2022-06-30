from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.functional import accuracy
import pytorch_lightning as pl

class SkipMlpBlock(nn.Module):
    def __init__(self, in_features, out_features, negative_slope):
        super(SkipMlpBlock, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
    def forward(self, x):
        return x + self.activation(self.fc(x))

class SkipConnection(nn.Module):
    def __init__(self):
        super(SkipConnection, self).__init__()

def skip_mlp_block_list(in_features, out_features, negative_slope):
        fc = nn.Linear(in_features=in_features, out_features=out_features)
        activation = nn.LeakyReLU(negative_slope=negative_slope)
        skip_connection = lambda x, y : x + y
        return fc, activation, skip_connection

class SkipMLP(pl.LightningModule):
    def __init__(self, hidden_layer_config, weakly_relu_neg_slope=0.01):
        super().__init__()
        next_layer_input = 784
        #layers = []
        layers = OrderedDict()
        for _i, hidden_layer in enumerate(hidden_layer_config):
            # Create layer
            # layers.append(nn.Linear(in_features=next_layer_input, out_features=hidden_layer))
            # layers.append(nn.LeakyReLU(negative_slope=wealy_relu_neg_slope))
            if next_layer_input == hidden_layer:
                layers.update({'skip_fc'+str(_i): SkipMlpBlock(in_features=next_layer_input, out_features=hidden_layer, negative_slope=weakly_relu_neg_slope)})
            else:
                layers.update({'fc'+ str(_i): nn.Linear(in_features=next_layer_input, out_features=hidden_layer)})
                layers.update({'af'+ str(_i): nn.LeakyReLU(negative_slope=weakly_relu_neg_slope)})
                # Update input size
            next_layer_input = hidden_layer

        self.layers = nn.Sequential(layers)


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
        loss = F.cross_entropy(preds, target, label_smoothing=0.1)
        # Logging to TensorBoard by default
        self.log('train_acc', accuracy(preds, target), on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        data, target = valid_batch
        preds = self(data)
        _, max_pred = torch.max(preds, 1)
        loss = F.cross_entropy(preds, target, label_smoothing=0.1)
        self.log("valid_loss", loss)
        self.log('valid_acc', accuracy(preds, target), on_step=True, on_epoch=True)
        return loss
