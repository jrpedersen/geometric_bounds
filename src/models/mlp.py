from collections import OrderedDict
import pdb
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
import src.func_geometric_bounds as gb

class RunBase(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def configure_optimizers(self):pass

    def training_step(self, batch, batch_idx):
        data, target = batch

        # Bound statistics
        if batch_idx %16==0:
            bounds = gb.get_bounds(self, data)
            gradients_data, gradients_params = gb.norm_gradients(self, nn.CrossEntropyLoss(label_smoothing=0.1), data, target)
            self.log('bounds/Gradients_x', gradients_data.sum())
            self.log('bounds/Bound', bounds.sum())
            self.log('bounds/Bound div Gradients_x', (bounds.sum(dim=1)/gradients_data).mean())
            self.log('bounds/Gradients_x div Bound', (gradients_data/bounds.sum(dim=1)).mean())
            self.log('bounds/Gradients_x times bound',(
                gradients_data* bounds.sum(dim=1)
            ).sum(), on_step=True)
            self.log('bounds/Gradients parameters',gradients_params.sum(), on_step=True)

        preds = self(data)
        loss = F.cross_entropy(preds, target, label_smoothing=0.1)
        # Logging to TensorBoard by default
        self.log('train/train_acc', accuracy(preds, target), on_step=True, on_epoch=True)
        self.log("train/train_loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        data, target = valid_batch
        preds = self(data)
        _, max_pred = torch.max(preds, 1)
        loss = F.cross_entropy(preds, target, label_smoothing=0.1)
        self.log("validation/valid_loss", loss)
        self.log('validation/valid_acc', accuracy(preds, target), on_step=True, on_epoch=True)
        return loss


class SimpleMLP(RunBase):
    def __init__(self, config):
        super(RunBase, self).__init__()
        next_layer_input = 784
        #layers = []
        hidden_layers = config['hl_depth'] * [config['hl_width']]
        layers = OrderedDict()
        for _i, hidden_layer in enumerate(hidden_layers):
            # Create layer
            layers.update({'fc'+ str(_i): nn.Linear(in_features=next_layer_input, out_features=hidden_layer)})
            layers.update({'af'+ str(_i): nn.LeakyReLU(negative_slope=config['negative_slope'])})
            # Update input size
            next_layer_input = hidden_layer
        layers.update({'fc'+ str(len(hidden_layers)): nn.Linear(in_features=next_layer_input, out_features=10)})
        layers.update({'af'+ str(len(hidden_layers)): nn.LeakyReLU(negative_slope=config['negative_slope'])})
        self.layers = nn.Sequential(layers)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x_resized = x.view(batch_size, -1)
        return self.layers(x_resized)


class SkipMlpBlock(nn.Module):
    def __init__(self, in_features, out_features, negative_slope):
        super(SkipMlpBlock, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
    def forward(self, x):
        return x + self.activation(self.fc(x))


class SkipMLP(RunBase):
    def __init__(self, config):
        super(RunBase, self).__init__()
        next_layer_input = 784
        #layers = []
        hidden_layers = config['hl_depth'] * [config['hl_width']]
        layers = OrderedDict()
        for _i, hidden_layer in enumerate(hidden_layers):
            if next_layer_input == hidden_layer:
                layers.update({'skip_fc'+str(_i): SkipMlpBlock(in_features=next_layer_input, out_features=hidden_layer, negative_slope=config['negative_slope'])})
            else:
                layers.update({'fc'+ str(_i): nn.Linear(in_features=next_layer_input, out_features=hidden_layer)})
                layers.update({'af'+ str(_i): nn.LeakyReLU(negative_slope=config['negative_slope'])})
            # Update input size
            next_layer_input = hidden_layer
        layers.update({'fc'+ str(len(hidden_layers)): nn.Linear(in_features=next_layer_input, out_features=10)})
        layers.update({'af'+ str(len(hidden_layers)): nn.LeakyReLU(negative_slope=config['negative_slope'])})
        self.layers = nn.Sequential(layers)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x_resized = x.view(batch_size, -1)
        return self.layers(x_resized)
