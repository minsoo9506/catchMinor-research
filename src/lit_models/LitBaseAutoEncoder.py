from typing import List

import pytorch_lightning as pl
import torch
import torch.optim

from src.models.BaseAutoEncoder import BaseAutoEncoder


class LitBaseAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        n_layers: int = 3,
        features_list: List[int] = [16, 8, 4, 2],
        activation_func_name: str = "ReLU",
        dropout_p: float = 0.2,
        use_batch_norm: bool = False,
        optimizer: str = "Adam",
        loss_function: str = "MSELoss",
    ):
        """AE with fully-connected layer

        Parameters
        ----------
        n_layers : int, optional
            num of layers in encoder (= or decoder), by default 3
        features_list : List[int], optional
            in_features and out_features in each layers iteratively, by default [16, 8, 4, 2]
        activation_func_name : str, optional
            should match the name in torch.nn, by default "ReLU"
        dropout_p : float, optional
            if 0, no dropout layer, by default 0.2
        use_batch_norm : bool, optional
            _description_, by default False
        optimizer : str, optional
            name of optimizer in torch.optim, by default 'Adam'
        loss_function : str, optional
            name of loss function in torch.nn, by defaul 'MSELoss'
        """
        super().__init__()
        self.model = BaseAutoEncoder(
            n_layers, features_list, activation_func_name, dropout_p, use_batch_norm
        )
        self.optimizer = getattr(torch.optim, optimizer, torch.optim.Adam)
        self.loss_function = getattr(torch.nn, loss_function, torch.nn.MSELoss)
        self.loss_function = self.loss_function()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch
        output_x = self(x)
        loss = self.loss_function(output_x, x)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        output_x = self(x)
        loss = self.loss_function(output_x, x)
        self.log("val_loss", loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        output_x = self(x)
        loss = self.loss_function(output_x, x)
        self.log("test_loss", loss, on_epoch=True)