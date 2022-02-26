from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.models.BaseVAE import BaseVAE

# import wandb


def loss_function(output_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(output_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class LitBaseVAE(pl.LightningModule):
    def __init__(
        self,
        n_layers: int = 3,
        features_list: List[int] = [16, 8, 4, 2],
        activation_func_name: str = "ReLU",
        dropout_p: float = 0.2,
        use_batch_norm: bool = False,
        optimizer: str = "Adam",
    ):
        """VAE with fully-connected layer

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
        """
        super().__init__()
        self.model = BaseVAE(
            n_layers, features_list, activation_func_name, dropout_p, use_batch_norm
        )
        self.optimizer = getattr(torch.optim, optimizer, torch.optim.Adam)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch
        output_x, mu, logvar = self(x)
        loss = loss_function(output_x, x, mu, logvar)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        output_x, mu, logvar = self(x)
        loss = loss_function(output_x, x, mu, logvar)
        self.log("val_loss", loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        output_x, mu, logvar = self(x)
        loss = loss_function(output_x, x, mu, logvar)
        self.log("test_loss", loss, on_epoch=True)

    # def test_step_end(self, test_step_outputs):
    #     dummy_input = torch.zeros(784, device=self.config.cuda)
    #     model_filename = "model_final.onnx"
    #     torch.onnx.export(self, dummy_input, model_filename)
    #     wandb.save(model_filename)


# class ImagePredictionLogger(pl.Callback):
#     def __init__(self, val_samples):
#         super().__init__()
#         self.val_imgs = val_samples

#     def on_validation_epoch_end(self, trainer, pl_module):
#         val_imgs = self.val_imgs.to(device=pl_module.device)

#         outputs, _, _ = pl_module(val_imgs)
#         reconstructed_img = outputs.view(-1, 28, 28).unsqueeze(-1)

#         trainer.logger.experiment.log(
#             {
#                 "examples": [
#                     wandb.Image(reconstructed_img, caption="reconstructed image")
#                 ],
#             }
#         )
