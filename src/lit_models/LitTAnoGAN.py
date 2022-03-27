from typing import List

import pytorch_lightning as pl
import torch
import torch.optim

from src.models.TAnoGAN import LSTMDiscriminator, LSTMGenerator


class LitTAnoGAN(pl.LightningModule):
    def __init__(
        self,
        G: LSTMGenerator = LSTMGenerator(),
        D: LSTMDiscriminator = LSTMDiscriminator(),
        optimizer: str = "Adam",
        loss_function: str = "BCELoss",
        seq_len: int = 8,
    ):
        """TAnoGAn litmodel

        Parameters
        ----------
        G : LSTMGenerator, optional
            _description_, by default LSTMGenerator()
        D : LSTMDiscriminator, optional
            _description_, by default LSTMDiscriminator()
        optimizer : str, optional
            _description_, by default "Adam"
        loss_function : str, optional
            _description_, by default "BCELoss"
        seq_len : int, optional
            _description_, by default 8
        """
        super().__init__()
        self.g = G
        self.d = D
        self.seq_len = seq_len
        self.optimizerG = getattr(torch.optim, optimizer, torch.optim.Adam)
        self.optimizerD = getattr(torch.optim, optimizer, torch.optim.Adam)
        self.loss_function = getattr(torch.nn, loss_function, torch.nn.MSELoss)
        self.loss_function = self.loss_function()

    # def forward(self, x):
    #     return self.g(x)

    def configure_optimizers(self):
        optimizer_g = self.optimizerG(self.g.parameters())
        optimizer_d = self.optimizerD(self.d.parameters())
        return [optimizer_g, optimizer_d]

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        # |x| = (batch_size, seq_len, input_size)

        # generate latent_z
        z = torch.randn(x.size(0), self.seq_len, self.g.input_size)
        z = z.type_as(x)

        # train generator
        # maximize log(D(G(z)))
        if optimizer_idx == 0: 

            generated_data, _ = self.g(z)
            # |generated_data| = |x|

            fake, _ = self.d(generated_data)
            # |fake| = (batch_size, 1)
            fake_label = torch.ones(fake.size(0), 1)
            fake_label = fake_label.type_as(x)
            loss_g = self.loss_function(fake, fake_label)
            self.log("train_loss_g", loss_g, on_epoch=True)
            return loss_g        

        # train discriminator
        # maximize log(D(x)) + log(1 - D(G(z)))
        if optimizer_idx == 1:
            # real data loss
            real_label = torch.zeros(x.size(0), 1)
            real_label = real_label.type_as(x)
            real, _ = self.d(x)
            loss_d_real = self.loss_function(real, real_label)
            # fake data loss
            generated_data, _ = self.g(z)
            fake, _ = self.d(generated_data)
            fake_label = torch.ones(fake.size(0), 1)
            fake_label = fake_label.type_as(x)
            loss_d_fake = self.loss_function(fake, fake_label)
            # average
            loss_d = (loss_d_real + loss_d_fake) / 2
            self.log("train_loss_d", loss_d, on_epoch=True)
            return loss_d

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        # generator loss
        z = torch.randn(x.size(0), self.seq_len, self.g.input_size)
        z = z.type_as(x)
        generated_data, _ = self.g(z)
        fake, _ = self.d(generated_data)
        fake_label = torch.ones(fake.size(0), 1)
        fake_label = fake_label.type_as(x)
        loss_g = self.loss_function(fake, fake_label)
        # discriminator loss
        real_label = torch.zeros(x.size(0), 1)
        real_label = real_label.type_as(x)
        real, _ = self.d(x)
        loss_d_real = self.loss_function(real, real_label)
        generated_data, _ = self.g(z)
        fake, _ = self.d(generated_data)
        fake_label = torch.ones(fake.size(0), 1)
        fake_label = fake_label.type_as(x)
        loss_d_fake = self.loss_function(fake, fake_label)
        loss_d = (loss_d_real + loss_d_fake) / 2.0
        # val_loss = (generator loss + discriminator loss) / 2.0
        val_loss = (loss_g + loss_d) / 2.0
        self.log("val_loss", val_loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        output_x = self(x)
        loss = self.loss_function(output_x, x)
        self.log("test_loss", loss, on_epoch=True)
