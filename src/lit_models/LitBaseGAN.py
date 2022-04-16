from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.BaseGAN import BaseDiscriminator, BaseGenerator


class LitGAN(pl.LightningModule):
    def __init__(
        self,
        G: BaseGenerator = BaseGenerator(),
        D: BaseDiscriminator = BaseDiscriminator(),
        batch_size: int = 32,
        latent_dim: int = 2,
        **kwargs
    ):
        """LitGAN

        Parameters
        ----------
        G : BaseGenerator, optional
            _description_, by default BaseGenerator()
        D : BaseDiscriminator, optional
            _description_, by default BaseDiscriminator()
        batch_size : int, optional
            _description_, by default 32
        latent_dim : int, optional
            dimension of latent z, by default 2
        """
        super().__init__()
        self.latent_dim = latent_dim

        self.generator = G
        self.discriminator = D

    def forward(self, z):
        return self.generator(z)

    def _adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch

        # sample noise
        z = torch.randn(x.shape[0], self.hparams.latent_dim)
        z = z.type_as(x)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # ground truth result (ie: all fake)
            real = torch.ones(x.size(0), 1)
            real = real.type_as(x)

            # binary cross-entropy
            g_loss = self._adversarial_loss(self.discriminator(self(z)), real)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

        # train discriminator
        if optimizer_idx == 1:

            real = torch.ones(x.size(0), 1)
            real = real.type_as(x)
            real_loss = self.adversarial_loss(self.discriminator(x), real)

            fake = torch.zeros(x.size(0), 1)
            fake = fake.type_as(x)
            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake
            )

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters())
        opt_d = torch.optim.Adam(self.discriminator.parameters())
        return [opt_g, opt_d], []
