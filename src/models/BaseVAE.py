from typing import Any, List

import torch
import torch.nn as nn


class BaseVAE(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        features_list: List[int] = [16, 8, 4, 2],
        activation_func_name: str = "ReLU",
        dropout_p: float = 0.2,
        use_batch_norm: bool = False,
    ):
        """VAE with fully-connected layers

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
        """
        super().__init__()
        assert n_layers + 1 == len(
            features_list
        ), "should be: n_layers + 1 == len(features_list)"
        activation_func = getattr(nn, activation_func_name, nn.ReLU)

        encoder_layer: List[Any] = []
        for in_features, out_features in zip(features_list[:-2], features_list[1:-1]):
            # fully-connected layer
            encoder_layer.append(nn.Linear(in_features, out_features))
            # batchnorm1d
            if use_batch_norm:
                encoder_layer.append(nn.BatchNorm1d(out_features))
            # activation function
            encoder_layer.append(activation_func())
            # dropout
            if dropout_p != 0:
                encoder_layer.append(nn.Dropout(dropout_p))

        self.encoder = nn.Sequential(*encoder_layer)
        self.encoder_mu_layer = nn.Linear(features_list[-2], features_list[-1])
        self.encoder_logvar_layer = nn.Linear(features_list[-2], features_list[-1])

        decoder_layer: List[Any] = []
        for in_features, out_features in zip(
            features_list[::-1][:-1], features_list[::-1][1:]
        ):
            # fully-connected layer
            decoder_layer.append(nn.Linear(in_features, out_features))
            # batchnorm1d
            if use_batch_norm:
                decoder_layer.append(nn.BatchNorm1d(out_features))
            # activation function
            decoder_layer.append(activation_func())
            # dropout
            if dropout_p != 0:
                decoder_layer.append(nn.Dropout(dropout_p))
        self.decoder = nn.Sequential(*decoder_layer)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """encode part in vae

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            mu, logvar
        """
        h = self.encoder(x)
        mu = self.encoder_mu_layer(h)
        logvar = self.encoder_logvar_layer(h)
        return mu, logvar

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """reparameterize in vae

        Parameters
        ----------
        mu : torch.Tensor
            _description_
        logvar : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            reparameterized result of encoder (= mu + esp + std)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        # rand_like
        # Returns a Tensor with the same size as input
        # that is filled with random numbers from a uniform distribution
        return mu + eps * std

    def _decode(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """decode part in vae

        Parameters
        ----------
        z : torch.Tensor
            latent variable

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            z, mu, logvar
        """

        x = self.decoder(z)
        x = torch.sigmoid(x)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            reconstructed x, mu, logvar
        """
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        return self._decode(z), mu, logvar
