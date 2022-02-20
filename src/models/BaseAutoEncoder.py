from typing import List

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        features_list: List[int] = [16, 8, 4, 2],
        activation_func_name: str = "ReLU",
        dropout_p: float = 0.2,
        use_batch_norm: bool = False,
    ):
        """Encoder in BaseAutoEncoder

        Parameters
        ----------
        n_layers : int, optional
            num of layers, by default 3
        features_list : List[int], optional
            in_ and out_features in each layers iteratively
            features_list[0] is in_features of the first layer
            features_list[-1] is out_features of the last layer
            by default [16, 8, 4, 2]
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

        layers = []
        # order of layers (reference):
        # https://gaussian37.github.io/dl-concept-order_of_regularization_term/
        for in_features, out_features in zip(features_list[:-1], features_list[1:]):
            # fully-connected layer
            layers.append(nn.Linear(in_features, out_features))
            # batchnorm1d
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            # activation function
            layers.append(activation_func())
            # dropout
            if dropout_p != 0:
                layers.append(nn.Dropout(dropout_p))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model(x)
        # |z| = (batch_size, features_list[-1])
        return z


class BaseDecoder(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        features_list: List[int] = [2, 4, 8, 16],
        activation_func_name: str = "ReLU",
        dropout_p: float = 0.2,
        use_batch_norm: bool = False,
    ):
        """Decoder in BaseAutoEncoder

        Parameters
        ----------
        n_layers : int, optional
            num of layers, by default 3
        features_list : List[int], optional
            in_ and out_features in each layers iteratively
            features_list[0] is in_features of the first layer
            features_list[-1] is out_features of the last layer
            by default [2, 4, 8, 16]
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

        layers = []
        for in_features, out_features in zip(features_list[:-1], features_list[1:]):
            # fully-connected layer
            layers.append(nn.Linear(in_features, out_features))
            # batchnorm1d
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            # activation function
            layers.append(activation_func())
            # dropout
            if dropout_p != 0:
                layers.append(nn.Dropout(dropout_p))

        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.model(z)
        # |x| = (batch_size, features_list[-1])
        return x


class BaseAutoEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        features_list: List[int] = [16, 8, 4, 2],
        activation_func_name: str = "ReLU",
        dropout_p: float = 0.2,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        self.encoder = BaseEncoder(
            n_layers, features_list, activation_func_name, dropout_p, use_batch_norm
        )
        self.decoder = BaseDecoder(
            n_layers,
            features_list[::-1],
            activation_func_name,
            dropout_p,
            use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x
