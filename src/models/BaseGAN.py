from typing import Any, Dict, List

import torch
import torch.nn as nn


class BaseGenerator(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        features_list: List[int] = [2, 4, 8, 16],
        activation_func_name: str = "ReLU",
        dropout_p: float = 0.2,
        use_batch_norm: bool = False,
    ):
        """generator

        Parameters
        ----------
        n_layers : int, optional
            _description_, by default 3
        features_list : List[int], optional
            feature_list[0] should be latent_dim,
            feature_list[-1] should be original data dim,
            by default [2, 4, 8, 16]
        activation_func_name : str, optional
            _description_, by default "ReLU"
        dropout_p : float, optional
            _description_, by default 0.2
        use_batch_norm : bool, optional
            _description_, by default False
        """
        super().__init__()

        assert n_layers + 1 == len(
            features_list
        ), "should be: n_layers + 1 == len(features_list)"
        self.latent_dim = features_list[0]
        self.n_layers = n_layers
        self.output_size = features_list[-1]
        activation_func = getattr(nn, activation_func_name, nn.ReLU)

        layers: List[Any] = []
        for in_features, out_features in zip(features_list[:-2], features_list[1:-1]):
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

        layers.append(
            nn.Linear(
                in_features=features_list[-2],
                out_features=features_list[-1],
            ),
            nn.Tanh(),
        )
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        generated_x = self.model(z)
        return generated_x


class BaseDiscriminator(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        features_list: List[int] = [16, 8, 4, 1],
        activation_func_name: str = "ReLU",
        dropout_p: float = 0.2,
        use_batch_norm: bool = False,
    ):
        """disciminator

        Parameters
        ----------
        n_layers : int, optional
            _description_, by default 3
        features_list : List[int], optional
            feature_list[0] should be original data dim,
            feature_list[-1] should be 1,
            by default [16, 8, 4, 1]
        activation_func_name : str, optional
            _description_, by default "ReLU"
        dropout_p : float, optional
            _description_, by default 0.2
        use_batch_norm : bool, optional
            _description_, by default False
        """
        super().__init__()

        assert n_layers + 1 == len(
            features_list
        ), "should be: n_layers + 1 == len(features_list)"
        assert features_list[-1] == 1, "should be: feature_list[-1] == 1"
        self.n_layers = n_layers
        activation_func = getattr(nn, activation_func_name, nn.ReLU)

        layers: List[Any] = []
        for in_features, out_features in zip(features_list[:-2], features_list[1:-1]):
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

        layers.append(
            nn.Linear(
                in_features=features_list[-2],
                out_features=features_list[-1],
            ),
            nn.Sigmoid(),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        true_or_fake = self.model(x)
        return true_or_fake
