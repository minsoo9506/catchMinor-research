# https://arxiv.org/pdf/1905.13628.pdf
# cnn based time series segmentaion (U-net)

from typing import Any, List

import torch
import torch.nn as nn

# Conv + batch norma + ReLu
# |conv1d input| = (batch_size, feature_dim, seq_len)
# feature_dim -> channels을 의미

def CBR1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int =3,
    stride: int =1,
    padding: int =1,
    ) -> nn.Sequential:
    """conv1d + batch norm + ReLu layer

    Parameters
    ----------
    in_channels : int
        can think as feature dimension in conv1d
    out_channels : int
        _description_
    kernel_size : int, optional
        _description_, by default 3
    stride : int, optional
        _description_, by default 1
    padding : int, optional
        _description_, by default 1

    Returns
    -------
    nn.Sequential
        _description_
    """
    layers: List[Any] = []
    layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
    layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.ReLU())

    cbr1d = nn.Sequential(*layers)
    
    return cbr1d


class TSUNet(nn.Module):
    def __init__(
        self,
        feature_dim: int
        ):
        super().__init__()

        # encode
        self.enc1_1 = CBR1d(in_channels=feature_dim, out_channels=16)
        self.enc1_2 = CBR1d(in_channels=16, out_channels=16)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.enc2_1 = CBR1d(in_channels=16, out_channels=32)
        self.enc2_2 = CBR1d(in_channels=32, out_channels=32)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.enc3_1 = CBR1d(in_channels=32, out_channels=64)
        self.enc3_2 = CBR1d(in_channels=64, out_channels=64)
        self.pool3 = nn.MaxPool1d(kernel_size=4)

        self.enc4_1 = CBR1d(in_channels=64, out_channels=128)
        self.enc4_2 = CBR1d(in_channels=128, out_channels=128)
        self.pool4 = nn.MaxPool1d(kernel_size=4)

        self.enc5_1 = CBR1d(in_channels=128, out_channels=256)
        self.enc5_2 = CBR1d(in_channels=256, out_channels=256)
        self.pool5 = nn.MaxPool1d(kernel_size=4)

        # decode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass