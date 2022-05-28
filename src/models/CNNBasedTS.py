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
        feature_dim: int = 1 
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

        # decode
        self.dec1_1 = CBR1d(in_channels=128, out_channels=256)
        self.dec1_2 = CBR1d(in_channels=256, out_channels=256)
        self.upsample1 = nn.ConvTranspose1d(kernel_size=4)

        self.dec2_1 = CBR1d(in_channels=256, out_channels=128)
        self.dec2_2 = CBR1d(in_channels=128, out_channels=128)
        self.upsample2 = nn.ConvTranspose1d(kernel_size=4)

        self.dec3_1 = CBR1d(in_channels=128, out_channels=64)
        self.dec3_2 = CBR1d(in_channels=64, out_channels=64)
        self.upsample3 = nn.ConvTranspose1d(kernel_size=4)

        self.dec4_1 = CBR1d(in_channels=64, out_channels=32)
        self.dec4_2 = CBR1d(in_channels=32, out_channels=32)
        self.upsample4 = nn.ConvTranspose1d(kernel_size=4)

        self.dec5_1 = CBR1d(in_channels=32, out_channels=16)
        self.dec5_2 = CBR1d(in_channels=16, out_channels=16)
        
        self.last = nn.Conv1d(in_channels=16, out_channels=feature_dim, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encode
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc1_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        # decode
        dec1_1 = self.dec1_1(pool4)
        dec1_2 = self.dec1_2(dec1_1)
        upsample1 = self.upsample1(dec1_2)

        concat1 = torch.concat((enc4_2, upsample1), dim=1) # dim = 1 : channel dim에 합치는 것        
        dec2_1 = self.dec2_1(concat1)
        dec2_2 = self.dec2_2(dec2_1)
        upsample2 = self.upsample2(dec2_2)
        
        concat2 = torch.concat((enc3_2, upsample2), dim=1)
        dec3_1 = self.dec3_1(concat2)
        dec3_2 = self.dec3_2(dec3_1)
        upsample3 = self.upsample3(dec3_2)

        concat3 = torch.concat((enc2_2, upsample3), dim=1)
        dec4_1 = self.dec4_1(concat3)
        dec4_2 = self.dec4_2(dec4_1)
        upsample4 = self.upsample4(dec4_2)

        dec5_1 = self.dec5_1(upsample4)
        dec5_2 = self.dec5_2(dec5_1)
        
        last = self.last(dec5_2)
        output = self.sigmoid(last)
        
        return output