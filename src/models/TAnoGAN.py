from typing import Any, Dict, List

import torch
import torch.nn as nn


class LSTMGenerator(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        input_size: int = 1,
        hidden_size_list: List[int] = [2, 4, 8],
        output_size: int = 8,
        num_layers: int = 1,
        dropout_p: float = 0.2,
        batch_first: bool = True,
        bidirectional: bool = False,
        seq_len: int = 60,
    ):
        """TAnoGAN Generator with LSTM layers

        Parameters
        ----------
        n_layers : int, optional
            num of LSTM layers, by default 3
        input_size : int, optional
            size of latent_z, by default 1
        hidden_size_list : List[int], optional
            list of LSTM's hidden_size, by default [2, 4, 8]
        output_size : int, optional
            output_size of last layers in nn.Linear, by default 8
        num_layers : int, optional
            num_layers hyperparameter in nn.LSTM, by default 1
        dropout_p : float, optional
            dropout hyperparameter in nn.LSTM, by default 0.2
        batch_first : bool, optional
            batch_first hyperparameter in nn.LSTM, by default True
        bidirectional : bool, optional
            bidirectional hyperparameter in nn.LSTM, by default False
        seq_len : int, optional
            sequence length of time series data, by default 60
        """
        super().__init__()

        assert n_layers == len(
            hidden_size_list
        ), "should be: n_layers == len(features_list)"
        self.input_size = input_size
        self.n_layers = n_layers
        self.output_size = output_size

        self.lstm: Dict[Any] = {}
        for i in range(n_layers):
            if i == 0:
                self.lstm[i] = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size_list[i],
                    num_layers=num_layers,
                    batch_first=batch_first,
                    dropout=dropout_p,
                    bidirectional=bidirectional,
                )
            else:
                self.lstm[i] = nn.LSTM(
                    input_size=hidden_size_list[i - 1],
                    hidden_size=hidden_size_list[i],
                    num_layers=num_layers,
                    batch_first=batch_first,
                    dropout=dropout_p,
                    bidirectional=bidirectional,
                )

        self.linear = nn.Sequential(
            nn.Linear(
                in_features=seq_len * hidden_size_list[-1],
                out_features=seq_len * output_size,
            ),
            nn.Tanh(),
        )

    def forward(self, input):
        recurrent_features, _ = self.lstm[0](input)
        for i in range(1, self.n_layers):
            recurrent_features, _ = self.lstm[i](recurrent_features)
        # |recurrent_features| = (batch_size, seq_len, hidden_size * bidirectional)
        batch_size, seq_len, h = (
            recurrent_features.size(0),
            recurrent_features.size(1),
            recurrent_features.size(3),
        )
        outputs = self.linear(
            recurrent_features.contiguous().view(batch_size, seq_len * h)
        )
        # |outputs| = (batch_size, seq_len * output_size)
        outputs = outputs.view(
            batch_size, seq_len, -1
        )  # : want to be same with the original data
        return outputs, recurrent_features


class LSTMDiscriminator(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        input_size: int = 8,
        hidden_size_list: List[int] = [8, 4, 2],
        output_size: int = 1,
        num_layers: int = 1,
        dropout_p: float = 0.2,
        batch_first: bool = True,
        bidirectional: bool = False,
        seq_len: int = 60,
    ):
        """TAnoGAN Discriminator with LSTM layers

        Parameters
        ----------
        n_layers : int, optional
            num of LSTM layers, by default 3
        input_size : int, optional
            size of latent_z, by default 1
        hidden_size_list : List[int], optional
            list of LSTM's hidden_size, by default [2, 4, 8]
        output_size : int, optional
            output_size of last layers in nn.Linear, by default 8
        num_layers : int, optional
            num_layers hyperparameter in nn.LSTM, by default 1
        dropout_p : float, optional
            dropout hyperparameter in nn.LSTM, by default 0.2
        batch_first : bool, optional
            batch_first hyperparameter in nn.LSTM, by default True
        bidirectional : bool, optional
            bidirectional hyperparameter in nn.LSTM, by default False
        seq_len : int, optional
            sequence length of time series data, by default 60
        """
        super().__init__()

        assert n_layers == len(
            hidden_size_list
        ), "should be: n_layers == len(features_list)"

        assert output_size == 1, "Discriminator's output_size should be 1"

        self.n_layers = n_layers
        self.output_size = output_size

        self.lstm: Dict[Any] = {}
        for i in range(n_layers):
            if i == 0:
                self.lstm[i] = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size_list[i],
                    num_layers=num_layers,
                    batch_first=batch_first,
                    dropout=dropout_p,
                    bidirectional=bidirectional,
                )
            else:
                self.lstm[i] = nn.LSTM(
                    input_size=hidden_size_list[i - 1],
                    hidden_size=hidden_size_list[i],
                    num_layers=num_layers,
                    batch_first=batch_first,
                    dropout=dropout_p,
                    bidirectional=bidirectional,
                )

        self.linear = nn.Sequential(
            nn.Linear(
                in_features=seq_len * hidden_size_list[-1], out_features=output_size
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        recurrent_features, _ = self.lstm[0](input)
        for i in range(1, self.n_layers):
            recurrent_features, _ = self.lstm[i](recurrent_features)
        # |recurrent_features| = (batch_size, seq_len, hidden_size * bidirectional)
        batch_size, seq_len, h = (
            recurrent_features.size(0),
            recurrent_features.size(1),
            recurrent_features.size(3),
        )
        outputs = self.linear(
            recurrent_features.contiguous().view(batch_size, seq_len * h)
        )
        # |outputs| = (batch_size, output_size=1)
        return outputs, recurrent_features
