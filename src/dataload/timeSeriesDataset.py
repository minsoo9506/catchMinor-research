from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class UnivariateBaseWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, window_size: int):
        """make window-based data for fully-connected layer
        Parameters
        ----------
        x : np.ndarray
            input data
        y : np.ndarray
            output data
        window_size : int
            window size
        """

        super().__init__()

        data_len = len(x) - window_size + 1
        self.x = np.zeros((data_len, window_size))
        self.y = np.zeros((data_len, window_size))

        for idx in range(data_len):
            self.x[idx, :] = x[idx : idx + window_size]
            self.y[idx, :] = y[idx : idx + window_size]

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self) -> int:
        """_summary_

        Returns
        -------
        int
            length of dataset
        """
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """_summary_

        Parameters
        ----------
        idx : int
            _description_

        Returns
        -------
        Tuple[torch.tensor, torch.tensor]
            _description_
        """
        return self.x[idx, :], self.y[idx, :]


# LSTM input = (N,L,H) when batch_first=True
# = (batch_size, seq_length, hidden_size)
class UnivariateLSTMWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, window_size: int):
        super().__init__()

        data_len = len(x) - window_size + 1
        self.x = np.zeros((data_len, window_size))
        self.y = np.zeros((data_len, window_size))

        for idx in range(data_len):
            self.x[idx, :] = x[idx : idx + window_size]
            self.y[idx, :] = y[idx : idx + window_size]

        # add axis (hidden_size)
        self.x = self.x[:, :, np.newaxis]
        self.y = self.y[:, :, np.newaxis]

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self) -> int:
        """_summary_

        Returns
        -------
        int
            length of dataset
        """
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """_summary_

        Parameters
        ----------
        idx : int
            _description_

        Returns
        -------
        Tuple[torch.tensor, torch.tensor]
            _description_
        """
        return self.x[idx, :], self.y[idx, :]


if __name__ == "__main__":
    # univariate time series data -> window-based approach
    dataset = UnivariateLSTMWindowDataset(
        x=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        y=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        window_size=3,
    )
    data_loader = DataLoader(dataset, batch_size=2)
    for x, _ in data_loader:
        print(x.shape)
        print(x)
        break
