from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def split_tabular_normal_only_train(
    df: pd.DataFrame,
    y_label: str = "label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.7,
    shuffle: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """make dataset: train(normal), valid(normal), test(normal, abnormal)

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    y_label : str, optional
        _description_, by default 'label'
    train_ratio : float, optional
        _description_, by default 0.7
    val_ratio : float, optional
        ratio between normal in valid and normal in test, by default 0.7
    shuffle : bool, optional
        shuffle when split the dataset, by default False

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        normal_train, normal_val, normal_abnormal_test
    """
    normal = df.loc[df[y_label] == 0, :].reset_index(drop=True)
    abnormal_test = df.loc[df[y_label] == 1, :].reset_index(drop=True)
    normal_train, normal_val = train_test_split(
        normal, train_size=train_ratio, shuffle=shuffle
    )
    normal_val, normal_test = train_test_split(
        normal_val, train_size=val_ratio, shuffle=shuffle
    )
    normal_val = normal_val.reset_index(drop=True)
    normal_abnormal_test = pd.concat([normal_test, abnormal_test], axis=0).reset_index(
        drop=True
    )
    return normal_train, normal_val, normal_abnormal_test


class tabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """make Dataset (type cast to torch.float32)

        Parameters
        ----------
        x : np.ndarray
            _description_
        y : np.ndarray
            _description_
        """
        super().__init__()

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx, :], self.y[idx]
