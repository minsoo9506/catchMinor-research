import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.dataload.tabularData import (split_tabular_normal_only_train,
                                      tabularDataset)
from src.lit_models.LitBaseAutoEncoder import LitBaseAutoEncoder
from src.lit_models.LitBaseVAE import LitBaseVAE


def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", default="Tabular Anomaly Detection")
    parser.add_argument("--model", default="LitBaseAutoEncoder")
    parser.add_argument(
        "--data_path",
        default="/Users/nhn/Workspace/catchMinor/data/tabular_data/abalone9-18.csv",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="input batch size for training (default: 512)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument("--cuda", type=int, default=0, help="0 for cpu -1 for all gpu")
    config = parser.parse_args()
    if config.cuda == 0 or torch.cuda.is_available() == False:
        config.cuda = 0

    return config


def main(config):
    # data
    df = pd.read_csv(config.data_path)
    normal_train, normal_val, normal_abnormal_test = split_tabular_normal_only_train(df)
    train_dataset = tabularDataset(
        np.array(normal_train.iloc[:, :-1]), np.array(normal_train.iloc[:, -1])
    )
    valid_dataset = tabularDataset(
        np.array(normal_val.iloc[:, :-1]), np.array(normal_val.iloc[:, -1])
    )
    test_dataset = tabularDataset(
        np.array(normal_abnormal_test.iloc[:, :-1]),
        np.array(normal_abnormal_test.iloc[:, -1]),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # model
    if config.model == "LitBaseAutoEncoder":
        model = LitBaseAutoEncoder(n_layers=2, features_list=[8, 4, 2])
    if config.model == "LitBaseVAE":
        model = LitBaseVAE()

    # trainer
    logger = pl.loggers.WandbLogger()
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=20
    )
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=10,  # set the logging frequency
        gpus=config.cuda,  # use all GPUs
        max_epochs=config.epochs,  # number of epochs
        deterministic=True,  # keep it deterministic
        callbacks=[early_stopping_callback],
    )

    # fit the model
    trainer.fit(model, train_loader, valid_loader)

    # error
    ############
    # # validate
    # trainer.validate(valid_loader)

    # # test
    # trainer.test(test_loader)

    # inference
    # result = trainer.predict(test_loader)
    # print(result.shape)
    #############


if __name__ == "__main__":
    config = define_argparser()
    main(config)
