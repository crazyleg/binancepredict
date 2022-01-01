import math
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

NEEDED_MOVE_UP = 1.006
NEEDED_MOVE_DOWN = 1 / NEEDED_MOVE_UP

SYMBOLS = set(
    [
        "DOGEUSDT",
        "AVAXUSDT",
        "SOLUSDT",
        "SHIBUSDT",
        "EURUSDT",
        "GBPUSDT",
        "BGBPUSDC",
        "ETCETH",
        "ETCBTC",
        "MKRUSDT",
        "MKRBTC",
        "IOTAUSDT",
        "ADAUSDT",
        "XLMUSDT",
        "TRXUSDT",
        "XMRUSDT",
        "EOSUSDT",
        "DOGEGBP",
        "BTCEUR",
        "BTCGBP",
        "BTCUSDT",
    ]
)

FEATURES = [
    "open_timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_timestamp",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

NEEDED_COLUMNS = [
    "open_timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]

TYPES = {
    "open_timestamp": np.uint64,
    "open": np.float32,
    "high": np.float32,
    "low": np.float32,
    "close": np.float32,
    "volume": np.float32,
    "close_timestamp": np.uint64,
    "quote_asset_volume": np.float32,
    "number_of_trades": np.float32,
    "taker_buy_base_asset_volume": np.float32,
    "taker_buy_quote_asset_volume": np.float32,
    "ignore": np.uint8,
}

PATH = "data/bin/"
PATH_DAILY = "data/bin/daily/"
min_time = 1577836800000
max_time = 1640606400000


class BinanceCoinDataset(Dataset):
    def __init__(
        self,
        test=(False, 0.7),
        window_size=1024,
        y_time_forward=15,
        transform=None,
        target_transform=None,
        currencies=["BTCUSDT", "ADAUSDT", "XMRUSDT"],
    ):
        self.y_time_forward = y_time_forward
        self.window_size = window_size
        self.currencies = currencies

        results = {}
        results_y = {}
        stats = {}
        for s in currencies:
            currency = []
            for year in range(2020, 2022, 1):
                for month in range(1, 13, 1):
                    try:
                        file = f"{s}-1m-{year}-{month:02d}.zip"
                        data = pd.read_csv(
                            PATH + f"{s}-1m-{year}-{month:02d}.zip",
                            names=FEATURES,
                            dtype=TYPES,
                        )
                        data = data[NEEDED_COLUMNS]
                        currency.append(data)
                    except FileNotFoundError as error:
                        pass

            for day in range(1, 31, 1):
                try:
                    file = f"{s}-1m-2021-12-{day:02d}.zip"
                    data = pd.read_csv(
                        PATH_DAILY + file,
                        names=FEATURES,
                        dtype=TYPES,
                    )
                    data = data[NEEDED_COLUMNS]
                    currency.append(data)
                except FileNotFoundError as error:
                    pass

            currency = pd.concat(currency).set_index(["open_timestamp"])
            ys = currency[["close"]]
            ys.columns = ["close_" + s]

            ys = ys.reindex(range(min_time, max_time, 1000 * 60), fill_value=-1)
            results_y[s] = ys

            stats[s] = {"mean": currency.mean(), "std": currency.std()}
            currency = (currency - currency.mean()) / currency.std()
            currency = currency.reindex(
                range(min_time, max_time, 1000 * 60), fill_value=-1
            )

            currency.columns = [c + "_" + s for c in currency.columns]
            data["pair"] = s

            results[s] = currency

        with open("stats.pkl", "wb") as file:
            pickle.dump(stats, file)

        self.data = results[currencies[0]]
        self.data_y = results_y[currencies[0]]
        for i in range(1, len(currencies), 1):
            self.data = self.data.join(results[currencies[i]])
            self.data_y = self.data_y.join(results_y[currencies[i]])

        self.y_target_columns = ["close_" + c for c in self.currencies]

        self.data = (
            self.data.iloc[: int(self.data.shape[0] * test[1])]
            if not test[0]
            else self.data.iloc[int(self.data.shape[0] * test[1]) :]
        )

        self.data_y = (
            self.data_y.iloc[: int(self.data_y.shape[0] * test[1])]
            if not test[0]
            else self.data_y.iloc[int(self.data_y.shape[0] * test[1]) :]
        )

    def get_size_of_features(self):
        return self.data.shape[1]

    def __len__(self):
        return self.data.shape[0] - self.window_size - self.y_time_forward

    def __getitem__(self, idx):
        sample = self.data.iloc[idx : idx + self.window_size]

        future_prices = self.data_y.iloc[
            idx + self.window_size : idx + self.window_size + self.y_time_forward
        ][self.y_target_columns]

        # Thats not cool cause actually it would be better to use HIGH and LOW
        # prices on a tick-level data. But I don't have those, so let's predict
        # a more general case.

        # So here we just are looking for a fast of having any closing price above or below threshold

        higher_bond = (
            self.data_y.iloc[idx + self.window_size - 1][self.y_target_columns]
            * NEEDED_MOVE_UP
        )

        lower_bond = (
            self.data_y.iloc[idx + self.window_size - 1][self.y_target_columns]
            * NEEDED_MOVE_DOWN
        )

        current_prices_are_valid = (
            (
                self.data_y.iloc[
                    idx
                    + self.window_size : idx
                    + self.window_size
                    + self.y_time_forward
                ][self.y_target_columns]
                != -1
            )
            .any()
            .values
        )

        up_markers = (future_prices > higher_bond).any().values.astype(np.float32)
        down_markers = (future_prices < lower_bond).any().values.astype(np.float32)

        up_markers = np.where(current_prices_are_valid, up_markers, -1)
        down_markers = np.where(current_prices_are_valid, down_markers, -1)

        x = sample.values.swapaxes(0, 1).astype(np.float32)

        return x, up_markers, down_markers
