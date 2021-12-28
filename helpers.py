import pickle

import numpy as np
import pandas as pd


def extract_windows_vectorized(clearing_time_index, max_time, sub_window_size):
    start = clearing_time_index + 1 - sub_window_size + 1

    sub_windows = (
        start
        +
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(sub_window_size), 0)
        + np.expand_dims(np.arange(max_time + 1), 0).T
    )

    return sub_windows


def prepare_data(config, data):
    features = config["models"]["resnet"]["features"]
    symbol = config["models"]["resnet"]["symbols"][0]

    with open("ml/models/stats.pkl", "rb") as file:
        stats = pickle.load(file)

    final_data = pd.DataFrame(data[symbol].ticks)[features]
    final_data = final_data.astype(np.float32)
    final_data = (final_data - stats[symbol]["mean"]) / stats[symbol]["std"]
    final_data.columns = [x + "_" + symbol for x in final_data.columns]

    for symbol in config["models"]["resnet"]["symbols"][1:]:
        symbol_data = pd.DataFrame(data[symbol].ticks)[features]
        symbol_data = symbol_data.astype(np.float32)
        symbol_data = (symbol_data - stats[symbol]["mean"]) / stats[symbol]["std"]
        symbol_data.columns = [x + "_" + symbol for x in symbol_data.columns]
        final_data = final_data.join(symbol_data)

    x = final_data.values.swapaxes(0, 1).astype(np.float32)

    idx = extract_windows_vectorized(512, 1000 - 512 - 17, 512)

    data_for_lr = final_data.values[idx].swapaxes(1, 2).astype(np.float32)
    column_names = ["close_" + x for x in config["models"]["resnet"]["symbols"]]

    prev_price = final_data[column_names].iloc[idx[:, -1]]
    cur_price = final_data[column_names].iloc[idx[:, -1] + 15]

    returns = (cur_price.values / prev_price.values) - 1

    return x[:, -512:], data_for_lr, returns
