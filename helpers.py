import pickle

import numpy as np
import pandas as pd


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

    return x
