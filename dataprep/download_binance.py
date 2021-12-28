import datetime
import os
import pickle
import sys
import time
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

symbols = [
    "DOGEUSDT",
    "AVAXUSDT",
    "SOLUSDT",
    "SHIBUSDT",
    "EURUSDT",
    "GBPUSDT",
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
import urllib

PATH = "../data/bin/daily/"
# for s in symbols:
#     for year in range(2018, 2022, 1):
#         for month in range(1, 13, 1):
#             file = f"{s}-1m-{year}-{month:02d}.zip"
#             fullfilename = os.path.join(PATH, file)
#             try:
#                 urllib.request.urlretrieve(
#                     f"https://data.binance.vision/data/spot/monthly/klines/{s}/1m/{file}",
#                     fullfilename,
#                 )
#             except urllib.error.HTTPError as exception:
#                 print(f"not found {file}")
for s in symbols:
    for day in range(19, 28, 1):
        file = f"{s}-1m-2021-12-{day:02d}.zip"
        fullfilename = os.path.join(PATH, file)
        try:
            urllib.request.urlretrieve(
                f"https://data.binance.vision/data/spot/daily/klines/{s}/1m/{file}",
                fullfilename,
            )
        except urllib.error.HTTPError as exception:
            print(f"not found {file}")
