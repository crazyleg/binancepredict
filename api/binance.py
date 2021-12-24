import asyncio
import logging
import time
from collections import ChainMap
from datetime import datetime
from typing import Mapping

import aiohttp
import numpy as np
import pandas as pd
import requests
from datastructures import TickerData, TickerStream


class BinanceAPI:
    def __init__(self, config):
        super(BinanceAPI, self).__init__()
        logging.info("Init Binance API")
        self.endpoint = config["endpoint"]

    def wait_for_1m_tick(self) -> datetime:
        response = requests.get(self.endpoint + "/api/v3/time")
        server_time = response.json()["serverTime"]
        sleep_for = 60 - datetime.fromtimestamp(server_time / 1000).second + 20
        logging.info(f"going to sleep for {sleep_for}")
        time.sleep(sleep_for)
        response = requests.get(self.endpoint + "/api/v3/time")
        server_time = response.json()["serverTime"]
        logging.info(
            f"getting data with delay of {datetime.fromtimestamp(server_time / 1000).second}"
        )

        return datetime.fromtimestamp(server_time / 1000)

    async def get_ticker_data_async(
        self, session, symbol: str, interval: str = "1m", limit: int = 512
    ) -> TickerStream:
        parameters = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        # TODO BLOCKER - monitor for 429 and other errors
        async with session.get(
            self.endpoint + "/api/v3/klines", params=parameters
        ) as response:
            tickers = [TickerData(symbol, interval, *x) for x in await response.json()]
            stream = TickerStream(tickers, symbol, interval)
            last_timestamp = stream.max_timestamp()

        logging.debug(
            f"get {len(tickers)} ticks of {symbol} with delay of {time.time()-last_timestamp/1000}"
        )
        return {symbol: stream}, last_timestamp

    async def start_async_fetching(self, symbols, interval, limit=1000):
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *[
                    self.get_ticker_data_async(session, s, interval, limit)
                    for s in symbols
                ]
            )
        results_data = dict(ChainMap(*[r for r, t in results]))
        return results_data, np.array([t for r, t in results]).max()

    def get_interence_data_async(
        self, symbols, interval, limit=1000
    ) -> Mapping[str, TickerStream]:
        # TODO make a specific dataclass
        # also push binance calls there and ensure same timestamp
        results, max_timestamp = asyncio.run(
            self.start_async_fetching(symbols, interval, limit)
        )

        return results, max_timestamp
