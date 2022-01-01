import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Mapping

import pandas as pd

from datastructures.log_writer import LogWriter
from datastructures.operations import Operation
from datastructures.tickerstream import TickerStream

BUY_THR = 0.005
SELL_THR = -0.005


class TriggerType(Enum):
    NN = 1
    LR = 2
    LRx3 = 3
    LRx4 = 4
    Elastic = 5
    LRx4Manual = 6
    Elasticx10 = 7
    LRx4Manualx10 = 8
    Q_THR = 9
    Q_THR_HARD = 10
    Q_THR_REVERSED = 11
    FOCAL = 12
    FOCAL_5 = 13


@dataclass
class Trade:
    timestamp: int
    currency: str
    entry_price: float
    type: Operation
    trigger: TriggerType
    prediction: float
    threshold: float


@dataclass
class TradesMemory:
    trades: List[Trade] = field(default_factory=list)

    def add_trade(self, trade):
        self.trades.append(trade)


class Trading:
    def __init__(self, cfg, thresholds, logger):
        self.thresholds = thresholds
        self.cfg = cfg
        self.logger = logger
        self.delay = 15  # TODO DATETIME AND TIMEDELTA here
        self.active_trades = 0
        self.trades = TradesMemory()
        self.logger = LogWriter()

        # self.thresholds_m = pd.DataFrame(
        #     [
        #         {"pair": "DOGEUSDT", "buy_thr": 0.21, "sell_thr": 0.21},
        #         {"pair": "AVAXUSDT", "buy_thr": 0.38, "sell_thr": 0.38},
        #         {"pair": "SOLUSDT", "buy_thr": 0.30, "sell_thr": 0.31},
        #         {"pair": "SHIBUSDT", "buy_thr": 0.39, "sell_thr": 0.39},
        #         {"pair": "EURUSDT", "buy_thr": 1, "sell_thr": 1},
        #         {"pair": "GBPUSDT", "buy_thr": 1, "sell_thr": 1},
        #         {"pair": "ETCETH", "buy_thr": 1, "sell_thr": 1},
        #         {"pair": "ETCBTC", "buy_thr": 1, "sell_thr": 1},
        #         {"pair": "MKRUSDT", "buy_thr": 0.2, "sell_thr": 0.2},
        #         {"pair": "MKRBTC", "buy_thr": 1, "sell_thr": 1},
        #         {"pair": "IOTAUSDT", "buy_thr": 0.30, "sell_thr": 0.30},
        #         {"pair": "ADAUSDT", "buy_thr": 0.22, "sell_thr": 0.22},
        #         {"pair": "XLMUSDT", "buy_thr": 1, "sell_thr": 1},
        #         {"pair": "TRXUSDT", "buy_thr": 0.12, "sell_thr": 0.12},
        #         {"pair": "XMRUSDT", "buy_thr": 0.16, "sell_thr": 0.16},
        #         {"pair": "EOSUSDT", "buy_thr": 0.18, "sell_thr": 0.18},
        #         {"pair": "DOGEGBP", "buy_thr": 1, "sell_thr": 1},
        #         {"pair": "BTCEUR", "buy_thr": 1, "sell_thr": 1},
        #         {"pair": "BTCGBP", "buy_thr": 1, "sell_thr": 1},
        #         {"pair": "BTCUSDT", "buy_thr": 0.12, "sell_thr": 0.12},
        #     ]
        # ).set_index("pair")
        self.thresholds_m = pd.DataFrame(
            [
                {"pair": "DOGEUSDT", "buy_thr": 0.24, "sell_thr": 0.24},
                {"pair": "AVAXUSDT", "buy_thr": 0.30, "sell_thr": 0.30},
                {"pair": "SOLUSDT", "buy_thr": 0.33, "sell_thr": 0.33},
                {"pair": "SHIBUSDT", "buy_thr": 0.30, "sell_thr": 0.30},
                {"pair": "EURUSDT", "buy_thr": 1, "sell_thr": 1},
                {"pair": "GBPUSDT", "buy_thr": 1, "sell_thr": 1},
                {"pair": "ETCETH", "buy_thr": 1, "sell_thr": 1},
                {"pair": "ETCBTC", "buy_thr": 1, "sell_thr": 1},
                {"pair": "MKRUSDT", "buy_thr": 0.4, "sell_thr": 0.4},
                {"pair": "MKRBTC", "buy_thr": 1, "sell_thr": 1},
                {"pair": "IOTAUSDT", "buy_thr": 0.28, "sell_thr": 0.28},
                {"pair": "ADAUSDT", "buy_thr": 0.40, "sell_thr": 0.40},
                {"pair": "XLMUSDT", "buy_thr": 1, "sell_thr": 1},
                {"pair": "TRXUSDT", "buy_thr": 1, "sell_thr": 1},
                {"pair": "XMRUSDT", "buy_thr": 0.4, "sell_thr": 0.4},
                {"pair": "EOSUSDT", "buy_thr": 0.4, "sell_thr": 0.4},
                {"pair": "DOGEGBP", "buy_thr": 1, "sell_thr": 1},
                {"pair": "BTCEUR", "buy_thr": 1, "sell_thr": 1},
                {"pair": "BTCGBP", "buy_thr": 1, "sell_thr": 1},
                {"pair": "BTCUSDT", "buy_thr": 0.25, "sell_thr": 0.25},
            ]
        ).set_index("pair")

        self.thresholds = pd.DataFrame(
            [
                {"pair": "DOGEUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "AVAXUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "SOLUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "SHIBUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "EURUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "GBPUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "ETCETH", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "ETCBTC", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "MKRUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "MKRBTC", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "IOTAUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "ADAUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "XLMUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "TRXUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "XMRUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "EOSUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "DOGEGBP", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "BTCEUR", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "BTCGBP", "buy_thr": 0.5, "sell_thr": 0.5},
                {"pair": "BTCUSDT", "buy_thr": 0.5, "sell_thr": 0.5},
            ]
        ).set_index("pair")

    def make_trade(self, trade: Trade, timestamp):
        self.trades.add_trade(trade)
        self.logger.log_trade_open(trade, timestamp)
        self.active_trades += 1

    def __len__(self):
        return len(self.trades.trades)

    def check_trade_for_closing(self, trade: Trade, timestamp, prices: TickerStream):
        if (
            datetime.fromtimestamp(prices.max_timestamp() / 1000)
            - datetime.fromtimestamp(trade.timestamp / 1000)
        ).total_seconds() > 60 * 15:
            self.logger.log_trade_close(
                timestamp,
                trade,
                prices.latest_close_price(),
            )

            self.active_trades -= 1

            return False

        return True

    def close_trades(self, timestamp, spot_prices: Mapping[str, TickerStream]):
        self.trades.trades = [
            trade
            for trade in self.trades.trades
            if self.check_trade_for_closing(
                trade, timestamp, spot_prices[trade.currency]
            )
        ]

    def update_status(
        self,
        max_timestamp,
        spot_prices: Mapping[str, TickerStream],
        predictions_up,
        predictions_down,
    ):
        # log per-currency and total profits
        # check for double trades!

        for thr_type, thrs in zip(
            [
                TriggerType.FOCAL,
                TriggerType.FOCAL_5,
            ],
            [
                self.thresholds_m,
                self.thresholds,
            ],
        ):

            for c, currency in enumerate(self.cfg["models"]["resnet"]["symbols"]):
                current_price = spot_prices[currency].latest_close_price()
                if currency not in thrs.index.values:
                    continue

                thr = thrs.loc[currency]

                buy_thr = thr.buy_thr
                sell_thr = thr.sell_thr

                self.logger.log_currency_tick_data(
                    ts=max_timestamp,
                    currency=currency,
                    spot_price=spot_prices[currency].latest_close_price(),
                    prediction_up=predictions_up[0, c],
                    prediction_down=predictions_down[0, c],
                    buy_thr=buy_thr,
                    sell_thr=sell_thr,
                    trigger_type=thr_type,
                )

                if predictions_up[0, c] > buy_thr:
                    trade = Trade(
                        timestamp=max_timestamp,
                        currency=currency,
                        entry_price=current_price,
                        type=Operation.BUY,
                        trigger=thr_type,
                        prediction=predictions_up[0, c],
                        threshold=buy_thr,
                    )
                    self.make_trade(trade, max_timestamp)

                if predictions_down[0, c] > sell_thr:
                    trade = Trade(
                        timestamp=max_timestamp,
                        currency=currency,
                        entry_price=current_price,
                        type=Operation.SELL,
                        trigger=thr_type,
                        prediction=predictions_down[0, c],
                        threshold=sell_thr,
                    )
                    self.make_trade(trade, max_timestamp)

        self.close_trades(max_timestamp, spot_prices)
