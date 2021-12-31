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

        self.lr_thresholds = pd.read_pickle("ml/models/lr_thresholds.pkl")
        self.C_thresholds = pd.read_pickle("ml/models/C_thresholds.pkl")

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
        predictions_nn,
        predictions_lr,
        predictions_el,
        q_thrs,
        q_thrs_hard,
    ):
        # log per-currency and total profits
        # check for double trades!

        self.just_thresholds = pd.DataFrame(
            [
                {"pair": c, "buy_thr": 0.002, "sell_thr": -0.002}
                for c in self.cfg["models"]["resnet"]["symbols"]
            ]
        )

        self.just_thresholds4 = pd.DataFrame(
            [
                {"pair": c, "buy_thr": 0.004, "sell_thr": -0.004}
                for c in self.cfg["models"]["resnet"]["symbols"]
            ]
        )

        self.just_thresholds10 = pd.DataFrame(
            [
                {"pair": c, "buy_thr": 0.01, "sell_thr": -0.01}
                for c in self.cfg["models"]["resnet"]["symbols"]
            ]
        )

        for thr_type, thrs in zip(
            [
                TriggerType.NN,
                TriggerType.LR,
                TriggerType.LRx3,
                TriggerType.LRx4,
                TriggerType.Elastic,
                TriggerType.LRx4Manual,
                TriggerType.Elasticx10,
                TriggerType.LRx4Manualx10,
                TriggerType.Q_THR,
                TriggerType.Q_THR_HARD,
            ],
            [
                self.C_thresholds,
                self.lr_thresholds,
                self.lr_thresholds,
                self.lr_thresholds,
                self.just_thresholds,
                self.just_thresholds4,
                self.just_thresholds10,
                self.just_thresholds10,
                q_thrs,
                q_thrs_hard,
            ],
        ):
            tmp_thrs = thrs.copy()

            if thr_type in [
                TriggerType.LRx3,
                TriggerType.LR,
                TriggerType.LRx4,
                TriggerType.LRx4Manual,
                TriggerType.LRx4Manualx10,
            ]:
                predictions = predictions_lr.copy()

            elif thr_type in [TriggerType.Elastic, TriggerType.Elasticx10]:
                predictions = predictions_el.copy()

            else:
                predictions = predictions_nn.numpy().copy()

            if thr_type == TriggerType.LRx3:
                tmp_thrs.buy_thr *= 3
                tmp_thrs.sell_thr *= 3

            if thr_type == TriggerType.LRx4:
                tmp_thrs.buy_thr *= 4
                tmp_thrs.sell_thr *= 4

            for c, currency in enumerate(self.cfg["models"]["resnet"]["symbols"]):
                current_price = spot_prices[currency].latest_close_price()
                if currency not in tmp_thrs.pair.values:
                    continue
                thr = tmp_thrs.set_index("pair").loc[currency]

                buy_thr = thr.buy_thr
                sell_thr = thr.sell_thr

                self.logger.log_currency_tick_data(
                    ts=max_timestamp,
                    currency=currency,
                    spot_price=spot_prices[currency].latest_close_price(),
                    prediction=predictions[0, c],
                    buy_thr=buy_thr,
                    sell_thr=sell_thr,
                    trigger_type=thr_type,
                )

                if predictions[0, c] > buy_thr:
                    trade = Trade(
                        timestamp=max_timestamp,
                        currency=currency,
                        entry_price=current_price,
                        type=Operation.BUY,
                        trigger=thr_type,
                        prediction=predictions[0, c],
                        threshold=buy_thr,
                    )
                    self.make_trade(trade, max_timestamp)

                if predictions[0, c] < sell_thr:
                    trade = Trade(
                        timestamp=max_timestamp,
                        currency=currency,
                        entry_price=current_price,
                        type=Operation.SELL,
                        trigger=thr_type,
                        prediction=predictions[0, c],
                        threshold=sell_thr,
                    )
                    self.make_trade(trade, max_timestamp)

        self.close_trades(max_timestamp, spot_prices)
