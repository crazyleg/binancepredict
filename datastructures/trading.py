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
    LR_P_3PCT = 3
    LRx2 = 4


@dataclass
class Trade:
    timestamp: int
    currency: str
    entry_price: float
    type: Operation
    trigger: TriggerType


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

    def make_trade(self, trade: Trade):
        self.trades.add_trade(trade)
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
        self, max_timestamp, spot_prices: Mapping[str, TickerStream], predictions
    ):
        # log per-currency and total profits
        # check for double trades!

        for thr_type, thrs in zip(
            [TriggerType.NN, TriggerType.LR, TriggerType.LR_P_3PCT, TriggerType.LRx2],
            [
                self.C_thresholds,
                self.lr_thresholds,
                self.lr_thresholds,
                self.lr_thresholds,
            ],
        ):
            tmp_thrs = thrs.copy()
            if thr_type == TriggerType.LR_P_3PCT:
                tmp_thrs.buy_thr *= 1.3
                tmp_thrs.sell_thr *= 1.3

            if thr_type == TriggerType.LRx2:
                tmp_thrs.buy_thr *= 2
                tmp_thrs.sell_thr *= 2

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
                )

                if predictions[0, c] > buy_thr:
                    trade = Trade(
                        timestamp=max_timestamp,
                        currency=currency,
                        entry_price=current_price,
                        type=Operation.BUY,
                        trigger=thr_type,
                    )
                    kibana_extra_data = {
                        "type": "open_trade",
                        "currency": trade.currency,
                        "open_price": trade.entry_price,
                        "trade_type": trade.type,
                        "trade_trigger": trade.trigger,
                    }
                    self.make_trade(trade)
                    logging.info(
                        f"opening the trade on ts: {max_timestamp} currency: {currency} entry_price: {trade.entry_price}  type: {Operation.BUY} type: {thr_type} as prediction was {predictions[0, c]} and thr {buy_thr}",
                        extra=kibana_extra_data,
                    )
                if predictions[0, c] < sell_thr:
                    trade = Trade(
                        timestamp=max_timestamp,
                        currency=currency,
                        entry_price=current_price,
                        type=Operation.SELL,
                        trigger=thr_type,
                    )
                    kibana_extra_data = {
                        "type": "open_trade",
                        "currency": trade.currency,
                        "open_price": trade.entry_price,
                        "trade_type": trade.type,
                        "trade_trigger": trade.trigger,
                    }
                    self.make_trade(trade)
                    logging.info(
                        f"opening the trade on ts: {max_timestamp} currency: {currency} entry_price: {trade.entry_price}  type: {Operation.SELL} type: {thr_type} as prediction was {predictions[0, c]} and thr {sell_thr}",
                        extra=kibana_extra_data,
                    )

        self.close_trades(max_timestamp, spot_prices)
