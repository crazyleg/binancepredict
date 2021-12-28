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
    AVG = 1
    SUM = 2
    MANUAL = 3
    LR = 4
    LR_AVG = 5
    LR_SUM = 6


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

        self.thresholds_avg = pd.read_pickle("thresholds_avg.pkl")
        self.thresholds_sum = pd.read_pickle("thresholds_sum.pkl")

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

        for c, currency in enumerate(self.cfg["models"]["resnet"]["symbols"]):
            current_price = spot_prices[currency].latest_close_price()
            thr = self.thresholds.set_index("pair").loc[currency]

            self.logger.log_currency_tick_data(
                ts=max_timestamp,
                currency=currency,
                spot_price=spot_prices[currency].latest_close_price(),
                prediction=predictions[0, c],
                buy_thr=thr.buy_thr,
                sell_thr=thr.sell_thr,
            )

            avg_buy_thr = self.thresholds_avg.set_index("pair").loc[currency].buy_thr
            avg_sell_thr = self.thresholds_avg.set_index("pair").loc[currency].sell_thr
            sum_buy_thr = self.thresholds_sum.set_index("pair").loc[currency].buy_thr
            sum_sell_thr = self.thresholds_sum.set_index("pair").loc[currency].sell_thr

            if predictions[0, c] > avg_buy_thr:
                trade = Trade(
                    timestamp=max_timestamp,
                    currency=currency,
                    entry_price=current_price,
                    type=Operation.BUY,
                    trigger=TriggerType.LR_AVG,
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
                    f"opening the trade on ts: {max_timestamp} currency: {currency} entry_price: {trade.entry_price}  type: {Operation.BUY} type: {TriggerType.LR_AVG} as prediction was {predictions[0, c]} and thr {avg_buy_thr}",
                    extra=kibana_extra_data,
                )
            if predictions[0, c] < avg_sell_thr:
                trade = Trade(
                    timestamp=max_timestamp,
                    currency=currency,
                    entry_price=current_price,
                    type=Operation.SELL,
                    trigger=TriggerType.LR_AVG,
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
                    f"opening the trade on ts: {max_timestamp} currency: {currency} entry_price: {trade.entry_price}  type: {Operation.SELL} type: {TriggerType.LR_SUM} as prediction was {predictions[0, c]} and thr {avg_sell_thr}",
                    extra=kibana_extra_data,
                )

            if predictions[0, c] > sum_buy_thr:
                trade = Trade(
                    timestamp=max_timestamp,
                    currency=currency,
                    entry_price=current_price,
                    type=Operation.BUY,
                    trigger=TriggerType.LR_SUM,
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
                    f"opening the trade on ts: {max_timestamp} currency: {currency} entry_price: {trade.entry_price}  type: {Operation.BUY} type: {TriggerType.LR_SUM} as prediction was {predictions[0, c]} and thr {sum_buy_thr}",
                    extra=kibana_extra_data,
                )
            if predictions[0, c] < sum_sell_thr:
                trade = Trade(
                    timestamp=max_timestamp,
                    currency=currency,
                    entry_price=current_price,
                    type=Operation.SELL,
                    trigger=TriggerType.LR_SUM,
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
                    f"opening the trade on ts: {max_timestamp} currency: {currency} entry_price: {trade.entry_price}  type: {Operation.SELL} type: {TriggerType.LR_AVG} as prediction was {predictions[0, c]} and thr {sum_sell_thr}",
                    extra=kibana_extra_data,
                )
        self.close_trades(max_timestamp, spot_prices)