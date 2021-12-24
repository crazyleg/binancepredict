import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Mapping

from datastructures.tickerstream import TickerStream


class Operation(Enum):
    BUY = 1
    SELL = 2


class TriggerType(Enum):
    AVG = 1
    SUM = 2


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

    def make_trade(self, trade: Trade):
        self.trades.add_trade(trade)
        self.active_trades += 1

    def __len__(self):
        return len(self.trades.trades)

    def check_trade_for_closing(self, trade: Trade, timestamp, prices: TickerStream):
        if (
            datetime.fromtimestamp(prices.max_timestamp() / 1000)
            - datetime.fromtimestamp(timestamp / 1000)
        ).total_seconds() > 59:
            logging.info(
                f"closing the trade on ts: {timestamp} currency: {trade.currency} price: {prices} type: {Operation.SELL} type: {TriggerType.SUM}"
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
        # TODO SUPER BLOCKER!!!! - do not open trade if expected profit < 0.03

        for c, currency in enumerate(self.cfg["models"]["resnet"]["symbols"]):
            current_price = float(
                [
                    x
                    for x in spot_prices[currency].ticks
                    if x.open_time == max_timestamp
                ][0].close
            )
            if (
                predictions[0, c]
                > self.thresholds.set_index("currency")
                .loc[currency]
                .best_thr_avg_positive
            ):
                trade = Trade(
                    timestamp=max_timestamp,
                    currency=currency,
                    entry_price=current_price,
                    type=Operation.BUY,
                    trigger=TriggerType.AVG,
                )
                self.make_trade(trade)
                logging.info(
                    f"opening the trade on ts: {max_timestamp} currency: {currency} price: {current_price} type: {Operation.BUY} type: {TriggerType.AVG}"
                )
            if (
                predictions[0, c]
                > self.thresholds.set_index("currency")
                .loc[currency]
                .best_thr_sum_positive
            ):
                trade = Trade(
                    timestamp=max_timestamp,
                    currency=currency,
                    entry_price=current_price,
                    type=Operation.BUY,
                    trigger=TriggerType.SUM,
                )
                self.make_trade(trade)
                logging.info(
                    f"opening the trade on ts: {max_timestamp} currency: {currency} price: {current_price} type: {Operation.BUY} type: {TriggerType.SUM}"
                )

            if (
                predictions[0, c]
                < self.thresholds.set_index("currency")
                .loc[currency]
                .best_thr_avg_negative
            ):
                trade = Trade(
                    timestamp=max_timestamp,
                    currency=currency,
                    entry_price=current_price,
                    type=Operation.SELL,
                    trigger=TriggerType.AVG,
                )
                self.make_trade(trade)
                logging.info(
                    f"opening the trade on ts: {max_timestamp} currency: {currency} price: {current_price} type: {Operation.SELL} type: {TriggerType.AVG}"
                )
            if (
                predictions[0, c]
                > self.thresholds.set_index("currency")
                .loc[currency]
                .best_thr_sum_negative
            ):
                trade = Trade(
                    timestamp=max_timestamp,
                    currency=currency,
                    entry_price=current_price,
                    type=Operation.SELL,
                    trigger=TriggerType.SUM,
                )
                self.make_trade(trade)
                logging.info(
                    f"opening the trade on ts: {max_timestamp} currency: {currency} price: {current_price} type: {Operation.SELL} type: {TriggerType.SUM}"
                )

        self.close_trades(max_timestamp, spot_prices)
