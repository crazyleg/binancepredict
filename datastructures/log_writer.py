import logging
from enum import Enum

from datastructures.operations import Operation


class LogEvent(Enum):
    TICK = 1
    OPEN_TRADE = 2
    CLOSE_TRADE = 3


class LogWriter:
    def log_currency_tick_data(
        self,
        ts,
        currency,
        spot_price,
        prediction,
        buy_thr,
        sell_thr,
    ):
        kibana_extra_data = {
            "log_event_type": LogEvent.TICK,
            "currency": currency,
            "spot_price": spot_price,
            "prediction": prediction.item(),
            "best_thr_avg_positive": buy_thr,
            "best_thr_sum_positive": sell_thr,
        }
        logging.info(
            f"logging spot data on ts: {ts} currency: {currency} spot_price: {spot_price} prediction: {prediction.item()} buy_thr: {buy_thr} sell_thr: {sell_thr} ",
            extra=kibana_extra_data,
        )

    def log_trade_close(self, timestamp, trade, close_price):
        profit_before_commission = (trade.entry_price - close_price) / trade.entry_price
        if trade.type == Operation.BUY:
            profit_before_commission *= -1

        profit = profit_before_commission - 0.001  # Commission accounted for

        kibana_extra_data = {
            "log_event_type": LogEvent.CLOSE_TRADE,
            "currency": trade.currency,
            "open_price": trade.entry_price,
            "close_price": close_price,
            "trade_type": trade.type,
            "trade_trigger": trade.trigger,
            "profit": profit_before_commission,
            "profit_real": profit,
        }

        logging.info(
            f"closing the trade on ts: {timestamp} currency: {trade.currency} open_price: {trade.entry_price} close_price: {close_price} type: {trade.type} type: {trade.trigger}",
            extra=kibana_extra_data,
        )
