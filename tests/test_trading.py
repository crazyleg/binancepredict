import logging
from time import time

import pandas as pd
import pytest
import yaml
from datastructures.trading import Operation, Trade, Trading, TriggerType

thresholds = pd.read_pickle("ml/models/thresholds.pkl")

test_logger = logging.getLogger("test_logger")

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


def test_add_trade():
    trades = Trading(cfg, thresholds, test_logger)
    test_trade = Trade(
        timestamp=time(),
        currency="ETCBTC",
        entry_price=1,
        type=Operation.BUY,
        trigger=TriggerType.AVG,
    )
    trades.make_trade(test_trade)

    assert len(trades) == 1


def test_match_trades():
    trades = Trading(cfg, thresholds, test_logger)
    test_trade = Trade(
        timestamp=100,
        currency="ETCBTC",
        entry_price=1,
        type=Operation.BUY,
        trigger=TriggerType.AVG,
    )
    trades.make_trade(test_trade)
    trades.close_trades(1000, spot_prices={"ETCBTC": []})
