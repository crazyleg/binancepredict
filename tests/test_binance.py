from datetime import datetime

import pytest
import vcr
import yaml
from api.binance import BinanceAPI

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


@vcr.use_cassette()
def test_getting_inference_data_async():
    api = BinanceAPI({"endpoint": "https://api.binance.com"})
    result = api.get_interence_data_async(
        cfg["models"]["resnet"]["symbols"], interval="1m"
    )
    # todo add asserts


@vcr.use_cassette()
def test_wait_for_1m_tick():
    api = BinanceAPI({"endpoint": "https://api.binance.com"})
    result = api.wait_for_1m_tick()
    assert result.second < 4
