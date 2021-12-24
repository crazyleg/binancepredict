import pytest
import vcr
import yaml
from api.binance import BinanceAPI
from helpers import prepare_data

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


@vcr.use_cassette()
def test_prepare_data():
    api = BinanceAPI({"endpoint": "https://api.binance.com"})
    result = api.get_inference_data(cfg["models"]["resnet"]["symbols"], interval="1m")

    result = prepare_data(cfg, result)
