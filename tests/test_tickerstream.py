import pytest
from datastructures.tickerstream import TickerStream
from datastructures.ticketdata import TickerData


class TestTickerStream:
    """A class with common parameters, `param1` and `param2`."""

    @pytest.fixture
    def ticker_stream(self):
        return TickerStream(
            ticks=[
                TickerData(
                    symbol="ETCBTC",
                    open=1,
                    close=2,
                    open_time=1,
                    close_time=2,
                    high=1,
                    low=2,
                    volume=3,
                    quote_asset_volume=1,
                    number_of_trades=3,
                    taker_buy_base_asset_volume=2,
                    taker_buy_quote_asset_volume=2,
                    ignore=0,
                    interval="1m",
                ),
                TickerData(
                    symbol="ETCBTC",
                    open=3,
                    close=4,
                    open_time=2,
                    close_time=3,
                    high=1,
                    low=2,
                    volume=3,
                    quote_asset_volume=1,
                    number_of_trades=3,
                    taker_buy_base_asset_volume=2,
                    taker_buy_quote_asset_volume=2,
                    ignore=0,
                    interval="1m",
                ),
            ],
            symbol="ETCBTC",
            interval="1m",
        )

    def test_max_timestamps(self, ticker_stream):
        assert ticker_stream.max_timestamp() == 2

    def test_min_timestamps(self, ticker_stream):
        assert ticker_stream.min_timestamp() == 1

    def test_latest_open_price(self, ticker_stream):
        assert ticker_stream.latest_open_price() == 3

    def test_latest_close_price(self, ticker_stream):
        assert ticker_stream.latest_close_price() == 4
