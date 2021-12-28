from dataclasses import dataclass


@dataclass
class TickerData:
    symbol: str
    interval: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_asset_volume: float
    number_of_trades: float
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float
    ignore: float

    def __post_init__(self):
        self.open_time = int(self.open_time)
        self.open = float(self.open)
        self.high = float(self.high)
        self.low = float(self.low)
        self.close = float(self.close)
        self.volume = float(self.volume)
        self.close_time = int(self.close_time)
        self.quote_asset_volume = float(self.quote_asset_volume)
        self.number_of_trades = float(self.number_of_trades)
        self.taker_buy_base_asset_volume = float(self.taker_buy_base_asset_volume)
        self.taker_buy_quote_asset_volume = float(self.taker_buy_quote_asset_volume)
        self.ignore = float(self.ignore)
