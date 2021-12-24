from dataclasses import dataclass

from .ticketdata import TickerData


@dataclass
class TickerStream:
    ticks: list[TickerData]
    symbol: str
    interval: str

    def min_timestamp(self) -> float:
        return min([x.open_time for x in self.ticks])

    def latest_price(self) -> float:
        # TODO!!!
        pass

    def max_timestamp(self) -> float:
        return max([x.open_time for x in self.ticks])
