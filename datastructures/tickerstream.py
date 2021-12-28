from dataclasses import dataclass

from .ticketdata import TickerData


@dataclass
class TickerStream:
    ticks: list[TickerData]
    symbol: str
    interval: str

    def min_timestamp(self) -> float:
        return min([x.open_time for x in self.ticks])

    def latest_open_price(self) -> float:
        latest_timestep = self.max_timestamp()
        return [x.open for x in self.ticks if x.open_time == latest_timestep][0]

    def latest_close_price(self) -> float:
        latest_timestep = self.max_timestamp()
        return [x.close for x in self.ticks if x.open_time == latest_timestep][0]

    def max_timestamp(self) -> float:
        return max([x.open_time for x in self.ticks])
