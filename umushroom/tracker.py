"""Since-prediction tracker: how has each pick moved since the run that
chose it?

For one run, every order is compared with the latest close:

    change = (latest close) / (close on the prediction date) - 1

Both closes come from the SAME Yahoo series, so the ratio is right even
when Yahoo quotes London stocks in pence (GBp) and the order in pounds.
"Price now" is then the order's own price scaled by that change, so it is
in the same units as the price you saw on the day of the prediction.

Values ignore currency moves since the prediction (a USD stock's GBP value
also moves with GBP/USD); the tracker says so on the page.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

FetchFn = Callable[[list[str], date], pd.DataFrame]   # (symbols, start) -> closes


def yahoo_closes(symbols: list[str], start: date) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(symbols, start=start.isoformat(), interval="1d",
                      auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    closes = raw.get("Close", raw)
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes


@dataclass
class TrackedPosition:
    sleeve: str
    ticker: str
    symbol: str
    name: str
    currency: str
    shares: float | None
    price_then: float | None
    value_then: float
    change: float | None = None          # None = no price found
    last_date: str | None = None
    path: pd.Series | None = None        # growth of 1 since the prediction

    @property
    def price_now(self) -> float | None:
        if self.price_then is None or self.change is None:
            return None
        return self.price_then * (1 + self.change)

    @property
    def value_now(self) -> float | None:
        return None if self.change is None else self.value_then * (1 + self.change)


@dataclass
class TrackerResult:
    as_of: str
    positions: list[TrackedPosition] = field(default_factory=list)

    @property
    def priced(self) -> list[TrackedPosition]:
        return [p for p in self.positions if p.change is not None]

    @property
    def value_then(self) -> float:
        return sum(p.value_then for p in self.priced)

    @property
    def value_now(self) -> float:
        return sum(p.value_now or 0.0 for p in self.priced)

    @property
    def change(self) -> float | None:
        return self.value_now / self.value_then - 1 if self.value_then > 0 else None

    def portfolio_path(self) -> pd.Series:
        """Value of the run's (priced) orders on each trading day since."""
        parts = [p.path * p.value_then for p in self.priced if p.path is not None]
        if not parts:
            return pd.Series(dtype=float)
        return pd.concat(parts, axis=1).ffill().bfill().sum(axis=1)


def _symbol(order: dict) -> str:
    return str(order.get("priced_from") or order["ticker"])


def track(payload: dict, fetch: FetchFn = yahoo_closes,
          today: date | None = None) -> TrackerResult:
    """Compare one run's orders with the latest prices."""
    as_of = str(payload["meta"].get("as_of", ""))
    result = TrackerResult(as_of)
    orders = payload.get("orders") or []
    if not orders:
        return result
    try:
        d0 = date.fromisoformat(as_of[:10])
    except ValueError:
        d0 = (today or date.today()) - timedelta(days=1)

    symbols = sorted({_symbol(o) for o in orders})
    closes = fetch(symbols, d0 - timedelta(days=10))
    for o in orders:
        lp = o.get("local_price")
        pos = TrackedPosition(
            sleeve=str(o.get("sleeve") or ""), ticker=str(o["ticker"]), symbol=_symbol(o),
            name=str(o.get("name") or ""), currency=str(o.get("listing_currency") or ""),
            shares=o.get("shares"),
            price_then=float(lp) if lp is not None else (
                float(o["price"]) if o.get("price") is not None else None),
            value_then=float(o.get("allocated_value") or 0.0))
        if pos.symbol in closes.columns:
            s = closes[pos.symbol].dropna()
            s = s[s > 0]
            base = s[s.index <= pd.Timestamp(d0)]
            if len(base):
                since = s[s.index >= base.index[-1]]
                pos.path = since / base.iloc[-1]
                pos.change = float(pos.path.iloc[-1] - 1)
                pos.last_date = since.index[-1].strftime("%Y-%m-%d")
        result.positions.append(pos)
    return result
