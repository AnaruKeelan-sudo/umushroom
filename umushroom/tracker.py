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

v4.21: `record` scores a whole run history book by book -- each holding
period's book from the close its signals were built on to the close the
next book was bought at -- against a benchmark over the same days. That is
the strategy's live track record, and the only honest answer to "were the
predictions right?": a run published this morning has no close newer than
its prediction yet (`TrackerResult.newer` is False), so every change is
exactly zero and there is nothing to score until tomorrow.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

FetchFn = Callable[[list[str], date], pd.DataFrame]   # (symbols, start) -> closes
BENCHMARK = "IWDA.L"          # iShares Core MSCI World: the record's yardstick
MARGIN_DAYS = 10              # history fetched before the earliest prediction


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
    until: str | None = None             # scored up to this close (None = latest)

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

    @property
    def up(self) -> int:
        return sum(1 for p in self.priced if p.change is not None and p.change > 0)

    @property
    def last_date(self) -> str | None:
        dates = [p.last_date for p in self.priced if p.last_date]
        return max(dates) if dates else None

    @property
    def newer(self) -> bool:
        """Is there a close AFTER the prediction date to score against? On
        the morning a run is published its latest close is the very close
        the model chose from, so every change is exactly zero: no data yet,
        not a result (the published site read "0 / 21 up" for that reason)."""
        last = self.last_date
        return bool(last and last > self.as_of[:10])

    def portfolio_path(self) -> pd.Series:
        """Value of the run's (priced) orders on each trading day since."""
        parts = [p.path * p.value_then for p in self.priced if p.path is not None]
        if not parts:
            return pd.Series(dtype=float)
        return pd.concat(parts, axis=1).ffill().bfill().sum(axis=1)


@dataclass(frozen=True)
class RecordRow:
    """One holding period's book, scored from its prediction close to the
    close the next book was bought at (`until` None = still held: so far)."""
    stamp: str
    as_of: str
    until: str | None
    change: float | None       # value-weighted change of the priced orders
    up: int
    n: int                     # priced holdings
    benchmark: float | None    # the benchmark over the same closes
    last_date: str | None

    @property
    def active(self) -> float | None:
        if self.change is None or self.benchmark is None:
            return None
        return self.change - self.benchmark


def _symbol(order: dict) -> str:
    return str(order.get("priced_from") or order["ticker"])


def _as_date(d: str | date) -> date:
    return d if isinstance(d, date) else date.fromisoformat(str(d)[:10])


def _prediction_date(payload: dict, today: date | None) -> date:
    try:
        return date.fromisoformat(str(payload["meta"].get("as_of", ""))[:10])
    except ValueError:
        return (today or date.today()) - timedelta(days=1)


def _since(series: pd.Series, d0: date, until: date | None) -> pd.Series | None:
    """Growth of 1 from the last close on or before `d0`, up to `until`."""
    s = series.dropna()
    s = s[s > 0]
    base = s[s.index <= pd.Timestamp(d0)]
    if not len(base):
        return None
    since = s[s.index >= base.index[-1]]
    if until is not None:
        since = since[since.index <= pd.Timestamp(until)]
    return since / base.iloc[-1]


def _score(payload: dict, closes: pd.DataFrame, until: str | date | None,
           today: date | None) -> TrackerResult:
    as_of = str(payload["meta"].get("as_of", ""))
    end = _as_date(until) if until else None
    result = TrackerResult(as_of, until=end.isoformat() if end else None)
    d0 = _prediction_date(payload, today)
    for o in payload.get("orders") or []:
        lp = o.get("local_price")
        pos = TrackedPosition(
            sleeve=str(o.get("sleeve") or ""), ticker=str(o["ticker"]), symbol=_symbol(o),
            name=str(o.get("name") or ""), currency=str(o.get("listing_currency") or ""),
            shares=o.get("shares"),
            price_then=float(lp) if lp is not None else (
                float(o["price"]) if o.get("price") is not None else None),
            value_then=float(o.get("allocated_value") or 0.0))
        if pos.symbol in closes.columns:
            path = _since(closes[pos.symbol], d0, end)
            if path is not None and len(path):
                pos.path = path
                pos.change = float(path.iloc[-1] - 1)
                pos.last_date = path.index[-1].strftime("%Y-%m-%d")
        result.positions.append(pos)
    return result


def track(payload: dict, fetch: FetchFn = yahoo_closes,
          today: date | None = None, until: str | date | None = None) -> TrackerResult:
    """Compare one run's orders with the latest close (or the close on
    `until`, for a book that has since been replaced)."""
    orders = payload.get("orders") or []
    if not orders:
        return TrackerResult(str(payload["meta"].get("as_of", "")))
    d0 = _prediction_date(payload, today)
    symbols = sorted({_symbol(o) for o in orders})
    closes = fetch(symbols, d0 - timedelta(days=MARGIN_DAYS))
    return _score(payload, closes, until, today)


def record(entries: Iterable[tuple[str, dict, str | date | None]],
           fetch: FetchFn = yahoo_closes, benchmark: str | None = BENCHMARK,
           today: date | None = None) -> list[RecordRow]:
    """Score every book in `entries` = (stamp, payload, until) with ONE price
    download; the benchmark is scored over exactly the same closes."""
    items = [(s, p, u) for s, p, u in entries if p.get("orders")]
    if not items:
        return []
    symbols = {_symbol(o) for _, p, _ in items for o in p["orders"]}
    if benchmark:
        symbols.add(benchmark)
    start = min(_prediction_date(p, today) for _, p, _ in items)
    closes = fetch(sorted(symbols), start - timedelta(days=MARGIN_DAYS))
    rows: list[RecordRow] = []
    for stamp, payload, until in items:
        res = _score(payload, closes, until, today)
        bench = None
        if benchmark and benchmark in closes.columns:
            path = _since(closes[benchmark], _prediction_date(payload, today),
                          _as_date(until) if until else None)
            if path is not None and len(path):
                bench = float(path.iloc[-1] - 1)
        rows.append(RecordRow(stamp, res.as_of, res.until, res.change, res.up,
                              len(res.priced), bench, res.last_date))
    return rows
