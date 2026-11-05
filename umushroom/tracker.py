"""Since-prediction tracker: how has each pick moved since the run that
chose it?

For one run, every order is compared with the latest close:

    change = (latest close) / (close on the prediction date) - 1

Both closes come from the SAME Yahoo series, so the ratio is right even
when Yahoo quotes London stocks in pence (GBp) and the order in pounds.
"Price now" is then the order's own price scaled by that change, so it is
in the same units as the price you saw on the day of the prediction.

v4.25: values are in the account's currency, currency moves included -- a
USD stock's GBP value moves with GBP/USD, which is what the UMushroom account
shows. The FX pairs (USDGBP=X ...) come in the same download; "price now"
stays in the listing's own currency. A position whose FX pair cannot be
fetched falls back to its local change, and the page says so.

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
BENCHMARK_CURRENCY = "USD"    # IWDA.L is the USD line
MARGIN_DAYS = 10              # history fetched before the earliest prediction


def fx_pair(currency: str | None, base: str) -> str | None:
    """Yahoo symbol of `base` per 1 unit of `currency` (None = no conversion)."""
    c = (currency or "").strip()
    c = "GBP" if c == "GBp" else c.upper()       # pence are pounds for FX purposes
    if not c or c == base.upper():
        return None
    return f"{c}{base.upper()}=X"


def _base(payload: dict) -> str:
    return str((payload.get("meta") or {}).get("base_currency") or "GBP")


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
    change: float | None = None          # None = no price found (v4.25: in base
                                         # currency, currency moves included)
    last_date: str | None = None
    path: pd.Series | None = None        # growth of 1 since the prediction
    local_change: float | None = None    # the price change in the listing currency
    fx: bool = True                      # currency moves included (or none needed)

    @property
    def price_now(self) -> float | None:
        change = self.local_change if self.local_change is not None else self.change
        if self.price_then is None or change is None:
            return None
        return self.price_then * (1 + change)

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
    def fx(self) -> bool:
        """Every priced holding is valued with currency moves included."""
        return all(p.fx for p in self.priced)

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
    fx: bool = True            # v4.25: book and benchmark in base currency

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


def _in_base(path: pd.Series, closes: pd.DataFrame, pair: str | None, d0: date,
             until: date | None) -> tuple[pd.Series, bool]:
    """A local growth path times the FX pair's growth over the same days (the
    pair's last rate on or before each day). (path, True) when converted or
    nothing needed converting; (path unchanged, False) when the rate is missing."""
    if pair is None:
        return path, True
    if pair not in closes.columns:
        return path, False
    fx = _since(closes[pair], d0, until)
    if fx is None or not len(fx):
        return path, False
    rate = (fx.reindex(fx.index.union(path.index)).ffill()
            .reindex(path.index).fillna(1.0))
    return path * rate, True


def _score(payload: dict, closes: pd.DataFrame, until: str | date | None,
           today: date | None) -> TrackerResult:
    as_of = str(payload["meta"].get("as_of", ""))
    end = _as_date(until) if until else None
    result = TrackerResult(as_of, until=end.isoformat() if end else None)
    d0 = _prediction_date(payload, today)
    base = _base(payload)
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
            local = _since(closes[pos.symbol], d0, end)
            if local is not None and len(local):
                path, pos.fx = _in_base(local, closes, fx_pair(pos.currency, base), d0, end)
                pos.path = path
                pos.change = float(path.iloc[-1] - 1)
                pos.local_change = float(local.iloc[-1] - 1)
                pos.last_date = path.index[-1].strftime("%Y-%m-%d")
        result.positions.append(pos)
    return result


def _symbols(payloads: Iterable[dict], benchmark: str | None = None) -> list[str]:
    """Every price and FX series the payloads' orders (and the benchmark) need."""
    out: set[str] = set()
    for p in payloads:
        base = _base(p)
        for o in p.get("orders") or []:
            out.add(_symbol(o))
            pair = fx_pair(str(o.get("listing_currency") or ""), base)
            if pair:
                out.add(pair)
        if benchmark:
            out.add(benchmark)
            pair = fx_pair(BENCHMARK_CURRENCY, base)
            if pair:
                out.add(pair)
    return sorted(out)


def track(payload: dict, fetch: FetchFn = yahoo_closes,
          today: date | None = None, until: str | date | None = None) -> TrackerResult:
    """Compare one run's orders with the latest close (or the close on
    `until`, for a book that has since been replaced)."""
    orders = payload.get("orders") or []
    if not orders:
        return TrackerResult(str(payload["meta"].get("as_of", "")))
    d0 = _prediction_date(payload, today)
    closes = fetch(_symbols([payload]), d0 - timedelta(days=MARGIN_DAYS))
    return _score(payload, closes, until, today)


def record(entries: Iterable[tuple[str, dict, str | date | None]],
           fetch: FetchFn = yahoo_closes, benchmark: str | None = BENCHMARK,
           today: date | None = None) -> list[RecordRow]:
    """Score every book in `entries` = (stamp, payload, until) with ONE price
    download; the benchmark is scored over exactly the same closes."""
    items = [(s, p, u) for s, p, u in entries if p.get("orders")]
    if not items:
        return []
    start = min(_prediction_date(p, today) for _, p, _ in items)
    closes = fetch(_symbols([p for _, p, _ in items], benchmark),
                   start - timedelta(days=MARGIN_DAYS))
    rows: list[RecordRow] = []
    for stamp, payload, until in items:
        res = _score(payload, closes, until, today)
        bench, bench_fx = None, True
        if benchmark and benchmark in closes.columns:
            d0, end = _prediction_date(payload, today), (_as_date(until) if until else None)
            path = _since(closes[benchmark], d0, end)
            if path is not None and len(path):
                path, bench_fx = _in_base(path, closes,
                                          fx_pair(BENCHMARK_CURRENCY, _base(payload)), d0, end)
                bench = float(path.iloc[-1] - 1)
        rows.append(RecordRow(stamp, res.as_of, res.until, res.change, res.up,
                              len(res.priced), bench, res.last_date,
                              fx=res.fx and bench_fx))
    return rows
