"""Market data acquisition layer.

Responsibilities:
  - Fetch daily close prices from yfinance
  - Retry transient network failures with backoff
  - Validate the result (no wrong shape, no all-NaN columns, no
    unreasonable gaps) before handing it to the rest of the pipeline
  - Cache to local parquet so repeated runs within the same trading
    day don't re-hit the network, and so the pipeline still works
    offline against the last good snapshot

This isolates the only part of the system that talks to the outside
world, which is also the only part that is non-deterministic and
hard to unit test — everything downstream consumes a plain
pandas.DataFrame and is fully testable with synthetic data.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from umushroom.exceptions import DataError, InsufficientHistoryError
from umushroom.logging_config import get_logger
from umushroom.universe import Listing, required_fx_tickers, resolve_universe

logger = get_logger("data")

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
MIN_TICKER_COVERAGE = 0.9   # a ticker must have prices on >=90% of days
                            # or it is dropped (see _validate_prices)
LOCAL_TZ = ZoneInfo("Europe/London")
# v4.19: point this at a price snapshot (see snapshot_fetcher) to run the
# whole pipeline offline on a fixed, reproducible set of prices.
SNAPSHOT_ENV = "UMUSHROOM_PRICE_SNAPSHOT"
SPIKE_THRESHOLD = 0.95      # |daily return| above this = presumed data error


def _cache_path(cache_dir: Path, tickers: list[str], period: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = "_".join(sorted(tickers)) + f"_{period}"
    # v4: hash the FULL key. v3 truncated it to 150 chars, so two large
    # universes sharing a prefix silently shared (stale) cached data.
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    prefix = "".join(c if c.isalnum() else "_" for c in key)[:40]
    return cache_dir / f"{prefix}_{digest}.parquet"


def _cache_is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < max_age_hours


def _fetch_with_retries(tickers: list[str], period: str
                        ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    # Imported lazily so the whole offline test suite (and any library
    # use that passes its own prices) never needs yfinance importable.
    import yfinance as yf

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Fetching data (attempt {attempt}/{MAX_RETRIES}) "
                        f"for {len(tickers)} tickers, period={period}")
            raw = yf.download(
                tickers, period=period, interval="1d",
                auto_adjust=True, progress=False, threads=True,
            )
            if raw is None or raw.empty:
                raise DataError("yfinance returned an empty result")

            closes = raw.get("Close", raw) if isinstance(raw, pd.DataFrame) else raw
            if isinstance(closes, pd.Series):
                closes = closes.to_frame(tickers[0])
            volumes = raw.get("Volume") if isinstance(raw, pd.DataFrame) else None
            if isinstance(volumes, pd.Series):
                volumes = volumes.to_frame(tickers[0])
            return closes, volumes
        except Exception as e:  # noqa: BLE001 — yfinance/requests/urllib3 raise
            # a wide and version-dependent set of exception types for
            # network failures; catching broadly here and re-raising as
            # our own typed DataError after retries is intentional, not
            # an oversight.
            last_error = e
            logger.warning(f"Fetch attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise DataError(
        f"Failed to fetch data after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def _validate_prices(data: pd.DataFrame, requested_tickers: list[str],
                      min_history_days: int,
                      spike_exempt: frozenset[str] = frozenset(),
                      ragged: bool = False) -> pd.DataFrame:
    """Validate a (date x ticker) close-price frame.

    ragged=False (v4 behaviour): every ticker must cover >= 90% of days and
    the result has no NaN at all -- simple, but a recent listing either
    gets dropped or truncates everyone's history.
    ragged=True (v4.4): each ticker keeps whatever history it has, with
    NaN before it listed; it is kept if it has >= min_history_days of
    prices. Features only become defined once a ticker has a full year of
    data, so new listings JOIN the model when ready instead of being
    discarded, and long lookbacks (10y) don't throw most tickers away.
    """
    if data.empty:
        raise DataError("Downloaded price data is empty")

    # Drop tickers that returned no usable data at all, but fail loudly
    # if we lose a meaningful fraction of the requested universe —
    # silently trading a shrunken universe is a correctness bug.
    fully_missing = [c for c in data.columns if data[c].isna().all()]
    if fully_missing:
        logger.warning(f"No data returned for: {fully_missing}")

    missing_from_response = set(requested_tickers) - set(data.columns)
    if missing_from_response:
        logger.warning(f"Tickers absent from response entirely: "
                        f"{sorted(missing_from_response)}")

    usable = data.drop(columns=fully_missing, errors="ignore")
    usable = usable.ffill(limit=3)   # tolerate short gaps (holidays, etc.)

    if ragged:
        n_obs = usable.notna().sum()
        short = n_obs[n_obs < min_history_days].index.tolist()
        if short:
            logger.warning(f"Dropping {len(short)} ticker(s) with fewer than "
                           f"{min_history_days} days of prices: {short}")
            usable = usable.drop(columns=short)
        usable = usable.dropna(how="all")
    else:
        # v4: drop tickers with short histories (recent IPOs, long
        # suspensions) BEFORE the row-wise dropna. In v3 a single late-listed
        # ticker silently truncated the history of the entire universe.
        coverage = usable.notna().mean()
        low_coverage = coverage[coverage < MIN_TICKER_COVERAGE].index.tolist()
        if low_coverage:
            logger.warning(
                f"Dropping {len(low_coverage)} ticker(s) with <"
                f"{MIN_TICKER_COVERAGE:.0%} price coverage: {low_coverage}"
            )
            usable = usable.drop(columns=low_coverage)
        usable = usable.dropna()

    # v4.3: a >95% single-day move in a stock or ETF is almost always a
    # data error -- typically Yahoo flipping a London price between pence
    # and pounds (a 100x jump). Real run on 2026-10-08 found 14 such
    # tickers; they made the equal-weight benchmark jump ~17% in a month
    # and corrupted the rankings. They are now DROPPED, not just flagged.
    # Crypto (spike_exempt) can genuinely move this much and is kept.
    daily_moves = usable.pct_change(fill_method=None).abs()
    spiky = [c for c in usable.columns
             if (daily_moves[c] > SPIKE_THRESHOLD).any()]
    to_drop = [c for c in spiky if c not in spike_exempt]
    kept = [c for c in spiky if c in spike_exempt]
    if to_drop:
        logger.warning(
            f"Dropping {len(to_drop)} ticker(s) with single-day moves >"
            f"{SPIKE_THRESHOLD:.0%} (likely data errors, e.g. pence/pound "
            f"flips): {to_drop}")
        usable = usable.drop(columns=to_drop)
    if kept:
        logger.info(f"Large single-day moves kept for exempt (crypto) tickers: {kept}")

    loss_fraction = 1 - (len(usable.columns) / max(len(requested_tickers), 1))
    if loss_fraction > 0.5:
        raise DataError(
            f"Lost more than half the requested universe after validation "
            f"({len(usable.columns)}/{len(requested_tickers)} tickers usable). "
            f"Aborting rather than running on a severely degraded universe."
        )

    if usable.shape[1] == 0 or len(usable) < min_history_days:
        raise InsufficientHistoryError(
            f"Only {len(usable)} valid trading days available; "
            f"need at least {min_history_days}."
        )

    return usable


def get_price_data(
    tickers: list[str],
    period: str,
    cache_dir: Path,
    cache_max_age_hours: float,
    min_history_days: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a validated DataFrame of daily close prices, ticker columns,
    date index. Uses a local cache to avoid redundant network calls.
    """
    cache_path = _cache_path(cache_dir, tickers, period)

    if not force_refresh and _cache_is_fresh(cache_path, cache_max_age_hours):
        logger.info(f"Using cached data: {cache_path} "
                     f"(age < {cache_max_age_hours}h)")
        data = pd.read_parquet(cache_path)
    else:
        data, _volumes = _fetch_with_retries(tickers, period)
        data.to_parquet(cache_path)
        logger.info(f"Cached fresh data to {cache_path} at "
                     f"{datetime.now(timezone.utc).isoformat()}")

    return _validate_prices(data, tickers, min_history_days)


# --------------------------------------------------------------------------
# v4: multi-market, base-currency data
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame          # base-currency closes; what EVERYTHING
                                  # downstream (features, backtests,
                                  # allocation sizing) consumes
    local_prices: pd.DataFrame    # closes in each listing's own currency
                                  # (pounds, not pence), for order entry
    listings: dict[str, Listing]  # only tickers that survived validation
    base_currency: str
    # v4.4: traded value per day in base currency (shares x base price);
    # None when the data source gave no volumes
    dollar_volume: pd.DataFrame | None = None
    # v4.6: tickers with NO actual price on the latest day (their last price
    # was carried forward from an earlier day). Never picked for new orders.
    stale_last: frozenset[str] = frozenset()


class SnapshotFetcher:
    """Replays prices from a JSON(.gz) snapshot instead of calling Yahoo.

    Snapshot shape: {"fetched_at": ISO timestamp, "meta": {symbol: {...}},
    "series": {symbol: {"d": [days since 1970-01-01], "c": [adjusted
    closes], "v": [volumes]}}}. Symbols missing from the snapshot come back
    as all-NaN columns, exactly as yfinance returns an unknown ticker, so
    every downstream validation behaves the same way as in a live run.
    Why: GitHub and the team's laptops can reach Yahoo, research sandboxes
    often cannot; a snapshot makes any run reproducible to the last price.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rt", encoding="utf-8") as fh:
            raw = json.load(fh)
        self._series = raw.get("series") or {}
        self.meta = raw.get("meta") or {}
        stamp = str(raw.get("fetched_at") or "")
        self.as_of: date | None = None
        if stamp:
            try:
                self.as_of = datetime.fromisoformat(stamp.replace("Z", "+00:00")).date()
            except ValueError:
                self.as_of = None
        logger.info(f"Price snapshot {self.path} ({len(self._series)} symbols, "
                    f"fetched {stamp or 'unknown'})")

    def __call__(self, symbols: list[str], period: str
                 ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        closes: dict[str, pd.Series] = {}
        volumes: dict[str, pd.Series] = {}
        for sym in symbols:
            s = self._series.get(sym)
            if not s or not s.get("c"):
                continue
            idx = pd.to_datetime(pd.Index(s["d"], dtype="int64"), unit="D")
            close = pd.Series(s["c"], index=idx, dtype=float)
            # Yahoo can repeat a day (the live quote beside the daily bar):
            # the later entry is the completed one
            close = close[~close.index.duplicated(keep="last")]
            closes[sym] = close
            vol = s.get("v")
            if vol is not None:
                v_ser = pd.Series([float("nan") if v is None else float(v) for v in vol],
                                  index=idx, dtype=float)
                volumes[sym] = v_ser[~v_ser.index.duplicated(keep="last")]
        if not closes:
            raise DataError(f"Price snapshot {self.path} has none of the requested symbols")
        close_df = pd.DataFrame(closes).sort_index()
        close_df = close_df[~close_df.index.duplicated(keep="last")]
        close_df = close_df.reindex(columns=symbols)        # unknown -> all NaN
        vol_df = pd.DataFrame(volumes).sort_index().reindex(index=close_df.index,
                                                            columns=symbols)
        return close_df, vol_df


def snapshot_fetcher(path: str | Path | None = None) -> SnapshotFetcher | None:
    """The configured snapshot fetcher, if UMUSHROOM_PRICE_SNAPSHOT (or
    `path`) names an existing file; otherwise None (= live Yahoo)."""
    p = path or os.environ.get(SNAPSHOT_ENV)
    if not p:
        return None
    if not Path(p).exists():
        raise DataError(f"{SNAPSHOT_ENV} points to a missing file: {p}")
    return SnapshotFetcher(p)


def drop_incomplete_last_bar(data: pd.DataFrame, today: date) -> pd.DataFrame:
    """Drop any bar dated `today` or later.

    UMushroom fills orders at the PREVIOUS day's close. If the pipeline
    runs while a market is open, yfinance returns a partial, still-moving
    bar for today: features would then be built on an intraday price
    while the fill is at yesterday's close -- a silent mismatch between
    what the model saw and what the trade actually gets. Using only
    completed daily bars keeps signal date == fill price date, which is
    exactly the assumption the backtests make.
    """
    idx_dates = pd.DatetimeIndex(data.index).tz_localize(None).normalize()
    keep = idx_dates < pd.Timestamp(today)
    dropped = int((~keep).sum())
    if dropped:
        logger.info(f"Dropped {dropped} incomplete bar(s) dated >= {today}")
    return data.loc[keep]


def align_to_trading_calendar(data: pd.DataFrame, calendar_tickers: list[str],
                              min_fraction: float = 0.5) -> pd.DataFrame:
    """Keep only days on which most `calendar_tickers` (the stocks) traded.

    Crypto trades 7 days a week; stocks 5. Without this, weekend rows would
    give every stock two zero-return days a week, silently shrinking the
    "252-day" factor windows to ~7 months and distorting volatility. Crypto
    keeps its weekend moves: they simply land in Monday's return.
    """
    cols = [c for c in calendar_tickers if c in data.columns]
    if not cols:
        return data
    keep = data[cols].notna().mean(axis=1) >= min_fraction
    dropped = int((~keep).sum())
    if dropped:
        logger.info(f"Dropped {dropped} non-trading day(s) (weekends/holidays) "
                    f"to align with the stock calendar")
    return data.loc[keep]


def convert_to_base_currency(
    local_quoted: pd.DataFrame,
    fx: pd.DataFrame,
    listings: dict[str, Listing],
    base_currency: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (base_currency_prices, local_prices_in_major_units).

    FX series are forward-filled only (never back-filled -- a back-fill
    would use a future rate). Rows before an FX series starts are
    dropped.
    """
    local_cols: dict[str, pd.Series] = {}
    base_cols: dict[str, pd.Series] = {}
    fx = fx.reindex(local_quoted.index).ffill()
    for ticker in local_quoted.columns:
        listing = listings[ticker]
        px = local_quoted[ticker] / listing.price_divisor
        local_cols[ticker] = px
        if listing.currency == base_currency:
            base_cols[ticker] = px
        else:
            pair = f"{listing.currency}{base_currency}=X"
            if pair not in fx.columns:
                raise DataError(f"Missing FX series {pair} needed for {ticker}")
            base_cols[ticker] = px * fx[pair]
    # build each frame in one go (column-by-column inserts fragment the
    # frame and flood the log with PerformanceWarnings on 600+ tickers)
    local = pd.DataFrame(local_cols, index=local_quoted.index)
    base = pd.DataFrame(base_cols, index=local_quoted.index)
    needed = sorted({f"{listings[t].currency}{base_currency}=X"
                     for t in local_quoted.columns
                     if listings[t].currency != base_currency})
    valid = (fx[needed].notna().all(axis=1) if needed
             else pd.Series(True, index=local_quoted.index))
    if not valid.all():
        logger.info(f"Dropped {int((~valid).sum())} leading row(s) lacking FX rates")
    return base.loc[valid], local.loc[valid]


def get_market_data(
    tickers: list[str],
    period: str,
    cache_dir: Path,
    cache_max_age_hours: float,
    min_history_days: int,
    base_currency: str,
    currency_overrides: dict[str, str] | None = None,
    force_refresh: bool = False,
    drop_today_bar: bool = True,
    today: date | None = None,
    fetcher=None,
    calendar_tickers: list[str] | None = None,
    ragged: bool = False,
    yahoo_symbols: dict[str, str] | None = None,
) -> MarketData:
    """Fetch, validate and currency-convert the full universe.

    `fetcher(tickers, period) -> DataFrame` is injectable so the whole
    path is testable offline; it defaults to the yfinance fetcher.
    """
    listings = resolve_universe(tickers, currency_overrides)
    fx_tickers = required_fx_tickers(listings, base_currency)
    sym_map = {t: (yahoo_symbols or {}).get(t, t) for t in tickers}
    back = {v: k for k, v in sym_map.items()}
    all_symbols = [sym_map[t] for t in tickers] + fx_tickers
    if fetcher is None:
        snapshot = snapshot_fetcher()
        if snapshot is not None:
            fetcher = snapshot
            # the snapshot's own date decides which bar is "today's" partial one
            today = today or snapshot.as_of
    fetch = fetcher or _fetch_with_retries

    cache_path = _cache_path(cache_dir, all_symbols, period)
    vol_cache = cache_path.with_name(cache_path.stem + "_volume.parquet")
    volumes: pd.DataFrame | None = None
    if fetcher is None and not force_refresh and _cache_is_fresh(
            cache_path, cache_max_age_hours):
        logger.info(f"Using cached data: {cache_path}")
        raw = pd.read_parquet(cache_path)
        if vol_cache.exists():
            volumes = pd.read_parquet(vol_cache)
    else:
        fetched = fetch(all_symbols, period)
        raw, volumes = fetched if isinstance(fetched, tuple) else (fetched, None)
        if fetcher is None:
            raw.to_parquet(cache_path)
            if volumes is not None:
                volumes.to_parquet(vol_cache)
            logger.info(f"Cached fresh data to {cache_path} at "
                        f"{datetime.now(timezone.utc).isoformat()}")

    # fetched under Yahoo symbols -> rename back to UMushroom tickers
    if any(k != v for k, v in sym_map.items()):
        raw = raw.rename(columns=back)
        if volumes is not None:
            volumes = volumes.rename(columns=back)

    if drop_today_bar:
        raw = drop_incomplete_last_bar(raw, today or datetime.now(LOCAL_TZ).date())

    if calendar_tickers:
        raw = align_to_trading_calendar(raw, calendar_tickers)

    missing_fx = [f for f in fx_tickers if f not in raw.columns or raw[f].isna().all()]
    if missing_fx:
        raise DataError(f"FX rates unavailable for {missing_fx}; cannot convert "
                        f"prices to {base_currency}")

    equities = raw[[t for t in tickers if t in raw.columns]]
    crypto = frozenset(t for t, lst in listings.items() if lst.exchange == "Crypto")
    validated_local = _validate_prices(equities, tickers, min_history_days,
                                       spike_exempt=crypto, ragged=ragged)
    surviving = {t: listings[t] for t in validated_local.columns}

    prices, local = convert_to_base_currency(
        validated_local, raw[fx_tickers], surviving, base_currency)
    if len(prices) < min_history_days:
        raise InsufficientHistoryError(
            f"Only {len(prices)} days remain after FX alignment; "
            f"need {min_history_days}")

    raw_last = equities.reindex(index=prices.index, columns=prices.columns).iloc[-1]
    stale = frozenset(raw_last[raw_last.isna()].index)
    if stale:
        logger.warning(f"No fresh close on {prices.index[-1].date()} for "
                       f"{len(stale)} instrument(s) (last price carried forward): "
                       f"{sorted(stale)}")

    dollar_volume = None
    if volumes is not None:
        vol = volumes.reindex(index=prices.index, columns=prices.columns)
        # zero volume = no trade recorded -> treat as missing, not "free"
        dollar_volume = (vol.where(vol > 0) * prices).astype(float)
    return MarketData(prices=prices, local_prices=local, listings=surviving,
                      base_currency=base_currency, dollar_volume=dollar_volume,
                      stale_last=stale)
