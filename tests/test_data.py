from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umushroom.data import _validate_prices
from umushroom.exceptions import DataError, InsufficientHistoryError


def test_validate_passes_clean_data(synthetic_prices):
    out = _validate_prices(synthetic_prices, list(synthetic_prices.columns),
                            min_history_days=70)
    assert not out.empty
    assert not out.isna().any().any()


def test_validate_raises_on_empty_frame():
    with pytest.raises(DataError):
        _validate_prices(pd.DataFrame(), ["AAA"], min_history_days=10)


def test_validate_raises_on_insufficient_rows():
    small = pd.DataFrame({"AAA": [1.0, 1.01, 1.02]})
    with pytest.raises(InsufficientHistoryError):
        _validate_prices(small, ["AAA"], min_history_days=70)


def test_validate_drops_fully_missing_tickers(synthetic_prices):
    tainted = synthetic_prices.copy()
    tainted["GHOST"] = np.nan
    out = _validate_prices(tainted, list(tainted.columns), min_history_days=70)
    assert "GHOST" not in out.columns
    # The 8 good tickers should all survive
    assert out.shape[1] == synthetic_prices.shape[1]


def test_validate_raises_when_losing_more_than_half_universe(synthetic_prices):
    tainted = synthetic_prices.copy()
    # Null out 6 of 8 tickers entirely -> only 2/8 survive -> should raise
    for col in list(tainted.columns)[:6]:
        tainted[col] = np.nan
    with pytest.raises(DataError):
        _validate_prices(tainted, list(tainted.columns), min_history_days=70)


def test_validate_forward_fills_short_gaps(synthetic_prices):
    gappy = synthetic_prices.copy()
    gappy.iloc[50, 0] = np.nan  # single-day gap
    out = _validate_prices(gappy, list(gappy.columns), min_history_days=70)
    assert not out.isna().any().any()


# ---------------------------------------------------------------- v4 tests
from datetime import date

from umushroom.data import (
    convert_to_base_currency,
    drop_incomplete_last_bar,
    get_market_data,
)
from umushroom.universe import resolve_universe


def test_short_history_ticker_is_dropped_not_truncating_universe(synthetic_prices):
    """v3 bug: one late-listed ticker truncated EVERY ticker's history
    via row-wise dropna. v4 drops the ticker instead."""
    tainted = synthetic_prices.copy()
    tainted.iloc[:300, 0] = np.nan          # TICK0 listed half-way through
    out = _validate_prices(tainted, list(tainted.columns), min_history_days=70)
    assert "TICK0" not in out.columns
    assert len(out) == len(synthetic_prices)


def test_drop_incomplete_last_bar():
    idx = pd.bdate_range("2026-10-05", periods=4)       # Mon..Thu
    df = pd.DataFrame({"A": [1.0, 2, 3, 4]}, index=idx)
    out = drop_incomplete_last_bar(df, today=date(2026, 10, 8))
    assert out.index.max() == pd.Timestamp("2026-10-07")
    assert len(drop_incomplete_last_bar(df, today=date(2026, 10, 10))) == 4


def test_convert_to_base_currency_handles_pence_and_fx():
    idx = pd.bdate_range("2026-01-05", periods=3)
    quoted = pd.DataFrame({"AZN.L": [10000.0, 10100, 10200],   # pence
                           "AAPL": [200.0, 202, 204]}, index=idx)
    fx = pd.DataFrame({"USDGBP=X": [0.8, 0.8, 0.75]}, index=idx)
    listings = resolve_universe(["AZN.L", "AAPL"])
    base, local = convert_to_base_currency(quoted, fx, listings, "GBP")
    assert base["AZN.L"].iloc[0] == pytest.approx(100.0)      # 10000p = GBP100
    assert local["AZN.L"].iloc[0] == pytest.approx(100.0)
    assert base["AAPL"].iloc[2] == pytest.approx(204 * 0.75)
    assert local["AAPL"].iloc[2] == pytest.approx(204.0)


def test_convert_never_backfills_fx():
    idx = pd.bdate_range("2026-01-05", periods=3)
    quoted = pd.DataFrame({"AAPL": [1.0, 1.0, 1.0]}, index=idx)
    fx = pd.DataFrame({"USDGBP=X": [np.nan, 0.8, np.nan]}, index=idx)
    base, _ = convert_to_base_currency(quoted, fx, resolve_universe(["AAPL"]), "GBP")
    assert list(base.index) == list(idx[1:])                   # row 0 dropped
    assert base["AAPL"].iloc[-1] == pytest.approx(0.8)          # forward-filled


def _fake_fetcher(n_days=320):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2025-01-01", periods=n_days)

    def fetch(symbols, period):
        cols = {}
        for s in symbols:
            if s.endswith("=X"):
                cols[s] = 0.8 + rng.normal(0, 0.001, n_days).cumsum()
            else:
                level = 10000.0 if s.endswith(".L") else 100.0
                cols[s] = level * np.cumprod(1 + rng.normal(0, 0.01, n_days))
        return pd.DataFrame(cols, index=idx)
    return fetch


def test_get_market_data_end_to_end_offline(tmp_path):
    tickers = ["AAPL", "MSFT", "AZN.L", "SHEL.L", "NESN.SW"]
    md = get_market_data(tickers, "2y", tmp_path, 12, 70, base_currency="GBP",
                         fetcher=_fake_fetcher(), drop_today_bar=False)
    assert list(md.prices.columns) == tickers
    assert set(md.listings) == set(tickers)
    # London pence converted to pounds; base == local for GBP listings
    assert md.prices["AZN.L"].iloc[0] < 1000
    pd.testing.assert_series_equal(md.prices["AZN.L"], md.local_prices["AZN.L"])
    assert not md.prices.isna().any().any()


def test_get_market_data_raises_when_fx_missing(tmp_path):
    base_fetch = _fake_fetcher()

    def no_fx(symbols, period):
        return base_fetch([s for s in symbols if not s.endswith("=X")], period)
    with pytest.raises(DataError):
        get_market_data(["AAPL", "AZN.L"], "2y", tmp_path, 12, 70, "GBP",
                        fetcher=no_fx, drop_today_bar=False)


def test_cache_path_unique_for_long_universes(tmp_path):
    """v3 bug: keys were truncated to 150 chars, so large universes that
    differ only late in sorted order collided on one cache file."""
    from umushroom.data import _cache_path
    base = [f"TICKER{i:03d}.L" for i in range(60)]
    a = _cache_path(tmp_path, base + ["ZZZ1"], "5y")
    b = _cache_path(tmp_path, base + ["ZZZ2"], "5y")
    assert a != b
    assert _cache_path(tmp_path, list(reversed(base)), "5y") == _cache_path(
        tmp_path, base, "5y")


def test_align_to_trading_calendar_drops_weekends_keeps_crypto_moves():
    from umushroom.data import align_to_trading_calendar
    idx = pd.date_range("2026-01-02", periods=5, freq="D")   # Fri..Tue
    df = pd.DataFrame({"STK": [10.0, np.nan, np.nan, 11.0, 12.0],
                       "BTC-USD": [100.0, 105.0, 90.0, 95.0, 96.0]}, index=idx)
    out = align_to_trading_calendar(df, ["STK"])
    assert list(out.index.dayofweek) == [4, 0, 1]             # Fri, Mon, Tue
    # Monday's crypto return includes the weekend: 95 / 100 - 1
    assert out["BTC-USD"].pct_change().iloc[1] == pytest.approx(-0.05)


def test_pence_pound_flip_ticker_is_dropped_but_crypto_spike_kept(synthetic_prices):
    """v4.3, from the first real run: Yahoo sometimes reports a London
    price in pounds instead of pence for a few days (a 100x jump). Such a
    ticker must be dropped; a genuine big crypto move must not be."""
    data = synthetic_prices.copy()
    data.iloc[200:203, 0] /= 100              # TICK0: pence -> pounds flip
    data["BTC-USD"] = synthetic_prices.iloc[:, 1].to_numpy()
    data.iloc[300:, -1] *= 2.5                # +150% day: real crypto move
    out = _validate_prices(data, list(data.columns), 70,
                           spike_exempt=frozenset({"BTC-USD"}))
    assert "TICK0" not in out.columns
    assert "BTC-USD" in out.columns
    assert len(out) == len(data)


def test_convert_to_base_currency_does_not_fragment(recwarn):
    idx = pd.bdate_range("2026-01-05", periods=5)
    tickers = [f"T{i}" for i in range(300)]
    quoted = pd.DataFrame(1.0, index=idx, columns=tickers)
    fx = pd.DataFrame({"USDGBP=X": 0.8}, index=idx)
    convert_to_base_currency(quoted, fx, resolve_universe(tickers), "GBP")
    assert not [w for w in recwarn if "fragmented" in str(w.message)]


def test_ragged_validation_keeps_late_listing_with_leading_nan(synthetic_prices):
    data = synthetic_prices.copy()
    data.iloc[:300, 0] = np.nan                    # listed half-way through
    data.iloc[:550, 1] = np.nan                    # only 50 days: too short
    out = _validate_prices(data, list(data.columns), 70, ragged=True)
    assert "TICK0" in out.columns and out["TICK0"].iloc[:300].isna().all()
    assert "TICK1" not in out.columns
    assert len(out) == len(data)                   # nobody else truncated


def test_get_market_data_returns_dollar_volume(tmp_path):
    base = _fake_fetcher()

    def with_volume(symbols, period):
        closes = base(symbols, period)
        vol = pd.DataFrame(1000.0, index=closes.index, columns=closes.columns)
        vol.iloc[5, 0] = 0.0                       # a zero-volume day
        return closes, vol
    md = get_market_data(["AAPL", "AZN.L"], "2y", tmp_path, 12, 70, "GBP",
                         fetcher=with_volume, drop_today_bar=False)
    assert md.dollar_volume is not None
    pd.testing.assert_series_equal(
        md.dollar_volume["AZN.L"].iloc[6:], (md.prices["AZN.L"] * 1000).iloc[6:],
        check_names=False)
    assert np.isnan(md.dollar_volume.iloc[5, 0])   # zero volume -> missing


def test_stale_last_price_is_flagged(tmp_path):
    """v4.6: if the source has no close for the latest day for a ticker, its
    price is carried forward -- it must be flagged so it isn't traded."""
    base = _fake_fetcher()

    def fetch(symbols, period):
        df = base(symbols, period)
        df.iloc[-1, df.columns.get_loc("AZN.L")] = np.nan     # no close today
        return df
    md = get_market_data(["AAPL", "MSFT", "AZN.L"], "2y", tmp_path, 12, 70, "GBP",
                         fetcher=fetch, drop_today_bar=False)
    assert md.stale_last == frozenset({"AZN.L"})
    assert md.prices["AZN.L"].iloc[-1] == md.prices["AZN.L"].iloc[-2]   # ffilled


def test_yahoo_symbols_fetch_under_other_symbol_but_keep_ticker(tmp_path):
    """v4.7: CRH.L is fetched as Yahoo 'CRH' (US line, USD) but stays CRH.L."""
    seen = {}
    base = _fake_fetcher()

    def fetch(symbols, period):
        seen["symbols"] = list(symbols)
        return base(symbols, period)
    md = get_market_data(["AAPL", "CRH.L"], "2y", tmp_path, 12, 70, "GBP",
                         currency_overrides={"CRH.L": "USD"},
                         yahoo_symbols={"CRH.L": "CRH"},
                         fetcher=fetch, drop_today_bar=False)
    assert "CRH" in seen["symbols"] and "CRH.L" not in seen["symbols"]
    assert "CRH.L" in md.prices.columns
    assert md.listings["CRH.L"].currency == "USD"
