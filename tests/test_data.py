from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from umushroom import data as data_mod
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


def test_get_market_data_drops_instruments_with_missing_fx(tmp_path):
    """A missing FX pair (Yahoo rate-limits a lone pair) must drop only the
    instruments that need it, not abort the whole run -- otherwise one niche
    AUD/NOK listing kills every pick."""
    base_fetch = _fake_fetcher()

    def no_fx(symbols, period):
        return base_fetch([s for s in symbols if not s.endswith("=X")], period)
    # AAPL is USD (needs USDGBP=X, now missing); AZN.L is GBP (needs no FX).
    md = get_market_data(["AAPL", "AZN.L"], "2y", tmp_path, 12, 70, "GBP",
                         fetcher=no_fx, drop_today_bar=False)
    assert "AZN.L" in md.prices.columns
    assert "AAPL" not in md.prices.columns


def test_get_market_data_raises_when_no_instrument_can_be_priced(tmp_path):
    """If every instrument needs a missing FX pair, there is nothing left to
    price and the run should still fail loudly."""
    base_fetch = _fake_fetcher()

    def no_fx(symbols, period):
        return base_fetch([s for s in symbols if not s.endswith("=X")], period)
    with pytest.raises(DataError):
        get_market_data(["AAPL"], "2y", tmp_path, 12, 70, "GBP",
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


# ------------------------------------------------------------ v4.19 snapshot replay

def _write_snapshot(path, gz=True):
    import gzip
    import json
    days = list(range(19000, 19000 + 400))        # 400 consecutive days from 2022-01-08
    payload = {"fetched_at": "2023-02-15T17:00:00Z",
               "meta": {"AAA.L": {"cur": "GBp", "n": 400}, "USDGBP=X": {"cur": "GBP"},
                        "BBB": {"cur": "USD", "n": 400}},
               "series": {"AAA.L": {"d": days, "c": [5000.0 + i for i in range(400)],
                                    "v": [1000] * 400},
                          "BBB": {"d": days + [days[-1]],          # a repeated last day
                                  "c": [100.0 + i * 0.1 for i in range(400)] + [140.5],
                                  "v": [None] * 401},
                          "USDGBP=X": {"d": days, "c": [0.8] * 400, "v": [None] * 400}}}
    text = json.dumps(payload)
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text)
    return path


def test_snapshot_fetcher_replays_prices_and_dedupes_days(tmp_path):
    snap = data_mod.SnapshotFetcher(_write_snapshot(tmp_path / "snap.json.gz"))
    assert snap.as_of == date(2023, 2, 15)
    closes, volumes = snap(["AAA.L", "BBB", "ZZZ", "USDGBP=X"], "10y")
    assert list(closes.columns) == ["AAA.L", "BBB", "ZZZ", "USDGBP=X"]
    assert closes["ZZZ"].isna().all()                 # unknown symbol -> NaN, like Yahoo
    assert not closes.index.duplicated().any()
    assert closes["BBB"].iloc[-1] == 140.5            # the later entry of a repeated day wins
    assert volumes is not None and volumes["AAA.L"].iloc[0] == 1000
    assert volumes["BBB"].isna().all()


def test_snapshot_env_var_drives_get_market_data(tmp_path, monkeypatch):
    _write_snapshot(tmp_path / "snap.json", gz=False)
    monkeypatch.setenv(data_mod.SNAPSHOT_ENV, str(tmp_path / "snap.json"))
    md = data_mod.get_market_data(
        tickers=["AAA.L", "BBB"], period="10y", cache_dir=tmp_path / ".cache",
        cache_max_age_hours=12, min_history_days=100, base_currency="GBP")
    # London pence converted to pounds, USD converted at the snapshot's FX
    assert md.prices["AAA.L"].iloc[0] == pytest.approx(50.0)
    assert md.prices["BBB"].iloc[0] == pytest.approx(80.0)
    # the snapshot's own date decides the partial bar: nothing dated >= 2023-02-15 survives
    assert md.prices.index.max() < pd.Timestamp("2023-02-15")
    monkeypatch.setenv(data_mod.SNAPSHOT_ENV, str(tmp_path / "missing.json"))
    with pytest.raises(DataError):
        data_mod.get_market_data(
            tickers=["AAA.L"], period="10y", cache_dir=tmp_path / ".cache",
            cache_max_age_hours=12, min_history_days=100, base_currency="GBP")


# ------------------------------------------- v4.20: rate-limit resilience
# Yahoo answers a big bulk download by serving part of it and returning the
# rest EMPTY (HTTP 429 per symbol, no exception). On 2026-10-19 that left
# 780 of 2,281 symbols as NaN columns, which were then cached for 12 hours:
# the ETF sleeve ran on 1,039 of ~1,700 instruments and 5 of 13 core ETFs.

def _flaky_download(fail_first: set[str], always_fail: set[str] = frozenset(),
                    n_days: int = 320):
    """A yfinance stand-in: symbols in `fail_first` come back empty on the
    first (bulk) call only; `always_fail` never return data. Records every
    call so tests can check what was re-fetched."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2025-01-01", periods=n_days)
    calls: list[list[str]] = []

    def download(symbols, period):
        calls.append(list(symbols))
        first = len(calls) == 1
        cols, vols = {}, {}
        for s in symbols:
            if s in always_fail or (first and s in fail_first):
                cols[s] = np.full(n_days, np.nan)
            elif s.endswith("=X"):
                cols[s] = 0.8 + rng.normal(0, 0.001, n_days).cumsum()
            else:
                level = 10000.0 if s.endswith(".L") else 100.0
                cols[s] = level * np.cumprod(1 + rng.normal(0, 0.01, n_days))
            vols[s] = np.where(np.isnan(cols[s]), np.nan, 1000.0)
        return pd.DataFrame(cols, index=idx), pd.DataFrame(vols, index=idx)
    download.calls = calls   # type: ignore[attr-defined]
    return download


def test_repair_missing_refetches_only_the_empty_symbols_in_small_batches():
    symbols = [f"S{i}" for i in range(12)]
    failed = {"S1", "S4", "S5", "S9", "S10"}
    dl = _flaky_download(fail_first=failed)
    closes, volumes = dl(symbols, "2y")
    assert set(data_mod.missing_symbols(closes, symbols)) == failed
    pauses: list[float] = []
    closes, volumes, still = data_mod.repair_missing(
        closes, volumes, symbols, "2y", download=dl, batch_size=2,
        sleep=pauses.append)
    assert still == []
    assert not closes[symbols].isna().any().any()
    assert volumes is not None and not volumes[symbols].isna().any().any()
    retries = dl.calls[1:]
    assert all(set(b) <= failed and len(b) <= 2 for b in retries)
    assert sorted(s for b in retries for s in b) == sorted(failed)   # each once
    assert pauses == [data_mod.RETRY_BACKOFF_SECONDS]                  # one round


def test_repair_missing_keeps_trying_dead_symbols_until_rounds_run_out():
    symbols = ["A", "B", "C"]
    dl = _flaky_download(fail_first=set(), always_fail={"C"})
    closes, volumes = dl(symbols, "2y")
    pauses: list[float] = []
    _c, _v, still = data_mod.repair_missing(closes, volumes, symbols, "2y",
                                            download=dl, rounds=3, sleep=pauses.append)
    assert still == ["C"]
    assert len(dl.calls) == 4                      # bulk + one batch per round
    assert pauses == [data_mod.RETRY_BACKOFF_SECONDS * r for r in (1, 2, 3)]


def test_merge_columns_replaces_empty_columns_on_the_union_of_dates():
    idx = pd.bdate_range("2025-01-01", periods=3)
    base = pd.DataFrame({"A": [1.0, 2, 3], "B": [np.nan] * 3}, index=idx)
    patch = pd.DataFrame({"B": [5.0, 6.0, 7.0]}, index=idx[1:].append(
        pd.bdate_range("2025-01-07", periods=1)))
    out = data_mod._merge_columns(base, patch)
    assert list(out.index) == list(idx) + [pd.Timestamp("2025-01-07")]
    assert out["B"].tolist()[1:] == [5.0, 6.0, 7.0] and np.isnan(out["B"].iloc[0])
    assert out["A"].tolist()[:3] == [1.0, 2.0, 3.0] and np.isnan(out["A"].iloc[-1])


def test_live_fetch_refuses_a_degraded_download_and_caches_nothing(tmp_path, monkeypatch):
    tickers = [f"T{i}" for i in range(10)] + ["AZN.L", "NESN.SW"]
    dead = {f"T{i}" for i in range(4)}                           # 4/14 symbols = 29%
    monkeypatch.setattr(data_mod, "_yf_download", _flaky_download(set(), dead))
    monkeypatch.setattr(data_mod, "_sleep", lambda s: None)
    with pytest.raises(DataError, match="rate-limiting"):
        get_market_data(tickers, "2y", tmp_path, 12, 70, "GBP", drop_today_bar=False)
    assert not list(tmp_path.glob("*.parquet"))                  # nothing cached


def test_live_fetch_repairs_a_partial_download_before_caching(tmp_path, monkeypatch):
    tickers = [f"T{i}" for i in range(10)] + ["AZN.L", "NESN.SW"]
    dl = _flaky_download(fail_first={"T2", "T7", "AZN.L", "USDGBP=X"})
    monkeypatch.setattr(data_mod, "_yf_download", dl)
    monkeypatch.setattr(data_mod, "_sleep", lambda s: None)
    md = get_market_data(tickers, "2y", tmp_path, 12, 70, "GBP", drop_today_bar=False)
    assert set(md.prices.columns) == set(tickers)
    price_cache = [f for f in tmp_path.glob("*.parquet") if "_volume" not in f.name]
    assert len(price_cache) == 1
    cached = pd.read_parquet(price_cache[0])
    assert data_mod.missing_symbols(cached, tickers + ["USDGBP=X", "CHFGBP=X"]) == []


def test_a_few_dead_symbols_are_tolerated_and_dropped(tmp_path, monkeypatch):
    tickers = [f"T{i}" for i in range(20)] + ["AZN.L"]
    monkeypatch.setattr(data_mod, "_yf_download", _flaky_download(set(), {"T19"}))
    monkeypatch.setattr(data_mod, "_sleep", lambda s: None)
    md = get_market_data(tickers, "2y", tmp_path, 12, 70, "GBP", drop_today_bar=False)
    assert "T19" not in md.prices.columns and len(md.prices.columns) == 20


def test_damaged_cache_is_repaired_in_place_on_the_next_run(tmp_path, monkeypatch):
    """The 2026-10-19 situation: a fresh cache whose rate-limited columns are
    all NaN. The next run must re-fetch just those, rewrite the cache, and
    run on the full universe."""
    tickers = ["AAPL", "MSFT", "AZN.L", "SHEL.L"]
    symbols = tickers + ["USDGBP=X"]
    good = _flaky_download(set())
    closes, volumes = good(symbols, "2y")
    closes[["MSFT", "SHEL.L"]] = np.nan                       # as the 429s left them
    volumes[["MSFT", "SHEL.L"]] = np.nan
    path = data_mod._cache_path(tmp_path, symbols, "2y")
    closes.to_parquet(path)
    volumes.to_parquet(path.with_name(path.stem + "_volume.parquet"))
    monkeypatch.setattr(data_mod, "_yf_download", good)
    monkeypatch.setattr(data_mod, "_sleep", lambda s: None)
    md = get_market_data(tickers, "2y", tmp_path, 12, 70, "GBP", drop_today_bar=False)
    assert set(md.prices.columns) == set(tickers)
    assert [c for c in good.calls[1:]] == [["MSFT", "SHEL.L"]]  # only the empty ones
    repaired = pd.read_parquet(path)
    assert data_mod.missing_symbols(repaired, symbols) == []
    assert md.dollar_volume is not None and md.dollar_volume["MSFT"].notna().all()


def test_damaged_cache_that_cannot_be_repaired_fails_loudly(tmp_path, monkeypatch):
    tickers = [f"T{i}" for i in range(10)]
    symbols = list(tickers)                                    # all USD -> no FX
    good = _flaky_download(set())
    closes, _volumes = good(symbols + ["USDGBP=X"], "2y")
    closes = closes[symbols]
    closes[[f"T{i}" for i in range(4)]] = np.nan
    closes.to_parquet(data_mod._cache_path(tmp_path, symbols, "2y"))
    monkeypatch.setattr(data_mod, "_yf_download",
                        _flaky_download(set(), {f"T{i}" for i in range(4)}))
    monkeypatch.setattr(data_mod, "_sleep", lambda s: None)
    with pytest.raises(DataError, match="cached download"):
        get_market_data(tickers, "2y", tmp_path, 12, 70, "USD", drop_today_bar=False)
    # a deliberately degraded run is still possible by raising the limit
    md = get_market_data(tickers, "2y", tmp_path, 12, 70, "USD", drop_today_bar=False,
                         max_missing_fraction=0.5)
    assert len(md.prices.columns) == 6


def test_injected_fetcher_and_snapshot_are_trusted_as_complete(tmp_path):
    """The guard is for live Yahoo data only: an injected fetcher (tests,
    research) that omits symbols keeps the old behaviour."""
    base = _fake_fetcher()

    def partial(symbols, period):
        return base([s for s in symbols if s not in ("MSFT", "NESN.SW")], period)
    md = get_market_data(["AAPL", "MSFT", "AZN.L", "NESN.SW"], "2y", tmp_path, 12, 70,
                         "GBP", fetcher=partial, drop_today_bar=False)
    assert set(md.prices.columns) == {"AAPL", "AZN.L"}



# ---------------------------------------------- v4.21: freshness, fixed start
from umushroom.data import START_PREFIX, check_freshness, last_trading_day_before
from umushroom.exceptions import StaleDataError


@pytest.mark.parametrize("today, expected", [
    (date(2026, 10, 27), date(2026, 10, 26)),     # Tuesday -> Monday
    (date(2026, 10, 26), date(2026, 10, 23)),     # Monday -> Friday
    (date(2026, 10, 24), date(2026, 10, 23)),     # Saturday -> Friday
    (date(2026, 10, 25), date(2026, 10, 23)),     # Sunday -> Friday
])
def test_last_trading_day_before(today, expected):
    assert last_trading_day_before(today) == expected


def test_check_freshness_accepts_yesterday_and_refuses_older():
    check_freshness(date(2026, 10, 26), date(2026, 10, 27))
    check_freshness(date(2026, 10, 23), date(2026, 10, 26))       # Monday run
    with pytest.raises(StaleDataError, match="2026-10-26 has already closed"):
        check_freshness(date(2026, 10, 23), date(2026, 10, 27))   # the 27 Oct live run
    with pytest.raises(StaleDataError):
        check_freshness(date(2026, 10, 22), date(2026, 10, 26))


def test_get_market_data_refuses_stale_data_only_when_asked(tmp_path):
    md = get_market_data(["AAPL", "MSFT"], "2y", tmp_path, 12, 70, "USD",
                         fetcher=_fake_fetcher(), drop_today_bar=False)
    last = md.prices.index[-1].date()
    day_after = np.busday_offset(np.datetime64(last), 1, roll="forward").astype(date)
    # fresh: the data reaches the last trading day before "today"
    get_market_data(["AAPL", "MSFT"], "2y", tmp_path, 12, 70, "USD", fetcher=_fake_fetcher(),
                    drop_today_bar=False, require_fresh=True, today=day_after)
    # stale: refused -- but only when asked (library default stays permissive)
    with pytest.raises(StaleDataError):
        get_market_data(["AAPL", "MSFT"], "2y", tmp_path, 12, 70, "USD",
                        fetcher=_fake_fetcher(), drop_today_bar=False, require_fresh=True,
                        today=date(2030, 1, 7))
    get_market_data(["AAPL", "MSFT"], "2y", tmp_path, 12, 70, "USD", fetcher=_fake_fetcher(),
                    drop_today_bar=False, today=date(2030, 1, 7))


def test_fixed_start_period_is_passed_to_yahoo_and_keyed_in_the_cache(tmp_path, monkeypatch):
    import sys
    import types
    calls = []
    idx = pd.bdate_range("2016-10-03", periods=5)

    def download(symbols, **kw):
        calls.append(kw)
        cols = pd.MultiIndex.from_product([["Close", "Volume"], list(symbols)])
        return pd.DataFrame(1.0, index=idx, columns=cols)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    closes, volumes = data_mod._yf_download(["A", "B"], START_PREFIX + "2016-10-01")
    assert calls[-1]["start"] == "2016-10-01" and "period" not in calls[-1]
    assert list(closes.columns) == ["A", "B"] and volumes is not None
    data_mod._yf_download(["A"], "10y")
    assert calls[-1]["period"] == "10y" and "start" not in calls[-1]
    rolling = data_mod._cache_path(tmp_path, ["A", "B"], "10y")
    fixed = data_mod._cache_path(tmp_path, ["A", "B"], START_PREFIX + "2016-10-01")
    assert rolling != fixed
