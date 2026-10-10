"""Macro exposures: known-answer tests on synthetic data."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from umushroom.macro import FACTORS, analyse, talking_points

DAYS = pd.bdate_range("2024-09-02", "2026-10-09")


def _closes(seed=0):
    """Factor price paths + two stocks with KNOWN exposures (daily data,
    constructed from weekly shocks so the weekly betas are exact-ish)."""
    rng = np.random.default_rng(seed)
    n = len(DAYS)
    rets = {f.symbol: rng.normal(0, 0.01, n) for f in FACTORS if f.kind == "ret"}
    lv = {"^TNX": 4 + np.cumsum(rng.normal(0, 0.03, n)),
          "^VIX": 18 + np.cumsum(rng.normal(0, 0.5, n)).clip(-8, 30)}
    px = {s: 100 * np.cumprod(1 + r) for s, r in rets.items()}
    m, oil = rets["ACWI"], rets["BZ=F"]
    # A: market beta 1.2, oil beta 0.5 ;  B: market beta 0.3, nothing else
    px["AAA.L"] = 1000 * np.cumprod(1 + 1.2 * m + 0.5 * oil + rng.normal(0, 0.002, n))
    px["BBB"] = 50 * np.cumprod(1 + 0.3 * m + rng.normal(0, 0.002, n))
    return pd.DataFrame({**px, **lv}, index=DAYS)


def _fetch(frame):
    def f(symbols, start):
        return frame[[s for s in symbols if s in frame.columns]]
    return f


PAYLOAD = {"meta": {"as_of": "2026-10-09"}, "orders": [
    {"ticker": "AAA.L", "sleeve": "equity", "allocated_value": 6000.0,
     "listing_currency": "GBP"},
    {"ticker": "BBB", "sleeve": "equity", "allocated_value": 4000.0,
     "listing_currency": "USD"}]}


def test_recovers_known_betas():
    res = analyse(PAYLOAD, _fetch(_closes()), today=date(2026, 10, 9))
    a = next(h for h in res.holdings if h.ticker == "AAA.L")
    b = next(h for h in res.holdings if h.ticker == "BBB")
    assert a.betas["market"] == pytest.approx(1.2, abs=0.1)
    assert a.betas["oil"] == pytest.approx(0.5, abs=0.1)
    assert abs(a.betas["gold"]) < 0.1                     # no real exposure
    assert b.betas["market"] == pytest.approx(0.3, abs=0.1)
    assert res.portfolio["market"] == pytest.approx(0.6 * 1.2 + 0.4 * 0.3, abs=0.08)
    assert res.coverage == pytest.approx(1.0)


def test_scenarios_and_currency_translation():
    res = analyse(PAYLOAD, _fetch(_closes()), today=date(2026, 10, 9))
    sc = {s["key"]: s for s in res.scenarios}
    assert sc["market"]["impact"] == pytest.approx(-0.10 * res.portfolio["market"])
    assert sc["oil"]["impact"] == pytest.approx(0.20 * res.portfolio["oil"])
    # GBP +5%: 40% in USD loses about 4.76% in GBP terms
    fx = 0.4 * (-0.05 / 1.05)
    assert sc["gbp"]["impact"] == pytest.approx(0.05 * res.portfolio["gbp"] + fx)
    assert "USD holdings" in sc["gbp"]["note"]
    assert res.currency_mix == {"GBP": pytest.approx(0.6), "USD": pytest.approx(0.4)}


def test_dashboard_and_talking_points():
    res = analyse(PAYLOAD, _fetch(_closes()), today=date(2026, 10, 9))
    keys = {r["key"] for r in res.dashboard}
    assert {"market", "us10y", "oil", "vix"} <= keys
    us = next(r for r in res.dashboard if r["key"] == "us10y")
    assert 0 <= us["pct_1y"] <= 1 and us["unit"] == "pp"
    pts = talking_points(res)
    assert any("Market beta" in p for p in pts)
    assert any("Brent oil" in p for p in pts)                # biggest real exposure
    assert any("40% of the money is in USD" in p for p in pts)


def test_holding_without_history_is_left_out_not_guessed():
    frame = _closes()
    frame.loc[frame.index[:-100], "BBB"] = np.nan             # only ~20 weeks
    res = analyse(PAYLOAD, _fetch(frame), today=date(2026, 10, 9))
    b = next(h for h in res.holdings if h.ticker == "BBB")
    assert b.betas == {}
    assert res.coverage == pytest.approx(0.6)


def test_old_x10_yield_quote_is_rescaled():
    frame = _closes()
    frame["^TNX"] = frame["^TNX"] * 10
    res = analyse(PAYLOAD, _fetch(frame), today=date(2026, 10, 9))
    us = next(r for r in res.dashboard if r["key"] == "us10y")
    assert us["level"] < 20
