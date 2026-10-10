"""Since-prediction tracker: known-answer tests with fake Yahoo data."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from umushroom.tracker import track


def _payload(orders, as_of="2026-10-02"):
    return {"meta": {"as_of": as_of, "base_currency": "GBP"}, "orders": orders}


def _fetch(frame):
    calls = []

    def f(symbols, start):
        calls.append((tuple(symbols), start))
        return frame[[s for s in symbols if s in frame.columns]]
    f.calls = calls
    return f


DAYS = pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"])


def test_change_is_right_even_when_yahoo_quotes_pence():
    # Yahoo: 1000p -> 1100p ; the order saw the price in pounds (10.00)
    closes = pd.DataFrame({"AAA.L": [990.0, 1000.0, 1050.0, 1100.0]}, index=DAYS)
    res = track(_payload([{"ticker": "AAA.L", "sleeve": "equity", "shares": 100,
                           "local_price": 10.0, "listing_currency": "GBP",
                           "allocated_value": 1000.0}]), _fetch(closes))
    p = res.positions[0]
    assert p.change == pytest.approx(0.10)
    assert p.price_now == pytest.approx(11.0)
    assert p.value_now == pytest.approx(1100.0)
    assert p.last_date == "2026-10-06"
    assert res.change == pytest.approx(0.10)


def test_weekend_prediction_uses_last_close_before_it():
    closes = pd.DataFrame({"B": [100.0, 110.0, 121.0, 121.0]}, index=DAYS)
    res = track(_payload([{"ticker": "B", "local_price": 110.0,
                           "allocated_value": 110.0}], as_of="2026-10-04"), _fetch(closes))
    assert res.positions[0].change == pytest.approx(0.10)     # 110 (Fri) -> 121


def test_missing_price_is_reported_not_counted():
    closes = pd.DataFrame({"A": [1.0, 1.0, 2.0, 2.0]}, index=DAYS)
    res = track(_payload([
        {"ticker": "A", "local_price": 1.0, "allocated_value": 100.0},
        {"ticker": "GONE", "local_price": 5.0, "allocated_value": 500.0}]), _fetch(closes))
    gone = next(p for p in res.positions if p.ticker == "GONE")
    assert gone.change is None and gone.price_now is None and gone.value_now is None
    assert res.value_then == 100.0 and res.value_now == pytest.approx(200.0)
    assert len(res.priced) == 1


def test_substitute_symbol_is_used():
    closes = pd.DataFrame({"CRH": [50.0, 50.0, 55.0, 60.0]}, index=DAYS)
    f = _fetch(closes)
    res = track(_payload([{"ticker": "CRH.L", "priced_from": "CRH", "local_price": 50.0,
                           "allocated_value": 500.0}]), f)
    assert res.positions[0].change == pytest.approx(0.2)
    assert f.calls[0][0] == ("CRH",)
    assert f.calls[0][1] == date(2026, 9, 22)                 # 10 days of margin


def test_portfolio_path_is_value_weighted():
    closes = pd.DataFrame({"A": [1.0, 1.0, 1.5, 2.0], "B": [1.0, 1.0, 1.0, 0.5]},
                          index=DAYS)
    res = track(_payload([{"ticker": "A", "local_price": 1.0, "allocated_value": 100.0},
                          {"ticker": "B", "local_price": 1.0, "allocated_value": 300.0}]),
                _fetch(closes))
    path = res.portfolio_path()
    assert path.iloc[0] == pytest.approx(400.0)
    assert path.iloc[-1] == pytest.approx(100 * 2.0 + 300 * 0.5)


def test_no_orders_and_no_data():
    assert track(_payload([]), _fetch(pd.DataFrame())).positions == []
    res = track(_payload([{"ticker": "A", "local_price": 1.0, "allocated_value": 1.0}]),
                _fetch(pd.DataFrame()))
    assert res.priced == [] and res.change is None
