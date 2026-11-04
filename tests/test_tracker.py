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


# ------------------------------------------------- v4.21: books and the record

def test_until_caps_the_path_at_the_next_books_close():
    closes = pd.DataFrame({"A": [100.0, 100.0, 120.0, 90.0]}, index=DAYS)
    res = track(_payload([{"ticker": "A", "local_price": 100.0, "allocated_value": 100.0}]),
                _fetch(closes), until="2026-10-05")
    assert res.positions[0].change == pytest.approx(0.20)       # not the later 90
    assert res.positions[0].last_date == "2026-10-05" and res.until == "2026-10-05"
    assert res.newer and res.up == 1


def test_no_close_after_the_prediction_is_not_a_result():
    """The morning a run is published, its latest close IS the prediction
    close: every change is 0.00% -- the site must not call that 0 of N up."""
    closes = pd.DataFrame({"A": [100.0, 110.0]}, index=DAYS[:2])
    res = track(_payload([{"ticker": "A", "local_price": 110.0, "allocated_value": 100.0}]),
                _fetch(closes))
    assert res.positions[0].change == 0.0 and res.up == 0
    assert res.last_date == "2026-10-02" and not res.newer


def test_record_scores_each_book_to_the_next_one_against_the_benchmark():
    from umushroom.tracker import record
    closes = pd.DataFrame({"A": [100.0, 100.0, 110.0, 121.0], "B": [50.0, 50.0, 40.0, 60.0],
                           "IWDA.L": [10.0, 10.0, 10.5, 10.5]}, index=DAYS)
    f = _fetch(closes)
    week1 = _payload([{"ticker": "A", "local_price": 100.0, "allocated_value": 300.0},
                      {"ticker": "B", "local_price": 50.0, "allocated_value": 100.0}],
                     as_of="2026-10-02")
    week2 = _payload([{"ticker": "A", "local_price": 110.0, "allocated_value": 100.0}],
                     as_of="2026-10-05")
    rows = record([("r1", week1, "2026-10-05"), ("r2", week2, None)], f)
    # one download -- v4.25: with the benchmark's FX pair (IWDA.L is the USD line)
    assert len(f.calls) == 1 and f.calls[0][0] == ("A", "B", "IWDA.L", "USDGBP=X")
    r1, r2 = rows
    assert r1.change == pytest.approx((300 * 1.1 + 100 * 0.8) / 400 - 1)  # +2.5%
    assert r1.up == 1 and r1.n == 2 and r1.until == "2026-10-05"
    assert r1.benchmark == pytest.approx(0.05) and r1.active == pytest.approx(0.025 - 0.05)
    assert r2.change == pytest.approx(0.10) and r2.until is None
    assert r2.benchmark == pytest.approx(0.0) and r2.last_date == "2026-10-06"
    assert record([], f) == []
    assert not r1.fx       # no USDGBP=X in this frame: the page must say so


# --------------------------------------------- v4.25: values in GBP, with FX

def test_a_usd_holding_is_valued_in_gbp_with_the_currency_move():
    """The stock is flat in dollars while the dollar gains 10% on the pound:
    the GBP value (what the UMushroom account shows) is up 10%; the price on
    the page stays the dollar price."""
    closes = pd.DataFrame({"AAPL": [200.0, 200.0, 200.0, 200.0],
                           "USDGBP=X": [0.75, 0.75, 0.80, 0.825]}, index=DAYS)
    f = _fetch(closes)
    res = track(_payload([{"ticker": "AAPL", "local_price": 200.0, "shares": 10,
                           "listing_currency": "USD", "allocated_value": 1500.0}]), f)
    p = res.positions[0]
    assert f.calls[0][0] == ("AAPL", "USDGBP=X")
    assert p.change == pytest.approx(0.10) and p.local_change == pytest.approx(0.0)
    assert p.value_now == pytest.approx(1650.0) and p.price_now == pytest.approx(200.0)
    assert p.fx and res.fx
    assert res.portfolio_path().iloc[-1] == pytest.approx(1650.0)


def test_fx_rate_on_a_day_the_stock_did_not_trade_is_carried_forward():
    days = pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"])
    closes = pd.DataFrame({"NESN.SW": [100.0, 100.0, None, 110.0],
                           "CHFGBP=X": [0.90, 0.90, 0.99, None]}, index=days)
    res = track(_payload([{"ticker": "NESN.SW", "local_price": 100.0,
                           "listing_currency": "CHF", "allocated_value": 900.0}]),
                _fetch(closes))
    # 10/06: price +10% in CHF, the rate's last print (10/05) +10% -> +21% in GBP
    assert res.positions[0].change == pytest.approx(1.1 * 1.1 - 1)


def test_missing_fx_pair_falls_back_to_the_local_change_and_says_so():
    from umushroom.web import tracker_body
    closes = pd.DataFrame({"AAPL": [200.0, 200.0, 210.0, 220.0],
                           "AZN.L": [100.0, 100.0, 101.0, 102.0]}, index=DAYS)
    res = track(_payload([
        {"ticker": "AAPL", "local_price": 200.0, "listing_currency": "USD",
         "allocated_value": 1500.0},
        {"ticker": "AZN.L", "local_price": 1.0, "listing_currency": "GBP",
         "allocated_value": 100.0}]), _fetch(closes))
    aapl, azn = res.positions
    assert aapl.change == pytest.approx(0.10) and not aapl.fx
    assert azn.fx and not res.fx                   # GBP needs no rate
    html = tracker_body(res, "GBP")
    assert "no exchange rate was found for AAPL" in html
    good = track(_payload([{"ticker": "AZN.L", "local_price": 1.0,
                            "listing_currency": "GBP", "allocated_value": 100.0}]),
                 _fetch(closes))
    assert "currency moves included" in tracker_body(good, "GBP")


def test_record_scores_book_and_benchmark_in_gbp():
    from umushroom.tracker import record
    from umushroom.web import record_body
    closes = pd.DataFrame({"AZN.L": [100.0, 100.0, 100.0, 100.0],
                           "IWDA.L": [10.0, 10.0, 10.0, 10.0],
                           "USDGBP=X": [0.80, 0.80, 0.78, 0.76]}, index=DAYS)
    book = _payload([{"ticker": "AZN.L", "local_price": 1.0, "listing_currency": "GBP",
                      "allocated_value": 100.0}], as_of="2026-10-02")
    (row,) = record([("20261002-080000", book, "2026-10-06")], _fetch(closes))
    assert row.change == pytest.approx(0.0)
    assert row.benchmark == pytest.approx(0.76 / 0.80 - 1)        # -5% in pounds
    assert row.active == pytest.approx(0.05) and row.fx
    assert "in GBP, currency moves included" in record_body([row], 5, ccy="GBP")


def test_fx_pair_names():
    from umushroom.tracker import fx_pair
    assert fx_pair("USD", "GBP") == "USDGBP=X" and fx_pair("chf", "GBP") == "CHFGBP=X"
    assert fx_pair("GBP", "GBP") is None and fx_pair("", "GBP") is None
    assert fx_pair("GBp", "GBP") is None and fx_pair("GBp", "USD") == "GBPUSD=X"
