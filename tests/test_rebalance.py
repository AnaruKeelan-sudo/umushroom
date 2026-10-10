"""When to sell: the strategy's exit rule as a trade list."""
from __future__ import annotations

from datetime import date

import pytest

from umushroom.rebalance import next_review, plan


def _p(orders, as_of="2026-10-02", top=None, sleeves=None):
    return {"meta": {"as_of": as_of}, "orders": orders, "top": top or [],
            "sleeves": sleeves or [{"name": "equity", "selection": "model"},
                                   {"name": "etf", "selection": "all"}]}


def _o(t, shares, price=10.0, sleeve="equity"):
    return {"ticker": t, "sleeve": sleeve, "shares": shares, "local_price": price,
            "listing_currency": "GBP", "name": t + " plc"}


def test_every_case_gets_the_right_action():
    held = _p([_o("DROP", 100), _o("KEEP", 100), _o("TRIM", 100), _o("TOPUP", 100)])
    latest = _p([_o("KEEP", 100), _o("TRIM", 60), _o("TOPUP", 150), _o("NEW", 50)],
                as_of="2026-10-09",
                top=[{"ticker": "NEW", "rank": 2}, {"ticker": "DROP", "rank": 17}])
    acts = {a.ticker: a for a in plan(held, latest)}
    assert acts["DROP"].action == "SELL ALL" and acts["DROP"].trade_shares == -100
    assert "#17" in acts["DROP"].reason
    assert acts["KEEP"].action == "HOLD" and acts["KEEP"].trade_shares == 0
    assert acts["TRIM"].action == "SELL" and acts["TRIM"].trade_shares == -40
    assert acts["TOPUP"].action == "BUY MORE" and acts["TOPUP"].trade_shares == 50
    assert acts["NEW"].action == "BUY NEW" and "#2" in acts["NEW"].reason


def test_sells_come_before_buys():
    held = _p([_o("A", 10), _o("B", 10)])
    latest = _p([_o("B", 20), _o("C", 10)])
    order = [a.action for a in plan(held, latest)]
    assert order == ["SELL ALL", "BUY NEW", "BUY MORE"]


def test_tiny_differences_are_held_not_traded():
    held = _p([_o("A", 100)])
    assert plan(held, _p([_o("A", 101)]))[0].action == "HOLD"     # 1% < 2% band
    assert plan(held, _p([_o("A", 105)]))[0].action == "BUY MORE"


def test_fractional_crypto_trims():
    held = _p([_o("BTC-USD", 0.150000, 60_000.0, "crypto")])
    latest = _p([_o("BTC-USD", 0.120000, 60_000.0, "crypto")])
    a = plan(held, latest)[0]
    assert a.action == "SELL" and a.trade_shares == pytest.approx(-0.03)


def test_hold_all_sleeve_explains_why_it_sells():
    held = _p([_o("SGLD.L", 5, 200.0, "etf")])
    a = plan(held, _p([]))[0]
    assert a.action == "SELL ALL" and "no longer in this sleeve" in a.reason.lower()


def test_same_picks_means_nothing_to_do():
    p = _p([_o("A", 10), _o("B", 3)])
    assert {a.action for a in plan(p, p)} == {"HOLD"}


def test_next_review_skips_weekends():
    assert next_review("2026-10-09", 5) == date(2026, 10, 16)     # Fri -> next Fri
    assert next_review("2026-10-08", 1) == date(2026, 10, 9)
    assert next_review("2026-10-09", 1) == date(2026, 10, 12)     # Fri -> Mon
