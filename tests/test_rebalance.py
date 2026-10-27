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



# ------------------------------------------------------- v4.21 weekly cadence
from umushroom.rebalance import (
    cadence,
    next_rebalance,
    period_index,
    period_start,
    trade_day,
)


def test_trade_day_is_the_next_trading_day():
    assert trade_day("2026-10-23") == date(2026, 10, 26)          # Friday -> Monday
    assert trade_day("2026-10-19") == date(2026, 10, 20)
    assert trade_day(date(2026, 10, 24)) == date(2026, 10, 26)    # Saturday -> Monday


def test_periods_are_calendar_weeks_for_five_days():
    week = [date(2026, 10, 26) + __import__("datetime").timedelta(days=i) for i in range(5)]
    assert len({period_index(d, 5) for d in week}) == 1
    assert period_index(date(2026, 11, 2), 5) == period_index(date(2026, 10, 30), 5) + 1
    assert period_start(date(2026, 10, 28), 5) == date(2026, 10, 26)
    with pytest.raises(ValueError):
        period_index(date(2026, 10, 28), 0)


def test_next_rebalance_is_the_first_day_of_the_next_period():
    assert next_rebalance("2026-10-23", 5) == date(2026, 11, 2)   # Fri signals -> Mon book
    assert next_rebalance("2026-10-19", 5) == date(2026, 10, 26)  # Mon signals -> Tue..Fri
    assert next_rebalance("2026-10-22", 5) == date(2026, 10, 26)


def test_cadence_rebalances_on_the_first_run_of_a_period_and_holds_after():
    runs = [("20261019-221955", "2026-10-16"), ("20261020-080050", "2026-10-19"),
            ("20261021-080257", "2026-10-19"), ("20261022-080441", "2026-10-20"),
            ("20261023-080204", "2026-10-21"), ("20261026-070800", "2026-10-23"),
            ("20261027-070313", "2026-10-23")]
    # 10-19 evening run: signals of Fri 10-16 -> trades Mon 10-19: the first (only)
    # run of that week, and nothing earlier to sell
    c = cadence(runs, "20261019-221955", 5)
    assert c.rebalance_now and c.held is None
    # 10-20 run: signals of Mon 10-19 -> trades Tue 10-20 (same week): HOLD the
    # 10-19 book -- the site must not order a new book every morning
    c = cadence(runs, "20261020-080050", 5)
    assert not c.rebalance_now and c.held == "20261019-221955"
    assert c.next_rebalance == date(2026, 10, 26)
    for stamp in ("20261021-080257", "20261022-080441", "20261023-080204"):
        assert cadence(runs, stamp, 5).held == "20261019-221955"
    # Monday 10-26 (signals of Fri 10-23): new period -> rebalance against last week's book
    c = cadence(runs, "20261026-070800", 5)
    assert c.rebalance_now and c.held == "20261019-221955"
    assert c.period_start == date(2026, 10, 26) and c.next_rebalance == date(2026, 11, 2)
    # Tuesday 10-27: hold Monday's book
    c = cadence(runs, "20261027-070313", 5)
    assert not c.rebalance_now and c.held == "20261026-070800"


def test_cadence_uses_the_first_successful_run_when_monday_failed():
    runs = [("20261019-221955", "2026-10-16"), ("20261027-090000", "2026-10-26"),
            ("20261028-090000", "2026-10-27")]
    c = cadence(runs, "20261027-090000", 5)          # Tuesday is this week's first run
    assert c.rebalance_now and c.held == "20261019-221955"
    assert cadence(runs, "20261028-090000", 5).held == "20261027-090000"
    with pytest.raises(ValueError):
        cadence(runs, "20261030-000000", 5)
