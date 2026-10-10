from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from umushroom.montecarlo import simulate
from umushroom.purged_backtest import run_purged_walk_forward
from umushroom.report import (
    SleeveReport,
    build_payload,
    render_html,
    stock_snapshot,
    write_report,
)
from umushroom.validation import decile_returns

KW = {"test_window_days": 21, "purge_days": 5, "embargo_days": 5,
      "rebalance_days": 5, "forward_return_days": 5, "top_n": 8,
      "min_train_days": 504, "model_alpha": 1.0, "random_seed": 42}


@pytest.fixture(scope="module")
def payload(planted_signal_prices):
    r = run_purged_walk_forward(planted_signal_prices, record_cross_sections=True, **KW)
    mc = simulate(r.period_returns, r.benchmark_period_returns, 8, 100_000,
                  n_paths=500)
    orders = pd.DataFrame({
        "sleeve": ["equity", "equity"], "shares": [10, 5],
        "allocated_value": [5000.0, 2500.0], "local_price": [500.0, np.nan],
        "listing_currency": ["GBP", "USD"],
        "name": ['Evil <script>alert("x")</script> PLC', "Normal Co"],
        "ret_1m": [0.05, -0.02],
    }, index=["S00", "S01"])
    top = pd.DataFrame({"rank": [1], "rank_pct": [1.0], "held": [1]}, index=["S00"])
    return build_payload(
        meta={"as_of": "2026-10-07", "generated": "now", "base_currency": "GBP",
              "version": "test"},
        sleeves=[SleeveReport("equity", 0.7, "model", r, 0.08, 2.1,
                              [("Placebo test", "PASS", "beat 20/20")])],
        orders=orders, top_ranked=top, deciles=decile_returns(r), mc=mc)


def test_payload_has_every_section(payload):
    assert {"meta", "orders", "top", "sleeves", "deciles", "mc"} <= set(payload)
    eq = payload["sleeves"][0]
    assert len(eq["curves"]["dates"]) == len(eq["curves"]["strategy"])
    assert eq["alpha"] == pytest.approx(0.08)
    assert set(payload["mc"]["bands"]) >= {"p5", "p25", "p50", "p75", "p95",
                                           "benchmark_p50"}
    assert len(payload["deciles"]) == 10


def test_payload_is_strict_json_without_nan(payload):
    text = json.dumps(payload, allow_nan=False)    # raises on NaN/inf
    assert "NaN" not in text
    assert payload["orders"][1]["local_price"] is None


def test_html_cannot_be_broken_by_data(payload):
    """A company name containing </script> must not end the data block
    (it is also only ever inserted with textContent)."""
    html = render_html(payload)
    data_block = re.search(r'<script id="report-data" type="application/json">(.*?)</script>',
                           html, re.DOTALL).group(1)
    assert "</script" not in data_block
    assert json.loads(data_block)["orders"][0]["name"].startswith("Evil <script>")
    assert "innerHTML" not in html


def test_html_is_self_contained(payload):
    html = render_html(payload)
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in html
    for section in ("Today's orders", "Top-ranked stocks", "Out-of-sample performance",
                    "Cumulative alpha", "Rolling 1-year alpha", "Monte Carlo",
                    "Validation checks"):
        assert section in html


def test_html_supports_dark_mode(payload):
    html = render_html(payload)
    assert "prefers-color-scheme: dark" in html and '[data-theme="dark"]' in html


def test_write_report(tmp_path, payload):
    p = write_report(tmp_path / "r.html", payload)
    assert p.exists() and p.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_stock_snapshot_known_values():
    idx = pd.bdate_range("2024-01-01", periods=300)
    px = pd.DataFrame({"A": np.linspace(100, 200, 300)}, index=idx)
    snap = stock_snapshot(px, ["A"])
    assert snap.loc["A", "ret_1m"] == pytest.approx(200 / px["A"].iloc[-22] - 1)
    assert snap.loc["A", "ret_12m"] == pytest.approx(200 / px["A"].iloc[-253] - 1)
    assert snap.loc["A", "vol_60d"] > 0


def test_tick_labels_never_negative_zero_and_match_step():
    from umushroom.report import _price, _tick
    assert _tick(-1e-17, "pct", 0.05) == "0%"
    assert _tick(0.0025, "pct", 0.0025) == "0.25%"
    assert _tick(0.008, "pct", 0.002) == "0.8%"
    assert _tick(-0.15, "pct", 0.05) == "-15%"
    assert _price(11030.4) == "11,030.40" and "e" not in _price(0.0000123)


def test_hold_all_sleeve_chart_shows_what_is_held(planted_signal_prices):
    """A hold-all sleeve EARNS the equal-weight basket; the model's picks are
    only a labelled reference -- never presented as what we hold."""
    r = run_purged_walk_forward(planted_signal_prices, **KW)
    p = build_payload(meta={"base_currency": "GBP"},
                      sleeves=[SleeveReport("crypto", 0.1, "all", r, 0.4, 1.8)],
                      orders=pd.DataFrame())
    c = p["sleeves"][0]["curves"]
    assert c["strategy"] == c["benchmark"] and c["model"] != c["strategy"]
    html = render_html(p)
    assert "Model&#x27;s picks (not used)" in html or "Model's picks (not used)" in html
    assert "alpha +40" not in html          # unused model's alpha not headlined


# ---- v4.11 graph-error regressions -------------------------------------

from umushroom.report import _hist, histogram, line_chart


def test_histogram_has_no_fake_edge_spikes_and_reports_hidden_tail():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(0, 1, 10_000), [500.0, -500.0] * 30])
    h = _hist(v)
    share = np.array(h["share"])
    # extremes are NOT squashed into the end bars
    assert share[0] < 0.01 and share[-1] < 0.01
    hidden = h["hidden_low"] + h["hidden_high"]
    assert 0 < hidden < 0.02
    assert share.sum() + hidden == pytest.approx(1.0)
    svg = histogram("t", "s.", h, "pct", 0.0, "m", [])
    assert "lie beyond the axis" in svg


def test_histogram_of_constant_values_does_not_crash():
    h = _hist(np.full(100, 0.05))
    assert sum(h["share"]) == pytest.approx(1.0)
    histogram("t", "s", h, "pct", 0.05, "m", [])
    histogram("t", "s", _hist(np.array([np.nan])), "pct", 0.0, "m", [])


def test_identical_series_are_merged_not_hidden():
    vals = [1.0, 1.1, 1.2]
    svg = line_chart("t", "s", ["a", "b", "c"],
                     [("Ours", vals, "--series-1"), ("Benchmark", list(vals), "--series-2")],
                     "x")
    assert "Ours = Benchmark" in svg
    assert svg.count('class="ln"') == 1
    assert svg.count('class="endlab"') == 1


def test_end_labels_never_overlap():
    svg = line_chart("t", "s", ["a", "b"],
                     [("A", [1.0, 2.0], "--series-1"), ("B", [1.0, 2.001], "--series-2")],
                     "x")
    assert svg.count('class="ln"') == 2
    assert svg.count('class="endlab"') == 1


def test_missing_benchmark_week_does_not_break_curves(planted_signal_prices):
    r = run_purged_walk_forward(planted_signal_prices, **KW)
    r.benchmark_period_returns.iloc[3] = np.nan
    p = build_payload({"as_of": "x", "base_currency": "GBP"},
                      [SleeveReport("equity", 1.0, "model", r)], pd.DataFrame())
    c = p["sleeves"][0]["curves"]
    assert all(v is not None for v in c["benchmark"])
    assert all(v is not None for v in c["cum_alpha"])
    render_html(p)


def test_short_history_shows_note_instead_of_empty_rolling_chart(planted_signal_prices):
    r = run_purged_walk_forward(planted_signal_prices, **KW)
    short = SleeveReport("equity", 1.0, "model", r)
    p = build_payload({"as_of": "x", "base_currency": "GBP"}, [short], pd.DataFrame())
    p["sleeves"][0]["curves"]["rolling_alpha"] = [None] * len(
        p["sleeves"][0]["curves"]["dates"])
    html_ = render_html(p)
    assert "rolling 1-year alpha needs at" in html_        # explained...
    assert "rolling 1-year alpha</h3>" not in html_        # ...not a blank chart


def test_worst_case_tile_has_single_sign(payload):
    html_ = render_html(payload)
    m = re.search(r"5% worst case: ([^<]*)<", html_)
    assert m and not re.search(r"[-+]{2}", m.group(1))
    assert "--" not in re.sub(r"var\(--[^)]*\)|--[a-z][\w-]*", "", html_.split("<style")[0])


def test_sleeve_acronyms_are_capitalised():
    from umushroom.report import _label
    assert _label("etf") == "ETF" and _label("equity") == "Equity"


def test_money_ticks_stay_distinct_for_small_steps():
    from umushroom.report import _tick
    assert [_tick(v, "money", 25) for v in (8700, 8725, 8750)] == ["8,700", "8,725", "8,750"]
    assert [_tick(v, "money", 500) for v in (8500, 9000)] == ["8.5k", "9.0k"]
    assert [_tick(v, "money", 10_000) for v in (90_000, 100_000)] == ["90k", "100k"]
    assert _tick(0.5, "money", 0.25) == "0.50"


# ---- v4.14 Value at Risk -----------------------------------------------

def test_payload_has_var_and_risk_section(payload):
    risk = payload["risk"]
    wk, hz = risk["weekly"], risk["horizon"]
    assert 0 < wk["var95"] <= wk["es95"] and wk["var95"] <= wk["var99"] <= wk["es99"]
    assert hz["weeks"] == 8 and hz["es95"] >= hz["var95"]
    assert risk["value"] == 100_000
    assert risk["backtest"]["status"] in ("PASS", "WARN", "FAIL")
    assert len(risk["backtest"]["dates"]) == len(risk["backtest"]["limit"])
    html_ = render_html(payload)
    assert "Value at Risk" in html_ and "VaR backtest" in html_
    assert "Value at Risk (95%, 1 week)" in html_
    json.dumps(payload, allow_nan=False)


def test_hold_all_sleeve_var_uses_what_it_holds(planted_signal_prices):
    r = run_purged_walk_forward(planted_signal_prices, **KW)
    p_model = build_payload({}, [SleeveReport("equity", 1.0, "model", r)], pd.DataFrame())
    p_all = build_payload({}, [SleeveReport("etf", 1.0, "all", r)], pd.DataFrame())
    from umushroom.risk import var_es
    assert p_model["risk"]["weekly"]["var95"] == pytest.approx(var_es(r.period_returns)[0])
    assert p_all["risk"]["weekly"]["var95"] == pytest.approx(
        var_es(r.benchmark_period_returns)[0])


def test_old_reports_without_risk_still_render(payload):
    old = {k: v for k, v in payload.items() if k != "risk"}
    html_ = render_html(old)
    assert "Not enough out-of-sample history" in html_


def test_risk_histogram_never_hides_the_worst_weeks(payload):
    h = payload["risk"]["weekly"]["hist"]
    assert h["hidden_low"] == 0 and h["hidden_high"] == 0
    assert h["edges"][0] == pytest.approx(-payload["risk"]["weekly"]["worst"])
