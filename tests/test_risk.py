"""VaR / expected shortfall: known answers, Kupiec test, VaR backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umushroom.montecarlo import simulate
from umushroom.risk import backtest_var, kupiec, var_es


def test_normal_returns_match_textbook_var_and_es():
    r = np.random.default_rng(0).normal(0, 0.02, 200_000)
    v95, e95 = var_es(r, 0.95)
    v99, e99 = var_es(r, 0.99)
    assert v95 == pytest.approx(1.645 * 0.02, rel=0.02)
    assert e95 == pytest.approx(2.063 * 0.02, rel=0.02)
    assert v99 == pytest.approx(2.326 * 0.02, rel=0.03)
    assert e99 == pytest.approx(2.665 * 0.02, rel=0.03)


def test_es_is_never_smaller_than_var_and_99_is_beyond_95():
    r = np.random.default_rng(1).standard_t(3, 500) * 0.02
    v95, e95 = var_es(r, 0.95)
    v99, e99 = var_es(r, 0.99)
    assert e95 >= v95 and e99 >= v99 >= v95


def test_var_es_ignores_nan_and_rejects_bad_input():
    r = np.r_[np.random.default_rng(2).normal(0, 0.01, 50), [np.nan] * 5]
    assert np.isfinite(var_es(r)[0])
    with pytest.raises(ValueError):
        var_es(np.zeros(5))
    with pytest.raises(ValueError):
        var_es(np.zeros(100), level=0.05)


def test_kupiec_known_answers():
    assert kupiec(100, 5, 0.05) == (pytest.approx(0.0), pytest.approx(1.0))
    lr0, p0 = kupiec(100, 0, 0.05)                 # no breaches at all
    assert lr0 == pytest.approx(-2 * 100 * np.log(0.95)) and p0 < 0.01
    _, p_many = kupiec(100, 20, 0.05)
    assert p_many < 1e-5


def test_backtest_passes_on_steady_risk():
    r = pd.Series(np.random.default_rng(3).normal(0.001, 0.02, 400),
                  index=pd.date_range("2018-01-05", periods=400, freq="W-FRI"))
    bt = backtest_var(r, 0.95, 52)
    assert bt.n == 348 and bt.status == "PASS"
    assert abs(bt.breach_rate - 0.05) < 0.03


def test_backtest_fails_when_risk_jumps_and_var_lags():
    rng = np.random.default_rng(4)
    # 100 calm weeks, then a sudden 8-week storm, repeated: each storm starts
    # when the 52-week window holds only calm weeks, so VaR is broken at once
    blocks = [np.r_[rng.normal(0, 0.005, 100), rng.normal(0, 0.06, 8)] for _ in range(5)]
    r = pd.Series(np.concatenate(blocks),
                  index=pd.date_range("2016-01-01", periods=540, freq="W-FRI"))
    bt = backtest_var(r, 0.95, 52)
    assert bt.breach_rate > 0.075 and bt.p_value < 0.01 and bt.status == "FAIL"


def test_backtest_var_uses_only_past_weeks():
    """Changing a future week must not change any earlier VaR limit."""
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.02, 150),
                  index=pd.date_range("2018-01-05", periods=150, freq="W-FRI"))
    a = backtest_var(r)
    r2 = r.copy()
    r2.iloc[-1] = -0.9
    b = backtest_var(r2)
    pd.testing.assert_series_equal(a.var, b.var)


def test_backtest_needs_enough_history():
    with pytest.raises(ValueError):
        backtest_var(pd.Series(np.zeros(60)))


def test_monte_carlo_reports_var_and_es():
    rng = np.random.default_rng(6)
    idx = pd.date_range("2020-01-03", periods=200, freq="W-FRI")
    s = pd.Series(rng.normal(0.002, 0.03, 200), index=idx)
    st = simulate(s, s * 0.5, 8, 100_000, n_paths=2000).stats()
    assert st["var_99"] >= st["var_95"] > 0
    assert st["es_95"] >= st["var_95"] and st["es_99"] >= st["var_99"]
