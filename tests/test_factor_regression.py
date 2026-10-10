from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umushroom.exceptions import DataError
from umushroom.factor_regression import (
    benchmark_alpha,
    compound_factors_over_periods,
    fama_french_alpha,
    ols_hac,
    parse_french_csv,
    regress_alpha,
)


def _planted(alpha, beta, n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    f = pd.Series(rng.normal(0.001, 0.02, n), index=idx, name="mkt")
    y = alpha + beta * f + rng.normal(0, 0.005, n)
    return y, f


def test_regression_recovers_planted_alpha_and_beta():
    y, f = _planted(alpha=0.002, beta=0.7)
    res = regress_alpha(y, f.to_frame(), periods_per_year=50, label="t")
    assert res.alpha_per_period == pytest.approx(0.002, abs=0.0006)
    assert res.betas["mkt"] == pytest.approx(0.7, abs=0.03)
    assert res.alpha_tstat > 4


def test_zero_alpha_is_not_significant():
    y, f = _planted(alpha=0.0, beta=1.1, seed=3)
    res = regress_alpha(y, f.to_frame(), 50, "t")
    assert abs(res.alpha_tstat) < 2
    assert "NOT statistically" in res.summary()


def test_ols_matches_numpy_lstsq():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(200), rng.normal(size=(200, 2))])
    y = X @ np.array([0.1, 0.5, -0.3]) + rng.normal(0, 0.1, 200)
    beta, se, r2 = ols_hac(y, X, lags=0)
    np.testing.assert_allclose(beta, np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-10)
    assert (se > 0).all() and 0 < r2 < 1


def test_newey_west_widens_se_under_autocorrelation():
    rng = np.random.default_rng(2)
    n = 500
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + rng.normal()
    X = np.ones((n, 1))
    _b0, se0, _ = ols_hac(e, X, lags=0)
    _b1, se1, _ = ols_hac(e, X, lags=10)
    assert se1[0] > 1.5 * se0[0]


def test_regression_rejects_too_few_obs():
    y, f = _planted(0.001, 1.0, n=5)
    with pytest.raises(ValueError):
        regress_alpha(y, f.to_frame(), 50, "t")


def test_benchmark_alpha_runs():
    y, f = _planted(0.001, 0.9)
    res = benchmark_alpha(y, f, 50)
    assert "universe" in res.betas


FRENCH_SAMPLE = """This file was created using the 202608 CRSP database.
The Tbill return is the simple daily rate.

,Mkt-RF,SMB,HML,RF
20260102,   0.50,  -0.10,   0.20,   0.02
20260105,  -1.00,   0.30,  -0.10,   0.02
20260106,   0.25,   0.00,   0.05,   0.02

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
2025,  10.0, 1.0, 2.0, 4.0
"""


def test_parse_french_csv():
    df = parse_french_csv(FRENCH_SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 3                                  # annual block ignored
    assert df.loc["2026-01-05", "Mkt-RF"] == pytest.approx(-0.01)


def test_parse_french_csv_rejects_garbage():
    with pytest.raises(DataError):
        parse_french_csv("nothing useful here\n1,2,3")


def test_compound_factors_over_periods_uses_open_closed_interval():
    daily = parse_french_csv(FRENCH_SAMPLE)
    ends = pd.Series([pd.Timestamp("2026-01-06")], index=[pd.Timestamp("2026-01-02")])
    per = compound_factors_over_periods(daily, ends)
    # (start, end] -> 01-05 and 01-06 only
    assert per.loc["2026-01-02", "Mkt-RF"] == pytest.approx(0.99 * 1.0025 - 1)


def test_fama_french_alpha_end_to_end():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2024-01-01", periods=600)
    daily = pd.DataFrame({"Mkt-RF": rng.normal(0.0004, 0.01, 600),
                          "SMB": rng.normal(0, 0.005, 600),
                          "HML": rng.normal(0, 0.005, 600),
                          "RF": 0.0001}, index=idx)
    starts = idx[:-5:5]
    ends = pd.Series(idx[5::5][: len(starts)], index=starts)
    per = compound_factors_over_periods(daily, ends)
    strat = per["RF"] + 0.003 + 1.0 * per["Mkt-RF"] + pd.Series(
        rng.normal(0, 0.002, len(per)), index=per.index)
    res = fama_french_alpha(strat, ends, daily, 50, "GBP", "Developed")
    assert res.alpha_per_period == pytest.approx(0.003, abs=0.001)
    assert res.betas["Mkt-RF"] == pytest.approx(1.0, abs=0.1)
    assert "USD" in res.caveat
