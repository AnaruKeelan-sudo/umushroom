from __future__ import annotations

import pytest

from umushroom.exceptions import DataError
from umushroom.portfolio_construction import equal_weight, hrp_weight


def test_equal_weight_sums_to_one():
    w = equal_weight(["A", "B", "C", "D"])
    assert w.sum() == pytest.approx(1.0)


def test_equal_weight_each_ticker_equal():
    w = equal_weight(["A", "B"])
    assert w["A"] == pytest.approx(0.5)
    assert w["B"] == pytest.approx(0.5)


def test_equal_weight_rejects_empty_list():
    with pytest.raises(ValueError):
        equal_weight([])


def test_hrp_weight_sums_to_one(synthetic_prices):
    tickers = list(synthetic_prices.columns[:4])
    w = hrp_weight(synthetic_prices, tickers, as_of_idx=len(synthetic_prices) - 1)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_hrp_weight_covers_all_requested_tickers(synthetic_prices):
    tickers = list(synthetic_prices.columns[:5])
    w = hrp_weight(synthetic_prices, tickers, as_of_idx=len(synthetic_prices) - 1)
    assert set(w.index) == set(tickers)


def test_hrp_weight_all_non_negative(synthetic_prices):
    """HRP (unlike Markowitz) should never recommend a short position
    in this context -- all weights should be >= 0."""
    tickers = list(synthetic_prices.columns[:6])
    w = hrp_weight(synthetic_prices, tickers, as_of_idx=len(synthetic_prices) - 1)
    assert (w >= -1e-9).all()


def test_hrp_weight_rejects_single_ticker(synthetic_prices):
    with pytest.raises(ValueError):
        hrp_weight(synthetic_prices, [synthetic_prices.columns[0]],
                   as_of_idx=len(synthetic_prices) - 1)


def test_hrp_weight_rejects_insufficient_history(synthetic_prices):
    tickers = list(synthetic_prices.columns[:3])
    with pytest.raises(DataError):
        hrp_weight(synthetic_prices, tickers, as_of_idx=10)


def test_hrp_weight_rejects_unknown_ticker(synthetic_prices):
    with pytest.raises(DataError):
        hrp_weight(synthetic_prices, ["NOT_A_REAL_TICKER", synthetic_prices.columns[0]],
                   as_of_idx=len(synthetic_prices) - 1)


def test_hrp_weight_no_lookahead(synthetic_prices):
    """Same discipline as features.py: weights computed as-of idx must
    not change if future rows are appended to the DataFrame."""
    idx = 150
    tickers = list(synthetic_prices.columns[:4])
    w_full = hrp_weight(synthetic_prices, tickers, as_of_idx=idx)
    w_truncated = hrp_weight(synthetic_prices.iloc[: idx + 1], tickers, as_of_idx=idx)
    for t in tickers:
        assert w_full[t] == pytest.approx(w_truncated[t])


# ---------------------------------------------------------------- v4 tests
import sys

import numpy as np
import pandas as pd


def _two_uncorrelated(vol_a=0.01, vol_b=0.02, n=600):
    rng = np.random.default_rng(9)
    rets = pd.DataFrame({"A": rng.normal(0, vol_a, n), "B": rng.normal(0, vol_b, n)},
                        index=pd.bdate_range("2023-01-02", periods=n))
    return 100 * (1 + rets).cumprod()


def test_hrp_two_assets_equals_inverse_variance():
    """Analytic check: for two assets, HRP's single bisection gives
    w_A = var_B / (var_A + var_B), i.e. inverse-variance weights."""
    prices = _two_uncorrelated()
    w = hrp_weight(prices, ["A", "B"], as_of_idx=len(prices) - 1)
    rets = prices.iloc[-252:].pct_change().iloc[1:]
    var = rets.var()
    assert w["A"] == pytest.approx(var["B"] / (var["A"] + var["B"]), rel=1e-9)
    assert w["A"] > w["B"]                       # lower vol -> more weight


def test_hrp_has_no_pyportfolioopt_dependency(synthetic_prices):
    """v4 regression test: HRP must work with PyPortfolioOpt/cvxpy absent
    (they do not install on Python 3.14)."""
    hrp_weight(synthetic_prices, list(synthetic_prices.columns[:5]),
               as_of_idx=len(synthetic_prices) - 1)
    assert "pypfopt" not in sys.modules
    assert "cvxpy" not in sys.modules


def test_hrp_rejects_duplicate_tickers(synthetic_prices):
    t = synthetic_prices.columns[0]
    with pytest.raises(ValueError):
        hrp_weight(synthetic_prices, [t, t], as_of_idx=len(synthetic_prices) - 1)
