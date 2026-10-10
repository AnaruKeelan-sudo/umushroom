"""Shared pytest fixtures.

All fixtures here generate synthetic data with a fixed random seed.
No test in this suite touches the network — this is deliberate: the
factor/model/backtest/allocation logic must be verifiable without
yfinance being reachable (e.g. in CI, or offline), and keeping the
data-fetch layer isolated behind umushroom.data is what makes that
possible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """600 trading days x 8 synthetic tickers, geometric-random-walk
    prices with distinct drift per ticker. 600 days because the v4
    factors (12-1 momentum, 52-week high, 1-year beta) need 253 bars
    before the first feature row exists.
    """
    rng = np.random.default_rng(seed=42)
    n_days = 600
    tickers = [f"TICK{i}" for i in range(8)]
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    data = {}
    for i, ticker in enumerate(tickers):
        drift = (i - 3.5) * 0.0005   # spread of drifts across tickers
        daily_returns = rng.normal(loc=drift, scale=0.015, size=n_days)
        prices = 100 * np.cumprod(1 + daily_returns)
        data[ticker] = prices

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def tiny_prices() -> pd.DataFrame:
    """Deliberately too-short history, for testing InsufficientHistoryError."""
    rng = np.random.default_rng(seed=1)
    dates = pd.bdate_range("2024-01-02", periods=10)
    data = {"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 10)),
            "B": 100 * np.cumprod(1 + rng.normal(0, 0.01, 10))}
    return pd.DataFrame(data, index=dates)


@pytest.fixture(scope="session")
def planted_signal_prices() -> pd.DataFrame:
    """KNOWN-ANSWER fixture: 40 tickers whose expected return is a
    slowly-varying, persistent latent drift. A stock that has done well
    over the past year therefore tends to KEEP doing well, i.e. 12-1
    momentum genuinely predicts future returns here. A correct pipeline
    must recover a positive, significant out-of-sample IC on this data;
    on pure noise (synthetic_noise_prices) it must not.
    """
    rng = np.random.default_rng(seed=7)
    n_days, n_tickers = 1100, 40
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    drift = np.zeros((n_days, n_tickers))
    drift[0] = rng.normal(0, 0.0012, n_tickers)
    for t in range(1, n_days):
        drift[t] = 0.998 * drift[t - 1] + rng.normal(0, 0.00008, n_tickers)
    market = rng.normal(0.0003, 0.009, (n_days, 1))
    rets = drift + market + rng.normal(0, 0.014, (n_days, n_tickers))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(prices, index=dates,
                        columns=[f"S{i:02d}" for i in range(n_tickers)])


@pytest.fixture(scope="session")
def synthetic_noise_prices() -> pd.DataFrame:
    """Same shape as planted_signal_prices but returns are pure iid
    noise: there is NO alpha to find."""
    rng = np.random.default_rng(seed=11)
    n_days, n_tickers = 1100, 40
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    market = rng.normal(0.0003, 0.009, (n_days, 1))
    rets = market + rng.normal(0, 0.014, (n_days, n_tickers))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(prices, index=dates,
                        columns=[f"N{i:02d}" for i in range(n_tickers)])
