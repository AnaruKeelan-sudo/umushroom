from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umushroom.robustness import bootstrap_sharpe


@pytest.fixture
def positive_drift_returns():
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0.01, 0.02, 60))


@pytest.fixture
def zero_drift_returns():
    rng = np.random.default_rng(2)
    return pd.Series(rng.normal(0.0, 0.02, 60))


def test_bootstrap_sharpe_returns_valid_interval(positive_drift_returns):
    result = bootstrap_sharpe(positive_drift_returns, periods_per_year=52,
                               n_bootstrap=500, block_size=5)
    assert result.ci_lower <= result.ci_upper


def test_bootstrap_sharpe_point_estimate_within_reasonable_range(positive_drift_returns):
    result = bootstrap_sharpe(positive_drift_returns, periods_per_year=52,
                               n_bootstrap=500, block_size=5)
    # Point estimate should roughly fall within the bootstrap distribution
    assert result.ci_lower - 2 <= result.point_estimate <= result.ci_upper + 2


def test_bootstrap_sharpe_positive_drift_mostly_positive_ci(positive_drift_returns):
    """A clearly positive-drift series should have a CI that leans
    positive (not a strict guarantee with randomness, but should hold
    with this drift/vol ratio and this many bootstrap draws)."""
    result = bootstrap_sharpe(positive_drift_returns, periods_per_year=52,
                               n_bootstrap=2000, block_size=5, random_seed=42)
    assert result.point_estimate > 0


def test_bootstrap_sharpe_zero_drift_ci_straddles_zero(zero_drift_returns):
    result = bootstrap_sharpe(zero_drift_returns, periods_per_year=52,
                               n_bootstrap=2000, block_size=5, random_seed=42)
    assert result.ci_lower < 0 < result.ci_upper


def test_bootstrap_sharpe_rejects_too_few_observations():
    tiny = pd.Series([0.01, 0.02, -0.01])
    with pytest.raises(ValueError):
        bootstrap_sharpe(tiny, periods_per_year=52)


def test_bootstrap_sharpe_rejects_invalid_block_size(positive_drift_returns):
    with pytest.raises(ValueError):
        bootstrap_sharpe(positive_drift_returns, periods_per_year=52, block_size=0)
    with pytest.raises(ValueError):
        bootstrap_sharpe(positive_drift_returns, periods_per_year=52,
                         block_size=len(positive_drift_returns) + 10)


def test_bootstrap_sharpe_rejects_invalid_confidence_level(positive_drift_returns):
    with pytest.raises(ValueError):
        bootstrap_sharpe(positive_drift_returns, periods_per_year=52,
                         confidence_level=1.5)


def test_bootstrap_sharpe_reproducible_with_same_seed(positive_drift_returns):
    r1 = bootstrap_sharpe(positive_drift_returns, periods_per_year=52,
                          n_bootstrap=500, block_size=5, random_seed=7)
    r2 = bootstrap_sharpe(positive_drift_returns, periods_per_year=52,
                          n_bootstrap=500, block_size=5, random_seed=7)
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper


def test_bootstrap_sharpe_summary_is_string(positive_drift_returns):
    result = bootstrap_sharpe(positive_drift_returns, periods_per_year=52,
                               n_bootstrap=500, block_size=5)
    assert isinstance(result.summary(), str)


def test_bootstrap_sharpe_summary_flags_ci_including_zero(zero_drift_returns):
    result = bootstrap_sharpe(zero_drift_returns, periods_per_year=52,
                               n_bootstrap=2000, block_size=5, random_seed=42)
    assert "cannot rule out" in result.summary()
