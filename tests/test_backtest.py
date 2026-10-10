from __future__ import annotations

import pytest

from umushroom.backtest import run_backtest
from umushroom.exceptions import InsufficientHistoryError
from umushroom.features import build_training_set
from umushroom.model import RankingModel


@pytest.fixture
def trained_model(synthetic_prices):
    X, y = build_training_set(synthetic_prices, forward_return_days=5,
                               step=5, min_history_days=70)
    return RankingModel(alpha=1.0, random_seed=42).fit(X, y)


def test_backtest_runs_and_returns_result(synthetic_prices, trained_model):
    result = run_backtest(
        synthetic_prices, trained_model, top_n=3, rebalance_days=5,
        min_history_days=70,
    )
    assert result.n_periods > 0
    assert 0.0 <= result.win_rate <= 1.0
    assert len(result.period_returns) == result.n_periods


def test_backtest_raises_on_insufficient_history(tiny_prices, trained_model):
    with pytest.raises(InsufficientHistoryError):
        run_backtest(tiny_prices, trained_model, top_n=1, rebalance_days=5,
                     min_history_days=70)


def test_backtest_top_n_cannot_exceed_universe(synthetic_prices, trained_model):
    # top_n larger than the number of tickers should skip every period
    # (feats available < top_n) and ultimately raise, not silently
    # return a partial portfolio.
    with pytest.raises(InsufficientHistoryError):
        run_backtest(
            synthetic_prices, trained_model, top_n=999, rebalance_days=5,
            min_history_days=70,
        )


def test_backtest_summary_is_a_string(synthetic_prices, trained_model):
    result = run_backtest(
        synthetic_prices, trained_model, top_n=3, rebalance_days=5,
        min_history_days=70,
    )
    summary = result.summary()
    assert isinstance(summary, str)
    assert "Backtest periods" in summary
    assert "Caveat" in summary


def test_backtest_benchmark_uses_full_universe(synthetic_prices, trained_model):
    """The benchmark series should reflect averaging over ALL tickers,
    not just the picked top_n — otherwise it isn't a fair comparison."""
    result_top3 = run_backtest(
        synthetic_prices, trained_model, top_n=3, rebalance_days=5,
        min_history_days=70,
    )
    result_top5 = run_backtest(
        synthetic_prices, trained_model, top_n=5, rebalance_days=5,
        min_history_days=70,
    )
    # Benchmark doesn't depend on top_n, only strategy selection does.
    pd_testing_equal = result_top3.benchmark_returns.equals(
        result_top5.benchmark_returns
    )
    assert pd_testing_equal
