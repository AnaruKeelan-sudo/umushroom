from __future__ import annotations

import pytest

from umushroom.exceptions import InsufficientHistoryError
from umushroom.purged_backtest import run_purged_walk_forward


def test_purged_walk_forward_runs_and_produces_folds(synthetic_prices):
    result = run_purged_walk_forward(
        synthetic_prices,
        test_window_days=20,
        purge_days=5,
        embargo_days=5,
        rebalance_days=5,
        forward_return_days=5,
        top_n=3,
        min_train_days=70,
        model_alpha=1.0,
        random_seed=42,
    )
    assert len(result.folds) > 0
    assert 0.0 <= result.overall_win_rate <= 1.0


def test_purged_folds_are_chronologically_non_overlapping(synthetic_prices):
    """Each fold's test window must start strictly after the previous
    fold's test window ends (the walk only ever moves forward)."""
    result = run_purged_walk_forward(
        synthetic_prices,
        test_window_days=20,
        purge_days=5,
        embargo_days=5,
        rebalance_days=5,
        forward_return_days=5,
        top_n=3,
        min_train_days=70,
        model_alpha=1.0,
        random_seed=42,
    )
    for prev, curr in zip(result.folds, result.folds[1:]):
        assert curr.test_start_date > prev.test_end_date


def test_purged_train_end_precedes_test_start_by_at_least_purge_gap(synthetic_prices):
    """Correctness check for the leakage guard itself: every fold's
    training data must end at least purge_days before its test window
    starts."""
    purge_days = 7
    result = run_purged_walk_forward(
        synthetic_prices,
        test_window_days=15,
        purge_days=purge_days,
        embargo_days=0,
        rebalance_days=5,
        forward_return_days=5,
        top_n=3,
        min_train_days=70,
        model_alpha=1.0,
        random_seed=42,
    )
    for fold in result.folds:
        gap_days = (fold.test_start_date - fold.train_end_date).days
        # Business-day gap of purge_days corresponds to a calendar gap
        # of at least purge_days (weekends only add to it).
        assert gap_days >= purge_days


def test_purged_walk_forward_raises_on_insufficient_history(tiny_prices):
    with pytest.raises(InsufficientHistoryError):
        run_purged_walk_forward(
            tiny_prices,
            test_window_days=20,
            purge_days=5,
            embargo_days=5,
            rebalance_days=5,
            forward_return_days=5,
            top_n=1,
            min_train_days=70,
            model_alpha=1.0,
            random_seed=42,
        )


def test_purged_walk_forward_exposes_period_returns(synthetic_prices):
    """period_returns must be the raw per-rebalance-period series (not
    just fold aggregates) so bootstrap_sharpe has something to resample."""
    result = run_purged_walk_forward(
        synthetic_prices,
        test_window_days=20,
        purge_days=5,
        embargo_days=5,
        rebalance_days=5,
        forward_return_days=5,
        top_n=3,
        min_train_days=70,
        model_alpha=1.0,
        random_seed=42,
    )
    assert len(result.period_returns) > len(result.folds)
    assert result.period_returns.index.is_monotonic_increasing


def test_purged_walk_forward_summary_is_string(synthetic_prices):
    result = run_purged_walk_forward(
        synthetic_prices,
        test_window_days=20,
        purge_days=5,
        embargo_days=5,
        rebalance_days=5,
        forward_return_days=5,
        top_n=3,
        min_train_days=70,
        model_alpha=1.0,
        random_seed=42,
    )
    summary = result.summary()
    assert isinstance(summary, str)
    assert "Information Coeff" in summary


def test_purged_walk_forward_reproducible_with_same_seed(synthetic_prices):
    kwargs = {
        "data": synthetic_prices, "test_window_days": 20, "purge_days": 5,
        "embargo_days": 5, "rebalance_days": 5, "forward_return_days": 5,
        "top_n": 3, "min_train_days": 70, "model_alpha": 1.0, "random_seed": 42,
    }
    result_a = run_purged_walk_forward(**kwargs)
    result_b = run_purged_walk_forward(**kwargs)
    assert result_a.overall_strategy_return == result_b.overall_strategy_return
    assert len(result_a.folds) == len(result_b.folds)


def test_summary_reports_test_period_and_annualised_returns(synthetic_prices):
    """v4.6: multi-year totals (e.g. crypto's 1,379%) must come with the
    test span and a per-year figure so they aren't misread."""
    r = run_purged_walk_forward(
        synthetic_prices, test_window_days=20, purge_days=5, embargo_days=5,
        rebalance_days=5, forward_return_days=5, top_n=3, min_train_days=70,
        model_alpha=1.0, random_seed=42)
    text = r.summary()
    assert "Test period:" in text and "years)" in text and "a year)" in text
    expected = (1 + r.overall_strategy_return) ** (1 / r.years) - 1
    assert f"{expected:+.1%} a year" in text
