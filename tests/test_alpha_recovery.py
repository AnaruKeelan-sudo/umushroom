"""Known-answer integration tests for the WHOLE research pipeline.

These are the tests that justify trusting the backtest verdict: the
pipeline must (a) find a signal that is genuinely there, and (b) report
NO skill when there is nothing to find. A pipeline that "finds" alpha in
pure noise has a leakage bug, however good its returns look.
"""
from __future__ import annotations

import pytest

from umushroom.purged_backtest import run_purged_walk_forward

KW = {"test_window_days": 21, "purge_days": 5, "embargo_days": 5,
      "rebalance_days": 5, "forward_return_days": 5, "top_n": 8,
      "min_train_days": 504, "model_alpha": 1.0, "random_seed": 42}


@pytest.mark.parametrize("target", ["rank", "excess"])
def test_pipeline_recovers_planted_alpha(planted_signal_prices, target):
    r = run_purged_walk_forward(planted_signal_prices, target=target, **KW)
    assert r.mean_information_coefficient > 0.03
    assert r.ic_tstat > 3
    assert r.has_evidence_of_skill
    assert r.active_returns.mean() > 0
    assert r.long_short_period_returns.mean() > 0


def test_pipeline_finds_no_alpha_in_pure_noise(synthetic_noise_prices):
    r = run_purged_walk_forward(synthetic_noise_prices, **KW)
    assert abs(r.ic_tstat) < 2
    assert not r.has_evidence_of_skill
    assert "NO statistically reliable" in r.summary()


def test_no_label_leakage_across_purge(planted_signal_prices):
    """Every training row's label window must close before the test
    window opens: idx(last feature date) + h < idx(test start)."""
    r = run_purged_walk_forward(planted_signal_prices, target="rank", **KW)
    index = planted_signal_prices.index
    for f in r.folds:
        last_feat = index.get_loc(f.train_last_feature_date)
        assert last_feat + KW["forward_return_days"] < index.get_loc(f.test_start_date)


def test_longer_horizon_scored_on_holding_period_without_leakage(planted_signal_prices):
    """21-day target, weekly rebalance: every scored period must span exactly
    one rebalance interval (not overlapping 21-day windows), and every
    training label (21 days long) must close before its test window."""
    kw = {**KW, "forward_return_days": 21, "purge_days": 21}
    r = run_purged_walk_forward(planted_signal_prices, target="rank", **kw)
    index = planted_signal_prices.index
    for start, end in r.period_end_dates.items():
        assert index.get_loc(end) - index.get_loc(start) == KW["rebalance_days"]
    for f in r.folds:
        assert (index.get_loc(f.train_last_feature_date) + 21
                < index.get_loc(f.test_start_date))
    assert r.has_evidence_of_skill        # planted signal still found
