"""Known-answer tests for the validation suite itself: it must PASS a real
(planted) signal and must NOT pass pure noise."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umushroom.purged_backtest import run_purged_walk_forward
from umushroom.validation import (
    FAIL,
    PASS,
    decile_returns,
    group_ic,
    pick_concentration,
    run_validation,
    yearly_stats,
)

KW = {"test_window_days": 21, "purge_days": 5, "embargo_days": 5,
      "rebalance_days": 5, "forward_return_days": 5, "top_n": 8,
      "min_train_days": 504, "model_alpha": 1.0, "random_seed": 42,
      "target": "rank"}


@pytest.fixture(scope="module")
def planted_report(planted_signal_prices):
    groups = {t: ("A" if i % 2 else "B")
              for i, t in enumerate(planted_signal_prices.columns)}
    return run_validation(planted_signal_prices, "test", KW,
                          groups_for_report=groups, placebo_runs=19)


@pytest.fixture(scope="module")
def noise_report(synthetic_noise_prices):
    return run_validation(synthetic_noise_prices, "test", KW, placebo_runs=19)


def _status(report, name):
    return next(c.status for c in report.checks if c.name == name)


def test_planted_signal_passes_placebo_test(planted_report):
    assert _status(planted_report, "Placebo test") == PASS


def test_noise_does_not_pass_placebo_test(noise_report):
    assert _status(noise_report, "Placebo test") != PASS


def test_planted_signal_deciles_are_ordered(planted_report):
    assert _status(planted_report, "Decile monotonicity") == PASS


def test_planted_signal_works_in_every_group(planted_report):
    assert _status(planted_report, "Works in every market") == PASS


def test_suite_reports_every_check(planted_report):
    names = {c.name for c in planted_report.checks}
    assert names == {"Placebo test", "Decile monotonicity", "Stable over time",
                     "Works in every market", "Survives 1-day execution delay",
                     "Benchmark plausible (data sanity)",
                     "Not dominated by a few names"}
    assert "Result:" in planted_report.summary()


def test_shuffled_signal_has_no_skill(planted_signal_prices):
    """The placebo really destroys the signal (else the test is meaningless)."""
    r = run_purged_walk_forward(planted_signal_prices, shuffle_seed=0, **KW)
    assert abs(r.ic_tstat) < 2.5
    assert not r.has_evidence_of_skill


def test_shuffle_keeps_feature_distribution(planted_signal_prices):
    from umushroom.features import build_feature_panel
    from umushroom.purged_backtest import _shuffle_within_dates
    panel = build_feature_panel(planted_signal_prices)
    shuffled = _shuffle_within_dates(panel, seed=1)
    d = panel.index.get_level_values("date")[0]
    a = np.sort(panel.xs(d, level="date").to_numpy(), axis=0)
    b = np.sort(shuffled.xs(d, level="date").to_numpy(), axis=0)
    np.testing.assert_allclose(a, b)
    assert not panel.equals(shuffled)


def test_entry_delay_shifts_holding_window(planted_signal_prices):
    r0 = run_purged_walk_forward(planted_signal_prices, **KW)
    r_default = run_purged_walk_forward(planted_signal_prices, entry_delay_days=0, **KW)
    pd.testing.assert_series_equal(r0.period_returns, r_default.period_returns)
    r2 = run_purged_walk_forward(planted_signal_prices, entry_delay_days=2, **KW)
    idx = planted_signal_prices.index
    for start, end in r2.period_end_dates.items():
        assert idx.get_loc(end) - idx.get_loc(start) == 2 + KW["rebalance_days"]


def test_negative_delay_rejected(planted_signal_prices):
    with pytest.raises(ValueError):
        run_purged_walk_forward(planted_signal_prices, entry_delay_days=-1, **KW)


def test_helpers_on_recorded_cross_sections(planted_signal_prices):
    r = run_purged_walk_forward(planted_signal_prices, record_cross_sections=True, **KW)
    assert len(r.cross_sections) == len(r.period_returns)
    dec = decile_returns(r)
    assert list(dec.index) == list(range(1, 11)) and dec.iloc[-1] > dec.iloc[0]
    yearly = yearly_stats(r)
    assert yearly["periods"].sum() == len(r.period_returns)
    gic = group_ic(r, {t: "X" for t in planted_signal_prices.columns})
    assert list(gic.index) == ["X"] and gic["X"] > 0
    conc = pick_concentration(r, KW["top_n"])
    assert (conc <= 1).all() and (conc > 0).all()


def test_benchmark_sanity_fails_on_absurd_data(planted_signal_prices):
    """Prices growing 0.5%/day (~250%/yr) for every stock = broken data."""
    n = len(planted_signal_prices)
    absurd = planted_signal_prices.mul(1.005 ** np.arange(n), axis=0)
    rep = run_validation(absurd, "test", KW, placebo_runs=0, delays=())
    assert _status(rep, "Benchmark plausible (data sanity)") == FAIL
