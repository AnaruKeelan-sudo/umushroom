"""Contest-window evaluation (window.py): windows are built only from
contiguous rebalance periods, compounded correctly, and the verdict
thresholds behave as declared."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umushroom.window import (
    FAIL,
    PASS,
    WARN,
    contest_windows,
    contiguous_runs,
    window_periods_for,
)


def _periods(n: int, start: str = "2024-01-01", step_days: int = 7):
    """n weekly periods where each one ends exactly where the next begins."""
    entry = pd.bdate_range(start, periods=n + 1, freq=f"{step_days}D")
    ends = pd.Series(entry[1:], index=entry[:-1])
    return pd.DatetimeIndex(entry[:-1]), ends


def test_window_periods_for_rounds_down_never_below_one():
    assert window_periods_for(30, 5) == 6
    assert window_periods_for(30, 7) == 4
    assert window_periods_for(3, 5) == 1
    with pytest.raises(ValueError):
        window_periods_for(0, 5)


def test_windows_compound_contiguous_periods():
    entry, ends = _periods(8)
    strat = pd.Series([0.01] * 8, index=entry)
    bench = pd.Series([0.0] * 8, index=entry)
    w = contest_windows(strat, bench, ends, window_periods=4, rebalance_days=5)
    assert w.n_windows == 5                      # 8 - 4 + 1 rolling starts
    assert w.n_independent == 2                  # starts at 0 and 4
    assert w.window_days == 20
    np.testing.assert_allclose(w.rolling_strategy.to_numpy(), 1.01 ** 4 - 1)
    np.testing.assert_allclose(w.rolling_benchmark.to_numpy(), 0.0)
    assert w.win_rate == 1.0 and w.positive_rate == 1.0
    assert w.worst_strategy == pytest.approx(1.01 ** 4 - 1)


def test_a_gap_between_periods_breaks_the_window():
    entry, ends = _periods(8)
    # break the chain after the 3rd period: period 3 ends a day early
    ends.iloc[2] = ends.iloc[2] - pd.Timedelta(days=1)
    strat = pd.Series(0.01, index=entry)
    bench = pd.Series(0.0, index=entry)
    runs = contiguous_runs(entry, ends)
    assert [len(r) for r in runs] == [3, 5]
    w = contest_windows(strat, bench, ends, window_periods=4, rebalance_days=5)
    # only the second run (5 periods) can host a 4-period window
    assert w.n_windows == 2 and w.n_independent == 1


def test_verdict_thresholds():
    entry, ends = _periods(60)
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0.0, 0.01, 60), index=entry)
    # strong, consistent edge -> PASS
    good = bench + 0.01
    w = contest_windows(good, bench, ends, 6, 5)
    assert w.verdict == PASS and w.active_tstat > 2 and w.win_rate > 0.5
    # no edge at all -> FAIL (win rate ~50%, t ~0)
    noise = bench + rng.normal(0.0, 0.0005, 60)
    w2 = contest_windows(noise, bench, ends, 6, 5)
    assert w2.verdict in (FAIL, WARN)
    assert w2.verdict != PASS
    # consistently WORSE than benchmark -> FAIL
    bad = bench - 0.01
    w3 = contest_windows(bad, bench, ends, 6, 5)
    assert w3.verdict == FAIL and w3.win_rate == 0.0


def test_summary_mentions_verdict_and_counts():
    entry, ends = _periods(12)
    strat = pd.Series(0.002, index=entry)
    bench = pd.Series(0.001, index=entry)
    w = contest_windows(strat, bench, ends, 6, 5)
    text = w.summary()
    assert "VERDICT" in text and "Rolling windows" in text
    assert f"{w.n_windows} ({w.n_independent} non-overlapping)" in text


def test_input_validation():
    entry, ends = _periods(5)
    strat = pd.Series(0.0, index=entry)
    with pytest.raises(ValueError):
        contest_windows(strat, strat, ends, 0, 5)
    with pytest.raises(ValueError):
        contest_windows(strat, strat.iloc[:2], ends, 2, 5)     # benchmark missing
    with pytest.raises(ValueError):
        contest_windows(strat, strat, ends.iloc[:2], 2, 5)      # end dates missing
    w = contest_windows(strat, strat, ends, 10, 5)             # too few periods
    assert w.n_windows == 0 and w.verdict == FAIL and np.isnan(w.win_rate)
