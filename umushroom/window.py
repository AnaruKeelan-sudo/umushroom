"""Contest-window evaluation (v4.19).

UMushroom's Stage 1 lasts six weeks (12 Oct - 22 Nov 2026: 30 trading
days). A signal that pays off over 6-12 months is worthless to us if
the payoff has not arrived by 22 Nov, however good its annualised
numbers look. This module asks the only question that matters for the
contest: over EVERY historical stretch the same length as the contest,
how did the strategy do against its benchmark?

It takes the out-of-sample, per-rebalance-period returns from the purged
walk-forward backtest and compounds them into contest-length windows:

  - CONTIGUOUS only. Rebalance periods are chained into a window only
    when each one ends exactly where the next begins (the purged backtest
    leaves purge/embargo gaps between folds; a "6-week window" that
    secretly spans a gap is not a 6-week window).
  - ROLLING windows (every possible start) give the full distribution of
    contest outcomes -- the fairest picture of "what could happen to us".
  - NON-OVERLAPPING windows feed the t-statistic. Rolling windows share
    most of their weeks, so a t-stat on them would be inflated several
    times over; the non-overlapping set is independent enough to test.

The verdict thresholds are fixed here, not tuned to the data:
  PASS  active-return t >= 2 (non-overlapping windows) AND the strategy
        beat its benchmark in more than half of all rolling windows
  WARN  one of the two
  FAIL  neither

These are the numbers the ETF sleeve's "auto" selection rule reads (see
config.yaml): it trades the model's picks only while the window verdict
is not FAIL and the walk-forward IC is significant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
WINDOW_T_THRESHOLD = 2.0
WINDOW_WIN_RATE_THRESHOLD = 0.5


def _tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def contiguous_runs(entry_dates: pd.DatetimeIndex, end_dates: pd.Series
                    ) -> list[list[int]]:
    """Split periods into runs where period k ends on the entry date of
    period k+1. Returns lists of positional indices."""
    runs: list[list[int]] = []
    current: list[int] = []
    ends = pd.DatetimeIndex(end_dates.reindex(entry_dates).to_numpy())
    for k in range(len(entry_dates)):
        if current and ends[current[-1]] != entry_dates[k]:
            runs.append(current)
            current = []
        current.append(k)
    if current:
        runs.append(current)
    return runs


@dataclass(frozen=True)
class WindowResult:
    window_periods: int           # rebalance periods per contest window
    window_days: int              # trading days per contest window
    rolling_strategy: pd.Series   # compounded window return, indexed by entry date
    rolling_benchmark: pd.Series
    nonoverlap_strategy: pd.Series
    nonoverlap_benchmark: pd.Series

    @property
    def n_windows(self) -> int:
        return len(self.rolling_strategy)

    @property
    def n_independent(self) -> int:
        return len(self.nonoverlap_strategy)

    @property
    def rolling_active(self) -> pd.Series:
        return (self.rolling_strategy - self.rolling_benchmark).rename("active")

    @property
    def nonoverlap_active(self) -> pd.Series:
        return (self.nonoverlap_strategy - self.nonoverlap_benchmark).rename("active")

    @property
    def win_rate(self) -> float:
        """Share of rolling windows in which the strategy beat the benchmark."""
        a = self.rolling_active
        return float((a > 0).mean()) if len(a) else float("nan")

    @property
    def positive_rate(self) -> float:
        s = self.rolling_strategy
        return float((s > 0).mean()) if len(s) else float("nan")

    @property
    def mean_active(self) -> float:
        a = self.rolling_active
        return float(a.mean()) if len(a) else float("nan")

    @property
    def median_strategy(self) -> float:
        s = self.rolling_strategy
        return float(s.median()) if len(s) else float("nan")

    @property
    def median_benchmark(self) -> float:
        b = self.rolling_benchmark
        return float(b.median()) if len(b) else float("nan")

    @property
    def worst_strategy(self) -> float:
        s = self.rolling_strategy
        return float(s.min()) if len(s) else float("nan")

    @property
    def best_strategy(self) -> float:
        s = self.rolling_strategy
        return float(s.max()) if len(s) else float("nan")

    @property
    def active_tstat(self) -> float:
        return _tstat(self.nonoverlap_active.to_numpy(dtype=float))

    @property
    def verdict(self) -> str:
        t_ok = bool(np.isfinite(self.active_tstat)
                    and self.active_tstat >= WINDOW_T_THRESHOLD)
        win_ok = bool(np.isfinite(self.win_rate)
                      and self.win_rate > WINDOW_WIN_RATE_THRESHOLD)
        if t_ok and win_ok:
            return PASS
        if t_ok or win_ok:
            return WARN
        return FAIL

    def summary(self) -> str:
        explain = {
            PASS: "the edge shows up inside a contest-length window.",
            WARN: "some evidence, not enough to rely on inside one window.",
            FAIL: "no evidence the picks beat the benchmark within one window.",
        }[self.verdict]
        lines = [
            (f"Contest window:             {self.window_periods} rebalance periods "
             f"(~{self.window_days} trading days)"),
            (f"Rolling windows:            {self.n_windows} "
             f"({self.n_independent} non-overlapping)"),
            (f"Median window return:       strategy {self.median_strategy:+.2%} vs "
             f"benchmark {self.median_benchmark:+.2%}"),
            (f"Mean active return/window:  {self.mean_active:+.2%}  "
             f"(t = {self.active_tstat:+.2f} on non-overlapping windows)"),
            f"Beat the benchmark:         {self.win_rate:.1%} of windows",
            f"Finished up:                {self.positive_rate:.1%} of windows",
            (f"Worst / best window:        {self.worst_strategy:+.2%} / "
             f"{self.best_strategy:+.2%}"),
            f"VERDICT ({self.verdict}): {explain}",
        ]
        return "\n".join(lines)


def contest_windows(period_returns: pd.Series, benchmark_period_returns: pd.Series,
                    period_end_dates: pd.Series, window_periods: int,
                    rebalance_days: int) -> WindowResult:
    """Compound contiguous rebalance periods into contest-length windows.

    `period_returns`/`benchmark_period_returns` are indexed by each
    period's ENTRY date; `period_end_dates` maps entry date -> exit date
    (all three come straight from PurgedWalkForwardResult).
    """
    if window_periods < 1:
        raise ValueError("window_periods must be >= 1")
    if rebalance_days < 1:
        raise ValueError("rebalance_days must be >= 1")
    s = period_returns.dropna()
    b = benchmark_period_returns.reindex(s.index)
    if b.isna().any():
        raise ValueError("benchmark returns missing for some periods")
    if not s.index.is_monotonic_increasing:
        raise ValueError("period returns must be sorted by entry date")
    entry = pd.DatetimeIndex(s.index)
    ends = pd.Series(pd.DatetimeIndex(period_end_dates.reindex(entry).to_numpy()),
                     index=entry)
    if ends.isna().any():
        raise ValueError("period_end_dates missing for some periods")

    roll_s: dict[pd.Timestamp, float] = {}
    roll_b: dict[pd.Timestamp, float] = {}
    nov_s: dict[pd.Timestamp, float] = {}
    nov_b: dict[pd.Timestamp, float] = {}
    sv, bv = s.to_numpy(dtype=float), b.to_numpy(dtype=float)
    for run in contiguous_runs(entry, ends):
        for pos, start in enumerate(run):
            if pos + window_periods > len(run):
                break
            sl = run[pos: pos + window_periods]
            cs = float(np.prod(1 + sv[sl]) - 1)
            cb = float(np.prod(1 + bv[sl]) - 1)
            roll_s[entry[start]] = cs
            roll_b[entry[start]] = cb
            if pos % window_periods == 0:
                nov_s[entry[start]] = cs
                nov_b[entry[start]] = cb

    def series(d: dict[pd.Timestamp, float], name: str) -> pd.Series:
        return pd.Series(d, dtype=float, name=name).sort_index()

    return WindowResult(
        window_periods=window_periods,
        window_days=window_periods * rebalance_days,
        rolling_strategy=series(roll_s, "strategy"),
        rolling_benchmark=series(roll_b, "benchmark"),
        nonoverlap_strategy=series(nov_s, "strategy"),
        nonoverlap_benchmark=series(nov_b, "benchmark"),
    )


def window_periods_for(contest_window_days: int, rebalance_days: int) -> int:
    """How many rebalance periods make one contest window (rounded down,
    never below one)."""
    if contest_window_days < 1 or rebalance_days < 1:
        raise ValueError("contest_window_days and rebalance_days must be >= 1")
    return max(1, contest_window_days // rebalance_days)
