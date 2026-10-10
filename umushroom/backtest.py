"""Backtesting engine.

Walks forward through history at a fixed rebalance cadence, at each
point building features from data strictly up to "today" (no
lookahead — see features.py), ranking the universe with the already
-fitted model, and measuring the realized forward return of the top-N
selection versus an equal-weight-all-tickers benchmark.

This module does NOT re-fit the model at each step (a "purged walk
-forward" retraining loop is the more rigorous version of this, and
is noted as a known limitation below) — it evaluates one fixed,
already-trained model's decisions across history. That is a
materially simpler and more honest claim than "this model would have
made money," so the result object makes the distinction explicit via
`BacktestResult.caveat`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from umushroom.exceptions import InsufficientHistoryError
from umushroom.features import MIN_BARS_REQUIRED, build_feature_panel
from umushroom.logging_config import get_logger
from umushroom.model import RankingModel

logger = get_logger("backtest")


@dataclass(frozen=True)
class BacktestResult:
    period_returns: pd.Series          # indexed by rebalance date
    benchmark_returns: pd.Series
    cumulative_return: float
    benchmark_cumulative_return: float
    win_rate: float
    sharpe_annualized: float
    n_periods: int
    caveat: str = (
        "WARNING -- IN-SAMPLE, NOT A REAL BACKTEST: this model was fit on "
        "the ENTIRE price history (including the period scored below) and "
        "then evaluated across that same history. The model has already "
        "seen the return patterns it's being 'tested' against, so these "
        "numbers are inflated and are NOT a valid estimate of real-world "
        "forward performance. This exists only as a fast sanity check "
        "(does the pipeline run, do picks look directionally sane). For "
        "an actual out-of-sample performance estimate, use "
        "purged_backtest.run_purged_walk_forward(), which retrains per "
        "fold on strictly prior data with a purge gap."
    )

    def summary(self) -> str:
        return (
            f"Backtest periods:         {self.n_periods}\n"
            f"Win rate:                 {self.win_rate:.1%}\n"
            f"Strategy total return:    {self.cumulative_return:.2%}\n"
            f"Benchmark total return:   {self.benchmark_cumulative_return:.2%}\n"
            f"Annualized Sharpe:        {self.sharpe_annualized:.2f}\n"
            f"\nCaveat: {self.caveat}"
        )


def run_backtest(
    data: pd.DataFrame,
    model: RankingModel,
    top_n: int,
    rebalance_days: int,
    min_history_days: int,
    groups: dict[str, str] | None = None,
    dollar_volume: pd.DataFrame | None = None,
) -> BacktestResult:
    n = len(data)
    start_idx = max(min_history_days, MIN_BARS_REQUIRED) - 1
    end_idx = n - rebalance_days

    if start_idx >= end_idx:
        raise InsufficientHistoryError(
            f"Not enough history to backtest: have {n} bars, "
            f"need > {start_idx + rebalance_days}."
        )

    panel = build_feature_panel(data, groups, dollar_volume=dollar_volume)
    panel_dates = set(panel.index.get_level_values("date"))

    strategy_returns: list[float] = []
    benchmark_returns: list[float] = []
    dates: list = []

    for idx in range(start_idx, end_idx, rebalance_days):
        date = data.index[idx]
        if date not in panel_dates:
            continue
        feats = panel.xs(date, level="date")
        if feats.empty or len(feats) < top_n:
            logger.debug(f"Skipping idx={idx}: insufficient clean feature rows")
            continue

        preds = model.predict(feats)
        picks = preds.sort_values(ascending=False).head(top_n).index.tolist()

        fwd_ratio = data.iloc[idx + rebalance_days][picks] / data.iloc[idx][picks]
        strategy_period_return = float((fwd_ratio - 1).mean())

        bench_ratio = data.iloc[idx + rebalance_days] / data.iloc[idx]
        benchmark_period_return = float((bench_ratio - 1).mean())

        strategy_returns.append(strategy_period_return)
        benchmark_returns.append(benchmark_period_return)
        dates.append(date)
    if not strategy_returns:
        raise InsufficientHistoryError(
            "Backtest produced zero valid periods — check data quality "
            "and min_history_days configuration."
        )

    strat_series = pd.Series(strategy_returns, index=dates, name="strategy")
    bench_series = pd.Series(benchmark_returns, index=dates, name="benchmark")

    strat_cum = float((1 + strat_series).prod() - 1)
    bench_cum = float((1 + bench_series).prod() - 1)
    win_rate = float((strat_series > 0).mean())

    periods_per_year = 252 / rebalance_days
    std = strat_series.std()
    sharpe = (
        float((strat_series.mean() / std) * np.sqrt(periods_per_year))
        if std > 0 else 0.0
    )

    result = BacktestResult(
        period_returns=strat_series,
        benchmark_returns=bench_series,
        cumulative_return=strat_cum,
        benchmark_cumulative_return=bench_cum,
        win_rate=win_rate,
        sharpe_annualized=sharpe,
        n_periods=len(strat_series),
    )
    logger.info(f"Backtest complete: {result.n_periods} periods, "
                f"strategy={strat_cum:.2%} vs benchmark={bench_cum:.2%}")
    return result
