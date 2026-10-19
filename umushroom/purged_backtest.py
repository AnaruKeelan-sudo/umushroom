"""Purged & embargoed walk-forward backtest.

This is a materially more rigorous evaluation than backtest.py's single
-fixed-model approach. The difference:

  backtest.py:          one model, fit ONCE, evaluated across all of
                         history walk-forward. Fast, but doesn't tell you
                         whether the strategy would have kept working as
                         market regimes changed, and a lucky/unlucky single
                         fit can dominate the whole result.

  purged_backtest.py:    the model is RETRAINED at every fold using only
                         data strictly before that fold's test window, with
                         two leakage-prevention gaps:

    - PURGE: a gap between the end of a fold's training data and the
      start of its test window, sized to >= forward_return_days. Without
      this, the last few training labels (which are computed using price
      moves forward_return_days AFTER their feature date) would partially
      overlap the "unseen" test period -- the model would be trained on
      information from inside the window it's being tested on. This is
      the single most common silent bug in homemade backtests.

    - EMBARGO: an additional gap after each fold's test window before
      that data is allowed back in as training data for the NEXT fold.
      This guards against serial correlation in returns bleeding
      information backward across the fold boundary.

Because folds are walked strictly chronologically (training data is
always entirely in the past relative to its test window), this does not
need the full combinatorial purged cross-validation scheme from Lopez de
Prado's "Advances in Financial Machine Learning" (which exists to handle
test folds scattered non-chronologically through time, where training
data can sit on both sides of a test fold). Here, purge is the
correctness-critical gap; embargo is extra conservatism. That
distinction is stated explicitly rather than left implicit, since it's
the kind of thing that's easy to over- or under-claim.

Adds an Information Coefficient (IC) per fold -- the Spearman rank
correlation between predicted scores and realized returns across the
universe at each rebalance point. IC is the standard way quant
research actually measures whether a ranking signal has any skill,
independent of position sizing or win-rate quirks. An IC consistently
near 0 across folds means the signal isn't predictive, even if a
particular backtest run happened to be profitable.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from umushroom.exceptions import DataError, InsufficientHistoryError
from umushroom.features import (
    MIN_BARS_REQUIRED,
    build_feature_panel,
    build_training_set,
)
from umushroom.logging_config import get_logger
from umushroom.model import RankingModel
from umushroom.portfolio_construction import cap_weights, equal_weight
from umushroom.selection import capped_top, correlation_filter

# v4.20: (prices, picks, as_of_idx) -> weights. The backtest scores every
# period with the SAME weighting scheme the live run trades (HRP or factor
# parity), capped exactly as allocation.py caps it; before, HRP was applied
# live but every backtest was equal weight.
WeightFn = Callable[[pd.DataFrame, list[str], int], pd.Series]

logger = get_logger("purged_backtest")

# Verdict thresholds. |t| >= 2 is the conventional ~95% bar; the
# Harvey-Liu-Zhu (2016) argument for t >= 3 applies to factor DISCOVERY
# across many trials, which is stated in the summary rather than enforced.
SKILL_T_THRESHOLD = 2.0


def _tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    n_train_samples: int
    strategy_return: float
    benchmark_return: float
    information_coefficient: float
    # Latest FEATURE date used in training. Its label needs prices up to
    # forward_return_days later, which must still be before test_start.
    train_last_feature_date: pd.Timestamp | None = None


@dataclass(frozen=True)
class PurgedWalkForwardResult:
    folds: list[FoldResult]
    overall_strategy_return: float
    overall_benchmark_return: float
    overall_win_rate: float
    overall_sharpe: float
    mean_information_coefficient: float
    period_returns: pd.Series   # every individual rebalance-period strategy
                                 # return across all folds, indexed by the
                                 # period's ENTRY date. This is the raw
                                 # series bootstrap_sharpe() resamples.
    benchmark_period_returns: pd.Series = field(default_factory=pd.Series)
    long_short_period_returns: pd.Series = field(default_factory=pd.Series)
    period_ic: pd.Series = field(default_factory=pd.Series)
    period_end_dates: pd.Series = field(default_factory=pd.Series)
    # v4.7 (validation): per-period (date, predictions, realized returns),
    # only filled when record_cross_sections=True
    cross_sections: list = field(default_factory=list)
    # v4.20: periods where the weighting scheme failed and equal weight was
    # used instead (reported, so a scheme that mostly falls back is visible)
    weight_fallbacks: int = 0

    # ---- period length / annualisation (v4.6: totals compound over many
    # years -- e.g. crypto's 1,379% was ~6 years -- so show per-year too)
    @property
    def years(self) -> float:
        if len(self.period_returns) == 0:
            return float("nan")
        start = self.period_returns.index.min()
        end = (self.period_end_dates.max() if len(self.period_end_dates)
               else self.period_returns.index.max())
        return max((end - start).days / 365.25, 1e-9)

    def _annualise(self, total: float) -> float:
        y = self.years
        return float((1 + total) ** (1 / y) - 1) if np.isfinite(y) else float("nan")

    def _span_text(self) -> str:
        if len(self.period_returns) == 0:
            return "n/a"
        start = self.period_returns.index.min().date()
        end = (self.period_end_dates.max() if len(self.period_end_dates)
               else self.period_returns.index.max()).date()
        return f"{start} .. {end}  ({self.years:.1f} years)"

    # ---- derived skill statistics
    @property
    def active_returns(self) -> pd.Series:
        return (self.period_returns - self.benchmark_period_returns).dropna()

    @property
    def ic_tstat(self) -> float:
        return _tstat(self.period_ic.to_numpy(dtype=float))

    @property
    def ic_hit_rate(self) -> float:
        ic = self.period_ic.dropna()
        return float((ic > 0).mean()) if len(ic) else float("nan")

    @property
    def active_tstat(self) -> float:
        return _tstat(self.active_returns.to_numpy(dtype=float))

    @property
    def has_evidence_of_skill(self) -> bool:
        return bool(np.isfinite(self.ic_tstat) and self.ic_tstat >= SKILL_T_THRESHOLD
                    and self.mean_information_coefficient > 0)

    def summary(self) -> str:
        header = (
            f"{'Fold':<5}{'TrainEnd':<12}{'Test window':<25}"
            f"{'Strategy':>10}{'Bench':>9}{'IC':>8}"
        )
        lines = [header]
        for f in self.folds:
            test_range = f"{f.test_start_date.date()}..{f.test_end_date.date()}"
            ic_str = f"{f.information_coefficient:+.3f}" if not np.isnan(
                f.information_coefficient) else "  n/a"
            lines.append(
                f"{f.fold_id:<5}{f.train_end_date.date()!s:<12}{test_range:<25}"
                f"{f.strategy_return:>9.2%} {f.benchmark_return:>8.2%} {ic_str:>8}"
            )
        act = self.active_returns
        ls = self.long_short_period_returns.dropna()
        lines += [
            "",
            f"Folds run:                  {len(self.folds)}",
            f"Rebalance periods:          {len(self.period_returns)}",
            f"Test period:                {self._span_text()}",
            (f"Overall strategy return:    {self.overall_strategy_return:.2%}  "
             f"(= {self._annualise(self.overall_strategy_return):+.1%} a year)"),
            (f"Overall benchmark return:   {self.overall_benchmark_return:.2%}  "
             f"(= {self._annualise(self.overall_benchmark_return):+.1%} a year)"),
            f"Overall win rate (folds):   {self.overall_win_rate:.1%}",
            f"Annualized Sharpe:          {self.overall_sharpe:.2f}",
            "",
            "--- Skill statistics (what actually measures alpha) ---",
            f"Mean Information Coeff:     {self.mean_information_coefficient:+.4f}",
            (f"IC t-statistic:             {self.ic_tstat:+.2f}   "
             f"(|t| >= {SKILL_T_THRESHOLD:.0f} ~ 95% confidence)"),
            (f"IC hit rate:                {self.ic_hit_rate:.1%}   "
             f"(share of periods with IC > 0)"),
            (f"Mean active return/period:  "
             f"{(act.mean() if len(act) else float('nan')):+.3%}  "
             f"(t={self.active_tstat:+.2f}) vs equal-weight universe"),
            (f"Top-minus-bottom quintile:  "
             f"{(ls.mean() if len(ls) else float('nan')):+.3%} per period"),
            "",
        ]
        if self.has_evidence_of_skill:
            lines.append("VERDICT: out-of-sample IC is positive and statistically "
                         "significant -- evidence of real ranking skill.")
        else:
            lines.append("VERDICT: NO statistically reliable evidence of skill "
                         "out-of-sample. Do not present backtest returns as alpha; "
                         "an equal-weight universe portfolio is the honest default.")
        lines.append("  (Caveat: if many feature/parameter variants were tried, "
                     "require a higher t -- e.g. 3 -- to allow for multiple testing.)")
        return "\n".join(lines)


def _shuffle_within_dates(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute which ticker each feature row belongs to, separately on each
    date. Feature distributions, dates and outcomes are unchanged; only the
    link between a stock's features and its own future return is broken."""
    rng = np.random.default_rng(seed)
    parts = []
    for _date, block in panel.groupby(level="date", sort=False):
        perm = rng.permutation(len(block))
        parts.append(pd.DataFrame(block.to_numpy()[perm], index=block.index,
                                  columns=block.columns))
    return pd.concat(parts)


def run_purged_walk_forward(
    data: pd.DataFrame,
    test_window_days: int,
    purge_days: int,
    embargo_days: int,
    rebalance_days: int,
    forward_return_days: int,
    top_n: int,
    min_train_days: int,
    model_alpha: float,
    random_seed: int,
    target: str = "rank",
    groups: dict[str, str] | None = None,
    dollar_volume: pd.DataFrame | None = None,
    entry_delay_days: int = 0,
    shuffle_seed: int | None = None,
    record_cross_sections: bool = False,
    sectors: dict[str, str] | None = None,
    max_per_sector: int | None = None,
    feature_set: str = "equity",
    model_type: str = "ridge",
    max_pick_correlation: float | None = None,
    weight_fn: WeightFn | None = None,
    max_weight_multiple: float = 2.0,
) -> PurgedWalkForwardResult:
    """Walk-forward evaluation.

    v4.20: `model_type` (ridge | mlp | gbm, model.py), `max_pick_correlation`
    (near-duplicate guard, selection.py) and `weight_fn` (the weighting
    scheme, capped at `max_weight_multiple` x equal weight like live
    orders) are applied in every period exactly as the live run applies
    them, so what is backtested is what is traded.

    The model is TRAINED on forward_return_days-ahead targets, but every
    test period is SCORED on the return actually earned while holding
    the picks until the next rebalance (rebalance_days). With a 21-day
    target and weekly rebalancing, scoring 21-day returns every 5 days
    would count each price move ~4 times and inflate every t-statistic.

    Validation switches (never used for live trading):
      entry_delay_days  enter `d` days after the signal instead of at the
                        signal-day close (robustness to execution timing)
      shuffle_seed      randomly re-assign each date's feature rows to
                        different tickers ("placebo"): keeps everything else
                        identical but destroys any real signal
      record_cross_sections  keep every period's predictions + outcomes
    """
    if entry_delay_days < 0:
        raise ValueError("entry_delay_days must be >= 0")
    if purge_days < forward_return_days:
        logger.warning(
            f"purge_days ({purge_days}) is less than forward_return_days "
            f"({forward_return_days}): training labels near the fold "
            f"boundary may overlap the test window. Recommend "
            f"purge_days >= forward_return_days."
        )

    n = len(data)
    if n < MIN_BARS_REQUIRED + forward_return_days + test_window_days:
        raise InsufficientHistoryError(
            f"Purged walk-forward needs at least "
            f"{MIN_BARS_REQUIRED + forward_return_days + test_window_days} bars, "
            f"have {n}."
        )

    # Features are causal (trailing windows) and normalised per date, so
    # one panel over the full history is identical, row for row, to
    # recomputing it inside every fold -- just ~50x faster. Labels are
    # NOT causal; which labels each fold may use is controlled strictly
    # by last_label_idx below.
    panel = build_feature_panel(data, groups, dollar_volume=dollar_volume,
                                feature_set=feature_set)
    if shuffle_seed is not None:
        panel = _shuffle_within_dates(panel, shuffle_seed)
    # return earned from entry (signal day + delay) until the next rebalance
    hold = (data.shift(-(entry_delay_days + rebalance_days))
            / data.shift(-entry_delay_days) - 1)
    panel_dates = panel.index.get_level_values("date")

    fold_id = 0
    folds: list[FoldResult] = []
    p_ret: list[float] = []
    p_bench: list[float] = []
    p_ls: list[float] = []
    p_ic: list[float] = []
    p_dates: list[pd.Timestamp] = []
    p_end: list[pd.Timestamp] = []
    cross_sections: list = []
    n_fallback = 0
    train_end = max(min_train_days, MIN_BARS_REQUIRED + forward_return_days + 1)

    while True:
        test_start = train_end + purge_days
        test_end = min(test_start + test_window_days, n)
        if test_start + test_window_days > n:
            break

        # A label at feature index i uses the close at i + h. For it to be
        # known strictly before train_end: i + h <= train_end - 1.
        last_label_idx = train_end - 1 - forward_return_days
        try:
            X_train, y_train = build_training_set(
                data,
                forward_return_days=forward_return_days,
                step=rebalance_days,
                min_history_days=MIN_BARS_REQUIRED,
                target=target,
                groups=groups,
                panel=panel,
                last_label_idx=last_label_idx,
                dollar_volume=dollar_volume,
                feature_set=feature_set,
            )
        except InsufficientHistoryError:
            logger.debug(f"Skipping fold ending at train_end={train_end}: "
                         f"not enough clean training rows yet")
            train_end = test_end + embargo_days
            continue

        model = RankingModel(alpha=model_alpha, random_seed=random_seed,
                             model_type=model_type,
                             purge_dates=math.ceil(forward_return_days / rebalance_days))
        model.fit(X_train, y_train)

        fold_id += 1
        strat_rets, bench_rets, ics = [], [], []

        for idx in range(test_start, test_end, rebalance_days):
            if idx + entry_delay_days + rebalance_days >= n:
                break
            date = data.index[idx]
            if date not in panel_dates:
                continue
            feats = panel.xs(date, level="date")
            if len(feats) < top_n:
                continue

            preds = model.predict(feats)
            realized = hold.loc[date].reindex(feats.index)
            ok = realized.notna()
            preds, realized = preds[ok], realized[ok]
            if len(preds) < top_n:
                continue
            ranked = preds.sort_values(ascending=False)
            too_similar = (correlation_filter(data, idx, max_pick_correlation)
                           if max_pick_correlation else None)
            picks = capped_top(ranked.index, top_n, sectors, max_per_sector, too_similar)

            if weight_fn is None:
                period_strat = float(realized[picks].mean())
            else:
                try:
                    w = weight_fn(data, picks, idx).reindex(picks).astype(float)
                    if w.isna().any() or (w < 0).any() or w.sum() <= 0:
                        raise ValueError("weighting returned invalid weights")
                    w = cap_weights(w, max_weight_multiple / len(picks))
                except (DataError, ValueError) as e:
                    n_fallback += 1
                    logger.debug(f"{date.date()}: weighting failed ({e}); equal weight")
                    w = equal_weight(picks)
                period_strat = float((realized[picks] * w).sum())
            period_bench = float(realized.mean())
            q = max(1, len(ranked) // 5)
            period_ls = float(realized[ranked.head(q).index].mean()
                              - realized[ranked.tail(q).index].mean())
            ic = (float(spearmanr(preds, realized)[0])
                  if len(preds) >= 3 and preds.nunique() > 1 else np.nan)

            strat_rets.append(period_strat)
            bench_rets.append(period_bench)
            if np.isfinite(ic):
                ics.append(ic)
            p_ret.append(period_strat)
            p_bench.append(period_bench)
            p_ls.append(period_ls)
            p_ic.append(ic)
            p_dates.append(date)
            p_end.append(data.index[idx + entry_delay_days + rebalance_days])
            if record_cross_sections:
                cross_sections.append((date, preds, realized))

        if strat_rets:
            folds.append(FoldResult(
                fold_id=fold_id,
                train_end_date=data.index[train_end - 1],
                test_start_date=data.index[test_start],
                test_end_date=data.index[test_end - 1],
                n_train_samples=len(X_train),
                strategy_return=float(np.prod([1 + r for r in strat_rets]) - 1),
                benchmark_return=float(np.prod([1 + r for r in bench_rets]) - 1),
                information_coefficient=(
                    float(np.mean(ics)) if ics else float("nan")
                ),
                train_last_feature_date=X_train.index.get_level_values("date").max(),
            ))

        train_end = test_end + embargo_days

    if not folds:
        raise InsufficientHistoryError(
            "Purged walk-forward produced zero valid folds: data history "
            "is too short for the configured train/test/purge/embargo "
            "windows. Shorten test_window_days or provide more history."
        )

    overall_strategy = float(np.prod([1 + f.strategy_return for f in folds]) - 1)
    overall_benchmark = float(np.prod([1 + f.benchmark_return for f in folds]) - 1)
    win_rate = float(np.mean([f.strategy_return > 0 for f in folds]))

    idx_dates = pd.DatetimeIndex(p_dates)
    period_returns = pd.Series(p_ret, index=idx_dates, name="strategy_period_return")
    rets = period_returns.to_numpy()
    periods_per_year = 252 / rebalance_days
    sharpe = (float(rets.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year))
              if len(rets) > 1 and rets.std(ddof=1) > 0 else 0.0)

    ic_series = pd.Series(p_ic, index=idx_dates, name="ic")
    mean_ic = float(ic_series.mean()) if ic_series.notna().any() else float("nan")

    result = PurgedWalkForwardResult(
        folds=folds,
        overall_strategy_return=overall_strategy,
        overall_benchmark_return=overall_benchmark,
        overall_win_rate=win_rate,
        overall_sharpe=sharpe,
        mean_information_coefficient=mean_ic,
        period_returns=period_returns,
        benchmark_period_returns=pd.Series(p_bench, index=idx_dates, name="benchmark"),
        long_short_period_returns=pd.Series(p_ls, index=idx_dates, name="long_short"),
        period_ic=ic_series,
        period_end_dates=pd.Series(pd.DatetimeIndex(p_end), index=idx_dates,
                                   name="period_end"),
        cross_sections=cross_sections,
        weight_fallbacks=n_fallback,
    )
    if n_fallback:
        logger.warning(f"Weighting scheme fell back to equal weight in {n_fallback} of "
                       f"{len(period_returns)} periods")
    logger.info(f"Purged walk-forward complete: {len(folds)} folds, "
                f"{len(period_returns)} periods, mean IC={mean_ic:+.4f} "
                f"(t={result.ic_tstat:+.2f})")
    return result
