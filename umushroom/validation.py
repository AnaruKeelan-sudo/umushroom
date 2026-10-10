"""Validation suite: extra backtests that try to BREAK the result.

None of this changes the model. Each check asks a different "could the
alpha be fake?" question and reports PASS / WARN / FAIL:

  1. Placebo (permutation) test -- re-run the identical walk-forward N times
     with each date's features shuffled across stocks. A real signal must
     beat (nearly) all of these; luck or a pipeline artefact won't.
  2. Decile monotonicity -- sort each week's stocks into 10 groups by
     prediction. Real skill shows returns rising from the bottom to the top
     group, not just one lucky extreme.
  3. Stability over time -- IC and active return per calendar year. Alpha
     that comes from one lucky year is not a strategy.
  4. Every market -- IC inside each exchange separately (LSE, SIX, US).
  5. Execution delay -- enter 1 and 2 days AFTER the signal. A genuine
     effect decays gradually; one that vanishes instantly was probably
     exploiting a data timing quirk.
  6. Benchmark sanity -- the equal-weight universe's own return must be
     plausible; absurd values mean bad price data (e.g. pence/pound flips).
  7. Concentration -- how often the same names are picked; a "strategy"
     that is really 2-3 stocks held forever is a bet, not a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from umushroom.factor_regression import benchmark_alpha
from umushroom.logging_config import get_logger
from umushroom.purged_backtest import PurgedWalkForwardResult, run_purged_walk_forward

logger = get_logger("validation")

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


@dataclass
class ValidationReport:
    sleeve: str
    checks: list[Check] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    def summary(self) -> str:
        lines = [f"=== Validation suite: {self.sleeve} sleeve ==="]
        for c in self.checks:
            lines.append(f"[{c.status}] {c.name}: {c.detail}")
        for title, table in self.tables.items():
            lines.append(f"\n{title}\n{table.to_string()}")
        n_fail = sum(c.status == FAIL for c in self.checks)
        n_warn = sum(c.status == WARN for c in self.checks)
        lines.append(f"\nResult: {len(self.checks) - n_fail - n_warn} pass, "
                     f"{n_warn} warn, {n_fail} fail")
        return "\n".join(lines)


# ----------------------------------------------------------------- helpers

def _alpha_t(r: PurgedWalkForwardResult, periods_per_year: float) -> float:
    try:
        return benchmark_alpha(r.period_returns, r.benchmark_period_returns,
                               periods_per_year).alpha_tstat
    except ValueError:
        return float("nan")


def decile_returns(r: PurgedWalkForwardResult, n_groups: int = 10) -> pd.Series:
    """Mean realised return per prediction decile (1 = lowest predicted)."""
    rows = []
    for _date, preds, realized in r.cross_sections:
        if len(preds) < n_groups:
            continue
        g = pd.qcut(preds.rank(method="first"), n_groups, labels=False) + 1
        rows.append(realized.groupby(g).mean())
    if not rows:
        return pd.Series(dtype=float)
    out = pd.concat(rows, axis=1).mean(axis=1)
    out.index.name = "decile"
    return out.rename("mean_return_per_period")


def yearly_stats(r: PurgedWalkForwardResult) -> pd.DataFrame:
    df = pd.DataFrame({"ic": r.period_ic,
                       "active": r.period_returns - r.benchmark_period_returns})
    g = df.groupby(df.index.year)
    out = pd.DataFrame({
        "periods": g.size(),
        "mean_IC": g["ic"].mean(),
        "IC_hit_rate": g["ic"].apply(lambda x: float((x > 0).mean())),
        "active_per_period": g["active"].mean(),
    })
    out.index.name = "year"
    return out


def group_ic(r: PurgedWalkForwardResult, groups: dict[str, str],
             min_size: int = 5) -> pd.Series:
    """Mean per-period Spearman IC computed within each group separately."""
    acc: dict[str, list[float]] = {}
    for _date, preds, realized in r.cross_sections:
        labels = pd.Series({t: groups.get(t) for t in preds.index})
        for g, members in labels.groupby(labels):
            idx = members.index
            if len(idx) >= min_size:
                ic = spearmanr(preds[idx], realized[idx])[0]
                if np.isfinite(ic):
                    acc.setdefault(str(g), []).append(float(ic))
    return pd.Series({g: float(np.mean(v)) for g, v in acc.items()},
                     name="mean_IC").sort_index()


def pick_concentration(r: PurgedWalkForwardResult, top_n: int) -> pd.Series:
    """How often each ticker was in the top-N picks (share of periods)."""
    counts: dict[str, int] = {}
    n = 0
    for _date, preds, _realized in r.cross_sections:
        n += 1
        for t in preds.sort_values(ascending=False).head(top_n).index:
            counts[t] = counts.get(t, 0) + 1
    return (pd.Series(counts, dtype=float) / max(n, 1)).sort_values(ascending=False)


# ------------------------------------------------------------------- suite

def run_validation(
    data: pd.DataFrame,
    sleeve: str,
    backtest_kwargs: dict[str, Any],
    groups_for_report: dict[str, str] | None = None,
    placebo_runs: int = 20,
    delays: tuple[int, ...] = (1, 2),
    benchmark_annual_range: tuple[float, float] = (-0.6, 1.0),
) -> ValidationReport:
    """Run every check for one sleeve. `backtest_kwargs` are exactly the
    arguments of run_purged_walk_forward used for the live model."""
    ppy = 252 / backtest_kwargs["rebalance_days"]
    top_n = backtest_kwargs["top_n"]
    rep = ValidationReport(sleeve)

    real = run_purged_walk_forward(data, record_cross_sections=True, **backtest_kwargs)
    real_t = _alpha_t(real, ppy)
    real_active = float(real.active_returns.mean())

    # 1. placebo / permutation test
    if placebo_runs > 0:
        placebo_active = []
        for seed in range(placebo_runs):
            pr = run_purged_walk_forward(data, shuffle_seed=seed, **backtest_kwargs)
            placebo_active.append(float(pr.active_returns.mean()))
        arr = np.array(placebo_active)
        # +1 smoothing: the real run counts as one of the possible outcomes
        p = float((np.sum(arr >= real_active) + 1) / (len(arr) + 1))
        status = PASS if p <= 0.05 else WARN if p <= 0.20 else FAIL
        rep.checks.append(Check(
            "Placebo test", status,
            f"real active return {real_active:+.3%}/period beat "
            f"{int(np.sum(arr < real_active))}/{len(arr)} shuffled-signal runs "
            f"(p = {p:.3f}; placebo mean {arr.mean():+.3%})"))

    # 2. deciles
    dec = decile_returns(real)
    if len(dec) >= 5:
        rho = float(spearmanr(dec.index, dec.to_numpy())[0])
        spread = float(dec.iloc[-1] - dec.iloc[0])
        status = PASS if rho >= 0.6 and spread > 0 else WARN if spread > 0 else FAIL
        rep.checks.append(Check(
            "Decile monotonicity", status,
            f"top-minus-bottom decile {spread:+.3%}/period; rank correlation of "
            f"decile vs return {rho:+.2f} (1.0 = perfectly ordered)"))
        rep.tables["Mean return per period by prediction decile (1 = lowest):"] = \
            dec.to_frame()

    # 3. stability over time
    yearly = yearly_stats(real)
    if len(yearly):
        good = int((yearly["mean_IC"] > 0).sum())
        share = good / len(yearly)
        status = PASS if share >= 0.7 else WARN if share >= 0.5 else FAIL
        rep.checks.append(Check(
            "Stable over time", status,
            f"IC positive in {good}/{len(yearly)} calendar years"))
        rep.tables["By calendar year:"] = yearly

    # 4. every market
    if groups_for_report:
        gic = group_ic(real, groups_for_report)
        if len(gic):
            neg = gic[gic <= 0].index.tolist()
            status = PASS if not neg else WARN if len(neg) < len(gic) else FAIL
            rep.checks.append(Check(
                "Works in every market", status,
                ", ".join(f"{g} IC {v:+.3f}" for g, v in gic.items())))

    # 5. execution delay
    delay_rows = {0: (real_active, real_t)}
    for d in delays:
        dr = run_purged_walk_forward(data, entry_delay_days=d, **backtest_kwargs)
        delay_rows[d] = (float(dr.active_returns.mean()), _alpha_t(dr, ppy))
    d1_active = delay_rows.get(1, (np.nan, np.nan))[0]
    if 1 in delay_rows and np.isfinite(d1_active):
        status = PASS if d1_active > 0 else WARN if real_active <= 0 else FAIL
        rep.checks.append(Check(
            "Survives 1-day execution delay", status,
            "; ".join(f"delay {d}d: active {a:+.3%}, alpha t {t:+.2f}"
                      for d, (a, t) in delay_rows.items())))

    # 6. benchmark sanity
    bench_ann = real._annualise(real.overall_benchmark_return)
    lo, hi = benchmark_annual_range
    status = PASS if lo <= bench_ann <= hi else FAIL
    rep.checks.append(Check(
        "Benchmark plausible (data sanity)", status,
        f"equal-weight universe {bench_ann:+.1%} a year "
        f"(allowed range {lo:+.0%} .. {hi:+.0%})"))

    # 7. concentration
    conc = pick_concentration(real, top_n)
    if len(conc):
        top_share = float(conc.iloc[0])
        status = PASS if top_share <= 0.5 else WARN
        rep.checks.append(Check(
            "Not dominated by a few names", status,
            f"{len(conc)} different instruments picked; most frequent "
            f"{conc.index[0]} in {top_share:.0%} of periods"))

    return rep
