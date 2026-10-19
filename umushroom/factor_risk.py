"""Factor-risk-parity weights (v4.20).

Equal weight treats every pick as one bet. It is not: on 2026-10-19 the
ETF sleeve's eight "bets" were two MSCI Korea trackers, three USD
high-yield bond funds, a battery/hydrogen fund, tin and a 80/20
multi-asset fund -- about four distinct exposures. This module sizes
the picks by the RISK FACTORS they load on instead of by name:

  1. FACTORS. One return series per group of the sleeve's universe (its
     asset classes for ETFs: equity, bond, commodity, ...; the exchanges
     for stocks): the equal-weight daily return of every member with a
     price that day. Built from the sleeve's own prices, so it needs no
     extra data and works inside the walk-forward backtest.
  2. EXPOSURES. Each pick's betas to the factors from an OLS of its daily
     returns on the factor returns over the trailing year, plus its
     residual (idiosyncratic) variance.
  3. FACTOR COVARIANCE. The factors are correlated (bond and equity
     averages move together in a rates shock): their full covariance
     matrix, not just their variances, prices every pick's exposure.
  4. WEIGHTS. With b = B'w the portfolio's factor exposures, the
     contribution of factor k to systematic variance is
     b_k (Sigma_F b)_k (Euler decomposition). Weights are chosen so that
     those contributions match the risk BUDGETS -- equal by default --
     subject to long-only, fully invested and a per-name cap
     (Roncalli & Weisang 2016, "Risk parity portfolios with risk
     factors"). Two near-identical picks share one factor exposure and
     are sized as one bet; a pick that alone carries a whole factor gets
     that factor's full budget.

Everything is causal: only prices up to and including `as_of_idx` are
used (the same discipline as features.py, tested the same way).
Whether the scheme beats equal weight is NOT assumed -- the walk-forward
scores it exactly as traded (`weight_fn` in purged_backtest.py) and the
pre-declared rule in `--compare-weighting` decides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from umushroom.exceptions import DataError
from umushroom.logging_config import get_logger

logger = get_logger("factor_risk")

FACTOR_LOOKBACK_DAYS = 252   # trailing window for factors and exposures
MIN_FACTOR_OBS = 60          # fewer overlapping days = not estimable
MIN_GROUP_MEMBERS = 5        # a group factor needs this many members priced
MIN_EXPOSURE = 0.10          # a factor nobody in the book loads on gets no budget
# A factor with (almost) no variance cannot carry risk, but as an OLS regressor
# it explodes every beta (the money-market average: ~0.02%/day; the first live
# run showed betas of +7.7 on it). Below this daily vol a group is not a factor.
MIN_FACTOR_DAILY_VOL = 0.0005
# Bounds on each pick's weight as multiples of equal weight: the cap is the one
# every scheme gets in allocation.py; the floor keeps every pick investable
# (a 2% slice of a sleeve cannot buy one unit of most ETFs) and stops the
# optimiser from parking a pick at zero. Applied in the backtest and live.
MAX_WEIGHT_MULTIPLE = 2.0
MIN_WEIGHT_MULTIPLE = 0.25
TIE_BREAK = 1e-3             # pull toward equal weight among equally good solutions
N_STARTS = 6                 # seeded random starts besides equal weight (non-convex)


@dataclass(frozen=True)
class FactorRisk:
    weights: pd.Series            # per pick, sums to 1
    exposures: pd.DataFrame       # picks x factors (betas)
    factor_cov: pd.DataFrame      # factors x factors, daily return units
    contributions: pd.Series      # share of systematic variance per factor
    budgets: pd.Series            # the targets those shares were fitted to
    idiosyncratic_share: float    # share of total variance the factors miss
    residual_var: pd.Series       # per pick, daily

    def summary(self) -> str:
        lines = ["Factor-risk parity weights (share of systematic risk per factor):"]
        for f in self.budgets.index:
            lines.append(f"  {f:<14} contribution {self.contributions[f]:6.1%}   "
                         f"budget {self.budgets[f]:6.1%}")
        lines.append(f"  idiosyncratic (not explained by the factors): "
                     f"{self.idiosyncratic_share:.1%} of total variance")
        lines.append("  exposures (beta to each factor) and weight:")
        cols = list(self.exposures.columns)
        for t in self.weights.index:
            betas = "  ".join(f"{c}={self.exposures.loc[t, c]:+.2f}" for c in cols)
            lines.append(f"    {t:<12} w={self.weights[t]:5.1%}   {betas}")
        return "\n".join(lines)


def group_factor_returns(prices: pd.DataFrame, groups: dict[str, str], as_of_idx: int,
                         lookback: int = FACTOR_LOOKBACK_DAYS,
                         min_members: int = MIN_GROUP_MEMBERS) -> pd.DataFrame:
    """Equal-weight daily return of each group, from prices up to and
    including `as_of_idx` only. Groups with too few members or too few
    days are left out; fewer than two usable factors is an error (there
    would be nothing to balance)."""
    if not 0 <= as_of_idx < len(prices):
        raise ValueError(f"as_of_idx {as_of_idx} outside the price history")
    window = prices.iloc[max(0, as_of_idx + 1 - lookback): as_of_idx + 1]
    rets = window.pct_change(fill_method=None).iloc[1:]
    out: dict[str, pd.Series] = {}
    for g in sorted({v for v in groups.values() if v}):
        cols = [c for c in rets.columns if groups.get(c) == g]
        if len(cols) < min_members:
            continue
        r = rets[cols]
        mean = r.mean(axis=1).where(r.notna().sum(axis=1) >= min_members)
        if int(mean.notna().sum()) < MIN_FACTOR_OBS:
            continue
        if float(mean.std()) < MIN_FACTOR_DAILY_VOL:
            logger.debug(f"factor {g!r} dropped: daily vol {mean.std():.5f} below "
                         f"{MIN_FACTOR_DAILY_VOL}")
            continue
        out[g] = mean
    if len(out) < 2:
        raise DataError(f"need at least 2 factors with {min_members}+ members, "
                        f"{MIN_FACTOR_OBS}+ days and daily vol >= "
                        f"{MIN_FACTOR_DAILY_VOL}; have {sorted(out)}")
    return pd.DataFrame(out)


def factor_exposures(asset_returns: pd.DataFrame, factor_returns: pd.DataFrame
                     ) -> tuple[pd.DataFrame, pd.Series]:
    """OLS betas of each asset on the factors (with intercept) and the
    residual variance. Returns (exposures: assets x factors, residual_var)."""
    names = list(factor_returns.columns)
    betas: dict[str, np.ndarray] = {}
    resid: dict[str, float] = {}
    for t in asset_returns.columns:
        d = pd.concat([asset_returns[t].rename("__r__"), factor_returns], axis=1).dropna()
        if len(d) < MIN_FACTOR_OBS:
            raise DataError(f"{t}: only {len(d)} days overlap the factors; "
                            f"need {MIN_FACTOR_OBS}")
        X = np.column_stack([np.ones(len(d)), d[names].to_numpy(dtype=float)])
        y = d["__r__"].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        dof = max(len(d) - X.shape[1], 1)
        betas[t] = coef[1:]
        resid[t] = float((res @ res) / dof)
    exposures = pd.DataFrame(betas, index=names).T
    exposures.index.name = None
    return exposures, pd.Series(resid, name="residual_var")


def factor_contributions(weights: pd.Series, exposures: pd.DataFrame,
                         factor_cov: pd.DataFrame) -> pd.Series:
    """Variance contributed by each factor: b_k (Sigma_F b)_k with b = B'w."""
    b = exposures.T.to_numpy() @ weights.reindex(exposures.index).to_numpy(dtype=float)
    rc = b * (factor_cov.loc[exposures.columns, exposures.columns].to_numpy() @ b)
    return pd.Series(rc, index=exposures.columns, name="variance_contribution")


def factor_parity_weights(exposures: pd.DataFrame, factor_cov: pd.DataFrame,
                          budgets: pd.Series | None = None,
                          max_weight: float | None = None,
                          min_weight: float | None = None) -> tuple[pd.Series, pd.Series]:
    """Long-only, fully invested weights whose factor risk contributions
    match `budgets` (equal by default) as closely as the book allows, each
    weight within [min_weight, max_weight] (default MIN/MAX_WEIGHT_MULTIPLE
    times equal weight). Returns (weights, budgets actually used). Factors
    no pick is exposed to (|beta| < MIN_EXPOSURE for every pick) are
    dropped first: a budget nobody can fill would only distort the rest."""
    n = len(exposures)
    if n < 2:
        raise ValueError("factor parity needs at least 2 assets")
    if exposures.index.duplicated().any():
        raise ValueError("duplicate assets")
    floor = MIN_WEIGHT_MULTIPLE / n if min_weight is None else float(min_weight)
    if floor < 0 or floor * n > 1.0:
        raise ValueError(f"min_weight {floor} is infeasible for {n} assets")
    active = [f for f in exposures.columns
              if (exposures[f].abs() >= MIN_EXPOSURE).any()]
    if len(active) < 2:
        raise DataError(f"the picks are exposed to {len(active)} factor(s) "
                        f"({active}); nothing to balance")
    B = exposures[active].to_numpy(dtype=float)
    cov = factor_cov.loc[active, active].to_numpy(dtype=float)
    if budgets is None:
        target = np.full(len(active), 1.0 / len(active))
    else:
        tb = budgets.reindex(active).fillna(0.0).to_numpy(dtype=float)
        if tb.sum() <= 0 or (tb < 0).any():
            raise ValueError("budgets must be non-negative with a positive total")
        target = tb / tb.sum()
    cap = MAX_WEIGHT_MULTIPLE / n if max_weight is None else float(max_weight)
    if cap * n < 1.0:
        cap = 1.0 / n
    floor = min(floor, cap)
    # Contributions are QUADRATIC in a lone asset's weight, so matching them
    # directly has a spurious stationary point at weight zero (the optimiser
    # would accept "all in one factor"). Matching square-root contributions
    # to square-root budgets has the same optimum (rc_k proportional to
    # budget_k) and is linear near zero. The problem is still non-convex,
    # hence several seeded starts; a tiny tie-breaker toward equal weight
    # picks the most diversified point when several are equally good.
    sqrt_target = np.sqrt(target)
    sqrt_target /= sqrt_target.sum()
    equal = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        b = B.T @ w
        rc = b * (cov @ b)
        t = np.sign(rc) * np.sqrt(np.abs(rc))
        tot = t.sum()
        if tot <= 0:
            return 1.0 + float(w @ w)            # no systematic risk: penalise
        dev = float(((t / tot - sqrt_target) ** 2).sum())
        return dev + TIE_BREAK * float(((w - equal) ** 2).sum())

    rng = np.random.default_rng(0)
    starts = [equal] + [rng.dirichlet(np.ones(n)) for _ in range(N_STARTS)]
    best = None
    for x0 in starts:
        res = minimize(objective, np.clip(x0, floor, cap), method="SLSQP",
                       bounds=[(floor, cap)] * n,
                       constraints=[{"type": "eq",
                                     "fun": lambda w: float(w.sum() - 1.0)}],
                       options={"maxiter": 500, "ftol": 1e-14})
        if np.all(np.isfinite(res.x)) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise DataError("factor parity optimisation produced invalid weights")
    w = np.clip(best.x, floor, cap)
    if w.sum() <= 0:
        raise DataError("factor parity optimisation produced invalid weights")
    w = w / w.sum()
    return (pd.Series(w, index=exposures.index, name="weight"),
            pd.Series(target, index=active, name="budget"))


def factor_risk(prices: pd.DataFrame, tickers: list[str], as_of_idx: int,
                groups: dict[str, str], lookback: int = FACTOR_LOOKBACK_DAYS,
                budgets: pd.Series | None = None,
                max_weight: float | None = None,
                min_weight: float | None = None) -> FactorRisk:
    """The whole pipeline for one sizing decision, using only prices up to
    and including `as_of_idx`. `prices` should be the sleeve's full
    universe (the factors are its group averages); `tickers` the picks."""
    if len(tickers) < 2:
        raise ValueError("factor-risk weights need at least 2 tickers")
    if len(set(tickers)) != len(tickers):
        raise ValueError("duplicate tickers")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise DataError(f"tickers not in the price data: {missing}")
    factors = group_factor_returns(prices, groups, as_of_idx, lookback)
    window = prices.iloc[max(0, as_of_idx + 1 - lookback): as_of_idx + 1][tickers]
    asset_rets = window.pct_change(fill_method=None).iloc[1:]
    exposures, resid = factor_exposures(asset_rets, factors)
    cov = factors.cov()
    weights, used = factor_parity_weights(exposures, cov, budgets, max_weight, min_weight)
    rc = factor_contributions(weights, exposures[used.index], cov)
    systematic = float(rc.sum())
    idio = float((weights.to_numpy() ** 2 * resid.reindex(weights.index).to_numpy()).sum())
    total = systematic + idio
    shares = (rc / systematic) if systematic > 0 else rc * 0.0
    return FactorRisk(weights=weights, exposures=exposures, factor_cov=cov,
                      contributions=shares.rename("share"), budgets=used,
                      idiosyncratic_share=(idio / total if total > 0 else float("nan")),
                      residual_var=resid)


def factor_weight(prices: pd.DataFrame, tickers: list[str], as_of_idx: int,
                  groups: dict[str, str], **kwargs) -> pd.Series:
    """`WeightFn`-shaped entry point (prices, picks, as_of_idx) -> weights."""
    return factor_risk(prices, tickers, as_of_idx, groups, **kwargs).weights
