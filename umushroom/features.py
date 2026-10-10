"""Factor / feature engineering (v4).

Every function here is a PURE function of a price DataFrame: no I/O, no
global state. Every raw factor is a TRAILING rolling computation, so the
value at date t depends only on prices at or before t. This NO-LOOKAHEAD
property is unit-tested directly (features at t are identical whether or
not rows after t exist).

What changed from v3, and why it matters for real alpha
---------------------------------------------------------
1. Factors replaced with ones that have decades of published,
   out-of-sample evidence, instead of overlapping short-horizon
   momentum/MA/RSI variants (which are mostly the same signal 4 times):

     mom_12_1        12-month return skipping the latest month
                     (Jegadeesh & Titman 1993; Carhart 1997)
     mom_6_1         6-month return skipping the latest month
     resid_mom_12_1  momentum of the stock's market-beta-adjusted
                     (idiosyncratic) returns, volatility-scaled
                     (Blitz, Huij & Martens 2011) -- crashes far less
                     than raw momentum in market reversals
     high_52w_ratio  price / 52-week high (George & Hwang 2004)
     reversal_21d    minus last-month return: short-term reversal
                     (Jegadeesh 1990; Lehmann 1990)
     reversal_5d     minus last-week return: matches the 5-day horizon
     vol_60d         60-day realised volatility (low-vol anomaly,
                     Ang et al. 2006 -- model learns a NEGATIVE weight)
     beta_252        1-year market beta ("betting against beta",
                     Frazzini & Pedersen 2014)
     rsi_14          kept from v3 as a short-horizon overbought signal

2. CROSS-SECTIONAL normalisation. Each factor is converted, on each
   date, to its centred percentile rank across the universe in
   [-0.5, +0.5]. v3 fed RAW values to Ridge: RSI (0-100) and volatility
   (~0.01) on wildly different scales, so the L2 penalty shrank them
   unequally, and a market-wide move shifted every stock's raw value at
   once. Ranks are scale-free, outlier-proof and compare stocks only to
   each other on the same day -- which is the actual decision being made.

3. Exchange-neutral ranking (optional, default on). Ranks are computed
   WITHIN each exchange, so the model compares a London stock to London
   stocks, not to Swiss stocks whose returns also contain a GBP/CHF
   currency move and a different country's market move.

4. TARGET is the cross-sectional rank of the forward return (default),
   or forward return minus the universe mean ("excess"). v3 trained on
   RAW forward returns pooled over dates, so most of what the model
   was asked to predict was "will the whole market go up this week" --
   unforecastable noise that swamped the stock-selection signal.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from umushroom.exceptions import InsufficientHistoryError
from umushroom.logging_config import get_logger

logger = get_logger("features")

FEATURE_NAMES: list[str] = [
    "mom_12_1", "mom_6_1", "resid_mom_12_1", "high_52w_ratio",
    "reversal_21d", "reversal_5d", "vol_60d", "beta_252", "rsi_14",
]

# v4.4 volume-based factors (equity sleeve, when volume data is available):
#   abn_volume_21_252  log(avg traded value last month / last year):
#                      "high-volume return premium" (Gervais, Kaniel &
#                      Mingelgrin 2001) -- unusual attention predicts returns
#   amihud_illiq_63    log of avg |daily return| per unit traded value over
#                      3 months (Amihud 2002) -- illiquidity premium
VOLUME_FEATURE_NAMES: list[str] = ["abn_volume_21_252", "amihud_illiq_63"]


def feature_names(with_volume: bool) -> list[str]:
    return FEATURE_NAMES + (VOLUME_FEATURE_NAMES if with_volume else [])


TargetType = Literal["rank", "excess", "raw"]
TARGET_TYPES: tuple[str, ...] = ("rank", "excess", "raw")

YEAR = 252
SKIP_MONTH = 21
# Prices needed for every factor to be defined on the last row: 252
# daily returns need 253 prices.
MIN_BARS_REQUIRED = YEAR + 1
# Exchanges with fewer members than this are pooled together for ranking
# rather than ranked alone (a rank among 3 stocks is mostly noise).
MIN_GROUP_SIZE = 5
POOLED_GROUP = "__pooled__"


# ------------------------------------------------------------------ helpers

def compute_rsi(series: pd.Series | pd.DataFrame, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)  # neutral RSI when undefined (flat/insufficient data)


def effective_groups(tickers: list[str], groups: dict[str, str] | None
                     ) -> dict[str, str]:
    """Map ticker -> ranking group, pooling undersized groups. With no
    groups, every ticker is in one global group."""
    if not groups:
        return {t: POOLED_GROUP for t in tickers}
    sizes = pd.Series([groups.get(t, POOLED_GROUP) for t in tickers]).value_counts()
    out = {}
    for t in tickers:
        g = groups.get(t, POOLED_GROUP)
        out[t] = g if sizes.get(g, 0) >= MIN_GROUP_SIZE else POOLED_GROUP
    return out


def hierarchical_groups(tickers: list[str], levels: list[dict[str, str]]
                        ) -> dict[str, str]:
    """Most specific group with at least MIN_GROUP_SIZE members.

    `levels` run finest -> coarsest, e.g. [sector|exchange, sector]. A
    ticker is ranked within its finest group that is big enough, else the
    next level, else the global pool -- so a Swiss utility (one of two)
    is ranked against all utilities, not alone and not against banks.
    """
    out: dict[str, str] = {}
    sizes = [pd.Series([lvl.get(t) for t in tickers]).value_counts() for lvl in levels]
    for t in tickers:
        out[t] = POOLED_GROUP
        for lvl, sz in zip(levels, sizes):
            g = lvl.get(t)
            if g is not None and sz.get(g, 0) >= MIN_GROUP_SIZE:
                out[t] = g
                break
    return out


def centered_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (per-date) rank mapped linearly onto [-0.5, +0.5].
    NaNs stay NaN; a row with a single valid value maps to 0."""
    ranks = df.rank(axis=1, method="average")
    count = df.notna().sum(axis=1)
    denom = (count - 1).where(count > 1)
    out = (ranks - 1).div(denom, axis=0) - 0.5
    single = count == 1
    if single.any():
        out.loc[single] = out.loc[single].where(df.loc[single].isna(), 0.0)
    return out.where(df.notna())


def grouped_centered_rank(df: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for g in sorted({groups[c] for c in df.columns}):
        cols = [c for c in df.columns if groups[c] == g]
        out[cols] = centered_rank(df[cols])
    return out


def grouped_demean(df: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for g in sorted({groups[c] for c in df.columns}):
        cols = [c for c in df.columns if groups[c] == g]
        out[cols] = df[cols].sub(df[cols].mean(axis=1), axis=0)
    return out


# ------------------------------------------------------------- raw factors

def compute_raw_factors(data: pd.DataFrame,
                        dollar_volume: pd.DataFrame | None = None
                        ) -> dict[str, pd.DataFrame]:
    """All factors as (date x ticker) frames. Purely trailing windows.
    Tickers may have leading NaN (not yet listed): every rolling window
    requires a full window of real data, so a ticker's factors switch on
    only once it has enough history."""
    rets = data.pct_change(fill_method=None)
    mkt = rets.mean(axis=1)                       # equal-weight universe return

    mom_12_1 = data.shift(SKIP_MONTH) / data.shift(YEAR) - 1
    mom_6_1 = data.shift(SKIP_MONTH) / data.shift(126) - 1
    high_52w = data / data.rolling(YEAR).max()
    rev_21 = -(data / data.shift(SKIP_MONTH) - 1)
    rev_5 = -(data / data.shift(5) - 1)
    vol_60 = rets.rolling(60).std()

    # Rolling 1-year beta to the equal-weight universe.
    mean_r = rets.rolling(YEAR).mean()
    mean_m = mkt.rolling(YEAR).mean()
    mean_rm = rets.mul(mkt, axis=0).rolling(YEAR).mean()
    var_m = mkt.rolling(YEAR).var(ddof=0)
    cov_rm = mean_rm.sub(mean_r.mul(mean_m, axis=0))
    beta = cov_rm.div(var_m.where(var_m > 0), axis=0)

    # Residual (idiosyncratic) 12-1 momentum, volatility-scaled:
    #   sum of (r - beta*m) over [t-251, t-21] / (resid_sd * sqrt(231))
    window = YEAR - SKIP_MONTH
    sum_r = rets.shift(SKIP_MONTH).rolling(window).sum()
    sum_m = mkt.shift(SKIP_MONTH).rolling(window).sum()
    resid_sum = sum_r.sub(beta.mul(sum_m, axis=0))
    var_r = rets.rolling(YEAR).var(ddof=0)
    resid_var = (var_r - beta.pow(2).mul(var_m, axis=0)).clip(lower=1e-12)
    resid_mom = resid_sum / np.sqrt(resid_var * window)

    factors = {
        "mom_12_1": mom_12_1,
        "mom_6_1": mom_6_1,
        "resid_mom_12_1": resid_mom,
        "high_52w_ratio": high_52w,
        "reversal_21d": rev_21,
        "reversal_5d": rev_5,
        "vol_60d": vol_60,
        "beta_252": beta,
        "rsi_14": compute_rsi(data),
    }
    assert list(factors) == FEATURE_NAMES, "factor order drifted from FEATURE_NAMES"
    if dollar_volume is not None:
        dv = dollar_volume.reindex(index=data.index, columns=data.columns)
        avg_21 = dv.rolling(21, min_periods=15).mean()
        avg_252 = dv.rolling(YEAR, min_periods=200).mean()
        factors["abn_volume_21_252"] = np.log(avg_21 / avg_252)
        illiq = rets.abs() / dv
        illiq = illiq.where(np.isfinite(illiq))
        factors["amihud_illiq_63"] = np.log(
            illiq.rolling(63, min_periods=40).mean().where(lambda x: x > 0))
    return factors


def _stack(frames: dict[str, pd.DataFrame], dates: pd.Index) -> pd.DataFrame:
    """dict of (date x ticker) -> long frame indexed (date, ticker)."""
    cols = {}
    for name, f in frames.items():
        cols[name] = f.loc[dates].stack(future_stack=True)
    long = pd.DataFrame(cols)
    long.index.names = ["date", "ticker"]
    return long


def build_feature_panel(
    data: pd.DataFrame,
    groups: dict[str, str] | None = None,
    normalize: bool = True,
    dollar_volume: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalised features for every date that has full history,
    as a long frame indexed (date, ticker) with FEATURE_NAMES columns.
    Rows with any undefined factor are dropped."""
    if len(data) < MIN_BARS_REQUIRED:
        raise InsufficientHistoryError(
            f"Need at least {MIN_BARS_REQUIRED} bars to compute features, "
            f"have {len(data)}."
        )
    raw = compute_raw_factors(data, dollar_volume)
    if normalize:
        g = effective_groups(list(data.columns), groups)
        raw = {k: grouped_centered_rank(v, g) for k, v in raw.items()}
    dates = data.index[MIN_BARS_REQUIRED - 1:]
    long = _stack(raw, dates)
    return _neutral_fill_volume(long).dropna()


def _neutral_fill_volume(feats: pd.DataFrame) -> pd.DataFrame:
    """Missing volume factors -> 0 (the centre of the rank scale), so a
    stock Yahoo has no volume for is scored on its price factors alone
    instead of being dropped from the universe."""
    vol_cols = [c for c in VOLUME_FEATURE_NAMES if c in feats.columns]
    if vol_cols:
        feats = feats.copy()
        feats[vol_cols] = feats[vol_cols].fillna(0.0)
    return feats


def build_features_at_index(
    data: pd.DataFrame,
    as_of_idx: int,
    groups: dict[str, str] | None = None,
    normalize: bool = True,
    dollar_volume: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One feature row per ticker, using ONLY data[: as_of_idx + 1].

    Raises InsufficientHistoryError if fewer than MIN_BARS_REQUIRED bars
    exist up to as_of_idx. Tickers with an undefined factor are
    returned with NaN (callers dropna()).
    """
    if as_of_idx < MIN_BARS_REQUIRED - 1:
        raise InsufficientHistoryError(
            f"as_of_idx={as_of_idx} has only {as_of_idx + 1} bars of "
            f"history; need at least {MIN_BARS_REQUIRED}."
        )
    # Every factor uses at most MIN_BARS_REQUIRED trailing prices, so a
    # trailing window of exactly that length gives the same values as the
    # full history -- and is ~5x faster on multi-year data.
    window = data.iloc[as_of_idx + 1 - MIN_BARS_REQUIRED: as_of_idx + 1]
    dv_window = (None if dollar_volume is None
                 else dollar_volume.reindex(index=window.index, columns=window.columns))
    raw = compute_raw_factors(window, dv_window)
    names = list(raw)
    last = pd.DataFrame({k: v.iloc[-1] for k, v in raw.items()})
    if normalize:
        g = effective_groups(list(data.columns), groups)
        last = pd.DataFrame(
            {k: grouped_centered_rank(last[[k]].T, g).iloc[0] for k in names})
    last.index.name = None
    out = last[names]
    if normalize:
        # mirror build_feature_panel exactly (live features == trained features)
        has_price_factors = out[FEATURE_NAMES].notna().all(axis=1)
        out = _neutral_fill_volume(out)
        out.loc[~has_price_factors] = np.nan
    return out


# ------------------------------------------------------------------ labels

def compute_forward_returns(data: pd.DataFrame, forward_return_days: int) -> pd.DataFrame:
    """(date x ticker) return from close at t to close at t+h. The value
    at t is only KNOWN at t+h; callers must respect that when choosing
    which labels may be used for training (see purged_backtest)."""
    return data.shift(-forward_return_days) / data - 1  # NaN where unlisted


def make_target(fwd: pd.DataFrame, target: str, groups: dict[str, str] | None
                ) -> pd.DataFrame:
    if target not in TARGET_TYPES:
        raise ValueError(f"target must be one of {TARGET_TYPES}, got {target!r}")
    if target == "raw":
        return fwd
    g = effective_groups(list(fwd.columns), groups)
    if target == "excess":
        return grouped_demean(fwd, g)
    return grouped_centered_rank(fwd, g)


def build_training_set(
    data: pd.DataFrame,
    forward_return_days: int,
    step: int,
    min_history_days: int,
    target: str = "rank",
    groups: dict[str, str] | None = None,
    panel: pd.DataFrame | None = None,
    last_label_idx: int | None = None,
    dollar_volume: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """(features, target) pairs at every `step`-th date.

    X and y are indexed by (date, ticker) so downstream CV can split on
    DATES (never splitting one date's cross-section across folds).

    `panel` may be passed in precomputed (from build_feature_panel on the
    same `data`) to avoid recomputation in the walk-forward loop.
    `last_label_idx` caps the latest feature date used, so that the
    label (which needs prices forward_return_days later) is fully known
    by a given cut-off.
    """
    n = len(data)
    start_idx = max(min_history_days, MIN_BARS_REQUIRED) - 1
    end_idx = n - forward_return_days            # exclusive
    if last_label_idx is not None:
        end_idx = min(end_idx, last_label_idx + 1)

    if start_idx >= end_idx:
        raise InsufficientHistoryError(
            f"Not enough history to build any training samples: "
            f"have {n} bars, need > {start_idx + forward_return_days}."
        )

    if panel is None:
        panel = build_feature_panel(data, groups, dollar_volume=dollar_volume)
    fwd = compute_forward_returns(data, forward_return_days)
    y_wide = make_target(fwd, target, groups)

    sample_dates = data.index[list(range(start_idx, end_idx, step))]
    X = panel[panel.index.get_level_values("date").isin(sample_dates)]
    y_long = y_wide.loc[sample_dates].stack(future_stack=True)
    y_long.index.names = ["date", "ticker"]
    y = y_long.reindex(X.index)
    valid = y.notna()
    X, y = X[valid], y[valid].rename("target")

    if X.empty:
        raise InsufficientHistoryError(
            "No valid training rows survived feature/label construction "
            "(likely too many NaNs in the price history)."
        )

    logger.info(f"Built training set: {X.shape[0]} samples over "
                f"{X.index.get_level_values('date').nunique()} dates, "
                f"{X.shape[1]} features, target={target}")
    return X, y
