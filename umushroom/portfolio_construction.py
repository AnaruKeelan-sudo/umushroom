"""Portfolio weighting schemes.

Converts a set of selected tickers into WEIGHTS (summing to 1), as a
separate concern from WHICH tickers get selected (that's model.py's
job) and HOW weights become share counts (that's allocation.py's job).
Keeping this separate means swapping the weighting scheme never touches
ranking or share-rounding logic.

Two schemes:

  equal_weight: 1/N for N picks. Simple, no extra assumptions, but
    ignores that some picks may be far more volatile or far more
    correlated with each other than others.

  hrp_weight: Hierarchical Risk Parity (Lopez de Prado, 2016).
    Clusters assets by correlation structure and allocates more weight
    to less-correlated, lower-volatility clusters, without needing
    expected-return estimates (unlike Markowitz/Black-Litterman, which
    are notoriously sensitive to small errors in expected-return
    inputs).

Implementation note (v4): HRP is implemented NATIVELY on top of
scipy.cluster.hierarchy rather than via PyPortfolioOpt. PyPortfolioOpt
pulls in cvxpy, whose compiled solvers (ecos/scs/osqp) lag behind new
Python releases on prebuilt wheels -- it fails to install on Python
3.14. HRP itself needs no optimiser, so the dependency was never
necessary. The three steps of the algorithm are:

  1. Tree clustering: single-linkage clustering on the correlation
     distance d_ij = sqrt((1 - rho_ij) / 2).
  2. Quasi-diagonalisation: reorder assets by the dendrogram's leaf
     order so correlated assets sit next to each other.
  3. Recursive bisection: split the ordered list in half repeatedly,
     giving each half capital in inverse proportion to its cluster
     variance (computed with inverse-variance weights inside the
     cluster).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from umushroom.exceptions import DataError
from umushroom.logging_config import get_logger

logger = get_logger("portfolio_construction")

MIN_HRP_LOOKBACK_DAYS = 60   # need enough return history for a stable
                              # correlation/covariance estimate; below
                              # this, correlations are too noisy to
                              # cluster on meaningfully
HRP_COV_LOOKBACK_DAYS = 252  # covariance window: one trading year


def cap_weights(weights: pd.Series, cap: float) -> pd.Series:
    """Cap every weight at `cap` (e.g. 2x equal weight), redistributing the
    excess pro rata to uncapped names. v4.4's real run had HRP put 59% of
    the stock sleeve in one name; nothing should ever dominate like that.
    v4.20: shared by live sizing AND the walk-forward, so a weighting scheme
    is backtested exactly as it is traded.
    """
    if cap * len(weights) < 1.0:
        cap = 1.0 / len(weights)          # infeasible cap -> equal weight
    w = weights.astype(float) / weights.sum()
    for _ in range(len(w)):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over & (w < cap)
        if not under.any():
            break
        base = w[under]
        # pro rata to the uncapped names; equally if they are all zero
        share = base / base.sum() if base.sum() > 0 else pd.Series(1.0 / len(base),
                                                                   index=base.index)
        w[under] += excess * share
    return w / w.sum()


def equal_weight(tickers: list[str]) -> pd.Series:
    if not tickers:
        raise ValueError("Cannot compute weights for an empty ticker list")
    w = 1.0 / len(tickers)
    return pd.Series({t: w for t in tickers}, name="weight")


def _correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """Lopez de Prado's correlation distance, guaranteed a valid metric
    (symmetric, zero diagonal, non-negative) even under floating-point
    noise in the correlation matrix."""
    dist = np.sqrt(np.clip((1.0 - corr.to_numpy()) / 2.0, 0.0, 1.0))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return dist


def _inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _cluster_variance(cov: pd.DataFrame, members: list[str]) -> float:
    sub = cov.loc[members, members].to_numpy()
    w = _inverse_variance_weights(sub).reshape(-1, 1)
    # .item() rather than float(): float() on a (1,1) ndarray is an
    # error on newer numpy versions.
    return (w.T @ sub @ w).item()


def _recursive_bisection(cov: pd.DataFrame, ordered: list[str]) -> pd.Series:
    weights = pd.Series(1.0, index=ordered)
    clusters: list[list[str]] = [ordered]
    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            half = len(cluster) // 2
            left, right = cluster[:half], cluster[half:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters
    return weights


def hrp_weight(data: pd.DataFrame, tickers: list[str], as_of_idx: int) -> pd.Series:
    """Hierarchical Risk Parity weights for `tickers`, using only
    data up to and including as_of_idx (no lookahead -- same
    discipline as features.py).
    """
    if len(tickers) < 2:
        raise ValueError(
            f"HRP needs at least 2 tickers to compute a correlation "
            f"structure, got {len(tickers)}"
        )
    if len(set(tickers)) != len(tickers):
        raise ValueError("HRP received duplicate tickers")
    if as_of_idx < MIN_HRP_LOOKBACK_DAYS - 1:
        raise DataError(
            f"as_of_idx={as_of_idx} has only {as_of_idx + 1} bars of "
            f"history; HRP needs at least {MIN_HRP_LOOKBACK_DAYS} for a "
            f"stable covariance estimate."
        )

    missing = set(tickers) - set(data.columns)
    if missing:
        raise DataError(f"Tickers requested for HRP not present in data: {missing}")

    start = max(0, as_of_idx + 1 - HRP_COV_LOOKBACK_DAYS)
    window = data.iloc[start: as_of_idx + 1][tickers]
    returns = window.pct_change().iloc[1:]

    if returns.isna().any().any():
        raise DataError("NaNs in HRP return window; check input data")

    cov = returns.cov()
    variances = np.diag(cov.to_numpy())
    if not np.all(np.isfinite(variances)) or (variances <= 0).any():
        flat = [t for t, v in zip(tickers, variances) if not v > 0]
        raise DataError(f"Zero/invalid return variance for {flat}; cannot run HRP")

    corr = returns.corr()
    dist = _correlation_distance(corr)
    link = linkage(squareform(dist, checks=False), method="single")
    ordered = [tickers[i] for i in leaves_list(link)]

    weights = _recursive_bisection(cov, ordered).reindex(tickers)
    weights.name = "weight"

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise DataError("HRP produced invalid total weight; cannot renormalize")
    weights = weights / total

    logger.info(f"HRP weights computed for {len(tickers)} tickers "
                f"(min={weights.min():.3f}, max={weights.max():.3f})")
    return weights
