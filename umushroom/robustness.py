"""Robustness statistics for backtest results.

A single point-estimate Sharpe ratio from one backtest run tells you
almost nothing about how reliable that number is -- with only a
handful of rebalance periods (which is typical for a 6-week contest
or even a 2-year daily backtest sampled every few days), the Sharpe
ratio itself has wide sampling variance. Two strategies with Sharpe
1.2 and Sharpe 0.8 might be statistically indistinguishable once you
account for that.

This module implements a circular block bootstrap (resampling
contiguous BLOCKS of returns, not individual returns) to build a
confidence interval around the Sharpe ratio. Block (rather than iid)
resampling matters here because daily/period returns are
autocorrelated (momentum, mean-reversion, volatility clustering) --
resampling individual returns would destroy that structure and give
an artificially narrow, overconfident interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

logger_name = "robustness"


@dataclass(frozen=True)
class SharpeConfidenceInterval:
    point_estimate: float
    mean_bootstrap: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    block_size: int
    n_observations: int

    def summary(self) -> str:
        return (
            f"Sharpe ratio (point estimate):  {self.point_estimate:.2f}\n"
            f"Bootstrap mean:                 {self.mean_bootstrap:.2f}\n"
            f"{self.confidence_level:.0%} confidence interval:        "
            f"[{self.ci_lower:.2f}, {self.ci_upper:.2f}]\n"
            f"  (from {self.n_bootstrap} block-bootstrap resamples, "
            f"block size {self.block_size}, n={self.n_observations} periods)\n"
            + (
                "  NOTE: interval includes 0 -- cannot rule out that the "
                "strategy has no real edge over this sample.\n"
                if self.ci_lower <= 0 <= self.ci_upper else ""
            )
        )


def bootstrap_sharpe(
    period_returns: pd.Series,
    periods_per_year: float,
    n_bootstrap: int = 5000,
    block_size: int = 5,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> SharpeConfidenceInterval:
    """Circular block bootstrap confidence interval for an annualized
    Sharpe ratio, computed from a series of per-rebalance-period
    returns (NOT daily returns necessarily -- whatever periodicity
    period_returns is in, pass the matching periods_per_year so the
    annualization is consistent).

    block_size should be chosen large enough to capture the
    return series' autocorrelation structure (a handful of periods is
    typically enough for short-horizon equity momentum/reversal
    signals) but small relative to len(period_returns), or the
    circular wrap-around will dominate the resampling.
    """
    if len(period_returns) < 10:
        raise ValueError(
            f"Need at least 10 return observations for a meaningful "
            f"bootstrap, got {len(period_returns)}"
        )
    if block_size < 1 or block_size > len(period_returns):
        raise ValueError(
            f"block_size ({block_size}) must be between 1 and "
            f"len(period_returns) ({len(period_returns)})"
        )
    if not (0 < confidence_level < 1):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    returns = period_returns.dropna().values
    n = len(returns)
    rng = np.random.default_rng(random_seed)

    std = returns.std()
    point_estimate = (
        float((returns.mean() / std) * np.sqrt(periods_per_year)) if std > 0 else 0.0
    )

    sharpe_samples = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        start_points = rng.integers(0, n, size=(n // block_size) + 1)
        indices = []
        for start in start_points:
            indices.extend((start + np.arange(block_size)) % n)
        sample = returns[indices[:n]]
        s = sample.std()
        sharpe_samples[b] = (sample.mean() / s) * np.sqrt(periods_per_year) if s > 0 else 0.0

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(sharpe_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(sharpe_samples, 100 * (1 - alpha / 2)))

    return SharpeConfidenceInterval(
        point_estimate=point_estimate,
        mean_bootstrap=float(sharpe_samples.mean()),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        n_observations=n,
    )
