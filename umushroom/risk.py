"""Value at Risk (VaR) and Expected Shortfall (ES), and a check that the
VaR would actually have worked.

  VaR 95%  "in 19 weeks out of 20, we lose less than this"
  ES 95%   "in the worst 1 week in 20, the AVERAGE loss is this"
           (ES looks inside the tail; VaR only marks where it starts)

Both are reported as positive loss fractions (0.04 = a 4% loss).

Two sources:
  * historical -- the purged walk-forward's real out-of-sample weekly
    returns (no model of the distribution: fat tails are kept as they were)
  * Monte Carlo -- the simulated contest horizon (montecarlo.py)

VaR backtest (Kupiec "proportion of failures" test): walk forward through
the history, each week compute VaR from ONLY the previous `window` weeks,
then count how often the next week lost more. A correct 95% VaR is broken
about 5% of the time; far more breaks means it understates risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

MIN_OBS = 20


def var_es(returns: np.ndarray | pd.Series, level: float = 0.95) -> tuple[float, float]:
    """Historical VaR and ES at `level` as positive loss fractions."""
    if not 0.5 < level < 1:
        raise ValueError("level must be between 0.5 and 1, e.g. 0.95")
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < MIN_OBS:
        raise ValueError(f"only {len(r)} returns; need at least {MIN_OBS}")
    q = float(np.quantile(r, 1 - level))
    tail = r[r <= q]
    return -q, float(-tail.mean())


def kupiec(n: int, breaches: int, p: float) -> tuple[float, float]:
    """Kupiec POF likelihood ratio and its p-value (chi-squared, 1 dof).
    Small p-value = the breach rate is inconsistent with the VaR level."""
    if n <= 0:
        raise ValueError("n must be positive")
    x = breaches
    phat = x / n
    log_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_alt = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = float(max(0.0, -2 * (log_null - log_alt)))
    return lr, float(chi2.sf(lr, 1))


@dataclass(frozen=True)
class VarBacktest:
    level: float
    window: int
    returns: pd.Series            # the weeks that were tested
    var: pd.Series                # VaR known BEFORE each of those weeks
    lr: float
    p_value: float

    @property
    def n(self) -> int:
        return len(self.returns)

    @property
    def breaches(self) -> int:
        return int((self.returns < -self.var).sum())

    @property
    def breach_rate(self) -> float:
        return self.breaches / self.n if self.n else float("nan")

    @property
    def expected_rate(self) -> float:
        return 1 - self.level

    @property
    def status(self) -> str:
        if self.p_value >= 0.05:
            return "PASS"
        # too FEW breaches = cautious (safe side); too many = understates risk
        return "WARN" if self.breach_rate < self.expected_rate else "FAIL"


def backtest_var(returns: pd.Series, level: float = 0.95, window: int = 52) -> VarBacktest:
    r = returns.dropna()
    if len(r) < window + MIN_OBS:
        raise ValueError(f"need at least {window + MIN_OBS} weeks for a VaR backtest; "
                         f"have {len(r)}")
    vals = r.to_numpy()
    var = np.array([-np.quantile(vals[t - window:t], 1 - level)
                    for t in range(window, len(vals))])
    tested = r.iloc[window:]
    var_s = pd.Series(var, index=tested.index)
    x = int((tested.to_numpy() < -var).sum())
    lr, p = kupiec(len(tested), x, 1 - level)
    return VarBacktest(level, window, tested, var_s, lr, p)
