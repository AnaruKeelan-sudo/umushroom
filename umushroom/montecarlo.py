"""Monte Carlo simulation of the contest, from OUT-OF-SAMPLE history.

Method: circular BLOCK bootstrap of the weekly (rebalance-period) returns
the purged walk-forward actually produced. Whole blocks of consecutive
weeks are resampled, keeping week-to-week momentum and volatility
clustering that iid resampling would destroy. The strategy and the
benchmark are always drawn from the SAME weeks, so "did we beat holding
everything?" compares like with like (a bad market week hits both).

What it answers: "if the next N weeks look like a random stretch of the
last ~8 years, what range of outcomes should we expect?" -- the spread of
final portfolio values, the chance of a loss, and the chance of beating
the benchmark.

What it does NOT do: foresee a crash or boom with no precedent in the
sample. It is a range, not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_HISTORY = 20
MIN_PATHS = 100


@dataclass(frozen=True)
class MonteCarloResult:
    horizon_weeks: int
    start_value: float
    block_size: int
    n_history: int
    strategy_paths: np.ndarray      # n_paths x (horizon + 1), column 0 = start
    benchmark_paths: np.ndarray

    @property
    def n_paths(self) -> int:
        return int(self.strategy_paths.shape[0])

    @property
    def final_strategy(self) -> np.ndarray:
        return self.strategy_paths[:, -1]

    @property
    def final_benchmark(self) -> np.ndarray:
        return self.benchmark_paths[:, -1]

    def percentile_bands(self) -> pd.DataFrame:
        q = {"p5": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95}
        out = pd.DataFrame({k: np.percentile(self.strategy_paths, v, axis=0)
                            for k, v in q.items()})
        out["benchmark_p50"] = np.median(self.benchmark_paths, axis=0)
        out.index.name = "week"
        return out

    def stats(self) -> dict[str, float]:
        f, b, s0 = self.final_strategy, self.final_benchmark, self.start_value
        return {
            "median_final": float(np.median(f)),
            "p5_final": float(np.percentile(f, 5)),
            "p95_final": float(np.percentile(f, 95)),
            "median_return": float(np.median(f) / s0 - 1),
            "prob_loss": float((f < s0).mean()),
            "prob_beat_benchmark": float((f > b).mean()),
            "var_95": float(s0 - np.percentile(f, 5)),   # 5% worst-case loss
            "var_99": float(s0 - np.percentile(f, 1)),
            # expected shortfall: AVERAGE loss in the worst 5% / 1% of runs
            "es_95": float(s0 - f[f <= np.percentile(f, 5)].mean()),
            "es_99": float(s0 - f[f <= np.percentile(f, 1)].mean()),
            "benchmark_median_final": float(np.median(b)),
        }


def block_bootstrap_indices(n: int, horizon: int, n_paths: int, block_size: int,
                            rng: np.random.Generator) -> np.ndarray:
    """n_paths x horizon indices into a history of length n, built from
    blocks of `block_size` CONSECUTIVE weeks (wrapping around the end)."""
    n_blocks = int(np.ceil(horizon / block_size))
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_size)[None, None, :]) % n
    return idx.reshape(n_paths, n_blocks * block_size)[:, :horizon]


def combine_sleeves(parts: list[tuple[float, pd.Series]]) -> pd.Series:
    """Weighted sum of sleeve period returns on the dates ALL sleeves share
    (each simulated week is a real week in which every sleeve existed)."""
    if not parts:
        raise ValueError("no sleeves to combine")
    frame = pd.concat([s.rename(i) for i, (_, s) in enumerate(parts)], axis=1,
                      join="inner").dropna()
    weights = np.array([w for w, _ in parts], dtype=float)
    return pd.Series(frame.to_numpy() @ weights, index=frame.index,
                     name="portfolio_return")


def simulate(strategy: pd.Series, benchmark: pd.Series, horizon_weeks: int,
             start_value: float, n_paths: int = 5000, seed: int = 42,
             block_size: int = 4) -> MonteCarloResult:
    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be >= 1")
    if n_paths < MIN_PATHS:
        raise ValueError(f"n_paths must be >= {MIN_PATHS}")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    joined = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1,
                       join="inner").dropna()
    n = len(joined)
    if n < MIN_HISTORY:
        raise ValueError(f"only {n} weeks of history; need at least {MIN_HISTORY}")
    rng = np.random.default_rng(seed)
    idx = block_bootstrap_indices(n, horizon_weeks, n_paths, block_size, rng)
    s, b = joined["s"].to_numpy()[idx], joined["b"].to_numpy()[idx]

    def paths(r: np.ndarray) -> np.ndarray:
        out = np.empty((n_paths, horizon_weeks + 1))
        out[:, 0] = start_value
        out[:, 1:] = start_value * np.cumprod(1 + r, axis=1)
        return out
    return MonteCarloResult(horizon_weeks, float(start_value), block_size, n,
                            paths(s), paths(b))


def bootstrap_annual_alpha(active_returns: pd.Series, periods_per_year: float,
                           n_sims: int = 5000, block_size: int = 4,
                           seed: int = 42) -> np.ndarray:
    """Distribution of the annualised mean active return (strategy minus
    benchmark) under block resampling of the out-of-sample history. Its
    spread is the honest uncertainty around the headline alpha."""
    a = active_returns.dropna().to_numpy()
    n = len(a)
    if n < MIN_HISTORY:
        raise ValueError(f"only {n} periods; need at least {MIN_HISTORY}")
    rng = np.random.default_rng(seed)
    idx = block_bootstrap_indices(n, n, n_sims, block_size, rng)
    return a[idx].mean(axis=1) * periods_per_year
