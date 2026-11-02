"""Picking the top-N with an optional per-sector cap (v4.18) and an
optional near-duplicate guard (v4.20).

The model's ranking is unchanged. The caps only change WHICH of the
top-ranked names are bought: walk down the ranking, take a name unless its
sector already has `max_per_sector` picks, and skip to the next-ranked name
instead. If there aren't enough sectors to fill top_n, the remaining slots
are filled by rank ignoring the cap (never fewer holdings than asked for).

Tickers with no known sector are each treated as their own sector (we
can't tell that two unknowns are both banks, so we don't pretend to).

v4.20 near-duplicate guard (`max_pick_correlation`). On 2026-10-19 the ETF
model's eight picks included two MSCI Korea trackers (1-year return
correlation 0.99) and three USD high-yield bond funds (0.6-0.92): one bet
counted three times. An asset-class cap cannot see that -- "equity" holds
1,200 ETFs on the platform. The guard skips any candidate whose trailing
daily-return correlation with a name already picked exceeds the threshold.
It is CAUSAL (only prices up to the signal date, see correlation_filter) and
is applied identically in the walk-forward backtest and in live orders. If
the universe really cannot fill top_n without duplicates, the skipped names
are used last, by rank -- a smaller book is never produced silently.

Decision rule for switching either cap on (pre-declared, see decide_cap):
adopt it only if it keeps at least 75% of the uncapped alpha AND does not
increase the weekly 95% VaR.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

ALPHA_RETENTION = 0.75
# v4.21: two weekly VaRs within this are the same number (0.05 pp). The ETF
# near-duplicate guard was rejected on 2026-10-19 for "raising" VaR from
# 2.86% to 2.88% while its alpha was higher -- a tie read as a loss.
VAR_TOLERANCE = 0.0005
# v4.20: trailing window and minimum overlap for the near-duplicate guard
CORR_LOOKBACK_DAYS = 252
MIN_CORR_OBS = 60

SimilarFn = Callable[[str, list[str]], bool]


def _sector(t: str, sectors: Mapping[str, str]) -> str:
    s = sectors.get(t)
    return s if s else f"?{t}"


def capped_top(ranked: Iterable[str], top_n: int,
               sectors: Mapping[str, str] | None = None,
               max_per_sector: int | None = None,
               too_similar: SimilarFn | None = None) -> list[str]:
    """First `top_n` of `ranked` (best first), at most `max_per_sector` per
    sector and never a name `too_similar(candidate, picks_so_far)` to one
    already taken; topped up by rank if the caps can't fill every slot
    (sector-capped names first, near-duplicates last)."""
    order = list(ranked)
    use_cap = bool(sectors) and bool(max_per_sector)
    if not use_cap and too_similar is None:
        return order[:top_n]
    picks: list[str] = []
    skipped_cap: list[str] = []
    skipped_dup: list[str] = []
    counts: dict[str, int] = {}
    for t in order:
        if too_similar is not None and too_similar(t, picks):
            skipped_dup.append(t)
            continue
        s = _sector(t, sectors) if use_cap and sectors is not None else t
        if not use_cap or counts.get(s, 0) < (max_per_sector or 0):
            picks.append(t)
            counts[s] = counts.get(s, 0) + 1
            if len(picks) == top_n:
                return picks
        else:
            skipped_cap.append(t)
    for t in skipped_cap + skipped_dup:     # not enough names: fill by rank
        if len(picks) == top_n:
            break
        picks.append(t)
    return picks


def correlation_filter(prices: pd.DataFrame, as_of_idx: int, max_correlation: float,
                       lookback: int = CORR_LOOKBACK_DAYS,
                       min_obs: int = MIN_CORR_OBS) -> SimilarFn:
    """The near-duplicate predicate for `capped_top`, built from prices up
    to and including `as_of_idx` ONLY (no lookahead, as features.py).

    `too_similar(candidate, picks)` is True when the candidate's daily
    return correlation over the trailing `lookback` days with ANY current
    pick exceeds `max_correlation`. A pair with fewer than `min_obs`
    overlapping returns (a recent listing) cannot be judged and is allowed.
    """
    if not 0 < max_correlation <= 1:
        raise ValueError(f"max_correlation must be in (0, 1], got {max_correlation}")
    if not 0 <= as_of_idx < len(prices):
        raise ValueError(f"as_of_idx {as_of_idx} outside the price history")
    window = prices.iloc[max(0, as_of_idx + 1 - lookback): as_of_idx + 1]
    rets = window.pct_change(fill_method=None).iloc[1:]
    cache: dict[tuple[str, str], float] = {}

    def corr(a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        if key not in cache:
            pair = rets[[a, b]].dropna()
            # a constant window (suspended line) has no correlation: NaN = allowed
            with np.errstate(invalid="ignore", divide="ignore"):
                cache[key] = (float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                              if len(pair) >= min_obs else float("nan"))
        return cache[key]

    def too_similar(candidate: str, picks: list[str]) -> bool:
        if candidate not in rets.columns:
            return False
        for p in picks:
            if p in rets.columns and p != candidate:
                c = corr(candidate, p)
                if np.isfinite(c) and c > max_correlation:
                    return True
        return False
    return too_similar


def top_sector_share(picks: Iterable[str], sectors: Mapping[str, str]) -> float:
    """Share of the picks in the single most-held sector."""
    ps = list(picks)
    if not ps:
        return 0.0
    counts: dict[str, int] = {}
    for t in ps:
        s = _sector(t, sectors)
        counts[s] = counts.get(s, 0) + 1
    return max(counts.values()) / len(ps)


@dataclass(frozen=True)
class CapVerdict:
    adopt: bool
    reason: str


def decide_cap(alpha_uncapped: float, alpha_capped: float,
               var_uncapped: float, var_capped: float,
               retention: float = ALPHA_RETENTION) -> CapVerdict:
    """The pre-declared rule. VaRs are positive loss fractions; a difference
    within VAR_TOLERANCE counts as unchanged."""
    if var_capped > var_uncapped + VAR_TOLERANCE:
        return CapVerdict(False, f"the cap made weekly risk WORSE (VaR {var_capped:.2%} "
                                 f"vs {var_uncapped:.2%})")
    if alpha_capped >= alpha_uncapped:
        return CapVerdict(True, "the cap kept all the alpha and lowered (or kept) risk")
    if alpha_uncapped <= 0:
        return CapVerdict(False, "the cap lowered alpha (and uncapped alpha wasn't "
                                 "positive to begin with)")
    kept = alpha_capped / alpha_uncapped
    if kept >= retention:
        return CapVerdict(True, f"the cap kept {kept:.0%} of the alpha (rule: at least "
                                f"{retention:.0%}) without adding risk")
    return CapVerdict(False, f"the cap kept only {kept:.0%} of the alpha (rule: at "
                             f"least {retention:.0%})")
