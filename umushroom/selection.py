"""Picking the top-N with an optional per-sector cap (v4.18).

The model's ranking is unchanged. The cap only changes WHICH of the
top-ranked names are bought: walk down the ranking, take a name unless its
sector already has `max_per_sector` picks, and skip to the next-ranked name
instead. If there aren't enough sectors to fill top_n, the remaining slots
are filled by rank ignoring the cap (never fewer holdings than asked for).

Tickers with no known sector are each treated as their own sector (we
can't tell that two unknowns are both banks, so we don't pretend to).

Decision rule for switching it on (pre-declared, see decide_cap):
adopt the cap only if it keeps at least 75% of the uncapped alpha AND does
not increase the weekly 95% VaR.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

ALPHA_RETENTION = 0.75


def _sector(t: str, sectors: Mapping[str, str]) -> str:
    s = sectors.get(t)
    return s if s else f"?{t}"


def capped_top(ranked: Iterable[str], top_n: int,
               sectors: Mapping[str, str] | None = None,
               max_per_sector: int | None = None) -> list[str]:
    """First `top_n` of `ranked` (best first), at most `max_per_sector` per
    sector; topped up by rank if the cap can't fill every slot."""
    order = list(ranked)
    if not sectors or not max_per_sector:
        return order[:top_n]
    picks: list[str] = []
    skipped: list[str] = []
    counts: dict[str, int] = {}
    for t in order:
        s = _sector(t, sectors)
        if counts.get(s, 0) < max_per_sector:
            picks.append(t)
            counts[s] = counts.get(s, 0) + 1
            if len(picks) == top_n:
                return picks
        else:
            skipped.append(t)
    for t in skipped:                       # not enough sectors: fill by rank
        if len(picks) == top_n:
            break
        picks.append(t)
    return picks


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
    """The pre-declared rule. VaRs are positive loss fractions."""
    if var_capped > var_uncapped + 1e-12:
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
