"""When to sell: the strategy's OWN exit rule, turned into a trade list.

The backtests validate exactly one exit rule, so that is the one shown:

  Every `rebalance_days` trading days (5 = weekly) the model re-ranks the
  universe. A stock is SOLD when it is no longer among the picks; the
  money goes into the new picks. Nothing else -- there is no stop-loss or
  profit target, because none was ever tested and an untested rule could
  quietly destroy the measured alpha.

Compare what you hold (the orders of the run you bought from) with the
latest run:

  SELL ALL   held, but no longer a pick              -> sell every share
  SELL       still a pick, but above its target size -> trim
  HOLD       still a pick, within the tolerance band
  BUY MORE   still a pick, below its target size     -> top up
  BUY NEW    a new pick                               -> buy

Small differences (under `band` of the position's value, default 2%) are
HOLD: trading them would cost more than it changes.

Sells are listed first: do them first, so the cash is there for the buys.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

DEFAULT_REBALANCE_DAYS = 5
# v4.21: rebalance periods are counted in trading days from a fixed Monday,
# so with rebalance_days = 5 a period is a calendar week (Mon-Fri). Before,
# the sell tab compared each day's run with the previous day's and told the
# team to turn the whole book over daily (16 sells, 13 buys on 2026-10-27);
# the strategy was only ever validated holding its picks for 5 trading days.
EPOCH = date(2000, 1, 3)
ORDER = {"SELL ALL": 0, "SELL": 1, "BUY NEW": 2, "BUY MORE": 3, "HOLD": 4}


@dataclass(frozen=True)
class Action:
    sleeve: str
    ticker: str
    name: str
    action: str
    held_shares: float
    target_shares: float
    price: float | None           # latest known local price (for value estimates)
    currency: str
    reason: str

    @property
    def trade_shares(self) -> float:
        """Shares to trade (positive buy, negative sell)."""
        if self.action == "HOLD":
            return 0.0
        return self.target_shares - self.held_shares


def next_review(as_of: str, rebalance_days: int = DEFAULT_REBALANCE_DAYS) -> date:
    """The trading day the holding period chosen on `as_of` ends."""
    d0 = pd.Timestamp(as_of[:10])
    return (d0 + pd.offsets.BDay(rebalance_days)).date()


def _as_date(d: str | date) -> date:
    return d if isinstance(d, date) else pd.Timestamp(str(d)[:10]).date()


def trade_day(as_of: str | date) -> date:
    """Orders built on the close of `as_of` are entered the next trading day."""
    # roll a weekend back to Friday first, so Saturday -> Monday (not Tuesday)
    return np.busday_offset(np.datetime64(_as_date(as_of)), 1, roll="backward").astype(date)


def period_index(d: str | date, rebalance_days: int = DEFAULT_REBALANCE_DAYS) -> int:
    """Which holding period a trading day belongs to (weeks for 5 days)."""
    if rebalance_days < 1:
        raise ValueError("rebalance_days must be >= 1")
    return int(np.busday_count(np.datetime64(EPOCH), np.datetime64(_as_date(d)))) \
        // rebalance_days


def period_start(d: str | date, rebalance_days: int = DEFAULT_REBALANCE_DAYS) -> date:
    k = period_index(d, rebalance_days)
    return np.busday_offset(np.datetime64(EPOCH), k * rebalance_days).astype(date)


def next_rebalance(as_of: str | date, rebalance_days: int = DEFAULT_REBALANCE_DAYS) -> date:
    """First trading day of the period after the one `as_of`'s orders fall in."""
    start = period_start(trade_day(as_of), rebalance_days)
    return np.busday_offset(np.datetime64(start), rebalance_days).astype(date)


@dataclass(frozen=True)
class Cadence:
    held: str | None          # the run whose orders you hold (None = nothing yet)
    rebalance_now: bool       # is the latest run the first of a new period?
    period_start: date        # first trading day of the latest run's period
    next_rebalance: date      # first trading day of the next period


def cadence(runs: Iterable[tuple[str, str]], latest: str,
            rebalance_days: int = DEFAULT_REBALANCE_DAYS) -> Cadence:
    """The weekly rule applied to a run history.

    `runs` are (stamp, as_of) pairs of successful runs. A run's orders fall
    in the period of their trade day (as_of + 1 trading day). The FIRST run
    of a period is that period's book: when the latest run is it, rebalance
    (sell what the previous period's book held that is no longer picked);
    otherwise hold the first run's book until the next period starts."""
    table = {stamp: as_of for stamp, as_of in runs}
    if latest not in table:
        raise ValueError(f"latest run {latest!r} is not in the history")
    idx = {stamp: period_index(trade_day(a), rebalance_days) for stamp, a in table.items()}
    p = idx[latest]
    same = sorted(s for s, k in idx.items() if k == p)
    start = period_start(trade_day(table[latest]), rebalance_days)
    nxt = np.busday_offset(np.datetime64(start), rebalance_days).astype(date)
    if same[0] != latest:
        return Cadence(same[0], False, start, nxt)
    earlier = [k for k in idx.values() if k < p]
    if not earlier:
        return Cadence(None, True, start, nxt)
    prev = max(earlier)
    held = min(s for s, k in idx.items() if k == prev)
    return Cadence(held, True, start, nxt)


def _num(v: object) -> float:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(f) else f


def _by_ticker(payload: dict) -> dict[str, dict]:
    return {str(o["ticker"]): o for o in payload.get("orders") or []}


def _ranks(payload: dict) -> dict[str, int]:
    return {str(t["ticker"]): int(t["rank"]) for t in payload.get("top") or []
            if t.get("rank") is not None}


def _price(o: dict) -> float | None:
    p = o.get("local_price", o.get("price"))
    return None if p is None else _num(p) or None


def plan(held: dict, latest: dict, band: float = 0.02) -> list[Action]:
    """Turn 'what I bought' + 'what the model wants now' into trades."""
    have, want = _by_ticker(held), _by_ticker(latest)
    ranks = _ranks(latest)
    sel = {s["name"]: s.get("selection", "model") for s in latest.get("sleeves") or []}
    out: list[Action] = []

    for t in sorted(set(have) | set(want)):
        h, w = have.get(t), want.get(t)
        o = w or h or {}
        sleeve = str(o.get("sleeve") or "")
        held_sh, tgt_sh = _num(h and h.get("shares")), _num(w and w.get("shares"))
        price = _price(w) if w else (_price(h) if h else None)
        name, ccy = str(o.get("name") or ""), str(o.get("listing_currency") or "")

        def act(kind: str, why: str, t: str = t, sleeve: str = sleeve, name: str = name,
                held_sh: float = held_sh, tgt_sh: float = tgt_sh,
                price: float | None = price, ccy: str = ccy) -> Action:
            return Action(sleeve, t, name, kind, held_sh, tgt_sh, price, ccy, why)
        if h and not w:
            if sel.get(sleeve, "model") == "model":
                r = ranks.get(t)
                why = (f"Dropped out of the picks: the model now ranks it #{r}." if r
                       else "Dropped out of the picks: no longer ranked near the top.")
            else:
                why = ("No longer in this sleeve today (removed from the list, or no "
                       "clean price) -- the sleeve holds only what it can price.")
            out.append(act("SELL ALL", why))
        elif w and not h:
            r = ranks.get(t)
            why = (f"New pick (ranked #{r})." if r and sel.get(sleeve) == "model"
                   else "New pick." if sel.get(sleeve, "model") == "model"
                   else "Part of this sleeve (holds every instrument, equal weight).")
            out.append(act("BUY NEW", why))
        else:
            diff = tgt_sh - held_sh
            size = max(tgt_sh, held_sh) * (price or 0.0)
            small = abs(diff) < 1e-9 or (size > 0 and abs(diff) * (price or 0.0)
                                          <= band * size)
            if small:
                out.append(act("HOLD", "Still a pick; size is on target."))
            elif diff < 0:
                out.append(act("SELL", (
                    "Still a pick, but above its target size (it rose, or the money is "
                    "spread over more names now): trim back.")))
            else:
                out.append(act("BUY MORE",
                               "Still a pick, but below its target size: top up."))
    return sorted(out, key=lambda a: (ORDER[a.action], a.sleeve, a.ticker))
