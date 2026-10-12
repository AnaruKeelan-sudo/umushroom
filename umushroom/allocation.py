"""Portfolio allocation generation.

Converts today's model rankings into a concrete, integer-share order
list sized to the contest's starting balance. Kept separate from the
model/backtest modules because this is the one place real money
(virtual or not) gets committed, so it has the tightest input/output
validation in the package: every invariant (shares >= 0, total spend
<= balance, no duplicate tickers) is checked and will raise rather
than silently producing an invalid order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd

from umushroom.exceptions import DataError
from umushroom.features import build_features_at_index
from umushroom.logging_config import get_logger
from umushroom.model import RankingModel
from umushroom.portfolio_construction import equal_weight, hrp_weight
from umushroom.selection import capped_top, top_sector_share

logger = get_logger("allocation")


def cap_weights(weights: pd.Series, cap: float) -> pd.Series:
    """Cap every weight at `cap` (e.g. 2x equal weight), redistributing the
    excess pro rata to uncapped names. v4.4's real run had HRP put 59% of
    the stock sleeve in one name; nothing should ever dominate like that.
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
        w[under] += excess * w[under] / w[under].sum()
    return w / w.sum()

WeightingScheme = Literal["equal", "hrp"]


@dataclass(frozen=True)
class Allocation:
    orders: pd.DataFrame          # columns: predicted_score, price, shares, allocated_value
    total_deployed: float
    cash_remaining: float
    balance: float
    all_scores: pd.Series          # predicted score for EVERY ticker with clean
                                    # features today, not just the top_n picks.
                                    # Needed for prediction logging / next-run
                                    # accuracy evaluation and IC computation --
                                    # scoring only the picks would bias those
                                    # metrics toward whatever the model already
                                    # favored, rather than measuring it fairly
                                    # across the whole universe.

    def __post_init__(self) -> None:
        if self.total_deployed > self.balance + 1e-6:
            raise DataError(
                f"Allocation overspends balance: deployed={self.total_deployed:.2f} "
                f"> balance={self.balance:.2f}"
            )
        if (self.orders["shares"] < 0).any():
            raise DataError("Allocation contains negative share counts")
        if self.orders.index.duplicated().any():
            raise DataError("Allocation contains duplicate tickers")

    def to_csv(self, path: str) -> None:
        self.orders.to_csv(path)
        logger.info(f"Allocation written to {path}")


def generate_allocation(
    data: pd.DataFrame,
    model: RankingModel,
    top_n: int,
    balance: float,
    weighting: WeightingScheme = "equal",
    groups: dict[str, str] | None = None,
    local_prices: pd.DataFrame | None = None,
    currencies: dict[str, str] | None = None,
    fractional: bool = False,
    min_local_price: float = 0.0,
    dollar_volume: pd.DataFrame | None = None,
    hold_all: bool = False,
    max_weight_multiple: float = 2.0,
    exclude: frozenset[str] = frozenset(),
    sectors: dict[str, str] | None = None,
    max_per_sector: int | None = None,
    feature_set: str = "equity",
) -> Allocation:
    """Rank today's universe and size integer-share orders.

    `data` must be in the ACCOUNT's base currency (data.get_market_data
    guarantees this), so `balance * weight // price` buys the right
    number of shares regardless of where the stock is listed.
    `local_prices`/`currencies` are optional and only add the
    listing-currency price to the output for cross-checking against
    what UMushroom displays.
    """
    if balance <= 0:
        raise ValueError(f"balance must be > 0, got {balance}")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if weighting not in ("equal", "hrp"):
        raise ValueError(f"weighting must be 'equal' or 'hrp', got {weighting!r}")

    as_of_idx = len(data) - 1
    feats_today = build_features_at_index(
        data, as_of_idx, groups=groups, dollar_volume=dollar_volume,
        feature_set=feature_set).dropna()
    preds = model.predict(feats_today)

    stale = sorted(set(preds.index) & set(exclude))
    if stale:
        logger.warning(f"Excluding {stale}: no fresh price for the latest day "
                       f"(would trade on an old price).")
        preds = preds.drop(stale)
        feats_today = feats_today.drop(stale)

    # v4.5: drop untradeable candidates BEFORE choosing the top N, so the
    # next-ranked instrument takes the slot (v4.4 skipped them AFTER
    # weighting, leaving that money in cash).
    excluded_any = bool(stale)
    if local_prices is not None and min_local_price > 0:
        last_local = local_prices.iloc[-1].reindex(preds.index)
        too_low = last_local[last_local < min_local_price].index.tolist()
        if too_low:
            excluded_any = True
            logger.warning(
                f"Excluding {too_low}: local price below {min_local_price} -- "
                f"probably a pence/pound or USD-line error. Check on UMushroom "
                f"and fix currency_overrides.")
            preds = preds.drop(too_low)
            feats_today = feats_today.drop(too_low)
    if excluded_any and 0 < len(feats_today) < top_n and not hold_all:
        logger.warning(f"Only {len(feats_today)} tradeable candidates left after "
                       f"exclusions; holding all of them instead of {top_n}.")
        top_n = len(feats_today)
    if hold_all:
        top_n = len(feats_today)       # "all" = every instrument usable today
        if top_n == 0:
            raise DataError("No instruments have clean features today")
    if len(feats_today) < top_n:
        raise DataError(
            f"Only {len(feats_today)} tickers have clean features today; "
            f"cannot select top_n={top_n}."
        )

    cap = None if hold_all else max_per_sector     # hold-all sleeves hold everything
    ranked_all = preds.sort_values(ascending=False).index.tolist()
    picked_tickers = capped_top(ranked_all, top_n, sectors, cap)
    if cap and sectors:
        logger.info(f"Sector cap: at most {cap} per sector "
                    f"(largest sector now {top_sector_share(picked_tickers, sectors):.0%} "
                    f"of picks)")

    def size(selected: list[str]) -> tuple[pd.Series, dict[str, dict[str, object]]]:
        picked_tickers = selected
        if weighting == "hrp":
            try:
                weights = hrp_weight(data, picked_tickers, as_of_idx)
            except (DataError, ValueError) as e:
                logger.warning(f"HRP weighting failed ({e}); falling back to "
                                f"equal weight for this run.")
                weights = equal_weight(picked_tickers)
        else:
            weights = equal_weight(picked_tickers)
        weights = cap_weights(weights, max_weight_multiple / len(picked_tickers))

        rows: dict[str, dict[str, object]] = {}
        for ticker in picked_tickers:
            price = float(data[ticker].iloc[-1])
            if price <= 0:
                logger.warning(f"Skipping {ticker}: non-positive price {price}")
                continue
            if local_prices is not None and ticker in local_prices.columns:
                local_px = float(local_prices[ticker].iloc[-1])
                if local_px < min_local_price:
                    # v4.3.1: a price this low for a stock/ETF almost always means
                    # a pence/pound mix-up (IGLT.L showed 0.0952 instead of 9.48 on
                    # the first real run -- a 100x oversized order). Never trade it.
                    logger.warning(
                        f"SKIPPING {ticker}: price {local_px:.4f} is below "
                        f"{min_local_price} -- probably a pence/pound error. Check "
                        f"its price on UMushroom and fix currency_overrides.")
                    continue
            dollars = balance * float(weights[ticker])
            if fractional:
                # floor (never round up) to 6 dp so the order can't overspend
                shares: float = math.floor(dollars / price * 1e6) / 1e6
            else:
                shares = int(dollars // price)
            row: dict[str, object] = {
                "predicted_score": round(float(preds[ticker]), 5),
                "target_weight": round(float(weights[ticker]), 4),
                "price": round(price, 4),
                "shares": shares,
                "allocated_value": math.floor(shares * price * 100) / 100,
            }
            if currencies is not None:
                row["listing_currency"] = currencies.get(ticker, "")
            if local_prices is not None and ticker in local_prices.columns:
                row["local_price"] = round(float(local_prices[ticker].iloc[-1]), 4)
            rows[ticker] = row
        return weights, rows

    # v4.11: an instrument whose single unit costs more than its share of the
    # budget would come out as a 0-share order (the money silently unused).
    # Drop it, re-spread the budget, and -- for model picks -- bring in the
    # next-ranked candidate so the sleeve still holds top_n names.
    ranked = preds.sort_values(ascending=False).index.tolist()
    selected = list(picked_tickers)
    unaffordable: list[str] = []
    while True:
        _weights, rows = size(selected)
        zero = [t for t, r in rows.items() if float(cast(float, r["shares"])) <= 0]
        if not zero or len(zero) == len(rows):
            break
        unaffordable += zero
        selected = [t for t in selected if t not in zero]
        if not hold_all:
            # refill by rank, still respecting the sector cap where possible
            pool = [t for t in ranked if t not in unaffordable]
            selected = capped_top([t for t in pool if t in selected]
                                  + [t for t in pool if t not in selected],
                                  top_n, sectors, cap)
    if unaffordable:
        logger.warning(f"Not buying {unaffordable}: one unit costs more than its share "
                       f"of the budget; the money was spread over the other holdings.")
    rows = {t: r for t, r in rows.items() if float(cast(float, r["shares"])) > 0}
    picked_tickers = list(rows)

    if not rows:
        raise DataError("No valid picks survived price sanity checks")

    orders = pd.DataFrame(rows).T
    orders.index.name = "ticker"
    for col in ("predicted_score", "target_weight", "price", "allocated_value"):
        orders[col] = orders[col].astype(float)
    orders["shares"] = orders["shares"].astype(float if fractional else int)
    total_deployed = float(orders["allocated_value"].sum())

    allocation = Allocation(
        orders=orders,
        total_deployed=total_deployed,
        cash_remaining=balance - total_deployed,
        balance=balance,
        all_scores=preds,
    )
    logger.info(f"Generated allocation: {len(orders)} positions, "
                f"{total_deployed:,.2f}/{balance:,.2f} deployed")
    return allocation
