"""Prediction accuracy tracking across runs.

Every time the pipeline generates today's rankings, it logs each
ticker's predicted score and the price it was predicted at. The next
time you run the pipeline (e.g. the next morning), BEFORE generating
new predictions it looks up the most recent PRIOR log entry, compares
each predicted score against the return actually realized since then
(using today's prices), and prints a report.

This is entirely self-contained: it does not require a live feed. It
simply compares "what did the model say last time" against "what does
the price data show now" using the same price source you already pull
for everything else.

Metrics reported:
  - Directional accuracy: did the predicted score's SIGN match the
    realized return's sign, averaged across tickers.
  - Information Coefficient (IC): Spearman rank correlation between
    predicted score and realized return across the universe. This is
    the standard measure of whether a ranking signal has real skill
    (near 0 = no skill, regardless of whether yesterday happened to
    be profitable).
  - Top-pick vs rest-of-universe realized return: whether the stocks
    actually selected into the portfolio outperformed the ones that
    weren't picked -- the thing that matters most practically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from umushroom.logging_config import get_logger

logger = get_logger("prediction_log")

LOG_COLUMNS = ["run_date", "ticker", "predicted_score", "price_at_prediction", "is_top_pick"]


def log_predictions(
    log_path: Path,
    run_date: pd.Timestamp,
    all_scores: pd.Series,
    prices: pd.Series,
    picked_tickers: set[str],
) -> None:
    """Append today's predictions to the log. Re-running on the same
    calendar date replaces that date's prior entries (idempotent),
    rather than duplicating them.
    """
    run_date = pd.Timestamp(run_date).normalize()
    rows = [
        {
            "run_date": run_date,
            "ticker": ticker,
            "predicted_score": float(score),
            "price_at_prediction": float(prices[ticker]),
            "is_top_pick": ticker in picked_tickers,
        }
        for ticker, score in all_scores.items()
        if ticker in prices.index
    ]
    new_rows = pd.DataFrame(rows, columns=LOG_COLUMNS)

    if log_path.exists():
        existing = pd.read_csv(log_path, parse_dates=["run_date"])
        existing = existing[existing["run_date"] != run_date]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_rows

    combined.to_csv(log_path, index=False)
    logger.info(f"Logged {len(new_rows)} predictions for {run_date.date()} "
                f"to {log_path}")


@dataclass(frozen=True)
class PredictionAccuracyReport:
    evaluated_run_date: pd.Timestamp
    n_tickers: int
    directional_accuracy: float
    information_coefficient: float
    ic_p_value: float
    mean_predicted_return: float
    mean_realized_return: float
    top_pick_realized_return: float
    rest_of_universe_realized_return: float

    def summary(self) -> str:
        ic_str = (f"{self.information_coefficient:+.3f} (p={self.ic_p_value:.3f})"
                  if not np.isnan(self.information_coefficient) else "n/a (too few tickers)")
        top_str = (f"{self.top_pick_realized_return:+.2%}"
                   if not np.isnan(self.top_pick_realized_return) else "n/a")
        rest_str = (f"{self.rest_of_universe_realized_return:+.2%}"
                    if not np.isnan(self.rest_of_universe_realized_return) else "n/a")
        return (
            f"Evaluating predictions from: {self.evaluated_run_date.date()}\n"
            f"Tickers evaluated:           {self.n_tickers}\n"
            f"Directional accuracy:        {self.directional_accuracy:.1%}  "
            f"(predicted sign matched realized sign)\n"
            f"Information coefficient:     {ic_str}\n"
            f"Mean predicted return:       {self.mean_predicted_return:+.2%}\n"
            f"Mean realized return:        {self.mean_realized_return:+.2%}\n"
            f"Top-pick realized return:    {top_str}\n"
            f"Rest-of-universe realized:   {rest_str}"
        )


def evaluate_latest_predictions(
    log_path: Path,
    current_prices: pd.Series,
    current_date: pd.Timestamp,
) -> PredictionAccuracyReport | None:
    """Score the most recent PRIOR run's predictions against current
    prices. Returns None if there's no prior log entry to evaluate
    (e.g. this is the first run, or the log is empty before today).
    """
    if not log_path.exists():
        logger.info("No prediction log found yet: nothing to evaluate "
                    "(this looks like the first run).")
        return None

    log = pd.read_csv(log_path, parse_dates=["run_date"])
    current_date = pd.Timestamp(current_date).normalize()
    prior_dates = log["run_date"][log["run_date"] < current_date]
    if prior_dates.empty:
        logger.info(f"Prediction log has no entries before "
                    f"{current_date.date()}: nothing to evaluate yet.")
        return None

    last_run_date = prior_dates.max()
    prior = log[log["run_date"] == last_run_date].copy()
    prior = prior[prior["ticker"].isin(current_prices.index)]
    if prior.empty:
        logger.warning("None of the previously logged tickers have current "
                       "prices available: cannot evaluate.")
        return None

    prior["current_price"] = prior["ticker"].map(current_prices)
    prior["realized_return"] = prior["current_price"] / prior["price_at_prediction"] - 1

    directional_accuracy = float(
        (np.sign(prior["predicted_score"]) == np.sign(prior["realized_return"])).mean()
    )

    if len(prior) >= 3:
        ic, p_value = spearmanr(prior["predicted_score"], prior["realized_return"])
        ic, p_value = float(ic), float(p_value)
    else:
        ic, p_value = float("nan"), float("nan")

    top_picks = prior[prior["is_top_pick"]]
    rest = prior[~prior["is_top_pick"]]

    report = PredictionAccuracyReport(
        evaluated_run_date=last_run_date,
        n_tickers=len(prior),
        directional_accuracy=directional_accuracy,
        information_coefficient=ic,
        ic_p_value=p_value,
        mean_predicted_return=float(prior["predicted_score"].mean()),
        mean_realized_return=float(prior["realized_return"].mean()),
        top_pick_realized_return=(
            float(top_picks["realized_return"].mean()) if not top_picks.empty
            else float("nan")
        ),
        rest_of_universe_realized_return=(
            float(rest["realized_return"].mean()) if not rest.empty
            else float("nan")
        ),
    )
    logger.info(f"Evaluated predictions from {last_run_date.date()}: "
                f"directional accuracy={directional_accuracy:.1%}, IC={ic:.3f}")
    return report
