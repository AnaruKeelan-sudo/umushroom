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


# --------------------------------------------------------------------------
# v4.25: the live record -- is the model right, at its own horizon?
# --------------------------------------------------------------------------
# evaluate_latest_predictions above scores the previous run against today's
# price: one day of a 5-day forecast, mostly noise. Below, every logged
# cross-section is scored once its full horizon has passed, on the same
# base-currency closes the model is trained on, and the live record is set
# against the walk-forward's own week-to-week spread: "how often did the
# backtest have a stretch this bad?" is the honest yardstick for a few weeks
# of live data, which on their own can neither confirm nor refute 8 years.
# a rank correlation over fewer names is mostly noise (and the pre-4.25 ETF log
# held only the ~12 core ETFs the sleeve bought, not the universe it ranked)
MIN_LIVE_NAMES = 30


@dataclass(frozen=True)
class LiveScore:
    """One logged cross-section, scored once `horizon` closes have passed."""
    run_date: str           # the close the scores were built on
    scored_to: str          # the close `horizon` trading days later
    n: int                  # instruments scored
    ic: float               # Spearman rank correlation of scores and returns
    picks: float | None     # mean realised return of the picks (None = none)
    universe: float         # mean realised return of every scored instrument
    independent: bool = False   # True = its window overlaps no earlier counted one

    @property
    def active(self) -> float | None:
        return None if self.picks is None else self.picks - self.universe


def score_logged(log_path: Path, prices: pd.DataFrame, horizon: int,
                 min_names: int | None = None) -> tuple[list[LiveScore], list[str]]:
    """(scored cross-sections oldest first, run dates still waiting for their
    horizon). A cross-section's window starts at the last close on or before
    its run date; the windows of the `independent` rows never overlap."""
    min_names = MIN_LIVE_NAMES if min_names is None else min_names
    if not log_path.exists() or horizon < 1 or prices.empty:
        return [], []
    log = pd.read_csv(log_path, parse_dates=["run_date"])
    idx = prices.index
    rows: list[LiveScore] = []
    pending: list[str] = []
    next_free = -1                      # first close an independent window may use
    for d, g in sorted(log.groupby("run_date"), key=lambda kv: kv[0]):
        i = int(idx.searchsorted(pd.Timestamp(d), side="right")) - 1
        if i < 0:
            continue
        j = i + horizon
        if j >= len(idx):
            pending.append(pd.Timestamp(d).strftime("%Y-%m-%d"))
            continue
        g = g.drop_duplicates("ticker").set_index("ticker")
        g = g[g.index.isin(prices.columns)]
        realised = (prices.iloc[j] / prices.iloc[i] - 1).reindex(g.index)
        ok = realised.notna() & np.isfinite(realised) & g["predicted_score"].notna()
        g, realised = g[ok], realised[ok]
        if len(g) < min_names or g["predicted_score"].nunique() < 2:
            continue
        ic = float(spearmanr(g["predicted_score"], realised)[0])
        top = realised[g["is_top_pick"].astype(str).str.lower() == "true"]
        indep = i >= next_free
        if indep:
            next_free = j
        rows.append(LiveScore(pd.Timestamp(d).strftime("%Y-%m-%d"),
                              idx[j].strftime("%Y-%m-%d"), len(g), ic,
                              float(top.mean()) if len(top) else None,
                              float(realised.mean()), indep))
    return rows, pending


@dataclass(frozen=True)
class LiveVerdict:
    """The live record against the walk-forward it is supposed to resemble."""
    n: int                        # independent live periods
    mean_ic: float
    mean_active: float | None
    bt_mean_ic: float             # walk-forward mean IC per period
    bt_sd_ic: float
    share_as_bad: float | None    # share of n-period stretches of the walk-forward
                                  # whose mean IC was this low or lower
    streak: int                   # latest periods in a row with negative active
    after_streak_n: int = 0       # walk-forward periods that followed such a streak
    after_streak_next: float | None = None   # their mean active return
    all_next: float | None = None            # mean active return of every period


def live_verdict(rows: list[LiveScore], period_ic: pd.Series,
                 active: pd.Series) -> LiveVerdict | None:
    """Compare the independent live periods with the walk-forward's per-period
    IC and active returns (both out of sample). None without both."""
    live = [r for r in rows if r.independent]
    ics = period_ic.dropna()
    if not live or len(ics) < 10:
        return None
    n = len(live)
    mean_ic = float(np.mean([r.ic for r in live]))
    acts = [r.active for r in live if r.active is not None]
    windows = ics.rolling(n).mean().dropna()
    share = float((windows <= mean_ic).mean()) if len(windows) else None
    streak = 0
    for r in reversed(live):
        if r.active is None or r.active >= 0:
            break
        streak += 1
    a = active.dropna()
    after_n, after_next = 0, None
    if streak and len(a) > streak:
        lagging = (a < 0).astype(int).rolling(streak).sum().shift(1) == streak
        nxt = a[lagging]
        after_n = len(nxt)
        after_next = float(nxt.mean()) if after_n else None
    return LiveVerdict(n, mean_ic, float(np.mean(acts)) if acts else None,
                       float(ics.mean()), float(ics.std(ddof=1)), share, streak,
                       after_n, after_next, float(a.mean()) if len(a) else None)


def live_payload(rows: list[LiveScore], pending: list[str], horizon: int,
                 verdict: LiveVerdict | None) -> dict:
    """JSON-ready live record for the report (all floats finite or None)."""
    def f(v: float | None) -> float | None:
        return None if v is None or not np.isfinite(v) else float(v)
    out: dict = {
        "horizon": horizon, "pending": list(pending),
        "rows": [{"run_date": r.run_date, "scored_to": r.scored_to, "n": r.n,
                  "ic": f(r.ic), "picks": f(r.picks), "universe": f(r.universe),
                  "active": f(r.active), "independent": r.independent} for r in rows],
        "verdict": None,
    }
    if verdict is not None:
        out["verdict"] = {k: (f(v) if isinstance(v, float) else v)
                          for k, v in verdict.__dict__.items()}
    return out


def live_summary(name: str, rows: list[LiveScore], horizon: int,
                 verdict: LiveVerdict | None) -> str:
    """Plain-text live record for the run log."""
    head = f"--- Live record ({name}, {horizon}-day horizon) ---"
    if not rows:
        return head + "\nNo logged predictions old enough to score yet."
    lines = [head]
    for r in rows:
        act = "n/a" if r.active is None else f"{r.active:+.2%}"
        lines.append(f"{r.run_date} -> {r.scored_to}: IC {r.ic:+.3f}, picks minus "
                     f"universe {act}{'' if r.independent else '  (overlaps)'}")
    if verdict is not None:
        share = ("n/a" if verdict.share_as_bad is None
                 else f"{verdict.share_as_bad:.1%}")
        lines.append(f"Independent periods: {verdict.n}; mean IC {verdict.mean_ic:+.3f} vs "
                     f"walk-forward {verdict.bt_mean_ic:+.3f} (sd {verdict.bt_sd_ic:.3f}); "
                     f"walk-forward {verdict.n}-period stretches this bad: {share}")
        if verdict.streak and verdict.after_streak_next is not None:
            lines.append(f"Walk-forward after {verdict.streak} lagging period(s) in a row "
                         f"({verdict.after_streak_n} times): next period "
                         f"{verdict.after_streak_next:+.2%} vs {verdict.all_next:+.2%} on "
                         "average")
    return "\n".join(lines)
