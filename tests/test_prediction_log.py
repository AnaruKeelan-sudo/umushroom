from __future__ import annotations

import pandas as pd
import pytest

from umushroom.prediction_log import evaluate_latest_predictions, log_predictions


@pytest.fixture
def sample_scores_and_prices():
    scores = pd.Series({"AAA": 0.02, "BBB": -0.01, "CCC": 0.05}, name="predicted_return")
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 200.0})
    return scores, prices


def test_evaluate_with_no_log_returns_none(tmp_path, sample_scores_and_prices):
    _scores, prices = sample_scores_and_prices
    log_path = tmp_path / "pred_log.csv"
    report = evaluate_latest_predictions(log_path, prices, pd.Timestamp("2024-01-02"))
    assert report is None


def test_log_then_evaluate_next_day(tmp_path, sample_scores_and_prices):
    scores, prices = sample_scores_and_prices
    log_path = tmp_path / "pred_log.csv"

    day1 = pd.Timestamp("2024-01-02")
    log_predictions(log_path, day1, scores, prices, picked_tickers={"AAA", "CCC"})

    # Simulate the next day: AAA up 5%, BBB down 2%, CCC up 1%
    day2_prices = pd.Series({"AAA": 105.0, "BBB": 49.0, "CCC": 202.0})
    day2 = pd.Timestamp("2024-01-03")

    report = evaluate_latest_predictions(log_path, day2_prices, day2)
    assert report is not None
    assert report.n_tickers == 3
    assert report.evaluated_run_date == day1
    # AAA predicted +0.02, realized +0.05 -> same sign
    # BBB predicted -0.01, realized -0.02 -> same sign
    # CCC predicted +0.05, realized +0.01 -> same sign
    assert report.directional_accuracy == 1.0


def test_evaluate_only_scores_strictly_prior_dates(tmp_path, sample_scores_and_prices):
    """Running the pipeline twice on the SAME day should not have that
    day's own predictions evaluate themselves."""
    scores, prices = sample_scores_and_prices
    log_path = tmp_path / "pred_log.csv"
    today = pd.Timestamp("2024-01-02")

    log_predictions(log_path, today, scores, prices, picked_tickers={"AAA"})
    report = evaluate_latest_predictions(log_path, prices, today)
    assert report is None  # no STRICTLY PRIOR entries exist yet


def test_log_is_idempotent_for_same_run_date(tmp_path, sample_scores_and_prices):
    scores, prices = sample_scores_and_prices
    log_path = tmp_path / "pred_log.csv"
    day1 = pd.Timestamp("2024-01-02")

    log_predictions(log_path, day1, scores, prices, picked_tickers={"AAA"})
    log_predictions(log_path, day1, scores, prices, picked_tickers={"AAA"})  # re-run same day

    log = pd.read_csv(log_path)
    assert len(log) == 3  # not 6 -- second write replaced, not appended


def test_top_pick_vs_rest_split(tmp_path):
    scores = pd.Series({"WINNER": 0.05, "LOSER": -0.05})
    prices_day1 = pd.Series({"WINNER": 100.0, "LOSER": 100.0})
    log_path = tmp_path / "pred_log.csv"
    day1 = pd.Timestamp("2024-01-02")
    log_predictions(log_path, day1, scores, prices_day1, picked_tickers={"WINNER"})

    # WINNER actually goes up, LOSER actually goes down further
    prices_day2 = pd.Series({"WINNER": 110.0, "LOSER": 90.0})
    day2 = pd.Timestamp("2024-01-03")
    report = evaluate_latest_predictions(log_path, prices_day2, day2)

    assert report.top_pick_realized_return == pytest.approx(0.10)
    assert report.rest_of_universe_realized_return == pytest.approx(-0.10)


def test_missing_ticker_in_current_prices_is_excluded(tmp_path, sample_scores_and_prices):
    scores, prices = sample_scores_and_prices
    log_path = tmp_path / "pred_log.csv"
    day1 = pd.Timestamp("2024-01-02")
    log_predictions(log_path, day1, scores, prices, picked_tickers={"AAA"})

    # CCC dropped out of the universe by day 2 (e.g. delisted)
    day2_prices = pd.Series({"AAA": 105.0, "BBB": 49.0})
    day2 = pd.Timestamp("2024-01-03")
    report = evaluate_latest_predictions(log_path, day2_prices, day2)
    assert report.n_tickers == 2


# ------------------------------------------------- v4.25: the live record
import numpy as np

from umushroom.prediction_log import (
    LiveScore,
    live_payload,
    live_summary,
    live_verdict,
    score_logged,
)


def _panel(n_days=30, n_names=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-10-01", periods=n_days)
    names = [f"S{i:02d}" for i in range(n_names)]
    rets = rng.normal(0, 0.01, (n_days, n_names))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=names)


def _log_day(log_path, prices, day, scores, picks):
    log_predictions(log_path, pd.Timestamp(day), scores, prices.loc[pd.Timestamp(day)],
                    picked_tickers=set(picks))


def test_scores_a_perfect_and_a_backwards_forecast_at_the_horizon(tmp_path):
    px = _panel()
    log = tmp_path / "log.csv"
    d1, d2 = px.index[2], px.index[9]
    fwd1 = px.iloc[2 + 5] / px.iloc[2] - 1           # the realised 5-day returns
    fwd2 = px.iloc[9 + 5] / px.iloc[9] - 1
    _log_day(log, px, d1, fwd1, fwd1.nlargest(3).index)        # knew the future
    _log_day(log, px, d2, -fwd2, fwd2.nsmallest(3).index)      # exactly backwards
    _log_day(log, px, px.index[-2], fwd1, [])                  # too recent to score
    rows, pending = score_logged(log, px, horizon=5)
    assert [r.run_date for r in rows] == [d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d")]
    assert rows[0].ic == pytest.approx(1.0) and rows[1].ic == pytest.approx(-1.0)
    assert rows[0].scored_to == px.index[7].strftime("%Y-%m-%d")
    assert rows[0].picks == pytest.approx(fwd1.nlargest(3).mean())
    assert rows[0].universe == pytest.approx(fwd1.mean())
    assert rows[0].active > 0 > rows[1].active
    assert all(r.independent for r in rows)                    # 7 closes apart >= 5
    assert pending == [px.index[-2].strftime("%Y-%m-%d")]


def test_overlapping_cross_sections_are_scored_but_not_counted_twice(tmp_path):
    px = _panel()
    log = tmp_path / "log.csv"
    for k in (2, 3, 4, 7, 12):          # 3 and 4 overlap 2's window; 7 starts at its end
        _log_day(log, px, px.index[k], px.iloc[k], [])
    rows, _ = score_logged(log, px, horizon=5)
    assert [r.independent for r in rows] == [True, False, False, True, True]


def test_live_verdict_against_the_walk_forward_spread():
    rows = [LiveScore("2026-10-07", "2026-10-14", 500, -0.10, -0.01, 0.0, True),
            LiveScore("2026-10-08", "2026-10-15", 500, -0.90, 0.0, 0.0, False),
            LiveScore("2026-10-14", "2026-10-21", 500, -0.20, -0.02, 0.0, True)]
    dates = pd.bdate_range("2018-01-01", periods=12, freq="W-MON")
    ic = pd.Series([0.05, -0.20, -0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                    0.05], index=dates)
    active = pd.Series([-0.01, -0.01, 0.03, -0.01, -0.01, 0.02, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0], index=dates)
    v = live_verdict(rows, ic, active)
    assert v.n == 2 and v.mean_ic == pytest.approx(-0.15)       # the overlap is ignored
    assert v.mean_active == pytest.approx(-0.015)
    # 2-period stretches of the walk-forward with a mean IC <= -0.15: only (-0.2, -0.2)
    assert v.share_as_bad == pytest.approx(1 / 11)
    assert v.streak == 2
    # after two lagging periods in a row the walk-forward's next period was +3%, +2%
    assert v.after_streak_n == 2 and v.after_streak_next == pytest.approx(0.025)
    assert v.bt_mean_ic == pytest.approx(ic.mean())
    payload = live_payload(rows, ["2026-10-21"], 5, v)
    import json
    json.dumps(payload, allow_nan=False)                         # strict JSON
    assert payload["verdict"]["streak"] == 2 and len(payload["rows"]) == 3
    text = live_summary("equity", rows, 5, v)
    assert "Independent periods: 2" in text and "(overlaps)" in text


def test_live_verdict_needs_independent_rows_and_a_walk_forward():
    assert live_verdict([], pd.Series([0.1] * 20, dtype=float), pd.Series(dtype=float)) is None
    one = [LiveScore("2026-10-07", "2026-10-14", 500, 0.1, None, 0.0, True)]
    assert live_verdict(one, pd.Series([0.1] * 5, dtype=float), pd.Series(dtype=float)) is None


def test_score_logged_without_a_log_or_with_too_few_names(tmp_path):
    px = _panel(n_names=5)
    assert score_logged(tmp_path / "none.csv", px, 5) == ([], [])
    log = tmp_path / "log.csv"
    _log_day(log, px, px.index[2], px.iloc[2], [])
    assert score_logged(log, px, 5) == ([], [])                  # 5 < MIN_LIVE_NAMES (30)
