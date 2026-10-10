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
