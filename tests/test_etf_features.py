"""The ETF feature set (features.py, feature_set="etf"): same correctness
properties as the stock factors -- no lookahead, panel == live path --
plus known-answer recovery of a planted trend signal through the whole
purged walk-forward pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umushroom.features import (
    ETF_FEATURE_NAMES,
    FEATURE_NAMES,
    MIN_BARS_REQUIRED,
    build_feature_panel,
    build_features_at_index,
    build_training_set,
    compute_raw_factors,
    feature_names,
    price_feature_names,
)
from umushroom.purged_backtest import run_purged_walk_forward

FIRST = MIN_BARS_REQUIRED - 1
KW = {"test_window_days": 21, "purge_days": 5, "embargo_days": 5,
      "rebalance_days": 5, "forward_return_days": 5, "top_n": 8,
      "min_train_days": 504, "model_alpha": 1.0, "random_seed": 42}


def test_feature_set_registry():
    assert price_feature_names("equity") == FEATURE_NAMES
    assert price_feature_names("etf") == ETF_FEATURE_NAMES
    assert feature_names(True, "etf") == ETF_FEATURE_NAMES + ["abn_volume_21_252",
                                                              "amihud_illiq_63"]
    assert len(set(ETF_FEATURE_NAMES)) == len(ETF_FEATURE_NAMES) == 13
    with pytest.raises(ValueError):
        price_feature_names("bonds")
    with pytest.raises(ValueError):
        compute_raw_factors(pd.DataFrame({"A": [1.0] * 300}), feature_set="bonds")


def test_etf_features_expected_columns_and_one_row_per_ticker(synthetic_prices):
    feats = build_features_at_index(synthetic_prices, FIRST + 10, feature_set="etf")
    assert list(feats.columns) == ETF_FEATURE_NAMES
    assert list(feats.index) == list(synthetic_prices.columns)
    assert feats.notna().all().all()
    # ranks are centred in [-0.5, 0.5]
    assert feats.min().min() >= -0.5 - 1e-9 and feats.max().max() <= 0.5 + 1e-9


def test_etf_raw_factors_no_lookahead(synthetic_prices):
    idx = FIRST + 60
    corrupted = synthetic_prices.copy()
    corrupted.iloc[idx + 1:] *= 3.7
    a = compute_raw_factors(synthetic_prices, feature_set="etf")
    b = compute_raw_factors(corrupted, feature_set="etf")
    for name in ETF_FEATURE_NAMES:
        pd.testing.assert_frame_equal(a[name].iloc[: idx + 1], b[name].iloc[: idx + 1])


def test_etf_features_no_lookahead_at_index(synthetic_prices):
    idx = FIRST + 100
    full = build_features_at_index(synthetic_prices, idx, feature_set="etf")
    trunc = build_features_at_index(synthetic_prices.iloc[: idx + 1], idx, feature_set="etf")
    pd.testing.assert_frame_equal(full, trunc)


def test_etf_panel_matches_live_path(synthetic_prices):
    groups = {c: ("bond" if i % 2 else "equity") for i, c in
              enumerate(synthetic_prices.columns)}
    panel = build_feature_panel(synthetic_prices, groups, feature_set="etf")
    assert list(panel.columns) == ETF_FEATURE_NAMES
    for idx in (FIRST, FIRST + 37, len(synthetic_prices) - 1):
        date = synthetic_prices.index[idx]
        from_panel = panel.xs(date, level="date").sort_index()
        direct = build_features_at_index(synthetic_prices, idx, groups=groups,
                                         feature_set="etf").sort_index()
        pd.testing.assert_frame_equal(from_panel, direct, check_names=False,
                                      atol=1e-9, rtol=0)


def test_trend_factors_have_the_right_sign():
    """A steadily rising series sits above its 200-day average, near its
    52-week high, with a positive 1-month time-series momentum; a falling
    one is the mirror image. Checked on raw (un-ranked) values."""
    dates = pd.bdate_range("2022-01-03", periods=400)
    up = 100 * np.exp(np.linspace(0, 0.4, 400))
    down = 100 * np.exp(np.linspace(0, -0.4, 400))
    px = pd.DataFrame({"UP": up, "DOWN": down}, index=dates)
    raw = compute_raw_factors(px, feature_set="etf")
    last = {k: v.iloc[-1] for k, v in raw.items()}
    assert last["trend_200"]["UP"] > 0 > last["trend_200"]["DOWN"]
    assert last["tsmom_21"]["UP"] > 0 > last["tsmom_21"]["DOWN"]
    assert last["high_52w_ratio"]["UP"] == pytest.approx(1.0)
    assert last["drawdown_63"]["UP"] == pytest.approx(0.0)
    assert last["drawdown_63"]["DOWN"] < 0
    assert last["trend_50_200"]["UP"] > 0 > last["trend_50_200"]["DOWN"]


def test_training_set_uses_etf_columns(synthetic_prices):
    X, y = build_training_set(synthetic_prices, forward_return_days=5, step=5,
                              min_history_days=300, feature_set="etf")
    assert list(X.columns) == ETF_FEATURE_NAMES and len(X) == len(y) > 0


def test_etf_pipeline_recovers_planted_trend(planted_signal_prices):
    """Known answer: on prices with a persistent latent drift the ETF
    trend factors must show real, significant ranking skill out of sample."""
    r = run_purged_walk_forward(planted_signal_prices, target="rank", feature_set="etf",
                                **KW)
    assert r.mean_information_coefficient > 0.03
    assert r.ic_tstat > 3
    assert r.has_evidence_of_skill


def test_etf_pipeline_finds_no_alpha_in_noise(synthetic_noise_prices):
    r = run_purged_walk_forward(synthetic_noise_prices, feature_set="etf", **KW)
    assert abs(r.ic_tstat) < 2
    assert not r.has_evidence_of_skill
