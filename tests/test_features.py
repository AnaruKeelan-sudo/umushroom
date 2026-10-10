from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umushroom.exceptions import InsufficientHistoryError
from umushroom.features import (
    FEATURE_NAMES,
    MIN_BARS_REQUIRED,
    POOLED_GROUP,
    build_feature_panel,
    build_features_at_index,
    build_training_set,
    centered_rank,
    compute_forward_returns,
    compute_raw_factors,
    compute_rsi,
    effective_groups,
    make_target,
)

FIRST = MIN_BARS_REQUIRED - 1   # first index with full feature history


def test_build_features_returns_expected_columns(synthetic_prices):
    feats = build_features_at_index(synthetic_prices, as_of_idx=FIRST)
    assert list(feats.columns) == FEATURE_NAMES


def test_build_features_one_row_per_ticker(synthetic_prices):
    feats = build_features_at_index(synthetic_prices, as_of_idx=FIRST + 50)
    assert set(feats.index) == set(synthetic_prices.columns)
    assert not feats.isna().any().any()


def test_build_features_raises_on_insufficient_history(synthetic_prices):
    with pytest.raises(InsufficientHistoryError):
        build_features_at_index(synthetic_prices, as_of_idx=FIRST - 1)


def test_build_features_no_lookahead(synthetic_prices):
    """The defining correctness property: features at index i must be
    identical whether or not data AFTER index i exists."""
    idx = FIRST + 100
    feats_full = build_features_at_index(synthetic_prices, as_of_idx=idx)
    feats_trunc = build_features_at_index(synthetic_prices.iloc[: idx + 1], as_of_idx=idx)
    pd.testing.assert_frame_equal(feats_full, feats_trunc)


def test_raw_factors_no_lookahead(synthetic_prices):
    """Stronger version: corrupting EVERY future price must not change
    any raw factor value at or before idx."""
    idx = FIRST + 60
    corrupted = synthetic_prices.copy()
    corrupted.iloc[idx + 1:] *= 3.7
    a = compute_raw_factors(synthetic_prices)
    b = compute_raw_factors(corrupted)
    for name in FEATURE_NAMES:
        pd.testing.assert_frame_equal(a[name].iloc[: idx + 1], b[name].iloc[: idx + 1])


def test_panel_matches_per_index_features(synthetic_prices):
    """The fast precomputed panel (used for training/backtests) must give
    the same features as the per-date path (used for today's live
    allocation). If these diverged, the model would trade on inputs
    different from those it was trained/validated on."""
    panel = build_feature_panel(synthetic_prices)
    for idx in (FIRST, FIRST + 37, len(synthetic_prices) - 1):
        date = synthetic_prices.index[idx]
        from_panel = panel.xs(date, level="date").sort_index()
        direct = build_features_at_index(synthetic_prices, idx).sort_index()
        pd.testing.assert_frame_equal(from_panel, direct, check_names=False,
                                      atol=1e-9, rtol=0)


def test_features_are_cross_sectionally_normalized(synthetic_prices):
    feats = build_features_at_index(synthetic_prices, FIRST + 10)
    assert feats.min().min() >= -0.5 - 1e-12
    assert feats.max().max() <= 0.5 + 1e-12
    np.testing.assert_allclose(feats.mean().to_numpy(), 0.0, atol=1e-12)


def test_normalized_features_invariant_to_common_price_scaling(synthetic_prices):
    """Multiplying every price by a common factor (e.g. an FX move that hits
    a whole market at once) must not change ranked features."""
    idx = FIRST + 20
    a = build_features_at_index(synthetic_prices, idx)
    b = build_features_at_index(synthetic_prices * 1.37, idx)
    pd.testing.assert_frame_equal(a, b, atol=1e-12, rtol=0)


def test_mom_12_1_skips_latest_month():
    """mom_12_1 at t must equal P[t-21]/P[t-252]-1 exactly, so the most
    recent month (short-term reversal territory) is excluded."""
    n = MIN_BARS_REQUIRED + 5
    rng = np.random.default_rng(0)
    data = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, 2)), axis=0),
                        columns=["A", "B"], index=pd.bdate_range("2020-01-01", periods=n))
    raw = compute_raw_factors(data)
    t = n - 1
    expected = data.iloc[t - 21] / data.iloc[t - 252] - 1
    pd.testing.assert_series_equal(raw["mom_12_1"].iloc[t], expected, check_names=False)


def test_beta_of_market_proxy_is_one():
    """A ticker whose returns ARE the universe's mean return direction
    should have beta ~1; a pure-noise ticker has beta ~0 relative to it
    as the universe grows. Here: identical tickers -> beta exactly 1."""
    n = MIN_BARS_REQUIRED + 5
    rng = np.random.default_rng(3)
    base = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    data = pd.DataFrame({"A": base, "B": base, "C": base},
                        index=pd.bdate_range("2020-01-01", periods=n))
    beta = compute_raw_factors(data)["beta_252"].iloc[-1]
    np.testing.assert_allclose(beta.to_numpy(), 1.0, atol=1e-8)


def test_centered_rank_bounds_and_nans():
    df = pd.DataFrame([[3.0, 1.0, np.nan, 2.0], [np.nan, 5.0, np.nan, np.nan]])
    r = centered_rank(df)
    assert r.iloc[0].tolist()[:2] == [0.5, -0.5]
    assert r.iloc[0, 3] == 0.0
    assert np.isnan(r.iloc[0, 2])
    assert r.iloc[1, 1] == 0.0          # single valid value -> 0
    assert r.iloc[1].isna().sum() == 3


def test_effective_groups_pools_small_groups():
    tickers = [f"L{i}" for i in range(6)] + ["S1", "S2"]
    groups = {t: ("LSE" if t.startswith("L") else "SIX") for t in tickers}
    g = effective_groups(tickers, groups)
    assert all(g[t] == "LSE" for t in tickers if t.startswith("L"))
    assert g["S1"] == POOLED_GROUP and g["S2"] == POOLED_GROUP


def test_exchange_neutral_ranking_removes_group_level_shift():
    """With exchange-neutral ranking, adding a constant to one group's
    factor values (a country-wide move) must not change its ranks."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(size=(4, 10)), columns=[f"T{i}" for i in range(10)])
    groups = {f"T{i}": ("A" if i < 5 else "B") for i in range(10)}
    shifted = df.copy()
    shifted[[f"T{i}" for i in range(5)]] += 100.0
    from umushroom.features import grouped_centered_rank
    pd.testing.assert_frame_equal(grouped_centered_rank(df, groups),
                                  grouped_centered_rank(shifted, groups))


@pytest.mark.parametrize("target", ["rank", "excess"])
def test_targets_are_cross_sectionally_centered(synthetic_prices, target):
    fwd = compute_forward_returns(synthetic_prices, 5)
    y = make_target(fwd, target, None).dropna(how="all")
    np.testing.assert_allclose(y.mean(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_make_target_rejects_unknown_type(synthetic_prices):
    with pytest.raises(ValueError):
        make_target(compute_forward_returns(synthetic_prices, 5), "bogus", None)


def test_compute_rsi_bounds(synthetic_prices):
    rsi = compute_rsi(synthetic_prices["TICK0"])
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_compute_rsi_handles_flat_series():
    flat = pd.Series([100.0] * 30)
    rsi = compute_rsi(flat)
    assert not rsi.isna().any()


def test_build_training_set_shapes(synthetic_prices):
    X, y = build_training_set(synthetic_prices, forward_return_days=5, step=5,
                              min_history_days=70)
    assert len(X) == len(y) > 0
    assert list(X.columns) == FEATURE_NAMES
    assert X.index.names == ["date", "ticker"]
    assert (X.index == y.index).all()


def test_build_training_set_raises_on_tiny_data(tiny_prices):
    with pytest.raises(InsufficientHistoryError):
        build_training_set(tiny_prices, forward_return_days=5, step=5,
                           min_history_days=70)


def test_build_training_set_raw_labels_match_actual_forward_return(synthetic_prices):
    """Label at (date, ticker) must equal P[t+h]/P[t]-1 exactly (no
    off-by-one, no wrong direction)."""
    h = 5
    _X, y = build_training_set(synthetic_prices, forward_return_days=h, step=7,
                               min_history_days=70, target="raw")
    for date, ticker in list(y.index)[:: max(1, len(y) // 25)]:
        i = synthetic_prices.index.get_loc(date)
        expected = synthetic_prices[ticker].iloc[i + h] / synthetic_prices[ticker].iloc[i] - 1
        assert y.loc[(date, ticker)] == pytest.approx(expected, abs=1e-12)


def test_build_training_set_respects_last_label_idx(synthetic_prices):
    """No training row may have a label that needs prices beyond the
    cut-off: feature index + h <= last_label_idx + h."""
    h, cut = 5, FIRST + 120
    X, _y = build_training_set(synthetic_prices, forward_return_days=h, step=1,
                               min_history_days=70, last_label_idx=cut)
    max_idx = synthetic_prices.index.get_loc(X.index.get_level_values("date").max())
    assert max_idx <= cut


def test_build_training_set_labels_never_use_prices_after_cutoff(synthetic_prices):
    """Leakage test: corrupting all prices after the label cut-off's
    horizon must leave every training row (X and y) unchanged."""
    h, cut = 5, FIRST + 120
    corrupted = synthetic_prices.copy()
    corrupted.iloc[cut + h + 1:] *= 5.0
    a = build_training_set(synthetic_prices, h, 1, 70, last_label_idx=cut)
    b = build_training_set(corrupted, h, 1, 70, last_label_idx=cut)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_series_equal(a[1], b[1])


def test_panel_matches_per_index_features_with_exchange_groups(planted_signal_prices):
    groups = {t: ("A" if i % 2 else "B") for i, t in enumerate(planted_signal_prices)}
    panel = build_feature_panel(planted_signal_prices, groups)
    idx = len(planted_signal_prices) - 1
    from_panel = panel.xs(planted_signal_prices.index[idx], level="date").sort_index()
    direct = build_features_at_index(planted_signal_prices, idx, groups=groups).sort_index()
    pd.testing.assert_frame_equal(from_panel, direct, check_names=False, atol=1e-9, rtol=0)


# ---------------------------------------------------------------- v4.4
from umushroom.features import (
    VOLUME_FEATURE_NAMES,
    hierarchical_groups,
)


def _ragged(prices: pd.DataFrame) -> pd.DataFrame:
    """TICK0 only lists half-way through the history."""
    out = prices.copy()
    out.iloc[:300, 0] = np.nan
    return out


def test_ragged_ticker_joins_once_it_has_a_year_of_history(synthetic_prices):
    data = _ragged(synthetic_prices)
    panel = build_feature_panel(data)
    first_date = panel.xs("TICK0", level="ticker").index.min()
    assert data.index.get_loc(first_date) == 300 + MIN_BARS_REQUIRED - 1
    # other tickers are unaffected and present from the start
    assert panel.xs("TICK1", level="ticker").index.min() == data.index[FIRST]


def test_ragged_features_no_lookahead(synthetic_prices):
    data = _ragged(synthetic_prices)
    idx = 560
    a = build_features_at_index(data, idx)
    b = build_features_at_index(data.iloc[: idx + 1], idx)
    pd.testing.assert_frame_equal(a, b)


def _dollar_volume(prices: pd.DataFrame, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shares = pd.DataFrame(rng.lognormal(12, 0.5, prices.shape),
                          index=prices.index, columns=prices.columns)
    return shares * prices


def test_volume_features_added_and_no_lookahead(synthetic_prices):
    dv = _dollar_volume(synthetic_prices)
    idx = FIRST + 120
    full = build_features_at_index(synthetic_prices, idx, dollar_volume=dv)
    assert list(full.columns) == FEATURE_NAMES + VOLUME_FEATURE_NAMES
    corrupted = dv.copy()
    corrupted.iloc[idx + 1:] *= 50
    pd.testing.assert_frame_equal(
        full, build_features_at_index(synthetic_prices, idx, dollar_volume=corrupted))


def test_panel_matches_per_index_with_volume(synthetic_prices):
    dv = _dollar_volume(synthetic_prices)
    panel = build_feature_panel(synthetic_prices, dollar_volume=dv)
    idx = len(synthetic_prices) - 1
    a = panel.xs(synthetic_prices.index[idx], level="date").sort_index()
    b = build_features_at_index(synthetic_prices, idx, dollar_volume=dv).sort_index()
    pd.testing.assert_frame_equal(a, b, check_names=False, atol=1e-9, rtol=0)


def test_missing_volume_is_neutral_not_dropped(synthetic_prices):
    """A stock Yahoo has no volume for must stay in the universe, scored
    as neutral (0) on the volume factors."""
    dv = _dollar_volume(synthetic_prices)
    dv["TICK3"] = np.nan
    feats = build_features_at_index(synthetic_prices, FIRST + 50, dollar_volume=dv)
    assert "TICK3" in feats.dropna().index
    assert (feats.loc["TICK3", VOLUME_FEATURE_NAMES] == 0).all()


def test_amihud_ranks_illiquid_stock_highest():
    n = MIN_BARS_REQUIRED + 5
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, 6)), axis=0),
                          index=idx, columns=list("ABCDEF"))
    dv = pd.DataFrame(1e7, index=idx, columns=prices.columns)
    dv["F"] = 1e4                                   # F trades 1000x less
    feats = build_features_at_index(prices, n - 1, dollar_volume=dv)
    assert feats["amihud_illiq_63"].idxmax() == "F"


def test_hierarchical_groups_falls_back_from_fine_to_coarse():
    tickers = [f"B{i}" for i in range(6)] + ["U1", "U2"] + ["X1"]
    sector = {**{f"B{i}": "bank" for i in range(6)}, "U1": "util", "U2": "util"}
    exch = {f"B{i}": ("LSE" if i < 5 else "SIX") for i in range(6)}
    exch.update({"U1": "LSE", "U2": "SIX", "X1": "US"})
    fine = {t: f"{sector[t]}|{exch[t]}" for t in sector}
    g = hierarchical_groups(tickers, [fine, sector])
    assert g["B0"] == "bank|LSE"          # 5 LSE banks: big enough
    assert g["B5"] == "bank"              # lone SIX bank -> all banks
    assert g["U1"] == POOLED_GROUP        # only 2 utilities -> global pool
    assert g["X1"] == POOLED_GROUP        # no sector at all
