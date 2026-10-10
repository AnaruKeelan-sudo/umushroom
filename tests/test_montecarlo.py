from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umushroom.montecarlo import (
    block_bootstrap_indices,
    combine_sleeves,
    simulate,
)

IDX = pd.bdate_range("2019-01-04", periods=300, freq="W-FRI")


def _series(values) -> pd.Series:
    return pd.Series(values, index=IDX[: len(values)])


def test_constant_returns_give_deterministic_paths():
    """Known answer: +1% every week for 8 weeks -> exactly 1.01**8 on every path."""
    s = _series(np.full(100, 0.01))
    b = _series(np.zeros(100))
    mc = simulate(s, b, horizon_weeks=8, start_value=100_000, n_paths=500)
    np.testing.assert_allclose(mc.final_strategy, 100_000 * 1.01 ** 8)
    st = mc.stats()
    assert st["prob_beat_benchmark"] == 1.0 and st["prob_loss"] == 0.0


def test_paths_start_at_start_value_and_have_right_shape():
    rng = np.random.default_rng(0)
    s, b = _series(rng.normal(0.002, 0.02, 200)), _series(rng.normal(0.001, 0.02, 200))
    mc = simulate(s, b, horizon_weeks=6, start_value=50_000, n_paths=1000)
    assert mc.strategy_paths.shape == (1000, 7)
    assert (mc.strategy_paths[:, 0] == 50_000).all()
    bands = mc.percentile_bands()
    assert (bands["p5"] <= bands["p25"]).all() and (bands["p25"] <= bands["p50"]).all()
    assert (bands["p50"] <= bands["p75"]).all() and (bands["p75"] <= bands["p95"]).all()


def test_pairing_keeps_strategy_and_benchmark_in_the_same_week():
    """Strategy = benchmark + 0.5% every week -> strategy wins on EVERY path.
    If the two were drawn independently this would fail."""
    rng = np.random.default_rng(1)
    b = _series(rng.normal(0, 0.03, 200))
    s = b + 0.005
    mc = simulate(s, b, horizon_weeks=8, start_value=1.0, n_paths=2000)
    assert mc.stats()["prob_beat_benchmark"] == 1.0


def test_blocks_are_consecutive_weeks():
    rng = np.random.default_rng(2)
    idx = block_bootstrap_indices(50, horizon=12, n_paths=10, block_size=4, rng=rng)
    assert idx.shape == (10, 12)
    for path in idx:
        for k in range(0, 12, 4):
            block = path[k:k + 4]
            assert all((block[i + 1] - block[i]) % 50 == 1 for i in range(len(block) - 1))


def test_reproducible_with_seed():
    rng = np.random.default_rng(3)
    s, b = _series(rng.normal(0, 0.02, 120)), _series(rng.normal(0, 0.02, 120))
    a = simulate(s, b, 8, 1.0, n_paths=500, seed=7)
    c = simulate(s, b, 8, 1.0, n_paths=500, seed=7)
    np.testing.assert_array_equal(a.strategy_paths, c.strategy_paths)


def test_mean_growth_matches_history_on_average():
    """With many paths the average weekly log-growth matches the sample's."""
    rng = np.random.default_rng(4)
    r = rng.normal(0.003, 0.02, 280)
    mc = simulate(_series(r), _series(np.zeros(280)), 10, 1.0, n_paths=20_000)
    sim = np.log(mc.final_strategy).mean() / 10
    assert sim == pytest.approx(np.log1p(r).mean(), abs=0.0007)


@pytest.mark.parametrize("kw", [{"horizon_weeks": 0}, {"n_paths": 10}])
def test_rejects_bad_parameters(kw):
    rng = np.random.default_rng(5)
    s = _series(rng.normal(0, 0.02, 50))
    args = {"horizon_weeks": 8, "start_value": 1.0, "n_paths": 500, **kw}
    with pytest.raises(ValueError):
        simulate(s, s, **args)


def test_rejects_too_little_history():
    s = _series(np.zeros(10))
    with pytest.raises(ValueError):
        simulate(s, s, 8, 1.0, n_paths=500)


def test_combine_sleeves_weights_and_aligns():
    a = _series([0.01, 0.02, 0.03])
    b = pd.Series([0.10, 0.20], index=IDX[1:3])
    out = combine_sleeves([(0.7, a), (0.3, b)])
    assert list(out.index) == list(IDX[1:3])          # common dates only
    assert out.iloc[0] == pytest.approx(0.7 * 0.02 + 0.3 * 0.10)


def test_bootstrap_annual_alpha_centres_on_sample_mean():
    from umushroom.montecarlo import bootstrap_annual_alpha
    rng = np.random.default_rng(6)
    a = _series(rng.normal(0.002, 0.01, 250))
    draws = bootstrap_annual_alpha(a, 50, n_sims=4000)
    assert draws.mean() == pytest.approx(a.mean() * 50, rel=0.05)
    assert draws.std() > 0
