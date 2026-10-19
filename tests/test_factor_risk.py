"""factor_risk.py (v4.20): known-answer factor parity, exposures, causality,
constraints and failure modes -- all on synthetic data with a planted
two-factor structure whose answer is known in closed form."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umushroom.exceptions import DataError
from umushroom.factor_risk import (
    MIN_FACTOR_DAILY_VOL,
    MIN_GROUP_MEMBERS,
    MIN_WEIGHT_MULTIPLE,
    factor_contributions,
    factor_exposures,
    factor_parity_weights,
    factor_risk,
    factor_weight,
    group_factor_returns,
)


def _two_factor_universe(vol1=0.010, vol2=0.020, n=600, seed=11, noise=0.0005):
    """10 'ga' assets = F1 + tiny noise, 10 'gb' assets = F2 + tiny noise,
    F1 and F2 independent. The group averages ARE the factors, so the
    factor-parity answer is known: w_A * vol1 == w_B * vol2."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    f1, f2 = rng.normal(0, vol1, n), rng.normal(0, vol2, n)
    cols, groups = {}, {}
    for i in range(10):
        cols[f"A{i}"], groups[f"A{i}"] = f1 + rng.normal(0, noise, n), "ga"
        cols[f"B{i}"], groups[f"B{i}"] = f2 + rng.normal(0, noise, n), "gb"
    px = pd.DataFrame({k: 100 * np.cumprod(1 + v) for k, v in cols.items()}, index=idx)
    return px, groups


PICKS = ["A0", "A1", "B0", "B1"]


def test_factor_parity_known_answer_two_independent_factors():
    px, groups = _two_factor_universe()
    fr = factor_risk(px, PICKS, len(px) - 1, groups)
    # equal risk contributions: w_A * sigma1 == w_B * sigma2 -> w_A = 2 w_B
    assert fr.weights[["A0", "A1"]].sum() == pytest.approx(2 / 3, abs=0.03)
    assert fr.weights[["B0", "B1"]].sum() == pytest.approx(1 / 3, abs=0.03)
    assert fr.contributions["ga"] == pytest.approx(0.5, abs=0.02)
    assert fr.contributions["gb"] == pytest.approx(0.5, abs=0.02)
    assert fr.weights.sum() == pytest.approx(1.0) and (fr.weights >= 0).all()
    assert list(fr.weights.index) == PICKS
    assert fr.idiosyncratic_share < 0.05                 # the factors explain the picks
    assert fr.exposures.loc["A0", "ga"] == pytest.approx(1.0, abs=0.1)
    assert abs(fr.exposures.loc["A0", "gb"]) < 0.1
    assert fr.exposures.loc["B0", "gb"] == pytest.approx(1.0, abs=0.1)


def test_budgets_tilt_the_weights():
    px, groups = _two_factor_universe()
    fr = factor_risk(px, PICKS, len(px) - 1, groups,
                     budgets=pd.Series({"ga": 0.8, "gb": 0.2}))
    # (w_A s1)^2 : (w_B s2)^2 = 4 : 1  ->  w_A s1 = 2 w_B s2  ->  w_A = 4 w_B
    assert fr.weights[["A0", "A1"]].sum() == pytest.approx(0.8, abs=0.03)
    assert fr.contributions["ga"] == pytest.approx(0.8, abs=0.03)
    assert list(fr.budgets.index) == ["ga", "gb"] and fr.budgets["ga"] == 0.8


def test_max_weight_bound_is_respected():
    px, groups = _two_factor_universe()
    # A0 alone carries the whole 'ga' factor: parity wants 2/3 in it
    fr = factor_risk(px, ["A0", "B0", "B1", "B2"], len(px) - 1, groups, max_weight=0.4)
    assert fr.weights.max() <= 0.4 + 1e-9
    assert fr.weights.sum() == pytest.approx(1.0)
    assert fr.weights["A0"] == pytest.approx(0.4, abs=1e-6)


def test_every_pick_keeps_an_investable_floor_weight():
    """A pick the optimiser would park near zero still gets MIN_WEIGHT_MULTIPLE
    of an equal share -- the live run must be able to buy one unit of it."""
    px, groups = _two_factor_universe(vol1=0.002, vol2=0.030)   # B is 15x riskier
    picks = ["A0", "A1", "A2", "B0"]
    fr = factor_risk(px, picks, len(px) - 1, groups)
    floor = MIN_WEIGHT_MULTIPLE / len(picks)
    assert (fr.weights >= floor - 1e-9).all() and fr.weights.sum() == pytest.approx(1.0)
    assert fr.weights["B0"] == pytest.approx(floor, abs=1e-6)       # floor binds
    assert fr.weights.max() <= 2.0 / len(picks) + 1e-9               # default cap
    with pytest.raises(ValueError):
        factor_risk(px, picks, len(px) - 1, groups, min_weight=0.5)   # 4 x 0.5 > 1


def test_a_factor_that_does_not_move_is_not_a_factor():
    """The money-market average (~0.02%/day) blew OLS betas up to +7.7 in the
    first live run: such a group is dropped before exposures are estimated."""
    px, groups = _two_factor_universe()
    rng = np.random.default_rng(3)
    n = len(px)
    for i in range(6):                     # a near-constant 'cash' group
        px[f"C{i}"] = 100 * np.cumprod(1 + rng.normal(0.00001, MIN_FACTOR_DAILY_VOL / 5, n))
        groups[f"C{i}"] = "cash"
    f = group_factor_returns(px, groups, len(px) - 1)
    assert list(f.columns) == ["ga", "gb"]
    fr = factor_risk(px, ["A0", "B0", "C0"], len(px) - 1, groups)
    assert "cash" not in fr.exposures.columns
    assert fr.exposures.abs().max().max() < 2.0                      # sane betas


def test_factor_exposures_known_answer():
    rng = np.random.default_rng(2)
    n = 300
    f = pd.DataFrame({"f1": rng.normal(0, 0.01, n), "f2": rng.normal(0, 0.01, n)},
                     index=pd.bdate_range("2024-01-01", periods=n))
    a = pd.DataFrame({"X": 2 * f["f1"] - 1 * f["f2"] + rng.normal(0, 0.001, n)})
    B, resid = factor_exposures(a, f)
    assert B.loc["X", "f1"] == pytest.approx(2.0, abs=0.05)
    assert B.loc["X", "f2"] == pytest.approx(-1.0, abs=0.05)
    assert resid["X"] == pytest.approx(1e-6, rel=0.5)
    with pytest.raises(DataError):
        factor_exposures(a.iloc[:30], f)


def test_contributions_decompose_the_systematic_variance():
    px, groups = _two_factor_universe()
    fr = factor_risk(px, PICKS, len(px) - 1, groups)
    B = fr.exposures[fr.budgets.index]
    rc = factor_contributions(fr.weights, B, fr.factor_cov)
    b = B.T.to_numpy() @ fr.weights.to_numpy()
    cov = fr.factor_cov.loc[fr.budgets.index, fr.budgets.index].to_numpy()
    assert rc.sum() == pytest.approx(float(b @ cov @ b))
    assert fr.contributions.sum() == pytest.approx(1.0)


def test_factor_risk_is_causal_and_deterministic():
    px, groups = _two_factor_universe()
    idx = 400
    corrupt = px.copy()
    corrupt.iloc[idx + 1:] *= 5.0
    a = factor_weight(px, PICKS, idx, groups)
    b = factor_weight(corrupt, PICKS, idx, groups)
    c = factor_weight(px.iloc[: idx + 1], PICKS, idx, groups)
    pd.testing.assert_series_equal(a, b)
    pd.testing.assert_series_equal(a, c)
    pd.testing.assert_series_equal(a, factor_weight(px, PICKS, idx, groups))


def test_group_factor_returns_drops_small_groups_and_needs_two():
    px, groups = _two_factor_universe()
    f = group_factor_returns(px, groups, len(px) - 1)
    assert list(f.columns) == ["ga", "gb"] and len(f) == 251
    few = dict(groups)
    for i in range(10 - MIN_GROUP_MEMBERS + 1):      # leaves 'gb' with 4 members
        few[f"B{i}"] = f"solo{i}"
    with pytest.raises(DataError, match="at least 2 factors"):
        group_factor_returns(px, few, len(px) - 1)
    with pytest.raises(ValueError):
        group_factor_returns(px, groups, len(px))


def test_picks_exposed_to_one_factor_only_is_an_error():
    px, groups = _two_factor_universe()
    with pytest.raises(DataError, match="nothing to balance"):
        factor_risk(px, ["A0", "A1", "A2"], len(px) - 1, groups)


def test_bad_inputs_raise_typed_errors():
    px, groups = _two_factor_universe()
    with pytest.raises(ValueError):
        factor_risk(px, ["A0"], len(px) - 1, groups)
    with pytest.raises(ValueError):
        factor_risk(px, ["A0", "A0"], len(px) - 1, groups)
    with pytest.raises(DataError):
        factor_risk(px, ["A0", "ZZ"], len(px) - 1, groups)
    with pytest.raises(DataError):                      # 30 days: not estimable
        factor_risk(px, ["A0", "B0"], 30, groups)
    expo = pd.DataFrame({"f1": [1.0, 0.0], "f2": [0.0, 1.0]}, index=["a", "b"])
    cov = pd.DataFrame(np.eye(2) * 1e-4, index=["f1", "f2"], columns=["f1", "f2"])
    with pytest.raises(ValueError):
        factor_parity_weights(expo, cov, budgets=pd.Series({"f1": -1.0, "f2": 2.0}))
    with pytest.raises(ValueError):
        factor_parity_weights(expo.iloc[:1], cov)


def test_summary_names_every_factor_and_pick():
    px, groups = _two_factor_universe()
    text = factor_risk(px, ["A0", "B0"], len(px) - 1, groups).summary()
    for word in ("ga", "gb", "A0", "B0", "idiosyncratic", "budget"):
        assert word in text
