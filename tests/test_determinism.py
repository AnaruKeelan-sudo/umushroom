"""v4.22: nothing on the path from scores to orders is random.

The picks are the model's ranking, top-N by predicted score; equal scores
are broken alphabetically (a stable sort), never by memory layout; the same
inputs give the same orders every time; and the gradient-boosting learner
gives the same predictions on every fit with the same seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umushroom.allocation import generate_allocation
from umushroom.features import build_training_set
from umushroom.model import RankingModel
from umushroom.selection import rank_order


@pytest.fixture
def trained_model(synthetic_prices):
    X, y = build_training_set(synthetic_prices, forward_return_days=5,
                              step=5, min_history_days=70)
    return RankingModel(alpha=1.0, random_seed=42).fit(X, y)


def test_rank_order_is_best_first_with_alphabetical_ties():
    preds = pd.Series({"ZZZ": 0.5, "BBB": 0.5, "AAA": 0.5, "MMM": 0.9, "QQQ": -0.1})
    assert rank_order(preds) == ["MMM", "AAA", "BBB", "ZZZ", "QQQ"]
    # the input order never matters
    assert rank_order(preds.iloc[::-1]) == rank_order(preds)
    assert rank_order(pd.Series(dtype=float)) == []


class _Constant:
    """A 'model' that scores every stock the same: the picks must then be
    the alphabetically first names, not whatever the sort happened to do."""
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(0.25, index=X.index)


def test_equal_scores_pick_alphabetically(synthetic_prices):
    a = generate_allocation(synthetic_prices, _Constant(), top_n=3, balance=10_000.0)
    expected = sorted(a.all_scores.index)[:3]
    assert list(a.orders.index) == expected


def test_same_inputs_give_the_same_orders(synthetic_prices, trained_model):
    kw: dict = {"top_n": 5, "balance": 100_000.0, "weighting": "hrp"}
    a = generate_allocation(synthetic_prices, trained_model, **kw)
    b = generate_allocation(synthetic_prices, trained_model, **kw)
    pd.testing.assert_frame_equal(a.orders, b.orders)
    pd.testing.assert_series_equal(a.all_scores, b.all_scores)


@pytest.mark.parametrize("model_type", ["ridge", "gbm", "mlp"])
def test_every_learner_is_reproducible_with_its_seed(synthetic_prices, model_type):
    X, y = build_training_set(synthetic_prices, forward_return_days=5, step=5,
                              min_history_days=60)
    m1 = RankingModel(random_seed=42, model_type=model_type).fit(X, y)
    m2 = RankingModel(random_seed=42, model_type=model_type).fit(X, y)
    np.testing.assert_array_equal(m1.predict(X).to_numpy(), m2.predict(X).to_numpy())


# ------------------------------------------ v4.23: seed-independent gbm bins

def test_quantize_ranks_snaps_to_255_levels_and_keeps_nan():
    from umushroom.model import GBM_BINS, quantize_ranks
    X = pd.DataFrame(np.random.default_rng(0).uniform(-0.5, 0.5, (20_000, 2)),
                     columns=["a", "b"])
    X.iloc[0, 0] = np.nan
    Q = quantize_ranks(X)
    assert Q["b"].nunique() <= GBM_BINS and np.isnan(Q.iloc[0, 0])
    assert Q.min().min() >= -0.5 and Q.max().max() <= 0.5
    assert (Q - X).abs().max().max() <= 0.5 / (GBM_BINS - 1) + 1e-12
    pd.testing.assert_frame_equal(quantize_ranks(Q), Q)           # idempotent
    assert quantize_ranks(pd.DataFrame({"a": [0.7, -0.9]}))["a"].tolist() == [0.5, -0.5]


def _big_panel(n_rows: int, seed: int = 0):
    """Centred-rank-like features over MORE than scikit-learn's 200,000-row
    binning subsample, indexed (date, ticker) like build_training_set."""
    rng = np.random.default_rng(seed)
    n_dates, n_tickers = n_rows // 100, 100
    idx = pd.MultiIndex.from_product([pd.bdate_range("2010-01-01", periods=n_dates),
                                      [f"T{i}" for i in range(n_tickers)]],
                                     names=["date", "ticker"])
    X = pd.DataFrame(rng.uniform(-0.5, 0.5, (len(idx), 3)), index=idx,
                     columns=["f1", "f2", "f3"])
    y = pd.Series(0.3 * X["f1"] - 0.2 * X["f2"] + rng.normal(0, 0.5, len(idx)),
                  index=idx)
    return X, y


def test_gbm_fixed_bins_do_not_depend_on_the_seed(monkeypatch):
    import umushroom.model as model_mod
    monkeypatch.setattr(model_mod, "GBM_ITER", 4)
    X, y = _big_panel(210_000)                     # above the subsample threshold
    with pytest.raises(ValueError):
        RankingModel(model_type="gbm", gbm_bins="bogus")
    fixed = [RankingModel(random_seed=s, model_type="gbm", gbm_bins="fixed").fit(X, y)
             .predict(X.iloc[:5000]).to_numpy() for s in (1, 2)]
    np.testing.assert_array_equal(fixed[0], fixed[1])


def test_gbm_bin_mode_survives_save_and_load(tmp_path, synthetic_prices):
    X, y = build_training_set(synthetic_prices, forward_return_days=5, step=5,
                              min_history_days=60)
    m = RankingModel(model_type="gbm", gbm_bins="fixed").fit(X, y)
    m.save(tmp_path / "m")
    loaded = RankingModel.load(tmp_path / "m")
    assert loaded.gbm_bins == "fixed"
    np.testing.assert_array_equal(loaded.predict(X).to_numpy(), m.predict(X).to_numpy())
