from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from umushroom.exceptions import ModelNotFittedError
from umushroom.features import build_training_set
from umushroom.model import RankingModel


@pytest.fixture
def trained_model(synthetic_prices):
    X, y = build_training_set(synthetic_prices, forward_return_days=5,
                               step=5, min_history_days=70)
    model = RankingModel(alpha=1.0, random_seed=42)
    model.fit(X, y)
    return model, X, y


def test_predict_before_fit_raises():
    model = RankingModel()
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ModelNotFittedError):
        model.predict(X)


def test_feature_importance_before_fit_raises():
    model = RankingModel()
    with pytest.raises(ModelNotFittedError):
        model.feature_importance()


def test_fit_rejects_mismatched_lengths():
    model = RankingModel()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 2])
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_rejects_empty_data():
    model = RankingModel()
    with pytest.raises(ValueError):
        model.fit(pd.DataFrame(), pd.Series(dtype=float))


def test_predict_rejects_mismatched_columns(trained_model):
    model, X, _y = trained_model
    bad_X = X.rename(columns={X.columns[0]: "renamed_column"})
    with pytest.raises(ValueError):
        model.predict(bad_X)


def test_predict_returns_series_indexed_like_input(trained_model):
    model, X, _y = trained_model
    preds = model.predict(X)
    assert isinstance(preds, pd.Series)
    assert list(preds.index) == list(X.index)
    assert len(preds) == len(X)


def test_feature_importance_covers_all_features(trained_model):
    model, X, _y = trained_model
    importance = model.feature_importance()
    assert set(importance.index) == set(X.columns)


def test_cross_validate_returns_finite_stats(trained_model):
    model, X, y = trained_model
    stats = model.cross_validate(X, y)
    assert np.isfinite(stats["rmse_mean"])
    assert stats["rmse_mean"] >= 0


def test_save_load_round_trip_predictions_match(trained_model):
    model, X, _y = trained_model
    preds_before = model.predict(X)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model"
        model.save(path)
        loaded = RankingModel.load(path)

    preds_after = loaded.predict(X)
    pd.testing.assert_series_equal(preds_before, preds_after)


def test_save_unfitted_model_raises():
    model = RankingModel()
    with pytest.raises(ModelNotFittedError):
        model.save("/tmp/should_not_be_created")


def test_same_seed_same_data_gives_identical_predictions(synthetic_prices):
    """Reproducibility: two independently trained models with the same
    seed and data must produce bit-identical predictions. If this ever
    fails, something in the pipeline has introduced non-determinism.
    """
    X, y = build_training_set(synthetic_prices, forward_return_days=5,
                               step=5, min_history_days=70)

    model_a = RankingModel(alpha=1.0, random_seed=7).fit(X, y)
    model_b = RankingModel(alpha=1.0, random_seed=7).fit(X, y)

    pd.testing.assert_series_equal(model_a.predict(X), model_b.predict(X))


# ---------------------------------------------------------------- v4 tests
def test_cross_validate_reports_ic(trained_model):
    model, X, y = trained_model
    stats = model.cross_validate(X, y, purge_dates=1)
    assert -1 <= stats["ic_mean"] <= 1
    assert stats["ic_n_dates"] > 0


def test_cross_validate_finds_skill_on_planted_signal(planted_signal_prices):
    X, y = build_training_set(planted_signal_prices, forward_return_days=5,
                              step=5, min_history_days=300)
    stats = RankingModel().cross_validate(X, y, purge_dates=1)
    assert stats["ic_mean"] > 0.03 and stats["ic_tstat"] > 2


def test_cross_validate_never_splits_a_date(trained_model, monkeypatch):
    """A date's cross-section must be wholly in train or wholly in test."""
    model, X, y = trained_model
    seen = []
    import umushroom.model as m
    real_ridge = m.Ridge

    class SpyRidge(real_ridge):
        def fit(self, Xf, yf, *a, **k):
            seen.append(set(Xf.index.get_level_values(0)))
            return super().fit(Xf, yf, *a, **k)

        def predict(self, Xp):
            test_dates = set(Xp.index.get_level_values(0))
            assert not (seen[-1] & test_dates)
            return super().predict(Xp)

    monkeypatch.setattr(m, "Ridge", SpyRidge)
    model.cross_validate(X, y, purge_dates=1)
    assert seen
