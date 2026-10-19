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


# ---------------------------------------------------------- v4.20 model types
from umushroom.model import MODEL_TYPES, mean_date_ic


@pytest.fixture
def small_training_set(synthetic_prices):
    return build_training_set(synthetic_prices, forward_return_days=5, step=5,
                              min_history_days=70)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_every_model_type_fits_predicts_and_round_trips(model_type, small_training_set):
    X, y = small_training_set
    model = RankingModel(alpha=1.0, random_seed=3, model_type=model_type).fit(X, y)
    preds = model.predict(X)
    assert list(preds.index) == list(X.index) and np.isfinite(preds).all()
    again = RankingModel(alpha=1.0, random_seed=3, model_type=model_type).fit(X, y)
    pd.testing.assert_series_equal(preds, again.predict(X))       # seeded
    with tempfile.TemporaryDirectory() as tmp:
        model.save(Path(tmp) / "m")
        loaded = RankingModel.load(Path(tmp) / "m")
    assert loaded.model_type == model_type
    pd.testing.assert_series_equal(preds, loaded.predict(X))
    imp = model.feature_importance()
    assert set(imp.index) == set(X.columns) and np.isfinite(imp).all()


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError):
        RankingModel(model_type="lstm")
    with pytest.raises(ValueError):
        RankingModel(purge_dates=-1)


def test_nonlinear_models_report_permutation_importance(small_training_set):
    X, y = small_training_set
    m = RankingModel(model_type="gbm").fit(X, y)
    imp = m.feature_importance()
    assert imp.name == "ic_drop" and "permutation" in m.importance_label
    assert RankingModel().importance_label == "Ridge coefficients"
    with tempfile.TemporaryDirectory() as tmp:
        m.save(Path(tmp) / "m")
        loaded = RankingModel.load(Path(tmp) / "m")
    with pytest.raises(ModelNotFittedError):      # no training data in memory
        loaded.feature_importance()


def test_mlp_early_stopping_validates_on_the_latest_dates_with_a_purge_gap(
        planted_signal_prices, monkeypatch):
    """The validation block is the LAST tenth of the dates, no training row
    may come from it, and `purge_dates` dates are skipped in between."""
    X, y = build_training_set(planted_signal_prices, forward_return_days=5,
                              step=5, min_history_days=300)
    import umushroom.model as m
    seen_train_dates: list = []
    real = m.MLPRegressor.partial_fit

    def spy(self, Xf, yf):
        seen_train_dates.append(Xf.index.get_level_values(0).max())
        return real(self, Xf, yf)
    monkeypatch.setattr(m.MLPRegressor, "partial_fit", spy)
    model = RankingModel(model_type="mlp", purge_dates=2).fit(X, y)
    s = model.training_summary
    dates = sorted(X.index.get_level_values(0).unique())
    n_val = int(s["n_validation_dates"])
    assert n_val == max(1, -(-len(dates) // 10))
    assert pd.Timestamp(s["first_validation_date"]) == dates[-n_val]
    assert pd.Timestamp(s["last_train_date"]) == dates[-n_val - 2 - 1]
    assert all(d == dates[-n_val - 3] for d in seen_train_dates)
    assert 1 <= s["best_epoch"] <= s["epochs_run"] <= m.MLP_MAX_EPOCHS


def test_mlp_on_few_dates_falls_back_to_fixed_epochs(small_training_set):
    X, y = small_training_set
    few = X.index.get_level_values(0).unique()[:5]
    mask = X.index.get_level_values(0).isin(few)
    model = RankingModel(model_type="mlp").fit(X[mask], y[mask])
    assert model.training_summary["n_validation_dates"] == 0
    assert np.isfinite(model.predict(X[mask])).all()


@pytest.mark.parametrize("model_type", ["gbm", "mlp"])
def test_nonlinear_models_find_the_planted_signal_in_cv(model_type, planted_signal_prices):
    X, y = build_training_set(planted_signal_prices, forward_return_days=5,
                              step=5, min_history_days=300)
    stats = RankingModel(model_type=model_type).cross_validate(X, y, purge_dates=1)
    assert stats["ic_mean"] > 0.02 and stats["ic_tstat"] > 2


def test_mean_date_ic_known_answer():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2024-01-01", periods=2),
                                      list("abcd")], names=["date", "ticker"])
    y = pd.Series([1, 2, 3, 4, 4, 3, 2, 1], index=idx, dtype=float)
    assert mean_date_ic(y, y) == pytest.approx(1.0)
    assert mean_date_ic(-y, y) == pytest.approx(-1.0)
