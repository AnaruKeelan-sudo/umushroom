"""Model layer.

Wraps a scikit-learn estimator behind a small class that:
  - tracks whether it has been fit, raising a typed error if predict()
    is called first
  - exposes feature importances for defensibility ("why did the model
    pick this stock")
  - supports save/load so a trained model is a versioned artifact,
    not something silently re-trained differently each run
  - runs leakage-safe time-series cross-validation internally

Design note: for a dataset this size (a few hundred stocks x a couple
years of daily bars, subsampled every few days), a regularized linear
model is deliberately preferred over gradient boosting or deep
learning. More flexible models fit noise in financial return data
with this little signal-to-noise and this few independent samples;
Ridge regression's coefficients are also directly explainable, which
matters when you need to justify the strategy to a judge or to
yourself before risking contest capital.

v4.20 -- `model_type`. Ridge stays the default. Two non-linear learners
are available behind the SAME interface so that they can be compared
in the purged walk-forward on identical data (`--compare-model`) and
adopted only if a pre-declared rule says so:

  "mlp"  a small feed-forward neural network (one hidden layer of 32
         units) on the SAME cross-sectionally ranked factors. Trained with
         early stopping judged on the LATEST tenth of the training DATES
         (never a random row split: rows of one date share a label
         window), with `purge_dates` dates dropped between the two blocks
         so no training label overlaps a validation date. Every
         hyper-parameter is a module constant fixed before any result was
         seen; nothing is tuned on the contest data.
  "gbm"  histogram gradient boosting with a fixed number of rounds and
         strong regularisation -- the standard tabular benchmark any
         neural network must beat before it earns its complexity.

Why no LSTM / CNN over raw price sequences: the trend factors already
summarise the last year of prices at the horizons the literature
documents, the sample holds ~430 weekly cross-sections (8 years), and a
sequence model would have to re-learn those summaries from ~250 noisy
daily returns per instrument. Gu, Kelly & Xiu (2020) find the gains of
neural networks in return prediction come from shallow nets on
engineered characteristics over decades of data, not from depth. The
MLP here is that shallow net; the walk-forward decides whether even it
earns its place.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

from umushroom.exceptions import ModelNotFittedError
from umushroom.logging_config import get_logger

logger = get_logger("model")

MODEL_TYPES: tuple[str, ...] = ("ridge", "mlp", "gbm")

# ---- fixed in advance; never tuned on the contest data ----
MLP_HIDDEN: tuple[int, ...] = (32,)
MLP_L2 = 1e-3
MLP_LEARNING_RATE = 1e-3
MLP_BATCH = 512
MLP_MAX_EPOCHS = 40
MLP_PATIENCE = 5
MLP_VALIDATION_FRACTION = 0.10      # of DATES, the latest ones
MLP_MIN_DATES_FOR_VALIDATION = 20   # fewer dates: fixed epochs, no validation
MLP_FIXED_EPOCHS = 10
GBM_ITER = 150
GBM_LEARNING_RATE = 0.05
GBM_LEAVES = 15
GBM_MIN_LEAF = 100
GBM_L2 = 1.0


def _row_dates(X: pd.DataFrame) -> pd.Index:
    if isinstance(X.index, pd.MultiIndex):
        return X.index.get_level_values(0)
    return pd.Index(np.arange(len(X)))


def mean_date_ic(preds: pd.Series, y: pd.Series) -> float:
    """Mean per-date Spearman IC (dates with < 3 rows skipped)."""
    dates = _row_dates(y.to_frame())
    ics = []
    for d in pd.Index(dates).unique():
        m = dates == d
        if m.sum() >= 3:
            ic = spearmanr(preds[m], y[m])[0]
            if np.isfinite(ic):
                ics.append(float(ic))
    return float(np.mean(ics)) if ics else float("nan")


class RankingModel:
    def __init__(self, alpha: float = 1.0, random_seed: int = 42,
                 n_cv_splits: int = 5, model_type: str = "ridge",
                 purge_dates: int = 1):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}, "
                             f"got {model_type!r}")
        if purge_dates < 0:
            raise ValueError("purge_dates must be >= 0")
        self.alpha = alpha
        self.random_seed = random_seed
        self.n_cv_splits = n_cv_splits
        self.model_type = model_type
        self.purge_dates = purge_dates
        self._model = self._make_estimator()
        self._is_fitted = False
        self._feature_names: list[str] | None = None
        self._cv_rmse_mean: float | None = None
        self._cv_rmse_std: float | None = None
        self._cv_ic_mean: float | None = None
        self._cv_ic_tstat: float | None = None
        self._train_ref: tuple[pd.DataFrame, pd.Series] | None = None
        self._importance: pd.Series | None = None
        # how the final fit went (mlp: epochs and the validation block)
        self.training_summary: dict[str, float | str] = {}

    # ------------------------------------------------------------ estimators
    def _make_estimator(self):
        if self.model_type == "ridge":
            return Ridge(alpha=self.alpha, random_state=self.random_seed)
        if self.model_type == "gbm":
            return HistGradientBoostingRegressor(
                max_iter=GBM_ITER, learning_rate=GBM_LEARNING_RATE,
                max_leaf_nodes=GBM_LEAVES, min_samples_leaf=GBM_MIN_LEAF,
                l2_regularization=GBM_L2, early_stopping=False,
                random_state=self.random_seed)
        return MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN, activation="relu", solver="adam",
            alpha=MLP_L2, batch_size=MLP_BATCH, learning_rate_init=MLP_LEARNING_RATE,
            max_iter=MLP_MAX_EPOCHS, shuffle=True, random_state=self.random_seed)

    def _fit_estimator(self, est, X: pd.DataFrame, y: pd.Series
                       ) -> dict[str, float | str]:
        """Fit one estimator. For the MLP this is the early-stopping loop
        judged on the latest block of DATES (see module docstring)."""
        if self.model_type != "mlp":
            est.fit(X, y)
            return {}
        dates = _row_dates(X)
        unique = pd.Index(sorted(pd.Index(dates).unique()))
        n_val = max(1, math.ceil(len(unique) * MLP_VALIDATION_FRACTION))
        n_train = len(unique) - n_val - self.purge_dates
        if len(unique) < MLP_MIN_DATES_FOR_VALIDATION or n_train < 1:
            est.set_params(batch_size=min(MLP_BATCH, len(X)))
            for _ in range(MLP_FIXED_EPOCHS):
                est.partial_fit(X, y)
            return {"best_epoch": MLP_FIXED_EPOCHS, "epochs_run": MLP_FIXED_EPOCHS,
                    "validation_ic": float("nan"), "n_validation_dates": 0}
        train_dates, val_dates = unique[:n_train], unique[-n_val:]
        tr = dates.isin(train_dates)
        va = dates.isin(val_dates)
        X_tr, y_tr, X_va, y_va = X[tr], y[tr], X[va], y[va]
        est.set_params(batch_size=min(MLP_BATCH, len(X_tr)))
        best_ic, best_epoch, bad, epoch = -np.inf, 0, 0, 0
        best_weights: tuple[list, list] | None = None
        for epoch in range(1, MLP_MAX_EPOCHS + 1):
            est.partial_fit(X_tr, y_tr)
            ic = mean_date_ic(pd.Series(est.predict(X_va), index=X_va.index), y_va)
            if np.isfinite(ic) and ic > best_ic + 1e-6:
                best_ic, best_epoch, bad = ic, epoch, 0
                best_weights = (copy.deepcopy(est.coefs_), copy.deepcopy(est.intercepts_))
            else:
                bad += 1
                if bad >= MLP_PATIENCE:
                    break
        if best_weights is not None:
            est.coefs_, est.intercepts_ = best_weights
        return {"best_epoch": best_epoch, "epochs_run": epoch,
                "validation_ic": float(best_ic) if np.isfinite(best_ic) else float("nan"),
                "n_validation_dates": int(n_val),
                "last_train_date": str(train_dates[-1]),
                "first_validation_date": str(val_dates[0])}

    # ------------------------------------------------------------ public API
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       purge_dates: int = 1) -> dict[str, float]:
        """Time-series CV that splits on DATES, not rows.

        v3 split the pooled rows with TimeSeriesSplit, which could cut
        one date's cross-section in half (the model then trained on half
        of the same day it was validated on), and scored RMSE, which
        mostly measures market noise. v4:
          - splits unique dates chronologically, so a date is wholly in
            train or wholly in validation;
          - drops `purge_dates` dates before each validation block so
            overlapping forward-return labels cannot leak across it;
          - reports the per-date Information Coefficient (Spearman rank
            correlation of prediction vs. target), the standard measure
            of ranking skill, plus its t-statistic.
        Rows must be indexed (date, ticker), as build_training_set does;
        a flat index falls back to treating each row as its own date.
        """
        row_dates = _row_dates(X)
        unique_dates = pd.Index(sorted(row_dates.unique()))
        n_splits = min(self.n_cv_splits, max(2, len(unique_dates) // 10))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_rmse: list[float] = []
        date_ics: list[float] = []
        for fold, (tr_d, te_d) in enumerate(tscv.split(unique_dates), start=1):
            if purge_dates > 0:
                tr_d = tr_d[: max(0, len(tr_d) - purge_dates)]
            if len(tr_d) == 0:
                continue
            tr_mask = row_dates.isin(unique_dates[tr_d])
            te_mask = row_dates.isin(unique_dates[te_d])
            fold_model = self._make_estimator()
            self._fit_estimator(fold_model, X[tr_mask], y[tr_mask])
            preds = pd.Series(fold_model.predict(X[te_mask]), index=X.index[te_mask])
            y_te = y[te_mask]
            fold_rmse.append(float(np.sqrt(mean_squared_error(y_te, preds))))
            for d in unique_dates[te_d]:
                m = row_dates[te_mask] == d
                if m.sum() >= 3:
                    ic = spearmanr(preds[m], y_te[m])[0]
                    if np.isfinite(ic):
                        date_ics.append(float(ic))
            logger.debug(f"CV fold {fold}: RMSE={fold_rmse[-1]:.5f}")

        if not fold_rmse:
            raise ValueError("Cross-validation produced no usable folds")
        self._cv_rmse_mean = float(np.mean(fold_rmse))
        self._cv_rmse_std = float(np.std(fold_rmse))
        ic_arr = np.array(date_ics)
        ic_mean = float(ic_arr.mean()) if len(ic_arr) else float("nan")
        ic_t = (float(ic_mean / ic_arr.std(ddof=1) * np.sqrt(len(ic_arr)))
                if len(ic_arr) > 1 and ic_arr.std(ddof=1) > 0 else float("nan"))
        self._cv_ic_mean, self._cv_ic_tstat = ic_mean, ic_t
        logger.info(f"Cross-validated over {n_splits} date-folds: "
                    f"RMSE={self._cv_rmse_mean:.5f}, IC={ic_mean:+.4f} "
                    f"(t={ic_t:.2f}, {len(ic_arr)} dates)")
        return {"rmse_mean": self._cv_rmse_mean, "rmse_std": self._cv_rmse_std,
                "ic_mean": ic_mean, "ic_tstat": ic_t, "ic_n_dates": float(len(ic_arr))}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RankingModel:
        if X.empty or y.empty:
            raise ValueError("Cannot fit on empty training data")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        self._model = self._make_estimator()
        self.training_summary = self._fit_estimator(self._model, X, y)
        self._is_fitted = True
        self._feature_names = list(X.columns)
        self._train_ref = (X, y)
        self._importance = None
        extra = ""
        if self.model_type == "mlp" and self.training_summary:
            extra = (f", early-stopped at epoch {self.training_summary['best_epoch']}"
                     f" (validation IC {self.training_summary['validation_ic']:+.4f})")
        logger.info(f"Model fit on {len(X)} samples, {X.shape[1]} features "
                    f"({self.model_type}{extra})")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._is_fitted:
            raise ModelNotFittedError("Call fit() or load() before predict()")
        if list(X.columns) != self._feature_names:
            raise ValueError(
                f"Feature columns at predict time {list(X.columns)} don't "
                f"match training-time columns {self._feature_names}"
            )
        preds = self._model.predict(X)
        return pd.Series(preds, index=X.index, name="predicted_return")

    @property
    def importance_label(self) -> str:
        if self.model_type == "ridge":
            return "Ridge coefficients"
        return ("permutation importance: drop in training-set IC when the factor "
                "is shuffled")

    def feature_importance(self) -> pd.Series:
        if not self._is_fitted:
            raise ModelNotFittedError("Call fit() before feature_importance()")
        if self.model_type == "ridge":
            return pd.Series(
                self._model.coef_, index=self._feature_names, name="coefficient"
            ).sort_values(key=abs, ascending=False)
        if self._importance is None:
            if self._train_ref is None:
                raise ModelNotFittedError(
                    "Permutation importance needs the training data; a loaded "
                    f"{self.model_type} model has none (fit() it in this process)")
            self._importance = self._permutation_importance(*self._train_ref)
        return self._importance

    def _permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Drop in mean per-date IC when one factor's column is shuffled
        (seeded). Positive = the model relies on it."""
        base = mean_date_ic(self.predict(X), y)
        rng = np.random.default_rng(self.random_seed)
        out = {}
        for col in X.columns:
            shuffled = X.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            out[col] = base - mean_date_ic(self.predict(shuffled), y)
        return pd.Series(out, name="ic_drop").sort_values(key=abs, ascending=False)

    def save(self, path: str | Path) -> None:
        if not self._is_fitted:
            raise ModelNotFittedError("Cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path.with_suffix(".joblib"))
        meta = {
            "alpha": self.alpha,
            "random_seed": self.random_seed,
            "model_type": self.model_type,
            "purge_dates": self.purge_dates,
            "feature_names": self._feature_names,
            "cv_rmse_mean": self._cv_rmse_mean,
            "cv_rmse_std": self._cv_rmse_std,
            "training_summary": self.training_summary,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Model saved to {path.with_suffix('.joblib')}")

    @classmethod
    def load(cls, path: str | Path) -> RankingModel:
        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        instance = cls(alpha=meta["alpha"], random_seed=meta["random_seed"],
                       model_type=meta.get("model_type", "ridge"),
                       purge_dates=int(meta.get("purge_dates", 1)))
        instance._model = joblib.load(path.with_suffix(".joblib"))
        instance._feature_names = meta["feature_names"]
        instance._cv_rmse_mean = meta.get("cv_rmse_mean")
        instance._cv_rmse_std = meta.get("cv_rmse_std")
        instance.training_summary = dict(meta.get("training_summary") or {})
        instance._is_fitted = True
        logger.info(f"Model loaded from {path.with_suffix('.joblib')}")
        return instance
