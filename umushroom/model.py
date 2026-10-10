"""Model layer.

Wraps a Ridge regression (linear, regularized, interpretable — see
the design note below) behind a small class that:
  - tracks whether it has been fit, raising a typed error if predict()
    is called first
  - exposes feature importances in the training's own units, for
    defensibility ("why did the model pick this stock")
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
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from umushroom.exceptions import ModelNotFittedError
from umushroom.logging_config import get_logger

logger = get_logger("model")


class RankingModel:
    def __init__(self, alpha: float = 1.0, random_seed: int = 42,
                 n_cv_splits: int = 5):
        self.alpha = alpha
        self.random_seed = random_seed
        self.n_cv_splits = n_cv_splits
        self._model = Ridge(alpha=alpha, random_state=random_seed)
        self._is_fitted = False
        self._feature_names: list[str] | None = None
        self._cv_rmse_mean: float | None = None
        self._cv_rmse_std: float | None = None
        self._cv_ic_mean: float | None = None
        self._cv_ic_tstat: float | None = None

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
        if isinstance(X.index, pd.MultiIndex):
            row_dates = X.index.get_level_values(0)
        else:
            row_dates = pd.Index(np.arange(len(X)))
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
            fold_model = Ridge(alpha=self.alpha, random_state=self.random_seed)
            fold_model.fit(X[tr_mask], y[tr_mask])
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

        self._model.fit(X, y)
        self._is_fitted = True
        self._feature_names = list(X.columns)
        logger.info(f"Model fit on {len(X)} samples, {X.shape[1]} features")
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

    def feature_importance(self) -> pd.Series:
        if not self._is_fitted:
            raise ModelNotFittedError("Call fit() before feature_importance()")
        return pd.Series(
            self._model.coef_, index=self._feature_names, name="coefficient"
        ).sort_values(key=abs, ascending=False)

    def save(self, path: str | Path) -> None:
        if not self._is_fitted:
            raise ModelNotFittedError("Cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path.with_suffix(".joblib"))
        meta = {
            "alpha": self.alpha,
            "random_seed": self.random_seed,
            "feature_names": self._feature_names,
            "cv_rmse_mean": self._cv_rmse_mean,
            "cv_rmse_std": self._cv_rmse_std,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Model saved to {path.with_suffix('.joblib')}")

    @classmethod
    def load(cls, path: str | Path) -> RankingModel:
        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        instance = cls(alpha=meta["alpha"], random_seed=meta["random_seed"])
        instance._model = joblib.load(path.with_suffix(".joblib"))
        instance._feature_names = meta["feature_names"]
        instance._cv_rmse_mean = meta.get("cv_rmse_mean")
        instance._cv_rmse_std = meta.get("cv_rmse_std")
        instance._is_fitted = True
        logger.info(f"Model loaded from {path.with_suffix('.joblib')}")
        return instance
