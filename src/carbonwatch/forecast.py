"""Baselines and gradient-boosted model for hourly CI forecasting.

All predictors consume the design matrix from features.build_direct_dataset,
so every method is evaluated on exactly the same (origin, horizon) rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .features import feature_cols


def _target_times(X: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(X.index + pd.to_timedelta(X["horizon"], unit="h"))


def predict_persistence(ci: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Last value known at the origin, held flat."""
    ref = X.index - pd.Timedelta(hours=1)
    return pd.Series(ci.reindex(ref).to_numpy(), index=X.index, name="persistence")


def predict_seasonal_naive(ci: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Same hour on the most recent fully observed day (24h-periodic naive)."""
    target = _target_times(X)
    shift_h = 24 * np.ceil(X["horizon"].to_numpy() / 24)
    ref = target - pd.to_timedelta(shift_h, unit="h")
    return pd.Series(ci.reindex(ref).to_numpy(), index=X.index, name="seasonal_naive")


def predict_climatology(train_ci: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Train-period mean CI by (hour of day, month)."""
    table = train_ci.groupby([train_ci.index.hour, train_ci.index.month]).mean()
    target = _target_times(X)
    keys = pd.MultiIndex.from_arrays([target.hour, target.month])
    return pd.Series(table.reindex(keys).to_numpy(), index=X.index, name="climatology")


def fit_lgbm(train: pd.DataFrame, extra_features: list[str] | None = None) -> LGBMRegressor:
    cols = feature_cols() + (extra_features or [])
    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        verbose=-1,
    )
    model.fit(train[cols], train["y"])
    return model


def predict_lgbm(model: LGBMRegressor, X: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(X[model.feature_name_]), index=X.index, name="lgbm")


def mae_by_horizon(y: pd.Series, preds: dict[str, pd.Series], horizon: pd.Series) -> pd.DataFrame:
    """MAE per model, grouped into horizon buckets."""
    err = pd.DataFrame({name: (p - y).abs() for name, p in preds.items()})
    buckets = pd.cut(horizon, bins=[0, 6, 12, 24, 36], labels=["1-6h", "7-12h", "13-24h", "25-36h"])
    out = err.groupby(buckets, observed=True).mean()
    out.loc["overall"] = err.mean()
    return out
