"""Feature engineering for direct multi-horizon carbon-intensity forecasting.

One training row per (forecast origin, horizon): all features are values known
at the origin, the target is CI at origin + horizon hours. A single model
learns all horizons jointly, with the horizon itself as a feature.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 48, 168)
FEATURE_COLS_BASE = ["horizon", "target_hour", "target_dow", "target_month"]


def feature_cols(lags: Iterable[int] = DEFAULT_LAGS) -> list[str]:
    return FEATURE_COLS_BASE + [f"lag_{lag}" for lag in lags]


def build_direct_dataset(
    ci: pd.Series,
    horizons: Iterable[int] = range(1, 37),
    lags: Iterable[int] = DEFAULT_LAGS,
    origin_hour: int = 0,
    exog: pd.DataFrame | None = None,
    exog_max_horizon: int = 24,
) -> pd.DataFrame:
    """Build the (origin, horizon) design matrix from an hourly CI series.

    Index: forecast origin (repeated across horizons). Column `y` is the target.
    Rows whose target lies beyond the series end are dropped; missing lags at
    the start of the series stay NaN (LightGBM handles them natively).

    `exog`: hourly day-ahead published values (e.g. ENTSO-E wind/solar/load
    forecasts) indexed by target time. They are only attached for horizons up
    to `exog_max_horizon`: beyond the day ahead those publications would not
    yet exist at the origin, and joining them would leak future information.
    """
    ci = ci.asfreq("1h")
    origins = ci.index[ci.index.hour == origin_hour]
    frames = []
    for h in horizons:
        target_time = origins + pd.Timedelta(hours=h)
        df = pd.DataFrame(index=origins)
        df["horizon"] = h
        df["y"] = ci.reindex(target_time).to_numpy()
        df["target_hour"] = target_time.hour
        df["target_dow"] = target_time.dayofweek
        df["target_month"] = target_time.month
        for lag in lags:
            df[f"lag_{lag}"] = ci.reindex(origins - pd.Timedelta(hours=lag)).to_numpy()
        if exog is not None:
            for col in exog.columns:
                values = exog[col].reindex(target_time).to_numpy()
                # LightGBM sanitizes feature names, so avoid spaces from the start
                df[f"exog_{col.replace(' ', '_')}"] = values if h <= exog_max_horizon else float("nan")
        frames.append(df)
    return pd.concat(frames).sort_index().dropna(subset=["y"])
