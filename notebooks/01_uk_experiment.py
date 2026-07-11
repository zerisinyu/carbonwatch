"""UK end-to-end experiment: forecast backtest + scheduling simulation.

Train on 2024, test on 2025. Saves processed outputs for the site:
  data/processed/uk_ci_hourly.parquet      hourly actual CI 2024-2025
  data/processed/uk_forecast_2025.parquet  hourly known-at-midnight forecast
  data/processed/uk_backtest_mae.parquet   MAE by horizon bucket per model
  data/processed/uk_sim_daily.parquet      per-day emissions per policy

Run: uv run python notebooks/01_uk_experiment.py
"""

import pandas as pd

from carbonwatch import forecast as fc
from carbonwatch.config import DATA_PROCESSED
from carbonwatch.features import build_direct_dataset
from carbonwatch.fetch_uk import fetch_uk_intensity_cached
from carbonwatch.schedule_sim import simulate, summarize

SPLIT = "2025-01-01"

ci = pd.concat([fetch_uk_intensity_cached(y) for y in (2024, 2025)]).ci_actual.resample("1h").mean()
ci.to_frame().to_parquet(DATA_PROCESSED / "uk_ci_hourly.parquet")

# --- forecast backtest, horizons 1-36h ---
data = build_direct_dataset(ci, horizons=range(1, 37))
train, test = data[data.index < SPLIT], data[data.index >= SPLIT]
model = fc.fit_lgbm(train)
preds = {
    "persistence": fc.predict_persistence(ci, test),
    "seasonal_naive": fc.predict_seasonal_naive(ci, test),
    "climatology": fc.predict_climatology(ci[ci.index < SPLIT], test),
    "lgbm": fc.predict_lgbm(model, test),
}
mae = fc.mae_by_horizon(test["y"], preds, test["horizon"])
mae.to_parquet(DATA_PROCESSED / "uk_backtest_mae.parquet")
print(mae.round(1), "\n")

# --- scheduling simulation on the day-ahead (1-24h) slice ---
day_ahead = test[test["horizon"] <= 24]
target_time = day_ahead.index + pd.to_timedelta(day_ahead["horizon"], unit="h")
forecast_hourly = pd.Series(
    fc.predict_lgbm(model, day_ahead).to_numpy(), index=target_time, name="ci_pred"
).sort_index()
forecast_hourly.to_frame().to_parquet(DATA_PROCESSED / "uk_forecast_2025.parquet")

sim = simulate(ci[SPLIT:], forecast_hourly)
sim.to_parquet(DATA_PROCESSED / "uk_sim_daily.parquet")
for k, v in summarize(sim).items():
    print(f"{k}: {v:.1f}")
