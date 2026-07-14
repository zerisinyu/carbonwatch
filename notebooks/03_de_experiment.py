"""German end-to-end experiment: CI forecast backtest + scheduling simulation.

Mirrors the UK experiment (01) but with ENTSO-E day-ahead wind/solar/load
forecasts as exogenous features. Train 2024, test 2025.

Saves:
  data/processed/de_backtest_mae.parquet
  data/processed/de_forecast_2025.parquet
  data/processed/de_sim_daily.parquet

Run after 02_de_pipeline.py:  uv run python notebooks/03_de_experiment.py
"""

import pandas as pd

from carbonwatch import forecast as fc
from carbonwatch.config import DATA_PROCESSED
from carbonwatch.features import build_direct_dataset, feature_cols
from carbonwatch.fetch_entsoe import fetch_load_forecast, fetch_wind_solar_forecast
from carbonwatch.schedule_sim import simulate, summarize

SPLIT = "2025-01-01"
YEARS = (2024, 2025)

ci = pd.read_parquet(DATA_PROCESSED / "de_ci_hourly.parquet")["ci"].tz_convert("UTC")

ws = pd.concat([fetch_wind_solar_forecast(y) for y in YEARS]).sort_index()
lf = pd.concat([fetch_load_forecast(y) for y in YEARS]).sort_index()
exog = ws.join(lf, how="outer")
exog = exog[~exog.index.duplicated()].tz_convert("UTC").resample("1h").mean()
print("exogenous features:", list(exog.columns))

data = build_direct_dataset(ci, horizons=range(1, 37), exog=exog)
train, test = data[data.index < SPLIT], data[data.index >= SPLIT]
print(f"train rows {len(train)}, test rows {len(test)}")

exog_cols = [c for c in data.columns if c.startswith("exog_")]
model = fc.fit_lgbm(train, extra_features=exog_cols)
model_noexog = fc.fit_lgbm(train)

preds = {
    "persistence": fc.predict_persistence(ci, test),
    "seasonal_naive": fc.predict_seasonal_naive(ci, test),
    "climatology": fc.predict_climatology(ci[ci.index < SPLIT], test),
    "lgbm": fc.predict_lgbm(model_noexog, test),
    "lgbm_exog": fc.predict_lgbm(model, test),
}
mae = fc.mae_by_horizon(test["y"], preds, test["horizon"])
mae.to_parquet(DATA_PROCESSED / "de_backtest_mae.parquet")
print("\nMAE (gCO2/kWh) by horizon bucket:")
print(mae.round(1))
skill = 1 - mae.loc["overall", "lgbm_exog"] / mae.loc["overall", "seasonal_naive"]
print(f"\nLGBM+exog skill vs seasonal-naive: {skill:.1%}")

# --- scheduling simulation on the day-ahead slice ---
day_ahead = test[test["horizon"] <= 24]
target_time = day_ahead.index + pd.to_timedelta(day_ahead["horizon"], unit="h")
forecast_hourly = pd.Series(
    fc.predict_lgbm(model, day_ahead).to_numpy(), index=target_time, name="ci_pred"
).sort_index()
forecast_hourly.to_frame().to_parquet(DATA_PROCESSED / "de_forecast_2025.parquet")

sim = simulate(ci[SPLIT:], forecast_hourly)
sim.to_parquet(DATA_PROCESSED / "de_sim_daily.parquet")
print()
for k, v in summarize(sim).items():
    print(f"{k}: {v:.1f}")
