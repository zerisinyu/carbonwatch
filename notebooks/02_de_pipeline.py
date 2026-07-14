"""German carbon-intensity pipeline: build hourly CI from ENTSO-E generation,
validate against Energy-Charts (Fraunhofer ISE) reference.

Saves:
  data/processed/de_ci_hourly.parquet   our hourly DE CI 2024-2025 (+ ref column)
  data/processed/de_generation_hourly.parquet  hourly MW by fuel (for the site)

Run: uv run python notebooks/02_de_pipeline.py
"""

import pandas as pd

from carbonwatch.carbon_intensity import compute_carbon_intensity
from carbonwatch.config import DATA_PROCESSED
from carbonwatch.fetch_energy_charts import fetch_co2eq_cached
from carbonwatch.fetch_entsoe import fetch_generation

YEARS = (2024, 2025)

gen = pd.concat([fetch_generation(y) for y in YEARS]).sort_index()
gen = gen[~gen.index.duplicated()]
gen_hourly = gen.resample("1h").mean()
gen_hourly.to_parquet(DATA_PROCESSED / "de_generation_hourly.parquet")

ci = compute_carbon_intensity(gen_hourly)
ref = pd.concat([fetch_co2eq_cached(y) for y in YEARS]).resample("1h").mean()

df = ci.to_frame().join(ref.rename("ci_ref"), how="left")
df.to_parquet(DATA_PROCESSED / "de_ci_hourly.parquet")

both = df.dropna()
corr = both["ci"].corr(both["ci_ref"])
bias = (both["ci"] - both["ci_ref"]).mean()
print(f"hours: {len(both)}")
print(f"our CI:  mean {both['ci'].mean():.0f}, range {both['ci'].min():.0f}-{both['ci'].max():.0f} gCO2/kWh")
print(f"ref CI:  mean {both['ci_ref'].mean():.0f}, range {both['ci_ref'].min():.0f}-{both['ci_ref'].max():.0f}")
print(f"correlation: {corr:.3f}   mean bias: {bias:+.0f} g/kWh")
assert corr >= 0.95, "validation gate failed: correlation below 0.95"
print("validation gate PASSED")
