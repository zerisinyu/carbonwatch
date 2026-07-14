"""Descriptive analysis + spatial (DE vs UK) arm. Saves site-ready artifacts.

  data/processed/heatmap_de.parquet / heatmap_uk.parquet   hour x month mean CI
  data/processed/daily_windows.parquet   per-day best/worst 4h window stats, both countries
  data/processed/spatial_sim.parquet     per-day emissions: DE/UK naive & oracle, joint best
  data/processed/example_day.parquet     one representative DE day for the interactive

Run: uv run python notebooks/04_descriptive.py
"""

import pandas as pd

from carbonwatch.config import DATA_PROCESSED
from carbonwatch.schedule_sim import DURATION_H, POWER_KW, best_start, window_means

ci_de = pd.read_parquet(DATA_PROCESSED / "de_ci_hourly.parquet")["ci"].tz_convert("UTC")
ci_uk = pd.read_parquet(DATA_PROCESSED / "uk_ci_hourly.parquet")["ci_actual"]

# --- hour x month heatmaps ---
for name, s in [("de", ci_de), ("uk", ci_uk)]:
    heat = s.groupby([s.index.hour, s.index.month]).mean().unstack()
    heat.index.name, heat.columns.name = "hour", "month"
    heat.to_parquet(DATA_PROCESSED / f"heatmap_{name}.parquet")

# --- per-day window stats (both countries) ---
rows = []
for country, s in [("DE", ci_de), ("UK", ci_uk)]:
    for day, ci_day in s.groupby(s.index.date):
        if len(ci_day) != 24 or ci_day.isna().any():
            continue
        w = window_means(ci_day, DURATION_H)
        rows.append(
            {
                "country": country,
                "day": day,
                "best_start": int(w.idxmin()),
                "best_ci": float(w.min()),
                "worst_ci": float(w.max()),
                "mean_ci": float(ci_day.mean()),
            }
        )
daily = pd.DataFrame(rows)
daily["gap"] = daily["worst_ci"] - daily["best_ci"]
daily["saving_vs_worst"] = 1 - daily["best_ci"] / daily["worst_ci"]
daily.to_parquet(DATA_PROCESSED / "daily_windows.parquet")

for c in ("DE", "UK"):
    d = daily[daily.country == c]
    midday = d.best_start.between(9, 14).mean()
    print(
        f"{c}: median gap {d.gap.median():.0f} g/kWh, median saving vs worst {d.saving_vs_worst.median():.0%}, "
        f"clean window 9-14h on {midday:.0%} of days"
    )

# --- spatial arm: per-day emissions (kg) for the 4h/5kW job ---
energy_kwh = DURATION_H * POWER_KW
per_day = {}
for country, s in [("de", ci_de), ("uk", ci_uk)]:
    days = {}
    for day, ci_day in s.groupby(s.index.date):
        if len(ci_day) != 24 or ci_day.isna().any():
            continue
        w = window_means(ci_day, DURATION_H)
        days[day] = {
            f"{country}_naive": float(w.mean() * energy_kwh / 1000),
            f"{country}_oracle": float(w.min() * energy_kwh / 1000),
        }
    per_day[country] = pd.DataFrame.from_dict(days, orient="index")

spatial = per_day["de"].join(per_day["uk"], how="inner")
spatial = spatial[spatial.index >= pd.Timestamp("2025-01-01").date()]
spatial["joint_oracle"] = spatial[["de_oracle", "uk_oracle"]].min(axis=1)
spatial.to_parquet(DATA_PROCESSED / "spatial_sim.parquet")

t = spatial.sum()
print(f"\nspatial arm 2025 ({len(spatial)} days), kg CO2 for the daily 4h job:")
print(f"  stay in DE, random start : {t.de_naive:.0f}")
print(f"  stay in DE, perfect time : {t.de_oracle:.0f}  ({1 - t.de_oracle / t.de_naive:+.0%})")
print(f"  move to UK, random start : {t.uk_naive:.0f}  ({1 - t.uk_naive / t.de_naive:+.0%})")
print(f"  move to UK, perfect time : {t.uk_oracle:.0f}  ({1 - t.uk_oracle / t.de_naive:+.0%})")
print(f"  best country each day    : {t.joint_oracle:.0f}  ({1 - t.joint_oracle / t.de_naive:+.0%})")
print(f"  UK cleaner on {(spatial.uk_oracle < spatial.de_oracle).mean():.0%} of days")

# --- representative DE day for the drag-the-job interactive ---
d25 = daily[(daily.country == "DE") & (daily.day >= pd.Timestamp("2025-01-01").date())]
candidates = d25[(d25.gap > d25.gap.quantile(0.75)) & d25.best_start.between(10, 13)]
pick = candidates.sort_values("gap").iloc[len(candidates) // 2]
example = ci_de[ci_de.index.date == pick.day]
example.to_frame().to_parquet(DATA_PROCESSED / "example_day.parquet")
print(f"\nexample day: {pick.day} (gap {pick.gap:.0f} g/kWh, best start {pick.best_start}:00)")
