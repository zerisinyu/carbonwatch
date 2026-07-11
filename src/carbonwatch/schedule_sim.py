"""Counterfactual scheduling: emissions of a fixed daily job under timing policies.

The job: `duration_h` contiguous hours at `power_kw`, one run per day, free to
start at any hour such that it finishes within the same day. Policies choose
the start hour; emissions are always settled against *actual* CI.
"""

from __future__ import annotations

import pandas as pd

DURATION_H = 4
POWER_KW = 5.0  # roughly one 8xA100 node under load


def window_means(ci_day: pd.Series, duration_h: int = DURATION_H) -> pd.Series:
    """Mean CI of each contiguous window, indexed by start hour (0..24-duration)."""
    means = ci_day.rolling(duration_h).mean().shift(-(duration_h - 1))
    means.index = ci_day.index.hour
    return means.iloc[: len(ci_day) - duration_h + 1]


def best_start(ci_day: pd.Series, duration_h: int = DURATION_H) -> int:
    return int(window_means(ci_day, duration_h).idxmin())


def job_emissions_kg(
    actual_day: pd.Series,
    start_hour: int,
    duration_h: int = DURATION_H,
    power_kw: float = POWER_KW,
) -> float:
    """kg CO2 of the job settled against actual CI (gCO2/kWh * kWh / 1000)."""
    window = actual_day[(actual_day.index.hour >= start_hour) & (actual_day.index.hour < start_hour + duration_h)]
    return float(window.mean() * power_kw * duration_h / 1000.0)


def simulate(
    actual_hourly: pd.Series,
    forecast_hourly: pd.Series,
    duration_h: int = DURATION_H,
    power_kw: float = POWER_KW,
) -> pd.DataFrame:
    """Per-day emissions (kg CO2) under three policies.

    - naive: expected emissions of a uniformly random feasible start
    - forecast: cleanest window according to `forecast_hourly`
    - oracle: cleanest window with perfect knowledge of actuals

    Days with incomplete actual or forecast coverage are skipped.
    """
    rows = []
    for day, actual_day in actual_hourly.groupby(actual_hourly.index.date):
        forecast_day = forecast_hourly[forecast_hourly.index.date == day]
        if len(actual_day) != 24 or len(forecast_day) != 24 or actual_day.isna().any() or forecast_day.isna().any():
            continue
        energy = power_kw * duration_h / 1000.0
        rows.append(
            {
                "day": day,
                "naive": float(window_means(actual_day, duration_h).mean() * energy),
                "forecast": job_emissions_kg(actual_day, best_start(forecast_day, duration_h), duration_h, power_kw),
                "oracle": job_emissions_kg(actual_day, best_start(actual_day, duration_h), duration_h, power_kw),
            }
        )
    return pd.DataFrame(rows).set_index("day")


def summarize(sim: pd.DataFrame) -> dict[str, float]:
    """Headline numbers: savings vs naive, and how much oracle headroom the forecast captures."""
    totals = sim.sum()
    return {
        "days": len(sim),
        "naive_kg": totals["naive"],
        "forecast_kg": totals["forecast"],
        "oracle_kg": totals["oracle"],
        "forecast_saving_pct": 100 * (1 - totals["forecast"] / totals["naive"]),
        "oracle_saving_pct": 100 * (1 - totals["oracle"] / totals["naive"]),
        "headroom_captured_pct": 100
        * (totals["naive"] - totals["forecast"])
        / (totals["naive"] - totals["oracle"]),
    }
