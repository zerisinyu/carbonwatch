"""Fetch Germany's CO2eq intensity from Energy-Charts (Fraunhofer ISE) — free, no key.

Used only as an independent reference series to validate our own ENTSO-E-derived
carbon intensity. Docs: https://api.energy-charts.info
"""

from __future__ import annotations

import pandas as pd
import requests

from .config import DATA_RAW

API = "https://api.energy-charts.info/co2eq"


def fetch_co2eq(start: str, end: str, country: str = "de") -> pd.Series:
    """15-min CO2eq intensity (gCO2eq/kWh) for [start, end], UTC-indexed."""
    resp = requests.get(API, params={"country": country, "start": start, "end": end}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    idx = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    return pd.Series(payload["co2eq"], index=idx, name="co2eq_ref", dtype=float)


def fetch_co2eq_cached(year: int, country: str = "de") -> pd.Series:
    cache = DATA_RAW / f"energy_charts_co2eq_{country}_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["co2eq_ref"]
    # fetch in quarters to keep responses small
    parts = []
    for q_start, q_end in [
        (f"{year}-01-01", f"{year}-03-31"),
        (f"{year}-04-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-12-31"),
    ]:
        parts.append(fetch_co2eq(q_start, q_end, country))
    s = pd.concat(parts)
    s = s[~s.index.duplicated()].sort_index()
    s.to_frame().to_parquet(cache)
    return s
