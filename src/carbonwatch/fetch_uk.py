"""Fetch historical carbon intensity for Great Britain from the NESO Carbon Intensity API.

Free, no API key. Docs: https://carbon-intensity.github.io/api-definitions/
The /intensity/{from}/{to} endpoint returns 30-min settlement periods and
accepts at most 14 days per request, so longer ranges are fetched in chunks.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from .config import DATA_RAW

API_BASE = "https://api.carbonintensity.org.uk"
MAX_CHUNK = timedelta(days=13)
REQUEST_PAUSE_S = 0.5


def date_chunks(start: datetime, end: datetime, step: timedelta = MAX_CHUNK) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into API-sized windows."""
    chunks = []
    cursor = start
    while cursor < end:
        chunks.append((cursor, min(cursor + step, end)))
        cursor += step
    return chunks


def _fetch_chunk(start: datetime, end: datetime) -> list[dict]:
    fmt = "%Y-%m-%dT%H:%MZ"
    url = f"{API_BASE}/intensity/{start.strftime(fmt)}/{end.strftime(fmt)}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


def fetch_uk_intensity(start: datetime, end: datetime) -> pd.DataFrame:
    """Return GB national carbon intensity (30-min resolution) for [start, end).

    Columns: ci_actual, ci_forecast (gCO2/kWh), indexed by UTC period start.
    NESO's own forecast is kept so our model can later be benchmarked against it.
    """
    records = []
    for lo, hi in date_chunks(start, end):
        for row in _fetch_chunk(lo, hi):
            records.append(
                {
                    "time": row["from"],
                    "ci_actual": row["intensity"].get("actual"),
                    "ci_forecast": row["intensity"].get("forecast"),
                }
            )
        time.sleep(REQUEST_PAUSE_S)
    df = pd.DataFrame.from_records(records)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    return df


def fetch_uk_intensity_cached(year: int) -> pd.DataFrame:
    """Fetch one calendar year, cached as parquet under data/raw/."""
    cache = DATA_RAW / f"uk_intensity_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = fetch_uk_intensity(datetime(year, 1, 1), datetime(year + 1, 1, 1))
    df.to_parquet(cache)
    return df
