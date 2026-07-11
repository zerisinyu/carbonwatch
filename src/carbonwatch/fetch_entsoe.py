"""Fetch German (DE-LU bidding zone) data from the ENTSO-E Transparency Platform.

Requires ENTSOE_API_KEY in the environment or a .env file at the repo root
(register at transparency.entsoe.eu, then email transparency@entsoe.eu with
subject "Restful API access" to have the Web API enabled on the account).

Everything is fetched per calendar year and cached as parquet so long pulls
are resumable and re-runs are free.
"""

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from .config import DATA_RAW, PROJECT_ROOT

ZONE = "DE_LU"


def _client() -> EntsoePandasClient:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError("ENTSOE_API_KEY not set (put it in .env at the repo root)")
    return EntsoePandasClient(api_key=key)


def _cached_year(kind: str, year: int, fetch) -> pd.DataFrame:
    cache = DATA_RAW / f"entsoe_{ZONE}_{kind}_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Berlin")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Berlin")
    df = fetch(start, end)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.to_parquet(cache)
    return df


def fetch_generation(year: int) -> pd.DataFrame:
    """Actual generation per production type, 15-min MW."""
    client = _client()

    def _get(start, end):
        df = client.query_generation(ZONE, start=start, end=end, psr_type=None)
        # entsoe-py returns MultiIndex columns (type, Actual Aggregated/Consumption);
        # keep aggregated output only.
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs("Actual Aggregated", axis=1, level=1)
        return df

    return _cached_year("generation", year, _get)


def fetch_load(year: int) -> pd.DataFrame:
    """Actual total load, 15-min MW."""
    client = _client()
    return _cached_year("load", year, lambda s, e: client.query_load(ZONE, start=s, end=e))


def fetch_wind_solar_forecast(year: int) -> pd.DataFrame:
    """Day-ahead wind and solar generation forecasts (forecast features later)."""
    client = _client()
    return _cached_year(
        "wind_solar_forecast",
        year,
        lambda s, e: client.query_wind_and_solar_forecast(ZONE, start=s, end=e),
    )


def fetch_load_forecast(year: int) -> pd.DataFrame:
    """Day-ahead total load forecast."""
    client = _client()
    return _cached_year(
        "load_forecast",
        year,
        lambda s, e: client.query_load_forecast(ZONE, start=s, end=e),
    )
