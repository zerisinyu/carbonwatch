import pandas as pd
import pytest

from carbonwatch.schedule_sim import best_start, job_emissions_kg, window_means


def _day(values):
    idx = pd.date_range("2025-06-01", periods=24, freq="1h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_best_start_finds_clean_valley():
    # dirty everywhere (300) except a clean 4h valley at hours 10-13
    values = [300.0] * 24
    values[10:14] = [50.0] * 4
    assert best_start(_day(values), duration_h=4) == 10


def test_window_means_count_and_edges():
    means = window_means(_day(range(24)), duration_h=4)
    assert len(means) == 21  # starts 0..20
    assert means.loc[0] == pytest.approx(1.5)  # mean(0,1,2,3)
    assert means.loc[20] == pytest.approx(21.5)  # mean(20..23)


def test_job_emissions():
    # flat 100 g/kWh, 4h at 5kW -> 20 kWh -> 2.0 kg
    assert job_emissions_kg(_day([100.0] * 24), start_hour=0) == pytest.approx(2.0)
