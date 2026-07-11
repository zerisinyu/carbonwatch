import pandas as pd
import pytest

from carbonwatch.carbon_intensity import compute_carbon_intensity


def test_weighted_average():
    # 50/50 gas (490) and wind onshore (11) -> mean of the two factors
    gen = pd.DataFrame({"Fossil Gas": [100.0], "Wind Onshore": [100.0]})
    ci = compute_carbon_intensity(gen)
    assert ci.iloc[0] == pytest.approx((490 + 11) / 2)


def test_negative_generation_excluded():
    # Pumped storage drawing from the grid must not enter the mix
    gen = pd.DataFrame({"Fossil Gas": [100.0], "Hydro Pumped Storage": [-50.0]})
    ci = compute_carbon_intensity(gen)
    assert ci.iloc[0] == pytest.approx(490.0)


def test_unknown_type_raises():
    gen = pd.DataFrame({"Fusion": [100.0]})
    with pytest.raises(KeyError):
        compute_carbon_intensity(gen)


def test_nan_handled():
    gen = pd.DataFrame({"Fossil Gas": [100.0], "Solar": [float("nan")]})
    ci = compute_carbon_intensity(gen)
    assert ci.iloc[0] == pytest.approx(490.0)
