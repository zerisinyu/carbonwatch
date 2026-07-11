"""Compute grid carbon intensity from generation by production type.

Emission factors are lifecycle values in gCO2eq/kWh, following the convention
ElectricityMaps uses (IPCC AR5 2014 medians for the major technologies), so
that our derived series is directly comparable to theirs during validation.

Known simplifications, stated rather than hidden:
- Lignite uses the IPCC coal median (820); plant-level lignite is typically
  higher, so German CI is likely slightly underestimated in coal-heavy hours.
- Pumped-storage discharge is treated as hydro (24) instead of carrying the
  carbon of the electricity that charged it.
- Production-based intensity: imports/exports are not traced.
"""

from __future__ import annotations

import pandas as pd

# Keyed by ENTSO-E "Actual Generation per Production Type" column names
# (entsoe-py returns human-readable PSR type names).
EMISSION_FACTORS: dict[str, float] = {
    "Biomass": 230.0,
    "Fossil Brown coal/Lignite": 820.0,
    "Fossil Coal-derived gas": 820.0,
    "Fossil Gas": 490.0,
    "Fossil Hard coal": 820.0,
    "Fossil Oil": 650.0,
    "Fossil Oil shale": 820.0,
    "Fossil Peat": 820.0,
    "Geothermal": 38.0,
    "Hydro Pumped Storage": 24.0,
    "Hydro Run-of-river and poundage": 24.0,
    "Hydro Water Reservoir": 24.0,
    "Marine": 24.0,
    "Nuclear": 12.0,
    "Other renewable": 230.0,
    "Solar": 45.0,
    "Waste": 700.0,
    "Wind Offshore": 12.0,
    "Wind Onshore": 11.0,
    "Other": 700.0,
}


def compute_carbon_intensity(generation: pd.DataFrame) -> pd.Series:
    """Generation-weighted average carbon intensity in gCO2eq/kWh.

    `generation` is a wide frame of MW per production type. Negative values
    (pumped-storage consumption) are excluded from the mix. Columns without a
    known factor raise, so a new ENTSO-E category can't silently pass as zero.
    """
    unknown = [c for c in generation.columns if c not in EMISSION_FACTORS]
    if unknown:
        raise KeyError(f"No emission factor for production type(s): {unknown}")

    gen = generation.clip(lower=0).fillna(0.0)
    factors = pd.Series({c: EMISSION_FACTORS[c] for c in gen.columns})
    emissions = gen.mul(factors, axis=1).sum(axis=1)  # MW * g/kWh == g-scale numerator
    total = gen.sum(axis=1)
    return (emissions / total).rename("ci")
