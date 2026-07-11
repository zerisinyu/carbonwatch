# When Should a Computer Work? — Carbon-Aware Computing

How much CO₂ does a deferrable compute job (say, an ML training run) save by
running when the grid is cleanest — and how well can we predict those windows?

A data project on the German power grid (ENTSO-E data, UK as comparison arm),
ending in a data-journalism site.

**Status: work in progress.**

## What's here

- `src/carbonwatch/` — data pipeline: ENTSO-E (DE-LU) and NESO (GB) fetchers,
  carbon-intensity computation from generation mix, forecasting, scheduling simulation
- `notebooks/` — exploratory analysis
- `site/` — Quarto narrative site
- `tests/` — unit tests

## Reproduce

```bash
uv sync
cp .env.example .env   # add your ENTSO-E API key
uv run pytest
```

## Data sources

- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — German generation by type, load, day-ahead forecasts
- [NESO Carbon Intensity API](https://carbonintensity.org.uk/) — GB carbon intensity + official forecast
- [ElectricityMaps data portal](https://www.electricitymaps.com/data-portal) & [Energy-Charts](https://energy-charts.info) — validation
