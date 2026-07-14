"""Render the README hero figure: one day's carbon intensity, dirty vs clean
4h windows highlighted, Berlin dark theme. Static PNG via kaleido.

Run: uv run python notebooks/05_readme_figure.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from carbonwatch import viz
from carbonwatch.viz import ACCENT, CYAN, GRID, INK, LIME, MUTED, PAGE, SURFACE

P = Path("data/processed")
ENERGY = 20.0  # kWh: 4h x 5kW

day = pd.read_parquet(P / "example_day.parquet")["ci"].tz_convert("Europe/Berlin")
vals = day.to_numpy()
hours = list(range(24))

kg = {s: float(vals[s : s + 4].mean() * ENERGY / 1000) for s in range(21)}
best_s, worst_s = min(kg, key=kg.get), max(kg, key=kg.get)
saving = 1 - kg[best_s] / kg[worst_s]

fig = go.Figure()

fig.add_shape(
    type="rect", x0=worst_s - 0.5, x1=worst_s + 3.5, y0=0, y1=1, yref="paper",
    fillcolor="rgba(211,79,146,0.12)", line=dict(color=CYAN, width=0),
)
fig.add_shape(
    type="rect", x0=best_s - 0.5, x1=best_s + 3.5, y0=0, y1=1, yref="paper",
    fillcolor="rgba(200,240,67,0.14)", line=dict(color=ACCENT, width=1.5),
)

fig.add_annotation(
    x=worst_s + 1.5, y=vals[worst_s : worst_s + 4].max() + 25, showarrow=False,
    text=f"dirtiest hour<br><b>{kg[worst_s]:.1f} kg CO₂</b>",
    font=dict(family=viz.MONO, size=15, color="#e9a0c0"), align="center",
)
fig.add_annotation(
    x=best_s + 1.5, y=vals[best_s : best_s + 4].min() - 45, showarrow=False,
    text=f"cleanest hour<br><b>{kg[best_s]:.1f} kg CO₂</b>",
    font=dict(family=viz.MONO, size=15, color=ACCENT), align="center",
)
fig.add_annotation(
    x=12, y=vals.max() + 90, showarrow=False,
    text=f"<b>−{saving:.0%} CO₂ · same job, same day, better hour</b>",
    font=dict(family="Space Grotesk, sans-serif", size=20, color=INK),
)

fig.add_scatter(
    x=hours, y=vals, mode="lines", line=dict(color=LIME, width=3),
    fill="tozeroy", fillcolor="rgba(132,160,16,0.08)", showlegend=False,
)

fig.update_layout(
    width=1200, height=630, paper_bgcolor=PAGE, plot_bgcolor=SURFACE,
    margin=dict(l=70, r=40, t=110, b=60),
    xaxis=dict(
        tickvals=[0, 6, 12, 18, 23], ticktext=["00:00", "06:00", "12:00", "18:00", "23:00"],
        gridcolor=GRID, linecolor=GRID, tickfont=dict(family=viz.MONO, size=13, color=MUTED),
        range=[-0.5, 23.5],
    ),
    yaxis=dict(
        title=dict(text="gCO₂ / kWh — German grid, 14 Jul 2025", font=dict(family=viz.MONO, size=12, color=MUTED)),
        gridcolor=GRID, linecolor=GRID, tickfont=dict(family=viz.MONO, size=13, color=MUTED),
        range=[0, vals.max() + 140],
    ),
)

out = Path("assets")
out.mkdir(exist_ok=True)
fig.write_image(out / "hero.png", scale=2)
print(f"wrote {out / 'hero.png'}  (saving {saving:.0%}, worst_s={worst_s}, best_s={best_s})")
