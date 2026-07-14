"""Shared plotly styling for the site: Berlin dark theme.

Categorical palette validated with the dataviz six-checks script against the
chart surface #161614 (lightness band, chroma floor, CVD ΔE, contrast).
The acid accent #c8f043 is brand chrome only — never a data series color.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# data series (validated set, fixed slot order)
LIME = "#84a010"
CYAN = "#0da2c7"
MAGENTA = "#d34f92"
ORANGE = "#c9762e"
SERIES = [LIME, CYAN, MAGENTA, ORANGE]

# chrome
ACCENT = "#c8f043"  # non-data accent only
SURFACE = "#161614"
PAGE = "#0d0d0d"
INK = "#f2f2ec"
INK2 = "#b9b8ad"
MUTED = "#8a887f"
GRID = "#2a2a27"
REF_GRAY = "#5c5a52"  # reference/baseline marks

# sequential "dirtiness" ramp: one rose hue, dark (clean, recedes) -> bright (dirty)
SEQ = ["#241016", "#3c1622", "#571d2e", "#73243b", "#902c48", "#ae3556", "#cc4364", "#e85a72"]

FONT = "Inter, 'Helvetica Neue', system-ui, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"


def template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK2),
        title=dict(font=dict(family=FONT, size=14, color=INK)),
        colorway=SERIES,
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis=dict(
            gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID,
            tickfont=dict(family=MONO, size=11, color=MUTED), automargin=True,
        ),
        yaxis=dict(
            gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID,
            tickfont=dict(family=MONO, size=11, color=MUTED), automargin=True,
        ),
        legend=dict(
            font=dict(size=12, color=INK2), bgcolor="rgba(0,0,0,0)",
            orientation="h", yanchor="bottom", y=1.02, x=0,
        ),
        hoverlabel=dict(
            bgcolor="#0d0d0d", bordercolor=GRID,
            font=dict(family=MONO, size=12, color=INK),
        ),
        hovermode="closest",
    )
    return t


pio.templates["berlin"] = template()
pio.templates.default = "berlin"

CONFIG = {"displayModeBar": False, "responsive": True}
