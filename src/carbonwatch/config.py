"""Paths and shared configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

for _d in (DATA_RAW, DATA_PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)
