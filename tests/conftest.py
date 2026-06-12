"""Shared pytest fixtures."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from ird.config import Settings
from ird.data.synthetic import generate_history


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings pointing at an isolated temp data dir, no FRED key."""
    s = Settings(data_dir=tmp_path / "data", fred_api_key=None)
    s.ensure_dirs()
    return s


@pytest.fixture
def small_history():
    """A short synthetic history for fast tests."""
    return generate_history(
        start=dt.date(2022, 1, 3), end=dt.date(2022, 3, 31), seed=7
    )
