"""Tests for the data validation and cleaning layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ird.data.validation import clean_history, validate_history


def test_clean_synthetic_is_ok(small_history: pd.DataFrame) -> None:
    cleaned = clean_history(small_history)
    report = validate_history(cleaned)
    assert report.nan_cells == 0
    assert not report.missing_business_days
    assert report.ok


def test_detects_large_jump() -> None:
    idx = pd.bdate_range("2022-01-03", periods=4)
    df = pd.DataFrame(
        {"2Y": [0.01, 0.01, 0.05, 0.05], "5Y": [0.02, 0.02, 0.02, 0.02]}, index=idx
    )
    report = validate_history(df)
    assert any(t == "2Y" for _, t, _ in report.large_jumps)
    assert not report.ok


def test_detects_missing_business_day() -> None:
    idx = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-06"])  # 01-05 missing
    df = pd.DataFrame({"2Y": [0.01, 0.011, 0.012]}, index=idx)
    report = validate_history(df)
    assert len(report.missing_business_days) == 1


def test_detects_nan_and_cleaning_fills() -> None:
    idx = pd.bdate_range("2022-01-03", periods=3)
    df = pd.DataFrame({"2Y": [0.01, np.nan, 0.012]}, index=idx)
    assert validate_history(df).nan_cells == 1
    assert validate_history(clean_history(df)).nan_cells == 0


def test_detects_inversion() -> None:
    idx = pd.bdate_range("2022-01-03", periods=1)
    df = pd.DataFrame({"2Y": [0.05], "10Y": [0.03]}, index=idx)  # inverted
    report = validate_history(df)
    assert len(report.inverted_dates) == 1
