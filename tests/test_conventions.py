"""Tests for tenor parsing and day-count conventions."""

from __future__ import annotations

import datetime as dt

import pytest

from ird.core.conventions import DayCount, tenor_to_years, year_fraction


@pytest.mark.parametrize(
    ("tenor", "expected"),
    [("1Y", 1.0), ("10Y", 10.0), ("3M", 0.25), ("6M", 0.5), ("2W", 14 / 365)],
)
def test_tenor_to_years(tenor: str, expected: float) -> None:
    assert tenor_to_years(tenor) == pytest.approx(expected)


def test_tenor_to_years_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        tenor_to_years("banana")
    with pytest.raises(ValueError):
        tenor_to_years("5Q")


def test_act_360() -> None:
    yf = year_fraction(dt.date(2022, 1, 1), dt.date(2022, 1, 31), DayCount.ACT_360)
    assert yf == pytest.approx(30 / 360)


def test_act_365f() -> None:
    yf = year_fraction(dt.date(2022, 1, 1), dt.date(2023, 1, 1), DayCount.ACT_365F)
    assert yf == pytest.approx(365 / 365)


def test_thirty_360() -> None:
    yf = year_fraction(dt.date(2022, 1, 15), dt.date(2022, 7, 15), DayCount.THIRTY_360)
    assert yf == pytest.approx(0.5)


def test_year_fraction_rejects_reversed_dates() -> None:
    with pytest.raises(ValueError):
        year_fraction(dt.date(2022, 2, 1), dt.date(2022, 1, 1))
