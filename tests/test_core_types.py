"""Tests for CurveDate and VolSurface value types."""

from __future__ import annotations

import datetime as dt

import pytest

from ird.core.curve_date import CurveDate
from ird.core.vol_surface import VolSurface, VolType


def test_curve_date_sorts_tenors_by_maturity() -> None:
    cd = CurveDate(dt.date(2022, 6, 15), {"10Y": 0.034, "3M": 0.012, "2Y": 0.028})
    assert cd.tenors == ["3M", "2Y", "10Y"]
    mats, rates = cd.as_arrays()
    assert mats == [0.25, 2.0, 10.0]
    assert rates == [0.012, 0.028, 0.034]


def test_curve_date_rejects_empty() -> None:
    with pytest.raises(ValueError):
        CurveDate(dt.date(2022, 6, 15), {})


def test_curve_date_rejects_nan() -> None:
    with pytest.raises(ValueError):
        CurveDate(dt.date(2022, 6, 15), {"5Y": float("nan")})


def test_curve_date_rate_lookup() -> None:
    cd = CurveDate(dt.date(2022, 6, 15), {"5Y": 0.03})
    assert cd.rate("5Y") == 0.03
    with pytest.raises(KeyError):
        cd.rate("7Y")


def test_vol_surface_get_and_shapes() -> None:
    vs = VolSurface(
        date=dt.date(2022, 6, 15),
        expiries=["1Y", "2Y"],
        tenors=["5Y", "10Y"],
        vols=[[0.0085, 0.0090], [0.0080, 0.0088]],
        vol_type=VolType.NORMAL,
    )
    assert vs.get("2Y", "10Y") == pytest.approx(0.0088)
    assert vs.expiry_years() == [1.0, 2.0]
    assert vs.tenor_years() == [5.0, 10.0]


def test_vol_surface_validates_dimensions() -> None:
    with pytest.raises(ValueError):
        VolSurface(
            date=dt.date(2022, 6, 15),
            expiries=["1Y", "2Y"],
            tenors=["5Y"],
            vols=[[0.0085]],  # only one row, but two expiries
        )
