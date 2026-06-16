"""Tests for Greeks and risk metrics."""

from __future__ import annotations

import datetime as dt

import pytest

from ird.core.curve_date import CurveDate
from ird.curve import bootstrap_curve
from ird.greeks import (
    bond_duration_convexity,
    bond_price,
    dv01,
    key_rate_dv01,
    swaption_price_fn,
    swaption_vega,
)
from ird.models import HullWhite1F

UPWARD = {
    "1M": 0.0530, "3M": 0.0525, "6M": 0.0515, "1Y": 0.0490, "2Y": 0.0450,
    "3Y": 0.0430, "5Y": 0.0415, "7Y": 0.0420, "10Y": 0.0430, "20Y": 0.0450,
    "30Y": 0.0455,
}
A, SIGMA = 0.05, 0.01


@pytest.fixture
def curve():
    return bootstrap_curve(CurveDate(dt.date(2023, 6, 15), dict(UPWARD)), "loglinear")


@pytest.mark.parametrize("T", [2.0, 5.0, 10.0])
def test_zero_coupon_duration_convexity(curve, T) -> None:
    # Continuously-compounded zero: modified duration == T, convexity == T^2.
    _, dur, conv = bond_duration_convexity(curve, T, coupon=0.0, freq=1)
    assert dur == pytest.approx(T, abs=1e-2)
    assert conv == pytest.approx(T * T, rel=1e-2)


def test_bond_price_zero_equals_discount_factor(curve) -> None:
    assert bond_price(curve, 5.0, coupon=0.0, freq=1) == pytest.approx(
        curve.discount_factor(5.0), rel=1e-9
    )


def test_krd_sum_equals_parallel_dv01(curve) -> None:
    K = HullWhite1F(A, SIGMA, curve).forward_swap_rate(2.0, 5.0, 2)
    pf = swaption_price_fn(A, SIGMA, 2.0, 5.0, K, 2, True)
    total = dv01(pf, curve)
    krd_sum = sum(key_rate_dv01(pf, curve).values())
    assert krd_sum == pytest.approx(total, rel=0.02)


def test_krd_locality(curve) -> None:
    K = HullWhite1F(A, SIGMA, curve).forward_swap_rate(2.0, 5.0, 2)
    pf = swaption_price_fn(A, SIGMA, 2.0, 5.0, K, 2, True)
    krd = key_rate_dv01(pf, curve)
    total = abs(dv01(pf, curve))
    # A 2Y-into-5Y swaption (spans ~2y..7y) has ~no 30Y sensitivity.
    assert abs(krd[30.0]) < 0.02 * total


def test_dv01_signs(curve) -> None:
    K = HullWhite1F(A, SIGMA, curve).forward_swap_rate(2.0, 5.0, 2)
    payer = swaption_price_fn(A, SIGMA, 2.0, 5.0, K, 2, True)
    receiver = swaption_price_fn(A, SIGMA, 2.0, 5.0, K, 2, False)
    assert dv01(payer, curve) < 0 < dv01(receiver, curve)


def test_vega_positive(curve) -> None:
    assert swaption_vega(curve, A, SIGMA, 2.0, 5.0) > 0
