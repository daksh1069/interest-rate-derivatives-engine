"""Tests for short-rate models: Vasicek, Hull-White, and calibration."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from ird.core.curve_date import CurveDate
from ird.curve import bootstrap_curve
from ird.models import (
    HullWhite1F,
    VasicekModel,
    calibrate_hull_white,
    calibrate_vasicek_mle,
    simulate_vasicek,
    synth_vol_surface,
)

UPWARD = {
    "1M": 0.0530, "3M": 0.0525, "6M": 0.0515, "1Y": 0.0490, "2Y": 0.0450,
    "3Y": 0.0430, "5Y": 0.0415, "7Y": 0.0420, "10Y": 0.0430, "20Y": 0.0450,
    "30Y": 0.0455,
}


@pytest.fixture
def curve():
    return bootstrap_curve(CurveDate(dt.date(2023, 6, 15), dict(UPWARD)), "loglinear")


# --- Vasicek -------------------------------------------------------------- #
def test_vasicek_bond_price_bounds_and_limit() -> None:
    m = VasicekModel(kappa=0.3, theta=0.04, sigma=0.01)
    p = m.bond_price(0.0, 5.0, r=0.03)
    assert 0.0 < p < 1.0
    # Bond price -> 1 as maturity -> t.
    assert m.bond_price(0.0, 1e-6, r=0.03) == pytest.approx(1.0, abs=1e-4)


def test_vasicek_mle_recovers_params() -> None:
    true = VasicekModel(kappa=0.5, theta=0.04, sigma=0.012)
    path = simulate_vasicek(true, r0=0.03, dt=1 / 252, n=60_000, seed=1)
    est = calibrate_vasicek_mle(path, dt=1 / 252)
    assert est.kappa == pytest.approx(true.kappa, rel=0.25)
    assert est.theta == pytest.approx(true.theta, abs=0.01)
    assert est.sigma == pytest.approx(true.sigma, rel=0.10)


# --- Hull-White ----------------------------------------------------------- #
def test_hw_reprices_initial_curve(curve) -> None:
    hw = HullWhite1F(a=0.05, sigma=0.01, curve=curve)
    r0 = hw.r0
    for T in (1.0, 2.0, 5.0, 10.0, 30.0):
        assert hw.bond_price(0.0, T, r0) == pytest.approx(curve.discount_factor(T), rel=1e-6)


def test_hw_atm_payer_equals_receiver(curve) -> None:
    hw = HullWhite1F(a=0.05, sigma=0.01, curve=curve)
    payer = hw.swaption_price(2.0, 5.0, strike=None, payer=True)
    receiver = hw.swaption_price(2.0, 5.0, strike=None, payer=False)
    assert payer == pytest.approx(receiver, rel=1e-6)  # ATM put-call parity
    assert payer > 0


def test_hw_swaption_increases_with_sigma(curve) -> None:
    lo = HullWhite1F(0.05, 0.005, curve).swaption_price(2.0, 5.0)
    hi = HullWhite1F(0.05, 0.015, curve).swaption_price(2.0, 5.0)
    assert hi > lo > 0


def test_hw_forward_swap_rate_reasonable(curve) -> None:
    hw = HullWhite1F(0.05, 0.01, curve)
    fwd = hw.forward_swap_rate(2.0, 5.0, freq=2)
    assert 0.02 < fwd < 0.07  # plausible for this curve


# --- Calibration ---------------------------------------------------------- #
def test_calibration_recovers_known_params(curve) -> None:
    expiries = ["1Y", "2Y", "5Y", "7Y"]
    tenors = ["2Y", "5Y", "10Y"]
    a_true, sig_true = 0.08, 0.012
    surface = synth_vol_surface(curve, a_true, sig_true, expiries, tenors, freq=2)
    res = calibrate_hull_white(curve, surface, freq=2)
    assert res.a == pytest.approx(a_true, abs=0.02)
    assert res.sigma == pytest.approx(sig_true, abs=0.001)
    assert res.rmse_bps < 0.5
