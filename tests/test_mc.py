"""Tests for the Monte Carlo swaption engine."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from ird.core.curve_date import CurveDate
from ird.curve import bootstrap_curve
from ird.mc import (
    price_bermudan_swaption_mc,
    price_european_swaption_mc,
    simulate_short_rate,
)
from ird.models import HullWhite1F

UPWARD = {
    "1M": 0.0530, "3M": 0.0525, "6M": 0.0515, "1Y": 0.0490, "2Y": 0.0450,
    "3Y": 0.0430, "5Y": 0.0415, "7Y": 0.0420, "10Y": 0.0430, "20Y": 0.0450,
    "30Y": 0.0455,
}


@pytest.fixture
def hw():
    curve = bootstrap_curve(CurveDate(dt.date(2023, 6, 15), dict(UPWARD)), "loglinear")
    return HullWhite1F(a=0.05, sigma=0.01, curve=curve)


def test_european_mc_matches_analytic(hw) -> None:
    analytic = hw.swaption_price(2.0, 5.0, strike=None, freq=2, payer=True)
    mc = price_european_swaption_mc(hw, 2.0, 5.0, n_paths=200_000, seed=1)
    assert abs(mc.price - analytic) < 5 * mc.std_error


def test_control_variate_reduces_variance(hw) -> None:
    with_cv = price_european_swaption_mc(hw, 2.0, 5.0, n_paths=100_000,
                                         control_variate=True, seed=2)
    no_cv = price_european_swaption_mc(hw, 2.0, 5.0, n_paths=100_000,
                                       control_variate=False, seed=2)
    assert with_cv.std_error < no_cv.std_error


def test_qmc_converges(hw) -> None:
    analytic = hw.swaption_price(2.0, 5.0, strike=None, freq=2, payer=True)
    mc = price_european_swaption_mc(hw, 2.0, 5.0, n_paths=16_384, method="qmc")
    assert abs(mc.price - analytic) < 5e-4  # within 5 vol-bp at modest N


def test_discounting_is_martingale(hw) -> None:
    grid = np.linspace(0.0, 10.0, 121)
    _, bank, grid = simulate_short_rate(hw, grid, n_paths=60_000, seed=3)
    for T in (1.0, 5.0, 10.0):
        k = int(np.argmin(np.abs(grid - T)))
        mc_df = float((1.0 / bank[:, k]).mean())
        assert mc_df == pytest.approx(hw.curve.discount_factor(T), rel=0.012)


def test_bermudan_at_least_european(hw) -> None:
    analytic = hw.swaption_price(2.0, 5.0, strike=None, freq=2, payer=True)
    berm = price_bermudan_swaption_mc(hw, 2.0, 5.0, n_paths=40_000, seed=5)
    assert berm.price >= analytic - 5 * berm.std_error
    assert berm.price > 0


def test_payer_receiver_atm_parity_mc(hw) -> None:
    payer = price_european_swaption_mc(hw, 2.0, 5.0, payer=True, n_paths=80_000, seed=7)
    receiver = price_european_swaption_mc(hw, 2.0, 5.0, payer=False, n_paths=80_000, seed=7)
    # ATM payer == receiver up to MC error.
    assert abs(payer.price - receiver.price) < 5 * (payer.std_error + receiver.std_error)
