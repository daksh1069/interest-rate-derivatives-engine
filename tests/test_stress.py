"""Tests for stress testing: scenarios, historical replays, VaR/CVaR."""

from __future__ import annotations

import datetime as dt

import pytest

from ird.core.curve_date import CurveDate
from ird.curve import bootstrap_curve
from ird.data.synthetic import generate_history
from ird.models import HullWhite1F
from ird.stress import (
    Portfolio,
    Position,
    historical_var,
    reverse_stress_parallel,
    run_curve_scenarios,
    run_historical_scenarios,
    run_vol_scenarios,
)

UPWARD = {
    "1M": 0.0530, "3M": 0.0525, "6M": 0.0515, "1Y": 0.0490, "2Y": 0.0450,
    "3Y": 0.0430, "5Y": 0.0415, "7Y": 0.0420, "10Y": 0.0430, "20Y": 0.0450,
    "30Y": 0.0455,
}
A, SIGMA = 0.05, 0.01


@pytest.fixture
def setup():
    curve = bootstrap_curve(CurveDate(dt.date(2023, 6, 15), dict(UPWARD)), "loglinear")
    K = HullWhite1F(A, SIGMA, curve).forward_swap_rate(5.0, 10.0, 2)
    pf = Portfolio([Position(+1, 5.0, 10.0, K, 2, False)], A, SIGMA)  # long receiver
    return pf, curve


def test_receiver_directional_signs(setup) -> None:
    pf, curve = setup
    cs = run_curve_scenarios(pf, curve)
    assert cs["parallel -100bp"] > 0  # receiver gains as rates fall
    assert cs["parallel +100bp"] < 0


def test_long_option_positive_vega(setup) -> None:
    pf, curve = setup
    assert run_vol_scenarios(pf, curve)["vol +50%"] > 0


def test_historical_replay_directions(setup) -> None:
    pf, curve = setup
    h = run_historical_scenarios(pf, curve)
    assert h["COVID crash (Mar 2020)"] > 0  # rates fell -> receiver gains
    assert h["Fed hike peak (2022)"] < 0


def test_var_cvar_ordering(setup) -> None:
    pf, curve = setup
    hist = generate_history(dt.date(2020, 1, 1), dt.date(2023, 12, 31))
    vr = historical_var(pf, curve, hist, n_samples=1000, seed=1)
    assert vr.var99 >= vr.var95 >= 0
    assert vr.cvar99 >= vr.var99
    assert vr.cvar95 >= vr.var95


def test_reverse_stress_hits_target(setup) -> None:
    pf, curve = setup
    target = pf.value(curve) * 0.10
    rs = reverse_stress_parallel(pf, curve, target_loss=target)
    assert rs["loss"] >= target - 1e-9
    assert abs(rs["shock_bp"]) > 0
