"""Tests for yield-curve construction (bootstrap, interpolation, NSS)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from ird.core.curve_date import CurveDate
from ird.curve import bootstrap_curve, fit_nss, repricing_error_bps
from ird.curve.interpolation import METHODS
from ird.curve.zero_curve import ZeroCurve

# A realistic upward-sloping curve and an inverted one.
UPWARD = {
    "1M": 0.0530, "3M": 0.0525, "6M": 0.0515, "1Y": 0.0490, "2Y": 0.0450,
    "3Y": 0.0430, "5Y": 0.0415, "7Y": 0.0420, "10Y": 0.0430, "20Y": 0.0450,
    "30Y": 0.0455,
}
INVERTED = {
    "1M": 0.0540, "3M": 0.0545, "6M": 0.0535, "1Y": 0.0500, "2Y": 0.0470,
    "3Y": 0.0440, "5Y": 0.0405, "7Y": 0.0390, "10Y": 0.0380, "20Y": 0.0375,
    "30Y": 0.0372,
}


def _cd(rates: dict[str, float]) -> CurveDate:
    return CurveDate(dt.date(2023, 6, 15), dict(rates))


@pytest.mark.parametrize("rates", [UPWARD, INVERTED])
def test_bootstrap_reprices_inputs(rates: dict[str, float]) -> None:
    curve = bootstrap_curve(_cd(rates), method="loglinear")
    errs = repricing_error_bps(curve, _cd(rates))
    assert max(abs(e) for e in errs.values()) < 0.1  # sub-0.1bp


@pytest.mark.parametrize("method", list(METHODS))
def test_interpolation_reprices_pillars(method: str) -> None:
    curve = bootstrap_curve(_cd(UPWARD), method=method)
    for t, df in zip(curve.pillars, curve.dfs[1:]):
        assert curve.discount_factor(float(t)) == pytest.approx(df, abs=1e-9)


@pytest.mark.parametrize("method", list(METHODS))
def test_discount_factors_monotone_and_forwards_positive(method: str) -> None:
    curve = bootstrap_curve(_cd(UPWARD), method=method)
    grid = np.linspace(0.05, 30.0, 400)
    dfs = curve.discount_factor(grid)
    assert np.all(np.diff(dfs) < 0)  # strictly decreasing for positive rates
    fwd = curve.instantaneous_forward(grid)
    assert np.all(fwd > -1e-6)  # non-negative forwards


def test_zero_and_forward_consistency() -> None:
    curve = bootstrap_curve(_cd(UPWARD), method="loglinear")
    # z(T)*T == integral of forwards: check df relation directly.
    T1, T2 = 2.0, 5.0
    f = curve.forward_rate(T1, T2)
    df1, df2 = curve.discount_factor(T1), curve.discount_factor(T2)
    assert np.exp(-f * (T2 - T1)) == pytest.approx(df2 / df1, rel=1e-9)


def test_par_rate_roundtrip() -> None:
    curve = bootstrap_curve(_cd(UPWARD), method="loglinear")
    assert curve.par_rate(5.0, freq=1) == pytest.approx(UPWARD["5Y"], abs=1e-5)
    assert curve.par_rate(10.0, freq=1) == pytest.approx(UPWARD["10Y"], abs=1e-5)


def test_nss_recovers_known_params() -> None:
    from ird.curve.nss_fitter import _loadings

    t = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    true = np.array([0.045, -0.015, 0.02, 0.01])
    y = _loadings(t, 1.5, 6.0) @ true
    fit = fit_nss(t, y)
    assert fit.rmse_bps < 1.0
    assert np.allclose(fit.zero_rate(t), y, atol=2e-4)


def test_nss_fits_bootstrapped_curve() -> None:
    curve = bootstrap_curve(_cd(UPWARD), method="loglinear")
    t = curve.pillars
    z = np.atleast_1d(curve.zero_rate(t))
    fit = fit_nss(t, z)
    assert fit.rmse_bps < 8.0  # parametric fit close to the bootstrapped zeros


def test_zero_curve_rejects_non_monotone_times() -> None:
    with pytest.raises(ValueError):
        ZeroCurve(np.array([1.0, 1.0, 2.0]), np.array([0.95, 0.94, 0.90]))
