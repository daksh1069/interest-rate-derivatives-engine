"""Calibrate Hull-White (a, sigma) to an ATM swaption volatility surface.

Market swaption vols are quoted as normal (Bachelier, basis-point) vols. We
price each ATM cell under Hull-White (Jamshidian), convert the model price to a
normal vol via the ATM Bachelier formula, and least-squares fit (a, sigma) to
the surface. The two parameters are found by a coarse-to-fine grid search
(robust, derivative-free, pure NumPy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ird.core.conventions import tenor_to_years
from ird.core.vol_surface import VolSurface, VolType
from ird.curve.zero_curve import ZeroCurve
from ird.models.hull_white import HullWhite1F

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def atm_bachelier_price(annuity: float, sigma_n: float, expiry: float) -> float:
    """ATM swaption price under the normal model."""
    return annuity * sigma_n * math.sqrt(expiry) / _SQRT_2PI


def atm_normal_vol(price: float, annuity: float, expiry: float) -> float:
    """Invert the ATM Bachelier formula for the normal vol."""
    return price / annuity * _SQRT_2PI / math.sqrt(expiry)


def model_atm_normal_vol(
    hw: HullWhite1F, expiry: float, tenor: float, freq: int = 2
) -> float:
    """Hull-White ATM swaption normal (bp) vol for an (expiry, tenor) cell."""
    price = hw.swaption_price(expiry, tenor, strike=None, freq=freq, payer=True)
    annuity = hw.annuity(expiry, tenor, freq)
    return atm_normal_vol(price, annuity, expiry)


def synth_vol_surface(
    curve: ZeroCurve,
    a: float,
    sigma: float,
    expiries: list[str],
    tenors: list[str],
    freq: int = 2,
    date=None,
) -> VolSurface:
    """Build a normal-vol surface implied by a Hull-White model (for tests/demos)."""
    hw = HullWhite1F(a, sigma, curve)
    vols = [
        [
            model_atm_normal_vol(hw, tenor_to_years(e), tenor_to_years(t), freq)
            for t in tenors
        ]
        for e in expiries
    ]
    return VolSurface(
        date=date or (curve.date or __import__("datetime").date.today()),
        expiries=list(expiries),
        tenors=list(tenors),
        vols=vols,
        vol_type=VolType.NORMAL,
    )


@dataclass
class HWCalibrationResult:
    a: float
    sigma: float
    rmse_bps: float


def _surface_sse(
    hw: HullWhite1F, surface: VolSurface, freq: int
) -> tuple[float, int]:
    sse, count = 0.0, 0
    for i, e in enumerate(surface.expiries):
        for j, t in enumerate(surface.tenors):
            model = model_atm_normal_vol(
                hw, tenor_to_years(e), tenor_to_years(t), freq
            )
            sse += (model - surface.vols[i][j]) ** 2
            count += 1
    return sse, count


def calibrate_hull_white(
    curve: ZeroCurve,
    surface: VolSurface,
    freq: int = 2,
    a_bounds: tuple[float, float] = (1e-3, 1.0),
    sigma_bounds: tuple[float, float] = (1e-4, 0.05),
    n_grid: int = 12,
    n_refine: int = 5,
) -> HWCalibrationResult:
    """Fit (a, sigma) to a normal-vol surface by coarse-to-fine grid search."""
    a_lo, a_hi = a_bounds
    s_lo, s_hi = sigma_bounds
    best = None
    for _ in range(n_refine):
        a_vals = np.linspace(a_lo, a_hi, n_grid)
        s_vals = np.linspace(s_lo, s_hi, n_grid)
        for a in a_vals:
            for s in s_vals:
                hw = HullWhite1F(float(a), float(s), curve)
                sse, count = _surface_sse(hw, surface, freq)
                if best is None or sse < best[0]:
                    best = (sse, float(a), float(s), count)
        # shrink bounds around the incumbent for the next refinement pass.
        _, ba, bs, _ = best
        da = (a_hi - a_lo) / n_grid
        ds = (s_hi - s_lo) / n_grid
        a_lo, a_hi = max(a_bounds[0], ba - da), min(a_bounds[1], ba + da)
        s_lo, s_hi = max(sigma_bounds[0], bs - ds), min(sigma_bounds[1], bs + ds)

    sse, a, s, count = best
    rmse_bps = math.sqrt(sse / count) * 1e4
    return HWCalibrationResult(a=a, sigma=s, rmse_bps=rmse_bps)
