"""Zero-curve bump helpers for bump-and-reprice risk.

Bumps are applied in continuously-compounded zero-rate space and the curve is
rebuilt with the same interpolation method, so the model/pricer re-anchors to
the perturbed curve cleanly.
"""

from __future__ import annotations

import numpy as np

from ird.curve.zero_curve import ZeroCurve


def _pillar_zero_rates(curve: ZeroCurve) -> tuple[np.ndarray, np.ndarray]:
    """Pillar maturities (>0) and their continuously-compounded zero rates."""
    t = curve.times[1:]
    z = -np.log(curve.dfs[1:]) / t
    return t, z


def parallel_bumped_curve(curve: ZeroCurve, dz: float) -> ZeroCurve:
    """Return a copy with every pillar zero rate shifted by ``dz``."""
    t, z = _pillar_zero_rates(curve)
    dfs = np.exp(-(z + dz) * t)
    return ZeroCurve(t, dfs, method=curve.method, date=curve.date)


def key_rate_bumped_curve(curve: ZeroCurve, pillar_index: int, dz: float) -> ZeroCurve:
    """Return a copy with only pillar ``pillar_index`` shifted by ``dz``."""
    t, z = _pillar_zero_rates(curve)
    z = z.copy()
    z[pillar_index] += dz
    dfs = np.exp(-z * t)
    return ZeroCurve(t, dfs, method=curve.method, date=curve.date)


def pillar_tenors(curve: ZeroCurve) -> np.ndarray:
    """Pillar maturities (years, excluding the t=0 anchor)."""
    return curve.times[1:]
