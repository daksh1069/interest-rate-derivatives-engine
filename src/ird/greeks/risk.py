"""Greeks and risk metrics by bump-and-reprice.

All sensitivities use central differences. A ``price_fn`` maps a
:class:`ZeroCurve` to a present value, so the same machinery prices swaps,
bonds, and (re-anchored) Hull-White swaptions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ird.curve.zero_curve import ZeroCurve
from ird.greeks.curve_bumps import (
    key_rate_bumped_curve,
    parallel_bumped_curve,
    pillar_tenors,
)
from ird.models.hull_white import HullWhite1F

PriceFn = Callable[[ZeroCurve], float]
_BP = 1e-4


def dv01(price_fn: PriceFn, curve: ZeroCurve, h: float = _BP) -> float:
    """Dollar value of 1bp: -dV/dy for a 1bp parallel shift (central diff).

    Positive DV01 means value falls as rates rise (bond-like).
    """
    up = price_fn(parallel_bumped_curve(curve, h))
    dn = price_fn(parallel_bumped_curve(curve, -h))
    return -(up - dn) / 2.0  # change per 1bp when h == 1bp


def parallel_gamma(price_fn: PriceFn, curve: ZeroCurve, h: float = 1e-3) -> float:
    """Second-order parallel sensitivity d2V/dy2 (curve convexity, raw)."""
    up = price_fn(parallel_bumped_curve(curve, h))
    base = price_fn(curve)
    dn = price_fn(parallel_bumped_curve(curve, -h))
    return (up - 2.0 * base + dn) / h**2


def key_rate_dv01(price_fn: PriceFn, curve: ZeroCurve, h: float = _BP) -> dict[float, float]:
    """Key-rate DV01 per pillar: -dV/dy_pillar for a 1bp bump of each pillar."""
    out: dict[float, float] = {}
    for i, t in enumerate(pillar_tenors(curve)):
        up = price_fn(key_rate_bumped_curve(curve, i, h))
        dn = price_fn(key_rate_bumped_curve(curve, i, -h))
        out[float(t)] = -(up - dn) / 2.0
    return out


def bond_price(curve: ZeroCurve, maturity: float, coupon: float, freq: int = 2) -> float:
    """Price of a fixed-coupon bond (unit face) off the curve."""
    n = int(round(maturity * freq))
    tau = 1.0 / freq
    times = tau * np.arange(1, n + 1)
    dfs = np.array([float(np.atleast_1d(curve.discount_factor(t))[0]) for t in times])
    return float(coupon * tau * dfs.sum() + dfs[-1])


def bond_duration_convexity(
    curve: ZeroCurve, maturity: float, coupon: float, freq: int = 2, h: float = _BP
) -> tuple[float, float, float]:
    """(price, modified duration, convexity) of a bond via parallel curve bumps."""
    def pf(c: ZeroCurve) -> float:
        return bond_price(c, maturity, coupon, freq)

    p0 = pf(curve)
    up = pf(parallel_bumped_curve(curve, h))
    dn = pf(parallel_bumped_curve(curve, -h))
    dP = (up - dn) / (2.0 * h)
    d2P = (up - 2.0 * p0 + dn) / h**2
    mod_dur = -dP / p0
    convexity = d2P / p0
    return p0, mod_dur, convexity


def swaption_vega(
    curve: ZeroCurve,
    a: float,
    sigma: float,
    expiry: float,
    tenor: float,
    strike: float | None = None,
    freq: int = 2,
    payer: bool = True,
    dsigma: float = _BP,
) -> float:
    """Hull-White swaption vega: dPrice per 1bp change in sigma (central diff)."""
    up = HullWhite1F(a, sigma + dsigma, curve).swaption_price(
        expiry, tenor, strike, freq, payer
    )
    dn = HullWhite1F(a, sigma - dsigma, curve).swaption_price(
        expiry, tenor, strike, freq, payer
    )
    return (up - dn) / 2.0


def swaption_price_fn(
    a: float, sigma: float, expiry: float, tenor: float,
    strike: float | None = None, freq: int = 2, payer: bool = True,
) -> PriceFn:
    """Build a curve->price function for a Hull-White swaption (for DV01/KRD)."""
    def pf(curve: ZeroCurve) -> float:
        return HullWhite1F(a, sigma, curve).swaption_price(
            expiry, tenor, strike, freq, payer
        )
    return pf
