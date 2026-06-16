"""Scenario stress testing: portfolio valuation under curve and vol shocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ird.curve.zero_curve import ZeroCurve
from ird.models.hull_white import HullWhite1F


@dataclass
class Position:
    """A single Hull-White swaption position (unit notional x sign)."""

    sign: float
    expiry: float
    tenor: float
    strike: float
    freq: int = 2
    payer: bool = True


class Portfolio:
    """A book of Hull-White swaptions valued off a curve and a vol level."""

    def __init__(self, positions: list[Position], a: float, sigma: float) -> None:
        self.positions = positions
        self.a = a
        self.sigma = sigma

    def value(self, curve: ZeroCurve, sigma: float | None = None) -> float:
        sig = self.sigma if sigma is None else sigma
        hw = HullWhite1F(self.a, sig, curve)
        return float(sum(
            p.sign * hw.swaption_price(p.expiry, p.tenor, p.strike, p.freq, p.payer)
            for p in self.positions
        ))


def apply_zero_shift(curve: ZeroCurve, shift_by_tenor: dict[float, float]) -> ZeroCurve:
    """Shift the zero curve by a per-tenor amount (basis points), interpolated.

    ``shift_by_tenor`` maps maturity (years) -> shift in bp; the shift is
    linearly interpolated (flat-extrapolated) onto the curve pillars.
    """
    t = curve.times[1:]
    z = -np.log(curve.dfs[1:]) / t
    spec_t = np.array(sorted(shift_by_tenor))
    spec_v = np.array([shift_by_tenor[k] / 1e4 for k in sorted(shift_by_tenor)])
    dz = np.interp(t, spec_t, spec_v)
    dfs = np.exp(-(z + dz) * t)
    return ZeroCurve(t, dfs, method=curve.method, date=curve.date)


# Standard curve-shape scenarios as per-tenor bp shifts (anchors interpolated).
def standard_scenarios() -> dict[str, dict[float, float]]:
    return {
        "parallel +100bp": {1: 100, 30: 100},
        "parallel +200bp": {1: 200, 30: 200},
        "parallel +300bp": {1: 300, 30: 300},
        "parallel -100bp": {1: -100, 30: -100},
        "parallel -200bp": {1: -200, 30: -200},
        "parallel -300bp": {1: -300, 30: -300},
        "steepener +100bp": {2: 0, 10: 100},
        "flattener -100bp": {2: 0, 10: -100},
        "bear flattener": {2: 100, 10: 50},
        "bull steepener": {2: -100, 10: -25},
        "butterfly": {2: 0, 5: 50, 10: 0},
    }


def run_curve_scenarios(
    portfolio: Portfolio, base_curve: ZeroCurve
) -> dict[str, float]:
    """P&L (value change) for each standard curve scenario."""
    base = portfolio.value(base_curve)
    out: dict[str, float] = {}
    for name, spec in standard_scenarios().items():
        bumped = apply_zero_shift(base_curve, spec)
        out[name] = portfolio.value(bumped) - base
    return out


def run_vol_scenarios(
    portfolio: Portfolio, base_curve: ZeroCurve
) -> dict[str, float]:
    """P&L for relative ATM-vol shocks (sigma scaled)."""
    base = portfolio.value(base_curve)
    shocks = {"vol +20%": 1.2, "vol +50%": 1.5, "vol -20%": 0.8}
    return {
        name: portfolio.value(base_curve, sigma=portfolio.sigma * mult) - base
        for name, mult in shocks.items()
    }
