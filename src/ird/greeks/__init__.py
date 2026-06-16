"""Greeks and risk metrics.

Modules:
    curve_bumps.py - parallel and key-rate zero-curve bump helpers
    risk.py        - DV01, key-rate DV01, duration, convexity, vega
"""

from __future__ import annotations

from ird.greeks.curve_bumps import (
    key_rate_bumped_curve,
    parallel_bumped_curve,
    pillar_tenors,
)
from ird.greeks.risk import (
    bond_duration_convexity,
    bond_price,
    dv01,
    key_rate_dv01,
    parallel_gamma,
    swaption_price_fn,
    swaption_vega,
)

__all__ = [
    "bond_duration_convexity",
    "bond_price",
    "dv01",
    "key_rate_bumped_curve",
    "key_rate_dv01",
    "parallel_bumped_curve",
    "parallel_gamma",
    "pillar_tenors",
    "swaption_price_fn",
    "swaption_vega",
]
