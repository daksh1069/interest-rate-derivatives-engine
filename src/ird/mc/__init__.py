"""Vectorized Monte Carlo swaption engine (pure NumPy).

Prices European swaptions under the T-forward measure (validated against the
Jamshidian analytics in :mod:`ird.models.hull_white`) and Bermudan swaptions via
Longstaff-Schwartz least-squares regression on simulated Hull-White paths, with
antithetic and control variates and optional quasi-Monte Carlo.
"""

from __future__ import annotations

from ird.mc.engine import (
    McResult,
    price_bermudan_swaption_mc,
    price_european_swaption_mc,
    simulate_short_rate,
)

__all__ = [
    "McResult",
    "price_bermudan_swaption_mc",
    "price_european_swaption_mc",
    "simulate_short_rate",
]
