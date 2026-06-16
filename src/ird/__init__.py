"""IRD Pricing Engine.

A modular, pure-Python fixed-income derivatives library: Treasury/SOFR
yield-curve construction, short-rate model calibration (Vasicek, Hull-White 1F),
vectorized Monte Carlo swaption pricing, Greeks, backtesting, and stress testing.

Package layout:
    ird.core      - shared value types (CurveDate, VolSurface) and conventions
    ird.data      - data ingestion, storage, validation
    ird.curve     - bootstrapping, NSS, interpolation
    ird.models    - Vasicek, Hull-White, calibration
    ird.mc        - vectorized Monte Carlo swaption engine
    ird.greeks    - DV01, key-rate durations, pathwise Greeks
    ird.backtest  - walk-forward delta hedging, P&L attribution
    ird.stress    - scenario analysis, VaR/CVaR
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
