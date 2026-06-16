"""IRD Pricing Engine.

A modular, pure-Python fixed-income derivatives library: Treasury/SOFR
yield-curve construction, short-rate model calibration (Vasicek, Hull-White 1F),
vectorized Monte Carlo swaption pricing, Greeks, backtesting, and stress testing.

Package layout (mirrors the project plan):
    ird.core      - shared value types (CurveDate, VolSurface) and conventions
    ird.data      - Phase 1: data ingestion, storage, validation
    ird.curve     - Phase 2: bootstrapping, NSS, interpolation
    ird.models    - Phase 3: Vasicek, Hull-White, calibration
    ird.mc        - Phase 4: vectorized Monte Carlo swaption engine
    ird.greeks    - Phase 5: DV01, key-rate durations, pathwise Greeks
    ird.backtest  - Phase 6: walk-forward delta hedging, P&L attribution
    ird.stress    - Phase 7: scenario analysis, VaR/CVaR
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
