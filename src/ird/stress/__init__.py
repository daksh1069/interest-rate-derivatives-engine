"""Stress testing.

Modules:
    scenarios.py            - portfolio, parallel/shape scenarios, vol shocks
    historical_scenarios.py - 2020 COVID, 2022 hikes, 2023 SVB replays
    var_cvar.py             - portfolio VaR/CVaR + reverse stress test
"""

from __future__ import annotations

from ird.stress.historical_scenarios import (
    HISTORICAL_SCENARIOS,
    run_historical_scenarios,
)
from ird.stress.scenarios import (
    Portfolio,
    Position,
    apply_zero_shift,
    run_curve_scenarios,
    run_vol_scenarios,
    standard_scenarios,
)
from ird.stress.var_cvar import VarResult, historical_var, reverse_stress_parallel

__all__ = [
    "HISTORICAL_SCENARIOS",
    "Portfolio",
    "Position",
    "VarResult",
    "apply_zero_shift",
    "historical_var",
    "reverse_stress_parallel",
    "run_curve_scenarios",
    "run_historical_scenarios",
    "run_vol_scenarios",
    "standard_scenarios",
]
