"""Historical scenario replays.

Representative one-move curve shifts (basis points by maturity) reconstructed
from notable episodes, applied instantaneously to the current curve.
"""

from __future__ import annotations

from ird.curve.zero_curve import ZeroCurve
from ird.stress.scenarios import Portfolio, apply_zero_shift

# Per-tenor bp shifts approximating the curve move in each episode.
HISTORICAL_SCENARIOS: dict[str, dict[float, float]] = {
    "COVID crash (Mar 2020)": {0.25: -150, 2: -90, 5: -70, 10: -60, 30: -55},
    "Fed hike peak (2022)": {0.25: 220, 2: 280, 5: 200, 10: 150, 30: 110},
    "SVB front-end (Mar 2023)": {0.25: -120, 2: -100, 5: -55, 10: -35, 30: -25},
}


def run_historical_scenarios(
    portfolio: Portfolio, base_curve: ZeroCurve
) -> dict[str, float]:
    """P&L of the portfolio under each historical replay."""
    base = portfolio.value(base_curve)
    out: dict[str, float] = {}
    for name, spec in HISTORICAL_SCENARIOS.items():
        bumped = apply_zero_shift(base_curve, spec)
        out[name] = portfolio.value(bumped) - base
    return out
