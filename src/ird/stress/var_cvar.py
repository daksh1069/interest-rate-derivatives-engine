"""Portfolio VaR / CVaR by historical simulation, and reverse stress testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ird.core.conventions import tenor_to_years
from ird.curve.zero_curve import ZeroCurve
from ird.stress.scenarios import Portfolio, apply_zero_shift


@dataclass
class VarResult:
    pnl: np.ndarray
    var95: float
    var99: float
    cvar95: float
    cvar99: float

    def as_dict(self) -> dict[str, float]:
        return {"VaR95": self.var95, "VaR99": self.var99,
                "CVaR95": self.cvar95, "CVaR99": self.cvar99}


def historical_var(
    portfolio: Portfolio,
    base_curve: ZeroCurve,
    history: pd.DataFrame,
    n_samples: int = 2000,
    seed: int = 0,
) -> VarResult:
    """1-day VaR/CVaR by sampling historical daily curve changes.

    Daily per-tenor changes are sampled (with replacement) from ``history``,
    applied to the current curve, and the portfolio is repriced.
    """
    diffs = history.diff().dropna()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=min(n_samples, len(diffs) * 4))
    years = [tenor_to_years(c) for c in history.columns]
    base = portfolio.value(base_curve)

    pnl = np.empty(len(idx))
    for j, i in enumerate(idx):
        row = diffs.iloc[i]
        shift = {years[k]: float(row.iloc[k]) * 1e4 for k in range(len(years))}
        pnl[j] = portfolio.value(apply_zero_shift(base_curve, shift)) - base

    losses = -pnl
    var95 = float(np.percentile(losses, 95))
    var99 = float(np.percentile(losses, 99))
    cvar95 = float(losses[losses >= var95].mean())
    cvar99 = float(losses[losses >= var99].mean())
    return VarResult(pnl, var95, var99, cvar95, cvar99)


def reverse_stress_parallel(
    portfolio: Portfolio,
    base_curve: ZeroCurve,
    target_loss: float,
    max_bp: float = 400.0,
) -> dict[str, float]:
    """Smallest parallel shift (bp) producing a loss >= ``target_loss``.

    Scans both directions; returns the minimal-magnitude shock and its P&L.
    """
    base = portfolio.value(base_curve)
    best = None
    for bp in np.arange(1.0, max_bp + 1.0, 1.0):
        for sign in (+1.0, -1.0):
            shift = {1: sign * bp, 30: sign * bp}
            pnl = portfolio.value(apply_zero_shift(base_curve, shift)) - base
            if -pnl >= target_loss:
                if best is None or bp < best["shock_bp"]:
                    best = {"shock_bp": float(sign * bp), "loss": float(-pnl)}
                break
        if best is not None:
            break
    return best or {"shock_bp": float("nan"), "loss": 0.0}
