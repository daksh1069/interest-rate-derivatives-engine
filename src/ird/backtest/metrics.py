"""Performance metrics for a P&L series."""

from __future__ import annotations

import numpy as np


def sharpe(pnl: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio of a per-period P&L series (zero risk-free)."""
    pnl = np.asarray(pnl, float)
    sd = pnl.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(pnl.mean() / sd * np.sqrt(periods_per_year))


def max_drawdown(cumulative: np.ndarray) -> float:
    """Largest peak-to-trough drop of a cumulative P&L series (>= 0)."""
    c = np.asarray(cumulative, float)
    peak = np.maximum.accumulate(c)
    return float(np.max(peak - c))


def hit_rate(pnl: np.ndarray) -> float:
    """Fraction of periods with non-negative P&L."""
    pnl = np.asarray(pnl, float)
    return float(np.mean(pnl >= 0))
