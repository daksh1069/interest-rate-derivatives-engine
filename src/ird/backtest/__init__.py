"""Walk-forward delta-hedging backtest.

Modules:
    delta_hedger.py - walk-forward delta-hedge with theta/delta/gamma/vega P&L
    metrics.py      - Sharpe, hit rate, max drawdown
"""

from __future__ import annotations

from ird.backtest.delta_hedger import BacktestResult, run_delta_hedge_backtest
from ird.backtest.metrics import hit_rate, max_drawdown, sharpe

__all__ = [
    "BacktestResult",
    "hit_rate",
    "max_drawdown",
    "run_delta_hedge_backtest",
    "sharpe",
]
