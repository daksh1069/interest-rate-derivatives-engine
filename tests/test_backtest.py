"""Tests for the delta-hedging backtest and performance metrics."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ird.backtest import (
    hit_rate,
    max_drawdown,
    run_delta_hedge_backtest,
    sharpe,
)
from ird.data.synthetic import generate_history


def test_sharpe_constant_positive() -> None:
    pnl = np.full(252, 0.01)
    assert sharpe(pnl) == 0.0  # zero variance -> defined as 0


def test_sharpe_sign() -> None:
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.001, 0.01, 1000)
    assert sharpe(pnl) > 0


def test_max_drawdown_known() -> None:
    cum = np.array([0.0, 1.0, 3.0, 2.0, 5.0, 1.0])
    assert max_drawdown(cum) == pytest.approx(4.0)  # peak 5 -> trough 1


def test_hit_rate() -> None:
    assert hit_rate(np.array([1.0, -1.0, 2.0, -3.0])) == pytest.approx(0.5)


@pytest.fixture(scope="module")
def history():
    return generate_history(dt.date(2018, 1, 1), dt.date(2024, 12, 31))


def test_backtest_delta_hedge_reduces_variance(history) -> None:
    res = run_delta_hedge_backtest(history, pd.Timestamp("2022-01-03"),
                                   a=0.05, sigma=0.01, expiry=1.0, tenor=5.0,
                                   payer=False)
    assert res.summary["variance_reduction"] > 1.5
    assert res.summary["n_days"] > 200


def test_backtest_attribution_identity(history) -> None:
    res = run_delta_hedge_backtest(history, pd.Timestamp("2022-01-03"),
                                   a=0.05, sigma=0.01)
    f = res.frame.dropna(subset=["dV"])
    recon = f["theta"] + f["delta"] + f["gamma_pnl"] + f["vega_pnl"] + f["residual"]
    assert (recon - f["dV"]).abs().max() < 1e-12
