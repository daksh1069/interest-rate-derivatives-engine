"""Walk-forward delta-hedging backtest with P&L attribution.

Holds a fixed-strike swaption (ATM at inception) through historical curve moves,
delta-hedged daily with a DV01-neutral position in the underlying forward swap.
Each day's swaption P&L is decomposed into theta / delta / gamma / vega /
residual, mirroring the structure of an equity-vol hedging backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ird.core.curve_date import CurveDate
from ird.curve import bootstrap_curve
from ird.curve.zero_curve import ZeroCurve
from ird.greeks import dv01, parallel_gamma, swaption_price_fn, swaption_vega
from ird.models import HullWhite1F


def _swap_value_fn(t0: float, tenor: float, k_swap: float, freq: int, payer: bool):
    """Curve -> value of a forward-starting swap (start t0, length tenor)."""
    n = int(round(tenor * freq))
    tau = 1.0 / freq
    times = t0 + tau * np.arange(1, n + 1)

    def pf(curve: ZeroCurve) -> float:
        dfs = np.array([float(np.atleast_1d(curve.discount_factor(t))[0]) for t in times])
        annuity = tau * dfs.sum()
        v = float(np.atleast_1d(curve.discount_factor(t0))[0]) - dfs[-1] - k_swap * annuity
        return v if payer else -v

    return pf


@dataclass
class BacktestResult:
    frame: pd.DataFrame
    a: float
    sigma: float
    summary: dict = field(default_factory=dict)


def run_delta_hedge_backtest(
    history: pd.DataFrame,
    inception: pd.Timestamp,
    a: float,
    sigma: float,
    expiry: float = 1.0,
    tenor: float = 5.0,
    freq: int = 2,
    payer: bool = False,
    method: str = "loglinear",
) -> BacktestResult:
    """Run a daily delta-hedged backtest over the swaption's life.

    Args:
        history: Wide curve history (index=date, columns=tenor labels, decimals).
        inception: Start date (must be in ``history``).
        a, sigma: Hull-White parameters (held fixed over the run).
        expiry, tenor: Swaption option expiry and underlying swap tenor (years).
        payer: Payer vs receiver swaption.
    """
    end = inception + pd.Timedelta(days=int(expiry * 365))
    dates = history.index[(history.index >= inception) & (history.index <= end)]
    cols = list(history.columns)

    rows = []
    strike = None
    prev = None  # holds (curve, V, hedge_notional, par5, sigma)
    for d in dates:
        elapsed = (d - inception).days / 365.0
        tau = max(expiry - elapsed, 1e-4)
        row = history.loc[d]
        cd = CurveDate(d.date(), {c: float(row[c]) for c in cols})
        curve = bootstrap_curve(cd, method=method)
        hw = HullWhite1F(a, sigma, curve)
        if strike is None:
            strike = hw.forward_swap_rate(expiry, tenor, freq)
            k_swap = strike

        pf = swaption_price_fn(a, sigma, tau, tenor, strike, freq, payer)
        V = pf(curve)
        DV01 = dv01(pf, curve)
        GAMMA = parallel_gamma(pf, curve)
        VEGA = swaption_vega(curve, a, sigma, tau, tenor, strike, freq, payer)
        swap_pf = _swap_value_fn(tau, tenor, k_swap, freq, payer)
        swap_val = swap_pf(curve)
        swap_dv01 = dv01(swap_pf, curve)
        hedge = DV01 / swap_dv01 if swap_dv01 != 0 else 0.0
        par5 = float(np.atleast_1d(curve.zero_rate(5.0))[0])

        rec = {"date": d, "tau": tau, "V": V, "swap_val": swap_val,
               "DV01": DV01, "gamma": GAMMA, "vega": VEGA, "hedge": hedge, "par5": par5}

        if prev is not None:
            dV = V - prev["V"]
            dy = par5 - prev["par5"]               # parallel-proxy rate move
            dy_bp = dy * 1e4
            # Theta: reprice yesterday's curve with today's reduced expiry.
            hw_prev = HullWhite1F(a, sigma, prev["curve"])
            V_theta = hw_prev.swaption_price(tau, tenor, strike, freq, payer)
            theta = V_theta - prev["V"]
            delta = -prev["DV01"] * dy_bp
            gamma = 0.5 * prev["gamma"] * dy**2
            vega = prev["vega"] * (sigma - prev["sigma"]) * 1e4
            residual = dV - theta - delta - gamma - vega
            unhedged = dV
            hedged = dV - prev["hedge"] * (swap_val - prev["swap_val"])
            rec.update({"dV": dV, "theta": theta, "delta": delta, "gamma_pnl": gamma,
                        "vega_pnl": vega, "residual": residual,
                        "unhedged_pnl": unhedged, "hedged_pnl": hedged})
        rows.append(rec)
        prev = {"curve": curve, "V": V, "DV01": DV01, "gamma": GAMMA, "vega": VEGA,
                "hedge": hedge, "par5": par5, "swap_val": swap_val, "sigma": sigma}

    frame = pd.DataFrame(rows).set_index("date")
    step = frame.dropna(subset=["hedged_pnl"])
    summary = {
        "inception": str(dates[0].date()),
        "expiry_date": str(dates[-1].date()),
        "n_days": len(frame),
        "hedged_pnl_total": float(step["hedged_pnl"].sum()),
        "unhedged_pnl_total": float(step["unhedged_pnl"].sum()),
        "hedged_pnl_std": float(step["hedged_pnl"].std(ddof=1)),
        "unhedged_pnl_std": float(step["unhedged_pnl"].std(ddof=1)),
        "variance_reduction": float(step["unhedged_pnl"].std(ddof=1)
                                    / step["hedged_pnl"].std(ddof=1)),
        "mean_abs_residual_bp": float(step["residual"].abs().mean() * 1e4),
    }
    return BacktestResult(frame=frame, a=a, sigma=sigma, summary=summary)
