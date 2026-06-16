"""Bootstrap a zero/discount curve from par rates.

Treats the short end (maturities <= 1Y) as single-payment money-market
instruments and longer maturities as annual-coupon par swaps satisfying the
no-arbitrage condition

    S * sum_i tau_i * P(0, t_i)  =  1 - P(0, T),

solved sequentially for each pillar's discount factor. Coupon dates that fall
between bootstrapped pillars are handled by log-linear (in log-DF) interpolation
of the curve built so far, with the unknown pillar discount factor found by
bisection. Pure NumPy, no SciPy dependency.
"""

from __future__ import annotations

import numpy as np

from ird.core.conventions import tenor_to_years
from ird.core.curve_date import CurveDate
from ird.curve.zero_curve import ZeroCurve


def _coupon_schedule(maturity: float) -> tuple[np.ndarray, np.ndarray]:
    """Annual coupon times up to ``maturity`` and their year fractions."""
    n_full = int(np.floor(maturity + 1e-9))
    times = list(range(1, n_full + 1))
    if not times or abs(times[-1] - maturity) > 1e-9:
        times.append(maturity)
    times_arr = np.array(times, dtype=float)
    prev = np.concatenate([[0.0], times_arr[:-1]])
    return times_arr, times_arr - prev


def _solve_df(residual, lo: float, hi: float, tol: float = 1e-14) -> float:
    """Bisection for a monotone residual on [lo, hi]."""
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0:
        # Fall back: expand toward zero discount factor.
        lo = 1e-9
        f_lo = residual(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def bootstrap_curve(curve_date: CurveDate, method: str = "loglinear") -> ZeroCurve:
    """Bootstrap a :class:`ZeroCurve` from a :class:`CurveDate` of par rates."""
    mats, rates = curve_date.as_arrays()
    mats = np.asarray(mats, float)
    rates = np.asarray(rates, float)
    order = np.argsort(mats)
    mats, rates = mats[order], rates[order]

    times = [0.0]
    dfs = [1.0]

    for T, S in zip(mats, rates):
        if T <= 1.0 + 1e-9:
            # Money-market: single payment, simple compounding over the period.
            df_T = 1.0 / (1.0 + S * T)
        else:
            cpn_times, taus = _coupon_schedule(float(T))

            def residual(df_T: float, cpn_times=cpn_times, taus=taus, S=S, T=T) -> float:
                xs = np.array(times + [T])
                ys = np.log(np.array(dfs + [df_T]))
                cpn_df = np.exp(np.interp(cpn_times, xs, ys))
                annuity = float(np.sum(taus * cpn_df))
                return S * annuity - (1.0 - df_T)

            df_T = _solve_df(residual, lo=1e-6, hi=float(dfs[-1]))

        times.append(float(T))
        dfs.append(float(df_T))

    return ZeroCurve(
        np.array(times), np.array(dfs), method=method, date=curve_date.date
    )


def repricing_error_bps(curve: ZeroCurve, curve_date: CurveDate) -> dict[str, float]:
    """Reprice each input par instrument from the curve; return errors in bps.

    For each pillar, recompute the implied par rate from the bootstrapped curve
    and compare to the input rate. Errors should be ~0 for a correct bootstrap.
    """
    errors: dict[str, float] = {}
    for tenor, input_rate in curve_date.rates.items():
        T = tenor_to_years(tenor)
        if T <= 1.0 + 1e-9:
            df_T = curve.discount_factor(T)
            implied = (1.0 / df_T - 1.0) / T
        else:
            implied = curve.par_rate(T, freq=1)
        errors[tenor] = (implied - input_rate) * 1e4
    return errors
