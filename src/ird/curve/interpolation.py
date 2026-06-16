"""Discount-factor interpolation schemes.

Each builder takes pillar maturities ``times`` (years, ascending, including the
anchor ``t=0`` with discount factor ``1.0``) and pillar discount factors
``dfs``, and returns a vectorized callable ``df(T)`` valid for ``T >= 0``.

Three schemes, matching the standard "compare these" set:

* ``log_linear_df``     - linear in log discount factor (piecewise-flat forwards)
* ``cubic_spline_zero`` - natural cubic spline on continuously-compounded zeros
* ``monotone_convex``   - Hagan-West monotone-convex forwards (positive forwards)

All three reprice the input pillars exactly: ``df(t_i) == dfs[i]``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DfFunc = Callable[[np.ndarray], np.ndarray]


def _as_array(T: float | np.ndarray) -> np.ndarray:
    return np.atleast_1d(np.asarray(T, dtype=float))


def _zero_rates(times: np.ndarray, dfs: np.ndarray) -> np.ndarray:
    """Continuously-compounded zero rates at pillars (z_0 set equal to z_1)."""
    z = np.empty_like(times)
    z[1:] = -np.log(dfs[1:]) / times[1:]
    z[0] = z[1] if len(times) > 1 else 0.0
    return z


# --------------------------------------------------------------------------- #
# Log-linear on discount factors
# --------------------------------------------------------------------------- #
def log_linear_df(times: np.ndarray, dfs: np.ndarray) -> DfFunc:
    times = np.asarray(times, float)
    log_df = np.log(dfs)

    def df(T: float | np.ndarray) -> np.ndarray:
        t = _as_array(T)
        # np.interp clamps outside the range; beyond t_max we extend the last
        # log-df slope (flat instantaneous forward) for a sensible extrapolation.
        out = np.interp(t, times, log_df)
        if len(times) >= 2:
            slope = (log_df[-1] - log_df[-2]) / (times[-1] - times[-2])
            beyond = t > times[-1]
            out[beyond] = log_df[-1] + slope * (t[beyond] - times[-1])
        return np.exp(out)

    return df


# --------------------------------------------------------------------------- #
# Natural cubic spline on zero rates
# --------------------------------------------------------------------------- #
def _natural_cubic_coeffs(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Second derivatives for a natural cubic spline (tridiagonal solve)."""
    n = len(xs)
    if n < 3:
        return np.zeros(n)
    h = np.diff(xs)
    rhs = np.zeros(n)
    rhs[1:-1] = 6.0 * (
        (ys[2:] - ys[1:-1]) / h[1:] - (ys[1:-1] - ys[:-2]) / h[:-1]
    )
    diag = np.ones(n)
    lower = np.zeros(n)
    upper = np.zeros(n)
    diag[1:-1] = 2.0 * (h[:-1] + h[1:])
    upper[1:-1] = h[1:]
    lower[1:-1] = h[:-1]
    # Thomas algorithm.
    for i in range(1, n):
        w = lower[i] / diag[i - 1]
        diag[i] -= w * upper[i - 1]
        rhs[i] -= w * rhs[i - 1]
    m = np.zeros(n)
    m[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        m[i] = (rhs[i] - upper[i] * m[i + 1]) / diag[i]
    return m


def cubic_spline_zero(times: np.ndarray, dfs: np.ndarray) -> DfFunc:
    times = np.asarray(times, float)
    z = _zero_rates(times, dfs)
    m = _natural_cubic_coeffs(times, z)

    def zero(T: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(times, T) - 1, 0, len(times) - 2)
        x0, x1 = times[idx], times[idx + 1]
        h = x1 - x0
        a = (x1 - T) / h
        b = (T - x0) / h
        return (
            a * z[idx]
            + b * z[idx + 1]
            + ((a**3 - a) * m[idx] + (b**3 - b) * m[idx + 1]) * (h**2) / 6.0
        )

    def df(T: float | np.ndarray) -> np.ndarray:
        t = _as_array(T)
        tc = np.clip(t, times[0], times[-1])
        zt = zero(tc)
        # Flat-zero extrapolation beyond the last pillar.
        zt = np.where(t > times[-1], zero(np.full_like(t, times[-1])), zt)
        out = np.exp(-zt * t)
        out = np.where(t <= 0, 1.0, out)
        return out

    return df


# --------------------------------------------------------------------------- #
# Hagan-West monotone convex
# --------------------------------------------------------------------------- #
def _hw_region(g0: float, g1: float) -> int:
    if g0 == 0.0 and g1 == 0.0:
        return 0
    if (g0 < 0 and -0.5 * g0 <= g1 <= -2.0 * g0) or (
        g0 > 0 and -0.5 * g0 >= g1 >= -2.0 * g0
    ):
        return 1
    if (g0 < 0 and g1 > -2.0 * g0) or (g0 > 0 and g1 < -2.0 * g0):
        return 2
    if (g0 > 0 and 0 > g1 > -0.5 * g0) or (g0 < 0 and 0 < g1 < -0.5 * g0):
        return 3
    return 4


def _hw_integral(region: int, g0: float, g1: float, x: float) -> float:
    """G(x) = integral_0^x g(u) du for the Hagan-West interval function."""
    if region == 0:
        return 0.0
    if region == 1:
        return g0 * (x - 2 * x**2 + x**3) + g1 * (-x**2 + x**3)
    if region == 2:
        eta = (g1 + 2 * g0) / (g1 - g0)
        if x <= eta:
            return g0 * x
        return g0 * x + (g1 - g0) / (3 * (1 - eta) ** 2) * (x - eta) ** 3
    if region == 3:
        eta = 3 * g1 / (g1 - g0)
        if x <= eta:
            return g1 * x + (g0 - g1) / (3 * eta**2) * (eta**3 - (eta - x) ** 3)
        g_eta = g1 * eta + (g0 - g1) * eta / 3
        return g_eta + g1 * (x - eta)
    # region 4
    eta = g1 / (g1 + g0)
    A = -g0 * g1 / (g0 + g1)
    if x <= eta:
        return A * x + (g0 - A) / (3 * eta**2) * (eta**3 - (eta - x) ** 3)
    g_eta = A * eta + (g0 - A) * eta / 3
    return g_eta + A * (x - eta) + (g1 - A) / (3 * (1 - eta) ** 2) * (x - eta) ** 3


def monotone_convex(times: np.ndarray, dfs: np.ndarray) -> DfFunc:
    times = np.asarray(times, float)
    n = len(times) - 1
    rt = -np.log(dfs)  # cumulative R_i * t_i, with rt[0] = 0
    h = np.diff(times)
    fd = np.diff(rt) / h  # discrete forwards per interval, length n

    # Instantaneous node forwards with positivity collar.
    f = np.zeros(n + 1)
    for i in range(1, n):
        f[i] = (h[i - 1] * fd[i] + h[i] * fd[i - 1]) / (times[i + 1] - times[i - 1])
        hi = 2.0 * min(fd[i - 1], fd[i])
        if hi >= 0.0:
            f[i] = min(max(f[i], 0.0), hi)
    f[0] = fd[0] - 0.5 * (f[1] - fd[0]) if n >= 2 else fd[0]
    f[0] = min(max(f[0], 0.0), 2.0 * fd[0]) if fd[0] >= 0 else f[0]
    f[n] = fd[n - 1] - 0.5 * (f[n - 1] - fd[n - 1]) if n >= 2 else fd[n - 1]
    if fd[n - 1] >= 0:
        f[n] = min(max(f[n], 0.0), 2.0 * fd[n - 1])

    g0 = f[:-1] - fd
    g1 = f[1:] - fd
    regions = [_hw_region(g0[i], g1[i]) for i in range(n)]

    def df(T: float | np.ndarray) -> np.ndarray:
        t = _as_array(T)
        out = np.empty_like(t)
        for k, tk in enumerate(t):
            if tk <= 0:
                out[k] = 1.0
                continue
            if tk >= times[-1]:
                # flat instantaneous forward beyond the last node
                rt_t = rt[-1] + f[n] * (tk - times[-1])
                out[k] = np.exp(-rt_t)
                continue
            i = int(np.searchsorted(times, tk) - 1)
            i = max(0, min(i, n - 1))
            x = (tk - times[i]) / h[i]
            G = _hw_integral(regions[i], g0[i], g1[i], x)
            rt_t = rt[i] + h[i] * (fd[i] * x + G)
            out[k] = np.exp(-rt_t)
        return out

    return df


METHODS: dict[str, Callable[[np.ndarray, np.ndarray], DfFunc]] = {
    "loglinear": log_linear_df,
    "cubic": cubic_spline_zero,
    "monotone_convex": monotone_convex,
}
