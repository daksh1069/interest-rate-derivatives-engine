"""Monte Carlo swaption pricing under Hull-White (vectorized NumPy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ird.mc._qmc import standard_normals_1d, standard_normals_2d
from ird.models.hull_white import HullWhite1F


@dataclass
class McResult:
    """A Monte Carlo price estimate."""

    price: float
    std_error: float
    n_paths: int
    method: str = "pseudo"
    label: str = ""

    def __repr__(self) -> str:
        return (
            f"McResult({self.label or 'price'}={self.price:.6f} "
            f"+/- {self.std_error:.6f}, n={self.n_paths}, {self.method})"
        )


def _fixed_schedule(expiry: float, tenor: float, freq: int):
    n = int(round(tenor * freq))
    tau = 1.0 / freq
    times = expiry + tau * np.arange(1, n + 1)
    return times, np.full(n, tau)


def _bond_matrix(hw: HullWhite1F, t: float, times: np.ndarray, r: np.ndarray) -> np.ndarray:
    """P(t, times_j) for each path r_i -> array (n_paths, n_times)."""
    A = np.array([hw.A(t, T) for T in times])
    B = np.array([hw.B(t, T) for T in times])
    return A[None, :] * np.exp(-np.outer(r, B))


def price_european_swaption_mc(
    hw: HullWhite1F,
    expiry: float,
    tenor: float,
    strike: float | None = None,
    freq: int = 2,
    payer: bool = True,
    n_paths: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    method: str = "pseudo",
    seed: int = 0,
) -> McResult:
    """European swaption price by Monte Carlo under the T-forward measure.

    Exact in the Hull-White model up to MC error, so it converges to the
    Jamshidian analytic price.
    """
    times, taus = _fixed_schedule(expiry, tenor, freq)
    if strike is None:
        strike = hw.forward_swap_rate(expiry, tenor, freq)

    std_x = np.sqrt(hw.x_variance(expiry))
    mean_x = hw.forward_measure_x_mean(expiry)
    z = standard_normals_1d(n_paths, method=method, antithetic=antithetic, seed=seed)
    r = mean_x + std_x * z + hw.alpha(expiry)

    P = _bond_matrix(hw, expiry, times, r)
    annuity_paths = P @ taus
    v_payer = 1.0 - P[:, -1] - strike * annuity_paths
    v = v_payer if payer else -v_payer
    payoff = np.maximum(v, 0.0)

    p0 = hw.P0(expiry)
    if control_variate:
        # The swap value is a control with known forward-measure mean.
        annuity0 = float(np.sum(taus * np.array([hw.P0(T) for T in times])))
        v_swap_today = hw.P0(expiry) - hw.P0(times[-1]) - strike * annuity0
        ctrl = v_payer if payer else -v_payer
        mean_ctrl = v_swap_today / p0 * (1.0 if payer else -1.0)
        cov = np.cov(payoff, ctrl)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        payoff = payoff - beta * (ctrl - mean_ctrl)

    disc = p0 * payoff
    price = float(disc.mean())
    stderr = float(disc.std(ddof=1) / np.sqrt(n_paths))
    return McResult(price, stderr, n_paths, method, label="european")


def simulate_short_rate(
    hw: HullWhite1F,
    grid: np.ndarray,
    n_paths: int,
    antithetic: bool = True,
    seed: int = 0,
):
    """Simulate Hull-White short-rate paths on ``grid`` (must start at 0).

    Returns ``(r, bank, grid)`` where ``r`` and ``bank`` are ``(n_paths, len)``;
    ``bank[:, k] = exp(int_0^t_k r ds)`` (trapezoidal), so ``1/bank`` is the
    stochastic discount factor to time 0.
    """
    grid = np.asarray(grid, float)
    m = len(grid) - 1
    z = standard_normals_2d(n_paths, m, antithetic=antithetic, seed=seed)
    x = np.zeros((n_paths, m + 1))
    a, sig = hw.a, hw.sigma
    for k in range(m):
        dt = grid[k + 1] - grid[k]
        e = np.exp(-a * dt)
        sd = sig * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))
        x[:, k + 1] = x[:, k] * e + sd * z[:, k]
    alpha = np.array([hw.alpha(t) for t in grid])
    r = x + alpha[None, :]
    ln_bank = np.zeros((n_paths, m + 1))
    for k in range(m):
        dt = grid[k + 1] - grid[k]
        ln_bank[:, k + 1] = ln_bank[:, k] + 0.5 * (r[:, k] + r[:, k + 1]) * dt
    return r, np.exp(ln_bank), grid


def price_bermudan_swaption_mc(
    hw: HullWhite1F,
    expiry: float,
    tenor: float,
    exercise_dates: list[float] | None = None,
    strike: float | None = None,
    freq: int = 2,
    payer: bool = True,
    n_paths: int = 50_000,
    step: float = 1.0 / 12.0,
    antithetic: bool = True,
    seed: int = 0,
) -> McResult:
    """Bermudan (co-terminal) swaption via Longstaff-Schwartz.

    Exercisable at ``exercise_dates`` (default: annually from ``expiry`` to one
    year before swap maturity), entering a swap that runs to ``expiry + tenor``.
    """
    maturity = expiry + tenor
    if exercise_dates is None:
        n_ex = max(1, int(round(tenor)))
        exercise_dates = [expiry + i for i in range(n_ex)]
    exercise_dates = sorted(d for d in exercise_dates if d < maturity - 1e-9)
    full_times, full_taus = _fixed_schedule(expiry, tenor, freq)
    if strike is None:
        strike = hw.forward_swap_rate(expiry, tenor, freq)

    # Fine simulation grid that contains all exercise dates and t=0.
    base = np.arange(0.0, exercise_dates[-1] + step / 2, step)
    grid = np.unique(np.concatenate([base, [0.0], exercise_dates]))
    r, bank, grid = simulate_short_rate(hw, grid, n_paths, antithetic, seed)
    idx = {d: int(np.argmin(np.abs(grid - d))) for d in exercise_dates}

    def swap_value(d: float, r_d: np.ndarray) -> np.ndarray:
        rem = full_times[full_times > d + 1e-9]
        rem_tau = full_taus[full_times > d + 1e-9]
        P = _bond_matrix(hw, d, rem, r_d)
        v_payer = 1.0 - P[:, -1] - strike * (P @ rem_tau)
        return v_payer if payer else -v_payer

    cashflow_pv = np.zeros(n_paths)  # PV (to time 0) of the chosen strategy
    for d in reversed(exercise_dates):
        k = idx[d]
        ev = np.maximum(swap_value(d, r[:, k]), 0.0)  # immediate value at d
        ev_pv = ev / bank[:, k]
        itm = ev > 1e-12
        if d == exercise_dates[-1]:
            cashflow_pv = np.where(itm, ev_pv, cashflow_pv)
            continue
        if itm.sum() >= 8:
            xk = r[itm, k]
            basis = np.vander(xk, 3)  # [x^2, x, 1]
            coef, *_ = np.linalg.lstsq(basis, cashflow_pv[itm], rcond=None)
            cont_hat = basis @ coef
            do_ex = ev_pv[itm] > cont_hat
            sel = np.where(itm)[0][do_ex]
            cashflow_pv[sel] = ev_pv[sel]
    price = float(cashflow_pv.mean())
    stderr = float(cashflow_pv.std(ddof=1) / np.sqrt(n_paths))
    return McResult(price, stderr, n_paths, "pseudo", label="bermudan")
