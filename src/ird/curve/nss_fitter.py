"""Nelson-Siegel-Svensson curve fitting.

The NSS zero-rate function is

    r(t) = b0
         + b1 * ((1 - e^{-t/L1}) / (t/L1))
         + b2 * ((1 - e^{-t/L1}) / (t/L1) - e^{-t/L1})
         + b3 * ((1 - e^{-t/L2}) / (t/L2) - e^{-t/L2})

where b0 is the level, b1 the slope, and b2/b3 two curvature factors.

Fitting is **separable least squares**: for fixed decay parameters (L1, L2) the
model is linear in (b0..b3) and solved with ``numpy.linalg.lstsq``; the two
decays are chosen by a coarse-to-fine grid search. Pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _loadings(t: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    """Design matrix columns [level, slope, curv1, curv2] for given decays."""
    t = np.maximum(np.asarray(t, float), 1e-8)
    x1 = (1.0 - np.exp(-t / lam1)) / (t / lam1)
    c1 = x1 - np.exp(-t / lam1)
    c2 = (1.0 - np.exp(-t / lam2)) / (t / lam2) - np.exp(-t / lam2)
    return np.column_stack([np.ones_like(t), x1, c1, c2])


@dataclass
class NSSCurve:
    """A fitted Nelson-Siegel-Svensson curve."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    lam1: float
    lam2: float
    rmse_bps: float

    @property
    def betas(self) -> np.ndarray:
        return np.array([self.beta0, self.beta1, self.beta2, self.beta3])

    def zero_rate(self, T: float | np.ndarray) -> float | np.ndarray:
        t = np.atleast_1d(np.asarray(T, float))
        r = _loadings(t, self.lam1, self.lam2) @ self.betas
        return float(r[0]) if np.ndim(T) == 0 else r

    def discount_factor(self, T: float | np.ndarray) -> float | np.ndarray:
        t = np.atleast_1d(np.asarray(T, float))
        r = self.zero_rate(t)
        df = np.exp(-np.asarray(r) * t)
        return float(df[0]) if np.ndim(T) == 0 else df


def fit_nss(
    times: np.ndarray,
    zeros: np.ndarray,
    lam_grid: np.ndarray | None = None,
) -> NSSCurve:
    """Fit NSS parameters to (maturity, zero-rate) pairs.

    Args:
        times: Maturities in years (must be > 0).
        zeros: Continuously-compounded zero rates (decimals).
        lam_grid: Optional candidate decay values; defaults to a log-spaced grid.
    """
    t = np.asarray(times, float)
    y = np.asarray(zeros, float)
    if lam_grid is None:
        lam_grid = np.geomspace(0.2, 12.0, 40)

    best = None
    # L1 < L2 by convention; both drawn from the same candidate grid.
    for i, lam1 in enumerate(lam_grid):
        for lam2 in lam_grid[i + 1:]:
            X = _loadings(t, lam1, lam2)
            betas, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = X @ betas - y
            sse = float(resid @ resid)
            if best is None or sse < best[0]:
                best = (sse, betas, lam1, lam2)

    sse, betas, lam1, lam2 = best
    rmse_bps = float(np.sqrt(sse / len(t)) * 1e4)
    return NSSCurve(
        beta0=float(betas[0]),
        beta1=float(betas[1]),
        beta2=float(betas[2]),
        beta3=float(betas[3]),
        lam1=float(lam1),
        lam2=float(lam2),
        rmse_bps=rmse_bps,
    )
