"""Vasicek short-rate model.

    dr(t) = kappa * (theta - r(t)) dt + sigma dW(t)

A mean-reverting Gaussian short rate with closed-form zero-coupon bond prices.
Its limitations are deliberate teaching points: it produces an endogenous (not
market-fitted) term structure and admits negative rates. Hull-White (see
``hull_white.py``) fixes the term-structure fit; this model is the calibration
warm-up.

Calibration uses the exact OU transition density, which makes the maximum-
likelihood estimator a closed-form AR(1) regression of r_{t+1} on r_t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VasicekModel:
    """Vasicek model parameters."""

    kappa: float
    theta: float
    sigma: float
    r0: float | None = None  # current short rate, optional

    def B(self, t: float, T: float) -> float:
        tau = T - t
        return (1.0 - np.exp(-self.kappa * tau)) / self.kappa

    def _lnA(self, t: float, T: float) -> float:
        tau = T - t
        b = self.B(t, T)
        return (self.theta - self.sigma**2 / (2 * self.kappa**2)) * (b - tau) - (
            self.sigma**2 / (4 * self.kappa)
        ) * b**2

    def bond_price(self, t: float, T: float, r: float) -> float:
        """Zero-coupon bond price P(t, T) given short rate ``r`` at ``t``."""
        return float(np.exp(self._lnA(t, T) - self.B(t, T) * r))

    def zero_rate(self, t: float, T: float, r: float) -> float:
        """Continuously-compounded zero rate implied by the model."""
        return -np.log(self.bond_price(t, T, r)) / (T - t)

    def long_run_rate(self) -> float:
        """Asymptotic long-maturity zero rate."""
        return self.theta - self.sigma**2 / (2 * self.kappa**2)


def calibrate_vasicek_mle(rates: np.ndarray, dt: float) -> VasicekModel:
    """Closed-form MLE of Vasicek parameters from a short-rate time series.

    Uses the exact discrete transition r_{t+1} = alpha + beta*r_t + eps with
    beta = exp(-kappa*dt). Equivalent to OU maximum likelihood.

    Args:
        rates: Observed short-rate proxy (e.g. the front-pillar rate), decimals.
        dt: Time step between observations, in years.
    """
    r = np.asarray(rates, float)
    x, y = r[:-1], r[1:]
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx, sxy = (x * x).sum(), (x * y).sum()
    denom = n * sxx - sx**2
    beta = (n * sxy - sx * sy) / denom
    alpha = (sy - beta * sx) / n
    beta = min(max(beta, 1e-6), 1 - 1e-9)  # keep in (0,1) for a valid kappa

    kappa = -np.log(beta) / dt
    theta = alpha / (1.0 - beta)
    resid = y - (alpha + beta * x)
    s2 = float(resid @ resid) / n
    sigma = np.sqrt(s2 * 2.0 * kappa / (1.0 - beta**2))
    return VasicekModel(kappa=float(kappa), theta=float(theta), sigma=float(sigma),
                        r0=float(r[-1]))


def simulate_vasicek(
    model: VasicekModel, r0: float, dt: float, n: int, seed: int = 0
) -> np.ndarray:
    """Exact-scheme simulation of a Vasicek short-rate path (for tests/demos)."""
    rng = np.random.default_rng(seed)
    b = np.exp(-model.kappa * dt)
    mean_rev = model.theta * (1.0 - b)
    std = model.sigma * np.sqrt((1.0 - b**2) / (2.0 * model.kappa))
    r = np.empty(n + 1)
    r[0] = r0
    for i in range(n):
        r[i + 1] = r[i] * b + mean_rev + std * rng.standard_normal()
    return r
