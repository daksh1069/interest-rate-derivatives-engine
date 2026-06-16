"""Hull-White 1-factor short-rate model, fitted to an initial curve.

    dr(t) = [theta(t) - a * r(t)] dt + sigma dW(t)

theta(t) is implied by the observed discount curve, so the model reprices the
initial term structure exactly. We use the Brigo-Mercurio reconstruction:

    P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
    B(t,T) = (1 - e^{-a (T-t)}) / a
    A(t,T) = P(0,T)/P(0,t) * exp( B(t,T) f(0,t)
                                  - sigma^2/(4a) (1 - e^{-2 a t}) B(t,T)^2 )

European swaptions are priced by Jamshidian decomposition into a portfolio of
zero-coupon bond options, each with the Brigo-Mercurio closed form.
"""

from __future__ import annotations

import math

import numpy as np

from ird.curve.zero_curve import ZeroCurve

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT2)


class HullWhite1F:
    """Hull-White 1-factor model anchored to a :class:`ZeroCurve`."""

    def __init__(self, a: float, sigma: float, curve: ZeroCurve) -> None:
        if a <= 0:
            raise ValueError("mean reversion a must be positive")
        self.a = float(a)
        self.sigma = float(sigma)
        self.curve = curve

    # --- curve helpers ---------------------------------------------------- #
    def P0(self, T: float) -> float:
        return float(np.atleast_1d(self.curve.discount_factor(T))[0])

    def _f0(self, t: float) -> float:
        return float(np.atleast_1d(self.curve.instantaneous_forward(t))[0])

    def B(self, t: float, T: float) -> float:
        return (1.0 - math.exp(-self.a * (T - t))) / self.a

    def A(self, t: float, T: float) -> float:
        b = self.B(t, T)
        term = (
            b * self._f0(t)
            - (self.sigma**2 / (4.0 * self.a)) * (1.0 - math.exp(-2.0 * self.a * t)) * b**2
        )
        return (self.P0(T) / self.P0(t)) * math.exp(term)

    def bond_price(self, t: float, T: float, r: float) -> float:
        """P(t, T) given the short rate ``r`` at time ``t``."""
        return self.A(t, T) * math.exp(-self.B(t, T) * r)

    @property
    def r0(self) -> float:
        """Short rate consistent with the curve at t=0 (the instant forward)."""
        return self._f0(0.0)

    # --- zero-coupon bond options ---------------------------------------- #
    def zbo(self, expiry: float, bond_mat: float, strike: float, call: bool) -> float:
        """Price a European option on a zero-coupon bond P(.,bond_mat).

        Option expires at ``expiry`` with strike ``strike`` (Brigo-Mercurio).
        """
        sig_p = self.sigma * math.sqrt(
            (1.0 - math.exp(-2.0 * self.a * expiry)) / (2.0 * self.a)
        ) * self.B(expiry, bond_mat)
        pT, pS = self.P0(bond_mat), self.P0(expiry)
        h = math.log(pT / (pS * strike)) / sig_p + 0.5 * sig_p
        if call:
            return pT * _norm_cdf(h) - strike * pS * _norm_cdf(h - sig_p)
        return strike * pS * _norm_cdf(-h + sig_p) - pT * _norm_cdf(-h)

    # --- swaps & swaptions ------------------------------------------------ #
    def _schedule(self, expiry: float, tenor: float, freq: int):
        n = int(round(tenor * freq))
        tau = 1.0 / freq
        times = expiry + tau * np.arange(1, n + 1)
        return times, np.full(n, tau)

    def annuity(self, expiry: float, tenor: float, freq: int = 2) -> float:
        times, taus = self._schedule(expiry, tenor, freq)
        return float(np.sum(taus * np.array([self.P0(t) for t in times])))

    def forward_swap_rate(self, expiry: float, tenor: float, freq: int = 2) -> float:
        times, _ = self._schedule(expiry, tenor, freq)
        return (self.P0(expiry) - self.P0(times[-1])) / self.annuity(expiry, tenor, freq)

    def swaption_price(
        self,
        expiry: float,
        tenor: float,
        strike: float | None = None,
        freq: int = 2,
        payer: bool = True,
    ) -> float:
        """European swaption price via Jamshidian decomposition.

        ``strike=None`` prices the at-the-money swaption (strike = forward swap
        rate). Notional is 1.
        """
        times, taus = self._schedule(expiry, tenor, freq)
        if strike is None:
            strike = self.forward_swap_rate(expiry, tenor, freq)

        # Coupon-bond cashflows: c_i = K*tau_i, plus principal at the last date.
        c = strike * taus
        c[-1] += 1.0

        # Jamshidian critical rate r*: sum_i c_i P(expiry, T_i; r*) = 1.
        Avals = np.array([self.A(expiry, t) for t in times])
        Bvals = np.array([self.B(expiry, t) for t in times])

        def coupon_bond(r: float) -> float:
            return float(np.sum(c * Avals * np.exp(-Bvals * r))) - 1.0

        r_star = _bisect(coupon_bond, -0.5, 1.0)
        strikes = Avals * np.exp(-Bvals * r_star)  # X_i

        # Receiver swaption = call on coupon bond; payer = put. Decompose.
        call = not payer
        price = sum(
            c[i] * self.zbo(expiry, times[i], strikes[i], call=call)
            for i in range(len(times))
        )
        return float(price)


def _bisect(f, lo: float, hi: float, tol: float = 1e-12) -> float:
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        # widen once; coupon bond is monotone decreasing in r
        lo, hi = -2.0, 2.0
        flo, fhi = f(lo), f(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)
