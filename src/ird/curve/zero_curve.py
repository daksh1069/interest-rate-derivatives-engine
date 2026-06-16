"""The :class:`ZeroCurve` value type.

A discount curve defined by pillar maturities and discount factors plus an
interpolation scheme. Exposes the three quantities everything downstream needs:
discount factors, (continuously-compounded) zero rates, and forward rates.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from ird.curve.interpolation import METHODS, DfFunc


class ZeroCurve:
    """Interpolated discount curve.

    Args:
        times: Pillar maturities in years, ascending. A ``t=0`` anchor with
            discount factor 1.0 is added automatically if absent.
        discount_factors: Discount factor at each pillar maturity.
        method: Interpolation scheme: ``"loglinear"`` (default), ``"cubic"``,
            or ``"monotone_convex"``.
        date: Optional observation date, for reference.
    """

    def __init__(
        self,
        times: np.ndarray,
        discount_factors: np.ndarray,
        method: str = "loglinear",
        date: dt.date | None = None,
    ) -> None:
        t = np.asarray(times, dtype=float)
        d = np.asarray(discount_factors, dtype=float)
        if t[0] > 0:
            t = np.concatenate([[0.0], t])
            d = np.concatenate([[1.0], d])
        if not np.all(np.diff(t) > 0):
            raise ValueError("times must be strictly increasing")
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
        self.times = t
        self.dfs = d
        self.method = method
        self.date = date
        self._df: DfFunc = METHODS[method](t, d)

    @property
    def pillars(self) -> np.ndarray:
        """Pillar maturities excluding the t=0 anchor."""
        return self.times[1:]

    def discount_factor(self, T: float | np.ndarray) -> float | np.ndarray:
        """Discount factor P(0, T)."""
        out = self._df(T)
        return float(out[0]) if np.isscalar(T) or np.ndim(T) == 0 else out

    def zero_rate(self, T: float | np.ndarray) -> float | np.ndarray:
        """Continuously-compounded zero rate z(T) = -ln P(0,T) / T."""
        t = np.atleast_1d(np.asarray(T, dtype=float))
        df = self._df(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(t > 0, -np.log(df) / t, 0.0)
        return float(z[0]) if np.isscalar(T) or np.ndim(T) == 0 else z

    def forward_rate(
        self, T1: float | np.ndarray, T2: float | np.ndarray
    ) -> float | np.ndarray:
        """Continuously-compounded forward rate between T1 and T2."""
        t1 = np.atleast_1d(np.asarray(T1, dtype=float))
        t2 = np.atleast_1d(np.asarray(T2, dtype=float))
        df1, df2 = self._df(t1), self._df(t2)
        fwd = np.log(df1 / df2) / (t2 - t1)
        return float(fwd[0]) if np.ndim(T1) == 0 and np.ndim(T2) == 0 else fwd

    def instantaneous_forward(self, T: float | np.ndarray, eps: float = 1e-5):
        """Instantaneous forward rate f(T) via a central difference."""
        t = np.atleast_1d(np.asarray(T, dtype=float))
        lo = np.maximum(t - eps, 0.0)
        f = np.log(self._df(lo) / self._df(t + eps)) / ((t + eps) - lo)
        return float(f[0]) if np.ndim(T) == 0 else f

    def par_rate(self, T: float, freq: int = 1) -> float:
        """Par swap rate for maturity T with ``freq`` payments per year."""
        n = int(round(T * freq))
        taus = np.full(n, 1.0 / freq)
        pay_times = np.cumsum(taus)
        dfs = self._df(pay_times)
        annuity = float(np.sum(taus * dfs))
        return (1.0 - float(dfs[-1])) / annuity

    def __repr__(self) -> str:
        d = self.date.isoformat() if self.date else "n/a"
        return f"ZeroCurve(date={d}, pillars={len(self.pillars)}, method={self.method})"
