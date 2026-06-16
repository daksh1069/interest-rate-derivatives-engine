"""Random and quasi-random standard-normal generators (pure NumPy).

Provides pseudo-random normals (with optional antithetic pairing) and a
low-discrepancy alternative (Halton / van der Corput radical inverse mapped
through an inverse-normal CDF), so the engine can demonstrate quasi-Monte Carlo
convergence without SciPy.
"""

from __future__ import annotations

import numpy as np

# Acklam's rational approximation to the inverse standard-normal CDF.
_A = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
_B = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00]


def norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF (vectorized, Acklam approximation)."""
    p = np.asarray(p, float)
    out = np.empty_like(p)
    lo, hi = 0.02425, 1 - 0.02425
    # lower tail
    m = p < lo
    q = np.sqrt(-2 * np.log(p[m]))
    out[m] = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
        (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
    )
    # upper tail
    m = p > hi
    q = np.sqrt(-2 * np.log(1 - p[m]))
    out[m] = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
        (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
    )
    # central region
    m = (p >= lo) & (p <= hi)
    q = p[m] - 0.5
    r = q * q
    out[m] = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
        ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1
    )
    return out


def _van_der_corput(n: int, base: int) -> np.ndarray:
    """Radical-inverse (van der Corput) sequence for indices 1..n."""
    out = np.zeros(n)
    for i in range(1, n + 1):
        f, r, k = 1.0, 0.0, i
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        out[i - 1] = r
    return out


def standard_normals_1d(
    n: int, method: str = "pseudo", antithetic: bool = True, seed: int = 0
) -> np.ndarray:
    """Return ``n`` standard normals via pseudo-random or Halton QMC."""
    if method == "qmc":
        u = _van_der_corput(n, base=2)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return norm_ppf(u)
    rng = np.random.default_rng(seed)
    if antithetic:
        half = (n + 1) // 2
        z = rng.standard_normal(half)
        return np.concatenate([z, -z])[:n]
    return rng.standard_normal(n)


def standard_normals_2d(
    n_paths: int, n_steps: int, antithetic: bool = True, seed: int = 0
) -> np.ndarray:
    """Return an ``(n_paths, n_steps)`` array of pseudo-random normals."""
    rng = np.random.default_rng(seed)
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_steps))
        return np.concatenate([z, -z], axis=0)[:n_paths]
    return rng.standard_normal((n_paths, n_steps))
