"""Yield-curve construction.

Modules:
    bootstrapper.py  - zero curve from par swap rates
    nss_fitter.py    - Nelson-Siegel-Svensson parametric fit
    interpolation.py - log-linear, cubic spline, monotone convex
    zero_curve.py    - ZeroCurve discount/zero/forward object

Consumes :class:`ird.core.CurveDate` / :class:`ird.data.CurveStore`.
"""

from __future__ import annotations

from ird.curve.bootstrapper import bootstrap_curve, repricing_error_bps
from ird.curve.nss_fitter import NSSCurve, fit_nss
from ird.curve.zero_curve import ZeroCurve

__all__ = [
    "NSSCurve",
    "ZeroCurve",
    "bootstrap_curve",
    "fit_nss",
    "repricing_error_bps",
]
