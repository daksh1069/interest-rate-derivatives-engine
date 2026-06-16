"""Yield-curve construction.

Modules:
    bootstrapper.py  - zero curve from par swap rates
    nss_fitter.py    - Nelson-Siegel-Svensson parametric fit
    interpolation.py - log-linear, cubic spline, monotone convex

Consumes :class:`ird.core.CurveDate` / :class:`ird.data.CurveStore`.
"""

from __future__ import annotations

__all__: list[str] = []
