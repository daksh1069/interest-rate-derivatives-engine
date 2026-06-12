"""Phase 2 (Weeks 3-4): yield-curve construction.

Planned modules:
    bootstrapper.py  - zero curve from SOFR OIS par swap rates
    nss_fitter.py    - Nelson-Siegel-Svensson parametric fit
    interpolation.py - log-linear, cubic spline, monotone convex

Not yet implemented. The Phase 1 :class:`ird.core.CurveDate` and
:class:`ird.data.CurveStore` provide the inputs this phase consumes.
"""

from __future__ import annotations

__all__: list[str] = []
