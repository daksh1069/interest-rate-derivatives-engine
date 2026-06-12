"""Synthetic SOFR curve history generator (offline fallback).

Produces a realistic ~2018->present daily history without any API key, so the
whole pipeline is runnable and testable out of the box. The generator uses a
Nelson-Siegel level/slope/curvature structure whose factors follow a scripted
macro path with the major regimes baked in:

    * 2018-2019  : ~2.0-2.5% policy, gently inverted front end
    * Mar 2020   : COVID collapse to the zero lower bound
    * 2020-2021  : ZIRP, very low and flat
    * 2022-2023  : the +525 bp hike cycle, deep 2s10s inversion
    * 2024-2025  : plateau then gradual cuts

It is intentionally *not* a substitute for real data, but it reproduces the
curve *shapes* the engine must handle, which is what Phases 2-7 stress.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ird.config import DEFAULT_TENORS
from ird.core.conventions import tenor_to_years
from ird.logging_config import get_logger

logger = get_logger(__name__)

# Anchor points (date, short_rate, long_rate, curvature) describing the macro
# regime. Daily values are linearly interpolated between anchors, then a
# Nelson-Siegel curve is built per day and perturbed with small AR(1) noise.
_ANCHORS: list[tuple[dt.date, float, float, float]] = [
    (dt.date(2018, 1, 2), 0.0145, 0.0255, -0.004),
    (dt.date(2018, 12, 31), 0.0240, 0.0269, -0.003),
    (dt.date(2019, 9, 2), 0.0210, 0.0150, 0.002),   # 2019 inversion
    (dt.date(2020, 2, 19), 0.0158, 0.0147, 0.001),
    (dt.date(2020, 3, 23), 0.0005, 0.0070, 0.004),  # COVID ZLB
    (dt.date(2021, 6, 1), 0.0005, 0.0150, 0.006),
    (dt.date(2021, 12, 31), 0.0008, 0.0151, 0.004),
    (dt.date(2022, 6, 15), 0.0150, 0.0335, -0.002), # hikes underway
    (dt.date(2022, 12, 14), 0.0410, 0.0370, -0.006),# bear-flattening
    (dt.date(2023, 3, 8), 0.0490, 0.0395, -0.010),  # pre-SVB peak inversion
    (dt.date(2023, 3, 24), 0.0480, 0.0340, -0.013), # SVB whipsaw
    (dt.date(2023, 7, 26), 0.0533, 0.0390, -0.014), # terminal rate
    (dt.date(2024, 6, 3), 0.0533, 0.0440, -0.009),
    (dt.date(2024, 12, 18), 0.0445, 0.0445, -0.004),# cuts begin
    (dt.date(2025, 12, 1), 0.0360, 0.0430, 0.000),
]


def _interp_factors(target: dt.date) -> tuple[float, float, float]:
    """Piecewise-linear interpolation of (short, long, curvature) anchors."""
    if target <= _ANCHORS[0][0]:
        return _ANCHORS[0][1:]
    if target >= _ANCHORS[-1][0]:
        return _ANCHORS[-1][1:]
    for (d0, s0, l0, c0), (d1, s1, l1, c1) in zip(_ANCHORS, _ANCHORS[1:]):
        if d0 <= target <= d1:
            w = (target - d0).days / max((d1 - d0).days, 1)
            return (s0 + w * (s1 - s0), l0 + w * (l1 - l0), c0 + w * (c1 - c0))
    return _ANCHORS[-1][1:]


def _ns_curve(short: float, long: float, curv: float, taus: np.ndarray) -> np.ndarray:
    """Nelson-Siegel zero curve from level/slope/curvature factors."""
    lam = 0.6  # decay; ~peak curvature loading near the belly
    beta0 = long
    beta1 = short - long  # slope: short minus long (so r(0)=short, r(inf)=long)
    beta2 = curv          # belly curvature; kept small so slope drives inversion
    load_slope = (1 - np.exp(-taus * lam)) / (taus * lam)
    load_curv = load_slope - np.exp(-taus * lam)
    return beta0 + beta1 * load_slope + beta2 * load_curv


def generate_history(
    start: dt.date = dt.date(2018, 1, 1),
    end: dt.date | None = None,
    tenors: tuple[str, ...] = DEFAULT_TENORS,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic wide curve-history frame.

    Returns:
        DataFrame indexed by business date, one column per tenor, rates as
        decimals.
    """
    end = end or dt.date.today()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    taus = np.array([tenor_to_years(t) for t in tenors])

    # AR(1) common noise shared across the curve, plus small per-pillar jitter.
    ar = 0.0
    rows: list[np.ndarray] = []
    for d in dates:
        short, long, curv = _interp_factors(d.date())
        ar = 0.97 * ar + rng.normal(0.0, 0.0008)
        curve = _ns_curve(short + ar, long + 0.6 * ar, curv, taus)
        curve = curve + rng.normal(0.0, 0.00015, size=taus.shape)
        rows.append(np.maximum(curve, -0.005))  # allow mildly negative, floor it

    df = pd.DataFrame(rows, index=dates, columns=list(tenors))
    logger.info(
        "Generated synthetic history: %d dates x %d tenors (%s..%s)",
        len(df), len(tenors), dates[0].date(), dates[-1].date(),
    )
    return df
