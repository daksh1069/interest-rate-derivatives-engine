"""Market conventions: tenor parsing and day-count year fractions.

SOFR OIS conventions are Actual/360 for the floating leg accrual. We expose the
common day-count bases so downstream pricing code can be explicit about its
assumptions rather than hard-coding 1/360 or 1/365 everywhere.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

_TENOR_UNIT_MONTHS = {"D": None, "W": None, "M": 1, "Y": 12}


class DayCount(str, Enum):
    """Supported day-count conventions."""

    ACT_360 = "ACT/360"
    ACT_365F = "ACT/365F"
    THIRTY_360 = "30/360"


def tenor_to_years(tenor: str) -> float:
    """Convert a tenor label (e.g. ``"3M"``, ``"10Y"``, ``"2W"``) to year fraction.

    Uses calendar approximations: 1M = 1/12 yr, 1W = 7/365 yr, 1D = 1/365 yr.
    These are appropriate for pillar labelling, not for accrual; use
    :func:`year_fraction` with actual dates for cashflow accrual.

    Raises:
        ValueError: if the tenor string cannot be parsed.
    """
    s = tenor.strip().upper()
    if len(s) < 2 or not s[:-1].isdigit():
        raise ValueError(f"Unparseable tenor: {tenor!r}")
    n, unit = int(s[:-1]), s[-1]
    if unit == "Y":
        return float(n)
    if unit == "M":
        return n / 12.0
    if unit == "W":
        return n * 7.0 / 365.0
    if unit == "D":
        return n / 365.0
    raise ValueError(f"Unknown tenor unit {unit!r} in {tenor!r}")


def year_fraction(start: dt.date, end: dt.date, basis: DayCount = DayCount.ACT_360) -> float:
    """Accrual year fraction between two dates under ``basis``."""
    if end < start:
        raise ValueError(f"end ({end}) precedes start ({start})")
    if basis is DayCount.ACT_360:
        return (end - start).days / 360.0
    if basis is DayCount.ACT_365F:
        return (end - start).days / 365.0
    if basis is DayCount.THIRTY_360:
        d1 = min(start.day, 30)
        d2 = min(end.day, 30) if d1 == 30 else end.day
        return (
            360 * (end.year - start.year)
            + 30 * (end.month - start.month)
            + (d2 - d1)
        ) / 360.0
    raise ValueError(f"Unsupported day count: {basis}")
