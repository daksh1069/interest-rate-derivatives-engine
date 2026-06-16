"""The :class:`VolSurface` value type: an ATM swaption implied-vol grid.

Indexed by (option expiry, swap tenor). Stored as normal (bp) or lognormal
vols; the convention is carried on the object so calibration code can interpret
it correctly.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum

from ird.core.conventions import tenor_to_years


class VolType(str, Enum):
    """Quoting convention for swaption implied vols."""

    NORMAL = "normal"        # basis-point (Bachelier) vol
    LOGNORMAL = "lognormal"  # Black/percentage vol


@dataclass(frozen=True)
class VolSurface:
    """ATM swaption implied-vol grid for one observation date.

    Attributes:
        date: Observation date.
        expiries: Option-expiry tenor labels (rows), e.g. ``["1Y", "2Y", "5Y"]``.
        tenors: Underlying swap tenor labels (columns), e.g. ``["5Y", "10Y"]``.
        vols: ``vols[i][j]`` is the ATM vol for expiry ``expiries[i]`` into swap
            tenor ``tenors[j]``. Normal vols are decimals (0.0085 == 85 bp);
            lognormal vols are decimals (0.30 == 30%).
        vol_type: Quoting convention.
    """

    date: dt.date
    expiries: list[str]
    tenors: list[str]
    vols: list[list[float]]
    vol_type: VolType = VolType.NORMAL

    def __post_init__(self) -> None:
        if len(self.vols) != len(self.expiries):
            raise ValueError("vols row count must equal number of expiries")
        for i, row in enumerate(self.vols):
            if len(row) != len(self.tenors):
                raise ValueError(f"vols row {i} length must equal number of tenors")

    def get(self, expiry: str, tenor: str) -> float:
        """ATM vol for a given (expiry, tenor) cell."""
        return self.vols[self.expiries.index(expiry)][self.tenors.index(tenor)]

    def expiry_years(self) -> list[float]:
        return [tenor_to_years(e) for e in self.expiries]

    def tenor_years(self) -> list[float]:
        return [tenor_to_years(t) for t in self.tenors]
