"""The :class:`CurveDate` value type.

A ``CurveDate`` is the atomic market observation for the engine: a set of par
swap rates keyed by tenor for a single observation date. Phase 2 turns this into
a bootstrapped ``ZeroCurve``; everything upstream (data, validation, storage)
deals in ``CurveDate`` objects.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ird.core.conventions import tenor_to_years


@dataclass(frozen=True)
class CurveDate:
    """Par swap rates for one observation date.

    Attributes:
        date: Observation date.
        rates: Mapping of tenor label (e.g. ``"5Y"``) to par rate as a decimal
            (e.g. ``0.0425`` for 4.25%). Rates must be finite; NaNs are not
            permitted (run the data through :mod:`ird.data.validation` first).
    """

    date: dt.date
    rates: dict[str, float]

    def __post_init__(self) -> None:
        if not self.rates:
            raise ValueError(f"CurveDate {self.date} has no rates")
        for tenor, rate in self.rates.items():
            # Validates tenor label parses; raises ValueError otherwise.
            tenor_to_years(tenor)
            if rate != rate:  # NaN check without importing math
                raise ValueError(f"CurveDate {self.date} tenor {tenor} has NaN rate")

    @property
    def tenors(self) -> list[str]:
        """Tenor labels sorted by maturity in years (ascending)."""
        return sorted(self.rates, key=tenor_to_years)

    def as_arrays(self) -> tuple[list[float], list[float]]:
        """Return ``(maturities_in_years, rates)`` sorted by maturity."""
        ts = self.tenors
        return [tenor_to_years(t) for t in ts], [self.rates[t] for t in ts]

    def rate(self, tenor: str) -> float:
        """Par rate at ``tenor``; raises ``KeyError`` if absent."""
        return self.rates[tenor]

    def __str__(self) -> str:
        body = ", ".join(f"{t}={self.rates[t] * 100:.3f}%" for t in self.tenors)
        return f"CurveDate({self.date.isoformat()}: {body})"
