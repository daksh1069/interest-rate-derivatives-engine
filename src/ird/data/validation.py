"""Data validation for raw SOFR curve history.

Checks the three failure modes that silently corrupt downstream curve
construction: missing business days, implausible single-day jumps, and missing
pillars. Problems are flagged in a structured :class:`ValidationReport`; the
caller decides whether to forward-fill or drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ird.logging_config import get_logger

logger = get_logger(__name__)

# A single-day move larger than this (in basis points) is treated as suspicious.
MAX_DAILY_JUMP_BPS = 150.0


@dataclass
class ValidationReport:
    """Outcome of validating a curve-history frame."""

    n_rows: int
    n_cols: int
    missing_business_days: list[pd.Timestamp] = field(default_factory=list)
    nan_cells: int = 0
    large_jumps: list[tuple[pd.Timestamp, str, float]] = field(default_factory=list)
    inverted_dates: list[pd.Timestamp] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            not self.missing_business_days
            and self.nan_cells == 0
            and not self.large_jumps
        )

    def summary(self) -> str:
        return (
            f"rows={self.n_rows} cols={self.n_cols} "
            f"missing_bdays={len(self.missing_business_days)} "
            f"nan_cells={self.nan_cells} "
            f"large_jumps={len(self.large_jumps)} "
            f"inverted={len(self.inverted_dates)}"
        )


def validate_history(
    df: pd.DataFrame, max_jump_bps: float = MAX_DAILY_JUMP_BPS
) -> ValidationReport:
    """Validate a wide curve-history frame (index=date, columns=tenors, values=rates).

    Args:
        df: Rates as decimals (0.04 == 4%), indexed by date, one column per tenor.
        max_jump_bps: Threshold for flagging single-day moves.

    Returns:
        A :class:`ValidationReport`. This function never mutates ``df``.
    """
    df = df.sort_index()
    report = ValidationReport(n_rows=len(df), n_cols=df.shape[1])

    # Missing business days within the observed range.
    if len(df) >= 2:
        bdays = pd.bdate_range(df.index.min(), df.index.max())
        missing = bdays.difference(df.index)
        report.missing_business_days = list(missing)

    # NaN cells.
    report.nan_cells = int(df.isna().to_numpy().sum())

    # Implausible single-day jumps (bp).
    diff_bps = df.diff().abs() * 1e4
    jump_mask = diff_bps > max_jump_bps
    for date, row in diff_bps[jump_mask.any(axis=1)].iterrows():
        for tenor in row.index[jump_mask.loc[date]]:
            report.large_jumps.append((date, str(tenor), float(row[tenor])))

    # Curve inversions (informational, not an error — they genuinely happen).
    for date, row in df.iterrows():
        vals = row.dropna().to_numpy()
        if len(vals) >= 2 and np.any(np.diff(vals) < 0):
            report.inverted_dates.append(date)

    logger.info("Validation: %s", report.summary())
    return report


def clean_history(
    df: pd.DataFrame, report: ValidationReport | None = None
) -> pd.DataFrame:
    """Return a cleaned copy: reindex to business days and forward-fill gaps.

    Forward-filling is logged so the audit trail is explicit. Leading NaNs that
    cannot be filled forward are back-filled as a last resort.
    """
    df = df.sort_index()
    if len(df) >= 2:
        bdays = pd.bdate_range(df.index.min(), df.index.max())
        df = df.reindex(bdays)
    n_filled = int(df.isna().to_numpy().sum())
    if n_filled:
        logger.warning("clean_history: forward/back-filling %d cells", n_filled)
    return df.ffill().bfill()
