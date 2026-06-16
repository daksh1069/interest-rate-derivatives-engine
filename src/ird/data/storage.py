"""Curve-history storage: Parquet for the data, SQLite for a fast date index.

Design: the full wide rate history is persisted as a single columnar Parquet
file (cheap to read, compresses well). A lightweight SQLite table indexes which
dates are present and how many pillars each has, so date-range availability
queries don't require loading Parquet.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pandas as pd

from ird.core.curve_date import CurveDate
from ird.logging_config import get_logger

logger = get_logger(__name__)

_PARQUET_NAME = "sofr_history.parquet"
_CSV_FALLBACK_NAME = "sofr_history.csv"


def _parquet_available() -> bool:
    """True if a pandas Parquet engine (pyarrow or fastparquet) is importable."""
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


class CurveStore:
    """Persist and query SOFR curve history.

    Args:
        db_dir: Directory holding the Parquet file and SQLite index.
    """

    def __init__(self, db_dir: Path) -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.db_dir / _PARQUET_NAME
        self.csv_path = self.db_dir / _CSV_FALLBACK_NAME
        self.index_path = self.db_dir / "curves.sqlite"
        self._use_parquet = _parquet_available()
        if not self._use_parquet:
            logger.warning(
                "No Parquet engine (pyarrow/fastparquet) found; falling back to "
                "CSV storage. Install pyarrow for the intended columnar format."
            )
        self._init_index()

    def _init_index(self) -> None:
        with sqlite3.connect(self.index_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS curve_index (
                    obs_date  TEXT PRIMARY KEY,
                    n_pillars INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def write_history(self, df: pd.DataFrame) -> None:
        """Persist a wide history frame (index=date, columns=tenors) and reindex."""
        df = df.sort_index()
        if self._use_parquet:
            df.to_parquet(self.parquet_path, engine="auto")
        else:
            df.to_csv(self.csv_path, index_label="date")
        with sqlite3.connect(self.index_path) as conn:
            conn.execute("DELETE FROM curve_index")
            rows = [
                (idx.date().isoformat(), int(row.notna().sum()))
                for idx, row in df.iterrows()
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO curve_index VALUES (?, ?)", rows
            )
            conn.commit()
        logger.info(
            "Wrote %d curve dates (%d pillars) to %s",
            len(df), df.shape[1], self.parquet_path,
        )

    def read_history(
        self, start: dt.date | None = None, end: dt.date | None = None
    ) -> pd.DataFrame:
        """Load the history frame, optionally restricted to ``[start, end]``."""
        if self.parquet_path.exists():
            df = pd.read_parquet(self.parquet_path, engine="auto")
        elif self.csv_path.exists():
            df = pd.read_csv(self.csv_path, index_col="date", parse_dates=["date"])
        else:
            raise FileNotFoundError(
                f"No history in {self.db_dir}; run the fetcher first."
            )
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def available_dates(self) -> list[dt.date]:
        """Dates present in the index, ascending."""
        with sqlite3.connect(self.index_path) as conn:
            cur = conn.execute("SELECT obs_date FROM curve_index ORDER BY obs_date")
            return [dt.date.fromisoformat(r[0]) for r in cur.fetchall()]

    def get_curve_date(self, date: dt.date) -> CurveDate:
        """Materialise a single :class:`CurveDate` for ``date``."""
        df = self.read_history(start=date, end=date)
        if df.empty:
            raise KeyError(f"No curve for {date}")
        row = df.iloc[0].dropna()
        return CurveDate(date=date, rates={str(k): float(v) for k, v in row.items()})

    def __len__(self) -> int:
        with sqlite3.connect(self.index_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM curve_index").fetchone()[0])
