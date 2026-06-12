"""Central configuration.

Resolves paths and runtime settings from environment variables with sensible
defaults, so the library runs out-of-the-box (offline) but can be pointed at a
live FRED feed by setting ``FRED_API_KEY``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    # src/ird/config.py -> project root is three parents up.
    return Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dependency).

    Only sets variables that are not already present in the environment.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(_project_root() / ".env")

# The canonical SOFR OIS swap pillars used throughout the engine.
DEFAULT_TENORS: tuple[str, ...] = (
    "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y",
)

# FRED series IDs for the constant-maturity / OIS pillars we can pull live.
# (Treasury CMT used as a pragmatic, freely available proxy for several pillars;
# swap in true SOFR OIS series where you have entitlements.)
FRED_SERIES: dict[str, str] = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    data_dir: Path
    fred_api_key: str | None
    tenors: tuple[str, ...] = field(default=DEFAULT_TENORS)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def db_dir(self) -> Path:
        return self.data_dir / "db"

    @property
    def index_db(self) -> Path:
        return self.db_dir / "curves.sqlite"

    @property
    def has_fred_key(self) -> bool:
        return bool(self.fred_api_key)

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Build a :class:`Settings` from the current environment."""
    data_dir = Path(os.environ.get("IRD_DATA_DIR", _project_root() / "data"))
    return Settings(
        data_dir=data_dir,
        fred_api_key=os.environ.get("FRED_API_KEY") or None,
    )
