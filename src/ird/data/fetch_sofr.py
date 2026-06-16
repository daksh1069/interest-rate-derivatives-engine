"""Build the Treasury/SOFR curve-history dataset.

Pulls daily rates from FRED when ``FRED_API_KEY`` is set, otherwise falls back
to a deterministic synthetic history so the pipeline always runs. The output is
validated, cleaned, and persisted via :class:`ird.data.storage.CurveStore`.

Run it:
    python -m ird.data.fetch_sofr            # offline synthetic by default
    FRED_API_KEY=... python -m ird.data.fetch_sofr --source fred
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import pandas as pd

from ird.config import FRED_SERIES, Settings, get_settings
from ird.data.storage import CurveStore
from ird.data.synthetic import generate_history
from ird.data.validation import clean_history, validate_history
from ird.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

# Official FRED API (JSON). The older graph/fredgraph.csv endpoint is slow and
# prone to read timeouts; this is the documented, supported endpoint.
_FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT = 60
_MAX_RETRIES = 3


def fetch_fred_series(
    series_id: str, start: dt.date, api_key: str, session=None
) -> pd.Series:
    """Fetch one FRED series as a date-indexed percentage->decimal Series.

    Uses the official ``series/observations`` JSON endpoint, with retries and
    backoff for transient network timeouts.
    """
    import requests

    sess = session or requests.Session()
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.isoformat(),
    }
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = sess.get(_FRED_API_URL, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            frame = pd.DataFrame(obs)
            if frame.empty:
                return pd.Series(dtype=float, name=series_id)
            dates = pd.to_datetime(frame["date"])
            # FRED encodes missing values as ".".
            values = pd.to_numeric(frame["value"], errors="coerce") / 100.0
            return pd.Series(values.to_numpy(), index=dates, name=series_id)
        except requests.exceptions.RequestException as exc:
            last_err = exc
            wait = 2 ** attempt
            logger.warning(
                "FRED fetch %s failed (attempt %d/%d): %s; retrying in %ds",
                series_id, attempt, _MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"FRED fetch failed for {series_id}: {last_err}")


def fetch_fred_history(settings: Settings, start: dt.date) -> pd.DataFrame:
    """Assemble the wide curve history from FRED constant-maturity series."""
    import requests

    assert settings.fred_api_key is not None
    session = requests.Session()
    cols: dict[str, pd.Series] = {}
    for tenor in settings.tenors:
        series_id = FRED_SERIES.get(tenor)
        if series_id is None:
            logger.warning("No FRED series mapped for tenor %s; skipping", tenor)
            continue
        logger.info("Fetching FRED series %s for tenor %s", series_id, tenor)
        cols[tenor] = fetch_fred_series(
            series_id, start, settings.fred_api_key, session=session
        )
    df = pd.DataFrame(cols)
    df.index.name = None
    return df


def build_dataset(
    source: str = "auto",
    start: dt.date = dt.date(2018, 1, 1),
    end: dt.date | None = None,
    settings: Settings | None = None,
) -> CurveStore:
    """Build, validate, clean, and persist the curve-history dataset.

    Args:
        source: ``"fred"``, ``"synthetic"``, or ``"auto"`` (FRED if a key is
            present, else synthetic).
        start: First observation date.
        end: Last observation date (defaults to today).
        settings: Override settings (mainly for tests).

    Returns:
        The populated :class:`CurveStore`.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()

    use_fred = source == "fred" or (source == "auto" and settings.has_fred_key)
    if use_fred and not settings.has_fred_key:
        raise RuntimeError("source='fred' requested but FRED_API_KEY is not set")

    if use_fred:
        logger.info("Building dataset from FRED")
        df = fetch_fred_history(settings, start)
    else:
        logger.info("Building dataset from synthetic generator (no FRED key)")
        df = generate_history(start=start, end=end, tenors=settings.tenors)

    report = validate_history(df)
    if not report.ok:
        logger.warning("Validation flagged issues; cleaning. %s", report.summary())
    df = clean_history(df, report)

    store = CurveStore(settings.db_dir)
    store.write_history(df)
    return store


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Build the SOFR curve-history dataset.")
    parser.add_argument(
        "--source", choices=["auto", "fred", "synthetic"], default="auto"
    )
    parser.add_argument("--start", default="2018-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args(argv)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else None

    store = build_dataset(source=args.source, start=start, end=end)
    dates = store.available_dates()
    logger.info(
        "Done. %d curve dates stored (%s .. %s).",
        len(store), dates[0], dates[-1],
    )
    sample = store.get_curve_date(dates[-1])
    logger.info("Latest curve: %s", sample)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
