"""End-to-end pipeline tests (offline / synthetic)."""

from __future__ import annotations

import datetime as dt

from ird.config import Settings
from ird.data.fetch_sofr import build_dataset
from ird.data.storage import CurveStore


def test_build_dataset_synthetic(tmp_settings: Settings) -> None:
    store = build_dataset(
        source="synthetic",
        start=dt.date(2021, 1, 1),
        end=dt.date(2021, 6, 30),
        settings=tmp_settings,
    )
    dates = store.available_dates()
    assert len(dates) > 100
    assert dates == sorted(dates)
    # Round-trip a single curve date.
    cd = store.get_curve_date(dates[-1])
    assert set(cd.rates) == set(tmp_settings.tenors)
    mats, _ = cd.as_arrays()
    assert mats == sorted(mats)


def test_store_persists_and_reloads(tmp_settings: Settings) -> None:
    build_dataset(
        source="synthetic",
        start=dt.date(2021, 1, 1),
        end=dt.date(2021, 3, 31),
        settings=tmp_settings,
    )
    # New store instance reading the same dir sees the data.
    reopened = CurveStore(tmp_settings.db_dir)
    assert len(reopened) > 40
    df = reopened.read_history()
    assert not df.isna().to_numpy().any()


def test_auto_falls_back_to_synthetic_without_key(tmp_settings: Settings) -> None:
    assert not tmp_settings.has_fred_key
    store = build_dataset(
        source="auto",
        start=dt.date(2022, 1, 1),
        end=dt.date(2022, 2, 28),
        settings=tmp_settings,
    )
    assert len(store) > 30
