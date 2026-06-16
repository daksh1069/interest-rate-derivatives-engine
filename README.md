# Interest Rate Derivatives Pricing Engine

**Yield Curve Construction · Short-Rate Model Calibration · Monte Carlo Swaption Pricing**

An end-to-end fixed-income derivatives engine in **pure Python** (NumPy / SciPy):
bootstrap Treasury/SOFR zero curves, calibrate Vasicek and Hull-White short-rate
models, price European and Bermudan swaptions with a vectorized Monte Carlo
engine (antithetic + control variates, Longstaff-Schwartz LSM), compute Greeks,
and run walk-forward hedging backtests and stress tests through the 2020 COVID,
2022 hike-cycle, and 2023 SVB regimes.

## Quick start

```bash
python3 -m venv .venv          # or your pyenv 3.11 binary, see note below
source .venv/bin/activate
pip install -r requirements.txt
python main.py                 # runs the pipeline, prints results, saves figures/
```

`main.py` builds the dataset, bootstraps the latest curve, fits Nelson-Siegel-
Svensson, calibrates Hull-White to an ATM swaption vol surface, prints results,
and writes metrics to `results/` and plots to `figures/`: curve shapes across
regimes, the 3M/2Y/10Y rate history, the 2s10s inversion, the bootstrapped
zero/forward/NSS curve, the discount-factor curve, and the Hull-White vol-surface
calibration fit.

No API key required: the data pipeline ships with a deterministic synthetic
generator that reproduces the real curve *regimes* (COVID ZLB, the +525 bp hike
cycle, the SVB inversion). To pull live data instead, copy `.env.example` to
`.env`, add a free [FRED API key](https://fredaccount.stlouisfed.org/apikeys),
and run `python main.py --source fred`.

> **pyenv note:** if `python3` is a pyenv shim, build the venv from a concrete
> interpreter so it doesn't re-enter the shim:
> `~/.pyenv/versions/3.11.0/bin/python -m venv .venv`.

Want to run the tests too? `pip install pytest` then `pytest` (21 unit tests).

## Project layout

```
main.py            # run this — builds the dataset, prints results, saves figures/
requirements.txt   # the only deps you need
src/ird/
├── core/      # CurveDate, VolSurface, day-count & tenor conventions
├── data/      # FRED/synthetic ingestion, validation, Parquet+SQLite storage
├── curve/     # bootstrapping, Nelson-Siegel-Svensson, interpolation
├── models/    # Vasicek, Hull-White 1F, calibration
├── greeks/    # DV01, key-rate durations, pathwise MC Greeks
├── backtest/  # walk-forward delta hedging, P&L attribution
└── stress/    # scenarios, VaR/CVaR, reverse stress test
notebooks/     # narrative & visualization
tests/         # unit tests
```

## Data pipeline

`python main.py` builds a validated, business-day-complete daily history of
Treasury/SOFR pillars (1M–30Y) from 2018 to present, persisted as Parquet with a
SQLite date index. The validation layer flags missing business days, implausible
>150 bp single-day jumps, NaNs, and curve inversions before cleaning.

```python
from ird.data import CurveStore
from ird.config import get_settings

store = CurveStore(get_settings().db_dir)
curve = store.get_curve_date(store.available_dates()[-1])
print(curve)  # CurveDate(2026-06-12: 1M=..., 3M=..., ... 30Y=...)
```

## License

MIT — see [LICENSE](LICENSE).
