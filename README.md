# Interest Rate Derivatives Pricing Engine

**Yield Curve Construction · Short-Rate Model Calibration · Monte Carlo Swaption Pricing**

An end-to-end fixed-income derivatives engine in **Python + C++**: bootstrap SOFR
zero curves, calibrate Vasicek and Hull-White short-rate models, price European
and Bermudan swaptions with a high-performance Monte Carlo engine (pybind11 /
OpenMP / Sobol), compute Greeks, and run walk-forward hedging backtests and
stress tests through the 2020 COVID, 2022 hike-cycle, and 2023 SVB regimes.

> Status: **Phase 1 complete** (data infrastructure). Phases 2–8 scaffolded.
> See [`IRD_Pricing_Engine_Project_Plan.md`](IRD_Pricing_Engine_Project_Plan.md)
> for the full 16-week roadmap.

## Quick start

```bash
python3 -m venv .venv          # or your pyenv 3.11 binary, see note below
source .venv/bin/activate
pip install -r requirements.txt
python main.py                 # runs the pipeline, prints results, saves figures/
```

`main.py` builds the dataset, prints a results summary, and writes three plots to
`figures/` (curve shapes across regimes, the 3M/2Y/10Y rate history, and the
2s10s inversion).

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
├── core/      # CurveDate, VolSurface, day-count & tenor conventions  [done]
├── data/      # Phase 1: FRED/synthetic ingestion, validation, storage [done]
├── curve/     # Phase 2: bootstrapping, Nelson-Siegel-Svensson, interpolation
├── models/    # Phase 3: Vasicek, Hull-White 1F, calibration
├── greeks/    # Phase 5: DV01, key-rate durations, pathwise MC Greeks
├── backtest/  # Phase 6: walk-forward delta hedging, P&L attribution
└── stress/    # Phase 7: scenarios, VaR/CVaR, reverse stress test
notebooks/     # per-phase narrative & visualization (placeholders)
tests/         # unit tests
```

## Phase 1 — data pipeline (implemented)

Running `python main.py` builds a validated, business-day-complete daily history
of SOFR pillars (1M–30Y) from 2018 to present, persisted as Parquet with a SQLite
date index. The validation layer flags missing business days, implausible
>150 bp single-day jumps, NaNs, and curve inversions before cleaning.

```python
from ird.data import CurveStore
from ird.config import get_settings

store = CurveStore(get_settings().db_dir)
curve = store.get_curve_date(store.available_dates()[-1])
print(curve)  # CurveDate(2026-06-12: 1M=..., 3M=..., ... 30Y=...)
```

## ATS keywords

`yield curve` · `bootstrapping` · `Nelson-Siegel-Svensson` · `SOFR` ·
`forward rates` · `discount factors` · `Hull-White` · `Vasicek` ·
`short rate model` · `mean reversion` · `Monte Carlo simulation` ·
`variance reduction` · `antithetic variates` · `control variates` ·
`Sobol sequences` · `swaption pricing` · `Bermudan swaption` ·
`Longstaff-Schwartz` · `C++` · `pybind11` · `OpenMP` · `DV01` · `duration` ·
`convexity` · `key rate duration` · `interest rate risk` · `stress testing` ·
`P&L attribution` · `VaR` · `CVaR` · `fixed income` · `interest rate derivatives`

## License

MIT — see [LICENSE](LICENSE).
