# Notebooks

Each notebook is the narrative + visualization layer for one phase. Keep heavy
logic in `src/ird/`; notebooks should import and demonstrate, not implement.

| Notebook | Phase | Content |
|----------|-------|---------|
| `01_yield_curve.ipynb`     | 2 | Bootstrapping, NSS fit, curve shapes (COVID / 2022 / SVB) |
| `02_model_calibration.ipynb` | 3 | Vasicek MLE, Hull-White vol-surface calibration |
| `03_swaption_pricing.ipynb`  | 4 | MC vs analytical, Bermudan LSM, convergence & speedups |
| `04_greeks_risk.ipynb`       | 5 | DV01, key-rate durations, convexity, Vega |
| `05_backtesting.ipynb`       | 6 | Walk-forward delta hedge, P&L attribution |
| `06_stress_testing.ipynb`    | 7 | Scenarios, VaR/CVaR, reverse stress test |

A `00_data_pipeline.ipynb` demonstrating Phase 1 (this scaffold) is the natural
first addition.
