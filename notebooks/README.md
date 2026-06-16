# Notebooks

Each notebook is the narrative + visualization layer for a component. Keep heavy
logic in `src/ird/`; notebooks should import and demonstrate, not implement.

| Notebook | Content |
|----------|---------|
| `00_data_pipeline.ipynb`     | Curve ingestion, validation, regime curve shapes |
| `01_yield_curve.ipynb`       | Bootstrapping, NSS fit, curve shapes (COVID / 2022 / SVB) |
| `02_model_calibration.ipynb` | Vasicek MLE, Hull-White vol-surface calibration |
| `03_swaption_pricing.ipynb`  | MC vs analytical, Bermudan LSM, convergence & speedups |
| `04_greeks_risk.ipynb`       | DV01, key-rate durations, convexity, Vega |
| `05_backtesting.ipynb`       | Walk-forward delta hedge, P&L attribution |
| `06_stress_testing.ipynb`    | Scenarios, VaR/CVaR, reverse stress test |
