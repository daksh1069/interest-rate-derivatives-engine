# Interest Rate Derivatives Pricing Engine
## Full Project Plan — Daksh Kumar, NCSU MFM

**Project Title (for resume):** *Yield Curve Construction, Short Rate Model Calibration & Monte Carlo Swaption Pricing Engine*  
**Timeline:** ~16 weeks  
**Stack:** Pure Python (NumPy · SciPy · pandas) · FRED data  
**GitHub Deliverable:** Public repo with modular codebase, Jupyter notebooks, CI tests

---

## ATS Keywords Unlocked

`yield curve` · `bootstrapping` · `Nelson-Siegel-Svensson` · `SOFR` · `forward rates` · `discount factors` · `Hull-White` · `Vasicek` · `short rate model` · `interest rate model calibration` · `mean reversion` · `Monte Carlo simulation` · `variance reduction` · `antithetic variates` · `control variates` · `quasi-Monte Carlo` · `swaption pricing` · `Bermudan swaption` · `Longstaff-Schwartz` · `LSM` · `NumPy` · `SciPy` · `vectorization` · `DV01` · `duration` · `convexity` · `key rate duration` · `interest rate risk` · `stress testing` · `P&L attribution` · `VaR` · `CVaR` · `fixed income` · `interest rate derivatives`

---

## Project Architecture

```
ird-pricing-engine/
├── data/
│   ├── fetch_sofr.py          # FRED API ingestion
│   ├── fetch_swaption_vols.py # Vol surface data
│   └── db/                    # Parquet curve history
├── curve/
│   ├── bootstrapper.py        # Zero curve from swaps
│   ├── nss_fitter.py          # Nelson-Siegel-Svensson
│   └── interpolation.py       # Log-linear, cubic spline, monotone convex
├── models/
│   ├── vasicek.py             # Analytical bond/option pricing
│   ├── hull_white_1f.py       # Trinomial tree + analytical formulas
│   └── calibration.py         # MLE + least-squares to swaption surface
├── mc/
│   ├── paths.py               # Vectorized Hull-White path generator (NumPy)
│   ├── swaption_pricer.py     # European + Bermudan (Longstaff-Schwartz LSM)
│   └── variance_reduction.py  # Antithetic + control variates, Sobol QMC
├── greeks/
│   ├── bump_reprice.py        # DV01, KRDs, finite difference Greeks
│   └── pathwise.py            # Pathwise/likelihood ratio MC Greeks
├── backtest/
│   ├── delta_hedger.py        # Walk-forward delta hedging simulation
│   ├── pnl_attribution.py     # Theta, Delta, Gamma, Vega decomposition
│   └── metrics.py             # Sharpe, hit rate, max drawdown
├── stress/
│   ├── scenarios.py           # Parallel shifts, twist, butterfly
│   ├── historical_scenarios.py# 2020 COVID, 2022 hike cycle, 2023 SVB
│   └── var_cvar.py            # Portfolio VaR/CVaR via MC
├── tests/
│   └── unit tests for all modules
└── notebooks/
    ├── 01_yield_curve.ipynb
    ├── 02_model_calibration.ipynb
    ├── 03_swaption_pricing.ipynb
    ├── 04_greeks_risk.ipynb
    ├── 05_backtesting.ipynb
    └── 06_stress_testing.ipynb
```

---

## Phase 1: Infrastructure & Data Pipeline
**Duration:** Weeks 1–2

### Goal
Build the data foundation. Everything downstream depends on clean, correctly-dated SOFR curve data.

### Implementation
- **FRED API** — pull daily SOFR swap rates (1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y) from 2018–present. This covers pre-COVID baseline, 2020 crash, and the 2022–2023 hike cycle.
- **Swaption vol surface** — pull ATM implied vols from FRED or use academic datasets (Swaption vols from the NY Fed or reconstruct from historical data). Store as expiry × tenor grids.
- **Storage** — Parquet files partitioned by date. Use a lightweight SQLite index for fast date-range queries.
- **Data validation** — check for missing dates, rate inversions, implausible jumps (>150bps single-day). Flag and forward-fill with a warning log.

### Key Outputs
- `CurveDate` object: stores swap rates keyed by tenor for a given date
- `VolSurface` object: expiry × tenor ATM vol grid
- Clean history: ~1,500 curve dates from 2018–present

---

## Phase 2: Yield Curve Construction
**Duration:** Weeks 3–4

### Goal
Convert observable swap rates into a smooth, arbitrage-free zero/discount curve.

### Implementation

**Step 1 — Bootstrapping**  
Bootstrap the zero curve from SOFR OIS swap rates. For each maturity T:
- Use the no-arbitrage swap pricing condition: the fixed leg PV equals the floating leg PV
- Iteratively solve for zero rates / discount factors at each pillar
- Handle day count conventions (Actual/360 for SOFR)

**Step 2 — Interpolation**  
Implement three schemes and compare:
- Log-linear on discount factors (simple, no-arbitrage preserving for flat forwards)
- Cubic spline on zero rates (smooth but can produce negative forwards)
- Monotone convex (industry standard — guarantees positive forwards)

**Step 3 — Nelson-Siegel-Svensson (NSS) Fit**  
Fit the NSS parametric model to the bootstrapped zeros:
```
r(τ) = β₀ + β₁·(1-e^{-τ/λ₁})/(τ/λ₁) + β₂·[(1-e^{-τ/λ₁})/(τ/λ₁) - e^{-τ/λ₁}]
      + β₃·[(1-e^{-τ/λ₂})/(τ/λ₂) - e^{-τ/λ₂}]
```
Optimize {β₀,β₁,β₂,β₃,λ₁,λ₂} via least-squares. This gives:
- Level (β₀), slope (β₁), two curvature terms (β₂,β₃)
- A continuous, differentiable forward rate curve

**Step 4 — Validation**  
- Reprice input swaps from the bootstrapped curve — residuals should be <0.1 bps
- Check forward rates are positive across all maturities
- Plot curve shapes for key dates: 2020-03-16 (COVID), 2022-06-15 (aggressive hike), 2023-03-13 (SVB)

### Key Outputs
- `ZeroCurve` class with `discount_factor(T)`, `zero_rate(T)`, `forward_rate(T1,T2)` methods
- NSS parameter time series (β₀ history = level factor = useful for later stress tests)
- Curve shape analysis notebook

---

## Phase 3: Short Rate Models
**Duration:** Weeks 5–8

### 3A — Vasicek Model (Weeks 5–6)

The simplest mean-reverting short rate model:
```
dr(t) = κ(θ - r(t))dt + σdW(t)
```
Parameters: mean reversion speed κ, long-run mean θ, volatility σ.

**Analytical bond pricing:**
```
P(t,T) = A(t,T) · exp(-B(t,T) · r(t))
B(t,T) = (1 - e^{-κ(T-t)}) / κ
ln A(t,T) = (B(t,T) - (T-t))(κ²θ - σ²/2)/κ² - σ²B(t,T)²/(4κ)
```

**Calibration:**
- Maximum Likelihood Estimation (MLE) on daily SOFR rate time series
- Extract κ̂, θ̂, σ̂ with standard errors
- Diagnostic: plot mean reversion implied by κ against observed rate behavior

**Limitation to document:** Vasicek produces a constant term structure shape and can generate negative rates — deliberately discuss this limitation in your write-up and GitHub README. It motivates Hull-White.

### 3B — Hull-White 1-Factor Model (Weeks 7–8)

Extension that fits the initial term structure exactly:
```
dr(t) = [θ(t) - a·r(t)]dt + σ·dW(t)
```
θ(t) is a time-dependent drift chosen to match the observed zero curve.

**Analytical bond pricing:** Exact, closed-form (extends Vasicek formula with θ(t) implied from the initial curve).

**Analytical bond option pricing (and hence swaption pricing via jamshidian decomposition):**
```
V_cap = P(0,T₁)·N(d₁) - P(0,T₂)·N(d₂)
```
European swaptions = portfolio of bond options (Jamshidian's trick).

**Trinomial Tree:**
- Build a recombining trinomial tree for r
- Price Bermudan swaptions by backward induction
- Verify against analytical European prices (must match within <1bp)

**Calibration:**  
Calibrate (a, σ) to the swaption vol surface (ATM implied vols for a grid of option expiries × swap tenors). Objective function: minimize sum of squared vol errors across the surface. Use `scipy.optimize.minimize` with Levenberg-Marquardt. Report:
- Calibrated (a, σ) values and their economic interpretation
- Vol surface fit RMSE
- Stability of parameters across re-calibration dates

### Key Outputs
- `Vasicek` and `HullWhite1F` classes with `bond_price()`, `swaption_price()`, `calibrate()` methods
- Trinomial tree pricer
- Calibration results notebook with vol surface fit visualization

---

## Phase 4: Vectorized Monte Carlo Engine (Pure Python)
**Duration:** Weeks 9–10

### Goal
Build a fast, fully **vectorized** Monte Carlo swaption pricer in pure Python with NumPy. The performance story here is *vectorization done right* — generating all paths as NumPy arrays in a few operations rather than Python loops — plus variance reduction. No C++.

### Implementation

**Path Generation (Exact Scheme for HW1F):**  
The Hull-White model has an exact (non-Euler) discretization. Use it — Euler introduces bias for large timesteps. Generate the full `(n_paths × n_steps)` array of normals at once and build paths with `np.cumsum` / broadcasting — no per-path Python loop.
```python
# Exact discretization of dr = [theta(t) - a*r]dt + sigma*dW
# r(t+dt) | r(t) ~ Normal(mu(t,dt), v2(dt))
# mu = r(t)*exp(-a*dt) + (theta_bar/a)*(1 - exp(-a*dt))
# v2 = sigma**2/(2a) * (1 - exp(-2a*dt))
```

**Variance Reduction:**
1. **Antithetic variates** — generate paths in pairs (W, -W). Halves effective variance for smooth payoffs.
2. **Control variates** — use the analytically known bond price as a control variate for the discount factor. Reduces MC error dramatically.
3. **Quasi-Monte Carlo (Sobol sequences)** — replace pseudo-random normals with Sobol low-discrepancy sequences via `scipy.stats.qmc.Sobol`. For D-dimensional integration (D = number of timesteps), Sobol converges at O(log(N)^D / N) vs O(1/√N) for standard MC.

**Swaption Pricers:**
- **European swaption** — at expiry, the swaption pays max(swap_value, 0). Price by averaging discounted payoffs.
- **Bermudan swaption** — Longstaff-Schwartz (LSM) algorithm:
  1. Simulate N paths to final maturity
  2. Backward pass: at each exercise date, regress continuation value on basis functions (polynomials) of current state via `numpy.polynomial`
  3. Exercise when intrinsic value > estimated continuation value
  4. Price by forward pass using estimated exercise boundary

**Performance:**  
Vectorize fully so a 100k-path European price runs in well under a second. Optionally use `numpy.random.Generator` for fast normals and chunk very large path counts to bound memory. (If you ever want more speed later, `numba`'s `@njit` is a pure-Python-friendly option — but it is not required.)

**Benchmarking to document:**
- European swaption: MC vs Jamshidian analytical — should match within 0.1 vol bp at 1M paths
- Bermudan vs European: Bermudan must be >= European (early-exercise premium is non-negative)
- Speed: vectorized NumPy vs a naive Python-loop MC — expect a large speedup, document it
- Convergence: plot price standard error vs √N for standard MC, compare to Sobol

### Key Outputs
- `ird.mc` module: `price_european_swaption`, `price_bermudan_swaption`, `generate_rate_paths`
- Convergence and benchmarking notebook
- Performance comparison table (standard error, runtime, vectorized-vs-loop speedup)

---

## Phase 5: Greeks & Risk Metrics
**Duration:** Weeks 11–12

### DV01 & Key Rate Durations

**DV01 (Dollar Value of 01):**  
Shift the entire curve up by 1bp and reprice. DV01 = -(V_bumped - V_base).

**Key Rate Durations (KRDs):**  
Shift only one pillar of the curve by 1bp, keeping others fixed. Compute for pillars: 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y. This reveals which part of the curve the swaption is most sensitive to — crucial for hedging.

**Duration & Convexity (for underlying swap/bond):**
```
D = -1/P · dP/dy        (modified duration)
C = 1/P · d²P/dy²       (convexity)
ΔP ≈ -D·Δy·P + ½·C·(Δy)²·P
```

**Vega:**  
Bump σ (HW vol parameter) by 1%, reprice — captures sensitivity to swaption vol level.

**MC Pathwise Greeks:**  
For European swaptions, implement pathwise estimator for Delta (derivative of price w.r.t. initial rate r₀). More efficient than finite-differencing the MC pricer because it avoids re-running paths.

### Risk Report Output
For a sample swaption book (e.g., long 5Y expiry × 10Y tenor receiver + short 2Y × 5Y payer):
- Full KRD profile (bar chart)
- Net DV01 
- Net convexity
- Hedge ratios for a swap-based delta hedge

---

## Phase 6: Backtesting Framework
**Duration:** Weeks 13–14

### Goal
Simulate delta-hedging a swaption position through historical rate moves and decompose the resulting P&L. This directly mirrors your vol project's hedging backtest — you'll be able to draw explicit parallels in interviews.

### Setup
- **Instrument:** Long 1Y-into-5Y receiver swaption (ATM at inception)
- **Hedge:** Short the underlying swap in DV01-neutral quantity (delta hedge)
- **Period:** Jan 2019 – Dec 2024 (walk-forward, re-calibrate HW monthly)
- **Rebalancing:** Daily delta re-hedge

### Walk-Forward Calibration
On the first business day of each month:
1. Pull the current SOFR curve
2. Re-calibrate Hull-White (a, σ) to the current ATM swaption vol surface
3. Use new parameters for all pricing/Greek calculations that month

### P&L Attribution
Decompose daily P&L into:
- **Theta (carry):** P&L from time passing with rates unchanged
- **Delta P&L:** P&L from rate moves, approximated as DV01 × Δrate (first order)
- **Gamma P&L:** Second-order rate moves = ½ × Convexity × (Δrate)²
- **Vega P&L:** P&L from changes in implied vol × Vega
- **Residual:** Model error, higher-order terms

### Metrics to Report
- Total P&L of hedged portfolio (should be close to zero for a good hedge)
- Sharpe ratio of daily hedged P&L
- Max drawdown of cumulative P&L
- Hit rate (fraction of days with positive P&L post-hedge)
- RMSE of model price vs mark-to-market
- Comparison of HW vs Vasicek hedge quality (HW should win)

### Key Period Analysis
Explicitly call out performance during:
- March 2020 (COVID crash): rates fell 150bps in 2 weeks — how did the hedge hold?
- Mar 2022 – Jul 2023 (Fed hike cycle): rates rose 525bps — largest move in 40 years
- Mar 2023 (SVB crisis): short-end dislocation, curve whipsaw

---

## Phase 7: Stress Testing
**Duration:** Weeks 15–16

### Scenario Analysis

**Instantaneous parallel shifts** (apply to current curve, reprice portfolio):
- -300 bps, -200 bps, -100 bps (rally)
- +100 bps, +200 bps, +300 bps (selloff)

**Curve shape scenarios:**
- Steepening: 2s10s spread widens +100 bps (10Y up, 2Y unchanged)
- Flattening: 2s10s spread narrows -100 bps
- Bear flattening: 2Y up 100 bps, 10Y up 50 bps (typical hike cycle)
- Bull steepening: 2Y down 100 bps, 10Y down 25 bps (typical recession/easing)
- Butterfly: 5Y up 50 bps, 2Y and 10Y unchanged

**Volatility shocks:**
- ATM swaption vol +20%, +50% (vol spike, e.g., March 2020)
- ATM swaption vol -20% (vol crush, e.g., post-crisis recovery)

### Historical Scenarios (Replay)

Reconstruct the actual curve changes observed during:

| Scenario | Date | Key Move |
|----------|------|----------|
| COVID crash | Mar 16, 2020 | 2Y: -57bps single day |
| Fed emergency cut | Mar 3, 2020 | FF target cut 50bps |
| Peak rate hike | Jun 15, 2022 | +75bps hike, curve inversion |
| SVB crisis | Mar 13, 2023 | 2Y: -60bps in 2 days |
| GFC analog | Sep–Oct 2008 | Reference only (pre-SOFR) |

Apply each historical scenario's rate changes to the current portfolio and report P&L, Greeks under stress.

### Portfolio VaR & CVaR

Using MC-generated rate paths:
- Generate 10,000 1-day scenarios by sampling from historical rate changes
- Reprice portfolio under each scenario
- Compute 95% and 99% VaR (5th/1st percentile of P&L distribution)
- Compute 95% and 99% CVaR (mean loss beyond VaR threshold)
- Plot P&L distribution histogram with VaR/CVaR marked

### Reverse Stress Test
Find the minimum rate shock (in terms of 2-norm across the curve) that causes a 10% loss on the portfolio. This is implemented as a constrained optimization:
```
min  ||Δy||₂
s.t. P&L(Δy) ≤ -0.10 × Notional
```

---

## Phase 8: Validation & Documentation
**Duration:** Ongoing (complete in Week 16)

### Model Validation
- Reprice input swaption vols from calibrated HW model — residuals should be < 2 vol bps
- European swaption: compare MC prices to Jamshidian analytical (< 0.1 vol bp at 1M paths)
- Bermudan vs European: verify Bermudan ≥ European for all strikes/expiries
- Bond pricing: verify P(0,T) from HW equals bootstrapped discount factor at calibration date

### Unit Tests (pytest)
- `test_bootstrap`: reprice input swaps, verify residual < 0.1 bps
- `test_nss`: verify NSS fit RMSE < 5 bps on held-out curve
- `test_vasicek`: verify analytical bond price against known closed-form values
- `test_hw1f_tree`: trinomial tree European price == analytical within 0.5 bps
- `test_mc_convergence`: MC European swaption converges to analytical with √N rate
- `test_lsm_bermudan`: Bermudan >= European for 100 randomly sampled parameter sets
- `test_dv01`: parallel DV01 == sum of KRDs (within numerical tolerance)
- `test_pnl_attrib`: theta + delta + gamma + vega + residual == total daily P&L

### GitHub README Structure
1. Project overview and motivation (1 paragraph)
2. Architecture diagram
3. Quick start (pip install -r requirements.txt + run demo notebook)
4. Mathematical background (1 page with key equations)
5. Results summary (vol surface fit, backtest Sharpe, stress test table)
6. ATS keywords section (yes, put it in the README — recruiters skim repos)

---

## Resume Bullets (Write These Exactly)

**Project Title:** *Yield Curve Construction, Short Rate Model Calibration & Monte Carlo Swaption Pricing Engine*

> Engineered an end-to-end fixed income derivatives pricing system in pure Python (NumPy/SciPy), constructing Treasury/SOFR zero curves via bootstrapping and Nelson-Siegel-Svensson parameterization across 1,500+ historical curve dates (2018–2024).

> Calibrated Hull-White 1-factor and Vasicek short rate models to ATM swaption vol surfaces using least-squares optimization; achieved vol surface fit RMSE of [X] bps, enabling accurate swaption pricing across expiry-tenor grids.

> Built a fully vectorized NumPy Monte Carlo engine with antithetic variates, control variates, and Sobol quasi-random sequences; priced European and Bermudan swaptions (via Longstaff-Schwartz LSM), achieving [X]x speedup over a naive Python-loop implementation and convergence within 0.1 vol bp of analytical benchmarks at 1M paths.

> Computed DV01, key rate durations, duration, convexity, and Vega across a swaption book; conducted walk-forward delta-hedging backtest (2019–2024) with monthly HW recalibration, decomposing P&L into theta, delta, gamma, and vega components — maintained [X]% hedge effectiveness through the 2022 Fed rate hike cycle (+525 bps).

> Stress-tested portfolio against 15+ scenarios including parallel shifts (±300 bps), bear flattening, SVB (2023) and COVID (2020) historical replays, and vol shocks; computed 99% VaR and CVaR via MC simulation and performed reverse stress tests to identify loss-triggering rate paths.

---

## Timeline Summary

| Phase | Content | Weeks |
|-------|---------|-------|
| 1 | Data infrastructure & SOFR pipeline | 1–2 |
| 2 | Yield curve bootstrapping & NSS | 3–4 |
| 3A | Vasicek model & calibration | 5–6 |
| 3B | Hull-White 1F model & calibration | 7–8 |
| 4 | Vectorized NumPy MC engine (European + Bermudan LSM) | 9–10 |
| 5 | Greeks: DV01, KRDs, duration, convexity | 11–12 |
| 6 | Walk-forward delta-hedging backtest | 13–14 |
| 7 | Stress testing & VaR/CVaR | 15–16 |
| 8 | Validation, unit tests, GitHub docs | ongoing |

---

## What to Build Next (After This Project)

Once this is on your resume, your remaining gap is **statistical arbitrage**. A Kalman filter–based pairs trading engine (cointegration, OU process, dynamic hedge ratio, all in Python) would cover HF/prop trading roles and add: `stat arb`, `pairs trading`, `cointegration`, `Ornstein-Uhlenbeck`, `Kalman filter`, `mean reversion`. Together, the two projects make you competitive for derivatives quant, rates quant, quant research, and trading quant roles across all firm types.
