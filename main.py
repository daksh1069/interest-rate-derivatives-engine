"""Demo runner.

Builds the Treasury/SOFR curve dataset, prints a results summary, and saves
plots to ./figures/. No install needed: this script puts ./src on the path itself.

Usage:
    python main.py                 # offline synthetic data (no API key)
    python main.py --source fred   # live FRED data (needs FRED_API_KEY in .env)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

# Make `import ird` work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib

matplotlib.use("Agg")  # headless: save files instead of opening windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ird.config import get_settings  # noqa: E402
from ird.core.conventions import tenor_to_years  # noqa: E402
from ird.core.curve_date import CurveDate  # noqa: E402
from ird.core.vol_surface import VolSurface, VolType  # noqa: E402
from ird.curve import bootstrap_curve, fit_nss, repricing_error_bps  # noqa: E402
from ird.data.fetch_sofr import build_dataset  # noqa: E402
from ird.data.validation import validate_history  # noqa: E402
from ird.models import (  # noqa: E402
    HullWhite1F,
    calibrate_hull_white,
    model_atm_normal_vol,
)
from ird.mc import (  # noqa: E402
    price_bermudan_swaption_mc,
    price_european_swaption_mc,
)
from ird.greeks import (  # noqa: E402
    dv01,
    key_rate_dv01,
    parallel_gamma,
    swaption_price_fn,
    swaption_vega,
)
from ird.backtest import (  # noqa: E402
    hit_rate,
    max_drawdown,
    run_delta_hedge_backtest,
    sharpe,
)
from ird.stress import (  # noqa: E402
    Portfolio,
    Position,
    historical_var,
    reverse_stress_parallel,
    run_curve_scenarios,
    run_historical_scenarios,
    run_vol_scenarios,
)

# A representative ATM swaption normal-vol surface (basis points), expiry x tenor.
# Stands in for a market quote sheet; swap in real data when available.
SWAPTION_EXPIRIES = ["1Y", "2Y", "5Y", "10Y"]
SWAPTION_TENORS = ["2Y", "5Y", "10Y"]
MARKET_VOL_BP = [
    [116, 108, 98],
    [112, 104, 94],
    [101, 93, 85],
    [86, 80, 74],
]

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

# Representative dates -> regime label, for the curve-shape plot.
REGIMES = {
    "2018-02-01": "Pre-hike 2018",
    "2020-03-23": "COVID zero lower bound",
    "2022-12-14": "Peak hikes (inverted)",
    "2023-03-24": "SVB whipsaw",
}


def plot_curve_shapes(df, out: Path) -> None:
    tenors = list(df.columns)
    xs = [tenor_to_years(t) for t in tenors]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for date, label in REGIMES.items():
        row = df.loc[:date].iloc[-1]
        ax.plot(xs, row.values * 100, marker="o", label=f"{label} ({row.name.date()})")
    latest = df.iloc[-1]
    ax.plot(xs, latest.values * 100, marker="o", linewidth=2.5,
            label=f"Latest ({latest.name.date()})")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(tenors)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Par yield (%)")
    ax.set_title("U.S. Treasury par curve shape across regimes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_rate_history(df, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for tenor in ("3M", "2Y", "10Y"):
        ax.plot(df.index, df[tenor] * 100, label=tenor, linewidth=1.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rate (%)")
    ax.set_title("U.S. Treasury rate history: 3M / 2Y / 10Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_2s10s(df, out: Path) -> None:
    spread = (df["10Y"] - df["2Y"]) * 100
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df.index, spread, color="#534AB7", linewidth=1.2)
    ax.axhline(0, color="#D85A30", linewidth=1)
    ax.fill_between(df.index, spread, 0, where=(spread < 0),
                    color="#D85A30", alpha=0.25, label="inverted")
    ax.set_xlabel("Date")
    ax.set_ylabel("10Y - 2Y (%)")
    ax.set_title("2s10s slope (curve inversion below zero)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_zero_forward(curve, nss, grid, out: Path) -> None:
    zero = np.atleast_1d(curve.zero_rate(grid)) * 100
    fwd = np.atleast_1d(curve.instantaneous_forward(grid)) * 100
    nss_z = np.atleast_1d(nss.zero_rate(grid)) * 100
    pz = np.atleast_1d(curve.zero_rate(curve.pillars)) * 100
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(grid, zero, color="#185FA5", label="Zero rate", linewidth=2)
    ax.plot(grid, fwd, color="#D85A30", label="Instantaneous forward", linewidth=2, ls="--")
    ax.plot(grid, nss_z, color="#534AB7", label="NSS fit", linewidth=2, ls=":")
    ax.scatter(curve.pillars, pz, color="#185FA5", zorder=5, label="Bootstrapped pillars")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Bootstrapped zero / forward / NSS curve ({curve.date})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_discount_factors(curve, grid, out: Path) -> None:
    dfs = np.atleast_1d(curve.discount_factor(grid))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(grid, dfs, color="#1D9E75", linewidth=2)
    ax.fill_between(grid, dfs, 0, color="#1D9E75", alpha=0.08)
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Discount factor P(0,T)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Discount factor curve ({curve.date})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def build_curve_summary(curve, nss, errs: dict[str, float]) -> str:
    """Reproducible metrics for the bootstrapped curve and NSS fit."""
    sample_T = [t for t in (0.25, 1, 2, 5, 10, 30) if t <= curve.pillars[-1]]
    lines = [
        "=" * 56,
        "Yield curve construction metrics",
        "=" * 56,
        f"Curve date         : {curve.date}",
        f"Interpolation      : {curve.method}",
        f"Pillars            : {len(curve.pillars)}",
        f"Max reprice error  : {max(abs(e) for e in errs.values()):.4f} bps",
        "",
        "--- Nelson-Siegel-Svensson fit ---",
        f"RMSE               : {nss.rmse_bps:.3f} bps",
        f"level  b0          : {nss.beta0 * 100:.3f}%",
        f"slope  b1          : {nss.beta1 * 100:.3f}%",
        f"curv1  b2          : {nss.beta2 * 100:.3f}%",
        f"curv2  b3          : {nss.beta3 * 100:.3f}%",
        f"decays (L1, L2)    : ({nss.lam1:.2f}, {nss.lam2:.2f})",
        "",
        "--- Sample curve (zero / inst. forward / discount factor) ---",
    ]
    for T in sample_T:
        z = float(np.atleast_1d(curve.zero_rate(T))[0]) * 100
        f = float(np.atleast_1d(curve.instantaneous_forward(T))[0]) * 100
        d = float(np.atleast_1d(curve.discount_factor(T))[0])
        lines.append(f"  {T:>5.2f}y : zero {z:6.3f}%   fwd {f:6.3f}%   df {d:.5f}")
    lines.append("")
    return "\n".join(lines)


def plot_vol_surface(market_bp, model_bp, expiries, tenors, out: Path) -> None:
    market = np.array(market_bp, float)
    model = np.array(model_bp, float)
    err = model - market
    grids = [("Market (bp)", market, "Blues"),
             ("Hull-White (bp)", model, "Blues"),
             ("Error: model - market (bp)", err, "RdBu_r")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (title, data, cmap) in zip(axes, grids):
        vlim = max(abs(err.min()), abs(err.max())) if cmap == "RdBu_r" else None
        im = ax.imshow(data, cmap=cmap, aspect="auto",
                       vmin=-vlim if vlim else None, vmax=vlim if vlim else None)
        ax.set_xticks(range(len(tenors)), tenors)
        ax.set_yticks(range(len(expiries)), expiries)
        ax.set_xlabel("Swap tenor")
        ax.set_ylabel("Option expiry")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                        fontsize=9, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Hull-White calibration to ATM swaption vol surface")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_stress_scenarios(scen_bp, out: Path) -> None:
    names = list(scen_bp)
    vals = [scen_bp[n] for n in names]
    colors = ["#185FA5" if v >= 0 else "#D85A30" for v in vals]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Portfolio P&L (bp)")
    ax.set_title("Stress scenario P&L")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_var_cvar(vr, out: Path) -> None:
    pnl_bp = vr.pnl * 1e4
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pnl_bp, bins=60, color="#185FA5", alpha=0.75)
    ax.axvline(-vr.var95 * 1e4, color="#D85A30", ls="--", label=f"VaR95 {vr.var95*1e4:.1f}bp")
    ax.axvline(-vr.var99 * 1e4, color="#A32D2D", ls="--", label=f"VaR99 {vr.var99*1e4:.1f}bp")
    ax.set_xlabel("1-day P&L (bp)")
    ax.set_ylabel("Frequency")
    ax.set_title("1-day P&L distribution with VaR (historical simulation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def build_stress_summary(curve_s, vol_s, hist_s, vr, rstress, base_bp) -> str:
    """Reproducible stress-testing report."""
    lines = ["=" * 56, "Stress testing report", "=" * 56,
             f"Portfolio base value : {base_bp:+.2f} bp", "",
             "--- Curve scenarios (P&L, bp) ---"]
    for k, v in curve_s.items():
        lines.append(f"  {k:<18}: {v * 1e4:+9.2f}")
    lines.append("--- Volatility shocks (P&L, bp) ---")
    for k, v in vol_s.items():
        lines.append(f"  {k:<18}: {v * 1e4:+9.2f}")
    lines.append("--- Historical replays (P&L, bp) ---")
    for k, v in hist_s.items():
        lines.append(f"  {k:<26}: {v * 1e4:+9.2f}")
    lines += ["", "--- VaR / CVaR (1-day, bp; historical simulation) ---",
              f"  VaR 95%  : {vr.var95 * 1e4:7.2f}", f"  VaR 99%  : {vr.var99 * 1e4:7.2f}",
              f"  CVaR 95% : {vr.cvar95 * 1e4:7.2f}", f"  CVaR 99% : {vr.cvar99 * 1e4:7.2f}",
              "", "--- Reverse stress (parallel shock for 10% loss) ---",
              f"  shock : {rstress['shock_bp']:+.0f} bp -> loss {rstress['loss'] * 1e4:.1f} bp", ""]
    return "\n".join(lines)


def plot_backtest_pnl(frame, out: Path) -> None:
    f = frame.dropna(subset=["hedged_pnl"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f.index, np.cumsum(f["unhedged_pnl"]) * 1e4, color="#D85A30",
            label="Unhedged swaption", linewidth=1.5)
    ax.plot(f.index, np.cumsum(f["hedged_pnl"]) * 1e4, color="#1D9E75",
            label="Delta-hedged", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (bp)")
    ax.set_title("Delta-hedging backtest: hedged vs unhedged P&L")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_pnl_attribution(frame, out: Path) -> None:
    f = frame.dropna(subset=["dV"])
    comps = {"theta": f["theta"].sum(), "delta": f["delta"].sum(),
             "gamma": f["gamma_pnl"].sum(), "vega": f["vega_pnl"].sum(),
             "residual": f["residual"].sum()}
    vals = [v * 1e4 for v in comps.values()]
    colors = ["#185FA5" if v >= 0 else "#D85A30" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(comps), vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative P&L contribution (bp)")
    ax.set_title("Unhedged swaption P&L attribution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def build_backtest_summary(res) -> str:
    """Reproducible delta-hedging backtest report."""
    s = res.summary
    f = res.frame.dropna(subset=["hedged_pnl"])
    hp = f["hedged_pnl"].to_numpy()
    fa = res.frame.dropna(subset=["dV"])
    return "\n".join([
        "=" * 56,
        "Delta-hedging backtest (1Y x 5Y receiver swaption)",
        "=" * 56,
        f"Window             : {s['inception']} -> {s['expiry_date']} ({s['n_days']} days)",
        f"Hull-White         : a={res.a:.4f}, sigma={res.sigma * 1e4:.0f} bp",
        "",
        f"Unhedged P&L total : {s['unhedged_pnl_total'] * 1e4:+.2f} bp  "
        f"(daily std {s['unhedged_pnl_std'] * 1e4:.3f} bp)",
        f"Hedged P&L total   : {s['hedged_pnl_total'] * 1e4:+.2f} bp  "
        f"(daily std {s['hedged_pnl_std'] * 1e4:.3f} bp)",
        f"Variance reduction : {s['variance_reduction']:.1f}x",
        f"Hedged Sharpe      : {sharpe(hp):.2f}",
        f"Hedged max drawdown: {max_drawdown(np.cumsum(hp)) * 1e4:.2f} bp",
        f"Hedged hit rate    : {hit_rate(hp):.1%}",
        "",
        "--- Unhedged P&L attribution (cumulative, bp) ---",
        f"  theta    : {fa['theta'].sum() * 1e4:+.2f}",
        f"  delta    : {fa['delta'].sum() * 1e4:+.2f}",
        f"  gamma    : {fa['gamma_pnl'].sum() * 1e4:+.2f}",
        f"  vega     : {fa['vega_pnl'].sum() * 1e4:+.2f}",
        f"  residual : {fa['residual'].sum() * 1e4:+.2f}",
        f"  mean|resid|/day : {s['mean_abs_residual_bp']:.3f} bp",
        "",
    ])


def plot_krd_profile(tenors, krd_bp, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#185FA5" if v >= 0 else "#D85A30" for v in krd_bp]
    ax.bar([str(t) for t in tenors], krd_bp, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Curve pillar (years)")
    ax.set_ylabel("Key-rate DV01 (per 1bp, x1e4)")
    ax.set_title("Swaption book key-rate DV01 profile")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def build_risk_summary(book, net_dv01, net_vega, net_gamma, krd) -> str:
    """Reproducible risk report for the sample swaption book."""
    lines = [
        "=" * 56,
        "Swaption book risk report (bump-and-reprice)",
        "=" * 56,
        "Positions:",
    ]
    for desc, price in book:
        lines.append(f"  {desc:<34} PV = {price * 1e4:8.2f} bp")
    lines += [
        "",
        f"Net DV01           : {net_dv01 * 1e4:+.3f}  (x1e4, per 1bp parallel)",
        f"Net curve convexity: {net_gamma:+.4e}  (d2V/dy2)",
        f"Net vega           : {net_vega * 1e4:+.3f}  (x1e4, per 1bp sigma)",
        "",
        "--- Net key-rate DV01 profile (x1e4 per 1bp) ---",
    ]
    for t, v in krd.items():
        lines.append(f"  {t:>6.3f}y : {v * 1e4:+.4f}")
    lines.append("")
    return "\n".join(lines)


def plot_mc_convergence(ns, plain, cv, qmc, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(ns, plain, "o-", color="#185FA5", label="Standard MC")
    ax.loglog(ns, cv, "s--", color="#1D9E75", label="+ control variate")
    ax.loglog(ns, qmc, "^:", color="#534AB7", label="Quasi-MC (Halton)")
    ax.set_xlabel("Paths (N)")
    ax.set_ylabel("Absolute pricing error (bp)")
    ax.set_title("Monte Carlo swaption convergence vs Jamshidian analytic")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def build_mc_summary(analytic, mc, berm, ns, qmc_err) -> str:
    """Reproducible Monte Carlo pricing report (2Y x 5Y ATM payer)."""
    return "\n".join([
        "=" * 56,
        "Monte Carlo swaption pricing (2Y x 5Y ATM payer)",
        "=" * 56,
        f"Jamshidian analytic : {analytic * 1e4:.2f} bp",
        f"MC (200k, CV)       : {mc.price * 1e4:.2f} bp  "
        f"(+/- {mc.std_error * 1e4:.3f} bp, err {abs(mc.price - analytic) * 1e4:.3f} bp)",
        f"QMC error @ {ns[-1]:>6} : {qmc_err * 1e4:.4f} bp",
        "",
        "--- Bermudan vs European (Longstaff-Schwartz) ---",
        f"European            : {analytic * 1e4:.2f} bp",
        f"Bermudan (LSM)      : {berm.price * 1e4:.2f} bp (+/- {berm.std_error * 1e4:.3f} bp)",
        f"Early-exercise prem : {(berm.price - analytic) * 1e4:.1f} bp",
        "",
    ])


def build_hw_summary(res, market_bp, model_bp, expiries, tenors) -> str:
    """Reproducible Hull-White calibration report."""
    err = np.array(model_bp) - np.array(market_bp)
    lines = [
        "=" * 56,
        "Hull-White calibration to ATM swaption vol surface",
        "=" * 56,
        f"Mean reversion a   : {res.a:.4f}",
        f"Volatility sigma   : {res.sigma * 1e4:.1f} bp ({res.sigma:.5f})",
        f"Fit RMSE           : {res.rmse_bps:.2f} bp",
        f"Max |error|        : {np.abs(err).max():.2f} bp",
        "",
        "Grid: rows = expiry, cols = tenor " + str(tenors),
        "--- Market vols (bp) ---",
    ]
    for i, e in enumerate(expiries):
        lines.append(f"  {e:>4}: " + "  ".join(f"{v:5.1f}" for v in market_bp[i]))
    lines.append("--- Model vols (bp) ---")
    for i, e in enumerate(expiries):
        lines.append(f"  {e:>4}: " + "  ".join(f"{v:5.1f}" for v in model_bp[i]))
    lines.append("")
    return "\n".join(lines)


def build_summary(df, report, source: str, has_key: bool) -> str:
    """Assemble a human-readable, reproducible metrics report as a string."""
    spread = (df["10Y"] - df["2Y"]) * 100
    latest = df.iloc[-1]
    lines = [
        "=" * 56,
        "IRD Pricing Engine - data pipeline metrics",
        "=" * 56,
        f"Generated         : {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Data source        : {source}  (FRED key detected: {has_key})",
        "Instrument         : U.S. Treasury par yields (FRED CMT/DGS series),",
        "                     used as a public proxy for the SOFR OIS curve.",
        "",
        "--- Dataset ---",
        f"Curve dates stored : {len(df):,}",
        f"Date range         : {df.index.min().date()} -> {df.index.max().date()}",
        f"Pillars            : {', '.join(df.columns)}",
        f"Rate range         : {df.min().min() * 100:.2f}% to {df.max().max() * 100:.2f}%",
        "",
        "--- Validation ---",
        f"NaN cells              : {report.nan_cells}",
        f"Missing bus. days      : {len(report.missing_business_days)}",
        f">150bp jump flags      : {len(report.large_jumps)}",
        f"Days w/ any inversion  : {len(report.inverted_dates):,}  "
        "(any adjacent pillar dip, anywhere on the curve)",
        f"Days w/ 2s10s inverted : {int((spread < 0).sum()):,}  (10Y below 2Y)",
        f"Clean                  : "
        f"{report.nan_cells == 0 and not report.missing_business_days}",
        "",
        "--- 2s10s slope (10Y - 2Y, %) ---",
        f"Min (most inverted): {spread.min():.2f}  on {spread.idxmin().date()}",
        f"Max (steepest)     : {spread.max():.2f}  on {spread.idxmax().date()}",
        f"Latest             : {spread.iloc[-1]:.2f}",
        "",
        "--- Per-tenor (mean / min / max / latest, %) ---",
    ]
    for t in df.columns:
        col = df[t] * 100
        lines.append(
            f"  {t:>4} : mean {col.mean():5.2f}   min {col.min():5.2f}   "
            f"max {col.max():5.2f}   latest {latest[t] * 100:5.2f}"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "fred", "synthetic"], default="auto")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args(argv)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else None

    settings = get_settings()
    print(f"FRED key detected: {settings.has_fred_key} | source={args.source}")

    store = build_dataset(source=args.source, start=start, end=end, settings=settings)
    df = store.read_history()
    report = validate_history(df)

    summary = build_summary(df, report, args.source, settings.has_fred_key)
    print("\n" + summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics.txt"
    metrics_path.write_text(summary)
    # Machine-readable copy of the full curve history for reproducibility.
    df.to_csv(RESULTS_DIR / "curve_history.csv", index_label="date")

    FIG_DIR.mkdir(exist_ok=True)
    plot_curve_shapes(df, FIG_DIR / "01_curve_shapes.png")
    plot_rate_history(df, FIG_DIR / "02_rate_history.png")
    plot_2s10s(df, FIG_DIR / "03_2s10s_spread.png")

    # Bootstrap the latest curve, fit NSS, save curve outputs.
    latest_row = df.iloc[-1]
    latest_cd = CurveDate(
        latest_row.name.date(), {t: float(latest_row[t]) for t in df.columns}
    )
    curve = bootstrap_curve(latest_cd, method="monotone_convex")
    errs = repricing_error_bps(curve, latest_cd)
    nss = fit_nss(curve.pillars, np.atleast_1d(curve.zero_rate(curve.pillars)))

    grid = np.linspace(0.08, float(curve.pillars[-1]), 200)
    plot_zero_forward(curve, nss, grid, FIG_DIR / "04_zero_forward_curve.png")
    plot_discount_factors(curve, grid, FIG_DIR / "05_discount_factors.png")

    curve_summary = build_curve_summary(curve, nss, errs)
    print("\n" + curve_summary)
    (RESULTS_DIR / "curve_metrics.txt").write_text(curve_summary)

    # Calibrate Hull-White to the ATM swaption vol surface.
    market = [[v / 1e4 for v in row] for row in MARKET_VOL_BP]
    surface = VolSurface(curve.date, SWAPTION_EXPIRIES, SWAPTION_TENORS,
                         market, VolType.NORMAL)
    hw_res = calibrate_hull_white(curve, surface, freq=2)
    hw = HullWhite1F(hw_res.a, hw_res.sigma, curve)
    model_bp = [
        [round(model_atm_normal_vol(hw, tenor_to_years(e), tenor_to_years(t), 2) * 1e4, 1)
         for t in SWAPTION_TENORS]
        for e in SWAPTION_EXPIRIES
    ]
    plot_vol_surface(MARKET_VOL_BP, model_bp, SWAPTION_EXPIRIES, SWAPTION_TENORS,
                     FIG_DIR / "06_vol_surface_calibration.png")
    hw_summary = build_hw_summary(hw_res, MARKET_VOL_BP, model_bp,
                                  SWAPTION_EXPIRIES, SWAPTION_TENORS)
    print("\n" + hw_summary)
    (RESULTS_DIR / "hw_calibration.txt").write_text(hw_summary)

    # Monte Carlo: convergence vs analytic, variance reduction, Bermudan premium.
    mc_expiry, mc_tenor = 2.0, 5.0
    analytic = hw.swaption_price(mc_expiry, mc_tenor, None, freq=2, payer=True)
    ns = [256, 1024, 4096, 16384]
    plain = [abs(price_european_swaption_mc(hw, mc_expiry, mc_tenor, n_paths=n,
             control_variate=False, antithetic=False, seed=0).price - analytic) * 1e4
             for n in ns]
    cv = [abs(price_european_swaption_mc(hw, mc_expiry, mc_tenor, n_paths=n,
          control_variate=True, seed=0).price - analytic) * 1e4 for n in ns]
    qmc = [abs(price_european_swaption_mc(hw, mc_expiry, mc_tenor, n_paths=n,
           method="qmc").price - analytic) * 1e4 for n in ns]
    plot_mc_convergence(ns, plain, cv, qmc, FIG_DIR / "07_mc_convergence.png")

    mc_best = price_european_swaption_mc(hw, mc_expiry, mc_tenor, n_paths=200_000,
                                         control_variate=True, seed=0)
    berm = price_bermudan_swaption_mc(hw, mc_expiry, mc_tenor, n_paths=40_000, seed=0)
    mc_summary = build_mc_summary(analytic, mc_best, berm, ns, qmc[-1] / 1e4)
    print("\n" + mc_summary)
    (RESULTS_DIR / "mc_pricing.txt").write_text(mc_summary)

    # Greeks & risk: a sample swaption book (long 5Yx10Y receiver, short 2Yx5Y payer).
    a_c, sig_c = hw_res.a, hw_res.sigma
    k1 = HullWhite1F(a_c, sig_c, curve).forward_swap_rate(5.0, 10.0, 2)
    k2 = HullWhite1F(a_c, sig_c, curve).forward_swap_rate(2.0, 5.0, 2)
    positions = [  # (sign, description, price_fn, vega_args)
        (+1, "long  5Yx10Y receiver", swaption_price_fn(a_c, sig_c, 5.0, 10.0, k1, 2, False),
         (5.0, 10.0, k1, 2, False)),
        (-1, "short 2Yx5Y payer", swaption_price_fn(a_c, sig_c, 2.0, 5.0, k2, 2, True),
         (2.0, 5.0, k2, 2, True)),
    ]
    book = [(d, sgn * pf(curve)) for sgn, d, pf, _ in positions]
    net_dv01 = sum(sgn * dv01(pf, curve) for sgn, _, pf, _ in positions)
    net_gamma = sum(sgn * parallel_gamma(pf, curve) for sgn, _, pf, _ in positions)
    net_vega = sum(sgn * swaption_vega(curve, a_c, sig_c, *va) for sgn, _, _, va in positions)
    krd_net: dict[float, float] = {}
    for sgn, _, pf, _ in positions:
        for t, v in key_rate_dv01(pf, curve).items():
            krd_net[t] = krd_net.get(t, 0.0) + sgn * v
    plot_krd_profile(list(krd_net), [v * 1e4 for v in krd_net.values()],
                     FIG_DIR / "08_krd_profile.png")
    risk_summary = build_risk_summary(book, net_dv01, net_vega, net_gamma, krd_net)
    print("\n" + risk_summary)
    (RESULTS_DIR / "risk_report.txt").write_text(risk_summary)

    # Walk-forward delta-hedging backtest (1Y x 5Y receiver), over a 1-year window.
    last = df.index[-1]
    target = pd.Timestamp(last) - pd.Timedelta(days=400)
    inception = df.index[df.index.get_indexer([target], method="nearest")[0]]
    bt = run_delta_hedge_backtest(df, inception, a=a_c, sigma=sig_c,
                                  expiry=1.0, tenor=5.0, payer=False)
    plot_backtest_pnl(bt.frame, FIG_DIR / "09_backtest_pnl.png")
    plot_pnl_attribution(bt.frame, FIG_DIR / "10_pnl_attribution.png")
    bt_summary = build_backtest_summary(bt)
    print("\n" + bt_summary)
    (RESULTS_DIR / "backtest_report.txt").write_text(bt_summary)

    # Stress testing: same book under scenarios, historical replays, VaR/CVaR.
    pf = Portfolio(
        [Position(+1, 5.0, 10.0, k1, 2, False), Position(-1, 2.0, 5.0, k2, 2, True)],
        a_c, sig_c,
    )
    curve_s = run_curve_scenarios(pf, curve)
    vol_s = run_vol_scenarios(pf, curve)
    hist_s = run_historical_scenarios(pf, curve)
    vr = historical_var(pf, curve, df, n_samples=2000, seed=0)
    gross = sum(abs(p.sign) * abs(HullWhite1F(a_c, sig_c, curve).swaption_price(
        p.expiry, p.tenor, p.strike, p.freq, p.payer)) for p in pf.positions)
    rstress = reverse_stress_parallel(pf, curve, target_loss=0.10 * gross)
    scen_for_plot = {**{k: v * 1e4 for k, v in curve_s.items()},
                     **{k: v * 1e4 for k, v in hist_s.items()}}
    plot_stress_scenarios(scen_for_plot, FIG_DIR / "11_stress_scenarios.png")
    plot_var_cvar(vr, FIG_DIR / "12_var_cvar.png")
    stress_summary = build_stress_summary(curve_s, vol_s, hist_s, vr, rstress,
                                          pf.value(curve) * 1e4)
    print("\n" + stress_summary)
    (RESULTS_DIR / "stress_report.txt").write_text(stress_summary)
    # Machine-readable bootstrapped curve over a maturity grid.
    pd.DataFrame(
        {
            "maturity_yrs": grid,
            "zero_rate": np.atleast_1d(curve.zero_rate(grid)),
            "fwd_rate": np.atleast_1d(curve.instantaneous_forward(grid)),
            "nss_zero": np.atleast_1d(nss.zero_rate(grid)),
            "discount_factor": np.atleast_1d(curve.discount_factor(grid)),
        }
    ).to_csv(RESULTS_DIR / "bootstrapped_curve.csv", index=False)

    print(f"\nSaved metrics  -> {metrics_path}")
    print(f"Saved data     -> {RESULTS_DIR / 'curve_history.csv'}")
    print(f"Saved curve    -> {RESULTS_DIR / 'curve_metrics.txt'}, "
          f"{RESULTS_DIR / 'bootstrapped_curve.csv'}")
    print(f"Saved HW calib -> {RESULTS_DIR / 'hw_calibration.txt'}")
    print(f"Saved MC       -> {RESULTS_DIR / 'mc_pricing.txt'}")
    print(f"Saved risk     -> {RESULTS_DIR / 'risk_report.txt'}")
    print(f"Saved backtest -> {RESULTS_DIR / 'backtest_report.txt'}")
    print(f"Saved stress   -> {RESULTS_DIR / 'stress_report.txt'}")
    print(f"Saved 12 figures -> {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
