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
from ird.curve import bootstrap_curve, fit_nss, repricing_error_bps  # noqa: E402
from ird.data.fetch_sofr import build_dataset  # noqa: E402
from ird.data.validation import validate_history  # noqa: E402

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
    print(f"Saved 5 figures -> {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
