"""Phase 1 demo runner.

Builds the SOFR curve dataset, prints a results summary, and saves plots to
./figures/. No install needed: this script puts ./src on the path itself.

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

from ird.config import get_settings  # noqa: E402
from ird.core.conventions import tenor_to_years  # noqa: E402
from ird.data.fetch_sofr import build_dataset  # noqa: E402
from ird.data.validation import validate_history  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"

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
    ax.set_ylabel("Zero rate (%)")
    ax.set_title("SOFR curve shape across regimes")
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
    ax.set_title("SOFR rate history: 3M / 2Y / 10Y")
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

    print("\n=== Phase 1 results ===")
    print(f"Curve dates stored : {len(df):,}")
    print(f"Date range         : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Pillars            : {', '.join(df.columns)}")
    print(f"Rate range         : {df.min().min() * 100:.2f}% to {df.max().max() * 100:.2f}%")
    print(f"Inverted days      : {len(report.inverted_dates):,}")
    print(f"Validation clean   : {report.nan_cells == 0 and not report.missing_business_days}")
    latest = df.iloc[-1]
    print("\nLatest curve (%):")
    for t in df.columns:
        print(f"  {t:>4} : {latest[t] * 100:5.2f}")

    FIG_DIR.mkdir(exist_ok=True)
    plot_curve_shapes(df, FIG_DIR / "01_curve_shapes.png")
    plot_rate_history(df, FIG_DIR / "02_rate_history.png")
    plot_2s10s(df, FIG_DIR / "03_2s10s_spread.png")
    print(f"\nSaved 3 figures to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
