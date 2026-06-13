"""Generate a results chart for the README.

Runs the real backtest over the configured universe and saves a 3-panel PNG
(equity curve, drawdown, walk-forward Sharpe by fold) to docs/sample_backtest.png.

    python scripts/make_results_chart.py

If price data can't be fetched (offline / no network), it falls back to a
synthetic series and clearly labels the chart "ILLUSTRATIVE — synthetic data"
so a fake number is never passed off as a real backtest result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quant.config import load_config           # noqa: E402
from quant import data as datamod               # noqa: E402
from quant import backtest as bt                # noqa: E402


def _synthetic_universe(n_tickers: int = 8, n: int = 750) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-01", periods=n)
    out = {}
    for k in range(n_tickers):
        steps = rng.normal(0.0004, 0.016, n)
        close = 100 * np.exp(np.cumsum(steps))
        out[f"SYN{k}"] = pd.DataFrame({
            "Open": close,
            "High": close * (1 + rng.uniform(0, 0.012, n)),
            "Low": close * (1 - rng.uniform(0, 0.012, n)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        }, index=dates)
    return out


def main() -> int:
    cfg = load_config()
    d = cfg.section("data")
    price = datamod.load_universe(
        cfg.tickers, cfg.start_date, cfg.end_date,
        cache_dir=d.get("cache_dir", "data_cache"),
        cache_days=int(d.get("cache_days", 1)),
    )

    illustrative = False
    if not price:
        print("No live data available — using synthetic series (clearly labelled).")
        price = _synthetic_universe()
        illustrative = True

    portfolio, metrics, _trades = bt.simulate_portfolio(price, cfg)
    wf = bt.walk_forward(portfolio, cfg)
    equity = (1 + portfolio.fillna(0)).cumprod()
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max * 100

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [3, 2, 2]})

    axes[0].plot(equity.index, equity.values, color="#1f77b4", lw=1.8)
    axes[0].set_title("Portfolio Equity Curve (net of costs, no look-ahead)")
    axes[0].set_ylabel("Growth of 1.0")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    axes[1].set_title("Drawdown (%)")
    axes[1].grid(alpha=0.3)

    if not wf.empty:
        axes[2].bar(wf["fold"].astype(str), wf["sharpe"], color="#2ca02c", alpha=0.8)
        axes[2].axhline(0, color="black", lw=0.8)
        axes[2].set_title("Walk-Forward Sharpe by Out-of-Sample Fold")
        axes[2].set_xlabel("Fold")

    subtitle = (
        f"Sharpe {metrics.sharpe} | CAGR {metrics.cagr_pct}% | "
        f"MaxDD {metrics.max_drawdown_pct}% | WinRate {metrics.win_rate_pct}% | "
        f"Trades {metrics.num_trades}"
    )
    sup = "ILLUSTRATIVE — synthetic data (run with live data for real results)" if illustrative else subtitle
    fig.suptitle(sup, fontsize=11, y=0.995,
                 color=("#d62728" if illustrative else "black"))
    if illustrative:
        fig.text(0.5, 0.965, subtitle, ha="center", fontsize=9, color="#555")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sample_backtest.png"
    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")
    print(subtitle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
