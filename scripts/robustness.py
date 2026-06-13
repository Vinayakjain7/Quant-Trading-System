"""Robustness sweep: is the 'be more selective' edge real, or one lucky cell?

Sweeps a grid of trend-filter length x RSI entry threshold, runs each setting
through the realistic engine, and reports the Sharpe AND the walk-forward
consistency for every cell. A real edge shows up as a whole green NEIGHBOURHOOD,
not a single standout square surrounded by red (that would be overfitting).

    python scripts/robustness.py

Uses cached data in data_cache/ (offline-friendly).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quant.config import load_config, Config   # noqa: E402
from quant import data as datamod               # noqa: E402
from quant import backtest as bt                # noqa: E402

TREND_GRID = [100, 150, 200, 250]
RSI_GRID = [30, 35, 40, 45]


def _cfg_with(base: Config, trend: int, rsi: int) -> Config:
    raw = copy.deepcopy(base.raw)
    raw["strategy"]["trend_ma"] = trend
    raw["strategy"]["rsi_max_entry"] = rsi
    return Config(raw=raw)


def main() -> int:
    base = load_config()
    d = base.section("data")
    price = datamod.load_universe(
        base.tickers, base.start_date, base.end_date,
        cache_dir=d.get("cache_dir", "data_cache"),
        cache_days=int(d.get("cache_days", 1)),
    )
    if not price:
        print("No price data (cache empty and no network).")
        return 1
    print(f"Loaded {len(price)} tickers. Sweeping "
          f"{len(TREND_GRID)}x{len(RSI_GRID)} = {len(TREND_GRID)*len(RSI_GRID)} settings...\n")

    sharpe = np.full((len(RSI_GRID), len(TREND_GRID)), np.nan)
    wf_mean = np.full_like(sharpe, np.nan)
    posfolds = np.zeros_like(sharpe, dtype=int)
    rows = []

    for ri, rsi in enumerate(RSI_GRID):
        for ti, trend in enumerate(TREND_GRID):
            cfg = _cfg_with(base, trend, rsi)
            returns, m, _ = bt.simulate_portfolio(price, cfg)
            wf = bt.walk_forward(returns, cfg)
            folds = wf["sharpe"].tolist() if not wf.empty else []
            sharpe[ri, ti] = m.sharpe
            wf_mean[ri, ti] = round(float(np.mean(folds)), 2) if folds else np.nan
            posfolds[ri, ti] = sum(s > 0 for s in folds)
            rows.append({"trend_ma": trend, "rsi_max": rsi, "Sharpe": m.sharpe,
                         "CAGR%": m.cagr_pct, "MaxDD%": m.max_drawdown_pct,
                         "Trades": m.num_trades, "WF_mean": wf_mean[ri, ti],
                         "posFolds": f"{posfolds[ri, ti]}/{len(folds)}"})

    table = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 160)
    print("=== ROBUSTNESS SWEEP (sorted by Sharpe) ===")
    print(table.to_string(index=False))

    # Heatmap of full-period Sharpe, annotated with Sharpe / positive-fold count.
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sharpe, cmap="RdYlGn", vmin=-0.5, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(TREND_GRID)), labels=[str(t) for t in TREND_GRID])
    ax.set_yticks(range(len(RSI_GRID)), labels=[str(r) for r in RSI_GRID])
    ax.set_xlabel("Trend MA length (longer = more selective)")
    ax.set_ylabel("RSI entry max (lower = deeper dips only)")
    ax.set_title("Sharpe by parameter setting\n(green = beats cash; look for a green CLUSTER)")
    for ri in range(len(RSI_GRID)):
        for ti in range(len(TREND_GRID)):
            ax.text(ti, ri, f"{sharpe[ri, ti]:.2f}\n{posfolds[ri, ti]}/4 OOS",
                    ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    out = ROOT / "docs" / "robustness_heatmap.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nSaved heatmap: {out}")
    if (ROOT / "outputs").exists():
        table.to_csv(ROOT / "outputs" / "robustness_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
