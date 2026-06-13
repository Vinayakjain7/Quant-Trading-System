"""Compare strategy rule variations on a level playing field.

Runs several rule sets through the SAME realistic engine (top-N sizing, costs,
cash earns risk-free, no look-ahead) and prints a side-by-side table plus the
walk-forward Sharpe per out-of-sample fold. The point: find out whether any
variation has a real edge, judged out-of-sample — not by one in-sample number.

    python scripts/experiment.py

Uses cached data in data_cache/ when available (offline-friendly).
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


# Each variation is a set of overrides applied on top of config.yaml.
# Keys: "strategy" and/or "risk".
VARIATIONS: dict[str, dict] = {
    "baseline (current)": {},
    "let_winners_run":    {"strategy": {"exit_target": "upper"}},
    "looser_entry":       {"strategy": {"rsi_max_entry": 55, "bb_std": 2.0}},
    "tighter_trend_rsi":  {"strategy": {"trend_ma": 200, "rsi_max_entry": 35}},
    "wider_atr_stop":     {"risk": {"atr_stop_mult": 3.0}},
    "winners+looser":     {"strategy": {"exit_target": "upper", "rsi_max_entry": 55,
                                         "bb_std": 2.0}},
}


def _cfg_with(base: Config, overrides: dict) -> Config:
    raw = copy.deepcopy(base.raw)
    for section, vals in overrides.items():
        raw.setdefault(section, {}).update(vals)
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
        print("No price data (cache empty and no network). Run a backtest once online first.")
        return 1
    print(f"Loaded {len(price)} tickers from {base.start_date} to {base.end_date}\n")

    rows = []
    wf_by_variation: dict[str, pd.DataFrame] = {}
    for name, ov in VARIATIONS.items():
        cfg = _cfg_with(base, ov)
        returns, m, trades = bt.simulate_portfolio(price, cfg)
        wf = bt.walk_forward(returns, cfg)
        wf_by_variation[name] = wf
        fold_sharpes = wf["sharpe"].tolist() if not wf.empty else []
        rows.append({
            "variation": name,
            "Sharpe": m.sharpe,
            "CAGR%": m.cagr_pct,
            "MaxDD%": m.max_drawdown_pct,
            "Win%": m.win_rate_pct,
            "Trades": m.num_trades,
            "WF_minSharpe": round(min(fold_sharpes), 2) if fold_sharpes else float("nan"),
            "WF_meanSharpe": round(float(np.mean(fold_sharpes)), 2) if fold_sharpes else float("nan"),
            "WF_posFolds": f"{sum(s > 0 for s in fold_sharpes)}/{len(fold_sharpes)}" if fold_sharpes else "-",
        })

    table = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 160)
    print("=== RULE VARIATION COMPARISON (realistic engine) ===")
    print(table.to_string(index=False))
    print("\nWF_posFolds = how many out-of-sample folds had a POSITIVE Sharpe.")
    print("A variation only deserves trust if it beats cash (Sharpe > 0) across MOST folds.\n")

    # Chart: walk-forward Sharpe per fold, one line per variation.
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, wf in wf_by_variation.items():
        if wf.empty:
            continue
        ax.plot(wf["fold"], wf["sharpe"], marker="o", label=name)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Walk-Forward Sharpe by Out-of-Sample Fold")
    ax.set_xlabel("Fold (chronological)")
    ax.set_ylabel("Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = ROOT / "docs" / "experiment_walkforward.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved chart: {out}")

    table.to_csv(ROOT / "outputs" / "experiment_results.csv", index=False) if (ROOT / "outputs").exists() else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
