"""Command-line entry point.

    python -m quant.cli backtest        # full backtest + walk-forward + per-ticker table
    python -m quant.cli signals         # today's ranked BUY signals with sizing
    python -m quant.cli ml              # train & evaluate the ML ranker

Run from the repo root (so config.yaml and src/ are found), or install the
package. See README for details.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd

# Allow running as `python src/quant/cli.py` too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.config import load_config            # noqa: E402
from quant import data as datamod                # noqa: E402
from quant import backtest as bt                 # noqa: E402
from quant import ml as mlmod                     # noqa: E402
from quant import risk as riskmod                 # noqa: E402
from quant.strategy import generate_positions     # noqa: E402


def _load_universe(cfg):
    d = cfg.section("data")
    return datamod.load_universe(
        cfg.tickers, cfg.start_date, cfg.end_date,
        cache_dir=d.get("cache_dir", "data_cache"),
        cache_days=int(d.get("cache_days", 1)),
    )


def cmd_backtest(cfg) -> int:
    price = _load_universe(cfg)
    if not price:
        print("No price data could be loaded (network/yfinance unavailable?).")
        return 1

    # Realistic, event-driven portfolio (top-N sized positions, cash earns rf).
    portfolio, metrics, trades = bt.simulate_portfolio(price, cfg)
    # Per-name analytics via the SAME realistic engine (comparable, trustworthy).
    per_ticker = bt.per_name_diagnostics(price, cfg)

    print("\n=== PORTFOLIO BACKTEST (top-N sized, costs, cash earns rf, no look-ahead) ===")
    for k, v in metrics.as_dict().items():
        print(f"  {k:>18}: {v}")

    wf = bt.walk_forward(portfolio, cfg)
    if not wf.empty:
        print("\n=== WALK-FORWARD (out-of-sample folds) ===")
        print(wf.to_string(index=False))

    print("\n=== PER-NAME DIAGNOSTICS (realistic engine, standalone, by Sharpe) ===")
    print(per_ticker.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    portfolio.to_csv("outputs/portfolio_returns.csv")
    per_ticker.to_csv("outputs/backtest_per_ticker.csv", index=False)
    if not trades.empty:
        trades.to_csv("outputs/trades.csv", index=False)
    print("\nSaved: outputs/portfolio_returns.csv, outputs/backtest_per_ticker.csv, outputs/trades.csv")
    print("Tip: render the equity dashboard with:")
    print("     python -m quant.dashboard --trades outputs/trades.csv --out outputs/dashboard.html")
    return 0


def cmd_ml(cfg) -> int:
    price = _load_universe(cfg)
    res = mlmod.train_ranker(price, cfg)
    if res is None:
        print("ML disabled in config, or not enough data to train.")
        return 1
    print("\n=== ML RANKER (time-ordered holdout) ===")
    print(f"  test AUC      : {res.test_auc}")
    print(f"  test accuracy : {res.test_accuracy}")
    print(f"  train / test  : {res.n_train} / {res.n_test}")
    print("  feature importance:")
    for f, v in sorted(res.feature_importance.items(), key=lambda x: -abs(x[1])):
        print(f"      {f:>12}: {v}")
    return 0


def cmd_signals(cfg) -> int:
    price = _load_universe(cfg)
    if not price:
        print("No price data could be loaded.")
        return 1

    strat_cfg = cfg.section("strategy")
    risk_cfg = cfg.section("risk")
    capital = float(risk_cfg["capital"])
    top_n = int(risk_cfg["top_n"])

    # Optional ML ranker
    ml_res = mlmod.train_ranker(price, cfg)
    model = ml_res.model if ml_res else None

    rows = []
    for ticker, df in price.items():
        if len(df) < strat_cfg["trend_ma"] + 5:
            continue
        pos_df = generate_positions(df, strat_cfg, risk_cfg)
        last = pos_df.iloc[-1]
        in_buy = bool(last["signal"] == 1) or (
            last["position"] == 1 and pos_df["signal"].iloc[-5:].sum() > 0
        )
        if last["position"] != 1:
            continue
        price_now = float(last["Close"])
        atr_now = float(last.get("atr", 0.0) or 0.0)
        size = riskmod.position_size(capital, price_now, atr_now, risk_cfg)
        ml_prob = mlmod.score_latest(model, df, strat_cfg) if model else float("nan")
        rule_score = float(last["score"])
        # Blend rule score with ML probability when available.
        combined = rule_score + (0.0 if pd.isna(ml_prob) else (ml_prob - 0.5) * 4)
        rows.append({
            "Stock": ticker,
            "Price": round(price_now, 2),
            "Qty": size.quantity,
            "Stop": size.stop_price,
            "RuleScore": round(rule_score, 3),
            "ML_Prob": None if pd.isna(ml_prob) else round(ml_prob, 3),
            "Rank": round(combined, 3),
            "SizingNote": size.reason,
        })

    today = dt.date.today().isoformat()
    if not rows:
        print(f"[{today}] No BUY candidates today.")
        return 0

    sig = pd.DataFrame(rows).sort_values("Rank", ascending=False).head(top_n)
    print(f"\n=== TOP {top_n} BUY CANDIDATES — {today} ===")
    print(sig.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    sig.insert(0, "Date", today)
    sig.to_csv(f"outputs/signals_{today}.csv", index=False)
    hist = "outputs/signals_history.csv"
    sig.to_csv(hist, mode="a", header=not os.path.isfile(hist), index=False)
    print(f"\nSaved: outputs/signals_{today}.csv (appended to signals_history.csv)")

    _notify(sig)
    return 0


def _notify(sig: pd.DataFrame) -> None:
    try:
        from plyer import notification
        msg = "\n".join(f"{r.Stock} @ {r.Price} x{r.Qty}" for r in sig.itertuples())
        notification.notify(title="Trading Signals", message=msg, timeout=10)
    except Exception:
        pass  # notifications are best-effort


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quant trading system")
    parser.add_argument("command", choices=["backtest", "signals", "ml"])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    return {"backtest": cmd_backtest, "signals": cmd_signals, "ml": cmd_ml}[args.command](cfg)


if __name__ == "__main__":
    raise SystemExit(main())
