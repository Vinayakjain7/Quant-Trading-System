"""PnL / equity dashboard.

Reads a trades CSV (Date, Stock, Action, Price, Quantity, PnL) and renders an
interactive HTML dashboard: cumulative PnL, drawdown, and per-trade PnL.

    python -m quant.dashboard --trades outputs/trades.csv --out outputs/dashboard.html
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def build_dashboard(trades_csv: str, out_html: str) -> str:
    if not os.path.exists(trades_csv):
        raise FileNotFoundError(f"{trades_csv} not found. Run the tracker first.")

    df = pd.read_csv(trades_csv)
    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["Cumulative_PnL"] = df["PnL"].cumsum()
    df["Peak"] = df["Cumulative_PnL"].cummax()
    df["Drawdown"] = df["Cumulative_PnL"] - df["Peak"]

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Cumulative PnL", "Drawdown", "Per-trade PnL"),
    )
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Cumulative_PnL"],
                             name="Cumulative PnL", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Drawdown"], fill="tozeroy",
                             name="Drawdown"), row=2, col=1)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df["PnL"]]
    fig.add_trace(go.Bar(x=df["Date"], y=df["PnL"], marker_color=colors,
                         name="Trade PnL"), row=3, col=1)

    total = df["PnL"].sum()
    fig.update_layout(
        height=850,
        title=f"PnL Dashboard — Net PnL: {total:,.0f} | Trades: {len(df)}",
        template="plotly_white", showlegend=False,
    )
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    fig.write_html(out_html)
    return out_html


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render PnL dashboard")
    p.add_argument("--trades", default="outputs/trades.csv")
    p.add_argument("--out", default="outputs/dashboard.html")
    args = p.parse_args(argv)
    path = build_dashboard(args.trades, args.out)
    print(f"Dashboard written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
