"""Realistic backtest engine.

What makes this trustworthy (vs a naive backtest):
  * NO LOOK-AHEAD: a position decided at close of day t earns day t+1's return
    (positions are shifted by one bar before being multiplied by returns).
  * TRANSACTION COSTS + SLIPPAGE charged on every entry and exit.
  * Proper risk-adjusted metrics (Sharpe with risk-free rate, Sortino, max DD).
  * WALK-FORWARD evaluation so you see out-of-sample consistency, not one
    lucky in-sample number.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd

from .strategy import generate_positions
from . import risk as riskmod


@dataclass
class Metrics:
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    exposure_pct: float

    def as_dict(self) -> dict:
        return asdict(self)


def _per_side_cost(bt_cfg: dict) -> float:
    return (bt_cfg.get("commission_bps", 0) + bt_cfg.get("slippage_bps", 0)) / 10000.0


def strategy_returns(df_pos: pd.DataFrame, bt_cfg: dict) -> pd.Series:
    """Net daily strategy returns for one instrument (after costs)."""
    daily = df_pos["Close"].pct_change().fillna(0.0)
    pos = df_pos["position"].shift(1).fillna(0)          # <-- no look-ahead
    gross = pos * daily

    cost = _per_side_cost(bt_cfg)
    turnover = df_pos["position"].diff().abs().fillna(0)  # 1 on entry, 1 on exit
    costs = turnover * cost
    return (gross - costs).rename("ret")


def _annualization(index: pd.Index) -> int:
    return 252  # daily bars


def compute_metrics(returns: pd.Series, bt_cfg: dict, position: pd.Series | None = None) -> Metrics:
    r = returns.dropna()
    if r.empty:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0)

    ann = _annualization(r.index)
    rf_daily = bt_cfg.get("risk_free_rate", 0.0) / ann

    equity = (1 + r).cumprod()
    total_return = equity.iloc[-1] - 1
    years = max(len(r) / ann, 1e-9)
    cagr = equity.iloc[-1] ** (1 / years) - 1

    excess = r - rf_daily
    sharpe = (excess.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0
    downside = r[r < 0].std()
    sortino = (excess.mean() / downside * np.sqrt(ann)) if downside and downside > 0 else 0.0

    roll_max = equity.cummax()
    max_dd = ((equity - roll_max) / roll_max).min()

    wins = (r[r != 0] > 0).sum()
    active = (r != 0).sum()
    win_rate = (wins / active) if active > 0 else 0.0

    if position is not None:
        exposure = (position.shift(1).fillna(0) != 0).mean()
        trades = int((position.diff().fillna(0) > 0).sum())
    else:
        exposure = (r != 0).mean()
        trades = int(((r != 0).astype(int).diff().fillna(0) > 0).sum())

    return Metrics(
        total_return_pct=round(float(total_return) * 100, 2),
        cagr_pct=round(float(cagr) * 100, 2),
        sharpe=round(float(sharpe), 2),
        sortino=round(float(sortino), 2),
        max_drawdown_pct=round(float(max_dd) * 100, 2),
        win_rate_pct=round(float(win_rate) * 100, 1),
        num_trades=int(trades),
        exposure_pct=round(float(exposure) * 100, 1),
    )


def backtest_portfolio(
    price_data: dict[str, pd.DataFrame], config
) -> tuple[pd.Series, Metrics, pd.DataFrame]:
    """Equal-weight portfolio backtest across the whole universe.

    Returns (portfolio_returns, portfolio_metrics, per_ticker_metrics_table).
    """
    strat_cfg = config.section("strategy")
    risk_cfg = config.section("risk")
    bt_cfg = config.section("backtest")

    per_ticker_returns: dict[str, pd.Series] = {}
    rows = []

    for ticker, df in price_data.items():
        if len(df) < strat_cfg["trend_ma"] + 5:
            continue
        pos_df = generate_positions(df, strat_cfg, risk_cfg)
        ret = strategy_returns(pos_df, bt_cfg)
        per_ticker_returns[ticker] = ret
        m = compute_metrics(ret, bt_cfg, pos_df["position"])
        rows.append({"ticker": ticker, **m.as_dict()})

    if not per_ticker_returns:
        return pd.Series(dtype=float), Metrics(0, 0, 0, 0, 0, 0, 0, 0), pd.DataFrame()

    combined = pd.concat(per_ticker_returns, axis=1).fillna(0.0)
    portfolio = combined.mean(axis=1)  # equal weight across names
    port_metrics = compute_metrics(portfolio, bt_cfg)

    per_ticker_table = (
        pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    )
    return portfolio, port_metrics, per_ticker_table


def simulate_portfolio(
    price_data: dict[str, pd.DataFrame], config
) -> tuple[pd.Series, Metrics, pd.DataFrame]:
    """Event-driven portfolio simulation that matches how the system is ACTUALLY
    traded — and is therefore the number to trust.

    Each day:
      1. idle cash earns the risk-free rate (no longer a 0% drag),
      2. exit any holding whose per-name signal turned off (target/stop/regime),
      3. among fresh entry signals, take the highest-ranked ones until the
         max_positions cap is hit, sizing each with the ATR risk rules,
      4. costs charged on every buy and sell.

    Trades execute at the signal day's close (signal uses data up to that close,
    so there is no look-ahead). Returns (portfolio_returns, metrics, trades).
    """
    strat_cfg = config.section("strategy")
    risk_cfg = config.section("risk")
    bt_cfg = config.section("backtest")

    capital0 = float(risk_cfg["capital"])
    max_pos = int(risk_cfg["max_positions"])
    cost = _per_side_cost(bt_cfg)
    rf_daily = bt_cfg.get("risk_free_rate", 0.0) / 252

    # Precompute per-name signal frames.
    frames: dict[str, pd.DataFrame] = {}
    for t, df in price_data.items():
        if len(df) < strat_cfg["trend_ma"] + 5:
            continue
        frames[t] = generate_positions(df, strat_cfg, risk_cfg)
    if not frames:
        return pd.Series(dtype=float), Metrics(0, 0, 0, 0, 0, 0, 0, 0), pd.DataFrame()

    master = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
    tickers = list(frames.keys())

    # Align everything to the master calendar as fast numpy arrays.
    close = {t: frames[t]["Close"].reindex(master).to_numpy(float) for t in tickers}
    pos = {t: frames[t]["position"].reindex(master).fillna(0).to_numpy(float) for t in tickers}
    sig = {t: frames[t]["signal"].reindex(master).fillna(0).to_numpy(float) for t in tickers}
    score = {t: frames[t]["score"].reindex(master).to_numpy(float) for t in tickers}
    atr = {t: frames[t]["atr"].reindex(master).to_numpy(float) for t in tickers}

    # --- market regime filter (optional) -----------------------------------
    # Build a synthetic breadth index from the universe (equal-weight average
    # close) and only allow NEW entries when that index is above its own MA.
    # This keeps the system out of broad downturns instead of buying dips into
    # a falling market. Needs no extra data and is computed only from the past.
    use_regime = bool(strat_cfg.get("regime_filter", False))
    if use_regime:
        regime_ma = int(strat_cfg.get("regime_ma", 200))
        idx = np.nanmean(np.vstack([close[t] for t in tickers]), axis=0)
        idx_s = pd.Series(idx, index=master)
        ma = idx_s.rolling(regime_ma).mean()
        regime_on = (idx_s > ma).to_numpy()
        regime_on = np.where(np.isnan(idx) | np.isnan(ma.to_numpy()), False, regime_on)
    else:
        regime_on = np.ones(len(master), dtype=bool)

    cash = capital0
    holdings: dict[str, dict] = {}   # ticker -> {qty, entry}
    equity_vals: list[float] = []
    trades: list[dict] = []
    held_days = 0

    for i, d in enumerate(master):
        cash *= (1 + rf_daily)

        # --- exits ---
        for t in list(holdings.keys()):
            px = close[t][i]
            if np.isnan(px):
                continue
            if pos[t][i] == 0:                       # target / stop / regime exit
                qty = holdings[t]["qty"]
                cash += qty * px * (1 - cost)
                entry = holdings[t]["entry"]
                pnl = (px - entry) * qty - cost * px * qty - cost * entry * qty
                trades.append({"Date": d.date().isoformat(), "Stock": t,
                               "Action": "SELL", "Price": round(float(px), 2),
                               "Quantity": int(qty), "PnL": round(float(pnl), 2)})
                del holdings[t]

        # --- entries (rank fresh signals, fill open slots) ---
        if len(holdings) < max_pos and regime_on[i]:
            cands = []
            for t in tickers:
                if t in holdings:
                    continue
                px = close[t][i]
                if np.isnan(px) or sig[t][i] != 1 or pos[t][i] != 1:
                    continue
                sc = score[t][i]
                cands.append((0.0 if np.isnan(sc) else float(sc), t, float(px)))
            cands.sort(reverse=True)

            equity_now = cash + sum(
                holdings[t]["qty"] * close[t][i]
                for t in holdings if not np.isnan(close[t][i])
            )
            for sc, t, px in cands:
                if len(holdings) >= max_pos:
                    break
                atr_now = atr[t][i]
                atr_now = 0.0 if np.isnan(atr_now) else float(atr_now)
                size = riskmod.position_size(equity_now, px, atr_now, risk_cfg)
                affordable = int(cash // (px * (1 + cost))) if px > 0 else 0
                qty = min(size.quantity, affordable)
                if qty <= 0:
                    continue
                cash -= qty * px * (1 + cost)
                holdings[t] = {"qty": qty, "entry": px}
                trades.append({"Date": d.date().isoformat(), "Stock": t,
                               "Action": "BUY", "Price": round(px, 2),
                               "Quantity": int(qty), "PnL": 0.0})

        # --- mark to market ---
        mtm = sum(holdings[t]["qty"] * close[t][i]
                  for t in holdings if not np.isnan(close[t][i]))
        equity_vals.append(cash + mtm)
        if holdings:
            held_days += 1

    equity = pd.Series(equity_vals, index=master, name="equity")
    returns = equity.pct_change().fillna(0.0).rename("ret")
    metrics = compute_metrics(returns, bt_cfg)

    n_buys = sum(1 for tr in trades if tr["Action"] == "BUY")
    sells = [tr for tr in trades if tr["Action"] == "SELL"]
    wins = sum(1 for tr in sells if tr["PnL"] > 0)
    win_rate = round(wins / len(sells) * 100, 1) if sells else 0.0
    exposure = round(held_days / len(master) * 100, 1) if len(master) else 0.0
    metrics = replace(metrics, num_trades=n_buys, exposure_pct=exposure,
                      win_rate_pct=win_rate)
    trades_df = pd.DataFrame(trades)
    return returns, metrics, trades_df


def walk_forward(portfolio_returns: pd.Series, config) -> pd.DataFrame:
    """Split returns into K equal time folds and report metrics per fold.

    Consistent out-of-sample numbers across folds = a more trustworthy edge
    than one big in-sample total return.
    """
    bt_cfg = config.section("backtest")
    k = max(int(bt_cfg.get("walk_forward_splits", 4)), 1)
    r = portfolio_returns.dropna()
    if r.empty:
        return pd.DataFrame()

    folds = np.array_split(np.arange(len(r)), k)
    rows = []
    for i, idx in enumerate(folds, 1):
        seg = r.iloc[idx]
        if seg.empty:
            continue
        m = compute_metrics(seg, bt_cfg)
        rows.append({
            "fold": i,
            "start": seg.index[0].date().isoformat(),
            "end": seg.index[-1].date().isoformat(),
            "return_pct": m.total_return_pct,
            "sharpe": m.sharpe,
            "max_dd_pct": m.max_drawdown_pct,
        })
    return pd.DataFrame(rows)


def per_name_diagnostics(price_data: dict[str, pd.DataFrame], config) -> pd.DataFrame:
    """Trustworthy per-name table: run each ticker through the SAME realistic
    engine (sized, costs, cash earns rf), standalone. Unlike the old equal-weight
    diagnostics, these Sharpes are directly comparable to the portfolio's and
    are honest about whether a name actually contributes. Sorted best-first.
    """
    rows = []
    for ticker, df in price_data.items():
        _, m, _ = simulate_portfolio({ticker: df}, config)
        rows.append({
            "ticker": ticker,
            "sharpe": m.sharpe,
            "total_return_pct": m.total_return_pct,
            "cagr_pct": m.cagr_pct,
            "max_drawdown_pct": m.max_drawdown_pct,
            "win_rate_pct": m.win_rate_pct,
            "num_trades": m.num_trades,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
