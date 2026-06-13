"""Rule-based mean-reversion strategy (the transparent core).

Logic (long-only, dip-buying inside an uptrend):

  ENTRY  when ALL of:
    * price above long-term trend MA      (only buy in uptrends)
    * price at/below the lower Bollinger   (a genuine dip)
    * RSI below rsi_max_entry              (momentum oversold, not falling knife-ish)
    * volume >= min_vol_ratio * avg        (enough liquidity)

  EXIT   when ANY of:
    * price reverts up to the Bollinger mean   (target hit)
    * price falls below an ATR-based stop       (risk control)
    * price falls below the trend MA            (regime broke)

The position series is generated with a single O(n) numpy pass (fast — the old
code's slowness came from per-row pandas .iloc access, not the logic).
A position decided at the close of day t is applied to day t+1 returns in the
backtest, so there is no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_indicators


def entry_score(df: pd.DataFrame) -> pd.Series:
    """How attractive a dip is right now (higher = deeper oversold).

    Combines distance below the lower band (in std units) with RSI oversold.
    Used to RANK candidates when more signals exist than open slots.
    """
    depth = (df["bb_lower"] - df["Close"]) / df["bb_std"].replace(0, np.nan)
    rsi_term = (50 - df["rsi"]) / 50.0
    return (depth.fillna(0) + rsi_term).astype(float)


def generate_positions(df: pd.DataFrame, cfg: dict, risk_cfg: dict) -> pd.DataFrame:
    """Add 'position' (0/1), 'signal' (entry day), and 'score' columns.

    cfg      = config['strategy']
    risk_cfg = config['risk']   (for the ATR stop multiple)
    """
    data = add_indicators(df, cfg)
    data["score"] = entry_score(data)

    close = data["Close"].to_numpy(dtype=float)
    lower = data["bb_lower"].to_numpy(dtype=float)
    mean = data["bb_mean"].to_numpy(dtype=float)
    upper = data["bb_upper"].to_numpy(dtype=float)
    trend = data["trend_ma"].to_numpy(dtype=float)
    rsi = data["rsi"].to_numpy(dtype=float)
    atr = data["atr"].to_numpy(dtype=float)
    vol_ratio = data["vol_ratio"].to_numpy(dtype=float)

    rsi_max = cfg["rsi_max_entry"]
    min_vol = cfg["min_vol_ratio"]
    stop_mult = risk_cfg["atr_stop_mult"]
    # Exit target: revert to the band "mean" (default) or ride to the "upper" band.
    exit_target = cfg.get("exit_target", "mean")
    target = upper if exit_target == "upper" else mean

    n = len(data)
    position = np.zeros(n, dtype=int)
    signal = np.zeros(n, dtype=int)
    pos = 0
    entry_price = 0.0
    stop_price = 0.0

    for i in range(n):
        # Skip rows with undefined indicators (warm-up period).
        if np.isnan(lower[i]) or np.isnan(trend[i]) or np.isnan(atr[i]):
            position[i] = pos
            continue

        uptrend = close[i] > trend[i]

        if pos == 1:
            # exit conditions
            if (close[i] >= target[i]) or (close[i] <= stop_price) or (not uptrend):
                pos = 0
        else:
            # entry conditions
            if (
                uptrend
                and close[i] <= lower[i]
                and rsi[i] <= rsi_max
                and vol_ratio[i] >= min_vol
            ):
                pos = 1
                entry_price = close[i]
                stop_price = entry_price - stop_mult * atr[i]
                signal[i] = 1

        position[i] = pos

    data["position"] = position
    data["signal"] = signal
    return data
