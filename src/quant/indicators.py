"""Vectorized technical indicators.

Every indicator is computed with pandas/numpy vector ops — no per-row Python
loops. Critically, each value at row t uses ONLY data up to and including t
(rolling windows, shifted diffs), so there is no look-ahead leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bollinger(close: pd.Series, window: int = 20, n_std: float = 1.5) -> pd.DataFrame:
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mean + n_std * std
    lower = mean - n_std * std
    # %B: where price sits inside the bands (0 = lower, 1 = upper)
    width = (upper - lower).replace(0, np.nan)
    pct_b = (close - lower) / width
    return pd.DataFrame(
        {"bb_mean": mean, "bb_std": std, "bb_upper": upper,
         "bb_lower": lower, "bb_pct": pct_b}
    )


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach all configured indicators to an OHLCV frame.

    Returns a new frame; does not mutate the input.
    """
    out = df.copy()
    close = out["Close"]

    bb = bollinger(close, cfg["bb_window"], cfg["bb_std"])
    out = out.join(bb)

    out["rsi"] = rsi(close, cfg["rsi_window"])
    out["trend_ma"] = sma(close, cfg["trend_ma"])

    if {"High", "Low"}.issubset(out.columns):
        out["atr"] = atr(out["High"], out["Low"], close, cfg["atr_window"])
    else:  # only Close available
        out["atr"] = close.rolling(cfg["atr_window"]).std()

    if "Volume" in out.columns:
        out["vol_ma"] = out["Volume"].rolling(cfg["vol_window"]).mean()
        out["vol_ratio"] = out["Volume"] / out["vol_ma"].replace(0, np.nan)
    else:
        out["vol_ratio"] = 1.0

    return out
