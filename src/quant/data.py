"""Cached price-data access via yfinance.

Downloads are cached to parquet/csv on disk so repeated backtests don't
hammer Yahoo and so results are reproducible offline. One file per ticker.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - import guarded for test environments
    yf = None


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return cache_dir / f"{safe}.csv"


def _is_fresh(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return False
    age = _dt.datetime.now() - _dt.datetime.fromtimestamp(path.stat().st_mtime)
    return age <= _dt.timedelta(days=max_age_days)


def load_prices(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | Path = "data_cache",
    cache_days: int = 1,
) -> pd.DataFrame:
    """Return a tidy OHLCV DataFrame indexed by date for one ticker.

    Columns: Open, High, Low, Close, Volume. Uses on-disk cache.
    Returns an empty DataFrame if no data is available.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, ticker)

    if _is_fresh(path, cache_days):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return _clip(df, start, end)

    if yf is None:
        # No network/yfinance available: fall back to stale cache if present.
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return _clip(df, start, end)
        return pd.DataFrame()

    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance may return a MultiIndex column frame for single tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    df = raw[cols].copy()
    df.to_csv(path)
    return _clip(df, start, end)


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def load_universe(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data_cache",
    cache_days: int = 1,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV for every ticker. Silently skips tickers with no data."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = load_prices(t, start, end, cache_dir, cache_days)
        if not df.empty and len(df) > 0:
            out[t] = df
    return out
