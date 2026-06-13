import numpy as np

from quant import indicators as ind


def test_rsi_bounds(synthetic_ohlcv):
    r = ind.rsi(synthetic_ohlcv["Close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_bollinger_ordering(synthetic_ohlcv):
    bb = ind.bollinger(synthetic_ohlcv["Close"], 20, 2.0).dropna()
    assert (bb["bb_upper"] >= bb["bb_mean"]).all()
    assert (bb["bb_mean"] >= bb["bb_lower"]).all()


def test_atr_positive(synthetic_ohlcv):
    a = ind.atr(synthetic_ohlcv["High"], synthetic_ohlcv["Low"],
                synthetic_ohlcv["Close"], 14).dropna()
    assert (a >= 0).all()


def test_indicators_no_lookahead(synthetic_ohlcv):
    """An indicator value at time t must not change when FUTURE bars are added.
    This is the core anti-leakage guarantee."""
    cfg = dict(bb_window=20, bb_std=1.5, trend_ma=50, rsi_window=14,
               atr_window=14, vol_window=20, rsi_max_entry=45, min_vol_ratio=0.8)
    full = ind.add_indicators(synthetic_ohlcv, cfg)
    truncated = ind.add_indicators(synthetic_ohlcv.iloc[:200], cfg)
    # Compare overlapping region on a causal indicator (rolling mean).
    common = truncated.index
    np.testing.assert_allclose(
        full.loc[common, "bb_mean"].values,
        truncated["bb_mean"].values,
        rtol=1e-9, equal_nan=True,
    )
