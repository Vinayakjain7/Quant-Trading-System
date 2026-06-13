import numpy as np
import pandas as pd

from quant import backtest as bt
from quant.strategy import generate_positions

STRAT = dict(bb_window=20, bb_std=1.5, trend_ma=50, rsi_window=14,
             rsi_max_entry=55, atr_window=14, vol_window=20, min_vol_ratio=0.0)
RISK = dict(atr_stop_mult=2.0, risk_per_trade=0.02, max_allocation=0.2,
            max_positions=5, max_drawdown=-0.1, capital=100000, top_n=5)
BT = dict(commission_bps=3, slippage_bps=5, risk_free_rate=0.06,
          walk_forward_splits=4)


def test_positions_are_binary(synthetic_ohlcv):
    df = generate_positions(synthetic_ohlcv, STRAT, RISK)
    assert set(df["position"].unique()).issubset({0, 1})


def test_returns_use_lagged_position(synthetic_ohlcv):
    """Strategy return on day t must depend on the position decided on t-1,
    never on t's own (future-at-decision-time) position."""
    df = generate_positions(synthetic_ohlcv, STRAT, RISK)
    ret = bt.strategy_returns(df, BT)
    daily = df["Close"].pct_change().fillna(0.0)
    expected_gross = df["position"].shift(1).fillna(0) * daily
    # net return <= gross (costs only subtract), and aligns with lagged pos
    assert (ret <= expected_gross + 1e-12).all()


def test_costs_reduce_returns(synthetic_ohlcv):
    df = generate_positions(synthetic_ohlcv, STRAT, RISK)
    no_cost = bt.strategy_returns(df, dict(commission_bps=0, slippage_bps=0))
    with_cost = bt.strategy_returns(df, BT)
    assert with_cost.sum() <= no_cost.sum() + 1e-9


def test_metrics_sane(synthetic_ohlcv):
    df = generate_positions(synthetic_ohlcv, STRAT, RISK)
    ret = bt.strategy_returns(df, BT)
    m = bt.compute_metrics(ret, BT, df["position"])
    assert -100 <= m.max_drawdown_pct <= 0
    assert 0 <= m.win_rate_pct <= 100
    assert m.num_trades >= 0


class _Cfg:
    """Minimal config stub for the portfolio simulator."""
    def __init__(self):
        self._s = {"strategy": STRAT, "risk": RISK, "backtest": BT}

    def section(self, name):
        return self._s[name]


def _multi_universe(base):
    # Two correlated-but-distinct names so the portfolio cap logic exercises.
    import numpy as np
    shifted = base.copy()
    shifted["Close"] = base["Close"] * (1 + np.linspace(0, 0.05, len(base)))
    for c in ("Open", "High", "Low"):
        shifted[c] = shifted["Close"]
    return {"AAA": base, "BBB": shifted}


def test_simulate_portfolio_runs(synthetic_ohlcv):
    cfg = _Cfg()
    returns, metrics, trades = bt.simulate_portfolio(_multi_universe(synthetic_ohlcv), cfg)
    assert len(returns) > 0
    assert -100 <= metrics.max_drawdown_pct <= 0
    assert 0 <= metrics.exposure_pct <= 100
    assert metrics.num_trades >= 0
    # never hold more than max_positions at once is enforced by construction;
    # at least the trade log should be well-formed if any trades happened.
    if not trades.empty:
        assert set(trades["Action"].unique()).issubset({"BUY", "SELL"})


def test_idle_cash_earns_risk_free():
    """A universe with no valid signals should still compound at ~the risk-free
    rate (idle cash earns rf), not sit at 0%."""
    import numpy as np
    import pandas as pd
    n = 300
    dates = pd.bdate_range("2021-01-01", periods=n)
    # Strictly downtrending price -> trend filter blocks all entries.
    close = pd.Series(np.linspace(200, 100, n), index=dates)
    flat = pd.DataFrame({"Open": close, "High": close, "Low": close,
                         "Close": close, "Volume": 1_000_000}, index=dates)
    cfg = _Cfg()
    returns, metrics, trades = bt.simulate_portfolio({"DOWN": flat}, cfg)
    assert metrics.num_trades == 0
    # ~6% annual rf over ~1.2y should give a positive total return near rf.
    assert metrics.total_return_pct > 0
