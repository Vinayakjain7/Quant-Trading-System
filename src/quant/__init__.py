"""Quant Trading System — a configurable, rigorously backtested
mean-reversion + ML ranking engine built on free yfinance data.

Modules
-------
config      : load and validate config.yaml
data        : cached price downloads
indicators  : vectorized technical indicators (no look-ahead)
strategy    : rule-based signal generation
backtest    : realistic backtest with costs, slippage, walk-forward
ml          : optional ML ranking layer (leakage-guarded)
risk        : position sizing & portfolio risk controls
"""

__version__ = "2.0.0"
