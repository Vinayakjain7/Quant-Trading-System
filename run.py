"""Convenience launcher so you don't need to install the package.

    python run.py backtest     # realistic backtest + walk-forward + trades.csv
    python run.py signals      # today's ranked BUY candidates
    python run.py ml           # train/evaluate the ML ranker

(Equivalent to `python -m quant.cli ...` once you `pip install -e .`.)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from quant.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
