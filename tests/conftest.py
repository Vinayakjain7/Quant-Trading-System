import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the src/ package importable in tests.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def synthetic_ohlcv():
    """A deterministic upward-drifting price series with noise, so indicators
    and the strategy have something realistic (and offline) to chew on."""
    rng = np.random.default_rng(42)
    n = 400
    dates = pd.bdate_range("2021-01-01", periods=n)
    steps = rng.normal(0.0005, 0.015, n)
    close = 100 * np.exp(np.cumsum(steps))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
