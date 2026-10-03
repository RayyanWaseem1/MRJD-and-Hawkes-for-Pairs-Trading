"""Shared fixtures. Every stochastic test is seeded."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42


@pytest.fixture(autouse=True)
def _seed_everything():
    np.random.seed(SEED)


@pytest.fixture
def trading_index():
    """A business-day index of the right rough length for these tests."""
    return pd.bdate_range("2019-01-01", periods=600, tz="UTC")


@pytest.fixture
def flat_ohlc(trading_index):
    """Two price series with a common factor and no relative drift."""
    rng = np.random.default_rng(SEED)
    n = len(trading_index)
    common = np.cumsum(rng.normal(0, 0.01, n))
    price_a = 100 * np.exp(common)
    price_b = 50 * np.exp(common)

    def frame(px):
        return pd.DataFrame(
            {"Open": px, "High": px * 1.001, "Low": px * 0.999,
             "Close": px, "Volume": 1_000_000.0},
            index=trading_index,
        )

    return frame(price_a), frame(price_b)
