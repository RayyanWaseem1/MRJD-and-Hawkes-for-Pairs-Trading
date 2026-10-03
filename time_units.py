"""
One time unit, enforced: the TRADING DAY.

Everything now uses the trading-day index: observation `i` sits at time 
`i`, `dt = 1.0`, `T = n_observations - 1`, and durations are differences of
positional indices. `assert_trading_day_units` is available to make a 
regression fail loudly rather than print
"""

from __future__ import annotations 

from typing import Optional, Sequence 

import numpy as np 
from numpy.typing import NDArray
import pandas as pd 

__all__ = [
    "TIME_UNIT",
    "DT",
    "TRADING_DAYS_PER_YEAR",
    "positional_times",
    "observation_span",
    "trading_day_duration",
    "assert_trading_day_units",
]

# The single time unit used everywhere in this project 
TIME_UNIT = "trading_day"

# Step size in that unit. One observation == one trading day == 1.0
DT = 1.0

# Only for annualizing reported statistics -- never for model estimation 
TRADING_DAYS_PER_YEAR = 252 

def positional_times(
    index: pd.Index, mask: Optional[Sequence[bool] | NDArray[np.bool_]] = None
) -> np.ndarray:
    """
    Map index entries to trading-day times (0, 1, 2, ... by position)

    Params:
    index: pd.Index
        The full observation index; position defines the time.
    mask: sequence of bool, optional
        If given, only the positions wehre the mask is True are returned --
        this is how jump times are extracted

    Returns:
    np.ndarray of float trading-day times
    """

    positions = np.arange(len(index), dtype = float)
    if mask is None:
        return positions 

    mask_arr = np.asarray(mask, dtype = bool)
    if len(mask_arr) != len(index):
        raise ValueError(
            f"mask length {len(mask_arr)} does not match index length {len(index)}"
        )
    return positions[mask_arr]

def observation_span(index_or_length) -> float:
    """
    Total observation period T in trading days.

    For n observations at positions 0...n-1 the span is n-1. Always use this 
    rather than a calendar difference or a bare `len()`
    """

    n = len(index_or_length) if hasattr(index_or_length, "__len__") else int(index_or_length)
    return float(max(n-1,1))

def trading_day_duration(entry_position: int, exit_position: int) -> int:
    """ Holding period in trading days, from positional indices"""
    return int(exit_position - entry_position)

def assert_trading_day_units(dt: float, context: str = "") -> None:
    """
    Guard that a model is being estimated in trading-day units 

    Catches the reintroduction of `dt = 1/252`, which silently converts every 
    rate parameter to per-year while the rest of the pipeline reads days.
    """
    if not np.isclose(dt, DT):
        raise ValueError(
            f"{context or 'Model'} received dt = {dt!r}, expected dt = {DT} "
            f"({TIME_UNIT} units). A dt of 1/252 makes rate parameters per-YEAR "
            f"while half-lives, holding periods and Hawkes times are all in "
            f"trading days -- this is the factor-of-252 bug."
        )
