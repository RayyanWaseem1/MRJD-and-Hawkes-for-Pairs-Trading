"""
Corporate action adjustment for raw OHLCV price series.

The commited CSVs are raw Databento bars (`ts_event` + raw OHLCV schema),
which are **unadjusted by default**. Eactly one symbol in this repository
carries an unadjusted corporate action:

    NVDA 2021-07-20 4:1 split (raw close-to-close return -75.4%)
    NVDA 2024-06-10 10:1 split (raw close-to-close return -89.9%)

Verified by scanning every committed symbol for [close-to-close] return > 40%:
NVDA is the only one with any hits, and it has exactly these two.

Why this matters:
The previousu pipeline "handled" splits by deleting any day with |return| > 50%
in `clean_data()`. Deleting the split *day* does not remove the permanent
*level shift*: the log spread log(AMD) - h * log(NVDA) steps by h * log(4) and 
then h * log(10), against a spread whose true static-h standard deviation is 
0.69. Every AMD/NVDA result was therefore measuring two stock splits. 

Why the table is hard coded:
Split factors are discrete, publicly known, and verifiable against the data
itself (`verify_split_table` below re-derives each factor from the raw price
jump and asserts it matches). A committed table is reproducible; a network 
fetch is not.

Dividends:
Dividends are *NOT* adjusted for, and the divided differential is *NOT*
accrued in the backtest. 
"""

from __future__ import annotations 

from dataclasses import dataclass 
from typing import Dict, List, Optional, cast

import numpy as np 
import pandas as pd

__all__ = [
    "Split",
    "SPLIT_TABLE",
    "DIVIDEND_TREATMENT",
    "adjust_for_splits",
    "verify_no_unadjusted_actions",
    "verify_split_table",
    "estimate_dividend_drag",
    "accrue_dividend_differential",
    "UnadjustedCorporateActionError",
]

class UnadjustedCorporateActionError(RuntimeError):
    """Raised when a price series still contains an unadjusted corporate action"""

@dataclass(frozen = True)
class Split:
    """ A stock split. `ratio` is the new-shares-per-old-share factor (4.0 == 4.1)"""

    symbol: str 
    date: str 
    ratio: float 
    note: str = ""

# sourced from public split records; each entry is re-verified against the raw
# price series by `verify_split_table()`
SPLIT_TABLE: List[Split] = [
    Split("NVDA", "2021-07-20", 4.0, "4:1 split, raw close-to-close -75.4%"),
    Split("NVDA", "2024-06-10", 10.0, "10:1 split, raw close-to-close -89.9%"),
]

DIVIDEND_TREATMENT = """\
Dividends are exluded from this study, in both the spread construction and 
the backtest PnL. This is a real, unmodelled cash flow for a long/short book:
the position is long one name and short the other, so the strategy pays the 
dividend on the short leg and receives it on the long leg, and the net is the 
dividend *differential* between the two names. 

Estimated size of the omission (trailing indicative gross yields):

    CVX/XOM    ~4.2% vs ~3.3%   ->  ~0.9%/yr differential
    GS/MS      ~2.2% vs ~3.0%   ->  ~0.8%/yr differential
    GLD/GDX    0.0% vs ~1.3%    ->  ~1.3%/yr differential
    SPY/IVV    ~1.2% vs ~1.2%   ->  ~0.0%/yr differential (near-identical)
    AMD/NVDA   0.0% vs ~0.02%   ->  ~0.0%/yr differential
    
Against realized average gross exposure of roughly 10% of capital, a 0.9%/yr
gross differential is on the order of 9bp/yr on total capital -- small next to 
the strategy's own noise, but the SAME order of magnitude as the alpha being 
tested for. It is therefore a material omission for the CVX/XOM, GS/MS and
GLD/GDX pairs and must be stated, not assumed away.

The SPY benchmark used for the beta regression is likewise a PRICE series, not 
a total-return series. Comparisons against it understate the benchmark by 
roughly its ~1.2%/yr dividend yield. Because the headline test is now a direct
test of mean excess return (not a CAPM alpha), this affects only the 
neutrality regression, where it is second-order.
"""

def adjust_for_splits(
    df: pd.DataFrame,
    symbol: str, 
    splits: Optional[List[Split]] = None, 
    price_columns: tuple = ("Open", "High", "Low", "Close"),
    volume_column: str = "Volume",
) -> pd.DataFrame:
    """
    Back-adjust a raw OHLCV frame for stock splits. 

    Prices on or after a split are already in post-split terms, so every bar
    strictly *before* the split date is divided by the cumulative split factor
    and volume is multiplied by it. The most recent bar is left untouched,
    which is the standard back-adjustment convention. 

    Params:
    df: pd.DataFrame
        OHLCV frame with a DatetimeIndex and capitalized column names.
    symbol: str 
        symbol whose splits should be applied 
    splits: list[Split], optional
        defaults to the module-level SPLIT_TABLE

    Returns:
    pd.DataFrame
        A copy with split-adjusted prices and volume
    """
    if splits is None:
        splits = SPLIT_TABLE

    relevant = [s for s in splits if s.symbol.upper() == symbol.upper()]
    if not relevant:
        return df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"adjust_for_splits requires a DatetimeIndex for {symbol}")

    out = df.copy()
    # DataFrame.copy() widens the index type in pandas' type stubs, even
    # though the DatetimeIndex check above still applies to the copy.
    index = cast(pd.DatetimeIndex, out.index)

    # timezone alignment: the committed CSVs are tz-aware UTC
    tz = index.tz

    for split in relevant:
        split_ts = pd.Timestamp(split.date)
        if tz is not None and split_ts.tz is None:
            split_ts = split_ts.tz_localize(tz)

        pre = index < split_ts
        if not pre.any():
            continue 

        for col in price_columns:
            if col in out.columns:
                out.loc[pre, col] = out.loc[pre, col] / split.ratio
        if volume_column in out.columns:
            out.loc[pre, volume_column] = out.loc[pre, volume_column] * split.ratio

    return out 

def verify_split_table(
    raw: pd.DataFrame,
    symbol: str,
    splits: Optional[List[Split]] = None,
    close_column: str = "Close",
    tolerance: float = 0.15,
) -> Dict[str, float]:
    """
    Re-derive each tabled split factor from the raw price series.

    The point is that the table is not taken on faith: the implied factor 
    `prev_close / close` on the split date must match the tabled ratio.

    Returns a {date: implied_ratio} map. Raises if any entry disagrees with
    the table by more than `tolerance` (relative)
    """
    if splits is None:
        splits = SPLIT_TABLE

    relevant = [s for s in splits if s.symbol.upper() == symbol.upper()]
    implied: Dict[str, float] = {}
    if not relevant:
        return implied 

    if not isinstance(raw.index, pd.DatetimeIndex):
        raise TypeError(f"verify_split_table requires a DatetimeIndex for {symbol}")

    index = cast(pd.DatetimeIndex, raw.index)
    close = raw[close_column].astype(float)
    tz = index.tz

    for split in relevant:
        split_ts = pd.Timestamp(split.date)
        if tz is not None and split_ts.tz is None:
            split_ts = split_ts.tz_localize(tz)

        if split_ts not in close.index:
            raise UnadjustedCorporateActionError(
                f"{symbol}: tabled split date {split.date} is not in the price series"
            )

        pos = index.get_loc(split_ts)
        if not isinstance(pos, (int, np.integer)):
            raise ValueError(
                f"verify_split_table requires a unique DatetimeIndex for {symbol}"
            )
        pos = int(pos)
        if pos == 0:
            continue 

        ratio = float(close.iloc[pos - 1] / close.iloc[pos])
        implied[split.date] = ratio 

        rel_err = abs(ratio - split.ratio) / split.ratio 
        if rel_err > tolerance:
            raise UnadjustedCorporateActionError(
                f"{symbol}: tabled {split.ratio}:1 split on {split.date} does not match "
                f"the date (implied factor {ratio:.3f}, relative error {rel_err:.1%}). "
                "The split table is wrong or the CSV changed."
            )
    return implied 

def verify_no_unadjusted_actions(
    df: pd.DataFrame,
    symbol: str, 
    close_column: str = "Close",
    threshold: float = 0.40,
) -> None:
    """
    Assert that a price series contains no remaining unadjusted corporate actions.
    
    A single-day |close-to-close return| above `threshold` on a large-cap equity
    is almost always a split or a reverse split rather than a market move. The 
    previous pipeline silently *deleted* these days, which removed the return but 
    left the permanent level shift in place. This raises instead.
    
    Raises:
    UnadjustedCorporateActionError
        Naming the symbol, date, and return, so the split table can be updated
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"verify_no_unadjusted_actions requires a DatetimeIndex for {symbol}"
        )

    close = df[close_column].astype(float)
    returns = close.pct_change()
    offenders = returns[returns.abs() > threshold].dropna()

    if len(offenders) > 0:
        offender_dates = cast(pd.DatetimeIndex, offenders.index)
        detail = ", ".join(
            f"{ts.date()} ({float(val):+.1%})"
            for ts, val in zip(offender_dates, offenders.to_numpy())
        )
        raise UnadjustedCorporateActionError(
            f"{symbol}: {len(offenders)} bar(s) with |return| > {threshold:.0%} remain "
            f"after adjustment: {detail}. This is an unadjusted corporate action. "
            f"Add it to corporate_actions.SPLIT_TABLE -- do NOT delete the day, which "
            f"removes the return but leaves the level shift."
        )

def estimate_dividend_drag(symbol_a: str, symbol_b: str) -> Dict[str, float]:
    """
    Indicative annual dividend differential for a pair, used for disclosure only.
    
    These are trailing gross yields, hard-coded for reporting. They are NOT used in 
    PnL -- see DIVIDEND_TREATMENT.
    """
    yields = {
        "CVX": 0.042, "XOM": 0.033,
        "GS": 0.022, "MS": 0.030,
        "GLD": 0.000, "GDX": 0.013,
        "SPY": 0.012, "IVV": 0.012,
        "AMD": 0.000, "NVDA": 0.0002,
    }
    ya = yields.get(symbol_a.upper(), 0.0)
    yb = yields.get(symbol_b.upper(), 0.0)
    return {
        "yield_a": ya, 
        "yield_b": yb, 
        "differential_annual": ya - yb,
        "abs_differential_annual": abs(ya - yb),
    }

def accrue_dividend_differential(
    dates: pd.DatetimeIndex,
    exposure: pd.Series,
    symbol_a: str, 
    symbol_b: str, 
    enabled: bool = False
) -> pd.Series:
    """
    Hook for accruing the dividend differential into daily returns.
    
    Disabled by default (decision: dividends are disclosed as excluded rather 
    than modelled from an unsourced table). When a real, dated dividend table 
    is added, replace the flat-yield approximation here with actual ex-dates.
    
    Returns a per-date accrual series, all zeros when `enabled` is False.
    """

    accrual = pd.Series(0.0, index = dates)
    if not enabled:
        return accrual

    diff = estimate_dividend_drag(symbol_a, symbol_b)["differential_annual"]
    daily = diff / 252.0
    aligned = exposure.reindex(dates).fillna(0.0)
    return accrual.add(aligned * daily, fill_value = 0.0)
