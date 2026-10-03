"""
Intraday support: sampling frequency as a first-class parameter

Two of the study's central problems are frequency problems, not code problems:

    * Bipower variation is a HIGH FREQUENCY estimator. Its asymptotics require
    Delta -> 0, i.e. many returns *within* the period being tested. On daily
    bars there is no asymptotic regime in which the test is valid, which is why
    `jump_detector` demotes it to a robustness check.

    * The Hawkes layer needs EVENTS. Correctly dates, multiplicity-controlled
    detection finds 0-13 jumps per pairs in ~1,960 daily bars. A Hawkes process 
    cannot be estimated from that, and the branching-ratio confidence interval
    includes zero on every pair as a result. This is a sample-size limit, not 
    an estimator defect 

At 5-minute sampling the same 7.8-year span carries ~153,000 bars instead of 
1,960 -- a 78x increase in observations, and a comparable increase in detected
events. That is the change that makes the project's title testable. 

What this module provides:
    Frequency: Bars per day, annualization factor, sensible detector window 
    load_intraday: read an intraday OHLCV file and attach its frequency
    infer_frequency: detect bars-per-day from a DatetimeIndex
    resample_bars: aggregate intraday bars to a coarser frequency
    demonstrate_power: synthetic experiment: how many observations does the 
                        Hawkes layer actually need before it can recover a known 
                        branching ratio?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path 
from typing import Dict, Optional 

import numpy as np 
import pandas as pd 

__all__ = [
    "Frequency",
    "DAILY",
    "HOURLY",
    "FIVE_MINUTE",
    "ONE_MINUTE",
    "infer_frequency",
    "load_intraday",
    "resample_bars",
    "demonstrate_power",
]

TRADING_DAYS_PER_YEAR = 252

@dataclass(frozen = True)
class Frequency:
    """ A sampling frequency, and everything derived from it"""

    label: str 
    bars_per_day: float 
    pandas_rule: str 

    @property
    def bars_per_year(self) -> float:
        """ Annualization factor. Replaces the hardcoded 252 when intraday"""
        return self.bars_per_day * TRADING_DAYS_PER_YEAR

    @property
    def detector_window(self) -> int:
        """
        Local bipower volatility window, in bars

        Roughly one trading day of history for intraday data, and 20 bars for 
        daily data (where a one-day window is meaningless)
        """
        return int(max(round(self.bars_per_day), 20))

    @property
    def bipower_is_valid(self) -> bool:
        """
        Whether bipower variation has an asymptotic regime here.

        BPV needs many returns within the tested period. At one bar per day it 
        has none, which is the honest reason `jump_detector` demotes it
        """
        return self.bars_per_day >= 20 

    def scale_half_life(self, half_life_days: float) -> float:
        """ Convert a half-life in trading days into bars"""
        return half_life_days * self.bars_per_day

    def describe(self) -> str:
        return (
            f"{self.label}: {self.bars_per_day:g} bars/day, "
            f"{self.bars_per_year:,.0f} bars/year, detector window "
            f"{self.detector_window}, bipower "
            f"{'valid' if self.bipower_is_valid else 'INVALID (needs intraday)'}"
        )

DAILY = Frequency("daily", 1.0, "1D")
HOURLY = Frequency("hourly", 6.5, "1h")
FIVE_MINUTE = Frequency("5-minute", 78.0, "5min")
ONE_MINUTE = Frequency("1-minute", 390.0, "1min")

_KNOWN = {f.label: f for f in (DAILY, HOURLY, FIVE_MINUTE, ONE_MINUTE)}

def infer_frequency(index: pd.Index) -> Frequency:
    """
    Infer bars-per-day from an index, by counting bars on its busiest days.

    Uses the median count over days that have any bars, so a half-day or a 
    gap does not skew the estimate.
    """

    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return DAILY

    per_day = pd.Series(1, index = index).groupby(index.normalize()).sum()
    median = float(per_day.median())

    if median <= 1.5:
        return DAILY 
    # Snap to the nearest known frequency, otherwise describe it generically
    best = min(_KNOWN.values(), key = lambda f: abs(f.bars_per_day - median))
    if abs(best.bars_per_day - median) / max(best.bars_per_day, 1) < 0.25:
        return best 
    return Frequency(f"{median:g}-bars-per-day", median, "custom")

def load_intraday(
        path: str | Path,
        date_column: str = "ts_event",
        session_start: Optional[str] = None,
        session_end: Optional[str] = None,
) -> tuple[pd.DataFrame, Frequency]:
    """
    Load an intraday OHLCV file and report its frequency.

    Params:
    sesion_start, session_end: str, optional
        e.g. "09:30", "16:00". Bars outside the session are dropped, which
        matters because overnight gaps are not comparable to intraday returns
        and would be detected as jumps
    """
    frame = pd.read_csv(path)

    column = date_column if date_column in frame.columns else None
    if column is None:
        lowered = {str(c).lower(): c for c in frame.columns}
        column = (
            lowered.get(date_column.lower())
            or lowered.get("ts_event")
            or lowered.get("datetime")
            or lowered.get("date")
        )
    if column is None:
        raise ValueError(f"No timestamp column found in {path}")

    frame[column] = pd.to_datetime(frame[column], errors = "coerce")
    frame = frame.dropna(subset = [column]).set_index(column).sort_index()
    frame.columns = [str(c).capitalize() for c in frame.columns]

    if session_start and session_end:
        frame = frame.between_time(session_start, session_end)

    return frame, infer_frequency(frame.index)

def resample_bars(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """ Aggregate OHLCV bars to a coarser frequency"""
    agg = {}
    for column, how in (
        ("Open", "first"), ("High", "max"), ("Low", "min"),
        ("Close", "last"), ("Volume", "sum"),
    ):
        if column in frame.columns:
            agg[column] = how
    if "Close" not in agg:
        raise ValueError("resample_bars requires a Close column")
    return frame.resample(rule).agg(agg).dropna(subset=["Close"])

### The honest demonstration ###

def demonstrate_power(
    event_counts = (13, 40, 150, 600, 2400),
    branching_ratio: float = 0.5,
    beta: float = 0.6,
    n_replications: int = 40,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    How many EVENTS does the Hawkes layer need to recover a known branching ratio?

    Event count, not bar count, is the binding quantity -- and it is what 
    sampling frequency actually buys. This repository's daily data yields 0-13
    detected jumps per pair; 5-minute sampling of the same calendar span would
    yield roughly two orders of magnitude more, both becuase there are 78x more
    bars and because jumps invisible at daily resolution become detectable.

    For each target event count the observation span is solved from 

        E[events] = lambda_bar * T / (1 - eta)

    holding the true branching ratio fixed, and the fit is repeated
    `n_replications` times so the reported detection rate is a frequency rather
    than one draw. 

    Reports, per event count: the share of replications whose 95% confidence 
    interval on the branching ratio excludes zero, and the median interval 
    width. that is the honest answer to "would intraday data help?" -- it is a 
    statement about estimator power, not about returns
    """
    from hawkes_calibration import HawkesFitError, HawkesProcess

    lambda_bar = 0.05
    alpha = branching_ratio * beta 

    if verbose:
        print(
            f"True branching ratio {branching_ratio:.3f} "
            f"(lambda_bar = {lambda_bar}, alpha = {alpha:.3f}, beta = {beta})"
        )
        print(f"{n_replications} replications per target event count \n")

    rows = []
    rng = np.random.default_rng(seed)

    for target in event_counts:
        span = target * (1.0 - branching_ratio) / lambda_bar 

        detections, widths, estimates, actual_events = [], [], [], []
        for rep in range(n_replications):
            model = HawkesProcess(verbose = False)
            try:
                times = model.simulate(
                    span, lambda_bar = lambda_bar, alpha = alpha, beta = beta,
                    seed = int(rng.integers(0, 2**31 -1)),
                )
                actual_events.append(len(times))
                if len(times) < 5:
                    detections.append(False)
                    continue 
                model.fit(times, span)
                se = model.standard_errors()
                low, high = se.get("branching_ratio_ci95", (np.nan, np.nan))
                if not np.isfinite(low) or not np.isfinite(high):
                    detections.append(False)
                    continue 
                detections.append(bool(low > 0))
                widths.append(high - low)
                estimates.append(model.branching_ratio())
            except (HawkesFitError, Exception): # noqa: BLE001
                detections.append(False)

        rows.append(
            {
                "target_events": target, 
                "sampling_equivalent": _describe_events(target),
                "median_events": float(np.median(actual_events)) if actual_events else 0.0,
                "observation_span": span,
                "detection_rate": float(np.mean(detections)) if detections else 0.0,
                "median_ci_width": float(np.median(widths)) if widths else np.nan,
                "median_estimate": float(np.median(estimates)) if estimates else np.nan,
                "true_branching": branching_ratio,
                "n_replications": n_replications,
            }
        )

        if verbose:
            row = rows[-1]
            width = row["median_ci_width"]
            print(
                f" {row['sampling_equivalent']:<30} events~{row['median_events']:>7,.0f} "
                f"detect {row['detection_rate']:>5.0%} "
                f"CI width {width:.3f}" if np.isfinite(width) else 
                f" {row['sampling_equivalent']:<30} events~{row['median_events']:>7,.0f} "
                f"detect {row['detection_rate']:>5.0%} CI width n/a"
            )

    return pd.DataFrame(rows)

def _describe_events(n_events: int) -> str:
    """ Label an event count by the sampling frequency that would produce it."""
    if n_events <= 20:
        return "daily (this repo: 0-13)"
    if n_events <= 60:
        return "hourly (~6.5x daily)"
    if n_events <= 250:
        return "15-minute (~26x daily)"
    if n_events <= 1000:
        return "5-minute (~78x daily)"
    return "1-minute (~390x daily)"

def _describe_span(n_bars: int) -> str:
    """ Label a bar count by the sampling frequency it corresponds to"""
    daily_equivalent = 1_960 # this repository's actual daily sample 
    ratio = n_bars / daily_equivalent 
    if ratio < 2: 
        return "daily (this repo)"
    if ratio < 8:
        return "hourly (~6.5x daily)"
    if ratio < 32:
        return "15-minute (~26x daily)"
    if ratio < 120:
        return "5-minute (~78x daily)"
    return "1-minute (~390x daily)"

if __name__ == "__main__":
    print("=" * 78)
    print("SAMPLING FREQUENCY")
    print("=" * 78)
    for freq in (DAILY, HOURLY, FIVE_MINUTE, ONE_MINUTE):
        print(" " + freq.describe())

    print("\n" + "=" * 78)
    print("HOW MANY OBSERVATIONS DOES THE HAWKES LAYER NEED?")
    print("=" * 78)
    print(
        "Synthetic process with a KNOWN branching ratio, observed over spans \n"
        "matching daily / hourly / 5-min / 1-min sampling of the same window.\n"
    )
    frame = demonstrate_power() 

    print("\n" + "-" * 78)
    usable = frame[frame["detection_rate"] >= 0.80]
    if len(usable):
        smallest = int(usable["target_events"].min())
        row = usable[usable.target_events == smallest].iloc[0]
        print(
            f"80% detection power needs ~{smallest:,} events "
            f"({row['sampling_equivalent']})."
        )
        print(
            f"This repository's daily data yields 0-13 events per pair, i.e. "
            f"roughly {smallest / 13:.0f}x too few"
        )
    else:
        print("80% power not reached at any tested event count.")

    daily_row = frame.iloc[0]
    print(
        f"\n At the real daily event count (~13), detection power is "
        f"{daily_row['detection_rate']:.0%} and the median confidence interval "
        f" on the branching ratio is {daily_row['median_ci_width']:.2f} wide -- "
        f" which is why every pair's interval includes zero."
    )
    print(
        "\n NOTE: this is a statement about ESTIMATOR POWER, not about returns. \n"
        "No intraday data ships with this repository, so no intraday PnL is \n"
        "claimed anywhere."
    )
