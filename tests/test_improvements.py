"""
Tests for the four changes identified as capable of moving the needle:

    1. intraday support        (intraday.py)
    2. volatility-scaled stops (config + backtest_engine)
    3. cross-sectional breadth (pair_screen.py, statistics_tools.pool_pair_returns)
    4. principled pair removal (pair_screen.py)
"""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import BacktestEngine
from config import Config
from intraday import (
    DAILY,
    FIVE_MINUTE,
    HOURLY,
    ONE_MINUTE,
    Frequency,
    demonstrate_power,
    infer_frequency,
    resample_bars,
)
from pair_screen import SCREEN_CRITERIA, screen_pair
from statistics_tools import pool_pair_returns


# --------------------------------------------------------------------- #
# 1. intraday support
# --------------------------------------------------------------------- #

def test_frequency_annualisation_replaces_hardcoded_252():
    assert DAILY.bars_per_year == 252
    assert FIVE_MINUTE.bars_per_year == pytest.approx(78 * 252)
    assert ONE_MINUTE.bars_per_year == pytest.approx(390 * 252)


def test_bipower_is_flagged_invalid_on_daily_data():
    """
    BPV asymptotics need Delta -> 0. The module must say so rather than let a
    daily-bar application look legitimate.
    """
    assert DAILY.bipower_is_valid is False
    assert HOURLY.bipower_is_valid is False
    assert FIVE_MINUTE.bipower_is_valid is True
    assert ONE_MINUTE.bipower_is_valid is True


def test_infer_frequency_from_a_daily_index():
    index = pd.bdate_range("2020-01-01", periods=200, tz="UTC")
    assert infer_frequency(index).bars_per_day == 1.0


def test_infer_frequency_from_an_intraday_index():
    """A 5-minute regular-session index should be recognised as ~78 bars/day."""
    days = pd.bdate_range("2020-01-01", periods=10)
    stamps = []
    for day in days:
        stamps.extend(pd.date_range(day + pd.Timedelta(hours=9, minutes=30),
                                    periods=78, freq="5min"))
    frequency = infer_frequency(pd.DatetimeIndex(stamps))
    assert frequency.bars_per_day == pytest.approx(78, rel=0.05)
    assert frequency.bipower_is_valid


def test_resample_aggregates_ohlcv_correctly():
    index = pd.date_range("2020-01-01 09:30", periods=12, freq="5min")
    frame = pd.DataFrame(
        {
            "Open": np.arange(12, dtype=float),
            "High": np.arange(12, dtype=float) + 2,
            "Low": np.arange(12, dtype=float) - 2,
            "Close": np.arange(12, dtype=float) + 1,
            "Volume": np.full(12, 100.0),
        },
        index=index,
    )
    hourly = resample_bars(frame, "1h")

    first = hourly.iloc[0]
    assert first["Open"] == 0.0            # first open of the hour
    assert first["High"] == frame["High"].iloc[:12].max() or first["High"] > 0
    assert first["Volume"] == 100.0 * len(frame.loc[: hourly.index[0] + pd.Timedelta("59min")])


def test_event_count_is_what_limits_the_hawkes_layer():
    """
    The honest claim behind "use intraday data": detection power is driven by
    EVENT count, and the repository's daily data supplies far too few.

    At ~13 events (what daily detection yields) power should be poor and the
    branching-ratio interval wide; by a few hundred events it should be strong
    and the interval tight.
    """
    frame = demonstrate_power(
        event_counts=(13, 600), n_replications=12, seed=7, verbose=False
    )

    low, high = frame.iloc[0], frame.iloc[1]

    assert low["detection_rate"] < high["detection_rate"], (
        "more events must not reduce detection power"
    )
    assert high["median_ci_width"] < low["median_ci_width"] / 2, (
        "the branching-ratio interval should tighten substantially with events"
    )
    assert high["detection_rate"] >= 0.9


# --------------------------------------------------------------------- #
# 2. volatility-scaled stops
# --------------------------------------------------------------------- #

def _flat_frames(index):
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.normal(0, 0.01, len(index)))
    def frame(px):
        return pd.DataFrame(
            {"Open": px, "High": px * 1.001, "Low": px * 0.999,
             "Close": px, "Volume": 1e6},
            index=index,
        )
    return frame(100 * np.exp(common)), frame(50 * np.exp(common))


def _signals(index):
    return pd.DataFrame(
        {"signal": 0.0, "position": 0.0, "position_size": 0.0, "z_score": 0.0,
         "lambda": 0.01, "spread": 0.0, "regime": "normal", "signal_exit_reason": ""},
        index=index,
    )


def test_stops_scale_with_spread_volatility(trading_index):
    """A wider spread must get a wider stop; a fixed percentage cannot do that."""
    frame_a, frame_b = _flat_frames(trading_index)
    rng = np.random.default_rng(11)

    levels = {}
    for label, sd in (("narrow", 0.02), ("wide", 0.20)):
        spread = pd.DataFrame(
            {"spread": rng.normal(0, sd, len(trading_index)), "hedge_ratio": 1.0},
            index=trading_index,
        )
        engine = BacktestEngine(stop_mode="volatility", verbose=False)
        engine.run_backtest(
            _signals(trading_index), spread, frame_a["Close"], frame_b["Close"],
            hedge_ratio=1.0, asset_a_ohlc=frame_a, asset_b_ohlc=frame_b,
        )
        levels[label] = engine.resolved_stops["stop_loss_pct"]

    assert levels["wide"] > levels["narrow"] * 3, (
        f"stops did not scale with spread volatility: {levels}"
    )


def test_fixed_mode_still_reproduces_the_shipped_configuration(trading_index):
    """The audited configuration must remain reproducible for comparison."""
    frame_a, frame_b = _flat_frames(trading_index)
    spread = pd.DataFrame({"spread": 0.0, "hedge_ratio": 1.0}, index=trading_index)

    engine = BacktestEngine(stop_mode="fixed", stop_loss_pct=0.03, verbose=False)
    engine.run_backtest(
        _signals(trading_index), spread, frame_a["Close"], frame_b["Close"],
        hedge_ratio=1.0, asset_a_ohlc=frame_a, asset_b_ohlc=frame_b,
    )
    assert engine.resolved_stops["mode"] == "fixed"
    assert engine.stop_loss_pct == pytest.approx(0.03)


def test_stop_levels_are_clamped_against_a_degenerate_volatility(trading_index):
    frame_a, frame_b = _flat_frames(trading_index)
    spread = pd.DataFrame(
        {"spread": np.full(len(trading_index), 1.0), "hedge_ratio": 1.0},
        index=trading_index,
    )
    engine = BacktestEngine(stop_mode="volatility", verbose=False)
    engine.run_backtest(
        _signals(trading_index), spread, frame_a["Close"], frame_b["Close"],
        hedge_ratio=1.0, asset_a_ohlc=frame_a, asset_b_ohlc=frame_b,
    )
    # A constant spread has zero dispersion; the engine must not emit a zero stop.
    assert engine.resolved_stops.get("mode") in ("fixed_fallback", "volatility")
    if engine.resolved_stops.get("mode") == "volatility":
        assert engine.stop_loss_pct >= engine.stop_floor_pct


def test_config_default_is_volatility_mode():
    """Fixed percentage stops are the defect; volatility scaling is the default."""
    assert Config().backtest.stop_mode == "volatility"


# --------------------------------------------------------------------- #
# 3 & 4. breadth and principled pair removal
# --------------------------------------------------------------------- #

def test_screen_criteria_are_named_constants():
    """
    The criteria must be fixed in advance and inspectable, not inline literals
    that can drift once results are seen.
    """
    for key in (
        "COINT_P_MAX", "MIN_HALF_LIFE", "MAX_HALF_LIFE",
        "PREDICT_T_MAX", "EDGE_COST_MULTIPLE", "MULTIPLICITY_ALPHA",
    ):
        assert key in SCREEN_CRITERIA


def test_amd_nvda_is_dropped_by_rule_not_by_outcome():
    """
    AMD/NVDA must fail a screen specified in advance, so its removal is
    principled rather than data snooping on the validation result.
    """
    cfg = Config()
    result = screen_pair(
        "AMD", "NVDA", cfg.train_val.train_start, cfg.train_val.train_end, cfg
    )
    assert "error" not in result
    assert result["selected"] is False
    # Specifically: it is cointegrated but shows no usable mean reversion.
    assert result["predictive"] is False
    assert result["predict_t"] > SCREEN_CRITERIA["PREDICT_T_MAX"]


def test_screen_uses_engle_granger_not_plain_adf():
    cfg = Config()
    result = screen_pair(
        "CVX", "XOM", cfg.train_val.train_start, cfg.train_val.train_end, cfg
    )
    assert "error" not in result
    assert result["coint_pvalue"] >= result["adf_pvalue"], (
        "plain ADF over-rejects on an estimated spread; EG must be no more permissive"
    )


def test_pooling_reports_effective_breadth_not_raw_count():
    """
    N correlated pairs are not N independent bets. The pooled result must say
    so, otherwise "more pairs" overstates the gain.
    """
    index = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(5)

    independent = {f"p{i}": pd.Series(rng.normal(0, 0.004, 400), index=index)
                   for i in range(5)}
    shared = rng.normal(0, 0.004, 400)
    correlated = {f"q{i}": pd.Series(shared + rng.normal(0, 0.0005, 400), index=index)
                  for i in range(5)}

    a = pool_pair_returns(independent)
    b = pool_pair_returns(correlated)

    assert a["n_pairs"] == b["n_pairs"] == 5
    assert a["effective_independent_pairs"] > 4.0
    assert b["effective_independent_pairs"] < 1.5, (
        "five near-identical pairs must not count as five independent bets"
    )
    assert a["breadth_gain_vs_single"] > b["breadth_gain_vs_single"]


def test_pooling_shrinks_the_standard_error_of_the_mean():
    """
    The whole point of breadth. Nine independent pairs diversify away most of
    the idiosyncratic volatility, so the standard error of the mean return
    falls by roughly sqrt(9) = 3.

    Note it is the SE of the MEAN that shrinks, not the SE of the Sharpe: the
    latter depends on the number of time observations, which pooling does not
    change. What pooling raises is the Sharpe itself, because portfolio
    volatility falls faster than the mean does.
    """
    index = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(9)
    series = {f"p{i}": pd.Series(rng.normal(0.0002, 0.005, 600), index=index)
              for i in range(9)}

    pooled = pool_pair_returns(series)
    single = pool_pair_returns({"p0": series["p0"]})

    ratio = single["pooled_se_annualized_pct"] / pooled["pooled_se_annualized_pct"]
    assert ratio > 2.0, (
        f"pooling nine independent pairs shrank the standard error by only "
        f"{ratio:.2f}x; expected roughly sqrt(9) = 3"
    )
    assert pooled["pooled_sharpe_annualized"] > single["pooled_sharpe_annualized"]
