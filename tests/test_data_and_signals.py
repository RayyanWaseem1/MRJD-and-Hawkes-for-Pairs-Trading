"""
Data-integrity and signal-logic tests.

These pin the Tier 0 data findings and the Tier 3 signal-logic findings.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from corporate_actions import (
    SPLIT_TABLE,
    UnadjustedCorporateActionError,
    adjust_for_splits,
    verify_no_unadjusted_actions,
    verify_split_table,
)
from equity_pairs_loader import EquityPairsDataPipeline
from signal_generation import EXIT_PRIORITY, HawkesRegime, SignalType, TradingSignals

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NVDA_CSV = PROJECT_ROOT / "OHLCV_NVDA.csv"


def _load_raw(symbol):
    frame = pd.read_csv(PROJECT_ROOT / f"OHLCV_{symbol}.csv")
    frame["ts_event"] = pd.to_datetime(frame["ts_event"])
    frame = frame.set_index("ts_event").sort_index()
    frame.columns = [c.capitalize() for c in frame.columns]
    return frame


# --------------------------------------------------------------------- #
# corporate actions -- finding 0.3
# --------------------------------------------------------------------- #

@pytest.mark.skipif(not NVDA_CSV.exists(), reason="NVDA CSV not present")
def test_raw_nvda_contains_two_unadjusted_splits():
    """The bug the audit found must still be detectable in the raw file."""
    raw = _load_raw("NVDA")
    with pytest.raises(UnadjustedCorporateActionError, match="2021-07-20"):
        verify_no_unadjusted_actions(raw, "NVDA")


@pytest.mark.skipif(not NVDA_CSV.exists(), reason="NVDA CSV not present")
def test_split_adjustment_removes_the_level_shifts():
    raw = _load_raw("NVDA")
    adjusted = adjust_for_splits(raw, "NVDA")
    verify_no_unadjusted_actions(adjusted, "NVDA")  # must not raise

    assert float(adjusted["Close"].pct_change().abs().max()) < 0.40


@pytest.mark.skipif(not NVDA_CSV.exists(), reason="NVDA CSV not present")
def test_split_table_is_verified_against_the_data():
    """The tabled ratios must be re-derivable from the raw prices themselves."""
    raw = _load_raw("NVDA")
    implied = verify_split_table(raw, "NVDA")

    assert set(implied) == {"2021-07-20", "2024-06-10"}
    assert implied["2021-07-20"] == pytest.approx(4.0, rel=0.10)
    assert implied["2024-06-10"] == pytest.approx(10.0, rel=0.10)


def test_no_other_symbol_needs_adjustment():
    """NVDA was the only unadjusted symbol; that should stay true."""
    for symbol in ("AMD", "CVX", "XOM", "GS", "MS", "SPY", "IVV", "GLD", "GDX"):
        path = PROJECT_ROOT / f"OHLCV_{symbol}.csv"
        if not path.exists():
            continue
        verify_no_unadjusted_actions(_load_raw(symbol), symbol)


# --------------------------------------------------------------------- #
# spread construction -- findings 0.1, 0.2, 3.7, 3.8
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def cvx_xom_pipeline():
    pipeline = EquityPairsDataPipeline(
        asset_a_path=str(PROJECT_ROOT / "OHLCV_CVX.csv"),
        asset_b_path=str(PROJECT_ROOT / "OHLCV_XOM.csv"),
        asset_a_symbol="CVX", asset_b_symbol="XOM", verbose=False,
    )
    pipeline.load_from_csv(date_columns="ts_event")
    pipeline.clean_data()
    return pipeline


def test_static_hedge_dramatically_reduces_spread_variance(cvx_xom_pipeline):
    """
    A rolling hedge ratio inflated spread sigma by 22.7x on CVX/XOM, because
    the dh * log(B) term dominated the daily change.
    """
    static = cvx_xom_pipeline.construct_spread(method="johansen", hedge_mode="static")
    rolling = cvx_xom_pipeline.construct_spread(
        method="ols", hedge_mode="rolling", lookback=30
    )

    inflation = rolling["spread"].std() / static["spread"].std()
    assert inflation > 5.0, (
        f"rolling/static sigma ratio {inflation:.1f} -- expected a large inflation, "
        "which is the whole point of finding 0.1"
    )


def test_static_hedge_ratio_is_economically_sensible(cvx_xom_pipeline):
    """A negative hedge ratio between two oil majors is incoherent."""
    static = cvx_xom_pipeline.construct_spread(method="johansen", hedge_mode="static")
    h = float(static["hedge_ratio"].iloc[0])
    assert 0.0 < h < 3.0, f"hedge ratio {h:.3f} is not economically sensible"
    assert static["hedge_ratio"].nunique() == 1, "static hedge ratio must not vary"


def test_rolling_warmup_rows_are_dropped_not_backfilled(cvx_xom_pipeline):
    """
    `.bfill()` seeded the first `lookback` rows with an estimate computed from
    those same rows -- a look-ahead at the start of every sample.
    """
    lookback = 30
    rolling = cvx_xom_pipeline.construct_spread(
        method="ols", hedge_mode="rolling", lookback=lookback
    )
    full_length = len(cvx_xom_pipeline.data["asset_a"])
    assert len(rolling) == full_length - lookback
    assert rolling["hedge_ratio"].notna().all()


def test_clean_data_keeps_large_moves_and_the_first_row():
    """
    `clean_data` used to delete |return| > 50% days -- exactly the observations
    a jump study exists to study -- and always dropped row 0 because
    `NaN < 0.5` is False.
    """
    pipeline = EquityPairsDataPipeline(
        asset_a_path=str(PROJECT_ROOT / "OHLCV_AMD.csv"),
        asset_b_path=str(PROJECT_ROOT / "OHLCV_NVDA.csv"),
        asset_a_symbol="AMD", asset_b_symbol="NVDA", verbose=False,
    )
    raw = pipeline.load_from_csv(date_columns="ts_event")
    n_aligned = len(raw["asset_a"].index.intersection(raw["asset_b"].index))
    cleaned = pipeline.clean_data()

    assert len(cleaned["asset_a"]) == n_aligned, "clean_data dropped observations"
    assert cleaned["asset_a"].index[0] == raw["asset_a"].index[0], "first row dropped"


def test_engle_granger_pvalue_is_reported_alongside_adf(cvx_xom_pipeline):
    """
    ADF assumes an observed series; a spread from an estimated cointegrating
    vector needs Engle-Granger critical values. Both must be visible.
    """
    static = cvx_xom_pipeline.construct_spread(method="johansen", hedge_mode="static")
    stats = cvx_xom_pipeline.calculate_spread_statistics(
        static["spread"], static["log_a"], static["log_b"]
    )
    assert "adf_pvalue" in stats and "eg_pvalue" in stats
    assert stats["eg_pvalue"] >= stats["adf_pvalue"], (
        "Engle-Granger should be no more permissive than ADF -- plain ADF over-rejects"
    )


# --------------------------------------------------------------------- #
# signal logic -- findings 3.1, 3.2, 3.3
# --------------------------------------------------------------------- #

def _make_generator(**overrides):
    kwargs: dict[str, Any] = dict(
        z_entry_threshold=2.0, z_exit_threshold=0.5,
        min_lambda_decay_pct=0.15, use_hawkes_regimes=True,
        use_jump_entries=True, verbose=False,
    )
    kwargs.update(overrides)
    generator = TradingSignals(**kwargs)
    generator.set_lambda_baseline(0.01)
    generator.set_half_life(20.0)
    return generator


def test_max_hold_exit_can_actually_fire():
    """
    Finding 3.1: the exit `elif` chain made the time stop UNREACHABLE. All ten
    committed trade logs contained zero `max_hold` exits.
    """
    n = 300
    index = pd.bdate_range("2020-01-01", periods=n, tz="UTC")

    # z stays far from zero so mean reversion and profit target never trigger.
    z = pd.Series(3.0, index=index)
    z.iloc[0] = 0.0
    spread = pd.Series(np.linspace(0, 1, n), index=index)
    lam = pd.Series(0.01, index=index)

    generator = _make_generator(use_hawkes_regimes=False)
    signals = generator.generate_signals(
        spread=spread, lambda_intensity=lam,
        jump_indicator=pd.Series(0, index=index), z_score=z,
    )

    assert "max_hold" in generator.exit_reason_counts, (
        f"max_hold never fired; reasons seen: {generator.exit_reason_counts}"
    )


def test_emergency_stop_can_actually_fire():
    """Finding 3.1: the emergency stop could only fire before min_hold."""
    n = 200
    index = pd.bdate_range("2020-01-01", periods=n, tz="UTC")

    z = pd.Series(0.0, index=index)
    z.iloc[10:] = 3.0        # short entry
    z.iloc[40:] = 9.0        # then moves hard against the short
    spread = pd.Series(np.linspace(0, 1, n), index=index)
    lam = pd.Series(0.01, index=index)

    generator = _make_generator(use_hawkes_regimes=False)
    generator.generate_signals(
        spread=spread, lambda_intensity=lam,
        jump_indicator=pd.Series(0, index=index), z_score=z,
    )

    assert "emergency_stop" in generator.exit_reason_counts, (
        f"emergency_stop never fired; reasons seen: {generator.exit_reason_counts}"
    )


def test_exit_priority_is_ordered_most_urgent_first():
    assert EXIT_PRIORITY[0] == "emergency_stop"
    assert EXIT_PRIORITY.index("regime_crisis") < EXIT_PRIORITY.index("mean_reversion")
    assert set(EXIT_PRIORITY) == {
        "emergency_stop", "regime_crisis", "max_hold", "profit_target", "mean_reversion",
    }


def test_jump_entry_is_blocked_in_crisis_regime():
    """
    Finding 3.2: the jump-assisted entry was an `elif` OUTSIDE the CRISIS block
    and the lambda-decay filter, so it entered at a 35% looser threshold
    precisely on jump days -- mid-cascade, the opposite of the stated rule.
    """
    n = 120
    index = pd.bdate_range("2020-01-01", periods=n, tz="UTC")

    # In CRISIS the entry threshold is 2.0 * 1.5 = 3.0 and the jump-assisted
    # threshold is 0.65 * 3.0 = 1.95. z = 2.5 clears the jump threshold but not
    # the normal one, so ONLY the jump path wants in -- which is exactly the
    # back door finding 3.2 describes.
    z = pd.Series(0.0, index=index)
    z.iloc[50] = 2.5

    lam = pd.Series(0.01, index=index)
    lam.iloc[50] = 0.01 * 20  # excess = 19 -> CRISIS

    jumps = pd.Series(0, index=index)
    jumps.iloc[50] = 1

    generator = _make_generator()
    signals = generator.generate_signals(
        spread=pd.Series(np.linspace(0, 1, n), index=index),
        lambda_intensity=lam, jump_indicator=jumps, z_score=z,
    )

    assert signals["signal"].iloc[50] == 0, "jump entry bypassed the CRISIS block"
    assert generator.entries_blocked_by_regime >= 1


def test_calm_regime_is_reachable():
    """
    Finding 3.3: with percentile bucketing, p25 equalled lambda_bar EXACTLY for
    GS/MS and CVX/XOM, so CALM never fired. Excess-based cut-points fix that.
    """
    generator = _make_generator()
    generator.set_lambda_baseline(0.01)

    assert generator._get_regime(0.01) == HawkesRegime.CALM
    assert generator._get_regime(0.01 * 1.5) == HawkesRegime.NORMAL
    assert generator._get_regime(0.01 * 3.0) == HawkesRegime.ELEVATED
    assert generator._get_regime(0.01 * 20.0) == HawkesRegime.CRISIS


def test_regime_is_an_enum():
    """`HawkesRegime` was a dataclass of strings while `SignalType` was an Enum."""
    from enum import Enum

    assert issubclass(HawkesRegime, Enum)
    assert issubclass(SignalType, Enum)


def test_decay_filter_does_not_block_everything_when_intensity_is_flat():
    """
    When alpha ~ 0 the intensity is flat, no decay can ever be observed, and a
    naive 15%-decay requirement blocks EVERY entry for the whole sample.
    """
    n = 200
    index = pd.bdate_range("2020-01-01", periods=n, tz="UTC")

    z = pd.Series(0.0, index=index)
    z.iloc[30:] = 3.0
    lam = pd.Series(0.01, index=index)  # perfectly flat

    generator = _make_generator()
    signals = generator.generate_signals(
        spread=pd.Series(np.linspace(0, 1, n), index=index),
        lambda_intensity=lam, jump_indicator=pd.Series(0, index=index), z_score=z,
    )

    assert (signals["signal"].abs() == 1).sum() > 0, (
        "flat intensity blocked every entry -- the decay filter must only apply "
        "when there is actually a cascade to wait out"
    )
