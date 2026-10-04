"""
Regression tests for the second round of fixes:

    1. `hedge_ratio_method="johansen"` dispatched to Engle-Granger OLS
    2. `commission_rate` was 20bp under a "2bp" comment
    3. `max_hold_fraction` was never passed to the signal generator
    6. `is_tradeable` was computed but never blocked trading
    7. look-ahead in sizing, stops and jump flags
"""

import copy

import numpy as np
import pandas as pd
import pytest

import pipeline as pipeline_module
import walk_forward as walk_forward_module
from backtest_engine import BacktestEngine
from config import Config
from equity_pairs_loader import EquityPairsDataPipeline
from pipeline import PairPipeline
from signal_generation import TradingSignals
from walk_forward import WalkForwardEngine


def _fast_config(pair: str) -> Config:
    cfg = Config().for_pair(pair)
    cfg.hawkes.lr_bootstrap_reps = 0
    cfg.visualization.save_plots = False
    return cfg


@pytest.fixture(scope="module")
def cvx_xom_fit():
    cfg = _fast_config("CVX_XOM")
    tv = cfg.train_val
    pipe = PairPipeline(cfg, verbose=False)
    pipe.acquire_data(train_end=tv.train_end)
    spread = pipe.spread_df
    bundle = pipe.fit_models(spread.loc[tv.train_start:tv.train_end], train_end=tv.train_end)
    artifacts = pipe.compute_artifacts(spread, bundle)
    return cfg, pipe, bundle, artifacts


class _Recorder(TradingSignals):
    """TradingSignals that remembers the keyword arguments it was built with."""

    calls: list = []

    def __init__(self, **kwargs):
        _Recorder.calls.append(kwargs)
        super().__init__(**kwargs)


# --------------------------------------------------------------------- #
# 1. Johansen
# --------------------------------------------------------------------- #

def test_johansen_method_uses_the_johansen_vector():
    """The 'johansen' hedge used to equal OLS(A~B) to 14 digits."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    loader = EquityPairsDataPipeline(
        "OHLCV_AMD.csv", "OHLCV_NVDA.csv", "AMD", "NVDA", verbose=False
    )
    loader.load_from_csv(date_columns="ts_event")
    loader.clean_data()
    a = loader.data["asset_a"]["Close"].loc[:"2022-12-31"]
    b = loader.data["asset_b"]["Close"].loc[:"2022-12-31"]

    h, diag = loader.estimate_hedge_ratio_static(a, b, method="johansen")

    vec = coint_johansen(
        np.column_stack([np.log(a.to_numpy()), np.log(b.to_numpy())]), 0, 1
    ).evec[:, 0]
    assert h == pytest.approx(-vec[1] / vec[0])
    assert "johansen_trace_stat" in diag
    assert abs(h - diag["h_ols_a_on_b"]) > 0.05, "johansen still equals OLS"


# --------------------------------------------------------------------- #
# 2. costs
# --------------------------------------------------------------------- #

def test_commission_default_is_two_basis_points():
    bt = Config().backtest
    assert bt.commission_rate == pytest.approx(0.0002)
    round_trip = 2 * (bt.commission_rate + bt.slippage_bps / 1e4)
    assert round_trip == pytest.approx(0.0006)


# --------------------------------------------------------------------- #
# 3. holding period wiring
# --------------------------------------------------------------------- #

def test_signal_generator_default_matches_config():
    assert TradingSignals(verbose=False).max_hold_fraction == Config().trading.max_hold_fraction


def test_train_val_passes_hold_fractions(cvx_xom_fit, monkeypatch):
    cfg, pipe, bundle, artifacts = cvx_xom_fit
    _Recorder.calls = []
    monkeypatch.setattr(pipeline_module, "TradingSignals", _Recorder)

    tv = cfg.train_val
    pipe.evaluate_period(
        pipe.spread_df, pipe.cleaned_data, artifacts, bundle,
        tv.val_start, tv.val_end, "val", light=True,
    )
    kwargs = _Recorder.calls[-1]
    assert kwargs["min_hold_fraction"] == cfg.trading.min_hold_fraction
    assert kwargs["target_hold_fraction"] == cfg.trading.target_hold_fraction
    assert kwargs["max_hold_fraction"] == cfg.trading.max_hold_fraction
    assert kwargs["max_holding_period_cap"] == cfg.trading.max_holding_period_cap


# --------------------------------------------------------------------- #
# 6. validation gate
# --------------------------------------------------------------------- #

def test_failing_pair_opens_no_positions(cvx_xom_fit):
    """CVX/XOM fails Engle-Granger on the training window."""
    cfg, pipe, bundle, artifacts = cvx_xom_fit
    assert bundle.is_tradeable is False

    tv = cfg.train_val
    res = pipe.evaluate_period(
        pipe.spread_df, pipe.cleaned_data, artifacts, bundle,
        tv.val_start, tv.val_end, "val", light=True,
    )
    assert res["metrics"]["total_trades"] == 0
    assert res["signal_quality"]["entries_blocked_by_validation"] > 0

    ungated_cfg = copy.deepcopy(cfg)
    ungated_cfg.trading.require_tradeable = False
    ungated = PairPipeline(ungated_cfg, verbose=False)
    ungated.loader, ungated.spread_df, ungated.cleaned_data = (
        pipe.loader, pipe.spread_df, pipe.cleaned_data
    )
    res = ungated.evaluate_period(
        pipe.spread_df, pipe.cleaned_data, artifacts, bundle,
        tv.val_start, tv.val_end, "val", light=True,
    )
    assert res["metrics"]["total_trades"] > 0


def test_untradeable_quarter_is_skipped_by_tuning(cvx_xom_fit):
    cfg, pipe, bundle, artifacts = cvx_xom_fit
    tv = cfg.train_val
    *_, n_trials, trials = pipe.tune_thresholds(
        pipe.spread_df, pipe.cleaned_data, artifacts, bundle, tv.train_start, tv.train_end
    )
    assert n_trials == 0 and trials == []


# --------------------------------------------------------------------- #
# 7. look-ahead
# --------------------------------------------------------------------- #

def test_jump_flags_do_not_depend_on_later_data(cvx_xom_fit):
    """Truncating the sample must not change any earlier flag."""
    cfg, pipe, bundle, artifacts = cvx_xom_fit
    full = artifacts["jump_df"]["jump_indicator"]

    cut = pipe.spread_df.loc[:"2023-12-31"]
    truncated = pipe.compute_artifacts(cut, bundle)["jump_df"]["jump_indicator"]

    common = truncated.index
    pd.testing.assert_series_equal(full.loc[common], truncated, check_names=False)


def test_training_flags_are_reproduced_out_of_sample(cvx_xom_fit):
    """Frozen normalisers and cutoff reproduce the fitted training flags."""
    cfg, pipe, bundle, artifacts = cvx_xom_fit
    tv = cfg.train_val
    in_train = artifacts["jump_df"]["jump_indicator"].loc[tv.train_start:tv.train_end]
    expected = bundle.n_jumps_fdr if bundle.detection_basis == "fdr" else bundle.n_jumps_nominal
    assert int(in_train.sum()) == expected


def _two_trade_signals(index, hedges, stop_sd=None):
    n = len(index)
    sig = np.zeros(n)
    sig[[10, 60]] = 1.0
    sig[[40, 90]] = 2.0
    frame = pd.DataFrame(
        {"signal": sig, "position": 0.0, "position_size": 0.2, "z_score": 0.0,
         "lambda": 0.01, "spread": 0.0, "regime": "normal", "signal_exit_reason": "",
         "hedge_ratio": np.where(np.arange(n) < 50, hedges[0], hedges[1])},
        index=index,
    )
    if stop_sd is not None:
        frame["stop_reference_sd"] = stop_sd
    return frame


def test_each_position_is_sized_with_its_own_hedge(trading_index, flat_ohlc):
    frame_a, frame_b = flat_ohlc
    spread = pd.DataFrame({"spread": 0.0}, index=trading_index)
    engine = BacktestEngine(verbose=False)
    engine.run_backtest(
        _two_trade_signals(trading_index, (0.6, 1.4)), spread,
        frame_a["Close"], frame_b["Close"], hedge_ratio=1.0,
        asset_a_ohlc=frame_a, asset_b_ohlc=frame_b,
    )
    assert [round(t.hedge_ratio, 6) for t in engine.trades[:2]] == [0.6, 1.4]


def test_stops_use_the_frozen_reference_not_the_traded_window(trading_index, flat_ohlc):
    frame_a, frame_b = flat_ohlc
    rng = np.random.default_rng(3)
    levels = []
    for sd in (0.01, 0.30):
        spread = pd.DataFrame({"spread": rng.normal(0, sd, len(trading_index))},
                              index=trading_index)
        engine = BacktestEngine(stop_mode="volatility", verbose=False)
        engine.run_backtest(
            _two_trade_signals(trading_index, (1.0, 1.0)), spread,
            frame_a["Close"], frame_b["Close"], hedge_ratio=1.0,
            asset_a_ohlc=frame_a, asset_b_ohlc=frame_b, stop_reference_sd=0.05,
        )
        assert engine.resolved_stops["sd_source"] == "frozen_training_sd"
        levels.append(engine.stop_loss_pct)
    assert levels[0] == pytest.approx(levels[1])


@pytest.fixture(scope="module")
def short_walk_forward():
    """Two or three quarters at the end of the sample, recorded generator kwargs."""
    cfg = _fast_config("GS_MS")
    cfg.walk_forward.min_train_days = 1830
    cfg.walk_forward.tune_thresholds = False
    _Recorder.calls = []
    original = walk_forward_module.TradingSignals
    walk_forward_module.TradingSignals = _Recorder
    try:
        engine = WalkForwardEngine(cfg, verbose=False)
        results = engine.run()
    finally:
        walk_forward_module.TradingSignals = original
    return cfg, engine, results, list(_Recorder.calls)


def test_walk_forward_passes_hold_fractions(short_walk_forward):
    cfg, _, _, calls = short_walk_forward
    assert calls
    for kwargs in calls:
        assert kwargs["max_hold_fraction"] == cfg.trading.max_hold_fraction
        assert kwargs["min_hold_fraction"] == cfg.trading.min_hold_fraction


def test_walk_forward_carries_per_quarter_sizing_inputs(short_walk_forward):
    _, engine, results, _ = short_walk_forward
    signals = results["signals"]
    for q in engine.quarterly_results:
        window = signals.loc[q["eval_start"]:q["eval_end"]]
        assert (window["hedge_ratio"] == q["hedge_ratio"]).all()
        assert (window["stop_reference_sd"] == q["stop_reference_sd"]).all()
    for trade in results["engine"].trades:
        assert trade.hedge_ratio in {q["hedge_ratio"] for q in engine.quarterly_results}


def test_flat_book_reports_zero_sharpe_not_float_noise(trading_index, flat_ohlc):
    """A gated pair earns exactly rf; its Sharpe must be 0, not noise / noise."""
    frame_a, frame_b = flat_ohlc
    signals = _two_trade_signals(trading_index, (1.0, 1.0))
    signals["signal"] = 0.0
    engine = BacktestEngine(verbose=False)
    engine.run_backtest(
        signals, pd.DataFrame({"spread": 0.0}, index=trading_index),
        frame_a["Close"], frame_b["Close"], hedge_ratio=1.0,
    )
    metrics = engine.calculate_performance_metrics()
    assert metrics["total_trades"] == 0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["sortino_ratio"] == 0.0


def test_ungated_runs_never_overwrite_gated_artifacts():
    import main

    gated, ungated = Config(), Config()
    ungated.trading.require_tradeable = False
    assert main._dir_suffix(gated) == ""
    assert main._dir_suffix(ungated) == "_ungated"


def test_zero_trade_runs_overwrite_previous_trade_files(tmp_path):
    """An empty trade list must replace, not silently keep, an older file."""
    from results_io import ResultsWriter

    stale = tmp_path / "trades.csv"
    stale.write_text("pnl\n123\n")
    writer = ResultsWriter(tmp_path, verbose=False)
    writer.write_frame(BacktestEngine(verbose=False).get_trade_summary(), "trades.csv", index=False)
    written = pd.read_csv(stale)
    assert len(written) == 0 and "pnl" in written.columns
    assert "trades.csv" in writer.written

    stale.write_text("x\n1\n")
    writer.write_frame(pd.DataFrame(), "trades.csv")
    assert not stale.exists()
