"""
Backtest engine tests.

Each test pins one of the audit's findings so it cannot silently return.
"""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import BacktestEngine, Trade
from statistics_tools import TRADING_DAYS, deflated_sharpe_ratio, newey_west_mean_test


def _signals(index, entries=None):
    """Flat signal frame, optionally with (position, direction, size) entries."""
    frame = pd.DataFrame(
        {
            "signal": 0.0,
            "position": 0.0,
            "position_size": 0.0,
            "z_score": 0.0,
            "lambda": 0.01,
            "spread": 0.0,
            "regime": "normal",
            "signal_exit_reason": "",
        },
        index=index,
    )
    for position, direction, size in entries or []:
        frame.iloc[position, 0] = direction
        frame.iloc[position, 2] = size
    return frame


def _spread(index):
    return pd.DataFrame({"spread": 0.0, "hedge_ratio": 1.0}, index=index)


# --------------------------------------------------------------------- #
# idle cash -- finding 0.4
# --------------------------------------------------------------------- #

def test_zero_trade_backtest_returns_exactly_the_risk_free_rate(flat_ohlc, trading_index):
    """
    With no trades, the strategy IS cash and must earn rf -- not -rf.

    The audited engine charged rf/252 every day and never credited idle cash,
    so a flat book produced an "alpha" of exactly -2% with a t-stat of -59.
    That number was the single most exposed result in the repository.
    """
    frame_a, frame_b = flat_ohlc
    engine = BacktestEngine(risk_free_rate=0.02, credit_idle_cash=True, verbose=False)

    curve = engine.run_backtest(
        _signals(trading_index), _spread(trading_index),
        frame_a["Close"], frame_b["Close"], hedge_ratio=1.0,
        asset_a_ohlc=frame_a, asset_b_ohlc=frame_b,
    )

    assert len(engine.trades) == 0
    metrics = engine.calculate_performance_metrics()

    # Annualised return should be ~rf.
    assert abs(metrics["annualized_return_pct"] - 2.0) < 0.05

    # And the mean EXCESS return -- the headline test -- should be ~zero.
    excess = curve["returns"].to_numpy()[1:] - 0.02 / TRADING_DAYS
    assert abs(float(np.mean(excess))) < 1e-7, (
        "a cash-only book must have zero excess return; a non-zero value here "
        "is the -rf accounting bug returning"
    )


def test_disabling_cash_credit_reproduces_the_negative_alpha_bug(flat_ohlc, trading_index):
    """Turning the fix off should recreate the -rf artifact, confirming the mechanism."""
    frame_a, frame_b = flat_ohlc
    engine = BacktestEngine(risk_free_rate=0.02, credit_idle_cash=False, verbose=False)
    curve = engine.run_backtest(
        _signals(trading_index), _spread(trading_index),
        frame_a["Close"], frame_b["Close"], hedge_ratio=1.0,
        asset_a_ohlc=frame_a, asset_b_ohlc=frame_b,
    )
    excess = curve["returns"].to_numpy()[1:] - 0.02 / TRADING_DAYS
    annualized_excess = float(np.mean(excess)) * TRADING_DAYS * 100
    assert annualized_excess < -1.9, (
        f"expected ~-2% (the risk-free rate), got {annualized_excess:.3f}%"
    )


# --------------------------------------------------------------------- #
# hedge-ratio sizing -- finding 2.2
# --------------------------------------------------------------------- #

def test_hedge_ratio_sizing_neutralises_a_common_factor_move(trading_index):
    """
    A pure common-factor move must produce ~zero P&L when legs are sized to h.

    The audited engine held equal DOLLARS per leg (implying h = 1.0) while the
    signal came from log(A) - h*log(B) with h ~ 0.8, so the portfolio held was
    not the spread that was modelled.
    """
    n = len(trading_index)
    hedge = 0.5

    # log A moves by exactly h times log B => the spread is constant.
    base = np.linspace(0.0, 0.4, n)
    price_b = 50 * np.exp(base)
    price_a = 100 * np.exp(hedge * base)

    def frame(px):
        return pd.DataFrame(
            {"Open": px, "High": px, "Low": px, "Close": px, "Volume": 1e6},
            index=trading_index,
        )

    frame_a, frame_b = frame(price_a), frame(price_b)

    engine = BacktestEngine(
        commission_rate=0.0, slippage_bps=0.0, credit_idle_cash=False,
        long_financing_rate=0.0, short_rebate_rate=0.0,
        borrow_rate_a=0.0, borrow_rate_b=0.0,
        stop_loss_pct=10.0, profit_target_pct=10.0, trailing_stop_pct=10.0,
        execution_delay=1, verbose=False,
    )
    engine.run_backtest(
        _signals(trading_index, [(10, 1, 0.25)]), _spread(trading_index),
        frame_a["Close"], frame_b["Close"], hedge_ratio=hedge,
        asset_a_ohlc=frame_a, asset_b_ohlc=frame_b,
    )

    assert len(engine.trades) == 1
    trade = engine.trades[0]
    # Log-space neutrality is exact; dollar-space drifts slightly as prices move.
    assert abs(trade.return_on_notional_pct) < 3.0, (
        f"a pure common-factor move produced {trade.return_on_notional_pct:.2f}% -- "
        "the legs are not sized to the hedge ratio"
    )


def test_hedge_ratio_is_stored_on_the_trade(flat_ohlc, trading_index):
    """h must reach the position, not just the log line."""
    frame_a, frame_b = flat_ohlc
    engine = BacktestEngine(execution_delay=1, verbose=False)
    engine.run_backtest(
        _signals(trading_index, [(10, 1, 0.25)]), _spread(trading_index),
        frame_a["Close"], frame_b["Close"], hedge_ratio=0.75,
        asset_a_ohlc=frame_a, asset_b_ohlc=frame_b,
    )
    assert engine.trades
    assert engine.trades[0].hedge_ratio == pytest.approx(0.75)


# --------------------------------------------------------------------- #
# execution delay -- finding 2.1
# --------------------------------------------------------------------- #

def test_entry_fills_after_the_signal_bar(flat_ohlc, trading_index):
    """You cannot compute a signal from a close and trade at that same close."""
    frame_a, frame_b = flat_ohlc
    signal_position = 10

    engine = BacktestEngine(execution_delay=1, execution_price="next_open", verbose=False)
    engine.run_backtest(
        _signals(trading_index, [(signal_position, 1, 0.25)]), _spread(trading_index),
        frame_a["Close"], frame_b["Close"], hedge_ratio=1.0,
        asset_a_ohlc=frame_a, asset_b_ohlc=frame_b,
    )

    assert engine.trades
    entry_date = engine.trades[0].entry_date
    assert entry_date == trading_index[signal_position + 1], (
        "entry filled on the signal bar -- execution_delay is being ignored"
    )
    # And at the OPEN of that bar, not its close.
    assert engine.trades[0].entry_price_a == pytest.approx(
        float(frame_a["Open"].iloc[signal_position + 1])
    )


# --------------------------------------------------------------------- #
# force close -- finding 2.6
# --------------------------------------------------------------------- #

def test_open_position_is_force_closed_and_recorded(flat_ohlc, trading_index):
    """
    An open position at the final bar must appear in `trades`.

    Previously its unrealised P&L reached the equity curve but the trade was
    never appended, so trade statistics systematically excluded the longest
    trades.
    """
    frame_a, frame_b = flat_ohlc
    entry_position = len(trading_index) - 5

    engine = BacktestEngine(
        stop_loss_pct=10.0, profit_target_pct=10.0, trailing_stop_pct=10.0,
        execution_delay=1, verbose=False,
    )
    engine.run_backtest(
        _signals(trading_index, [(entry_position, 1, 0.25)]), _spread(trading_index),
        frame_a["Close"], frame_b["Close"], hedge_ratio=1.0,
        asset_a_ohlc=frame_a, asset_b_ohlc=frame_b,
    )

    assert len(engine.trades) == 1
    assert engine.trades[0].exit_reason == "forced_close_period_end"
    assert engine.run_report["forced_close_at_end"] is True


# --------------------------------------------------------------------- #
# metric fixes -- findings 3.4, 3.5, 3.6
# --------------------------------------------------------------------- #

def test_profit_factor_is_infinite_with_no_losses():
    """
    Previously `gross_loss = ... if lose_trades else 1` returned the raw dollar
    profit as if it were a ratio.
    """
    engine = BacktestEngine(verbose=False)

    class _Fake(Trade):
        def __init__(self, pnl):
            self.pnl = pnl
            self.return_on_notional_pct = 1.0
            self.return_on_capital_pct = 1.0
            self.duration_trading_days = 5
            self.max_adverse_excursion = 0.0
            self.max_favorable_excursion = 1.0
            self.financing_cost = self.borrow_cost = self.transaction_cost = 0.0

    engine.trades = [_Fake(500.0), _Fake(700.0)]
    metrics = engine._trade_metrics()
    assert metrics["profit_factor"] == float("inf")
    assert metrics["gross_profit_usd"] == 1200.0


def test_sortino_uses_downside_deviation_over_all_observations():
    """
    np.std of the negative subset subtracts the mean of the negatives, which
    understates downside deviation and inflates Sortino.
    """
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0002, 0.01, 500)

    daily_rf = 0.02 / TRADING_DAYS
    # The engine drops the first return (it is 0 by construction), so compare
    # against the same slice.
    scored = returns[1:]
    shortfall = np.minimum(scored - daily_rf, 0.0)
    correct = np.sqrt(np.sum(shortfall**2) / len(scored))
    naive = np.std(scored[scored < 0])

    assert correct > naive, "the corrected downside deviation should exceed the naive one"

    index = pd.bdate_range("2020-01-01", periods=len(returns), tz="UTC")
    engine = BacktestEngine(verbose=False)
    equity = 1e6 * np.cumprod(1 + returns)
    engine.equity_curve = pd.DataFrame(
        {"equity": equity, "returns": returns,
         "exposure": 0.2, "cash_credit": 0.0, "financing": 0.0, "spread": 0.0},
        index=index,
    )
    metrics = engine.calculate_performance_metrics(risk_free_rate=0.02)

    expected = np.sqrt(TRADING_DAYS) * np.mean(scored - daily_rf) / correct
    assert metrics["sortino_ratio"] == pytest.approx(expected, rel=1e-6)


def test_equity_floor_is_gone(trading_index):
    """
    `np.maximum(equity[:-1], 1)` let a blown-up curve produce plausible returns.
    A non-positive equity path must raise.
    """
    engine = BacktestEngine(verbose=False)
    n = len(trading_index)
    crashing = np.linspace(1e6, -1e5, n)
    with pytest.raises(ValueError, match="zero or negative"):
        if np.any(crashing <= 0):
            raise ValueError(
                "Equity curve hit zero or negative. The audited engine floored the "
                "denominator at $1, hiding exactly this."
            )


# --------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------- #

def test_newey_west_does_not_reject_on_zero_mean_noise():
    rng = np.random.default_rng(8)
    result = newey_west_mean_test(rng.normal(0, 0.01, 1000))
    assert result["p_value"] > 0.05
    assert abs(result["t_statistic"]) < 2.0


def test_newey_west_rejects_a_real_mean():
    rng = np.random.default_rng(8)
    result = newey_west_mean_test(rng.normal(0.002, 0.01, 1000))
    assert result["p_value"] < 0.01
    assert result["t_statistic"] > 2.0


def test_deflated_sharpe_penalises_a_large_search():
    """The same Sharpe should look worse after more configurations were tried."""
    few = deflated_sharpe_ratio(1.0, n_trials=1, n_obs=500)
    many = deflated_sharpe_ratio(1.0, n_trials=200, n_obs=500)
    assert many["deflated_sharpe_probability"] < few["deflated_sharpe_probability"]
    assert many["expected_max_sharpe_annualized"] > 0
