""" Backtest engine for the hedge-ratio-sized pair strategy"""

from __future__ import annotations 

from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional 

import numpy as np 
import pandas as pd 

from statistics_tools import (
    TRADING_DAYS,
    VOL_EPS,
    bootstrap_metric_ci,
    newey_west_mean_test,
    sharpe_standard_error,
)

__all__ = ["Trade", "BacktestEngine", "compute_alpha_tsta"]

@dataclass 
class Trade:
    """ One round-trip trade"""

    entry_date: pd.Timestamp 
    exit_date: pd.Timestamp 
    direction: int # 1 = long spread, -1 = short spread 
    entry_price_a: float 
    entry_price_b: float 
    exit_price_a: float 
    exit_price_b: float 
    entry_z: float 
    exit_z: float 
    position_size: float 
    hedge_ratio: float 
    gross_notional: float 
    capital_at_risk: float 
    pnl: float 
    return_on_notional_pct: float 
    return_on_capital_pct: float 
    duration_trading_days: int 
    entry_lambda: float
    exit_lambda: float 
    entry_regime: str 
    exit_regime: str 
    max_adverse_excursion: float 
    max_favorable_excursion: float 
    financing_cost: float 
    borrow_cost: float 
    transaction_cost: float 
    exit_reason: str 

def compute_alpha_tstat(
        equity_curve: pd.DataFrame,
        benchmark_csv: str, 
        risk_free_rate: float = 0.02,
) -> Dict:
    """
    CAPM regression of strategy excess returns on benchmark excess returns 
    
    RETAINED ONLY TO DEMONSTRATE MARKET NEUTRALITY. For a dollar-neutral book 
    beta ~ 0 by construction, so the intercept is not a measure of skill -- in 
    the audited results it equalled -rf for every pair with R-squared ~ 0.0005
    because idle cash earned nothing. Use 'newey_west_mean_test' on the excess
    return series for the headline. 
    
    NOTE: the SPY benchmark CSV is a PRICE series, not total return, so the
    benchmark is understated by roughly its dividend yield.
    """

    import statsmodels.api as sm 

    raw = pd.read_csv(benchmark_csv)
    date_col = next(
        (c for c in raw.columns if str(c).lower() in ("ts_event", "date", "datetime")), None 
    )
    if date_col:
        raw[date_col] = pd.to_datetime(raw[date_col], errors = "coerce")
        raw = raw.set_index(date_col)
    raw.columns = [str(c).capitalize() for c in raw.columns]

    bench_returns = raw["Close"].dropna().pct_change().dropna()
    strat_returns = equity_curve["returns"].copy()

    def strip_tz(s: pd.Series) -> pd.Series:
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
            s = s.copy()
            s.index = s.index.tz_convert(None)
        if isinstance(s.index, pd.DatetimeIndex):
            s.index = s.index.normalize()
        return s 
    
    strat_returns = strip_tz(strat_returns)
    bench_returns = strip_tz(bench_returns)

    common = strat_returns.index.intersection(bench_returns.index)
    empty = {
        "alpha_daily": 0.0, "alpha_annualized": 0.0, "alpha_tstat": 0.0,
        "alpha_pvalue": 1.0, "beta": 0.0, "beta_tstat": 0.0, "r_squared": 0.0,
        "information_ratio": 0.0, "n_observations": len(common),
        "capm_note": "Shown only to demonstrate neutrality; not the headline test.",
    }
    if len(common) < 30:
        return empty 

    strat = strat_returns.loc[common].to_numpy(dtype = float)
    bench = bench_returns.loc[common].to_numpy(dtype = float)
    if float(np.std(strat)) < 1e-12:
        return empty 
    
    daily_rf = risk_free_rate / TRADING_DAYS
    model = sm.OLS(
        strat - daily_rf, sm.add_constant(bench - daily_rf)
    ).fit(cov_type = "HAC", cov_kwds={"maxlags": 5})

    alpha_daily = float(model.params[0])
    tracking_error = float(np.std(np.asarray(model.resid, dtype = float)) * np.sqrt(TRADING_DAYS))

    return {
        "alpha_daily": alpha_daily,
        "alpha_annualized": alpha_daily * TRADING_DAYS,
        "alpha_tstat": float(model.tvalues[0]),
        "alpha_pvalue": float(model.pvalues[0]),
        "beta": float(model.params[1]),
        "beta_tstat": float(model.tvalues[1]),
        "r_squared": float(model.rsquared),
        "information_ratio": (alpha_daily * TRADING_DAYS / tracking_error)
        if tracking_error > 0
        else 0.0,
        "n_observations": int(len(common)),
        "capm_note": "Shown only to demonstrate neutrality; not the headline test.",
    }

class _OpenPosition:
    """ Mutable state for the currently-held position"""

    __slots__ = (
        "direction", "shares_a", "shares_b", "entry_price_a", "entry_price_b",
        "entry_date", "entry_idx", "entry_lambda", "entry_z", "entry_regime",
        "gross_notional", "capital_at_risk", "position_size",
        "max_adverse", "max_favorable", "trailing_stop_level", "trailing_activated",
        "financing", "borrow", "costs", "hedge_ratio",
        "stop_loss_pct", "trailing_stop_pct", "trailing_activation_pct", "profit_target_pct",
    )

    def __init__(self, **kwargs):
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key, 0.0))

class BacktestEngine:
    """ Event-driven backtest with realistic execution, financing, and reporting"""

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.0002,
        slippage_bps: float = 1.0,
        max_position_pct: float = 0.25,
        stop_loss_pct: float = 0.03,
        trailing_stop_pct: float = 0.015,
        trailing_activation_pct: float = 0.01,
        profit_target_pct: float = 0.06,
        execution_delay: int = 1,
        execution_price: str = "next_open",
        risk_free_rate: float = 0.02,
        credit_idle_cash: bool = True,
        long_financing_rate: float = 0.02,
        short_rebate_rate: float = 0.05,
        borrow_rate_a: float = 0.003,
        borrow_rate_b: float = 0.003,
        use_intraday_stops: bool = True,
        stop_mode: str = "fixed",
        stop_loss_sigma: float = 6.0,
        trailing_stop_sigma: float = 4.0,
        trailing_activation_sigma: float = 2.5,
        profit_target_sigma: float = 10.0,
        stop_floor_pct: float = 0.01,
        stop_cap_pct: float = 0.50,
        verbose: bool = True,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage_bps / 10_000.0
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_activation_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.profit_target_pct = profit_target_pct

        self.execution_delay = max(int(execution_delay), 0)
        self.execution_price = execution_price 
        self.risk_free_rate = risk_free_rate
        self.credit_idle_cash = credit_idle_cash

        self.long_financing_rate = long_financing_rate
        self.short_rebate_rate = short_rebate_rate
        self.borrow_rate_a = borrow_rate_a
        self.borrow_rate_b = borrow_rate_b
        self.use_intraday_stops = use_intraday_stops

        # Volatility scaled stops. See BacktestConfig.stop_mode for why this is 
        # the default: fixed percentage stops are in structural conflict with a 
        # mean-reverting spread, which by definition moves against the position
        # before it reverts
        self.stop_mode = stop_mode 
        self.stop_loss_sigma = stop_loss_sigma
        self.trailing_stop_sigma = trailing_stop_sigma
        self.trailing_activation_sigma = trailing_activation_sigma
        self.profit_target_sigma = profit_target_sigma
        self.stop_floor_pct = stop_floor_pct 
        self.stop_cap_pct = stop_cap_pct 
        self.resolved_stops: Dict = {}

        self.verbose = verbose 

        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.DataFrame] = None 
        self.performance_metrics: Dict = {}
        self.half_life: Optional[float] = None 
        self.run_report: Dict = []

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    def set_half_life(self, half_life: float) -> None:
        self.half_life = half_life 

    ### Helpers ###

    def _stop_levels(self, spread_sd: float, hedge_ratio: float) -> Optional[Dict]:
        """
        Volatility-scaled stop levels for one spread standard deviation.

        A one-unit move in the log spread produces a return of 1/(1+h) on gross
        notional, so a stop at k stationary sigma is k * sd(spread) / (1 + h).
        Returns None when the volatility estimate is degenerate.
        """
        stationary_sd = float(spread_sd) / (1.0 + abs(float(hedge_ratio)))
        if not np.isfinite(stationary_sd) or stationary_sd <= 0:
            return None

        def clamp(x: float) -> float:
            return float(min(max(x, self.stop_floor_pct), self.stop_cap_pct))

        return {
            "stationary_sd_pct": 100 * stationary_sd,
            "stop_loss_pct": clamp(self.stop_loss_sigma * stationary_sd),
            "trailing_stop_pct": clamp(self.trailing_activation_sigma * stationary_sd),
            "trailing_activation_pct": clamp(self.trailing_activation_sigma * stationary_sd),
            "profit_target_pct": clamp(self.profit_target_sigma * stationary_sd),
        }

    def _resolve_stops(
        self,
        spread_df: pd.DataFrame,
        hedge_ratio: float,
        reference_sd: Optional[float] = None,
    ) -> None:
        """
        Set the default stop levels used for this run.

        In 'volatility' mode the levels are multiples of the spread's 
        STATIONARY standard deviation, converted to position-return units.
        Stationary sigma, not daily sigma, is the right scale: entry happens at 
        z = 2, so a stop at k = 4 fires only after a further 2 sigma of adverse
        movement. Using daily sigma at a 60-day half-life gives ~3%, which 
        reproduces the very defect this is meant to fix -- the spread moves 
        several percent against the position before reverting, because that is
        what mean reversion is.

        `reference_sd` should be the spread standard deviation from the
        TRAINING window. Measuring it on the window being traded sizes every
        stop with knowledge of that window's realised volatility, a look-ahead.
        The evaluation-window value is used only as a labelled fallback when no
        reference is supplied (unit tests, ad-hoc use).
        """
        if self.stop_mode != "volatility":
            self.resolved_stops = {
                "mode": "fixed",
                "stop_loss_pct": self.stop_loss_pct,
                "trailing_stop_pct": self.trailing_stop_pct,
                "trailing_activation_pct": self.trailing_activation_pct,
                "profit_target_pct": self.profit_target_pct,
            }
            return 

        if reference_sd is not None:
            spread_sd, source = float(reference_sd), "frozen_training_sd"
        else:
            spread_sd, source = float(spread_df["spread"].std()), "evaluation_window_sd"

        levels = self._stop_levels(spread_sd, hedge_ratio)
        if levels is None:
            self._log(" WARNING: degenerate spread volatility; falling back to fixed stops")
            self.resolved_stops = {"mode": "fixed_fallback"}
            return 

        self.stop_loss_pct = levels["stop_loss_pct"]
        self.trailing_stop_pct = levels["trailing_stop_pct"]
        self.trailing_activation_pct = levels["trailing_activation_pct"]
        self.profit_target_pct = levels["profit_target_pct"]

        self.resolved_stops = {"mode": "volatility", "sd_source": source, **levels}
        self._log(
            f" Stops (scaled to stationary sigma {levels['stationary_sd_pct']:.2f}%, "
            f"{source}): stop {100 * self.stop_loss_pct:.2f}%, "
            f"trail {100 * self.trailing_stop_pct:.2f}%, "
            f"target {100 * self.profit_target_pct:.2f}%"
        )

    @staticmethod 
    def _pnl(pos: "_OpenPosition", price_a: float, price_b: float) -> float:
        return pos.shares_a * (price_a - pos.entry_price_a) + pos.shares_b * (
            price_b - pos.entry_price_b
        )

    def _net_return(self, pos: "_OpenPosition", price_a: float, price_b: float) -> float:
        """ Mark-to-market return net of accrued financing, borrow, and costs"""
        if pos.gross_notional <= 0:
            return 0.0
        net = self._pnl(pos, price_a, price_b) - pos.financing - pos.borrow - pos.costs
        return net / pos.gross_notional 

    def _worst_intraday_return(
            self, pos: "_OpenPosition",
            low_a: float, high_a: float, low_b: float, high_b: float,
    ) -> float:
        """ Worst mark the position could have touched inside the bar"""
        worst_a = low_a if pos.shares_a > 0 else high_a
        worst_b = low_b if pos.shares_b > 0 else high_b 
        return self._net_return(pos, worst_a, worst_b)

    def _close_position(
        self, pos: "_OpenPosition", i: int, date: pd.Timestamp,
        price_a: float, price_b: float, reason: str,
        exit_z: float, exit_lambda: float, exit_regime: str,
    ) -> float:
        """ Close, record the Trade, and return realized PnL. The ONLY exit path"""
        exit_costs = (abs(pos.shares_a * price_a) + abs(pos.shares_b * price_b)) * (
            self.commission_rate + self.slippage
        )
        total_costs = pos.costs + exit_costs 
        realized = self._pnl(pos, price_a, price_b) - total_costs - pos.financing - pos.borrow

        self.trades.append(
            Trade(
                entry_date = pos.entry_date,
                exit_date = date, 
                direction = int(pos.direction),
                entry_price_a = pos.entry_price_a,
                entry_price_b = pos.entry_price_b,
                exit_price_a = price_a,
                exit_price_b = price_b,
                entry_z = pos.entry_z,
                exit_z = exit_z,
                position_size = pos.position_size,
                hedge_ratio = float(pos.hedge_ratio),
                gross_notional = pos.gross_notional,
                capital_at_risk = pos.capital_at_risk,
                pnl = realized,
                return_on_notional_pct=(realized / pos.gross_notional * 100)
                if pos.gross_notional > 0 else 0.0,
                return_on_capital_pct=(realized / pos.capital_at_risk * 100)
                if pos.capital_at_risk > 0 else 0.0,
                duration_trading_days = int(i - pos.entry_idx),
                entry_lambda=pos.entry_lambda,
                exit_lambda = exit_lambda,
                entry_regime = str(pos.entry_regime),
                exit_regime = exit_regime,
                max_adverse_excursion = pos.max_adverse * 100,
                max_favorable_excursion=pos.max_favorable * 100,
                financing_cost = pos.financing, 
                borrow_cost = pos.borrow,
                transaction_cost = total_costs,
                exit_reason = reason,
            )
        )
        return realized 

    ####

    def run_backtest(
        self,
        signals_df: pd.DataFrame,
        spread_df: pd.DataFrame,
        asset_a_prices: pd.Series,
        asset_b_prices: pd.Series,
        hedge_ratio: Optional[float] = None,
        asset_a_ohlc: Optional[pd.DataFrame] = None, 
        asset_b_ohlc: Optional[pd.DataFrame] = None,
        stop_reference_sd: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Run the backtest. `hedge_ratio` drives POSITION SIZING, not just logging.

        Per-bar overrides, read from `signals_df` when present:
            hedge_ratio        -- hedge used to size a position entered on that
                                  bar (walk-forward passes each quarter's own h,
                                  so no position is sized with a later estimate)
            stop_reference_sd  -- training spread sd used to scale that
                                  position's stops
        Otherwise `hedge_ratio` and `stop_reference_sd` apply to every position.
        """
        common = signals_df.index
        for other in (spread_df.index, asset_a_prices.index, asset_b_prices.index):
            common = common.intersection(other)

        signals_df = signals_df.loc[common]
        spread_df = spread_df.loc[common]
        close_a = asset_a_prices.loc[common].astype(float)
        close_b = asset_b_prices.loc[common].astype(float)

        if hedge_ratio is None:
            hedge_ratio = (
                float(spread_df["hedge_ratio"].iloc[-1])
                if "hedge_ratio" in spread_df.columns
                else 1.0
            )
        hedge_ratio = float(hedge_ratio)

        has_ohlc = asset_a_ohlc is not None and asset_b_ohlc is not None
        if has_ohlc: 
            oa, ob = asset_a_ohlc.loc[common], asset_b_ohlc.loc[common]
            open_a, open_b = oa["Open"].astype(float), ob["Open"].astype(float)
            high_a, low_a = oa["High"].astype(float), oa["Low"].astype(float)
            high_b, low_b = ob["High"].astype(float), ob["Low"].astype(float)
        else:
            open_a, open_b = close_a, close_b
            high_a = low_a = close_a 
            high_b = low_b = close_b 

        use_open = has_ohlc and self.execution_price == "next_open"

        self._resolve_stops(spread_df, hedge_ratio, stop_reference_sd)

        hedge_values = (
            signals_df["hedge_ratio"].astype(float).to_numpy()
            if "hedge_ratio" in signals_df.columns
            else np.full(len(common), hedge_ratio)
        )
        stop_sd_values = (
            signals_df["stop_reference_sd"].astype(float).to_numpy()
            if "stop_reference_sd" in signals_df.columns
            else None
        )
        per_bar_hedge = "hedge_ratio" in signals_df.columns

        self._log(
            f"Backtest: h = {'per-bar' if per_bar_hedge else f'{hedge_ratio:.4f}'} (sized), delay = {self.execution_delay} bar(s), "
            f"fill at {'next open' if use_open else 'close'}, "
            f"intraday stops {'on' if (self.use_intraday_stops and has_ohlc) else 'off'}"
        )

        n = len(common)
        if n < 5:
            raise ValueError(f"Only {n} aligned observations; cannot backtest")

        signal_values = signals_df["signal"].to_numpy()
        size_values = signals_df["position_size"].abs().to_numpy()
        lambda_values = (
            signals_df["lambda"].to_numpy() if "lambda" in signals_df.columns else np.zeros(n)
        )
        z_values = (
            signals_df["z_score"].to_numpy() if "z_score" in signals_df.columns else np.zeros(n)
        )
        regime_values = (
            signals_df["regime"].astype(str).to_numpy()
            if "regime" in signals_df.columns
            else np.array(["normal"] * n)
        )
        signal_exit_reasons = (
            signals_df["signal_exit_reason"].astype(str).to_numpy()
            if "signal_exit_reason" in signals_df.columns
            else np.array([""] * n)
        )

        equity = np.zeros(n)
        equity[0] = self.initial_capital 
        cash = self.initial_capital
        daily_rf = self.risk_free_rate / TRADING_DAYS

        exposure_track = np.zeros(n)
        cash_credit_track = np.zeros(n)
        financing_track = np.zeros(n)

        pos: Optional[_OpenPosition] = None 
        pending_entry: Optional[dict] = None 
        pending_exit: Optional[dict] = None 

        def fill(i: int, leg: str) -> float:
            if use_open:
                return float(open_a.iloc[i] if leg == "a" else open_b.iloc[i])
            return float(close_a.iloc[i] if leg == "a" else close_b.iloc[i])

        for i in range(1, n):
            date = pd.Timestamp(common[i])
            price_a, price_b = float(close_a.iloc[i]), float(close_b.iloc[i])

            # 1. accrue and mark 
            if pos is not None:
                long_value = max(pos.shares_a, 0) * price_a + max(pos.shares_b, 0) * price_b
                short_value = (
                    abs(min(pos.shares_a, 0)) * price_a + abs(min(pos.shares_b, 0)) * price_b
                )
                financing = long_value * self.long_financing_rate / TRADING_DAYS
                rebate = short_value * self.short_rebate_rate / TRADING_DAYS
                borrow_rate = self.borrow_rate_a if pos.shares_a < 0 else self.borrow_rate_b
                borrow = short_value * borrow_rate / TRADING_DAYS

                pos.financing += financing - rebate 
                pos.borrow += borrow 
                financing_track[i] = financing - rebate + borrow 

                current_return = self._net_return(pos, price_a, price_b)
                pos.max_adverse = min(pos.max_adverse, current_return)
                pos.max_favorable = max(pos.max_favorable, current_return)

                if current_return > pos.trailing_activation_pct:
                    pos.trailing_activated = True 
                if pos.trailing_activated:
                    pos.trailing_stop_level = max(
                        pos.trailing_stop_level, pos.max_favorable - pos.trailing_stop_pct
                    )

            # 2. decide on an exit 
            if pos is not None and pending_exit is None:
                reason = None 
                forced_a = forced_b = None 

                touched = (
                    self._worst_intraday_return(
                        pos, float(low_a.iloc[i]), float(high_a.iloc[i]),
                        float(low_b.iloc[i]), float(high_b.iloc[i]),
                    )
                    if (self.use_intraday_stops and has_ohlc)
                    else self._net_return(pos, price_a, price_b)
                )

                if touched <= pos.trailing_stop_level:
                    reason = "trailing_stop" if pos.trailing_activated else "stop_loss"
                    # Gap through: if the open already breached, fill there
                    open_ret = self._net_return(
                        pos, float(open_a.iloc[i]), float(open_b.iloc[i])
                    )
                    if open_ret <= pos.trailing_stop_level:
                        forced_a, forced_b = float(open_a.iloc[i]), float(open_b.iloc[i])
                    else:
                        forced_a, forced_b = price_a, price_b 
                elif self._net_return(pos, price_a, price_b) > pos.profit_target_pct:
                    reason = "profit_target"
                    forced_a, forced_b = price_a, price_b

                if reason is not None:
                    # Stops and targets are resting orders: they fill when touched 
                    pending_exit = {
                        "reason": reason, "execute_at": i,
                        "price_a": forced_a, "price_b": forced_b,
                    }
                elif signal_values[i] == 2:
                    pending_exit = {
                        "reason": str(signal_exit_reasons[i]) or "signal",
                        "execute_at": i + self.execution_delay,
                        "price_a": None, "price_b": None,
                    }

            # 3. Execute and Exit
            if pos is not None and pending_exit is not None and i >= pending_exit["execute_at"]:
                px_a = pending_exit["price_a"] if pending_exit["price_a"] is not None else fill(i, "a")
                px_b = pending_exit["price_b"] if pending_exit["price_b"] is not None else fill(i, "b")
                cash += self._close_position(
                    pos, i, date, px_a, px_b, pending_exit["reason"],
                    float(z_values[i]), float(lambda_values[i]), str(regime_values[i]),
                )
                pos = None 
                pending_exit = None 

            # 4. Register an entry 
            if pos is None and pending_entry is None and signal_values[i] in (1, -1):
                pending_entry = {
                    "hedge_ratio": float(hedge_values[i]),
                    "stop_sd": (
                        float(stop_sd_values[i]) if stop_sd_values is not None else None
                    ),
                    "direction": int(signal_values[i]),
                    "size": float(size_values[i]),
                    "lambda": float(lambda_values[i]),
                    "z": float(z_values[i]),
                    "regime": str(regime_values[i]),
                    "execute_at": i + self.execution_delay,
                }

            # 5. Execute the entry 
            if pos is None and pending_entry is not None and i >= pending_entry["execute_at"]:
                px_a, px_b = fill(i, "a"), fill(i, "b")
                sign = pending_entry["direction"]
                size = min(pending_entry["size"], self.max_position_pct)
                entry_hedge = pending_entry["hedge_ratio"]

                # Stop levels for THIS position: from its own frozen reference
                # when one is supplied per bar, else the run-level defaults
                levels = None
                if self.stop_mode == "volatility" and pending_entry["stop_sd"] is not None:
                    levels = self._stop_levels(pending_entry["stop_sd"], entry_hedge)
                if levels is None:
                    levels = {
                        "stop_loss_pct": self.stop_loss_pct,
                        "trailing_stop_pct": self.trailing_stop_pct,
                        "trailing_activation_pct": self.trailing_activation_pct,
                        "profit_target_pct": self.profit_target_pct,
                    }

                #Leg B carries h times leg A's dollars, so the book matches 
                # log(A) - h * log(B) -- the spread that is actually modelled
                capital_per_leg = size * cash / (1.0 + abs(entry_hedge))
                shares_a = sign * capital_per_leg / px_a
                shares_b = -sign * entry_hedge * capital_per_leg / px_b
                gross_notional = abs(shares_a * px_a) + abs(shares_b * px_b)
                capital_at_risk = size * cash 

                entry_costs = gross_notional * (self.commission_rate + self.slippage)
                cash -= entry_costs 

                pos = _OpenPosition(
                    direction = sign, shares_a = shares_a, shares_b = shares_b,
                    entry_price_a = px_a, entry_price_b = px_b,
                    entry_date = date, entry_idx = i,
                    entry_lambda = pending_entry["lambda"], entry_z = pending_entry["z"],
                    entry_regime = pending_entry["regime"],
                    gross_notional = gross_notional, capital_at_risk = capital_at_risk,
                    position_size = size, hedge_ratio = entry_hedge,
                    max_adverse = 0.0, max_favorable = 0.0,
                    trailing_stop_level = -levels["stop_loss_pct"], trailing_activated = False, 
                    financing = 0.0, borrow = 0.0, costs = entry_costs,
                    stop_loss_pct = levels["stop_loss_pct"],
                    trailing_stop_pct = levels["trailing_stop_pct"],
                    trailing_activation_pct = levels["trailing_activation_pct"],
                    profit_target_pct = levels["profit_target_pct"],
                )
                pending_entry = None 

            # 6. Equity
            if pos is not None:
                unrealized = self._pnl(pos, price_a, price_b)
                gross_equity = cash + unrealized - pos.financing - pos.borrow 
                exposure = pos.gross_notional / equity[i - 1] if equity[i - 1] > 0 else 0.0
            else: 
                gross_equity = cash 
                exposure = 0.0

            if self.credit_idle_cash:
                credit = equity[i - 1] * max(0.0, 1.0 - exposure) * daily_rf 
                cash += credit
                gross_equity += credit 
                cash_credit_track[i] = credit

            equity[i] = gross_equity 
            exposure_track[i] = exposure 

        # force close at the final bar
        forced_close = False 
        if pos is not None:
            forced_close = True 
            i = n - 1
            cash += self._close_position(
                pos, i, pd.Timestamp(common[i]),
                float(close_a.iloc[i]), float(close_b.iloc[i]),
                "forced_close_period_end",
                float(z_values[i]), float(lambda_values[i]), str(regime_values[i]),
            )
            equity[i] = cash 
            pos = None 

        # No $1 floor: a non-positive equity curve is a bug, not a number to clip 
        if np.any(equity <= 0):
            raise ValueError(
                "Equity curve hit zero or negative. The audited engine floored the "
                "denominator at $1 (`np.maximum(equity[:-1], 1) `), hiding exactly this."
            )
        returns = np.concatenate(([0.0], np.diff(equity) / equity[:-1]))

        self.equity_curve = pd.DataFrame(
            {
                "equity": equity,
                "returns": returns,
                "exposure": exposure_track,
                "cash_credit": cash_credit_track,
                "financing": financing_track,
                "spread": spread_df["spread"].to_numpy(),
            },
            index = common,
        )

        self.run_report = {
            "n_trades": len(self.trades),
            "final_equity": float(equity[-1]),
            "total_return_pct": float((equity[-1] / self.initial_capital - 1) * 100),
            "avg_gross_exposure_pct": float(np.mean(exposure_track) * 100),
            "max_gross_exposure_pct": float(np.max(exposure_track) * 100),
            "pct_days_flat": float(np.mean(exposure_track == 0) * 100),
            "total_cash_credit": float(np.sum(cash_credit_track)),
            "total_financing_and_borrow": float(np.sum(financing_track)),
            "forced_close_at_end": forced_close,
            "hedge_ratio_used": "per-bar" if per_bar_hedge else hedge_ratio,
            **{f"stop_{k}": v for k, v in self.resolved_stops.items()},
        }

        self._log(
            f" {len(self.trades)} trades, final equity ${equity[-1]:,.0f} "
            f"({self.run_report['total_return_pct']:+.2f}%)"
        )
        self._log(
            f" avg gross exposure {self.run_report['avg_gross_exposure_pct']:.1f}%, "
            f"flat {self.run_report['pct_days_flat']:.0f} % of days, "
            f"cash credit ${self.run_report['total_cash_credit']:,.0f}"
        )
        if forced_close:
            self._log(" NOTE: open position force-closed at the final bar and RECORDED")
        if self.trades:
            reasons: Dict[str, int] = {}
            for t in self.trades:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            self._log(f" exit reasons: {reasons}")

        return self.equity_curve 

    ###

    def calculate_performance_metrics(self, risk_free_rate: Optional[float] = None) -> Dict:
        """ Performance metrics on both notional and capital-at-risk bases"""
        if self.equity_curve is None:
            raise ValueError("Run the backtest before calculating metrics")

        rf = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        equity = self.equity_curve["equity"].to_numpy(dtype = float)
        returns = self.equity_curve["returns"].to_numpy(dtype = float)[1:]
        exposure = self.equity_curve["exposure"].to_numpy(dtype = float)[1:]

        n_days = len(equity)
        annual_factor = TRADING_DAYS / n_days if n_days > 0 else 1.0
        total_return = (equity[-1] / equity[0] - 1) * 100
        annualized_return = ((equity[-1] / equity[0]) ** annual_factor - 1) * 100

        returns_std = float(np.std(returns, ddof = 1)) if len(returns) > 1 else 0.0
        annualized_vol = returns_std * np.sqrt(TRADING_DAYS) * 100

        daily_rf = rf / TRADING_DAYS
        excess = returns - daily_rf
        sharpe = (
            float(np.sqrt(TRADING_DAYS) * np.mean(excess) / returns_std)
            if returns_std > VOL_EPS
            else 0.0
        )

        # Sortino: downside deviation about the MAR, over ALL n observations
        shortfall = np.minimum(returns - daily_rf, 0.0)
        downside_std = (
            float(np.sqrt(np.sum(shortfall**2) / len(returns))) if len(returns) else 0.0
        )
        sortino = (
            float(np.sqrt(TRADING_DAYS) * np.mean(excess) / downside_std)
            if downside_std > VOL_EPS
            else 0.0
        )

        cumulative = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max 
        max_drawdown = float(np.min(drawdown) * 100) if len(drawdown) else 0.0
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0 

        metrics: Dict = {
            "total_return_pct": total_return,
            "annualized_return_pct": annualized_return,
            "annualized_volatility_pct": annualized_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_drawdown,
            "calmar_ratio": calmar, 
            "avg_gross_exposure_pct": float(np.mean(exposure) * 100),
            "pct_days_flat": float(np.mean(exposure == 0) * 100),
        }

        # Capital at risk: scale only the EXCESS return, then add rf back. The
        # risk-free credit accrues on the whole book and must not be levered -- 
        # scaling the total return would report the cash yield as if it were 
        # earned on the deployed sliver.
        mean_exposure = float(np.mean(exposure))
        if mean_exposure > 1e-9:
            scale = 1.0 / mean_exposure 
            excess_annualized = annualized_return - rf * 100 
            metrics.update(
                {
                    "car_annualized_return_pct": rf * 100 + excess_annualized * scale, 
                    "car_annualized_excess_pct": excess_annualized * scale,
                    "car_annualized_volatility_pct": annualized_vol * scale, 
                    "car_max_drawdown_pct": max_drawdown * scale,
                    "car_scale_factor": scale, 
                    "car_note": (
                        "Excess return scaled by 1/mean gross exposure; the "
                        "risk-free credit is not levered."
                    ),
                }
            )
        if len(returns) >= 30:
            nw = newey_west_mean_test(excess)
            metrics.update(
                {
                    "nw_mean_excess_annualized_pct": nw.get("mean_annualized_excess_pct", 0.0),
                    "nw_tstat": nw.get("t_statistic", 0.0),
                    "nw_pvalue": nw.get("p_value", 1.0),
                    "nw_ci95_low_pct": nw.get("ci95_annualized_pct", (0.0, 0.0))[0],
                    "nw_ci95_high_pct": nw.get("ci95_annualized_pct", (0.0, 0.0))[1],
                }
            )
            se = sharpe_standard_error(returns, rf)
            metrics.update(
                {
                    "sharpe_se_iid": se.get("se_annualized_iid", float("nan")),
                    "sharpe_se_hac": se.get("se_annualized_hac", float("nan")),
                    "sharpe_tstat_hac": se.get("t_statistic_hac", float("nan")),
                }
            )
        metrics.update(self._trade_metrics())
        self.performance_metrics = metrics 
        return metrics 

    def _trade_metrics(self) -> Dict:
        """ Trade statistics on consistent dollar AND percentage bases"""
        if not self.trades:
            return {
                "total_trades": 0, "win_rate_pct": 0.0,
                "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
                "avg_win_usd": 0.0, "avg_loss_usd": 0.0,
                "gross_profit_usd": 0.0, "gross_loss_usd": 0.0,
                "profit_factor": float("nan"),
                "expected_value_per_trade_pct": 0.0,
                "expected_value_per_trade_usd": 0.0,
                "avg_return_on_capital_pct": 0.0,
                "avg_trade_duration_days": 0.0,
                "avg_max_adverse_excursion_pct": 0.0,
                "avg_max_favorable_excursion_pct": 0.0,
                "total_financing_cost": 0.0, "total_borrow_cost": 0.0,
                "total_transaction_cost": 0.0,
            }

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        gross_profit = float(sum(t.pnl for t in wins))
        gross_loss = float(abs(sum(t.pnl for t in losses)))

        # inf when there are not losses -- NOT the raw dollar profit 
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = float("nan")

        return {
            "total_trades": len(self.trades),
            "win_rate_pct": len(wins) / len(self.trades) * 100,
            "avg_win_pct": float(np.mean([t.return_on_notional_pct for t in wins])) if wins else 0.0,
            "avg_loss_pct": float(np.mean([t.return_on_notional_pct for t in losses])) if losses else 0.0,
            "avg_win_usd": float(np.mean([t.pnl for t in wins])) if wins else 0.0,
            "avg_loss_usd": float(np.mean([t.pnl for t in losses])) if losses else 0.0,
            "gross_profit_usd": gross_profit,
            "gross_loss_usd": gross_loss,
            "profit_factor": profit_factor, 
            "expected_value_per_trade_usd": float(np.mean([t.pnl for t in self.trades])),
            "expected_value_per_trade_pct": float (
                np.mean([t.return_on_notional_pct for t in self.trades])
            ),
            "avg_return_on_capital_pct": float(
                np.mean([t.return_on_capital_pct for t in self.trades])
            ),
            "avg_trade_duration_days": float(
                np.mean([t.duration_trading_days for t in self.trades])
            ),
            "avg_max_adverse_excursion_pct": float(
                np.mean([abs(t.max_adverse_excursion) for t in self.trades])
            ),
            "avg_max_favorable_excursion_pct": float(
                np.mean([t.max_favorable_excursion for t in self.trades])
            ),
            "total_financing_cost": float(sum(t.financing_cost for t in self.trades)),
            "total_borrow_cost": float(sum(t.borrow_cost for t in self.trades)),
            "total_transaction_cost": float(sum(t.transaction_cost for t in self.trades)),
        }

    ###

    def bootstrap_metrics(
        self, n_boot: int = 1000, mean_block: float = 20.0, seed: int = 42
    ) -> Dict:
        """ Stationary-bootstrap confidence intervals for the headline metrics"""
        if self.equity_curve is None:
            raise ValueError("Run the backtest first")
        returns = self.equity_curve["returns"].to_numpy(dtype = float)[1:]

        out: Dict = {}
        for metric in ("sharpe", "mean", "max_drawdown"):
            res = bootstrap_metric_ci(
                returns, metric = metric, n_boot = n_boot, mean_block = mean_block,
                risk_free_rate = self.risk_free_rate, seed = seed,
            )
            if "error" in res:
                out[f"{metric}_error"] = res["error"]
            else:
                out[f"{metric}_point"] = res["point_estimate"]
                out[f"{metric}_ci95_low"] = res["ci95_low"]
                out[f"{metric}_ci95_high"] = res["ci95_high"]
        return out 

    def analyze_regime_performance(self, regime_threshold: float = 0.3) -> Dict:
        """ Trade performance split by entry-lambda. Threshold comes from config"""
        if not self.trades:
            return {"calm_regime": {"n_trades":0}, "volatile_regime": {"n_trades": 0}}

        calm = [t for t in self.trades if t.entry_lambda < regime_threshold]
        volatile = [t for t in self.trades if t.entry_lambda >= regime_threshold]

        def summarize(trades: List[Trade]) -> Dict:
            if not trades:
                return {"n_trades": 0, "win_rate": 0.0, "avg_return": 0.0, "total_pnl": 0.0}
            wins = [t for t in trades if t.pnl > 0]
            return {
                "n_trades": len(trades),
                "win_rate": len(wins) / len(trades) * 100,
                "avg_return": float(np.mean([t.return_on_notional_pct for t in trades])),
                "total_pnl": float(sum(t.pnl for t in trades)),
            }

        return {
            "calm_regime": summarize(calm),
            "volatile_regime": summarize(volatile),
            "regime_threshold_used": regime_threshold,
        }

    def analyze_by_exit_reason(self) -> Dict:
        """ Performance by exit reason -- shows which exits can actually fire"""
        if not self.trades:
            return {}
        grouped: Dict[str, List[Trade]] = {}
        for t in self.trades:
            grouped.setdefault(t.exit_reason, []).append(t)

        return {
            reason: {
                "count": len(trades),
                "win_rate": len([t for t in trades if t.pnl > 0]) / len(trades) * 100,
                "avg_return": float(np.mean([t.return_on_notional_pct for t in trades])),
                "total_pnl": float(sum(t.pnl for t in trades)),
            }
            for reason, trades in grouped.items()
        }

    def get_trade_summary(self) -> pd.DataFrame:
        if not self.trades:
            # Keep the columns so a zero-trade run writes a header-only file
            return pd.DataFrame(columns=[f.name for f in fields(Trade)])
        return pd.DataFrame([asdict(t) for t in self.trades])
