"""
Quarterly walk-forward, run as ONE CONTINUOUS BACKTEST.

Changes from the audited version
--------------------------------
1. CONTINUOUS, NOT 23 RESTARTS. The old engine restarted the backtest at every
   quarter with a fresh $1,000,000 and a flat book. Meanwhile the signal
   generator set min_hold = 0.5 x half-life and max_hold = 1.5 x half-life --
   for CVX/XOM that is 25 and 76 trading days against a ~63-day quarter. A
   trade entered mid-quarter often could not reach its minimum hold before the
   quarter ended, at which point it was silently liquidated and never recorded.
   That is not a test of the strategy; it is a test of a strategy that gets
   stopped out on an arbitrary calendar boundary.

   Parameters are now SWAPPED IN at each boundary while the book, the cash
   balance and any open position carry straight through.

2. NO STITCHING. `_stitch_equity_curves` rescaled each quarter so its first
   equity equalled the previous quarter's last. Because each quarter's
   `equity[0]` was the untouched initial capital, the boundary day contributed
   a hardcoded 0% return and the quarter's first genuine return was discarded
   -- 23 days zeroed out of the OOS series that then fed the Sharpe and the
   alpha t-stat. With one continuous curve there is nothing to stitch.

3. METRICS COME FROM REAL TRADES. The old code built a `temp_engine`, assigned
   it the stitched curve, and called `calculate_performance_metrics()` on it.
   `temp_engine.trades` was empty, so profit factor, average win, average loss,
   expected value, duration, MAE and MFE were all written to
   `walk_forward_metrics.csv` as zeros that looked like measurements.

4. FAILURES ARE REPORTED. Three `except Exception: print(...); continue` blocks
   dropped failed quarters and stitched the remainder as if continuous --
   survivorship bias that preferentially drops the volatile quarters a jump
   model most needs to be tested on. Failures are now counted, logged, and
   written to `quarter_failures.csv`.

5. THRESHOLDS ARE TUNED INSIDE THE LOOP, on each quarter's training window
   only, and the number of configurations tried is recorded so the deflated
   Sharpe can account for the search.

6. THE HEDGE RATIO IS RE-ESTIMATED at each boundary and held fixed within the
   quarter.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config
from backtest_engine import BacktestEngine, compute_alpha_tstat
from pipeline import ModelBundle, PairPipeline
from signal_generation import TradingSignals
from statistics_tools import deflated_sharpe_ratio, power_statement

__all__ = ["WalkForwardEngine"]


class WalkForwardEngine:
    """Quarterly refit, one continuous book."""

    def __init__(self, config: Config, verbose: bool = True):
        self.config = config
        self.verbose = verbose

        self.quarterly_results: List[Dict] = []
        self.failures: List[Dict] = []
        self.equity_curve: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        self.metrics: Dict = {}
        self.total_configurations_tried = 0

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    # ------------------------------------------------------------------ #

    def _quarter_ends(self, index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        min_train = self.config.walk_forward.min_train_days
        if len(index) <= min_train:
            return []

        naive = index.tz_convert(None) if index.tz is not None else index
        first, last = naive[min_train], naive[-1]

        try:
            calendar = pd.date_range(start=first, end=last, freq="QE")
        except ValueError:
            calendar = pd.date_range(start=first, end=last, freq="Q")

        ends: List[pd.Timestamp] = []
        for q in calendar:
            mask = naive <= q
            if not mask.any():
                continue
            actual = index[mask][-1]
            if not ends or actual != ends[-1]:
                ends.append(actual)
        return ends

    # ------------------------------------------------------------------ #

    def run(self) -> Dict:
        cfg = self.config
        wf = cfg.walk_forward

        self._log("\n" + "=" * 72)
        self._log("WALK-FORWARD (continuous book, quarterly parameter swap)")
        self._log(f"  min training: {wf.min_train_days} days")
        self._log("=" * 72)

        pipeline = PairPipeline(cfg, verbose=False)
        pipeline.acquire_data()

        full_spread = pipeline.spread_df
        full_cleaned = pipeline.cleaned_data
        loader = pipeline.loader
        if full_spread is None or full_cleaned is None or loader is None:
            raise RuntimeError("Data acquisition did not produce a complete pipeline state")
        index = full_spread.index
        if not isinstance(index, pd.DatetimeIndex):
            raise RuntimeError("Walk-forward evaluation requires a DatetimeIndex")

        quarter_ends = self._quarter_ends(index)
        if not quarter_ends:
            raise RuntimeError("Not enough data for the requested min_train_days")

        self._log(
            f"\nFull sample {index[0].date()} -> {index[-1].date()} "
            f"({len(index)} obs), {len(quarter_ends)} quarters"
        )

        # -------- build one signal series, swapping parameters per quarter --------
        signal_frames: List[pd.DataFrame] = []
        hedge_by_period: List[tuple] = []

        for i, q_end in enumerate(quarter_ends):
            after = index[index > q_end]
            if len(after) == 0:
                continue
            eval_start = after[0]
            eval_end = quarter_ends[i + 1] if i + 1 < len(quarter_ends) else index[-1]
            if eval_start >= eval_end:
                continue

            label = f"Q{i + 1} {eval_start.date()}->{eval_end.date()}"
            train_spread = full_spread.loc[:q_end]

            try:
                # Re-estimate the hedge ratio on this quarter's training window.
                if wf.reestimate_hedge_each_quarter:
                    h, _ = loader.estimate_hedge_ratio_static(
                        full_cleaned["asset_a"]["Close"].loc[:q_end],
                        full_cleaned["asset_b"]["Close"].loc[:q_end],
                        method=cfg.data.hedge_ratio_method,
                    )
                    quarter_spread = full_spread.copy()
                    quarter_spread["spread"] = (
                        quarter_spread["log_a"] - h * quarter_spread["log_b"]
                    )
                    quarter_spread["hedge_ratio"] = h
                    train_spread = quarter_spread.loc[:q_end]
                else:
                    quarter_spread = full_spread
                    h = float(full_spread["hedge_ratio"].iloc[-1])

                bundle = pipeline.fit_models(
                    train_spread,
                    train_start=str(index[0].date()),
                    train_end=str(q_end.date()),
                )
                artifacts = pipeline.compute_artifacts(quarter_spread, bundle)

                if wf.tune_thresholds:
                    z_entry, z_exit, n_trials, _ = pipeline.tune_thresholds(
                        quarter_spread, full_cleaned, artifacts, bundle,
                        str(index[0].date()), str(q_end.date()),
                    )
                    self.total_configurations_tried += n_trials
                else:
                    z_entry, z_exit = bundle.z_entry_threshold, bundle.z_exit_threshold

                # Signals for THIS quarter only, from frozen parameters.
                period = quarter_spread.loc[eval_start:eval_end]
                indicator = (
                    artifacts["jump_df"]["jump_indicator"]
                    .reindex(period.index).fillna(0).astype(int)
                )
                intensity = artifacts["hawkes_intensity"].reindex(period.index).ffill()
                z_score = artifacts["z_score"].reindex(period.index).fillna(0.0)

                generator = TradingSignals(
                    z_entry_threshold=z_entry,
                    z_exit_threshold=z_exit,
                    lambda_decay_lookback=cfg.trading.lambda_decay_lookback,
                    min_lambda_decay_pct=cfg.trading.min_lambda_decay_pct,
                    max_position_size=cfg.trading.max_position_size,
                    min_position_size=cfg.trading.min_position_size,
                    use_jump_entries=cfg.trading.use_jump_entries,
                    use_hawkes_regimes=bundle.use_hawkes_regimes,
                    z_lookback=cfg.trading.z_score_lookback,
                    emergency_z_move=cfg.trading.emergency_z_move,
                    regime_excess_calm=cfg.trading.regime_excess_calm,
                    regime_excess_elevated=cfg.trading.regime_excess_elevated,
                    regime_excess_crisis=cfg.trading.regime_excess_crisis,
                    verbose=False,
                )
                generator.set_lambda_baseline(bundle.hawkes_params.get("lambda_bar", 0.01))
                generator.set_half_life(bundle.half_life)

                quarter_signals = generator.generate_signals(
                    spread=period["spread"],
                    lambda_intensity=intensity,
                    jump_indicator=indicator,
                    z_score=z_score,
                )
                signal_frames.append(quarter_signals)
                hedge_by_period.append((eval_start, eval_end, h))

                self.quarterly_results.append(
                    {
                        "quarter": i + 1,
                        "train_end": q_end,
                        "eval_start": eval_start,
                        "eval_end": eval_end,
                        "hedge_ratio": h,
                        "half_life": bundle.half_life,
                        "z_entry": z_entry,
                        "z_exit": z_exit,
                        "hawkes_active": bundle.hawkes_active,
                        "n_jumps_fdr": bundle.n_jumps_fdr,
                        "n_jumps_nominal": bundle.n_jumps_nominal,
                        "branching_ratio": bundle.hawkes_inference.get("branching_ratio"),
                        "lr_pvalue": bundle.hawkes_inference.get(
                            "lr_p_bootstrap", bundle.hawkes_inference.get("lr_p_chi2_df2")
                        ),
                        "bundle": bundle,
                    }
                )
                self._log(
                    f"  {label}: h={h:.4f}, HL={bundle.half_life:.0f}d, "
                    f"z=({z_entry},{z_exit}), hawkes={'on' if bundle.hawkes_active else 'off'}"
                )

            except Exception as exc:  # noqa: BLE001 -- recorded, never silent
                self.failures.append(
                    {
                        "quarter": i + 1,
                        "train_end": str(q_end.date()),
                        "eval_start": str(eval_start.date()),
                        "eval_end": str(eval_end.date()),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                self._log(f"  {label}: FAILED -- {type(exc).__name__}: {exc}")
                continue

        if not signal_frames:
            raise RuntimeError(
                f"No quarter produced signals. {len(self.failures)} failed: "
                f"{[f['error_type'] for f in self.failures]}"
            )

        # -------- one continuous backtest over the concatenated signals --------
        combined = pd.concat(signal_frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        self.signals = combined

        spread_for_bt = full_spread.loc[combined.index]
        mean_hedge = float(np.mean([h for _, _, h in hedge_by_period])) if hedge_by_period else 1.0

        bt = cfg.backtest
        engine = BacktestEngine(
            initial_capital=bt.initial_capital,
            commission_rate=bt.commission_rate,
            slippage_bps=bt.slippage_bps,
            max_position_pct=bt.max_position_pct,
            stop_loss_pct=bt.stop_loss_pct,
            trailing_stop_pct=bt.trailing_stop_pct,
            trailing_activation_pct=bt.trailing_activation_pct,
            profit_target_pct=bt.profit_target_pct,
            execution_delay=bt.execution_delay,
            execution_price=bt.execution_price,
            risk_free_rate=bt.risk_free_rate,
            credit_idle_cash=bt.credit_idle_cash,
            long_financing_rate=bt.long_financing_rate,
            short_rebate_rate=bt.short_rebate_rate,
            borrow_rate_a=bt.borrow_rates.get(cfg.data.asset_a_symbol, bt.default_borrow_rate),
            borrow_rate_b=bt.borrow_rates.get(cfg.data.asset_b_symbol, bt.default_borrow_rate),
            use_intraday_stops=bt.use_intraday_stops,
            stop_mode=bt.stop_mode,
            stop_loss_sigma=bt.stop_loss_sigma,
            trailing_stop_sigma=bt.trailing_stop_sigma,
            trailing_activation_sigma=bt.trailing_activation_sigma,
            profit_target_sigma=bt.profit_target_sigma,
            stop_floor_pct=bt.stop_floor_pct,
            stop_cap_pct=bt.stop_cap_pct,
            verbose=self.verbose,
        )

        self._log(f"\nRunning ONE continuous backtest over {len(combined)} bars")
        self.equity_curve = engine.run_backtest(
            combined,
            spread_for_bt,
            full_cleaned["asset_a"]["Close"].loc[combined.index],
            full_cleaned["asset_b"]["Close"].loc[combined.index],
            hedge_ratio=mean_hedge,
            asset_a_ohlc=full_cleaned["asset_a"].loc[combined.index],
            asset_b_ohlc=full_cleaned["asset_b"].loc[combined.index],
        )

        # Metrics from the REAL trade list on the REAL curve.
        metrics = engine.calculate_performance_metrics(risk_free_rate=bt.risk_free_rate)
        metrics.update(compute_alpha_tstat(self.equity_curve, bt.benchmark_csv, bt.risk_free_rate))
        metrics.update(
            engine.bootstrap_metrics(
                n_boot=cfg.statistics.bootstrap_reps,
                mean_block=cfg.statistics.bootstrap_mean_block,
                seed=cfg.statistics.seed,
            )
        )

        metrics["quarters_evaluated"] = len(self.quarterly_results)
        metrics["quarters_failed"] = len(self.failures)
        metrics["configurations_tried"] = self.total_configurations_tried

        if self.total_configurations_tried > 0:
            dsr = deflated_sharpe_ratio(
                observed_sharpe=metrics.get("sharpe_ratio", 0.0),
                n_trials=max(self.total_configurations_tried, 1),
                n_obs=len(self.equity_curve),
            )
            if "error" not in dsr:
                metrics["deflated_sharpe_probability"] = dsr["deflated_sharpe_probability"]
                metrics["expected_max_sharpe_from_search"] = dsr[
                    "expected_max_sharpe_annualized"
                ]
                metrics["deflated_sharpe_passes"] = dsr["passes_at_95pct"]

        power = power_statement(
            n_obs=len(self.equity_curve),
            annual_volatility=metrics["annualized_volatility_pct"] / 100.0,
            target_effect_annual=cfg.statistics.power_target_effect,
        )
        if "error" not in power:
            metrics["power_to_detect_target"] = power["achieved_power"]
            metrics["mde_annualized_pct"] = power["mde_at_80pct_power_annual_pct"]
            metrics["power_statement"] = power["statement"]

        self.metrics = metrics
        self.engine = engine
        self._print_summary()

        return {
            "metrics": metrics,
            "equity_curve": self.equity_curve,
            "signals": combined,
            "quarterly_results": self.quarterly_results,
            "failures": self.failures,
            "engine": engine,
        }

    # ------------------------------------------------------------------ #

    def quarterly_frame(self) -> pd.DataFrame:
        rows = []
        for q in self.quarterly_results:
            row = {k: v for k, v in q.items() if k != "bundle"}
            row["eval_start"] = str(q["eval_start"].date())
            row["eval_end"] = str(q["eval_end"].date())
            row["train_end"] = str(q["train_end"].date())
            rows.append(row)
        return pd.DataFrame(rows)

    def failures_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.failures)

    def _print_summary(self) -> None:
        m = self.metrics
        self._log("\n" + "=" * 72)
        self._log("WALK-FORWARD OUT-OF-SAMPLE SUMMARY")
        self._log("=" * 72)
        self._log(f"  Quarters evaluated : {m.get('quarters_evaluated', 0)}")
        self._log(f"  Quarters FAILED    : {m.get('quarters_failed', 0)}")
        if self.failures:
            kinds: Dict[str, int] = {}
            for f in self.failures:
                kinds[f["error_type"]] = kinds.get(f["error_type"], 0) + 1
            self._log(f"    failure types    : {kinds}")

        self._log(f"\n  Annualised return  : {m.get('annualized_return_pct', 0):.3f}%")
        self._log(f"  Annualised vol     : {m.get('annualized_volatility_pct', 0):.3f}%")
        self._log(f"  Sharpe             : {m.get('sharpe_ratio', 0):.3f} "
                  f"(HAC SE {m.get('sharpe_se_hac', float('nan')):.3f})")
        self._log(f"  Max drawdown       : {m.get('max_drawdown_pct', 0):.2f}%")
        if "car_annualized_return_pct" in m:
            self._log(f"  On capital at risk : {m['car_annualized_return_pct']:.2f}% return, "
                      f"{m.get('car_max_drawdown_pct', 0):.2f}% drawdown "
                      f"({m.get('car_scale_factor', 1):.1f}x scale)")

        self._log(f"\n  HEADLINE TEST -- H0: mean excess return = 0 (Newey-West)")
        self._log(f"    mean excess     : {m.get('nw_mean_excess_annualized_pct', 0):+.3f}%/yr")
        self._log(f"    t-statistic     : {m.get('nw_tstat', 0):.3f}")
        self._log(f"    p-value         : {m.get('nw_pvalue', 1):.4f}")
        self._log(f"    95% CI          : [{m.get('nw_ci95_low_pct', 0):+.3f}%, "
                  f"{m.get('nw_ci95_high_pct', 0):+.3f}%]")

        self._log(f"\n  Neutrality check (CAPM vs SPY, NOT the headline)")
        self._log(f"    beta            : {m.get('beta', 0):.4f}   R^2 {m.get('r_squared', 0):.5f}")

        self._log(f"\n  Trades             : {m.get('total_trades', 0)}")
        self._log(f"  Win rate           : {m.get('win_rate_pct', 0):.1f}%")
        self._log(f"  Profit factor      : {m.get('profit_factor', float('nan')):.3f}")
        self._log(f"  Avg duration       : {m.get('avg_trade_duration_days', 0):.1f} trading days")

        if "deflated_sharpe_probability" in m:
            self._log(f"\n  Configurations tried : {m.get('configurations_tried', 0)}")
            self._log(f"  Deflated Sharpe prob : {m['deflated_sharpe_probability']:.4f} "
                      f"({'passes' if m.get('deflated_sharpe_passes') else 'FAILS'} at 95%)")

        if "power_statement" in m:
            self._log(f"\n  POWER: {m['power_statement']}")
        self._log("=" * 72)
