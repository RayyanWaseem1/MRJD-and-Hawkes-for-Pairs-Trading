"""
Model fitting and evaluation pipeline.

Split out of `main.py`, which was 1,325 lines carrying the pipeline,
validation, scoring, plotting, and entry point together. `main.py` is now the 
CLI; this module holds the research logic.

Discipline enforced here:
* The hedge ratio, all model parameters, the lambda baseline and every 
    threshold come from a `ModelBundle` fitted ONLY on the training window.
* Evaluation windows never re-fit anything.
* Nothing is silently skipped: a stage that cannot run records WHY in the 
    bundle, and the reason is written into the artifacts
"""

from __future__ import annotations 

from dataclasses import dataclass, field 
from typing import Dict, List, Optional, Tuple 

import numpy as np 
import pandas as pd 

from config import Config
from corporate_actions import DIVIDEND_TREATMENT, estimate_dividend_drag
from equity_pairs_loader import EquityPairsDataPipeline, SpreadStatistics
from hawkes_calibration import HawkesFitError, HawkesProcess
from jump_detector import JumpDetector
from mrjd_estimation import MRJDFitError, MRJDModel
from signal_generation import TradingSignals
from backtest_engine import BacktestEngine, compute_alpha_tstat
from statistics_tools import deflated_sharpe_ratio, power_statement
from time_units import observation_span 

__all__ = ["ModelBundle", "PairPipeline", "validate_pair"]

@dataclass
class ModelBundle:
    """ Everything fitted on the training window, frozen for evaluation"""

    hedge_ratio: float 
    hedge_diagnostics: Dict
    spread_stats: SpreadStatistics
    half_life: float
    pair_validation: Dict 

    hawkes_params: Dict = field(default_factory = dict)
    hawkes_inference: Dict = field(default_factory=dict)
    hawkes_active: bool = False 
    hawkes_inactive_reason: str = ""
    use_hawkes_regimes: bool = False 

    mrjd_params: Dict = field(default_factory = dict)
    mrjd_diagnostics: Dict = field(default_factory = dict)

    detection_basis: str = "fdr"
    n_jumps_fdr: int = 0
    n_jumps_nominal: int = 0

    z_entry_threshold: float = 2.0
    z_exit_threshold: float = 0.5

    train_start: Optional[str] = None 
    train_end: Optional[str] = None 

def validate_pair(
    spread: pd.Series,
    log_a: Optional[pd.Series] = None,
    log_b: Optional[pd.Series] = None,
    min_half_life: int = 5,
    max_half_life: int = 120,
    verbose: bool = True,
) -> Dict:
    """
    Five checks, and ALL FIVE now gate `is_tradeable`.

    The audited version computed `no_regime_change`, printed it, returned it --
    and then excluded it from `is_tradeable`, so the README advertised five 
    checks while four actually gated.

    Stationarity is judged on the Engle-Granger p-value when the log price
    legs are supplied, because the spread is a residual from an estimated
    cointegrating vector; plain ADF over-rejects
    """
    from statsmodels.tsa.stattools import adfuller, coint

    clean = spread.dropna()
    adf_stat, adf_p = adfuller(clean)[:2]

    eg_p = None
    if log_a is not None and log_b is not None:
        idx = log_a.index.intersection(log_b.index)
        _, eg_p, _ = coint(
            log_a.loc[idx].to_numpy(dtype = float),
            log_b.loc[idx].to_numpy(dtype = float),
            trend = "c",
        )
        eg_p = float(eg_p)

    decisive_p = eg_p if eg_p is not None else float(adf_p)
    is_stationary = decisive_p < 0.05

    lag = clean.shift(1).dropna()
    diff = clean.diff().dropna()
    idx = lag.index.intersection(diff.index)
    beta = float(np.polyfit(lag.loc[idx], diff.loc[idx], 1)[0])
    half_life = float(-np.log(2) / beta) if beta < 0 else float("inf")
    half_life_ok = bool(min_half_life <= half_life <= max_half_life)

    rolling_mean = clean.rolling(252).mean()
    mean_drift = float(rolling_mean.std() / clean.std()) if clean.std() > 0 else float("inf")
    stable_mean = mean_drift < 0.5 

    range_in_std = float((clean.max() - clean.min()) / clean.std()) if clean.std() > 0 else float("inf")
    reasonable_range = range_in_std < 10 

    recent = clean.iloc[-252:] if len(clean) > 252 else clean 
    mean_shift = float(abs(recent.mean() - clean.mean()) / clean.std()) if clean.std() > 0 else float("inf")
    no_regime_change = mean_shift < 1.0

    # All five gate. This is the fix 
    is_tradeable = bool(
        is_stationary and half_life_ok and stable_mean and reasonable_range and no_regime_change
    )

    result = {
        "is_tradeable": is_tradeable,
        "is_stationary": bool(is_stationary),
        "adf_pvalue": float(adf_p),
        "eg_pvalue": eg_p,
        "decisive_pvalue": decisive_p,
        "stationarity_test": "engle_granger" if eg_p is not None else "adf",
        "half_life": half_life,
        "half_life_ok": half_life_ok,
        "stable_mean": bool(stable_mean),
        "mean_drift": mean_drift,
        "range_in_std": range_in_std,
        "reasonable_range": bool(reasonable_range),
        "no_regime_change": bool(no_regime_change),
        "mean_shift": mean_shift,
        "n_checks_gating": 5,
    }

    if verbose:
        test_name = result["stationarity_test"]
        print(f" Pair validation ({test_name}) p = {decisive_p:.4f}:")
        print(f" stationary {is_stationary}")
        print(f" half-life {half_life:.1f}d ok={half_life_ok}")
        print(f" stable mean {stable_mean} (drift {mean_drift:.3f})")
        print(f" reasonable range {reasonable_range} ({range_in_std} sd)")
        print(f" no regime change {no_regime_change} (shift {mean_shift:.2f} sd)")
        print(f" => tradeable: {is_tradeable} (all 5 checks gate)")
        if eg_p is not None:
            print(f" [ADF would say p = {adf_p:.4g}; EG is the correct test here]")

    return result 

class PairPipeline:
    """ Fit on training data, evaluate out of sample, with frozen parameters"""

    def __init__(self, config: Optional[Config] = None, verbose: bool = True):
        self.config = config or Config()
        self.verbose = verbose 

        self.loader: Optional[EquityPairsDataPipeline] = None 
        self.spread_df: Optional[pd.DataFrame] = None 
        self.cleaned_data: Optional[Dict[str, pd.DataFrame]] = None 
        self.results: Dict = {}

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    ### data ###

    def acquire_data(self, train_end: Optional[str] = None) -> pd.DataFrame:
        """ Load, adjust, clean, and build the spread with a frozen hedge ratio"""
        cfg = self.config.data 

        self.loader = EquityPairsDataPipeline(
            asset_a_path = cfg.asset_a_csv,
            asset_b_path = cfg.asset_b_csv,
            asset_a_symbol = cfg.asset_a_symbol,
            asset_b_symbol = cfg.asset_b_symbol,
            adjust_splits = cfg.adjust_splits,
            verbose = self.verbose,
        )

        self.loader.load_from_csv(date_columns = cfg.date_columns)
        self.cleaned_data = self.loader.clean_data(
            flag_threshold = cfg.large_move_flag_threshold
        )

        self.spread_df = self.loader.construct_spread(
            method = cfg.hedge_ratio_method,
            hedge_mode = cfg.hedge_mode,
            lookback = cfg.lookback_period,
            train_end = train_end,
        )
        return self.spread_df

    ### fitting ###

    def fit_models(
        self,
        train_spread: pd.DataFrame,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
    ) -> ModelBundle:
        """ Fit everything on the training slice only """
        cfg = self.config 
        self._log(f"\n Fitting on {len(train_spread)} training observations")

        # Pair validation 
        pair_val = validate_pair(
            train_spread["spread"],
            train_spread.get("log_a"),
            train_spread.get("log_b"),
            verbose = self.verbose,
        )
        half_life = pair_val["half_life"]
        if not np.isfinite(half_life) or half_life <= 0:
            half_life = 30.0

        loader = self.loader
        if loader is None:
            raise RuntimeError("Data loader is unavailable; call acquire_data() before fit_models()")
        spread_stats = loader.calculate_spread_statistics(
            train_spread["spread"], train_spread.get("log_a"), train_spread.get("log_b")
        )

        # jump detection
        spread_diff = train_spread["spread"].diff().dropna()

        det_fdr = JumpDetector(
            cfg.jump_detection.significance_level, apply_fdr = True, verbose = self.verbose
        )
        jump_fdr = det_fdr.detect(
            spread_diff, method = cfg.jump_detection.method, window = cfg.jump_detection.window_size
        )
        n_fdr = int(jump_fdr["jump_indicator"].sum())

        det_nom = JumpDetector(
            cfg.jump_detection.significance_level, apply_fdr = False, verbose = False
        )
        jump_nom = det_nom.detect(
            spread_diff, method = cfg.jump_detection.method, window = cfg.jump_detection.window_size
        )
        n_nom = int(jump_nom["jump_indicator"].sum())

        self._log(f" Jumps on training: {n_fdr} (FDR) / {n_nom} (nominal)")

        # Which basis can support a Hawkes fit? Recorded, never silent
        basis, jump_df, n_used = "fdr", jump_fdr, n_fdr
        if n_fdr < cfg.jump_detection.min_jumps_for_hawkes:
            if cfg.jump_detection.fallback_to_nominal_for_hawkes and n_nom >= 5:
                basis, jump_df, n_used = "nominal", jump_nom, n_nom 
                self._log(
                    f" FDR basis has {n_fdr} jumps (, {cfg.jump_detection.min_jumps_for_hawkes}); "
                    f" fitting Hawkes on the NOMINAL basis ({n_nom}). Disclosed in artifacts."
                )

        # Hawkes 
        detector = JumpDetector(
            cfg.jump_detection.significance_level, apply_fdr = (basis == "fdr"), verbose = False 
        )
        jump_times = detector.extract_jump_times(jump_df)
        T = observation_span(jump_df.index)

        hawkes_params: Dict = {}
        hawkes_inference: Dict = {}
        hawkes_active = False 
        inactive_reason = ""

        if n_used >= 5:
            try:
                model = HawkesProcess(kernel = cfg.hawkes.kernel, verbose = self.verbose)
                hawkes_params = model.fit(
                    jump_times, T,
                    method = cfg.hawkes.estimation_method,
                    max_iterations = cfg.hawkes.max_iterations,
                    tolerance = cfg.hawkes.tolerance,
                    baseline_bounds = cfg.hawkes.baseline_bounds,
                    excitation_bounds=cfg.hawkes.excitation_bounds,
                    decay_bounds = cfg.hawkes.decay_bounds,
                    n_restarts = cfg.hawkes.n_restarts,
                )
                hawkes_active = True 

                # the inference that never existed 
                lr = model.likelihood_ratio_test(
                    n_bootstrap = cfg.hawkes.lr_bootstrap_reps, seed = cfg.statistics.seed
                )
                se = model.standard_errors()
                gof = model.goodness_of_fit()

                intensity = model.compute_intensity_at_dates(
                    jump_times, pd.DatetimeIndex(train_spread.index)
                )
                calib = model.intensity_calibration(
                    jump_df["jump_indicator"].reindex(train_spread.index).fillna(0), intensity
                )

                hawkes_inference = {
                    "lr_statistic": lr.get("lr_statistic"),
                    "lr_p_chi2_df2": lr.get("p_chi2_df2"),
                    "lr_p_bootstrap": lr.get("p_bootstrap"),
                    "lr_n_bootstrap": lr.get("n_bootstrap_effective"),
                    "ll_hawkes": lr.get("ll_hawkes"),
                    "ll_poisson": lr.get("ll_poisson"),
                    "branching_ratio": se.get("branching_ratio"),
                    "se_branching_ratio": se.get("se_branching_ratio"),
                    "branching_ci95_low": (se.get("branching_ratio_ci95") or (None, None))[0],
                    "branching_ci95_high": (se.get("branching_ratio_ci95") or (None, None))[1],
                    "se_lambda_bar": se.get("se_lambda_bar"),
                    "se_alpha": se.get("se_alpha"),
                    "se_beta": se.get("se_beta"),
                    "ks_statistic": gof.get("ks_statistic"),
                    "ks_pvalue": gof.get("ks_pvalue"),
                    "gof_is_good": gof.get("is_good_fit"),
                    "intensity_poisson_slope": calib.get("poisson_slope"),
                    "intensity_poisson_pvalue": calib.get("poisson_slope_pvalue"),
                    "n_jumps_fitted": n_used,
                    "detection_basis": basis, 
                    "_gof_full": gof,
                    "_calibration_full": calib,
                }

                p_head = lr.get("p_bootstrap", lr.get("p_chi2_df2"))
                self._log(
                    f" LR vs Poisson = {lr.get('lr_statistic', float('nan')):.2f}, "
                    f"p = {p_head:.4g} | KS p = {gof.get('ks_pvalue', float('nan')):.4g}"
                )
                ci = se.get("branching_ratio_ci95")
                if ci: 
                    self._log(
                        f" branching ratio {se.get('branching_ratio', float('nan')):.4f} "
                        f"CI95 [{ci[0]:.4f}, {ci[1]:.4f}]"
                    )

            except (HawkesFitError, Exception) as exc: #noqa: BLE001
                inactive_reason = f"{type(exc).__name__}: {exc}"
                self._log(f" Hawkes fit unavailable -- {inactive_reason}")
        else:
            inactive_reason = (
                f"only {n_used} jumps detected on the {basis} basis "
                f"(need >= 5). Under correct dating and multiplicity control "
                f"this pair does not produce enough events to establish a "
                f"self-excitation process."
            )
            self._log(f" Hawkes inactive -- {inactive_reason}")

        if not hawkes_active:
            hawkes_params = {
                "lambda_bar": max(n_used / T, 1e-6) if T > 0 else 1e-6,
                "alpha": 0.0,
                "beta": 1.0,
                "beta_H": 1.0,
            }

        # MRJD 
        mrjd_params: Dict = {}
        mrjd_diag: Dict = {}
        try:
            mrjd = MRJDModel(verbose = self.verbose)
            indicator = (
                jump_df["jump_indicator"].reindex(train_spread.index).fillna(0).astype(int)
            )
            mrjd_params = mrjd.fit(
                train_spread["spread"],
                indicator,
                dt = cfg.mrjd.dt,
                joint_refinement=cfg.mrjd.joint_refinement,
                empirical_half_life=half_life,
                raise_on_half_life_mismatch=cfg.mrjd.raise_on_half_life_mismatch,
            )
            mrjd_diag = dict(mrjd.diagnostics)
        except MRJDFitError as exc:
            mrjd_diag = {"error": f"{type(exc).__name__}: {exc}"}
            self._log(f" MRJD unavailable -- {exc}")

        use_regimes = bool(cfg.trading.use_hawkes_regimes and hawkes_active)

        return ModelBundle(
            hedge_ratio = float(train_spread["hedge_ratio"].iloc[-1]),
            hedge_diagnostics = dict(self.loader.hedge_report) if self.loader else {},
            spread_stats = spread_stats,
            half_life = half_life, 
            pair_validation = pair_val,
            hawkes_params = hawkes_params,
            hawkes_inference = hawkes_inference,
            hawkes_active = hawkes_active,
            hawkes_inactive_reason= inactive_reason,
            use_hawkes_regimes=use_regimes,
            mrjd_params = mrjd_params,
            mrjd_diagnostics=mrjd_diag,
            detection_basis = basis,
            n_jumps_fdr=n_fdr,
            n_jumps_nominal=n_nom,
            z_entry_threshold=cfg.trading.z_entry_threshold,
            z_exit_threshold=cfg.trading.z_exit_threshold,
            train_start = train_start, 
            train_end=train_end,
        )

    # Causal artifacts 

    def compute_artifacts(self, full_spread: pd.DataFrame, bundle: ModelBundle) -> Dict:
        """
        Detector output, Hawkes intensity, and z-score on the FULL sample using
        the bundle's FROZEN parameters. Causal: nothing here uses information
        from beyond each observation
        """
        cfg = self.config 

        spread_diff = full_spread["spread"].diff().dropna()
        detector = JumpDetector(
            cfg.jump_detection.significance_level,
            apply_fdr = (bundle.detection_basis == "fdr"),
            verbose = False,
        )
        jump_df = detector.detect(
            spread_diff,
            method = cfg.jump_detection.method,
            window = cfg.jump_detection.window_size,
        )
        jump_times = detector.extract_jump_times(jump_df)

        model = HawkesProcess(verbose = False)
        model.params = dict(bundle.hawkes_params)

        if len(jump_times) >= 1 and bundle.hawkes_active:
            intensity = model.compute_intensity_at_dates(
                jump_times, pd.DatetimeIndex(full_spread.index)
            )
        else:
            intensity = pd.Series(
                bundle.hawkes_params.get("lambda_bar", 0.01), index = full_spread.index
            )

        lookback = cfg.trading.z_score_lookback
        if cfg.trading.z_score_basis == "mrjd" and bundle.mrjd_params:
            theta = bundle.mrjd_params["theta"]
            sd = bundle.mrjd_params["sigma"] / np.sqrt(2 * bundle.mrjd_params["kappa"])
            z_score = (full_spread["spread"] - theta) / sd 
        else:
            prior = full_spread["spread"].shift(1)
            mean = prior.rolling(window = lookback, min_periods = 20).mean()
            sd = prior.rolling(window = lookback, min_periods = 20).std()
            z_score = ((full_spread["spread"] - mean) / sd).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)

        return {"jump_df": jump_df, "hawkes_intensity": intensity, "z_score": z_score}

    # Evaluation

    def evaluate_period(
        self,
        full_spread: pd.DataFrame,
        full_cleaned: Dict[str, pd.DataFrame],
        artifacts: Dict,
        bundle: ModelBundle,
        start: str, 
        end: str, 
        label: str, 
        z_entry: Optional[float] = None,
        z_exit: Optional[float] = None,
        light: bool = False,
    ) -> Dict:
        """
        Generate signals and backtest one window with frozen parameters.

        `light = True` skips the bootstrap and power calculations. Used by the 
        threshold grid search, where only the ranking metric is needed and the 
        full inference on every trial config would dominate runtime
        """
        cfg = self.config 
        self._log(f"\n Evaluating: {label}")

        period_spread = full_spread.loc[start:end]
        if len(period_spread) < 30:
            self._log(f" only {len(period_spread)} observations; skipping")
            return {}

        period_cleaned = {
            "asset_a": full_cleaned["asset_a"].loc[start:end],
            "asset_b": full_cleaned["asset_b"].loc[start:end],
        }

        jump_df = artifacts["jump_df"]
        indicator = (
            jump_df["jump_indicator"].reindex(period_spread.index).fillna(0).astype(int)
        )
        intensity = artifacts["hawkes_intensity"].reindex(period_spread.index).ffill()
        z_score = artifacts["z_score"].reindex(period_spread.index).fillna(0.0)

        generator = TradingSignals(
            z_entry_threshold=z_entry if z_entry is not None else bundle.z_entry_threshold,
            z_exit_threshold= z_exit if z_exit is not None else bundle.z_exit_threshold,
            lambda_decay_lookback= cfg.trading.lambda_decay_lookback,
            min_lambda_decay_pct=cfg.trading.min_lambda_decay_pct,
            max_position_size= cfg.trading.max_position_size,
            min_position_size=cfg.trading.min_position_size,
            min_hold_fraction = cfg.trading.min_hold_fraction,
            target_hold_fraction = cfg.trading.max_hold_fraction,
            max_holding_period_cap = cfg.trading.max_holding_period_cap,
            use_jump_entries = cfg.trading.use_jump_entries,
            use_hawkes_regimes= bundle.use_hawkes_regimes,
            z_lookback = cfg.trading.z_score_lookback,
            emergency_z_move = cfg.trading.emergency_z_move,
            regime_excess_calm = cfg.trading.regime_excess_calm,
            regime_excess_elevated = cfg.trading.regime_excess_elevated,
            regime_excess_crisis = cfg.trading.regime_excess_crisis,
            verbose = self.verbose,
        )
        generator.set_lambda_baseline(bundle.hawkes_params.get("lambda_bar", 0.01))
        generator.set_half_life(bundle.half_life)

        signals_df = generator.generate_signals(
            spread = period_spread["spread"],
            lambda_intensity=intensity,
            jump_indicator=indicator,
            z_score = z_score,
        )

        bt = cfg.backtest
        symbol_a = cfg.data.asset_a_symbol
        symbol_b = cfg.data.asset_b_symbol

        engine = BacktestEngine(
            initial_capital = bt.initial_capital,
            commission_rate = bt.commission_rate,
            slippage_bps=bt.slippage_bps,
            max_position_pct=bt.max_position_pct,
            stop_loss_pct = bt.stop_loss_pct,
            trailing_stop_pct = bt.trailing_stop_pct,
            trailing_activation_pct=bt.trailing_activation_pct,
            profit_target_pct=bt.profit_target_pct,
            execution_delay = bt.execution_delay,
            execution_price=bt.execution_price,
            risk_free_rate = bt.risk_free_rate,
            credit_idle_cash = bt.credit_idle_cash,
            long_financing_rate = bt.long_financing_rate,
            short_rebate_rate = bt.short_rebate_rate,
            borrow_rate_a = bt.borrow_rates.get(symbol_a, bt.default_borrow_rate),
            borrow_rate_b=bt.borrow_rates.get(symbol_b, bt.default_borrow_rate),
            use_intraday_stops = bt.use_intraday_stops,
            stop_mode = bt.stop_mode,
            stop_loss_sigma = bt.stop_loss_sigma,
            trailing_stop_sigma = bt.trailing_stop_sigma,
            trailing_activation_sigma=bt.trailing_activation_sigma,
            profit_target_sigma= bt.profit_target_sigma,
            stop_floor_pct = bt.stop_floor_pct,
            stop_cap_pct = bt.stop_cap_pct,
            verbose = self.verbose,
        )
        engine.set_half_life(bundle.half_life)

        equity_curve = engine.run_backtest(
            signals_df, 
            period_spread,
            period_cleaned["asset_a"]["Close"],
            period_cleaned["asset_b"]["Close"],
            hedge_ratio = bundle.hedge_ratio,
            asset_a_ohlc = period_cleaned["asset_a"],
            asset_b_ohlc = period_cleaned["asset_b"],
        )

        metrics = engine.calculate_performance_metrics(risk_free_rate = bt.risk_free_rate)

        if not light:
            metrics.update(
                compute_alpha_tstat(equity_curve, bt.benchmark_csv, bt.risk_free_rate)
            )
            metrics.update(
                engine.bootstrap_metrics(
                    n_boot = cfg.statistics.bootstrap_reps,
                    mean_block = cfg.statistics.bootstrap_mean_block,
                    seed = cfg.statistics.seed,
                )
            )

            power = power_statement(
                n_obs = len(equity_curve),
                annual_volatility = metrics["annualized_volatility_pct"] / 100.0,
                target_effect_annual = cfg.statistics.power_target_effect,
            )
            if "error" not in power:
                metrics["power_to_detect_target"] = power["achieved_power"]
                metrics["mde_annualized_pct"] = power["mde_at_80pct_power_annual_pct"]
                metrics["power_statement"] = power["statement"]

            metrics["dividend_differential_annual_pct"] = (
                estimate_dividend_drag(symbol_a, symbol_b)["differential_annual"] * 100
            )

        return {
            "label": label,
            "equity_curve": equity_curve,
            "signals_df": signals_df,
            "metrics": metrics, 
            "engine": engine, 
            "regime_performance": engine.analyze_regime_performance(bt.regime_threshold),
            "exit_reasons": engine.analyze_by_exit_reason(),
            "signal_quality": generator.calculate_signal_quality(signals_df),
        }

    # threshold tuning (training window only)

    def tune_thresholds(
        self,
        full_spread: pd.DataFrame,
        full_cleaned: Dict[str, pd.DataFrame],
        artifacts: Dict,
        bundle: ModelBundle,
        train_start: str, 
        train_end: str,
    ) -> Tuple[float, float, int, List[Dict]]:
        """
        Grid-search entry/exit thresholds ON THE TRAINING WINDOW ONLY.

        The audited pipeline hardcoded z_entry = 2.0 / z_exit = 0.5 in `main()`
        (with the comment "tuned on training data; frozen for val/walk-forward")
        and reused the same values verbatim in `walk_forward.py`, so all 23
        "out-of-sample" quarters traded with thresholds chosen knowing the full 
        sample. Returns the winner AND the number of configurations tried, so 
        the deflated Sharpe can account for the search.
        """
        wf = self.config.walk_forward
        trials: List[Dict] = []
        best = (bundle.z_entry_threshold, bundle.z_exit_threshold, -np.inf)

        was_verbose = self.verbose
        self.verbose = False 
        try: 
            for z_entry in wf.z_entry_grid:
                for z_exit in wf.z_exit_grid:
                    if z_exit >= z_entry:
                        continue 
                    try:
                        res = self.evaluate_period(
                            full_spread, full_cleaned, artifacts, bundle,
                            train_start, train_end, f"tune e{z_entry} x{z_exit}",
                            z_entry = z_entry, z_exit = z_exit, light = True,
                        )
                    except Exception as exc: #noqa: BLE001
                        trials.append(
                            {"z_entry": z_entry, "z_exit": z_exit,
                             "error": f"{type(exc).__name__}: {exc}"}
                        )
                        continue 
                    if not res:
                        continue 
                    sharpe = float(res["metrics"].get("sharpe_ratio", -np.inf))
                    trials.append(
                        {
                            "z_entry": z_entry, "z_exit": z_exit,
                            "sharpe": sharpe,
                            "n_trades": res["metrics"].get("total_trades", 0),
                            "ann_return_pct": res["metrics"].get("annualized_return_pct", 0.0),
                        }
                    )
                    if np.isfinite(sharpe) and sharpe > best[2]:
                        best = (z_entry, z_exit, sharpe)
        finally:
            self.verbose = was_verbose 

        n_trials = len([t for t in trials if "error" not in t])
        if was_verbose:
            print(
                f" tuned on training window: z_entry = {best[0]}, z_exit = {best[1]} "
                f"(best of {n_trials} configurations)"
            )
        return best[0], best[1], n_trials, trials 
