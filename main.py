"""
Command-line entry point 

    python main.py --pair CVX_XOM --mode train_val
    python main.py --pair all --mode all 
    python main.py --pair GS_MS --mode train_val --no-control 

Every run executes a CONTROL ARM: the same pipeline with the Hawkes layer
switched off (plain rolling z-score, no regimes, no jump entries, no 
lambda-decay filter). Without a control, "Hawkes does not add alpha" is not a 
measurable claim, because there is nothing to measure against. With it, the 
project is a controlled experiment rather than a strategy that lost money
"""

from __future__ import annotations 

import argparse
import copy
import sys 
import traceback 
from pathlib import Path 
from typing import Dict, List, Optional 

import numpy as np 
import pandas as pd 

from config import Config, PAIRS, set_seeds
from corporate_actions import DIVIDEND_TREATMENT
from jump_detector import JumpDetector
from pipeline import PairPipeline
from results_io import ResultsWriter 
from walk_forward import WalkForwardEngine

PROJECT_ROOT = Path(__file__).resolve().parent 
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


### train/ validation ###

def run_train_val(
    pair_key: str, config: Config, run_control: bool = True, verbose: bool = True
) -> Dict:
    """ Fit on the training window, evaluate in and out of sample, write artifacts"""
    tv = config.train_val
    out_dir = OUTPUT_ROOT / pair_key / "train_val"
    writer = ResultsWriter(out_dir, verbose = verbose) 

    if verbose:
        print("\n" + "=" * 72)
        print(f"TRAIN / VALIDATION -- {pair_key}")
        print(f" train {tv.train_start} -> {tv.train_end}")
        print(f" val {tv.val_start} -> {tv.val_end}")
        print("=" * 72)

    pipeline = PairPipeline(config, verbose = verbose)
    # the hedge ratio is estimated on the TRAINING window only and frozen 
    pipeline.acquire_data(train_end = tv.train_end)

    full_spread = pipeline.spread_df
    full_cleaned = pipeline.cleaned_data
    if full_spread is None or full_cleaned is None:
        raise RuntimeError("Data acquisition did not produce spread and cleaned data")
    train_spread = full_spread.loc[tv.train_start : tv.train_end]

    bundle = pipeline.fit_models(
        train_spread, train_start = tv.train_start, train_end = tv.train_end
    )
    artifacts = pipeline.compute_artifacts(full_spread, bundle)

    results: Dict[str, Dict] = {}
    for label, start, end in (
        ("train", tv.train_start, tv.train_end),
        ("validation", tv.val_start, tv.val_end),
    ):
        res = pipeline.evaluate_period(
            full_spread, full_cleaned, artifacts, bundle, start, end, label
        )
        if res:
            results[label] = res 

    ### control arm ###
    control_results: Dict[str, Dict] = {}
    if run_control:
        if verbose:
            print("\n -- CONTROL ARM: plain z-score, no Hawkes layer ---")
        # Preserve every user-selected setting (costs, detector, dates and
        # seed), changing only the Hawkes-specific signal layer.
        control_config = copy.deepcopy(config).as_control_arm()
        control_pipeline = PairPipeline(control_config, verbose = False)
        control_pipeline.loader = pipeline.loader 
        control_pipeline.spread_df = full_spread 
        control_pipeline.cleaned_data = full_cleaned

        control_bundle = control_pipeline.fit_models(
            train_spread, train_start = tv.train_start, train_end = tv.train_end
        )
        control_bundle.use_hawkes_regimes = False 
        control_artifacts = control_pipeline.compute_artifacts(full_spread, control_bundle)

        for label, start, end in (
            ("train", tv.train_start, tv.train_end),
            ("validation", tv.val_start, tv.val_end),
        ):
            res = control_pipeline.evaluate_period(
                full_spread, full_cleaned, control_artifacts, control_bundle,
                start, end, f"control_{label}",
            )
            if res:
                control_results[label] = res 
        if verbose:
            for label, res in control_results.items():
                m = res["metrics"]
                print(
                    f" control {label}: {m.get('total_trades', 0)} trades, "
                    f"ann {m.get('annualized_return_pct', 0):+.3f}%, "
                    f"Sharpe {m.get('sharpe_ratio', 0):+.3f}"
                )

    ### artifacts ###
    writer.write_spread(full_spread)
    writer.write_causal_artifacts(
        artifacts["jump_df"], artifacts["hawkes_intensity"], artifacts["z_score"]
    )
    writer.write_model_bundle(bundle)
    writer.write_row(bundle.pair_validation, "pair_validation.csv")
    writer.write_row(bundle.spread_stats, "spread_statistics.csv")
    writer.write_row(bundle.hedge_diagnostics, "hedge_ratio_diagnostics.csv")

    if bundle.hawkes_inference:
        clean_inference = {
            k: v for k, v in bundle.hawkes_inference.items() if not k.startswith("_")
        }
        writer.write_hawkes_inference(clean_inference)
    if bundle.mrjd_diagnostics:
        writer.write_row(bundle.mrjd_diagnostics, "mrjd_diagnostics.csv")

    detector = JumpDetector(
        config.jump_detection.significance_level, apply_fdr = True, verbose = False
    )
    comparison = detector.compare_detectors(
        full_spread["spread"].diff().dropna(), window = config.jump_detection.window_size
    )
    writer.write_jump_comparison(comparison)

    for label, res in results.items():
        writer.write_evaluation(
            label, res["equity_curve"], res["signals_df"],
            res["engine"].get_trade_summary(), res["metrics"],
        )
        writer.write_row(res["exit_reasons"], f"{label}_exit_reasons.csv")
        writer.write_row(res["regime_performance"], f"{label}_regime_performance.csv")
        writer.write_row(res["signal_quality"], f"{label}_signal_quality.csv")

    for label, res in control_results.items():
        writer.write_evaluation(
            f"control_{label}", res["equity_curve"], res["signals_df"],
            res["engine"].get_trade_summary(), res["metrics"]
        )

    # head-to-head comparison table 
    comparison_rows: List[Dict] = []
    for label in ("train", "validation"):
        for arm, source in (("hawkes", results), ("control", control_results)):
            if label in source:
                m = source[label]["metrics"]
                comparison_rows.append(
                    {
                        "arm": arm, "window": label,
                        "total_trades": m.get("total_trades", 0),
                        "annualized_return_pct": m.get("annualized_return_pct", 0.0),
                        "annualized_volatility_pct": m.get("annualized_volatility_pct", 0.0),
                        "sharpe_ratio": m.get("sharpe_ratio", 0.0),
                        "sharpe_se_hac": m.get("sharpe_se_hac", float("nan")),
                        "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
                        "nw_mean_excess_annualized_pct": m.get("nw_mean_excess_annualized_pct", 0.0),
                        "nw_tstat": m.get("nw_tstat", 0.0),
                        "nw_pvalue": m.get("nw_pvalue", 1.0),
                        "beta_vs_spy": m.get("beta", 0.0),
                        "profit_factor": m.get("profit_factor", float("nan")),
                    }
                )
    writer.write_comparison(comparison_rows, "arm_comparison.csv")

    # figures
    viz = config.visualization
    if viz.save_plots:
        title = f"{config.data.asset_a_symbol}/{config.data.asset_b_symbol}"
        writer.plot_spread_and_jumps(full_spread, artifacts["jump_df"],
                                    f"{title} spread and detected jumps", viz.dpi)
        writer.plot_intensity(artifacts["hawkes_intensity"],
                            bundle.hawkes_params.get("lambda_bar", 0.0),
                            f"{title} Hawkes intensity", viz.dpi)
        writer.plot_zscore(artifacts["z_score"], bundle.z_entry_threshold,
                            bundle.z_exit_threshold, f"{title} z-score", viz.dpi)
        
        curves = {}
        if "train" in results:
            curves["Hawkes (train)"] = results["train"]["equity_curve"]
        if "validation" in results:
            curves["Hawkes (val)"] = results["validation"]["equity_curve"]
        if "validation" in control_results:
            curves["Control (val)"] = control_results["validation"]["equity_curve"]
        if curves:
            writer.plot_equity(curves, f"{title} equity curves", "train_val_equity.png", viz.dpi)
        
        gof = bundle.hawkes_inference.get("_gof_full")
        if gof and "theoretical_quantiles" in gof:
            writer.plot_qq_residuals(gof, f"{title} Hawkes compensator residuals", viz.dpi)
        
    writer.manifest()
        
    if verbose:
        _print_comparison(pair_key, results, control_results, bundle)
        
    return {
        "bundle": bundle,
        "results": results,
        "control": control_results,
        "comparison": comparison_rows,
    }
        
        
def _print_comparison(
    pair_key: str, results: Dict, control: Dict, bundle
) -> None:
    print("\n" + "=" * 78)
    print(f"RESULTS -- {pair_key}")
    print("=" * 78)
        
    print(f"\n  Hedge ratio (frozen on train): {bundle.hedge_ratio:.4f}")
    print(f"  Half-life: {bundle.half_life:.1f} trading days")
    print(f"  Jumps on train: {bundle.n_jumps_fdr} (FDR) / {bundle.n_jumps_nominal} (nominal)")
        
    if bundle.hawkes_active:
        inf = bundle.hawkes_inference
        p_lr = inf.get("lr_p_bootstrap") or inf.get("lr_p_chi2_df2")
        print(f"\n  HAWKES SELF-EXCITATION TEST (H0: alpha = 0, homogeneous Poisson)")
        print(f"    LR statistic  : {inf.get('lr_statistic', float('nan')):.3f}")
        print(f"    p-value       : {p_lr:.4f}"
                f"  {'-> REJECT Poisson' if (p_lr or 1) < 0.05 else '-> CANNOT reject Poisson'}")
        br = inf.get("branching_ratio")
        lo, hi = inf.get("branching_ci95_low"), inf.get("branching_ci95_high")
        if br is not None and lo is not None:
            print(f"    branching     : {br:.4f}  CI95 [{lo:.4f}, {hi:.4f}]")
        print(f"    KS on residuals p = {inf.get('ks_pvalue', float('nan')):.4f}")
    else:
        print(f"\n  HAWKES INACTIVE: {bundle.hawkes_inactive_reason}")
        
    header = f"\n  {'metric':<34}{'Hawkes arm':>20}{'Control arm':>20}"
    for window in ("train", "validation"):
        if window not in results:
            continue
        print(f"\n  --- {window.upper()} ---")
        print(header)
        print("  " + "-" * 72)
        m = results[window]["metrics"]
        c = control.get(window, {}).get("metrics", {})
        rows = [
            ("trades", "total_trades", "{:>20.0f}"),
            ("annualised return %", "annualized_return_pct", "{:>20.3f}"),
            ("annualised vol %", "annualized_volatility_pct", "{:>20.3f}"),
            ("Sharpe", "sharpe_ratio", "{:>20.3f}"),
            ("Sharpe HAC SE", "sharpe_se_hac", "{:>20.3f}"),
            ("max drawdown %", "max_drawdown_pct", "{:>20.2f}"),
            ("return on capital-at-risk %", "car_annualized_return_pct", "{:>20.2f}"),
            ("mean excess %/yr (NW)", "nw_mean_excess_annualized_pct", "{:>20.3f}"),
            ("NW t-stat", "nw_tstat", "{:>20.3f}"),
            ("NW p-value", "nw_pvalue", "{:>20.4f}"),
            ("beta vs SPY", "beta", "{:>20.4f}"),
        ]
        for name, key, fmt in rows:
            a = m.get(key)
            b = c.get(key)
            a_s = fmt.format(a) if isinstance(a, (int, float)) and np.isfinite(a) else f"{'n/a':>20}"
            b_s = fmt.format(b) if isinstance(b, (int, float)) and np.isfinite(b) else f"{'n/a':>20}"
            print(f"  {name:<34}{a_s}{b_s}")
        
    if "validation" in results:
        stmt = results["validation"]["metrics"].get("power_statement")
        if stmt:
            print(f"\n  POWER: {stmt}")
    print("=" * 78)

### walk-forward ###

def _write_walk_forward_arm(
    pair_key: str, arm: str, config: Config, out_dir: Path, verbose: bool
) -> Dict:
    """Run and save one walk-forward arm in an isolated artifact directory."""
    writer = ResultsWriter(out_dir, verbose = verbose)

    engine = WalkForwardEngine(config, verbose = verbose)
    results = engine.run() 

    writer.write_frame(results["equity_curve"], "walk_forward_equity_curve.csv")
    writer.write_frame(results["signals"], "walk_forward_signals.csv")
    writer.write_row(results["metrics"], "walk_forward_metrics.csv")
    writer.write_frame(engine.quarterly_frame(), "quarterly_parameters.csv", index = False)
    writer.write_frame(engine.failures_frame(), "quarter_failures.csv", index = False)
    writer.write_frame(
        results["engine"].get_trade_summary(), "walk_forward_trade_summary.csv", index = False
    )
    writer.write_row(results["engine"].analyze_by_exit_reason(), "exit_reasons.csv")

    if config.visualization.save_plots and results["equity_curve"] is not None:
        title = f"{config.data.asset_a_symbol} / {config.data.asset_b_symbol}"
        writer.plot_equity(
            {"Walk-forward OOS": results["equity_curve"]},
            f"{title} walk-forward (continuous book)",
            "walk_forward_equity.png",
            config.visualization.dpi,
        )

    writer.write_json(
        {"pair": pair_key, "arm": arm, "config": config.to_dict()}, "run_config.json"
    )
    writer.manifest()
    return results


def _walk_forward_comparison(results_by_arm: Dict[str, Dict]) -> List[Dict]:
    """Create one like-for-like OOS row for each successfully run arm."""
    rows: List[Dict] = []
    for arm, results in results_by_arm.items():
        metrics = results["metrics"]
        rows.append(
            {
                "arm": arm,
                "window": "oos",
                "total_trades": metrics.get("total_trades", 0),
                "annualized_return_pct": metrics.get("annualized_return_pct", 0.0),
                "annualized_volatility_pct": metrics.get("annualized_volatility_pct", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "sharpe_se_hac": metrics.get("sharpe_se_hac", float("nan")),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
                "nw_mean_excess_annualized_pct": metrics.get(
                    "nw_mean_excess_annualized_pct", 0.0
                ),
                "nw_tstat": metrics.get("nw_tstat", 0.0),
                "nw_pvalue": metrics.get("nw_pvalue", 1.0),
                "quarters_evaluated": metrics.get("quarters_evaluated", 0),
                "quarters_failed": metrics.get("quarters_failed", 0),
                "configurations_tried": metrics.get("configurations_tried", 0),
                "deflated_sharpe_probability": metrics.get(
                    "deflated_sharpe_probability", float("nan")
                ),
            }
        )
    return rows


def run_walk_forward(
    pair_key: str, config: Config, run_control: bool = True, verbose: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Run Hawkes and matched no-Hawkes OOS arms using identical settings."""
    out_dir = output_dir or OUTPUT_ROOT / pair_key / "walk_forward"
    results_by_arm = {
        "hawkes": _write_walk_forward_arm(
            pair_key, "hawkes", config, out_dir / "hawkes", verbose
        )
    }
    if run_control:
        control_config = copy.deepcopy(config).as_control_arm()
        results_by_arm["control"] = _write_walk_forward_arm(
            pair_key, "control", control_config, out_dir / "control", verbose
        )

    comparison = _walk_forward_comparison(results_by_arm)
    writer = ResultsWriter(out_dir, verbose=verbose)
    writer.write_comparison(comparison, "oos_arm_comparison.csv")
    writer.write_json(
        {
            "pair": pair_key,
            "comparison_definition": (
                "Each arm uses the same data, walk-forward folds, execution "
                "settings, costs and seed. The control disables only Hawkes "
                "regimes and jump entries."
            ),
            "arms": list(results_by_arm),
        },
        "comparison_metadata.json",
    )
    writer.manifest()
    return {**results_by_arm, "comparison": comparison}


def _robustness_scenarios(config: Config) -> Dict[str, Dict]:
    """Pre-specified one-factor OOS scenarios; these are never tuned on OOS data."""
    return {
        "baseline": {"description": "Shipped walk-forward specification."},
        "half_cost": {
            "description": "Half the baseline commissions and slippage.",
            "commission_multiplier": 0.5,
            "slippage_multiplier": 0.5,
        },
        "double_cost": {
            "description": "Twice the baseline commissions and slippage.",
            "commission_multiplier": 2.0,
            "slippage_multiplier": 2.0,
        },
        "fixed_entry_1_5": {
            "description": "Fixed z-entry 1.5 and exit 0.5; no threshold tuning.",
            "z_entry": 1.5,
            "z_exit": 0.5,
        },
        "fixed_entry_2_5": {
            "description": "Fixed z-entry 2.5 and exit 0.5; no threshold tuning.",
            "z_entry": 2.5,
            "z_exit": 0.5,
        },
        "bipower_detector": {
            "description": "Use the pre-specified bipower jump detector.",
            "detector": "bipower",
        },
        "static_ols_hedge": {
            "description": "Use a static OLS hedge estimator rather than Johansen.",
            "hedge_mode": "static",
            "hedge_ratio_method": "ols",
        },
    }


def _apply_robustness_scenario(config: Config, scenario: Dict) -> Config:
    configured = copy.deepcopy(config)
    configured.backtest.commission_rate *= scenario.get("commission_multiplier", 1.0)
    configured.backtest.slippage_bps *= scenario.get("slippage_multiplier", 1.0)
    if "z_entry" in scenario:
        configured.walk_forward.tune_thresholds = False
        configured.trading.z_entry_threshold = scenario["z_entry"]
        configured.trading.z_exit_threshold = scenario["z_exit"]
    if "detector" in scenario:
        configured.jump_detection.method = scenario["detector"]
    if "hedge_mode" in scenario:
        configured.data.hedge_mode = scenario["hedge_mode"]
    if "hedge_ratio_method" in scenario:
        configured.data.hedge_ratio_method = scenario["hedge_ratio_method"]
    return configured


def run_robustness(
    pair_key: str, config: Config, run_control: bool = True, verbose: bool = True
) -> List[Dict]:
    """Execute the fixed robustness matrix and write a pair-level summary."""
    root = OUTPUT_ROOT / pair_key / "robustness"
    scenarios = _robustness_scenarios(config)
    writer = ResultsWriter(root, verbose=verbose)
    writer.write_json(scenarios, "scenario_definitions.json")
    rows: List[Dict] = []
    for name, definition in scenarios.items():
        if verbose:
            print(f"\nROBUSTNESS -- {pair_key} / {name}")
        try:
            results = run_walk_forward(
                pair_key,
                _apply_robustness_scenario(config, definition),
                run_control=run_control,
                verbose=verbose,
                output_dir=root / name,
            )
        except Exception as exc:  # noqa: BLE001 -- preserve the remaining scenarios
            rows.append(
                {
                    "pair": pair_key,
                    "scenario": name,
                    "arm": "unavailable",
                    "window": "oos",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if verbose:
                print(f"  scenario failed: {type(exc).__name__}: {exc}")
            continue
        for row in results["comparison"]:
            rows.append({"pair": pair_key, "scenario": name, **row})
    writer.write_comparison(rows, "walk_forward_robustness.csv")
    writer.manifest()
    return rows


def _write_walk_forward_portfolio(
    returns_by_pair: Dict[str, pd.Series], arm: str, risk_free_rate: float,
    verbose: bool,
) -> Optional[Dict]:
    """Write a transparent equal-weight portfolio for one OOS strategy arm."""
    if len(returns_by_pair) < 2:
        return None

    from statistics_tools import pool_pair_returns

    pooled = pool_pair_returns(returns_by_pair, risk_free_rate=risk_free_rate)
    if "error" in pooled:
        return None
    portfolio_returns = pooled.pop("portfolio_returns")
    out_dir = OUTPUT_ROOT / "portfolio" / "walk_forward" / arm
    writer = ResultsWriter(out_dir, verbose=verbose)
    pair_returns = pd.DataFrame(returns_by_pair).dropna(how="all").fillna(0.0)
    writer.write_frame(pair_returns, "pair_returns.csv")
    writer.write_frame(pair_returns.corr(), "pair_return_correlation.csv")
    writer.write_frame(
        pd.DataFrame(
            {"pair": list(pair_returns.columns), "weight": 1.0 / len(pair_returns.columns)}
        ),
        "weights.csv",
        index=False,
    )
    writer.write_frame(portfolio_returns.to_frame("returns"), "portfolio_returns.csv")
    writer.write_row(pooled, "portfolio_metrics.csv")
    writer.write_json(
        {
            "arm": arm,
            "weighting": "equal weight across pairs",
            "missing_return_policy": "Missing pair returns are treated as 0.0 (flat).",
            "risk_free_rate": risk_free_rate,
            "source": "Continuous walk-forward out-of-sample daily returns.",
        },
        "portfolio_definition.json",
    )
    writer.manifest()
    return pooled

### CLI ###

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description = "MRJD + Hawkes pairs trading study",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = "Pairs: " + ", ".join(sorted(PAIRS)) + ", or 'all'",
    )
    parser.add_argument("--pair", default = "CVX_XOM",
                        help = "pair key from the registry, or 'all'")
    parser.add_argument("--mode", default = "train_val",
                        choices = ["train_val", "walk_forward", "all", "portfolio", "robustness"])
    parser.add_argument("--screened", action = "store_true",
                        help = "use pairs surviving pair_screen.py instead of the registry")
    parser.add_argument("--no-control", action = "store_true",
                        help = "skip the no-Hawkes control arm")
    parser.add_argument("--no-fdr", action="store_true",
                        help="use nominal jump detection instead of BH-FDR control")
    parser.add_argument("--hedge-mode", default = None, 
                        choices = ["static", "periodic", "rolling"],
                        help = "override the hedge-ratio mode (default: static)")
    parser.add_argument("--detector", default = None,
                        choices = ["lee_mykland", "bipower", "threshold"])
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--quiet", action = "store_true")
    return parser 

def verbose_flag(args) -> bool:
    return not args.quiet 

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    set_seeds(args.seed)

    if args.screened:
        from pair_screen import screen_all, selected_pairs

        screen = screen_all(verbose = verbose_flag(args))
        chosen = selected_pairs(screen, criterion = "selected_fdr")
        if not chosen:
            print(
                "\n No pair survives the FDR-corrected screen. Trading the "
                "uncorrected survivors would be selected on multiplicity -- see "
                "pair_screen.py. Falling back to the registry, clearly labelled.",
                file = sys.stderr,
            )
            pair_keys = sorted(PAIRS)
        else:
            PAIRS.update(chosen)
            pair_keys = sorted(chosen)
    else:
        pair_keys = sorted(PAIRS) if args.pair == "all" else [args.pair]
    for key in pair_keys:
        if key not in PAIRS:
            print(f"Unknown pair '{key}'. Known: {', '.join(sorted(PAIRS))}", file = sys.stderr)
            return 2 
        
    verbose = not args.quiet 
    if args.mode == "all":
        modes = ["train_val", "walk_forward"]
    elif args.mode == "portfolio":
        modes = ["walk_forward"]
    elif args.mode == "robustness":
        modes = ["robustness"]
    else:
        modes = [args.mode]

    if verbose:
        print("=" * 72)
        print("MRJD + HAWKES PAIRS TRADING")
        print(f" pairs: {', '.join(pair_keys)}")
        print(f" modes: {', '.join(modes)}")
        print(f" seed: {args.seed}")
        print("=" * 72)
        print("\n DIVIDEND TREATMENT")
        print(DIVIDEND_TREATMENT)

    summary_rows: List[Dict] = []
    failures: List[Dict] = []
    portfolio_returns: Dict[str, Dict[str, pd.Series]] = {"hawkes": {}, "control": {}}

    for key in pair_keys:
        for mode in modes:
            config = Config().for_pair(key)
            config.statistics.seed = args.seed
            if args.hedge_mode:
                config.data.hedge_mode = args.hedge_mode
            if args.detector:
                config.jump_detection.method = args.detector
            if args.no_fdr:
                config.jump_detection.apply_fdr = False 

            try:
                if mode == "train_val":
                    out = run_train_val(
                        key, config, run_control = not args.no_control, verbose = verbose
                    )
                    for row in out["comparison"]:
                        summary_rows.append({"pair": key, "mode": mode, **row})
                elif mode == "walk_forward":
                    out = run_walk_forward(
                        key, config, run_control=not args.no_control, verbose=verbose
                    )
                    for row in out["comparison"]:
                        summary_rows.append({"pair": key, "mode": mode, **row})
                    for arm in ("hawkes", "control"):
                        result = out.get(arm)
                        if result is not None:
                            portfolio_returns[arm][key] = result["equity_curve"]["returns"]
                else:
                    rows = run_robustness(
                        key, config, run_control=not args.no_control, verbose=verbose
                    )
                    summary_rows.extend({"mode": mode, **row} for row in rows)
            except Exception as exc: # noqa: BLE001 -- reported, never silent
                failures.append(
                    {"pair": key, "mode": mode,
                     "error_type": type(exc).__name__, "error": str(exc)}
                )
                print(f"\n!! {key} / {mode} FAILED: {type(exc).__name__}: {exc}",
                      file = sys.stderr)
                if verbose:
                    traceback.print_exc()

    if summary_rows:
        OUTPUT_ROOT.mkdir(parents = True, exist_ok = True)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(OUTPUT_ROOT / "summary_all_pairs.csv", index = False)
        if verbose:
            print("\n" + "=" * 78)
            print("SUMMARY -- ALL PAIRS")
            print("=" * 78)
            cols = [c for c in ("pair", "mode", "arm", "window", "total_trades",
                                "annualized_return_pct", "sharpe_ratio", "nw_tstat",
                                "nw_pvalue") if c in summary.columns]
            print(summary[cols].to_string(index = False))
            print(f"\n Wrote {OUTPUT_ROOT / 'summary_all_pairs.csv'}")

    if args.mode == "portfolio":
        for arm, returns in portfolio_returns.items():
            pooled = _write_walk_forward_portfolio(
                returns, arm, Config().backtest.risk_free_rate, verbose
            )
            if pooled is None:
                continue
            print("\n" + "=" * 78)
            print(f"EQUAL-WEIGHT PORTFOLIO ACROSS PAIRS (walk-forward OOS, {arm})")
            print("=" * 78)
            print(f" pairs pooled {pooled['n_pairs']}")
            print(f" mean pairwise correlation {pooled['mean_pairwise_correlation']:+.3f}")
            print(f" effective independent pairs {pooled['effective_independent_pairs']:.2f}")
            print(f" breadth gain vs one pair {pooled['breadth_gain_vs_single']:.2f}x")
            print(f"\n pooled mean excess "
                  f"{pooled.get('pooled_mean_annualized_excess_pct', 0):+.3f}%/yr")
            print(f" pooled t-statistic {pooled.get('pooled_t_statistic', 0):+.3f}")
            print(f" pooled p-value {pooled.get('pooled_p_value', 1):.4f}")
            print(f" pooled Sharpe "
                  f"{pooled.get('pooled_sharpe_annualized', 0):+.3f} "
                  f"(HAC SE {pooled.get('pooled_se_annualized_hac', float('nan')):.3f})")
            print(f" artifacts {OUTPUT_ROOT / 'portfolio' / 'walk_forward' / arm}")
            print("=" * 78)

    if failures:
        pd.DataFrame(failures).to_csv(OUTPUT_ROOT / "run_failures.csv", index = False)
        print(f"\n{len(failures)} run(s) failed, see outputs/run_failures.csv",
              file = sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
