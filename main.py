"""
Optimized Main Integration Script for Self-Exciting Pairs Trading V4

Key Optimizations:
1. Half-life aware signal generation and position management
2. Trailing stop losses in backtest engine
3. Full Hawkes process utilization (regime detection, λ decay, dynamic thresholds)
4. Fewer, higher-quality trades
5. Hawkes suitability analysis for pair selection
"""

import sys
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import traceback
from typing import Callable, Dict, IO, Optional, TypeVar
import numpy as np 
import pandas as pd
import warnings
from dataclasses import dataclass
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import modules
from equity_pairs_loader import EquityPairsDataPipeline
from jump_detector import JumpDetector
from hawkes_calibration import HawkesProcess
from mrjd_estimation import MRJDModel
from config_old import ConfigV2, diagnose_signal_generation, print_diagnostics
from signal_generation_old import TradingSignals
from backtest_engine import BacktestEngineV2, compute_alpha_tstat

T = TypeVar("T")

@dataclass
class ModelBundle:
    """
    Frozen model parameters fitted exclusively on training data.

    Passed to evaluation phases so no in-sample information leaks
    into out-of-sample backtest periods.
    """
    hawkes_params: Dict #lambda_bar, alpha, beta
    mrjd_params: Dict #kappa, theta, sigma, jump_mean, jump_std
    half_life: float #empirical half-life from training spread 
    lambda_percentiles: Dict #p25/50/75/90/95 of training Hawkes intensity
    hawkes_suitability: Dict #Hawkes scoring dict (including use_hawkes_regimes)
    pair_validation: Dict #ADF/half-life/mean-stability results
    use_hawkes_regimes: bool #whether Hawkes regime logic is active


class TeeOutput:
    """Write script output to the terminal and a log file."""

    def __init__(self, *streams: IO[str]) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def run_with_output_capture(output_filename: str, func: Callable[[], T]) -> T:
    output_path = Path(__file__).with_name(output_filename)

    with output_path.open("w", encoding="utf-8") as output_file:
        stdout_tee = TeeOutput(sys.__stdout__, output_file)
        stderr_tee = TeeOutput(sys.__stderr__, output_file)

        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print(f"Saving terminal output to: {output_path}")
            try:
                return func()
            except Exception:
                traceback.print_exc()
                raise


def validate_pair(spread: pd.Series, min_half_life: int = 5, max_half_life: int = 60) -> dict:
    """Validate if the pair is suitable for mean-reversion trading"""
    from statsmodels.tsa.stattools import adfuller
    
    print("\n" + "="*70)
    print("PAIR VALIDATION")
    print("="*70)
    
    # 1. Check stationarity
    adf_result = adfuller(spread.dropna())
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]
    is_stationary = adf_pvalue < 0.05
    
    print(f"\n1. Stationarity Test (ADF):")
    print(f"   ADF Statistic: {adf_stat:.4f}")
    print(f"   P-value: {adf_pvalue:.4f}")
    print(f"   Stationary: {' YES' if is_stationary else ' NO'}")
    
    # 2. Calculate half-life
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    common_idx = spread_lag.index.intersection(spread_diff.index)
    
    beta = np.polyfit(spread_lag.loc[common_idx], spread_diff.loc[common_idx], 1)[0]
    half_life = -np.log(2) / beta if beta < 0 else np.inf
    
    print(f"\n2. Mean Reversion Half-Life:")
    print(f"   Half-life: {half_life:.1f} days")
    print(f"   Acceptable range: {min_half_life}-{max_half_life} days")
    
    half_life_ok = min_half_life <= half_life <= max_half_life
    print(f"   Within range: {' YES' if half_life_ok else ' NO'}")
    
    # 3. Mean stability
    rolling_mean_252 = spread.rolling(252).mean()
    mean_drift = rolling_mean_252.std() / spread.std()
    stable_mean = mean_drift < 0.5
    
    print(f"\n3. Mean Stability Check:")
    print(f"   Long-term mean drift: {mean_drift:.4f}")
    print(f"   Stable mean: {' YES' if stable_mean else ' NO (structural break likely)'}")
    
    # 4. Spread range
    spread_range = spread.max() - spread.min()
    range_in_std = spread_range / spread.std()
    reasonable_range = range_in_std < 10
    
    print(f"\n4. Spread Range Analysis:")
    print(f"   Range: {spread_range:.4f}")
    print(f"   Range in std units: {range_in_std:.1f}")
    print(f"   Reasonable range (<10 std): {' YES' if reasonable_range else ' NO (extreme moves)'}")
    
    # 5. Recent regime
    recent_252 = spread.iloc[-252:] if len(spread) > 252 else spread
    recent_mean = recent_252.mean()
    overall_mean = spread.mean()
    mean_shift = abs(recent_mean - overall_mean) / spread.std()
    no_regime_change = mean_shift < 1.0
    
    print(f"\n5. Recent vs Historical Mean:")
    print(f"   Overall mean: {overall_mean:.4f}")
    print(f"   Recent 1yr mean: {recent_mean:.4f}")
    print(f"   Shift in std: {mean_shift:.2f}")
    print(f"   No recent regime change: {' YES' if no_regime_change else ' NO'}")
    
    # Overall assessment
    is_tradeable = is_stationary and half_life_ok and stable_mean and reasonable_range
    
    print("\n" + "-"*70)
    print("OVERALL ASSESSMENT:")
    if is_tradeable:
        print(" PAIR IS SUITABLE FOR MEAN-REVERSION TRADING")
    else:
        print(" PAIR HAS ISSUES - TRADING NOT RECOMMENDED")
        print("\nIssues detected:")
        if not is_stationary:
            print("  - Spread is not stationary")
        if not half_life_ok:
            print(f"  - Half-life ({half_life:.1f}d) outside acceptable range")
        if not stable_mean:
            print("  - Long-term mean is unstable")
        if not reasonable_range:
            print(f"  - Spread has extreme moves ({range_in_std:.1f} std)")
    print("="*70)
    
    return {
        'is_tradeable': is_tradeable,
        'is_stationary': is_stationary,
        'adf_pvalue': adf_pvalue,
        'half_life': half_life,
        'half_life_ok': half_life_ok,
        'stable_mean': stable_mean,
        'mean_drift': mean_drift,
        'range_in_std': range_in_std,
        'reasonable_range': reasonable_range,
        'no_regime_change': no_regime_change,
        'mean_shift': mean_shift
    }


def analyze_hawkes_suitability(jump_df: pd.DataFrame, hawkes_params: Dict, 
                                lambda_intensity: pd.Series) -> Dict:
    """
    Analyze whether this pair is suitable for Hawkes-based trading
    
    High suitability = Hawkes adds significant value for timing
    Low suitability = Simple z-score sufficient
    """
    print("\n" + "="*70)
    print("HAWKES SUITABILITY ANALYSIS")
    print("="*70)
    
    # Calculate metrics
    n_jumps = jump_df['jump_indicator'].sum()
    n_obs = len(jump_df)
    jump_freq = n_jumps / n_obs
    
    branching_ratio = hawkes_params.get('alpha', 0) / hawkes_params.get('beta', 1)
    baseline_intensity = hawkes_params.get('lambda_bar', 0.01)
    
    # λ variability
    lambda_std = lambda_intensity.std()
    lambda_mean = lambda_intensity.mean()
    lambda_cv = lambda_std / lambda_mean if lambda_mean > 0 else 0
    
    # Time in elevated state
    lambda_90 = lambda_intensity.quantile(0.90)
    time_elevated = (lambda_intensity > lambda_90 * 0.5).mean()
    
    print(f"\n1. Jump Frequency: {jump_freq*100:.1f}%")
    if jump_freq < 0.07:
        print("   LOW - Not enough jumps for Hawkes to add value")
        jump_score = 0.3
    elif jump_freq < 0.12:
        print("   MODERATE - Hawkes can help with regime detection")
        jump_score = 0.6
    elif jump_freq < 0.25:
        print("   GOOD - Sufficient jumps for meaningful Hawkes modeling")
        jump_score = 0.9
    else:
        print("   HIGH - Excellent for Hawkes self-excitation modeling")
        jump_score = 1.0
    
    print(f"\n2. Branching Ratio (α/β): {branching_ratio:.3f}")
    if branching_ratio < 0.3:
        print("   LOW - Jumps don't cluster, limited self-excitation")
        branch_score = 0.4
    elif branching_ratio < 0.6:
        print("   MODERATE - Some jump clustering present")
        branch_score = 0.7
    elif branching_ratio < 0.9:
        print("   GOOD - Strong jump clustering, Hawkes very useful")
        branch_score = 1.0
    else:
        print("   HIGH - Near-critical process, may be unstable")
        branch_score = 0.7
    
    print(f"\n3. λ Coefficient of Variation: {lambda_cv:.2f}")
    if lambda_cv < 0.5:
        print("   LOW - Intensity doesn't vary much")
        cv_score = 0.4
    elif lambda_cv < 1.0:
        print("   MODERATE - Some regime variation")
        cv_score = 0.7
    else:
        print("   HIGH - Strong regime variation, Hawkes valuable")
        cv_score = 1.0
    
    print(f"\n4. Time in Elevated State: {time_elevated*100:.1f}%")
    if time_elevated < 0.1:
        print("   LOW - Rarely in high-intensity regime")
        elevated_score = 0.5
    elif time_elevated < 0.25:
        print("   GOOD - Meaningful time in elevated regimes")
        elevated_score = 0.9
    else:
        print("   HIGH - Often volatile, Hawkes useful for filtering")
        elevated_score = 0.7
    
    # Overall score
    overall_score = (jump_score + branch_score + cv_score + elevated_score) / 4
    
    print(f"\n" + "-"*70)
    print(f"OVERALL HAWKES SUITABILITY SCORE: {overall_score:.2f}/1.00")
    
    if overall_score >= 0.8:
        recommendation = "EXCELLENT - Hawkes should significantly improve timing"
        use_hawkes = True
    elif overall_score >= 0.6:
        recommendation = "GOOD - Hawkes useful for regime detection and filtering"
        use_hawkes = True
    elif overall_score >= 0.4:
        recommendation = "MARGINAL - Hawkes provides limited value"
        use_hawkes = False
    else:
        recommendation = "POOR - Consider simpler z-score model or different pairs"
        use_hawkes = False
    
    print(f"RECOMMENDATION: {recommendation}")
    
    if not use_hawkes:
        print("\n  Find different pair for trading")
    
    print("="*70)
    
    return {
        'overall_score': overall_score,
        'jump_frequency': jump_freq,
        'branching_ratio': branching_ratio,
        'lambda_cv': lambda_cv,
        'time_elevated': time_elevated,
        'recommendation': recommendation,
        'use_hawkes_regimes': use_hawkes
    }


class SelfExcitingPairsTradingV4:
    """
    Optimized self-exciting pairs trading system
    
    Key features:
    - Half-life aware exits
    - Trailing stop losses
    - Full Hawkes utilization when suitable
    - Dynamic regime-based thresholds
    """

    def __init__(self, config: Optional[ConfigV2] = None):
        self.config = config if config is not None else ConfigV2()
        
        # Components
        self.data_pipeline = None 
        self.jump_detector = None 
        self.hawkes_model = None 
        self.mrjd_model = None 
        self.signal_generator = None 
        self.backtest_engine = None 

        # Data
        self.cleaned_data = None
        self.spread_df = None 
        self.jump_df = None 
        self.signals_df = None 
        self.equity_curve = None 
        self.z_score = None
        self.hawkes_intensity = None
        self.half_life = None

        # Results
        self.results = {}
        self.diagnostics = {}
        self.pair_validation = {}
        self.hawkes_suitability = {}

    def run_complete_pipeline(self, skip_validation: bool = False):
        """Execute optimized trading pipeline"""

        print("\n" + "=" * 70)
        print("Self-Exciting Pairs Trading System")
        print(f"{self.config.data.asset_a_symbol}/{self.config.data.asset_b_symbol} "
              "Equity Pairs Strategy (Dollar-Neutral)")
        print("=" * 70)

        # Step 1: Data
        print("\n [Step 1/9] Data Acquisition")
        print("-" * 70)
        self._acquire_data()

        if self.spread_df is None:
            raise ValueError("Pair validation requires spread_df")
        spread_df = self.spread_df

        # Step 2: Pair Validation
        print("\n [Step 2/9] Pair Validation")
        print("-" * 70)
        self.pair_validation = validate_pair(spread_df['spread'])
        self.half_life = self.pair_validation['half_life']
        
        if not self.pair_validation['is_tradeable'] and not skip_validation:
            print("\n⚠ WARNING: Pair validation failed!")
            print("Continuing anyway for analysis...")

        # Step 3: Jump Detection
        print("\n [Step 3/9] Jump Detection")
        print("-" * 70)
        self._detect_jumps() 

        # Step 4: Hawkes Calibration
        print("\n [Step 4/9] Hawkes Process Calibration")
        print("-" * 70)
        self._calibrate_hawkes()

        # Step 5: Hawkes Suitability Analysis
        print("\n [Step 5/9] Hawkes Suitability Analysis")
        print("-" * 70)
        self._analyze_hawkes_suitability()

        # Step 6: MRJD Estimation
        print("\n [Step 6/9] MRJD Model Estimation")
        print("-" * 70)
        self._estimate_mrjd() 

        # Step 7: Diagnostics
        print("\n [Step 7/9] Signal Generation Diagnostics")
        print("-" * 70)
        self._run_diagnostics()

        # Step 8: Signal Generation
        print("\n [Step 8/9] Trading Signal Generation")
        print("-" * 70)
        self._generate_signals()

        # Step 9: Backtest
        print("\n [Step 9/9] Backtesting (Trailing Stops)")
        print("-" * 70)
        self._run_backtest()

        # Print Summary
        print("\n" + "="*70)
        print("Pipeline Complete - Generating Summary")
        print("="*70)
        self._print_summary()

        return self.results

    def _acquire_data(self):
        """Load and process data"""
        cfg = self.config.data

        self.data_pipeline = EquityPairsDataPipeline(
            asset_a_path=cfg.asset_a_csv,
            asset_b_path=cfg.asset_b_csv,
            asset_a_symbol=cfg.asset_a_symbol,
            asset_b_symbol=cfg.asset_b_symbol
        )

        self.data_pipeline.load_from_csv(
            asset_a_path=cfg.asset_a_csv,
            asset_b_path=cfg.asset_b_csv,
            date_columns=cfg.date_columns
        )

        self.cleaned_data = self.data_pipeline.clean_data() 
        self.spread_df = self.data_pipeline.construct_spread(
            method=cfg.hedge_ratio_method,
            lookback=cfg.lookback_period
        )

        stats = self.data_pipeline.calculate_spread_statistics(self.spread_df['spread'])
        
        print(f"\n Spread Statistics ({cfg.asset_a_symbol}/{cfg.asset_b_symbol})")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std: {stats['std']:.4f}")
        print(f"    Half-life: {stats['half_life']:.2f} days")
        print(f"    Stationary: {stats['is_stationary']}")

        self.results['spread_stats'] = stats

    def _detect_jumps(self):
        """Detect jumps in the spread"""
        cfg = self.config.jump_detection

        if self.spread_df is None:
            raise ValueError("Jump detection requires spread_df")
        spread_df = self.spread_df

        self.jump_detector = JumpDetector(significance_level=cfg.significance_level)
        returns = spread_df['spread'].pct_change().dropna()
        
        self.jump_df = self.jump_detector.detect_jumps_bipower_variation(
            returns, window=cfg.window_size
        )
        
        jump_stats = self.jump_detector.calculate_jump_statistics(self.jump_df)
        
        print(f"Using window size: {cfg.window_size} periods")
        print("Detecting jumps using Bipower Variation")
        print(f"Detected {jump_stats['n_jumps']} jumps ({jump_stats['jump_frequency']*100:.2f}% of observations)")
        
        print(f"\n Jump Statistics")
        print(f"    Total Jumps: {jump_stats['n_jumps']}")
        print(f"    Jump Frequency: {jump_stats['jump_frequency']:.4f}")
        print(f"    Clustering coefficient: {jump_stats['clustering_coefficient']:.4f}")

        self.results['jump_stats'] = jump_stats

    def _calibrate_hawkes(self):
        """Calibrate Hawkes process"""
        if self.jump_detector is None or self.jump_df is None or self.spread_df is None:
            raise ValueError("Hawkes calibration requires jump_detector, jump_df, and spread_df")

        jump_detector = self.jump_detector
        jump_df = self.jump_df
        spread_df = self.spread_df

        self.hawkes_model = HawkesProcess()
        jump_times = jump_detector.extract_jump_times(jump_df)
        
        if isinstance(jump_df.index, pd.DatetimeIndex):
            T = (jump_df.index[-1] - jump_df.index[0]).total_seconds() / (24*3600)
        else:
            T = float(len(jump_df) - 1)
        T = max(T, 1.0)

        if len(jump_times) >= 5:
            hawkes_params = self.hawkes_model.fit(jump_times, T)
            self.hawkes_intensity = self.hawkes_model.compute_intensity_at_dates(
                jump_times, pd.DatetimeIndex(spread_df.index), T
            )
        else:
            print("  Warning: Too few jumps for Hawkes calibration")
            hawkes_params = {'lambda_bar': 0.01, 'alpha': 0.1, 'beta': 1.0}
            self.hawkes_intensity = pd.Series(0.01, index=spread_df.index)
        
        print("Fitting Hawkes process using MLE")
        print(f" Hawkes parameters estimate: ")
        print(f" λ̄ (baseline intensity): {hawkes_params['lambda_bar']:.4f}")
        print(f" α (jump impact): {hawkes_params['alpha']:.4f}")
        print(f" β_H (decay rate): {hawkes_params['beta']:.4f}")
        
        branching = hawkes_params['alpha'] / hawkes_params['beta']
        print(f" Branching ratio (α/β_H): {branching:.4f}")
        
        print(f"\n Jump Intensity Statistics:")
        print(f"    Mean λ: {self.hawkes_intensity.mean():.6f}")
        print(f"    90th percentile: {self.hawkes_intensity.quantile(0.90):.6f}")

        self.results['hawkes_params'] = hawkes_params

    def _analyze_hawkes_suitability(self):
        """Analyze Hawkes suitability"""
        if self.jump_df is None or self.hawkes_intensity is None:
            raise ValueError("Hawkes suitability analysis requires jump_df and hawkes_intensity")
        if 'hawkes_params' not in self.results:
            raise ValueError("Hawkes suitability analysis requires hawkes_params in results")

        jump_df = self.jump_df
        hawkes_intensity = self.hawkes_intensity
        hawkes_params = self.results['hawkes_params']

        self.hawkes_suitability = analyze_hawkes_suitability(
            jump_df,
            hawkes_params,
            hawkes_intensity
        )
        self.results['hawkes_suitability'] = self.hawkes_suitability

    def _estimate_mrjd(self):
        """Estimate MRJD model"""
        cfg = self.config.mrjd 

        if self.spread_df is None or self.jump_df is None or self.pair_validation is None:
            raise ValueError("MRJD estimation requires spread_df, jump_df, and pair_validation")
        
        self.mrjd_model = MRJDModel()
        mrjd_model = self.mrjd_model
        
        spread_df = self.spread_df
        jump_df = self.jump_df
        pair_validation = self.pair_validation
        
        common_index = spread_df.index.intersection(jump_df.index)
        spread = spread_df.loc[common_index, 'spread']
        jump_indicator = jump_df.loc[common_index, 'jump_indicator']

        print("Fitting the MRJD model")
        print(" Step 1: Estimating OU params from continous periods")
        
        self.mrjd_params = mrjd_model.fit(spread, jump_indicator, dt=cfg.dt, method=cfg.estimation_method)
        
        # Check for half-life mismatch
        mrjd_half_life = np.log(2) / self.mrjd_params['kappa'] if self.mrjd_params['kappa'] > 0 else np.inf
        empirical_half_life = pair_validation['half_life']
        
        if abs(mrjd_half_life - empirical_half_life) > empirical_half_life * 0.5:
            print(f"\n{'='*70}")
            print("Critical: MRJD Half Life Mismatch Detected")
            print('='*70)
            print(f"    MRJD estimated half-life: {mrjd_half_life:.1f} days")
            print(f"    Empirical half-life: {empirical_half_life:.1f} days")
            print(f"\n Overiding MRJD kappa with the empirical estimate instead")
            
            corrected_kappa = np.log(2) / empirical_half_life
            print(f"\n Corrected kappa: {self.mrjd_params['kappa']:.6f} -> {corrected_kappa:.6f}")
            self.mrjd_params['kappa'] = corrected_kappa
            print('='*70)
        
        print(" Step 2: Estimating jump distribution")
        print(" Step 3: Joint MLE refinement")
        print("Warning: Skipping joint MLE to preserve the validated kappa")
        
        print(f"MRJD Model Parameters:")
        print(f" κ (mean reversion speed): {self.mrjd_params['kappa']:.4f}")
        print(f" θ (long-run mean): {self.mrjd_params['theta']:.4f}")
        print(f" σ (diffusive volatility): {self.mrjd_params['sigma']:.6f}")
        jump_mean = self.mrjd_params.get('jump_mean', self.mrjd_params.get('mu_J', 0.0))
        jump_std = self.mrjd_params.get('jump_std', self.mrjd_params.get('sigma_J', 0.0))
        print(f"  μ_J (jump mean): {jump_mean:.4f}")
        print(f"  σ_J (jump volatility): {jump_std:.4f}")

        self.z_score_mrjd = mrjd_model.calculate_z_score(spread_df['spread'])
        self.results['mrjd_params'] = self.mrjd_params

    def _run_diagnostics(self):
        """Run signal generation diagnostics"""
        if self.spread_df is None or self.hawkes_intensity is None:
            raise ValueError("Diagnostics require spread_df and hawkes_intensity")

        spread_df = self.spread_df
        hawkes_intensity = self.hawkes_intensity
        
        # Calculate empirical z-score
        spread = spread_df['spread']
        lookback = self.config.trading.z_score_lookback
        
        rolling_mean = spread.rolling(window=lookback, min_periods=20).mean()
        rolling_std = spread.rolling(window=lookback, min_periods=20).std()
        self.z_score = ((spread - rolling_mean) / rolling_std).fillna(0)
        
        print(f"\n Z-Score Comparison:")
        print(f"    Empirical Z-score range: [{self.z_score.min():.2f}, {self.z_score.max():.2f}]")
        print(f"    Empirical Z-score std: {self.z_score.std():.2f}")
        
        self.diagnostics = diagnose_signal_generation(
            spread_df['spread'],
            hawkes_intensity,
            self.z_score,
            self.config.trading
        )
        
        print_diagnostics(self.diagnostics)

    def _generate_signals(self):
        """Generate optimized trading signals"""
        cfg = self.config.trading

        if (
            self.spread_df is None
            or self.jump_df is None
            or self.hawkes_intensity is None
            or self.z_score is None
            or self.half_life is None
        ):
            raise ValueError(
                "Signal generation requires spread_df, jump_df, hawkes_intensity, z_score, and half_life"
            )

        spread_df = self.spread_df
        jump_df = self.jump_df
        hawkes_intensity = self.hawkes_intensity
        z_score = self.z_score
        half_life = self.half_life
        
        # Create jump indicator
        jump_indicator = pd.Series(0, index=spread_df.index)
        jump_times = jump_df[jump_df['jump_indicator'] == 1].index
        common_jumps = jump_indicator.index.intersection(jump_times)
        jump_indicator.loc[common_jumps] = 1
        
        # Determine if Hawkes regimes should be used
        use_hawkes = self.hawkes_suitability.get('use_hawkes_regimes', False)
        
        print("Generating improved jump-aware trading signals")
        if use_hawkes:
            print("  Using full Hawkes regime detection")
        else:
            print("  Using standard thresholds (Hawkes suitability low)")
        
        # Create signal generator with half-life awareness
        self.signal_generator = TradingSignals(
            z_entry_threshold=cfg.z_entry_threshold,
            z_exit_threshold=cfg.z_exit_threshold,
            lambda_threshold=cfg.lambda_threshold,
            scaling_constant=cfg.scaling_constant,
            max_position_size=0.25,  # Conservative
            use_jump_entries=cfg.use_jump_entries,
            use_hawkes_regimes=use_hawkes,
            z_lookback=cfg.z_score_lookback,
            max_holding_period=cfg.max_holding_period
        )
        
        # Set half-life for exit timing
        self.signal_generator.set_half_life(half_life)

        self.signals_df = self.signal_generator.generate_signals(
            spread=spread_df['spread'],
            lambda_intensity=hawkes_intensity,
            jump_indicator=jump_indicator,
            z_score=z_score,
            half_life=half_life
        )

        signal_quality = self.signal_generator.calculate_signal_quality(self.signals_df)

        print(f"\n Signal Quality:")
        print(f"    Time in Market: {signal_quality['time_in_market']:.2%}")
        print(f"    Entry Discipline: {signal_quality['entry_discipline']:.2%}")

        self.results['signal_quality'] = signal_quality

    def _run_backtest(self):
        """Run backtest with trailing stops"""
        cfg = self.config.backtest 

        if (
            self.spread_df is None
            or self.cleaned_data is None
            or self.signals_df is None
            or self.half_life is None
        ):
            raise ValueError("Backtest requires spread_df, cleaned_data, signals_df, and half_life")

        spread_df = self.spread_df
        cleaned_data = self.cleaned_data
        signals_df = self.signals_df
        half_life = float(self.half_life)

        print("Running backtest...")
        print(f"  Hedge ratio (for spread construction): {spread_df['hedge_ratio'].mean():.4f}")
        print(f"  Position sizing: Dollar-neutral (equal $ per leg)")
        print(f"  Max position: 25% of capital")
        print(f"  Stop loss: 3.0%")
        print(f"  Trailing stop: 1.5%")

        self.backtest_engine = BacktestEngineV2(
            initial_capital=cfg.initial_capital,
            commission_rate=cfg.commission_rate,
            slippage_bps=cfg.slippage_bps,
            max_position_pct=0.25,
            stop_loss_pct=0.03,
            trailing_stop_pct=0.015,
            trailing_activation_pct=0.01,
            profit_target_pct=0.06
        )

        # Set half-life
        self.backtest_engine.set_half_life(half_life)

        asset_a_prices = cleaned_data['asset_a']['Close']
        asset_b_prices = cleaned_data['asset_b']['Close']
        
        hedge_ratio = spread_df['hedge_ratio'].mean() if 'hedge_ratio' in spread_df.columns else 1.0

        self.equity_curve = self.backtest_engine.run_backtest(
            signals_df,
            spread_df,
            asset_a_prices,
            asset_b_prices,
            hedge_ratio=hedge_ratio
        )

        self.performance_metrics = self.backtest_engine.calculate_performance_metrics(
            risk_free_rate=cfg.risk_free_rate
        )

        #Alpha t-stat vs SPY benchmark
        alpha_stats = compute_alpha_tstat(
            self.equity_curve,
            cfg.benchmark_csv,
            cfg.risk_free_rate
        )

        regime_perf = self.backtest_engine.analyze_regime_performance(regime_threshold=0.1)

        self.results['performance'] = self.performance_metrics 
        self.results['regime_performance'] = regime_perf

    # Train / Validation Pipeline Helpers

    def _fit_models(self, train_spread_df: pd.DataFrame, train_cleaned_data: Dict) -> ModelBundle:
        """
        Fit all models on a training data slice and return frozen parameters
        """
        cfg_jd = self.config.jump_detection
        cfg_mrjd = self.config.mrjd 

        print(f"\n Training slice: {len(train_spread_df)} observations")

        print(" [1/5] Pair Validation")
        pair_val = validate_pair(train_spread_df['spread'])
        half_life = pair_val['half_life']
        if not np.isfinite(half_life) or half_life <= 0:
            half_life = 30.0

        print(" [2/5] Jump Detection")
        jd = JumpDetector(significance_level = cfg_jd.significance_level)
        train_returns = train_spread_df['spread'].pct_change().dropna()
        train_jump_df = jd.detect_jumps_bipower_variation(
            train_returns, window = cfg_jd.window_size
        )
        jump_times_numeric = jd.extract_jump_times(train_jump_df)

        print(" [3/5] Hawkes Calibration")
        hawkes = HawkesProcess()
        if isinstance(train_jump_df.index, pd.DatetimeIndex):
            T = (train_jump_df.index[-1] - train_jump_df.index[0]).total_seconds() / (24 * 3600)
        else:
            T = float(len(train_jump_df) - 1)
        T = max(T, 1.0)

        if len(jump_times_numeric) >= 5:
            hawkes_params = hawkes.fit(jump_times_numeric, T)
            train_intensity = hawkes.compute_intensity_at_dates(
                jump_times_numeric, pd.DatetimeIndex(train_spread_df.index), T
            )
        else:
            hawkes_params = {'lambda_bar': 0.01, 'alpha': 0.1, 'beta': 1.0, 'beta_H': 1.0}
            hawkes.params = hawkes_params.copy()
            train_intensity = pd.Series(hawkes_params['lambda_bar'], index = train_spread_df.index)

        print(" [4/5] Hawkes Suitability")
        hawkes_suit = analyze_hawkes_suitability(train_jump_df, hawkes_params, train_intensity)
        lambda_pcts = {
            'p25': float(train_intensity.quantile(0.25)),
            'p50': float(train_intensity.quantile(0.50)),
            'p75': float(train_intensity.quantile(0.75)),
            'p90': float(train_intensity.quantile(0.90)),
            'p95': float(train_intensity.quantile(0.95)),
        }

        print(" [5/5] MRJD estimation")
        mrjd = MRJDModel()
        common_idx = train_spread_df.index.intersection(train_jump_df.index)
        train_spread_s = train_spread_df.loc[common_idx, 'spread']
        jump_ind_train = pd.Series(0, index = train_spread_df.index)
        jt_dates = train_jump_df[train_jump_df['jump_indicator'] == 1].index
        jump_ind_train.loc[train_spread_df.index.intersection(jt_dates)] = 1


        mrjd_params = mrjd.fit(
            train_spread_s,
            jump_ind_train.loc[common_idx],
            dt = cfg_mrjd.dt,
            method = cfg_mrjd.estimation_method
        )

        #half-life correction (mirror of main pipeline)
        mrjd_hl = np.log(2) / mrjd_params['kappa'] if mrjd_params['kappa'] > 0 else np.inf
        if np.isfinite(mrjd_hl) and np.isfinite(half_life) and half_life > 0:
            if abs(mrjd_hl - half_life) > half_life * 0.5:
                mrjd_params['kappa'] = np.log(2) / half_life 

        return ModelBundle(
            hawkes_params = hawkes_params,
            mrjd_params = mrjd_params,
            half_life = half_life,
            lambda_percentiles = lambda_pcts,
            hawkes_suitability = hawkes_suit,
            pair_validation = pair_val,
            use_hawkes_regimes = hawkes_suit.get('use_hawkes_regimes', False),
        )
    
    def _compute_full_sample_artifacts(self, full_spread_df: pd.DataFrame, bundle: ModelBundle) -> Dict:
        """
        Compute jump_df, Hawkes intensity, and z-score on the FULL sample using
        FROZEN parameters from a training bundle
        """
        cfg_jd = self.config.jump_detection
        cfg_t = self.config.trading 

        jd = JumpDetector(significance_level = cfg_jd.significance_level)
        full_returns = full_spread_df['spread'].pct_change().dropna() 
        full_jump_df = jd.detect_jumps_bipower_variation(
            full_returns, window = cfg_jd.window_size
        )
        jump_times_numeric = jd.extract_jump_times(full_jump_df)
        hawkes = HawkesProcess()
        hawkes.params = bundle.hawkes_params.copy() 

        if len(jump_times_numeric) >= 1:
            full_intensity = hawkes.compute_intensity_at_dates(
                jump_times_numeric,
                pd.DatetimeIndex(full_spread_df.index),
                float(len(full_spread_df))
            )
        else:
            full_intensity = pd.Series(
                bundle.hawkes_params.get('lambda_bar', 0.01),
                index = full_spread_df.index
            )

        lookback = cfg_t.z_score_lookback
        rm = full_spread_df['spread'].rolling(window = lookback, min_periods = 20).mean() 
        rs = full_spread_df['spread'].rolling(window = lookback, min_periods = 20).std()
        full_z = ((full_spread_df['spread'] - rm) / rs).fillna(0)

        return {
            'jump_df': full_jump_df,
            'hawkes_intensity': full_intensity,
            'z_score': full_z,
        }
    
    def _evaluate_period(self,
                         full_spread_df: pd.DataFrame,
                         full_cleaned_data: Dict,
                         artifacts: Dict,
                         bundle: ModelBundle,
                         start: str, 
                         end: str,
                         label: str) -> Dict:
        """
        Generate signals and run a backtest on [start,end] using frozen parameters from 'bundle'
        """
        cfg_t = self.config.trading
        cfg_bt = self.config.backtest 

        print(f"\n Evaluating period: {label}")

        period_spread = full_spread_df.loc[start:end]
        period_cleaned = {
            'asset_a': full_cleaned_data['asset_a'].loc[start:end],
            'asset_b': full_cleaned_data['asset_b'].loc[start:end],
        }

        if len(period_spread) == 0:
            print(f" Warning: No data in window {start} to {end}")
            return {}
        
        jdf = artifacts['jump_df']
        period_jump_df = jdf.loc[start:end] if not jdf.empty else jdf
        period_intensity = artifacts['hawkes_intensity'].loc[start:end]
        period_z = artifacts['z_score'].loc[start:end]

        jump_indicator = pd.Series(0, index = period_spread.index)
        if not period_jump_df.empty:
            jt_dates = period_jump_df[period_jump_df['jump_indicator'] == 1].index
            common_jt = jump_indicator.index.intersection(jt_dates)
            jump_indicator.loc[common_jt] = 1

        sig_gen = TradingSignals(
            z_entry_threshold = cfg_t.z_entry_threshold,
            z_exit_threshold = cfg_t.z_exit_threshold,
            lambda_threshold = cfg_t.lambda_threshold,
            scaling_constant = cfg_t.scaling_constant,
            max_position_size = 0.25,
            use_jump_entries = cfg_t.use_jump_entries,
            use_hawkes_regimes = bundle.use_hawkes_regimes,
            z_lookback = cfg_t.z_score_lookback,
            max_holding_period = cfg_t.max_holding_period,
        )

        sig_gen.set_lambda_percentiles(bundle.lambda_percentiles)
        sig_gen.set_half_life(bundle.half_life)

        signals_df = sig_gen.generate_signals(
            spread = period_spread['spread'],
            lambda_intensity = period_intensity,
            jump_indicator = jump_indicator,
            z_score = period_z,
        )

        engine = BacktestEngineV2(
            initial_capital = cfg_bt.initial_capital,
            commission_rate = cfg_bt.commission_rate,
            slippage_bps= cfg_bt.slippage_bps,
            max_position_pct = 0.25,
            stop_loss_pct = 0.03,
            trailing_stop_pct = 0.015,
            trailing_activation_pct = 0.01,
            profit_target_pct = 0.06,
        )
        engine.set_half_life(bundle.half_life)

        hedge_ratio = (
            period_spread['hedge_ratio'].mean() 
            if 'hedge_ratio' in period_spread.columns else 1.0
        )
        equity_curve = engine.run_backtest(
            signals_df, period_spread,
            period_cleaned['asset_a']['Close'],
            period_cleaned['asset_b']['Close'],
            hedge_ratio = hedge_ratio,
        )

        metrics = engine.calculate_performance_metrics(risk_free_rate = cfg_bt.risk_free_rate)

        alpha_stats = compute_alpha_tstat(equity_curve, cfg_bt.benchmark_csv, cfg_bt.risk_free_rate)
        metrics.update(alpha_stats)

        return {
            'label': label,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'engine': engine,
            'signals_df': signals_df,
        }
    
    def _print_train_val_comparison(self, train_res: Dict, val_res: Dict):
        """ Print side-by-side in-sample vs out-of-sample performance"""
        tv = self.config.train_val 
        tm = train_res.get('metrics', {})
        vm = val_res.get('metrics', {})

        h_train = f"{tv.train_start[:10]} to {tv.train_end[:10]}"
        h_val = f"{tv.val_start[:10]} to {tv.val_end[:10]}"

        print("\n" + "="*80)
        print("TRAIN vs VALIDATION PERFORMANCE COMPARISON")
        print("=" * 80)
        print(f"\n{'Metric':<30} {'In-Sample':>22} {'Out-of-Sample':>22}")
        print(f"{'Period':<30} {h_train:>22} {h_val:>22}")
        print("-" * 80)

        def _fmt(val, spec):
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                return f"{'N/A':>22}"
            try:
                return f"{val:{spec}}"
            except (TypeError, ValueError):
                return f"{'N/A':>22}"
        
        rows = [
            ("Total Return (%)", "total_return_pct", ">22.2f"),
            ("Annualized Return (%)", "annualized_return_pct", ">22.2f"),
            ("Annualized Vol (%)", "annualized_volatility_pct", ">22.2f"),
            ("Sharpe Ratio", "sharpe_ratio", ">22.3f"),
            ("Sortino Ratio", "sortino_ratio", ">22.3f"),
            ("Max Drawdown (%)", "max_drawdown_pct", ">22.2f"),
            ("Calmar Ratio", "calmar_ratio", ">22.3f"),
            ("Alpha Annualized (%)", "alpha_annualized", ">22.4"),
            ("Alpha T-stat", "alpha_tstat", ">22.3f"),
            ("Alpha P-value", "alpha_pvalue", ">22.4f"),
            ("Beta vs SPY", "beta", ">22.3f"),
            ("R-Squared", "r_squared", ">22.4f"),
            ("Info Ratio", "information_ratio", ">22.3f"),
            ("Total Trades", "total_trades", ">22"),
            ("Win Rate (%)", "win_rate_pct", ">22.2f"),
            ("Profit Factor", "profit_factor", ">22.3f"),
        ]

        for label, key, spec in rows:
            tv_ = tm.get(key)
            vv_ = vm.get(key)
            if key == 'alpha_annualized':
                tv_ = tv_ * 100 if tv_ is not None else None 
                vv_ = vv_ * 100 if vv_ is not None else None 
            print(f" {label:<28} {_fmt(tv_, spec)} {_fmt(vv_, spec)}")

        print("=" * 76)

        t_sharpe = tm.get('sharpe_ratio', 0.0) or 0.0
        v_sharpe = vm.get('sharpe_ratio', 0.0) or 0.0
        if t_sharpe != 0:
            deg = (t_sharpe - v_sharpe) / abs(t_sharpe)
            tag = "WARNING: large Sharpe degradation" if deg > 0.5 else "Sharpe degradation"
            print(f"\n {tag} (train -> val): {deg:+.1%}")
        print("=" * 80)

    def run_train_val_pipeline(self) -> Dict:
        """ Full pipeline with proper in-sample / out-of-sample split"""
        tv = self.config.train_val 

        print("\n" + "="*80)
        print("TRAIN / VALIDATION PIPELINE")
        print(f" Train: {tv.train_start} -> {tv.train_end}")
        print(f" Val: {tv.val_start} -> {tv.val_end}")
        print("=" * 80)

        print("\n [Step 1] Data Acquisition (Full Sample)")
        self._acquire_data()
        if self.spread_df is None or self.cleaned_data is None:
            raise ValueError("Train/validation pipeline requires acquired spread and cleaned data")
        full_spread = self.spread_df
        full_cleaned = self.cleaned_data 

        train_spread = full_spread.loc[tv.train_start:tv.train_end]
        train_cleaned = {
            'asset_a': full_cleaned['asset_a'].loc[tv.train_start:tv.train_end],
            'asset_b': full_cleaned['asset_b'].loc[tv.train_start:tv.train_end],
        }
        n_val = len(full_spread.loc[tv.val_start:tv.val_end])
        print(f"\n Training obs: {len(train_spread)}")
        print(f" Validation obs: {n_val}")

        print("\n [Steps 2-6] Fitting Models on Training Data Only")
        bundle = self._fit_models(train_spread, train_cleaned)

        self.half_life = bundle.half_life 
        self.pair_validation = bundle.pair_validation
        self.hawkes_suitability = bundle.hawkes_suitability

        print("\n [Step 7] Full Sample Causal Artifacts (Frozen Params)")
        artifacts = self._compute_full_sample_artifacts(full_spread, bundle)

        print("\n [Step 8a] IN-SAMPLE Evaluation")
        print("-" * 70)
        train_res = self._evaluate_period(
            full_spread, full_cleaned, artifacts, bundle,
            tv.train_start, tv.train_end, "In-Sample (Train)"
        )

        print("\n [Step 8b] OUT-OF-SAMPLE Evaluation")
        print("-" * 70)
        val_res = self._evaluate_period(
            full_spread, full_cleaned, artifacts, bundle,
            tv.val_start, tv.val_end, "Out-of-Sample (Val)"
        )

        self._print_train_val_comparison(train_res, val_res)

        self.results['train_results'] = train_res.get('metrics', {})
        self.results['val_results'] = val_res.get('metrics', {})
        self.results['bundle'] = bundle

        return {'bundle': bundle, 'train': train_res, 'val': val_res}

    def _print_summary(self):
        """Print comprehensive summary"""
        if self.backtest_engine is None:
            raise ValueError("Summary requires a completed backtest")

        backtest_engine = self.backtest_engine

        print("\n" + "=" * 70)
        print("Performance Summary")
        print("=" * 70)

        perf = self.results['performance']

        print(f"\n Returns")
        print(f"    Total Return: {perf['total_return_pct']:.2f}%")
        print(f"    Annualized Return: {perf['annualized_return_pct']:.2f}%")
        print(f"    Volatility (annualized): {perf['annualized_volatility_pct']:.2f}%")

        print(f"\n Risk-Adjusted Metrics:")
        print(f"    Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
        print(f"    Sortino Ratio: {perf['sortino_ratio']:.3f}")
        print(f"    Max Drawdown: {perf['max_drawdown_pct']:.2f}%")

        print(f"\n Alpha vs Benchmark (SPY):")
        print(f" Alpha (annaulized): {perf.get('alpha_annualized', 0) * 100:.3f}%")
        print(f" Alpha T-Stat: {perf.get('alpha_tstat', 0):.3f}")
        print(f" Alpha P-Value: {perf.get('alpha_pvalue', 1):.4f}")
        print(f" Beta: {perf.get('beta', 0):.3f}")
        print(f" R-squared: {perf.get('r_squared', 0):.4f}")
        print(f" Information Ratio: {perf.get('information_ratio', 0):.3f}")
        
        print(f"\n Trading Activity:")
        print(f"    Total Trades: {perf['total_trades']}")
        print(f"    Win Rate: {perf['win_rate_pct']:.2f}%")
        print(f"    Profit Factor: {perf['profit_factor']:.2f}")
        print(f"    Avg Trade Duration: {perf['avg_trade_duration_days']:.1f} days")
        if 'expected_value_per_trade' in perf:
            print(f"    Expected Value/Trade: {perf['expected_value_per_trade']:.3f}%")
        
        print(f"\n Key Optimizations Applied:")
        print(f"    Dollar-neutral positions")
        print(f"    Half-life aware exits ({self.half_life:.1f}d)")
        print(f"    Trailing stop losses (1.5%)")
        print(f"    Hawkes regime detection: {self.hawkes_suitability.get('use_hawkes_regimes', False)}")
        
        # Exit reason analysis
        exit_analysis = backtest_engine.analyze_by_exit_reason()
        if exit_analysis:
            print(f"\n Exit Reason Breakdown:")
            for reason, stats in exit_analysis.items():
                print(f"    {reason}: {stats['count']} trades, "
                      f"{stats['win_rate']:.1f}% win rate")
        
        # Hawkes suitability
        hawkes_score = self.hawkes_suitability.get('overall_score', 0)
        print(f"\n Hawkes Utilization:")
        print(f"    Suitability Score: {hawkes_score:.2f}/1.00")
        print(f"    Recommendation: {self.hawkes_suitability.get('recommendation', 'N/A')}")
        
        # Pair quality warning
        if not self.pair_validation.get('is_tradeable', True):
            print(f"\n PAIR QUALITY WARNING:")
            print(f"    This pair has structural issues.")

    def _save_jump_visualization(self, output_dir: str) -> None:
        """Save jump detection visualization to output directory."""
        if self.spread_df is None or self.jump_df is None:
            return

        import matplotlib.pyplot as plt

        spread_df = self.spread_df
        jump_df = self.jump_df
        vis_cfg = self.config.visualization
        plot_format = vis_cfg.plot_format
        dpi = vis_cfg.dpi
        figsize = (vis_cfg.figure_size[0], vis_cfg.figure_size[1] + 2)

        fig, axes = plt.subplots(2, 1, figsize=figsize)

        axes[0].plot(spread_df.index, spread_df['spread'], color='blue', linewidth=0.9, alpha=0.8)
        jump_times = jump_df.index[jump_df['jump_indicator'] == 1]
        if len(jump_times) > 0:
            axes[0].scatter(
                jump_times,
                spread_df.loc[jump_times, 'spread'],
                color='red',
                s=35,
                marker='x',
                label=f'Jumps (n={len(jump_times)})',
                zorder=5
            )
            axes[0].legend(loc='upper right')
        axes[0].set_ylabel('Spread')
        axes[0].set_title('Spread with Detected Jumps')
        axes[0].grid(True, alpha=0.3)

        if 'z_statistic' in jump_df.columns:
            axes[1].plot(jump_df.index, jump_df['z_statistic'], color='green', linewidth=0.9)
            critical_value = float(jump_df['z_statistic'].quantile(1 - self.config.jump_detection.significance_level))
            axes[1].axhline(y=critical_value, color='red', linestyle='--', linewidth=1.0)
            axes[1].set_ylabel('Z-stat')
            axes[1].set_title('Jump Test Statistic')
        else:
            axes[1].plot(jump_df.index, jump_df['jump_indicator'].cumsum(), color='purple', linewidth=1.2)
            axes[1].set_ylabel('Cumulative Jumps')
            axes[1].set_title('Cumulative Jump Count')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        output_path = os.path.join(output_dir, f'jump_detection.{plot_format}')
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved jump visualization: {output_path}")

    def _save_hawkes_visualization(self, output_dir: str) -> None:
        """Save Hawkes intensity visualization to output directory."""
        if self.spread_df is None or self.hawkes_intensity is None:
            return

        import matplotlib.pyplot as plt

        spread_df = self.spread_df
        hawkes_intensity = self.hawkes_intensity
        vis_cfg = self.config.visualization
        plot_format = vis_cfg.plot_format
        dpi = vis_cfg.dpi
        figsize = (vis_cfg.figure_size[0], vis_cfg.figure_size[1] + 2)

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        axes[0].plot(spread_df.index, spread_df['spread'], color='blue', linewidth=0.9, alpha=0.8)
        if self.jump_df is not None:
            jump_times = self.jump_df.index[self.jump_df['jump_indicator'] == 1]
            if len(jump_times) > 0:
                axes[0].scatter(
                    jump_times,
                    spread_df.loc[jump_times, 'spread'],
                    color='red',
                    s=28,
                    marker='x',
                    zorder=5
                )
        axes[0].set_ylabel('Spread')
        axes[0].set_title('Spread and Jump Events')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(hawkes_intensity.index, hawkes_intensity.values, color='orange', linewidth=1.1)
        axes[1].axhline(
            y=self.config.trading.lambda_threshold,
            color='red',
            linestyle='--',
            linewidth=1.0,
            label=f"lambda_threshold={self.config.trading.lambda_threshold:.3f}"
        )
        axes[1].set_ylabel('Intensity λ(t)')
        axes[1].set_xlabel('Date')
        axes[1].set_title('Hawkes Intensity Path')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        output_path = os.path.join(output_dir, f'hawkes_intensity.{plot_format}')
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Hawkes visualization: {output_path}")
        
    def save_results(self, output_dir: Optional[str] = None):
        """Save results"""
        if self.backtest_engine is None:
            raise ValueError("Saving results requires a completed backtest")

        backtest_engine = self.backtest_engine

        if output_dir is None:
            output_dir = os.path.join(project_root, 'outputs')

        print(f"\nSaving results to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([self.results['performance']])
        metrics_df.to_csv(f'{output_dir}/performance_metrics.csv', index=False)
        
        # Save trades
        trade_summary = backtest_engine.get_trade_summary()
        if not trade_summary.empty:
            trade_summary.to_csv(f'{output_dir}/trade_summary.csv', index=False)
        
        # Save validation
        validation_df = pd.DataFrame([self.pair_validation])
        validation_df.to_csv(f'{output_dir}/pair_validation.csv', index=False)
        
        # Save Hawkes suitability
        hawkes_df = pd.DataFrame([self.hawkes_suitability])
        hawkes_df.to_csv(f'{output_dir}/hawkes_suitability.csv', index=False)

        # Save pipeline visualizations to the same output folder
        self._save_jump_visualization(output_dir)
        self._save_hawkes_visualization(output_dir)
        
        print("Results saved successfully")


def main():
    """Main execution"""
    
    config = ConfigV2()
    
    # Trading thresholds (tuned on training data; frozen for val/walk-forward)
    config.trading.z_entry_threshold = 2.0    # Increased from 1.5
    config.trading.z_exit_threshold = 0.5     # Increased from 0.3
    config.trading.lambda_threshold = 0.15    # Use 85th percentile
    
    system = SelfExcitingPairsTradingV4(config)
    results = system.run_train_val_pipeline()


    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    
    return system, results


if __name__ == "__main__":
    try:
        system, results = run_with_output_capture("main_output.txt", main)
    except Exception:
        sys.exit(1)
