"""
Improved Main Integration Script for Self-Exciting Pairs Trading V3

Key fixes from V2:
1. Uses dollar-neutral backtest engine
2. Adds pair validation (checks if pair is actually cointegrated)
3. Better stop losses
4. Validates signal direction
"""

import sys
import os
from typing import Optional
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import modules
from equity_pairs_loader import EquityPairsDataPipeline
from jump_detector import JumpDetector
from hawkes_calibration import HawkesProcess
from mrjd_estimation import MRJDModel
from config import ConfigV2, diagnose_signal_generation, print_diagnostics
from signal_generation import TradingSignals
from backtest_engine import BacktestEngineV2  # Use corrected engine


def validate_pair(spread: pd.Series, min_half_life: int = 5, max_half_life: int = 60) -> dict:
    """
    Validate if the pair is suitable for mean-reversion trading
    
    Returns dict with validation results and recommendations
    """
    from statsmodels.tsa.stattools import adfuller
    
    print("\n" + "="*70)
    print("PAIR VALIDATION")
    print("="*70)
    
    # 1. Check stationarity (ADF test)
    adf_result = adfuller(spread.dropna())
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]
    is_stationary = adf_pvalue < 0.05
    
    print(f"\n1. Stationarity Test (ADF):")
    print(f"   ADF Statistic: {adf_stat:.4f}")
    print(f"   P-value: {adf_pvalue:.4f}")
    print(f"   Stationary: {'✓ YES' if is_stationary else '✗ NO'}")
    
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
    print(f"   Within range: {'✓ YES' if half_life_ok else '✗ NO'}")
    
    # 3. Check for structural breaks (rolling mean stability)
    rolling_mean_60 = spread.rolling(60).mean()
    rolling_mean_252 = spread.rolling(252).mean()
    
    # Check if long-term mean drifts significantly
    mean_drift = rolling_mean_252.std() / spread.std()
    stable_mean = mean_drift < 0.5  # Long-term mean shouldn't vary too much
    
    print(f"\n3. Mean Stability Check:")
    print(f"   Long-term mean drift: {mean_drift:.4f}")
    print(f"   Stable mean: {'✓ YES' if stable_mean else '✗ NO (structural break likely)'}")
    
    # 4. Check spread range vs standard deviation
    spread_range = spread.max() - spread.min()
    range_in_std = spread_range / spread.std()
    
    print(f"\n4. Spread Range Analysis:")
    print(f"   Range: {spread_range:.4f}")
    print(f"   Standard deviation: {spread.std():.4f}")
    print(f"   Range in std units: {range_in_std:.1f}")
    
    reasonable_range = range_in_std < 10  # Shouldn't exceed 10 std
    print(f"   Reasonable range (<10 std): {'✓ YES' if reasonable_range else '✗ NO (extreme moves)'}")
    
    # 5. Recent performance check
    recent_252 = spread.iloc[-252:] if len(spread) > 252 else spread
    recent_mean = recent_252.mean()
    overall_mean = spread.mean()
    mean_shift = abs(recent_mean - overall_mean) / spread.std()
    
    print(f"\n5. Recent vs Historical Mean:")
    print(f"   Overall mean: {overall_mean:.4f}")
    print(f"   Recent 1yr mean: {recent_mean:.4f}")
    print(f"   Shift in std: {mean_shift:.2f}")
    
    no_regime_change = mean_shift < 1.0
    print(f"   No recent regime change: {'✓ YES' if no_regime_change else '✗ NO'}")
    
    # Overall assessment
    is_tradeable = is_stationary and half_life_ok and stable_mean and reasonable_range
    
    print("\n" + "-"*70)
    print("OVERALL ASSESSMENT:")
    if is_tradeable:
        print("✓ PAIR IS SUITABLE FOR MEAN-REVERSION TRADING")
    else:
        print("⚠ PAIR HAS ISSUES - TRADING NOT RECOMMENDED")
        print("\nIssues detected:")
        if not is_stationary:
            print("  - Spread is not stationary (not cointegrated)")
        if not half_life_ok:
            print(f"  - Half-life ({half_life:.1f}d) outside acceptable range")
        if not stable_mean:
            print("  - Long-term mean is unstable (structural breaks)")
        if not reasonable_range:
            print(f"  - Spread has extreme moves ({range_in_std:.1f} std)")
        if not no_regime_change:
            print("  - Recent regime change detected")
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


class SelfExcitingPairsTradingV3:
    """Self-exciting pairs trading system with corrected backtest"""

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

        # Results
        self.results = {}
        self.diagnostics = {}
        self.pair_validation = {}

    def run_complete_pipeline(self, skip_validation: bool = False):
        """Execute complete trading pipeline with validation"""

        print("\n" + "=" * 70)
        print("Self-Exciting Pairs Trading System V3")
        print("NVDA/AMD Equity Pairs Strategy (Dollar-Neutral)")
        print("=" * 70)

        # Step 1: Data Acquisition
        print("\n [Step 1/8] Data Acquisition")
        print("-" * 70)
        self._acquire_data()

        if self.spread_df is None:
            raise ValueError("Pair validation requires spread_df")
        spread_df = self.spread_df

        # Step 2: Pair Validation (NEW)
        print("\n [Step 2/8] Pair Validation")
        print("-" * 70)
        self.pair_validation = validate_pair(spread_df['spread'])
        
        if not self.pair_validation['is_tradeable'] and not skip_validation:
            print("\n⚠ WARNING: Pair validation failed!")
            print("Continuing anyway for educational purposes...")
            print("In production, you should choose a different pair.\n")

        # Step 3: Jump Detection
        print("\n [Step 3/8] Jump Detection")
        print("-" * 70)
        self._detect_jumps() 

        # Step 4: Hawkes Calibration
        print("\n [Step 4/8] Hawkes Process Calibration")
        print("-" * 70)
        self._calibrate_hawkes()

        # Step 5: MRJD Estimation
        print("\n [Step 5/8] MRJD Model Estimation")
        print("-" * 70)
        self._estimate_mrjd() 

        # Step 6: Diagnostics
        print("\n [Step 6/8] Signal Generation Diagnostics")
        print("-" * 70)
        self._run_diagnostics()

        # Step 7: Signal Generation
        print("\n [Step 7/8] Trading Signal Generation")
        print("-" * 70)
        self._generate_signals() 

        # Step 8: Backtesting (with corrected engine)
        print("\n [Step 8/8] Backtesting (Dollar-Neutral)")
        print("-" * 70)
        self._run_backtest() 

        # Summary
        print("\n" + "=" * 70)
        print("Pipeline Complete - Generating Summary")
        print("=" * 70)
        self._print_summary() 

        return self.results 
    
    def _acquire_data(self):
        """Step 1: Acquire data"""
        cfg = self.config.data

        self.data_pipeline = EquityPairsDataPipeline(
            asset_a_path=cfg.asset_a_csv,
            asset_b_path=cfg.asset_b_csv
        )

        data = self.data_pipeline.load_from_csv(
            asset_a_path=cfg.asset_a_csv,
            asset_b_path=cfg.asset_b_csv,
            date_columns=cfg.date_columns
        )

        self.cleaned_data = self.data_pipeline.clean_data() 
        self.spread_df = self.data_pipeline.construct_spread(
            method=cfg.hedge_ratio_method,
            lookback=cfg.lookback_period
        )

        stats = self.data_pipeline.calculate_spread_statistics(
            self.spread_df['spread']
        )

        print(f"\n Spread Statistics (NVDA/AMD)")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std: {stats['std']:.4f}")
        print(f"    Half-life: {stats['half_life']:.2f} days")
        print(f"    Stationary: {stats['is_stationary']}")

        self.results['spread_stats'] = stats 

    def _detect_jumps(self):
        """Step 3: Detect jumps"""
        cfg = self.config.jump_detection 

        if self.spread_df is None:
            raise ValueError("Jump detection requires spread_df")
        spread_df = self.spread_df

        self.jump_detector = JumpDetector(significance_level=cfg.significance_level)
        returns = spread_df['spread'].pct_change().dropna() 

        print(f"Using window size: {cfg.window_size} periods")
        self.jump_df = self.jump_detector.detect_jumps_bipower_variation(returns, window=cfg.window_size)

        jump_stats = self.jump_detector.calculate_jump_statistics(self.jump_df)

        print(f"\n Jump Statistics")
        print(f"    Total Jumps: {jump_stats['n_jumps']}")
        print(f"    Jump Frequency: {jump_stats['jump_frequency']:.4f}")
        print(f"    Clustering coefficient: {jump_stats['clustering_coefficient']:.4f}")

        self.results['jump_stats'] = jump_stats

    def _calibrate_hawkes(self):
        """Step 4: Calibrate Hawkes"""
        cfg = self.config.hawkes

        if self.jump_detector is None or self.jump_df is None or self.spread_df is None:
            raise ValueError("Hawkes calibration requires jump_detector, jump_df, and spread_df")

        jump_detector = self.jump_detector
        jump_df = self.jump_df
        spread_df = self.spread_df

        self.hawkes_model = HawkesProcess(kernel=cfg.kernel)
        jump_times = jump_detector.extract_jump_times(jump_df)
        
        if isinstance(jump_df.index, pd.DatetimeIndex):
            T = (jump_df.index[-1] - jump_df.index[0]).total_seconds() / (24 * 3600)
        else:
            T = float(len(jump_df) - 1)
        T = max(T, 1.0)

        if len(jump_times) < 2:
            print("Warning: Insufficient jumps for Hawkes calibration")
            self.hawkes_params = {'lambda_bar': 0.01, 'alpha': 0.1, 'beta': 1.0}
        else:
            self.hawkes_params = self.hawkes_model.fit(jump_times, T, method=cfg.estimation_method)

        # Compute intensity
        spread_dates = pd.DatetimeIndex(spread_df.index)
        
        if len(jump_times) >= 2:
            self.hawkes_intensity = self.hawkes_model.compute_intensity_at_dates(jump_times, spread_dates, T)
        else:
            self.hawkes_intensity = pd.Series(
                self.hawkes_params['lambda_bar'] * np.ones(len(spread_df)),
                index=spread_df.index
            )

        self.results['hawkes_params'] = self.hawkes_params

        print(f"\n Jump Intensity Statistics:")
        print(f"    Mean λ: {self.hawkes_intensity.mean():.6f}")
        print(f"    90th percentile: {self.hawkes_intensity.quantile(0.9):.6f}")

    def _estimate_mrjd(self):
        """Step 5: MRJD estimation"""
        cfg = self.config.mrjd

        if self.spread_df is None or self.jump_df is None:
            raise ValueError("MRJD estimation requires spread_df and jump_df")

        spread_df = self.spread_df
        jump_df = self.jump_df

        self.mrjd_model = MRJDModel() 

        common_index = spread_df.index.intersection(jump_df.index)
        spread = spread_df.loc[common_index, 'spread']
        jump_indicator = jump_df.loc[common_index, 'jump_indicator']

        self.mrjd_params = self.mrjd_model.fit(spread, jump_indicator, dt=cfg.dt, method=cfg.estimation_method)
        self.z_score_mrjd = self.mrjd_model.calculate_z_score(spread_df['spread'])
        self.results['mrjd_params'] = self.mrjd_params

    def _run_diagnostics(self):
        """Step 6: Diagnostics"""

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
        
        print("\n Z-Score Comparison:")
        print(f"    Empirical Z-score range: [{self.z_score.min():.2f}, {self.z_score.max():.2f}]")
        print(f"    Empirical Z-score std: {self.z_score.std():.2f}")
        
        # Run diagnostics
        self.diagnostics = diagnose_signal_generation(
            spread_df['spread'],
            hawkes_intensity,
            self.z_score,
            self.config.trading
        )
        
        print_diagnostics(self.diagnostics)

    def _generate_signals(self):
        """Step 7: Generate signals"""
        cfg = self.config.trading

        if self.spread_df is None or self.jump_df is None or self.hawkes_intensity is None or self.z_score is None:
            raise ValueError("Signal generation requires spread_df, jump_df, hawkes_intensity, and z_score")

        spread_df = self.spread_df
        jump_df = self.jump_df
        hawkes_intensity = self.hawkes_intensity
        z_score = self.z_score

        # Create jump indicator
        jump_indicator = pd.Series(0, index=spread_df.index)
        jump_times = jump_df[jump_df['jump_indicator'] == 1].index
        common_jumps = jump_indicator.index.intersection(jump_times)
        jump_indicator.loc[common_jumps] = 1

        # Use signal generator
        self.signal_generator = TradingSignals(
            z_entry_threshold=cfg.z_entry_threshold,
            z_exit_threshold=cfg.z_exit_threshold,
            lambda_threshold=cfg.lambda_threshold,
            scaling_constant=cfg.scaling_constant,
            max_position_size=cfg.max_position_size,
            use_jump_entries=cfg.use_jump_entries,
            z_lookback=cfg.z_score_lookback,
            max_holding_period=cfg.max_holding_period
        )

        self.signals_df = self.signal_generator.generate_signals(
            spread=spread_df['spread'],
            lambda_intensity=hawkes_intensity,
            jump_indicator=jump_indicator,
            z_score=z_score
        )

        signal_quality = self.signal_generator.calculate_signal_quality(self.signals_df)

        print(f"\n Signal Quality:")
        print(f"    Time in Market: {signal_quality['time_in_market']:.2%}")
        print(f"    Entry Discipline: {signal_quality['entry_discipline']:.2%}")

        self.results['signal_quality'] = signal_quality

    def _run_backtest(self):
        """Step 8: Backtest with CORRECTED engine"""
        cfg = self.config.backtest 

        if self.cleaned_data is None or self.spread_df is None or self.signals_df is None:
            raise ValueError("Backtest requires cleaned_data, spread_df, and signals_df")

        cleaned_data = self.cleaned_data
        spread_df = self.spread_df
        signals_df = self.signals_df

        # Use the CORRECTED backtest engine
        self.backtest_engine = BacktestEngineV2(
            initial_capital=cfg.initial_capital,
            commission_rate=cfg.commission_rate,
            slippage_bps=cfg.slippage_bps,
            max_position_pct=0.25,    # Conservative: 25% max per trade
            stop_loss_pct=0.05,       # 5% stop loss
            profit_target_pct=0.08    # 8% profit target
        )

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

        regime_perf = self.backtest_engine.analyze_regime_performance(regime_threshold=0.1)

        self.results['performance'] = self.performance_metrics 
        self.results['regime_performance'] = regime_perf 

    def _print_summary(self):
        """Print summary"""

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
        
        print(f"\n Trading Activity:")
        print(f"    Total Trades: {perf['total_trades']}")
        print(f"    Win Rate: {perf['win_rate_pct']:.2f}%")
        print(f"    Profit Factor: {perf['profit_factor']:.2f}")
        print(f"    Avg Trade Duration: {perf['avg_trade_duration_days']:.1f} days")
        
        print(f"\n Key Fixes Applied:")
        print(f"    ✓ Dollar-neutral positions")
        print(f"    ✓ P&L-based stop loss (5%)")
        print(f"    ✓ Conservative position sizing (25% max)")
        
        # Pair quality warning
        if not self.pair_validation.get('is_tradeable', True):
            print(f"\n ⚠ PAIR QUALITY WARNING:")
            print(f"    This pair has structural issues.")
            print(f"    Consider testing with XOM/CVX or other stable pairs.")
        
    def save_results(self, output_dir: Optional[str] = None):
        """Save results"""
        if output_dir is None:
            output_dir = os.path.join(project_root, 'outputs_v3')

        if self.backtest_engine is None:
            raise ValueError("Backtest engine is not available. Run the pipeline before saving.")

        backtest_engine = self.backtest_engine
            
        print(f"\nSaving results to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([self.results['performance']])
        metrics_df.to_csv(f'{output_dir}/performance_metrics_v3.csv', index=False)
        
        # Save trades
        trade_summary = backtest_engine.get_trade_summary()
        if not trade_summary.empty:
            trade_summary.to_csv(f'{output_dir}/trade_summary_v3.csv', index=False)
        
        # Save pair validation
        validation_df = pd.DataFrame([self.pair_validation])
        validation_df.to_csv(f'{output_dir}/pair_validation.csv', index=False)
        
        print("✓ Results saved successfully")


def main():
    """Main execution"""
    
    config = ConfigV2()
    
    # Adjust thresholds based on diagnostics from previous run
    # The recommended lambda_threshold was 0.0094, so let's use that
    config.trading.lambda_threshold = 0.1  # Use 80th percentile
    config.trading.z_entry_threshold = 1.5
    
    system = SelfExcitingPairsTradingV3(config)
    results = system.run_complete_pipeline(skip_validation=False)
    system.save_results()
    
    print("\n" + "="*70)
    print("✓ PIPELINE EXECUTION COMPLETE (V3)")
    print("="*70)
    
    return system, results


if __name__ == "__main__":
    system, results = main()
