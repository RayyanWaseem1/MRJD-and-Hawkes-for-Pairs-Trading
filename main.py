""" Main Integration Script for Self-Exciting Pairs Trading 

Pipeline:
1. Data acquisition and spread construction
2. Jump detection
3. Hawkes process calibration
4. MRJD model estimation
5. Trading signal generation
6. Backtesting and performance analysis
"""

import sys
import os
from typing import Optional
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')

#Add project root to path 
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

#import project modules
from config import Config
from equity_pairs_loader import EquityPairsDataPipeline
from jump_detector import JumpDetector
from hawkes_calibration import HawkesProcess
from mrjd_estimation import MRJDModel
from signal_generation import TradingSignals
from backtest_engine import BacktestEngine

class SelfExcitingPairsTrading:
    """ Complete the self-exciting pairs trading system"""

    def __init__(self, config: Optional[Config] = None):
        """Initialize trading system 
        
        Params:
        - config: Config
            - Configuration object
        """

        self.config = config if config is not None else Config() 

        #Components
        self.data_pipeline = None 
        self.jump_detector = None 
        self.hawkes_model = None 
        self.mrjd_model = None 
        self.signal_generator = None 
        self.backtest_engine = None 

        #Data
        self.cleaned_data = None  # Store cleaned price data
        self.spread_df = None 
        self.jump_df = None 
        self.signals_df = None 
        self.equity_curve = None 

        #Results
        self.results = {}

    def run_complete_pipeline(self):
        """ Execute complete trading pipeline"""

        print("\n" + "=" * 70)
        print("Self Exciting Pairs Trading System")
        print("XOM/CVX Equity Pairs Strategy")
        print("=" * 70)

        #Step 1: Data Acquisition
        print("\n [Step 1/6] Data Acquisition")
        print("-" * 70)
        self._acquire_data()

        #Step 2: Jump Detection
        print("\n [Step 2/6] Jump Detection")
        print("-" * 70)
        self._detect_jumps() 

        #Step 3: Hawkes Calibration
        print("\n [Step 3/6] Hawkes Process Calibration")
        print("-" * 70)
        self._calibrate_hawkes()

        #Step 4: MRJD Estimation
        print("\n [Step 4/6] MRJD Model Estimation")
        print("-" * 70)
        self._estimate_mrjd() 

        #Step 5: Signal Generation
        print("\n [Step 5/6] Trading Signal Generation")
        print("-" * 70)
        self._generate_signals() 

        #Step 6: Backtesting
        print("\n [Step 6/6] Backtesting")
        print("-" * 70)
        self._run_backtest() 

        #Summary
        print("\n" + "=" * 70)
        print("Pipeline Complete - Generating Summary")
        print("=" * 70)
        self._print_summary() 

        return self.results 
    
    def _acquire_data(self):
        """ Step 1: Acquiring and processing data from CSV files"""
        cfg = self.config.data

        self.data_pipeline = EquityPairsDataPipeline(
            asset_a_path = cfg.asset_a_csv,
            asset_b_path = cfg.asset_b_csv
        )

        #Loading data from CSV files
        data = self.data_pipeline.load_from_csv(
            asset_a_path = cfg.asset_a_csv,
            asset_b_path = cfg.asset_b_csv,
            date_columns = cfg.date_columns
        )

        #Clean data and store it
        self.cleaned_data = self.data_pipeline.clean_data() 

        #construct spread 
        self.spread_df = self.data_pipeline.construct_spread(
            method = cfg.hedge_ratio_method,
            lookback = cfg.lookback_period
        )

        #calculate spread statistics
        stats = self.data_pipeline.calculate_spread_statistics(
            self.spread_df['spread']
        )

        print(f"\n Spread Statistics (XOM/CVX)")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std: {stats['std']:.4f}")
        print(f"    Half-life: {stats['half_life']:.2f} days")
        print(f"    Stationary: {stats['is_stationary']}")

        self.results['spread_stats'] = stats 

    def _detect_jumps(self):
        """ Step 2: Detect jumps in the spread"""
        cfg = self.config.jump_detection 

        if self.spread_df is None:
            raise ValueError("Jump detection requires spread_df")

        spread_df = self.spread_df

        self.jump_detector = JumpDetector(
            significance_level = cfg.significance_level
        )

        #Calculate returns
        returns = spread_df['spread'].pct_change().dropna() 

        #Detecting jumps using bipower variation
        print(f"Using window size: {cfg.window_size} periods")
        self.jump_df = self.jump_detector.detect_jumps_bipower_variation(
            returns,
            window = cfg.window_size
        )

        #Calculate jump statistics
        jump_stats = self.jump_detector.calculate_jump_statistics(self.jump_df)

        print(f"\n Jump Statistics")
        print(f" Total Jumps: {jump_stats['n_jumps']}")
        print(f"    Jump Frequency: {jump_stats['jump_frequency']:.4f}")
        print(f"    Clustering coefficient: {jump_stats['clustering_coefficient']:.4f}")

        self.results['jump_stats'] = jump_stats

    def _calibrate_hawkes(self):
        """Step 3: Calibrate the Hawkes process"""
        cfg = self.config.hawkes

        if self.jump_detector is None or self.jump_df is None or self.spread_df is None:
            raise ValueError("Hawkes calibration requires jump_detector, jump_df, and spread_df")

        jump_detector = self.jump_detector
        jump_df = self.jump_df
        spread_df = self.spread_df
        spread_dates = pd.DatetimeIndex(spread_df.index)

        self.hawkes_model = HawkesProcess(kernel = cfg.kernel)

        #Extracting jump times
        jump_times = jump_detector.extract_jump_times(jump_df)
        if isinstance(jump_df.index, pd.DatetimeIndex):
            T = (jump_df.index[-1] - jump_df.index[0]).total_seconds() / (24 * 3600)
        else:
            index_values = jump_df.index.to_numpy()
            if np.issubdtype(index_values.dtype, np.number):
                T = float(index_values[-1] - index_values[0])
            else:
                T = float(len(jump_df) - 1)
        T = max(T, 1.0)

        if len(jump_times) < 2:
            print("Warning: Insufficient jumps for Hawkes calibration")
            print(" Using baseline model")
            self.hawkes_params = {
                'lambda_bar': 0.01,
                'alpha': 0.5,
                'beta': 2.0
            }
            #Manually set the params in the model 
            self.hawkes_model_params = self.hawkes_params 
            #Using baseline intensity
            self.hawkes_intensity = pd.Series(
                self.hawkes_params['lambda_bar'] * np.ones(len(spread_df)),
                index = spread_df.index
            )
        else:
            #Fitting hawkes process
            self.hawkes_params = self.hawkes_model.fit(
                jump_times,
                T,
                method = cfg.estimation_method
            )

            #Compute intensity at spread dates
            self.hawkes_intensity = self.hawkes_model.compute_intensity_at_dates(
                jump_times,
                spread_dates,
                T
            )

            #Goodness of fit
            gof = self.hawkes_model.goodness_of_fit(jump_times, T)
            print(f"\n Goodness of Fit: ")
            print(f" KS p-value: {gof['ks_pvalue']:.4f}")
            print(f" Good fit: {gof['is_good_fit']}")

        self.results['hawkes_params'] = self.hawkes_params

    def _estimate_mrjd(self):
        """ Step 4: Estimate MRJD Model"""
        cfg = self.config.mrjd

        if self.spread_df is None or self.jump_df is None:
            raise ValueError("MRJD estimation requires spread_df and jump_df")

        spread_df = self.spread_df
        jump_df = self.jump_df

        self.mrjd_model = MRJDModel() 

        #Align spread and jump indicator 
        common_index = spread_df.index.intersection(jump_df.index)
        spread = spread_df.loc[common_index, 'spread']
        jump_indicator = jump_df.loc[common_index, 'jump_indicator']

        #Fitting the MRJD model
        self.mrjd_params = self.mrjd_model.fit(
            spread,
            jump_indicator,
            dt = cfg.dt,
            method = cfg.estimation_method
        )

        #Calculate Z-scores
        self.z_score = self.mrjd_model.calculate_z_score(
            spread_df['spread']
        )

        #Prediction test
        current_spread = spread.iloc[-1]
        horizon = 5.0
        prediction = self.mrjd_model.predict_spread_statistics(
            current_spread,
            horizon
        )

        print(f"\n Spread Forecast ({horizon:.0f} days): ")
        print(f" Expected srpead: {prediction['expected_spread']:.4f}")
        print(f" Reversion: {prediction['reversion_pct']:.2f}%")

        self.results['mrjd_params'] = self.mrjd_params
        self.results['predictions'] = prediction 

    def _generate_signals(self):
        """ Step 5: Generate Trading Signals"""
        cfg = self.config.trading

        if self.z_score is None or self.hawkes_intensity is None or self.spread_df is None:
            raise ValueError("Signal generation requires z_score, hawkes_intensity, and spread_df")

        z_score_series = self.z_score
        hawkes_intensity_series = self.hawkes_intensity
        spread_df = self.spread_df

        self.signal_generator = TradingSignals(
            z_entry_threshold = cfg.z_entry_threshold,
            z_exit_threshold = cfg.z_exit_threshold,
            lambda_threshold = cfg.lambda_threshold,
            scaling_constant = cfg.scaling_constant,
            max_position_size = cfg.max_position_size
        )

        #Align data
        common_index = z_score_series.index.intersection(hawkes_intensity_series.index)
        z_score = z_score_series.loc[common_index]
        hawkes_intensity = hawkes_intensity_series.loc[common_index]
        spread = spread_df.loc[common_index, 'spread']

        #Generate signals
        self.signals_df = self.signal_generator.generate_signals(
            z_score,
            hawkes_intensity,
            spread
        )

        #Signal quality
        signal_quality = self.signal_generator.calculate_signal_quality(
            self.signals_df
        )

        print(f"\n Signal Quality:")
        print(f"    Time in Market: {signal_quality['time_in_market']:.2%}")
        print(f"    Entry discipline: {signal_quality['entry_discipline']:.2%}")

        self.results['signal_quality'] = signal_quality

    def _run_backtest(self):
        """ Step 6: Run Backtest"""
        cfg = self.config.backtest 

        if self.signals_df is None or self.spread_df is None:
            raise ValueError("Backtest requires signals_df and spread_df")
        
        if self.cleaned_data is None:
            raise ValueError("Backtest requires cleaned_data (run _acquire_data first)")

        signals_df = self.signals_df
        spread_df = self.spread_df

        self.backtest_engine = BacktestEngine(
            initial_capital = cfg.initial_capital,
            commission_rate = cfg.commission_rate,
            slippage_bps = cfg.slippage_bps
        )

        # Get asset prices from cleaned data
        asset_a_prices = self.cleaned_data['asset_a']['Close']
        asset_b_prices = self.cleaned_data['asset_b']['Close']
        
        # Get hedge ratio from spread_df
        if 'hedge_ratio' in spread_df.columns:
            hedge_ratio = spread_df['hedge_ratio'].iloc[0]
        else:
            # Fallback: try to get from spread_df or use default
            print("⚠ Warning: hedge_ratio not found in spread_df, using 1.0")
            hedge_ratio = 1.0

        # Run backtest with proper parameters
        self.equity_curve = self.backtest_engine.run_backtest(
            signals_df,
            spread_df,
            asset_a_prices,  # XOM prices
            asset_b_prices,  # CVX prices
            hedge_ratio=hedge_ratio
        )

        #Calculate performance metrics
        self.performance_metrics = self.backtest_engine.calculate_performance_metrics(
            risk_free_rate = cfg.risk_free_rate
        )

        #Regime analysis
        regime_perf = self.backtest_engine.analyze_regime_performance(
            regime_threshold = cfg.regime_threshold
        )

        self.results['performance'] = self.performance_metrics 
        self.results['regime_performance'] = regime_perf 

    def _print_summary(self):
        """ Print a comprehensive summary"""

        print("\n" + "=" * 70)
        print("Performance Summary")
        print("=" * 70)

        perf = self.results['performance']

        print(f"\n Returns")
        print(f"    Total Return: {perf['total_return_pct']:.2f}%")
        print(f"    Annualized Return: {perf['annualized_return_pct']:.2f}%")
        print(f"    Volatility (annualized): {perf['annualized_volatility_pct']:.2f}%")

        print(f"\n⚡ Risk-Adjusted Metrics:")
        print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {perf['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio: {perf['calmar_ratio']:.3f}")
        print(f"  Max Drawdown: {perf['max_drawdown_pct']:.2f}%")
        
        print(f"\n💼 Trading Activity:")
        print(f"  Total Trades: {perf['total_trades']}")
        print(f"  Win Rate: {perf['win_rate_pct']:.2f}%")
        print(f"  Profit Factor: {perf['profit_factor']:.2f}")
        print(f"  Avg Trade Duration: {perf['avg_trade_duration_days']:.1f} days")
        
        print(f"\n🎯 Risk Metrics:")
        print(f"  Avg MAE: {perf['avg_max_adverse_excursion_pct']:.2f}%")
        print(f"  Avg MFE: {perf['avg_max_favorable_excursion_pct']:.2f}%")
        
        # Model parameters
        print(f"\n🔬 Model Parameters:")
        print(f"  MRJD κ (reversion speed): {self.mrjd_params['kappa']:.4f}")
        print(f"  MRJD θ (equilibrium): {self.mrjd_params['theta']:.4f}")
        print(f"  Hawkes branching ratio: {self.hawkes_params['alpha']/self.hawkes_params['beta']:.4f}")
        
    def save_results(self, output_dir: str = os.path.join(project_root, 'outputs')):
        """
        Save all results to output directory
        
        Parameters:
        -----------
        output_dir : str
            Output directory path
        """
        print(f"\nSaving results to {output_dir}...")

        if self.backtest_engine is None or self.equity_curve is None or self.signals_df is None:
            raise ValueError("Results are incomplete. Run the full pipeline before saving.")

        backtest_engine = self.backtest_engine
        equity_curve = self.equity_curve
        signals_df = self.signals_df
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save performance metrics
        metrics_df = pd.DataFrame([self.results['performance']])
        metrics_df.to_csv(f'{output_dir}/performance_metrics.csv', index=False)
        
        # Save trade summary
        trade_summary = backtest_engine.get_trade_summary()
        if not trade_summary.empty:
            trade_summary.to_csv(f'{output_dir}/trade_summary.csv', index=False)
        
        # Save equity curve
        equity_curve.to_csv(f'{output_dir}/equity_curve.csv')
        
        # Save signals
        signals_df.to_csv(f'{output_dir}/trading_signals.csv')
        
        print("✓ Results saved successfully")


def main():
    """Main execution function"""
    
    # Initialize system
    config = Config()
    system = SelfExcitingPairsTrading(config)
    
    # Run complete pipeline
    results = system.run_complete_pipeline()
    
    # Save results
    system.save_results()
    
    print("\n" + "="*70)
    print("✓ PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review performance metrics in ./outputs/")
    print("2. Analyze trade summary for insights")
    print("3. Iterate on parameters to optimize performance")
    print("4. Test on different asset pairs")
    
    return system, results


if __name__ == "__main__":
    system, results = main()