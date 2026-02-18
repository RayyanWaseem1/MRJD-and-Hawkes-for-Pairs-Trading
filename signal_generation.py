"""
Trading Signal Generation for Self-Exciting Pairs Trading

Implements jump-aware entry and exit logic:
1. Entry: |Z_t| > Z_entry AND λ_t < λ_threshold
2. Position sizing: w_t ∝ Z_t / (1 + c*λ_t)
3. Exit: |Z_t| < Z_exit OR λ_t > λ_cascade_threshold

"""

import numpy as np 
import pandas as pd
from typing import Dict, Tuple, Optional
from enum import Enum 

class SignalType(Enum):
    """ Trading signal types"""
    NO_SIGNAL = 0
    LONG = 1 #Buy spread (long front, short back)
    SHORT = -1 #Sell spread (short front, long back)
    CLOSE = 2 #Close position

class TradingSignals:
    """ Generate jump-aware trading signals for pairs trading"""

    def __init__(self,
                 z_entry_threshold: float = 2.0,
                 z_exit_threshold: float = 0.5,
                 lambda_threshold: float = 5.0,
                 scaling_constant: float = 0.1,
                 max_position_size: float = 1.0):
        """
        Initializing the signal generator

        Params:
        - z_entry_threshold: float
            - minimum z-score magnitude for entry
        - z_exit_threshold: float
            - maximum z-score magnitude for exit
        - lambda_threshold: float
            - maximum jump intensity for safe entry
        - scaling_constant: float
            - position scaling constant (c in the formula)
        - max_position_size: float
            - maximum position size
        """

        self.z_entry = z_entry_threshold
        self.z_exit = z_exit_threshold 
        self.lambda_threshold = lambda_threshold 
        self.c = scaling_constant
        self.max_position = max_position_size 

        self.signals = None 
        self.positions = None 

    def generate_signals(self,
                         z_score: pd.Series,
                         lambda_intensity: pd.Series,
                         spread: pd.Series) -> pd.DataFrame:
        """
        Generating trading signals based on z-score and jump intensity

        Entry Conditions:
        - |Z_t| > z_entry_threshold
        - λ_t < lambda_threshold (avoid cascade risk)

        Exit Conditions:
        - |Z_t| < z_exit_threshold (mean reversion complete)
        - λ_t > lambda_threshold (elevated cascade risk)

        Params:
        - z_score: pd.Series
            - Standardized spread (z-scores)
        - lambda_intensity: pd.Series
            - jump intensity from Hawkes process
        - spread: pd.Series
            - raw spread values

        Returns:
        - pd.DataFrame
            - signals with columns: signal, position, z_score, lambda, spread
        """

        print("Generating jump-aware trading signals")

        #Aligning all the series
        common_index = z_score.index.intersection(lambda_intensity.index).intersection(spread.index)
        z_score = z_score.loc[common_index]
        lambda_intensity = lambda_intensity.loc[common_index]
        spread = spread.loc[common_index]

        n = len(z_score)
        signals = np.zeros(n)
        positions = np.zeros(n)
        position_sizes = np.zeros(n)

        current_position = 0 #0 = no position, 1 = long, -1 = short

        for i in range(1, n):
            z_t = z_score.iloc[i]
            lambda_t = lambda_intensity.iloc[i]

            #Entry logic
            if current_position == 0:
                #check entry conditions
                if z_t < -self.z_entry and lambda_t < self.lambda_threshold:
                    #the spread is too negative so BUY the spread (long front, short back)
                    signals[i] = SignalType.LONG.value
                    current_position = 1

                elif z_t > self.z_entry and lambda_t < self.lambda_threshold:
                    #the spread is too positive so SELL the spraed (short front, long back)
                    signals[i] = SignalType.SHORT.value
                    current_position = -1

            #Exit logic
            elif current_position != 0:
                #Exit Conditoin 1: Mean reversion complete
                exit_mean_reversion = abs(z_t) < self.z_exit

                #Exit Condition 2: Cascade risk elevated
                exit_cascade_risk = lambda_t > self.lambda_threshold

                #Exit Condition 3: Position reversal (z-score crossed zero)
                exit_reversal = (current_position == 1 and z_t > 0) or (current_position == -1 and z_t < 0)

                if exit_mean_reversion or exit_cascade_risk or exit_reversal:
                    signals[i] = SignalType.CLOSE.value
                    current_position = 0

            #Track position
            positions[i] = current_position

            #Calculate position size (only when in position)
            if current_position != 0:
                #Dynamic position scaling: w_t ∝ Z_t / (1 + c*λ_t)
                raw_size = abs(z_t) / (1 + self.c * lambda_t)
                position_sizes[i] = min(raw_size, self.max_position) * current_position

        #Create signals dataframe
        signals_df = pd.DataFrame({
            'signal': signals,
            'position': positions,
            'position_size': position_sizes,
            'z_score': z_score,
            'lambda': lambda_intensity,
            'spread': spread
        }, index = common_index)

        #Signal statistics
        n_signals = (signals != 0).sum()
        n_long = (signals == SignalType.LONG.value).sum()
        n_short = (signals == SignalType.SHORT.value).sum()
        n_close = (signals == SignalType.CLOSE.value).sum()

        print(f"Generaged {n_signals} total signals:")
        print(f"    - Long Entries: {n_long}")
        print(f"    - Short Entries: {n_short}")
        print(f"    - Closes: {n_close}")

        self.signals = signals_df
        return signals_df
        
    def calculate_signal_quality(self, signals_df: pd.DataFrame) -> Dict:
        """
        Calculating signal quality metrics

        Params:
        - signals_df: pd.DataFrame
            - signals dataframe

        Returns:
        - dict:
            - signal quality metrics
        """

        #Entry signal analysis
        long_entries = signals_df[signals_df['signal'] == SignalType.LONG.value]
        short_entries = signals_df[signals_df['signal'] == SignalType.SHORT.value]

        #Average conditions at entry
        if len(long_entries) > 0:
            avg_z_long = long_entries['z_score'].mean()
            avg_lambda_long = long_entries['lambda'].mean()
        else:
            avg_z_long = np.nan
            avg_lambda_long = np.nan 

        if len(short_entries) > 0:
            avg_z_short = short_entries['z_score'].mean()
            avg_lambda_short = short_entries['lambda'].mean() 
        else:
            avg_z_short = np.nan
            avg_lambda_short = np.nan

        #Average position size 
        in_position = signals_df[signals_df['position'] != 0]
        avg_position_size = in_position['position_size'].abs().mean() if len(in_position) > 0 else 0

        #Time in market 
        time_in_market = (signals_df['position'] != 0).sum() / len(signals_df)

        #Entry discipline (how often do we avoid the high lambda)
        all_extreme_z = signals_df[abs(signals_df['z_score']) > self.z_entry]
        safe_entries = all_extreme_z[all_extreme_z['lambda'] < self.lambda_threshold]
        entry_discipline = len(safe_entries) / len(all_extreme_z) if len(all_extreme_z) > 0 else 0

        metrics = {
            'avg_z_score_long_entry': avg_z_long,
            'avg_z_score_short_entry': avg_z_short,
            'avg_lambda_long_entry': avg_lambda_long,
            'avg_lambda_short_entry': avg_lambda_short,
            'avg_position_size': avg_position_size,
            'time_in_market': time_in_market,
            'entry_discipline': entry_discipline
        }

        return metrics
    
    def analyze_regime_performance(self,
                                   signals_df: pd.DataFrame,
                                   regime_threshold: float = 0.7) -> Dict:
        """
        Analyze signal characteristics in different regimes

        Regimes: 
        - Calm: λ_t < regime_threshold
        - Volatile: λ_t >= regime_threshold

        Params:
        - signals_df: pd.DataFrame
            - signals dataframe
        - regime_threshold: float
            - lambda threshold for regime classification

        Returns:
        - dict:
            - regime-specific metrics
        """

        #Classify regimes
        calm_regime = signals_df['lambda'] < regime_threshold
        volatile_regime = ~calm_regime 

        #Entries by regime 
        calm_entries = signals_df[calm_regime & (signals_df['signal'].isin([1,-1]))]
        volatile_entries = signals_df[volatile_regime & (signals_df['signal'].isin([1,-1]))]

        #Time in each regime
        time_calm = calm_regime.sum() / len(signals_df)
        time_volatile = volatile_regime.sum() / len(signals_df)

        #Signal frequency by regime 
        signal_freq_calm = len(calm_entries) / calm_regime.sum() if calm_regime.sum() > 0 else 0
        signal_freq_volatile = len(volatile_entries) / volatile_regime.sum() if volatile_regime.sum() > 0 else 0

        metrics = {
            'time_in_calm_regime': time_calm,
            'time_in_volatile_regime': time_volatile,
            'entries_in_calm': len(calm_entries),
            'entries_in_volatile': len(volatile_entries),
            'signal_frequency_calm': signal_freq_calm,
            'signal_frequency_volatile': signal_freq_volatile,
            'avg_lambda_calm': signals_df[calm_regime]['lambda'].mean(),
            'avg_lambda_volatile': signals_df[volatile_regime]['lambda'].mean()
        }

        return metrics
    
    def visualize_signals(self,
                          signals_df: pd.DataFrame,
                          save_path: Optional[str] = None):
        
        """
        Visualize trading signals and positions

        Params:
        - signals_df: pd.DataFrame
            - signals dataframe
        - save_path: str, optional
            - path to save plot
        """

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle 

        fig, axes = plt.subplots(4, 1, figsize = (16,12))

        #Plot 1: Spread with the entry/exit markers
        axes[0].plot(signals_df.index, signals_df['spread'], label = 'Spread', color = 'blue', alpha = 0.6, linewidth = 1)

        #Marking entries
        long_entries = signals_df[signals_df['signal'] == SignalType.LONG.value]
        short_entries = signals_df[signals_df['singal'] == SignalType.SHORT.value]
        exits = signals_df[signals_df['signal'] == SignalType.CLOSE.value]

        axes[0].scatter(long_entries.index, long_entries['spread'], color = 'green', marker = '^', s = 100, label = 'Long Entry', zorder = 5)
        axes[0].scatter(short_entries.index, short_entries['spread'], color = 'red', marker = 'v', s = 100, label = 'Short Entry', zorder = 5)
        axes[0].scatter(exits.index, exits['spread'], color = 'orange', marker = 'x', s = 100, label = 'Exit', zorder = 5)

        axes[0].set_ylabel('Spread')
        axes[0].set_title('Trading Signals on Spread')
        axes[0].legend(loc = 'best')
        axes[0].grid(True, alpha = 0.3)

        #Plot 2: Z-score with thresholds
        axes[1].plot(signals_df.index, signals_df['z_score'], color = 'purple', alpha = 0.7, linewidth = 1)
        axes[1].axhiline(y = self.z_entry, color = 'red', linestyle = '--', label = f"Entry threshold (+- {self.z_exit})")
        axes[1].axhline(y = -self.z_entry, color = 'red', linestyle = '--')
        axes[1].axhline(y = self.z_exit, color = 'green', linestyle = '--', label = f'Exit threshold (+= {self.z_exit})')
        axes[1].axhline(y = -self.z_exit, color = 'green', linestyle = '--')
        axes[1].axhline(y = 0, color = 'black', linestyle = '-', alpha = 0.3)

        axes[1].set_ylabel('Z-score')
        axes[1].set_title('Spread Z-score with Entry/Exit Thresholds')
        axes[1].legend(loc = 'best')
        axes[1].grid(True, alpha = 0.3)

        #Plot 3: Jump intensity with threshold 
        axes[2].plot(signals_df.index, signals_df['lambda'], color = 'orange', alpha = 0.7, linewidth = 1)
        axes[2].axhline(y = self.lambda_threshold, color = 'red', linestyle = '--', label = f"Cascade Threshold ({self.lambda_threshold})")
        axes[2].fill_between(signals_df.index, 0, self.lambda_threshold, alpha = 0.2, color = 'green', label = 'safe zone')

        axes[2].set_ylabel('Jump Intensity λ(t)')
        axes[2].set_title('Jump Intensity (Hawkes Process)')
        axes[2].legned(loc = 'best')
        axes[2].grid(True, alpha = 0.3)

        #Plot 4: Position and Position size
        axes[3].fill_between(signals_df.index, 0, signals_df['position_size'], where = (signals_df['position_size'] > 0), color = 'green', alpha = 0.5, label = 'Long')
        axes[3].fill_between(signals_df.index, 0, signals_df['position_size'], where = (signals_df['position_size'] < 0), color = 'red', alpha = 0.5, label = 'Short')

        axes[3].set_ylabel('Position Size')
        axes[3].set_xlabel('Date')
        axes[3].set_title('Position Sizing (jump-aware scaling)')
        axes[3].legend(loc = 'best')
        axes[3].grid(True, alpha = 0.3)
        axes[3].axhline(y = 0, color = 'black', linestyle = '-', linewidth = 0.5)

        plt.tight_layout() 

        if save_path:
            plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
            print(f" Signals plot saved to {save_path}")

            plt.show() 

if __name__ == "__main__":
    #Testing Signal Generation
    import sys
    import os 

    #Adding parent dir to path for imports 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

    from equity_pairs_loader import EquityPairsDataPipeline
    from jump_detector import JumpDetector
    from hawkes_calibration import HawkesProcess 

    print('=' * 70)
    print('Trading Signal Generation Test')
    print('=' * 70)

    #Initialize data pipeline
    print('\n [1] Loading data from CSV files...')
    pipeline = EquityPairsDataPipeline() 

    #Try to load CSV files from multiple possible locations
    csv_paths = [
        (os.path.join(current_dir, 'OHLCV_NVDA.csv'),
         os.path.join(current_dir, 'OHLCV_AMD.csv')), #script directory
        ('OHLCV_NVDA.csv', 'OHLCV_AMD.csv'), #current directory
    ]

    data_loaded = False
    for xom_path, cvx_path in csv_paths:
        if os.path.exists(xom_path) and os.path.exists(cvx_path):
            try:
                data = pipeline.load_from_csv(xom_path, cvx_path)
                data_loaded = True 
                print(f" Loaded from: {os.path.abspath(xom_path)}")
                break 
            except Exception as e:
                continue 

    if not data_loaded:
        print("Could not find CSV files. Testing with synthetic data instead")
        print("\n" + "=" * 70)
        print("Synthetic data test")
        print("=" * 70)

        #Generate synthetic data as a fallback
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2018-01-01', periods = n, freq = 'D')

        #Synthetic mean-reverting spread with jumps 
        spread = np.zeros(n)
        spread[0] = 0.0

        for t in range(1, n):
            #Mean reversion
            spread[t] = 0.9 * spread[t-1] + np.random.normal(0, 0.1)

            #Occasional jumps
            if np.random.random() < 0.02:
                spread[t] += np.random.choice([-0.3, 0.3])

        spread = pd.Series(spread, index = dates)

        #Synthetic z score 
        z_score = (spread - spread.mean()) / spread.std() 

        #synthetic hawkes intensity
        lambda_intensity = pd.Series(0.2 * np.ones(n), index = dates)
        jump_times = np.where(abs(spread.diff()) > 0.2) [0]
        for jt in jump_times:
            for i in range(jt, min(jt + 10, n)):
                decay = np.exp(-0.5 * (i - jt))
                lambda_intensity.iloc[i] += 5.0 * decay

        print(f"\n Generated synthetic data: {n} observations")

        #Generate signals with synthetic data
        signal_gen = TradingSignals(
            z_entry_threshold=2.0,
            z_exit_threshold=0.5,
            lambda_threshold=3.0,
            scaling_constant=0.1
        )
        
        signals_df = signal_gen.generate_signals(z_score, lambda_intensity, spread)
        
        # Print metrics
        print("\n" + "="*70)
        print("SIGNAL QUALITY METRICS")
        print("="*70)
        quality = signal_gen.calculate_signal_quality(signals_df)
        for metric, value in quality.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        sys.exit(0)
    
    # Clean and construct spread
    print("\n[2] Constructing spread...")
    cleaned_data = pipeline.clean_data()
    spread_df = pipeline.construct_spread(method='cointegration', lookback=60)
    
    spread = spread_df['spread']
    print(f"✓ Spread constructed: {len(spread)} observations")
    print(f"  Date range: {spread.index[0].date()} to {spread.index[-1].date()}")
    print(f"  Spread range: [{spread.min():.4f}, {spread.max():.4f}]")
    
    # Calculate Z-score
    print("\n[3] Calculating Z-score...")
    mean = spread.mean()
    std = spread.std()
    z_score = (spread - mean) / std
    
    print(f"✓ Z-score statistics:")
    print(f"  Mean: {z_score.mean():.4f}")
    print(f"  Std: {z_score.std():.4f}")
    print(f"  Range: [{z_score.min():.4f}, {z_score.max():.4f}]")
    
    # Detect jumps
    print("\n[4] Detecting jumps and calibrating Hawkes...")
    returns = spread.pct_change().dropna()
    
    detector = JumpDetector(significance_level=0.01)
    jump_df = detector.detect_jumps_bipower_variation(returns, window=20)
    
    # Extract jump times
    jump_times = jump_df[jump_df['jump_indicator'] == 1].index
    jump_times_numeric = (jump_times - spread.index[0]).days.values
    
    n_jumps = len(jump_times)
    print(f"✓ Detected {n_jumps} jumps ({100*n_jumps/len(returns):.2f}%)")
    
    # Fit Hawkes process
    T = (spread.index[-1] - spread.index[0]).days
    
    hawkes = HawkesProcess()
    if n_jumps >= 2:
        try:
            params = hawkes.fit(jump_times_numeric, T, method='MLE')
            print(f"✓ Hawkes parameters fitted:")
            print(f"  λ̄ = {params['lambda_bar']:.6f}")
            print(f"  α = {params['alpha']:.6f}")
            print(f"  β = {params['beta']:.6f}")
        except:
            print("⚠ Using baseline Hawkes parameters")
            params = {'lambda_bar': 0.001, 'alpha': 0.0, 'beta': 1.0}
    else:
        print("⚠ Too few jumps for Hawkes calibration, using baseline")
        params = {'lambda_bar': 0.001, 'alpha': 0.0, 'beta': 1.0}
    
    # Calculate time-varying jump intensity
    print("\n[5] Calculating jump intensity...")
    lambda_intensity = pd.Series(params['lambda_bar'], index=spread.index)
    
    for idx, t in enumerate(spread.index):
        lambda_t = params['lambda_bar']
        t_numeric = (t - spread.index[0]).days
        
        for jt in jump_times_numeric:
            if jt < t_numeric:
                lambda_t += params['alpha'] * np.exp(-params['beta'] * (t_numeric - jt))
        
        lambda_intensity.iloc[idx] = lambda_t
    
    print(f"✓ Jump intensity range: [{lambda_intensity.min():.6f}, {lambda_intensity.max():.6f}]")
    
    # Generate trading signals
    print("\n[6] Generating trading signals...")
    signal_gen = TradingSignals(
        z_entry_threshold=2.0,
        z_exit_threshold=0.5,
        lambda_threshold=5.0,
        scaling_constant=0.1
    )
    
    signals_df = signal_gen.generate_signals(z_score, lambda_intensity, spread)
    
    # Count signals
    n_long = len(signals_df[signals_df['signal'] == SignalType.LONG.value])
    n_short = len(signals_df[signals_df['signal'] == SignalType.SHORT.value])
    n_close = len(signals_df[signals_df['signal'] == SignalType.CLOSE.value])
    
    print(f"✓ Generated signals:")
    print(f"  Long entries: {n_long}")
    print(f"  Short entries: {n_short}")
    print(f"  Closes: {n_close}")
    print(f"  Total: {n_long + n_short + n_close}")
    
    # Signal quality metrics
    print("\n" + "="*70)
    print("SIGNAL QUALITY METRICS")
    print("="*70)
    quality = signal_gen.calculate_signal_quality(signals_df)
    for metric, value in quality.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Regime analysis
    print("\n" + "="*70)
    print("REGIME ANALYSIS")
    print("="*70)
    regime_metrics = signal_gen.analyze_regime_performance(
        signals_df, regime_threshold=2.0
    )
    for metric, value in regime_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Visualization
    print("\n[7] Creating visualization...")
    signal_gen.visualize_signals(signals_df, save_path='trading_signals_test_xom_cvx.png')
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nData: NVDA/AMD spread ({len(spread)} days)")
    print(f"Date range: {spread.index[0].date()} to {spread.index[-1].date()}")
    print(f"\nSpread Statistics:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std: {std:.4f}")
    print(f"  Z-score range: [{z_score.min():.2f}, {z_score.max():.2f}]")
    print(f"\nJump Activity:")
    print(f"  Total jumps: {n_jumps}")
    print(f"  Hawkes λ̄: {params['lambda_bar']:.6f}")
    print(f"  Branching: {params['alpha']/params['beta']:.6f}")
    print(f"\nTrading Signals:")
    print(f"  Total signals: {n_long + n_short + n_close}")
    print(f"  Entry discipline: {quality['entry_discipline']*100:.1f}%")
    print(f"  Time in market: {quality['time_in_market']*100:.1f}%")
    print(f"\n✓ All tests completed successfully!")
    print("="*70)
