"""
Improved Trading Signal Generation for Self-Exciting Pairs Trading

Key improvements over v1:
1. Jump-aware entry signals (use jumps to trigger entries)
2. Relaxed exit conditions (remove aggressive z-cross-zero exit)
3. Empirical z-score option (more stable than MRJD theoretical)
4. Adaptive thresholds based on regime
"""

import numpy as np 
import pandas as pd
from typing import Dict, Tuple, Optional
from enum import Enum 

class SignalType(Enum):
    """Trading signal types"""
    NO_SIGNAL = 0
    LONG = 1
    SHORT = -1
    CLOSE = 2

class TradingSignals:
    """
    Improved jump-aware trading signals for pairs trading
    
    Key changes:
    - Uses empirical z-score (rolling) instead of MRJD theoretical
    - Jump arrivals can trigger entries (not just z-score)
    - Relaxed exit conditions
    - Adaptive position sizing
    """

    def __init__(self,
                 z_entry_threshold: float = 1.5,
                 z_exit_threshold: float = 0.3,
                 lambda_threshold: float = 0.5,
                 scaling_constant: float = 0.1,
                 max_position_size: float = 1.0,
                 use_jump_entries: bool = True,
                 z_lookback: int = 60,
                 max_holding_period: int = 30):
        """
        Initialize signal generator with improved parameters
        
        Params:
        - z_entry_threshold: float - minimum z-score for entry (lowered from 2.0)
        - z_exit_threshold: float - z-score for mean reversion exit
        - lambda_threshold: float - max jump intensity for safe entry
        - scaling_constant: float - position scaling constant
        - max_position_size: float - maximum position size
        - use_jump_entries: bool - whether to use jump arrivals for entry signals
        - z_lookback: int - lookback for empirical z-score calculation
        - max_holding_period: int - maximum days to hold a position
        """
        self.z_entry = z_entry_threshold
        self.z_exit = z_exit_threshold 
        self.lambda_threshold = lambda_threshold 
        self.c = scaling_constant
        self.max_position = max_position_size 
        self.use_jump_entries = use_jump_entries
        self.z_lookback = z_lookback
        self.max_holding_period = max_holding_period

        self.signals = None 
        self.positions = None 

    def calculate_empirical_zscore(self, spread: pd.Series, lookback: Optional[int] = None) -> pd.Series:
        """
        Calculate empirical z-score using rolling statistics
        
        This is more stable than MRJD theoretical z-score because it uses
        actual observed mean and std, not model-implied values.
        """
        if lookback is None:
            lookback = self.z_lookback
            
        rolling_mean = spread.rolling(window=lookback, min_periods=20).mean()
        rolling_std = spread.rolling(window=lookback, min_periods=20).std()
        
        z_score = (spread - rolling_mean) / rolling_std
        
        # Fill initial NaN with expanding window
        for i in range(min(lookback, len(spread))):
            if pd.isna(z_score.iloc[i]) and i >= 5:
                expanding_mean = spread.iloc[:i+1].mean()
                expanding_std = spread.iloc[:i+1].std()
                if expanding_std > 0:
                    z_score.iloc[i] = (spread.iloc[i] - expanding_mean) / expanding_std
        
        return z_score.fillna(0)

    def generate_signals(self,
                         spread: pd.Series,
                         lambda_intensity: pd.Series,
                         jump_indicator: Optional[pd.Series] = None,
                         z_score: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate trading signals with improved logic
        
        Entry Conditions (ANY of the following when not in position):
        1. |Z_t| > z_entry_threshold AND λ_t < lambda_threshold
        2. Jump detected AND |Z_t| > z_entry_threshold * 0.7 (jump-assisted entry)
        
        Exit Conditions (ANY of the following when in position):
        1. |Z_t| < z_exit_threshold (mean reversion complete)
        2. Position held > max_holding_period
        3. Stop loss: z moved further against position by > 2 std
        
        REMOVED: Aggressive z-cross-zero exit (was killing trades prematurely)
        """
        print("Generating improved jump-aware trading signals")

        # Align all series
        common_index = spread.index.intersection(lambda_intensity.index)
        spread = spread.loc[common_index]
        lambda_intensity = lambda_intensity.loc[common_index]
        
        if jump_indicator is not None:
            common_index = common_index.intersection(jump_indicator.index)
            jump_indicator = jump_indicator.loc[common_index]
            spread = spread.loc[common_index]
            lambda_intensity = lambda_intensity.loc[common_index]
        
        # Calculate empirical z-score if not provided
        if z_score is None:
            z_score = self.calculate_empirical_zscore(spread)
        else:
            z_score = z_score.loc[common_index]

        n = len(spread)
        signals = np.zeros(n)
        positions = np.zeros(n)
        position_sizes = np.zeros(n)

        current_position = 0
        entry_idx = 0
        entry_z = 0.0

        for i in range(1, n):
            z_t = z_score.iloc[i]
            lambda_t = lambda_intensity.iloc[i]
            
            # Check for jump at this time
            is_jump = False
            if jump_indicator is not None:
                is_jump = jump_indicator.iloc[i] == 1

            # Entry logic
            if current_position == 0:
                # Standard entry: large z-score, low jump intensity
                standard_entry_long = (z_t < -self.z_entry) and (lambda_t < self.lambda_threshold)
                standard_entry_short = (z_t > self.z_entry) and (lambda_t < self.lambda_threshold)
                
                # Jump-assisted entry: jump detected with moderately extreme z-score
                jump_entry_long = False
                jump_entry_short = False
                if self.use_jump_entries and is_jump:
                    # Lower threshold for jump-assisted entries
                    jump_threshold = self.z_entry * 0.6
                    jump_entry_long = z_t < -jump_threshold
                    jump_entry_short = z_t > jump_threshold
                
                if standard_entry_long or jump_entry_long:
                    signals[i] = SignalType.LONG.value
                    current_position = 1
                    entry_idx = i
                    entry_z = z_t
                    
                elif standard_entry_short or jump_entry_short:
                    signals[i] = SignalType.SHORT.value
                    current_position = -1
                    entry_idx = i
                    entry_z = z_t

            # Exit logic (when in position)
            elif current_position != 0:
                holding_days = i - entry_idx
                
                # Exit conditions
                exit_mean_reversion = abs(z_t) < self.z_exit
                exit_max_holding = holding_days >= self.max_holding_period
                
                # Stop loss: z moved further against us
                # Long position: entered at negative z, stop if z becomes MORE negative
                # Short position: entered at positive z, stop if z becomes MORE positive
                stop_loss_threshold = 2.0  # Additional z-score move against position
                if current_position == 1:
                    exit_stop_loss = z_t < (entry_z - stop_loss_threshold)
                else:
                    exit_stop_loss = z_t > (entry_z + stop_loss_threshold)
                
                # Profit taking: z crossed zero significantly (optional, less aggressive)
                # Only exit if we've captured most of the move
                profit_target_reached = False
                if current_position == 1 and z_t > 0.5:  # Long: z went from negative to positive
                    profit_target_reached = True
                elif current_position == -1 and z_t < -0.5:  # Short: z went from positive to negative
                    profit_target_reached = True
                
                if exit_mean_reversion or exit_max_holding or exit_stop_loss or profit_target_reached:
                    signals[i] = SignalType.CLOSE.value
                    current_position = 0

            # Track position
            positions[i] = current_position

            # Calculate position size
            if current_position != 0:
                raw_size = abs(z_t) / (1 + self.c * lambda_t)
                position_sizes[i] = min(raw_size, self.max_position) * np.sign(current_position)

        # Create signals dataframe
        signals_df = pd.DataFrame({
            'signal': signals,
            'position': positions,
            'position_size': position_sizes,
            'z_score': z_score,
            'lambda': lambda_intensity,
            'spread': spread
        }, index=common_index)
        
        if jump_indicator is not None:
            signals_df['jump'] = jump_indicator

        # Signal statistics
        n_signals = (signals != 0).sum()
        n_long = (signals == SignalType.LONG.value).sum()
        n_short = (signals == SignalType.SHORT.value).sum()
        n_close = (signals == SignalType.CLOSE.value).sum()

        print(f"Generated {n_signals} total signals:")
        print(f"    - Long Entries: {n_long}")
        print(f"    - Short Entries: {n_short}")
        print(f"    - Closes: {n_close}")

        self.signals = signals_df
        return signals_df

    def calculate_signal_quality(self, signals_df: pd.DataFrame) -> Dict:
        """Calculate signal quality metrics"""
        long_entries = signals_df[signals_df['signal'] == SignalType.LONG.value]
        short_entries = signals_df[signals_df['signal'] == SignalType.SHORT.value]

        # Average conditions at entry
        avg_z_long = long_entries['z_score'].mean() if len(long_entries) > 0 else np.nan
        avg_lambda_long = long_entries['lambda'].mean() if len(long_entries) > 0 else np.nan
        avg_z_short = short_entries['z_score'].mean() if len(short_entries) > 0 else np.nan
        avg_lambda_short = short_entries['lambda'].mean() if len(short_entries) > 0 else np.nan

        # Position metrics
        in_position = signals_df[signals_df['position'] != 0]
        avg_position_size = in_position['position_size'].abs().mean() if len(in_position) > 0 else 0
        time_in_market = (signals_df['position'] != 0).sum() / len(signals_df)

        # Entry discipline
        all_extreme_z = signals_df[abs(signals_df['z_score']) > self.z_entry]
        safe_entries = all_extreme_z[all_extreme_z['lambda'] < self.lambda_threshold]
        entry_discipline = len(safe_entries) / len(all_extreme_z) if len(all_extreme_z) > 0 else 0

        # Jump-assisted entries
        if 'jump' in signals_df.columns:
            entries = signals_df[signals_df['signal'].isin([1, -1])]
            jump_assisted = entries[entries['jump'] == 1]
            jump_entry_pct = len(jump_assisted) / len(entries) if len(entries) > 0 else 0
        else:
            jump_entry_pct = 0

        metrics = {
            'avg_z_score_long_entry': avg_z_long,
            'avg_z_score_short_entry': avg_z_short,
            'avg_lambda_long_entry': avg_lambda_long,
            'avg_lambda_short_entry': avg_lambda_short,
            'avg_position_size': avg_position_size,
            'time_in_market': time_in_market,
            'entry_discipline': entry_discipline,
            'jump_assisted_entry_pct': jump_entry_pct
        }

        return metrics


class AdaptiveSignalGenerator:
    """
    Adaptive signal generator that adjusts thresholds based on market regime
    """
    
    def __init__(self, base_z_entry: float = 1.5, base_z_exit: float = 0.3):
        self.base_z_entry = base_z_entry
        self.base_z_exit = base_z_exit
        
    def calculate_adaptive_thresholds(self, 
                                       spread: pd.Series,
                                       lambda_intensity: pd.Series,
                                       lookback: int = 60) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate time-varying entry/exit thresholds based on:
        1. Recent spread volatility
        2. Jump intensity regime
        """
        # Rolling volatility of spread
        rolling_vol = spread.rolling(window=lookback, min_periods=20).std()
        vol_percentile = rolling_vol.rank(pct=True)
        
        # Adaptive entry threshold: higher when vol is high
        # Range: [base_z_entry * 0.7, base_z_entry * 1.5]
        z_entry_adaptive = self.base_z_entry * (0.7 + 0.8 * vol_percentile)
        
        # Adaptive exit threshold: tighter when vol is low (capture small moves)
        z_exit_adaptive = self.base_z_exit * (0.8 + 0.4 * vol_percentile)
        
        # Adjust for jump intensity regime
        lambda_percentile = lambda_intensity.rank(pct=True)
        
        # When jump intensity is high, be more conservative (wider thresholds)
        z_entry_adaptive = z_entry_adaptive * (1 + 0.3 * lambda_percentile)
        
        return z_entry_adaptive.fillna(self.base_z_entry), z_exit_adaptive.fillna(self.base_z_exit)
    
    def generate_signals(self,
                         spread: pd.Series,
                         lambda_intensity: pd.Series,
                         jump_indicator: Optional[pd.Series] = None) -> pd.DataFrame:
        """Generate signals with adaptive thresholds"""
        
        z_entry_adaptive, z_exit_adaptive = self.calculate_adaptive_thresholds(
            spread, lambda_intensity
        )
        
        # Calculate empirical z-score
        z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        z_score = z_score.fillna(0)
        
        common_index = spread.index.intersection(lambda_intensity.index)
        spread = spread.loc[common_index]
        lambda_intensity = lambda_intensity.loc[common_index]
        z_score = z_score.loc[common_index]
        z_entry_adaptive = z_entry_adaptive.loc[common_index]
        z_exit_adaptive = z_exit_adaptive.loc[common_index]
        
        if jump_indicator is not None:
            common_index = common_index.intersection(jump_indicator.index)
            jump_indicator = jump_indicator.loc[common_index]
            spread = spread.loc[common_index]
            lambda_intensity = lambda_intensity.loc[common_index]
            z_score = z_score.loc[common_index]
            z_entry_adaptive = z_entry_adaptive.loc[common_index]
            z_exit_adaptive = z_exit_adaptive.loc[common_index]
        
        n = len(spread)
        signals = np.zeros(n)
        positions = np.zeros(n)
        position_sizes = np.zeros(n)
        
        current_position = 0
        entry_idx = 0
        
        for i in range(1, n):
            z_t = z_score.iloc[i]
            lambda_t = lambda_intensity.iloc[i]
            z_entry_t = z_entry_adaptive.iloc[i]
            z_exit_t = z_exit_adaptive.iloc[i]
            
            is_jump = jump_indicator.iloc[i] == 1 if jump_indicator is not None else False
            
            if current_position == 0:
                # Entry with adaptive threshold
                if z_t < -z_entry_t and lambda_t < 0.5:
                    signals[i] = 1
                    current_position = 1
                    entry_idx = i
                elif z_t > z_entry_t and lambda_t < 0.5:
                    signals[i] = -1
                    current_position = -1
                    entry_idx = i
                # Jump-assisted entry
                elif is_jump and abs(z_t) > z_entry_t * 0.6:
                    signals[i] = -1 if z_t > 0 else 1
                    current_position = signals[i]
                    entry_idx = i
                    
            elif current_position != 0:
                holding_days = i - entry_idx
                
                # Exit conditions
                if abs(z_t) < z_exit_t or holding_days > 25:
                    signals[i] = 2
                    current_position = 0
                # Profit target
                elif (current_position == 1 and z_t > 0.5) or (current_position == -1 and z_t < -0.5):
                    signals[i] = 2
                    current_position = 0
            
            positions[i] = current_position
            if current_position != 0:
                position_sizes[i] = min(abs(z_t) / (1 + 0.1 * lambda_t), 1.0) * np.sign(current_position)
        
        signals_df = pd.DataFrame({
            'signal': signals,
            'position': positions,
            'position_size': position_sizes,
            'z_score': z_score,
            'lambda': lambda_intensity,
            'spread': spread,
            'z_entry_threshold': z_entry_adaptive,
            'z_exit_threshold': z_exit_adaptive
        }, index=common_index)
        
        n_signals = (signals != 0).sum()
        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()
        n_close = (signals == 2).sum()
        
        print(f"Generated {n_signals} total signals (adaptive):")
        print(f"    - Long Entries: {n_long}")
        print(f"    - Short Entries: {n_short}")
        print(f"    - Closes: {n_close}")
        
        return signals_df


if __name__ == "__main__":
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    print("="*70)
    print("IMPROVED SIGNAL GENERATION TEST")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Mean-reverting spread with jumps
    spread = np.zeros(n)
    spread[0] = 0.0
    jumps = np.zeros(n)
    
    for t in range(1, n):
        spread[t] = 0.95 * spread[t-1] + np.random.normal(0, 0.1)
        if np.random.random() < 0.03:  # 3% jump probability
            spread[t] += np.random.choice([-0.4, 0.4])
            jumps[t] = 1
    
    spread = pd.Series(spread, index=dates)
    jump_indicator = pd.Series(jumps, index=dates)
    
    # Synthetic lambda intensity
    lambda_intensity = pd.Series(0.1 * np.ones(n), index=dates)
    jump_times = np.where(jumps == 1)[0]
    for jt in jump_times:
        for i in range(jt, min(jt + 10, n)):
            lambda_intensity.iloc[i] += 0.3 * np.exp(-0.5 * (i - jt))
    
    print(f"\nTest data: {n} observations, {int(jumps.sum())} jumps")
    
    # Test improved signal generator
    print("\n" + "-"*70)
    print("Testing TradingSignalsV2")
    print("-"*70)
    
    signal_gen = TradingSignals(
        z_entry_threshold=1.5,
        z_exit_threshold=0.3,
        lambda_threshold=0.5,
        use_jump_entries=True,
        max_holding_period=25
    )
    
    signals_df = signal_gen.generate_signals(
        spread, 
        lambda_intensity,
        jump_indicator=jump_indicator
    )
    
    quality = signal_gen.calculate_signal_quality(signals_df)
    print(f"\nSignal Quality:")
    for k, v in quality.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Test adaptive signal generator
    print("\n" + "-"*70)
    print("Testing AdaptiveSignalGenerator")
    print("-"*70)
    
    adaptive_gen = AdaptiveSignalGenerator(base_z_entry=1.5, base_z_exit=0.3)
    adaptive_signals = adaptive_gen.generate_signals(spread, lambda_intensity, jump_indicator)
    
    time_in_market = (adaptive_signals['position'] != 0).mean()
    print(f"\nTime in market: {time_in_market:.2%}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
