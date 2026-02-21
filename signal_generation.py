"""
Optimized Trading Signal Generation for Self-Exciting Pairs Trading

Key Optimizations:
1. Half-life aware holding periods (match exit timing to spread dynamics)
2. Dynamic thresholds based on Hawkes λ (regime-adaptive entries)
3. λ decay detection (only enter when jump cascade is subsiding)
4. Fewer, higher-quality trades (stricter entry, more patient exits)
5. Full Hawkes utilization for timing and sizing
"""

import numpy as np 
import pandas as pd
from typing import Dict, Tuple, Optional
from enum import Enum 
from dataclasses import dataclass


class SignalType(Enum):
    """Trading signal types"""
    NO_SIGNAL = 0
    LONG = 1
    SHORT = -1
    CLOSE = 2


@dataclass
class HawkesRegime:
    """Hawkes-based market regime classification"""
    CALM = "calm"           # λ < 25th percentile - aggressive entries OK
    NORMAL = "normal"       # 25th <= λ < 75th percentile - standard entries
    ELEVATED = "elevated"   # 75th <= λ < 90th percentile - selective entries
    CRISIS = "crisis"       # λ >= 90th percentile - no new entries


class TradingSignals:
    """
    Optimized trading signal generator with full Hawkes utilization
    
    Key innovations:
    1. Half-life aware exits: Hold positions proportional to spread half-life
    2. λ-based regime detection: Adjust thresholds based on jump intensity
    3. λ decay entry: Only enter when intensity is falling (post-cascade)
    4. Dynamic position sizing: Inverse relationship with λ
    5. Regime-triggered exits: Exit early if regime deteriorates
    """

    def __init__(self,
                 # Entry thresholds (will be adjusted by regime)
                 z_entry_threshold: float = 2.0,      # Increased from 1.5 for selectivity
                 z_exit_threshold: float = 0.5,       # Increased from 0.3 for patience
                 
                 # Hawkes parameters
                 lambda_threshold: float = 0.5,       # Max λ for standard entry
                 lambda_decay_lookback: int = 5,      # Days to check λ trend
                 min_lambda_decay_pct: float = 0.15,  # λ must drop 15% from peak
                 
                 # Position sizing
                 scaling_constant: float = 0.1,
                 max_position_size: float = 0.25,     # Reduced from 1.0
                 min_position_size: float = 0.10,
                 
                 # Holding period (will be overridden by half-life)
                 max_holding_period: int = 30,
                 
                 # Features
                 use_jump_entries: bool = True,
                 use_hawkes_regimes: bool = True,
                 z_lookback: int = 60):
        
        # Entry/exit thresholds
        self.z_entry = z_entry_threshold
        self.z_exit = z_exit_threshold 
        self.lambda_threshold = lambda_threshold
        
        # Hawkes decay parameters
        self.lambda_decay_lookback = lambda_decay_lookback
        self.min_lambda_decay_pct = min_lambda_decay_pct
        
        # Position sizing
        self.c = scaling_constant
        self.max_position = max_position_size
        self.min_position = min_position_size
        
        # Holding period
        self.max_holding_period = max_holding_period
        self.half_life = None  # Will be set externally
        
        # Features
        self.use_jump_entries = use_jump_entries
        self.use_hawkes_regimes = use_hawkes_regimes
        self.z_lookback = z_lookback

        # State
        self.signals = None 
        self.positions = None
        self.lambda_percentiles = {}
        
        # Diagnostics
        self.entries_blocked_by_regime = 0
        self.entries_blocked_by_decay = 0
        self.regime_exits = 0

    def set_half_life(self, half_life: float):
        """
        Set half-life for holding period calibration
        
        This is CRITICAL for proper exit timing:
        - Min hold: 50% of half-life (let mean reversion start)
        - Target hold: 80% of half-life (capture most of reversion)
        - Max hold: 150% of half-life (time stop)
        """
        self.half_life = half_life
        self.min_hold = int(half_life * 0.5)
        self.target_hold = int(half_life * 0.8)
        self.max_holding_period = int(half_life * 1.5)
        
        print(f"  Half-life calibration:")
        print(f"    Spread half-life: {half_life:.1f} days")
        print(f"    Min hold period: {self.min_hold} days")
        print(f"    Target hold: {self.target_hold} days")
        print(f"    Max hold period: {self.max_holding_period} days")

    def _calculate_lambda_percentiles(self, lambda_intensity: pd.Series):
        """Calculate λ percentiles for regime classification"""
        self.lambda_percentiles = {
            'p25': lambda_intensity.quantile(0.25),
            'p50': lambda_intensity.quantile(0.50),
            'p75': lambda_intensity.quantile(0.75),
            'p90': lambda_intensity.quantile(0.90),
            'p95': lambda_intensity.quantile(0.95)
        }

    def _get_regime(self, lambda_t: float) -> str:
        """Determine current regime based on λ"""
        if not self.lambda_percentiles:
            return HawkesRegime.NORMAL
            
        if lambda_t < self.lambda_percentiles.get('p25', 0.02):
            return HawkesRegime.CALM
        elif lambda_t < self.lambda_percentiles.get('p75', 0.10):
            return HawkesRegime.NORMAL
        elif lambda_t < self.lambda_percentiles.get('p90', 0.20):
            return HawkesRegime.ELEVATED
        else:
            return HawkesRegime.CRISIS

    def _get_regime_thresholds(self, regime: str) -> Tuple[float, float, int]:
        """
        Get entry threshold, exit threshold, and max hold based on regime
        
        CALM: More aggressive (tighter thresholds, longer hold)
        CRISIS: More defensive (wider thresholds, shorter hold)
        """
        if regime == HawkesRegime.CALM:
            z_entry = self.z_entry * 0.85    # 15% easier entry
            z_exit = self.z_exit * 0.85      # Tighter exit (more patient)
            max_hold = int(self.max_holding_period * 1.2)  # Hold longer
        elif regime == HawkesRegime.NORMAL:
            z_entry = self.z_entry
            z_exit = self.z_exit
            max_hold = self.max_holding_period
        elif regime == HawkesRegime.ELEVATED:
            z_entry = self.z_entry * 1.25    # 25% stricter entry
            z_exit = self.z_exit * 1.2       # Looser exit (exit earlier)
            max_hold = int(self.max_holding_period * 0.75)
        else:  # CRISIS
            z_entry = self.z_entry * 1.5     # 50% stricter (very selective)
            z_exit = self.z_exit * 1.3       # Exit even earlier
            max_hold = int(self.max_holding_period * 0.5)
        
        return z_entry, z_exit, max_hold

    def _is_lambda_decaying(self, lambda_series: pd.Series, current_idx: int) -> bool:
        """
        Check if λ is in decay phase (safe to enter)
        
        Don't enter during a jump cascade - wait for intensity to subside.
        This is key for Hawkes utilization: predict the POST-jump period.
        """
        if current_idx < self.lambda_decay_lookback:
            return True
        
        lookback_values = lambda_series.iloc[current_idx - self.lambda_decay_lookback:current_idx + 1]
        peak_lambda = lookback_values.max()
        current_lambda = lookback_values.iloc[-1]
        
        if peak_lambda > 0:
            decay_pct = (peak_lambda - current_lambda) / peak_lambda
            return decay_pct >= self.min_lambda_decay_pct
        return True

    def _calculate_position_size(self, z_score: float, lambda_t: float, regime: str) -> float:
        """
        Dynamic position sizing based on signal strength and regime
        
        Larger position when:
        - |Z| is larger (stronger signal)
        - λ is lower (calmer market)
        - Regime is calm
        """
        # Z-score factor: stronger signal = larger position
        z_factor = min(abs(z_score) / 3.0, 1.5)
        
        # Lambda factor: lower λ = larger position (inverse relationship)
        lambda_90 = self.lambda_percentiles.get('p90', 0.20)
        if lambda_90 > 0:
            lambda_ratio = lambda_t / lambda_90
            lambda_factor = max(0.5, 1.5 - lambda_ratio)
        else:
            lambda_factor = 1.0
        
        # Regime factor
        regime_factors = {
            HawkesRegime.CALM: 1.2,
            HawkesRegime.NORMAL: 1.0,
            HawkesRegime.ELEVATED: 0.7,
            HawkesRegime.CRISIS: 0.5
        }
        regime_factor = regime_factors.get(regime, 1.0)
        
        # Combined
        position = self.max_position * z_factor * lambda_factor * regime_factor
        return max(self.min_position, min(position, self.max_position))

    def calculate_empirical_zscore(self, spread: pd.Series, lookback: Optional[int] = None) -> pd.Series:
        """Calculate empirical z-score using rolling statistics"""
        if lookback is None:
            lookback = self.z_lookback
            
        rolling_mean = spread.rolling(window=lookback, min_periods=20).mean()
        rolling_std = spread.rolling(window=lookback, min_periods=20).std()
        z_score = (spread - rolling_mean) / rolling_std
        return z_score.fillna(0)

    def generate_signals(self,
                         spread: pd.Series,
                         lambda_intensity: pd.Series,
                         jump_indicator: Optional[pd.Series] = None,
                         z_score: Optional[pd.Series] = None,
                         half_life: Optional[float] = None) -> pd.DataFrame:
        """
        Generate optimized trading signals with full Hawkes utilization
        
        Entry Conditions:
        1. |Z| > regime-adjusted threshold
        2. λ is in decay phase (not during cascade)
        3. Regime is not CRISIS
        4. (Optional) Jump-assisted entry with lower threshold
        
        Exit Conditions:
        1. |Z| < regime-adjusted exit threshold (mean reversion)
        2. Holding period > max (regime-adjusted)
        3. Profit target reached (Z crossed favorable)
        4. Regime escalated to CRISIS
        5. Minimum hold not yet reached (prevents early exit)
        """
        print("Generating optimized Hawkes-aware trading signals...")
        
        # Set half-life if provided
        if half_life is not None:
            self.set_half_life(half_life)
        elif self.half_life is None:
            print("  Warning: No half-life set, using default 30 days")
            self.half_life = 30.0
            self.min_hold = 15
            self.target_hold = 24
            self.max_holding_period = 45

        # Reset diagnostics
        self.entries_blocked_by_regime = 0
        self.entries_blocked_by_decay = 0
        self.regime_exits = 0

        # Align data
        common_index = spread.index.intersection(lambda_intensity.index)
        spread = spread.loc[common_index]
        lambda_intensity = lambda_intensity.loc[common_index]
        
        if jump_indicator is not None:
            common_index = common_index.intersection(jump_indicator.index)
            jump_indicator = jump_indicator.loc[common_index]
            spread = spread.loc[common_index]
            lambda_intensity = lambda_intensity.loc[common_index]
        
        # Calculate z-score
        if z_score is None:
            z_score = self.calculate_empirical_zscore(spread)
        else:
            z_score = z_score.loc[common_index]

        # Calculate λ percentiles for regime classification
        self._calculate_lambda_percentiles(lambda_intensity)
        
        print(f"  λ percentiles: p25={self.lambda_percentiles['p25']:.4f}, "
              f"p75={self.lambda_percentiles['p75']:.4f}, "
              f"p90={self.lambda_percentiles['p90']:.4f}")

        n = len(spread)
        signals = np.zeros(n)
        positions = np.zeros(n)
        position_sizes = np.zeros(n)
        regimes = []

        # Position state
        current_position = 0
        entry_idx = 0
        entry_z = 0.0
        entry_regime = None
        current_position_size = 0.0

        for i in range(1, n):
            z_t = z_score.iloc[i]
            lambda_t = lambda_intensity.iloc[i]
            
            # Determine regime
            if self.use_hawkes_regimes:
                regime = self._get_regime(lambda_t)
            else:
                regime = HawkesRegime.NORMAL
            regimes.append(regime)
            
            # Get regime-adjusted thresholds
            z_entry_adj, z_exit_adj, max_hold_adj = self._get_regime_thresholds(regime)
            
            # Check for jump
            is_jump = False
            if jump_indicator is not None:
                is_jump = jump_indicator.iloc[i] == 1

            # ENTRY LOGIC
            if current_position == 0:
                # Check basic z-score condition
                long_signal = z_t < -z_entry_adj
                short_signal = z_t > z_entry_adj
                
                if long_signal or short_signal:
                    # Block entry in CRISIS regime
                    if regime == HawkesRegime.CRISIS:
                        self.entries_blocked_by_regime += 1
                        continue
                    
                    # Check λ decay (only enter when cascade is subsiding)
                    if self.use_hawkes_regimes:
                        lambda_decaying = self._is_lambda_decaying(lambda_intensity, i)
                        if not lambda_decaying:
                            self.entries_blocked_by_decay += 1
                            continue
                    
                    # Execute entry
                    if long_signal:
                        signals[i] = SignalType.LONG.value
                        current_position = 1
                    else:
                        signals[i] = SignalType.SHORT.value
                        current_position = -1
                    
                    entry_idx = i
                    entry_z = z_t
                    entry_regime = regime
                    current_position_size = self._calculate_position_size(z_t, lambda_t, regime)
                
                # Jump-assisted entry (lower threshold)
                elif self.use_jump_entries and is_jump:
                    jump_threshold = z_entry_adj * 0.65
                    if z_t < -jump_threshold:
                        signals[i] = SignalType.LONG.value
                        current_position = 1
                        entry_idx = i
                        entry_z = z_t
                        entry_regime = regime
                        current_position_size = self._calculate_position_size(z_t, lambda_t, regime) * 0.8
                    elif z_t > jump_threshold:
                        signals[i] = SignalType.SHORT.value
                        current_position = -1
                        entry_idx = i
                        entry_z = z_t
                        entry_regime = regime
                        current_position_size = self._calculate_position_size(z_t, lambda_t, regime) * 0.8

            # EXIT LOGIC
            elif current_position != 0:
                holding_days = i - entry_idx
                
                # Get current regime thresholds
                _, z_exit_current, max_hold_current = self._get_regime_thresholds(regime)
                
                exit_signal = False
                exit_reason = None
                
                # 1. Mean reversion complete (but respect min hold)
                if abs(z_t) < z_exit_current and holding_days >= self.min_hold:
                    exit_signal = True
                    exit_reason = 'mean_reversion'
                
                # 2. Profit target: Z crossed favorable threshold (past zero)
                elif holding_days >= self.min_hold:
                    if current_position == 1 and z_t > 0.5:
                        exit_signal = True
                        exit_reason = 'profit_target'
                    elif current_position == -1 and z_t < -0.5:
                        exit_signal = True
                        exit_reason = 'profit_target'
                
                # 3. Time stop (regime-adjusted)
                elif holding_days >= max_hold_current:
                    exit_signal = True
                    exit_reason = 'max_hold'
                
                # 4. Regime escalated to CRISIS (exit early to protect)
                elif regime == HawkesRegime.CRISIS and entry_regime != HawkesRegime.CRISIS:
                    if holding_days >= self.min_hold // 2:  # Allow some min hold even in crisis
                        exit_signal = True
                        exit_reason = 'regime_crisis'
                        self.regime_exits += 1
                
                # 5. Emergency stop: Z moved significantly against position
                elif current_position == 1 and z_t < entry_z - 2.5:
                    exit_signal = True
                    exit_reason = 'emergency_stop'
                elif current_position == -1 and z_t > entry_z + 2.5:
                    exit_signal = True
                    exit_reason = 'emergency_stop'

                if exit_signal:
                    signals[i] = SignalType.CLOSE.value
                    current_position = 0

            # Track position
            positions[i] = current_position
            position_sizes[i] = current_position_size if current_position != 0 else 0

        # Create output dataframe
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
        
        # Add regime (pad for alignment)
        regimes = [HawkesRegime.NORMAL] + regimes
        signals_df['regime'] = regimes[:len(signals_df)]

        # Statistics
        n_long = (signals == SignalType.LONG.value).sum()
        n_short = (signals == SignalType.SHORT.value).sum()
        n_close = (signals == SignalType.CLOSE.value).sum()

        print(f"Generated {n_long + n_short + n_close} total signals:")
        print(f"    - Long Entries: {n_long}")
        print(f"    - Short Entries: {n_short}")
        print(f"    - Closes: {n_close}")
        
        if self.use_hawkes_regimes:
            print(f"\n  Hawkes Filtering Impact:")
            print(f"    Entries blocked by CRISIS regime: {self.entries_blocked_by_regime}")
            print(f"    Entries blocked by λ not decaying: {self.entries_blocked_by_decay}")
            print(f"    Exits triggered by regime escalation: {self.regime_exits}")

        self.signals = signals_df
        return signals_df

    def calculate_signal_quality(self, signals_df: pd.DataFrame) -> Dict:
        """Calculate signal quality metrics"""
        
        entries = signals_df[signals_df['signal'].isin([1, -1])]
        
        if len(entries) == 0:
            return {
                'n_entries': 0,
                'time_in_market': 0,
                'entry_discipline': 0
            }
        
        # Basic metrics
        total_obs = len(signals_df)
        time_in_market = (signals_df['position'] != 0).mean()
        
        # Entry quality: did we enter at extreme z-scores?
        avg_entry_z = entries['z_score'].abs().mean()
        entries_above_2std = (entries['z_score'].abs() > 2.0).mean()
        
        # Lambda discipline: did we avoid high-intensity entries?
        avg_entry_lambda = entries['lambda'].mean()
        lambda_90 = self.lambda_percentiles.get('p90', 0.20)
        low_lambda_entries = (entries['lambda'] < lambda_90).mean()
        
        metrics = {
            'n_entries': len(entries),
            'time_in_market': time_in_market,
            'entry_discipline': low_lambda_entries,
            'avg_entry_z_magnitude': avg_entry_z,
            'entries_above_2std': entries_above_2std,
            'avg_entry_lambda': avg_entry_lambda,
            'entries_blocked_regime': self.entries_blocked_by_regime,
            'entries_blocked_decay': self.entries_blocked_by_decay,
            'regime_exits': self.regime_exits
        }
        
        return metrics


# Legacy class name for backward compatibility
class AdaptiveSignalGenerator(TradingSignals):
    """Alias for backward compatibility"""
    pass


if __name__ == "__main__":
    print("="*70)
    print("OPTIMIZED SIGNAL GENERATION TEST")
    print("="*70)
    
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Mean-reverting spread with jumps
    spread = np.zeros(n)
    jumps = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.95 * spread[t-1] + np.random.normal(0, 0.1)
        if np.random.random() < 0.05:
            spread[t] += np.random.choice([-0.4, 0.4])
            jumps[t] = 1
    
    spread = pd.Series(spread, index=dates)
    jump_indicator = pd.Series(jumps, index=dates)
    
    # Synthetic Hawkes intensity
    lambda_intensity = pd.Series(0.05 * np.ones(n), index=dates)
    jump_times = np.where(jumps == 1)[0]
    for jt in jump_times:
        for i in range(jt, min(jt + 10, n)):
            lambda_intensity.iloc[i] += 0.4 * np.exp(-0.3 * (i - jt))
    
    print(f"\nTest data: {n} observations, {int(jumps.sum())} jumps")
    
    # Test with half-life awareness
    signal_gen = TradingSignals(
        z_entry_threshold=2.0,
        z_exit_threshold=0.5,
        use_hawkes_regimes=True
    )
    
    signals_df = signal_gen.generate_signals(
        spread, 
        lambda_intensity,
        jump_indicator=jump_indicator,
        half_life=25.0  # Provide half-life
    )
    
    quality = signal_gen.calculate_signal_quality(signals_df)
    print(f"\nSignal Quality:")
    for k, v in quality.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n Test complete!")