"""
Improved Configuration for Self-Exciting Pairs Trading

Key changes from original config:
1. Lower z_entry_threshold (1.5 instead of 2.0) for more entries
2. More reasonable lambda_threshold based on actual Hawkes intensity values
3. Longer max_holding_period to let trades develop
4. Removed aggressive exit conditions
"""

import numpy as np 
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

@dataclass
class DataConfig:
    """Data acquisition and processing params"""
    asset_a_symbol: str = "SPY"
    asset_b_symbol: str = "IVV"
    asset_a_csv: str = str(PROJECT_ROOT / "OHLCV_SPY.csv")
    asset_b_csv: str = str(PROJECT_ROOT / "OHLCV_IVV.csv")
    frequency: str = "1d"
    date_columns: str = "ts_event"
    hedge_ratio_method: str = 'cointegration'
    lookback_period: int = 30

@dataclass 
class JumpDetectionConfig:
    """Jump detection - slightly relaxed for more jump detection"""
    method: str = 'bipower_variation'
    window_size: int = 20
    significance_level: float = 0.05  # Relaxed from 0.01 to catch more jumps
    threshold_sigma: float = 3.0
    min_jump_size: float = 0.01

@dataclass
class HawkesConfig:
    """Hawkes process calibration"""
    kernel: str = "exponential"
    baseline_bounds: tuple = (0.001, 10.0)
    excitation_bounds: tuple = (0.1, 5.0)
    decay_bounds: tuple = (0.1, 10.0)
    estimation_method: str = "MLE"
    max_iterations: int = 1000
    tolerance: float = 1e-6
    test_fraction: float = 0.2

@dataclass
class MRJDConfig:
    """MRJD model - increased sigma bounds"""
    kappa_init: float = 0.5
    theta_init: float = 0.0
    sigma_init: float = 0.1
    jump_mean_init: float = 0.0
    jump_std_init: float = 0.05
    estimation_method: str = "MLE"
    dt: float = 1/252
    kappa_bounds: tuple = (0.01, 10.0)
    theta_bounds: tuple = (-10.0, 10.0)  # Wider for log-spread
    sigma_bounds: tuple = (0.001, 20.0)  # Increased from 5.0 to 20.0
    jump_mean_bounds: tuple = (-0.5, 0.5)
    jump_std_bounds: tuple = (0.001, 0.5)

@dataclass
class TradingConfigV2:
    """
    Improved trading parameters
    
    Key changes:
    - z_entry_threshold: 1.5 (was 2.0) - allows more entries
    - z_exit_threshold: 0.3 (was 0.5) - tighter exit for better profit capture
    - lambda_threshold: 0.3 (was 5.0/0.75) - based on actual Hawkes λ values (~0.01-0.1)
    - max_holding_period: 30 (was 25) - let trades develop
    - use_empirical_zscore: True - more stable than MRJD theoretical
    - use_jump_entries: True - jumps can trigger entries
    """
    # Entry conditions
    z_entry_threshold: float = 1.5  # Standard deviations for entry
    lambda_threshold: float = 0.3   # Based on actual Hawkes intensity scale
    
    # Position sizing
    max_position_size: float = 1.0
    scaling_constant: float = 0.1
    
    # Exit conditions  
    z_exit_threshold: float = 0.3  # Tighter for profit capture
    max_holding_period: int = 30   # Trading days
    stop_loss_sigma: float = 2.5   # Stop loss in std (wider to avoid whipsaws)
    profit_target_z: float = 0.5   # Take profit when z crosses this level
    
    # Risk management
    max_drawdown_threshold: float = 0.15
    position_limit: float = 2.0
    
    # New features
    use_empirical_zscore: bool = True   # Use rolling z-score instead of MRJD
    use_jump_entries: bool = True       # Allow jumps to trigger entries
    z_score_lookback: int = 60          # Lookback for empirical z-score
    
    # Removed: aggressive z-cross-zero exit

@dataclass
class BacktestConfig:
    """Backtesting params"""
    initial_capital: float = 1_000_000
    commission_rate: float = 0.0002
    slippage_bps: float = 1.0
    execution_delay: int = 1
    risk_free_rate: float = 0.02
    target_sharpe: float = 2.0
    regime_threshold: float = 0.3  # Based on actual lambda scale

@dataclass
class VisualizationConfig:
    """Visualization params"""
    figure_size: tuple = (14, 8)
    style: str = "seaborn-v0_8-darkgrid"
    long_color: str = "#2ecc71"
    short_color: str = "#e74c3c"
    neutral_color: str = "#95a5a6"
    jump_color: str = "#f39c12"
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300

class ConfigV2:
    """Improved master configuration"""
    def __init__(self):
        self.data = DataConfig() 
        self.jump_detection = JumpDetectionConfig() 
        self.hawkes = HawkesConfig() 
        self.mrjd = MRJDConfig() 
        self.trading = TradingConfigV2()  # Use V2
        self.backtest = BacktestConfig() 
        self.visualization = VisualizationConfig() 

    def to_dict(self) -> Dict:
        return {
            'data': self.data.__dict__,
            'jump_detection': self.jump_detection.__dict__,
            'hawkes': self.hawkes.__dict__,
            'mrjd': self.mrjd.__dict__,
            'trading': self.trading.__dict__,
            'backtest': self.backtest.__dict__,
            'visualization': self.visualization.__dict__
        }

# Default improved configuration
config_v2 = ConfigV2()


# Diagnostic function to help tune parameters
def diagnose_signal_generation(spread: pd.Series, 
                                lambda_intensity: pd.Series,
                                z_score: pd.Series,
                                config: Optional[TradingConfigV2] = None) -> Dict:
    """
    Diagnose why signals aren't being generated
    
    Returns dict with statistics about threshold crossings
    """
    import pandas as pd
    
    if config is None:
        config = TradingConfigV2()
    
    # Align data
    common_idx = spread.index.intersection(lambda_intensity.index).intersection(z_score.index)
    spread = spread.loc[common_idx]
    lambda_intensity = lambda_intensity.loc[common_idx]
    z_score = z_score.loc[common_idx]
    
    n = len(spread)
    
    # Count threshold crossings
    z_extreme_high = (z_score > config.z_entry_threshold).sum()
    z_extreme_low = (z_score < -config.z_entry_threshold).sum()
    lambda_safe = (lambda_intensity < config.lambda_threshold).sum()
    
    # Combined conditions
    long_opportunities = ((z_score < -config.z_entry_threshold) & 
                          (lambda_intensity < config.lambda_threshold)).sum()
    short_opportunities = ((z_score > config.z_entry_threshold) & 
                           (lambda_intensity < config.lambda_threshold)).sum()
    
    diagnostics = {
        'total_observations': n,
        'z_score_stats': {
            'mean': z_score.mean(),
            'std': z_score.std(),
            'min': z_score.min(),
            'max': z_score.max(),
            'pct_above_entry': 100 * z_extreme_high / n,
            'pct_below_neg_entry': 100 * z_extreme_low / n,
        },
        'lambda_stats': {
            'mean': lambda_intensity.mean(),
            'std': lambda_intensity.std(),
            'min': lambda_intensity.min(),
            'max': lambda_intensity.max(),
            'pct_below_threshold': 100 * lambda_safe / n,
        },
        'entry_opportunities': {
            'long_signals_possible': long_opportunities,
            'short_signals_possible': short_opportunities,
            'total_opportunities': long_opportunities + short_opportunities,
            'pct_of_data': 100 * (long_opportunities + short_opportunities) / n,
        },
        'recommended_thresholds': {
            'z_entry': z_score.std() * 1.5,  # 1.5 standard deviations
            'lambda_threshold': lambda_intensity.quantile(0.8),  # 80th percentile
        }
    }
    
    return diagnostics


def print_diagnostics(diagnostics: Dict):
    """Pretty print diagnostic results"""
    print("\n" + "="*70)
    print("SIGNAL GENERATION DIAGNOSTICS")
    print("="*70)
    
    print(f"\nTotal observations: {diagnostics['total_observations']}")
    
    print("\nZ-Score Statistics:")
    for k, v in diagnostics['z_score_stats'].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nLambda (Jump Intensity) Statistics:")
    for k, v in diagnostics['lambda_stats'].items():
        print(f"  {k}: {v:.6f}")
    
    print("\nEntry Opportunities:")
    for k, v in diagnostics['entry_opportunities'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    print("\nRecommended Thresholds (based on data):")
    for k, v in diagnostics['recommended_thresholds'].items():
        print(f"  {k}: {v:.4f}")
    
    print("="*70)