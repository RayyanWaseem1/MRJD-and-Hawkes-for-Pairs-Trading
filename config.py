"""
Configuration for Self-Exciting Pairs Trading

UPDATED VERSION:
- LOW TRANSACTION COSTS (2bp round trip - institutional level)
- Relaxed filtering defaults
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
    asset_a_symbol: str = "CVX"
    asset_b_symbol: str = "XOM"
    asset_a_csv: str = str(PROJECT_ROOT / "OHLCV_CVX.csv")
    asset_b_csv: str = str(PROJECT_ROOT / "OHLCV_XOM.csv")
    frequency: str = "1d"
    date_columns: str = "ts_event"
    hedge_ratio_method: str = 'cointegration'
    lookback_period: int = 30


@dataclass 
class JumpDetectionConfig:
    """Jump detection parameters"""
    method: str = 'bipower_variation'
    window_size: int = 20
    significance_level: float = 0.05
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
    """MRJD model parameters"""
    kappa_init: float = 0.5
    theta_init: float = 0.0
    sigma_init: float = 0.1
    jump_mean_init: float = 0.0
    jump_std_init: float = 0.05
    estimation_method: str = "MLE"
    dt: float = 1/252
    kappa_bounds: tuple = (0.01, 10.0)
    theta_bounds: tuple = (-10.0, 10.0)
    sigma_bounds: tuple = (0.001, 20.0)
    jump_mean_bounds: tuple = (-0.5, 0.5)
    jump_std_bounds: tuple = (0.001, 0.5)


@dataclass
class TradingConfig:
    """
    Trading parameters with RELAXED FILTERING
    
    Key changes from original:
    - min_lambda_decay_pct: 0.0 (DISABLED - was 0.15)
    - disable_crisis_block: True (ALLOW crisis entries)
    - skip_decay_in_calm: True (bypass decay check when calm)
    """
    # Entry conditions
    z_entry_threshold: float = 2.0
    lambda_threshold: float = 0.5
    
    # Position sizing
    max_position_size: float = 0.25
    scaling_constant: float = 0.1
    
    # Exit conditions  
    z_exit_threshold: float = 0.5
    max_holding_period: int = 30
    stop_loss_sigma: float = 2.5
    profit_target_z: float = 0.5
    
    # Risk management
    max_drawdown_threshold: float = 0.15
    position_limit: float = 2.0
    
    # RELAXED FILTERING OPTIONS (NEW)
    min_lambda_decay_pct: float = 0.0      # DISABLED (was 0.15)
    skip_decay_in_calm: bool = True        # Skip decay check in calm regimes
    adaptive_for_low_jumps: bool = True    # Auto-relax for low-jump pairs
    disable_crisis_block: bool = True      # ALLOW crisis entries (was False)
    
    # Features
    use_empirical_zscore: bool = True
    use_jump_entries: bool = True
    z_score_lookback: int = 60


@dataclass
class TrainValConfig:
    """Windows for train/validation split"""
    train_start: str = "2018-05-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2024-12-31"


@dataclass
class BacktestConfig:
    """
    Backtesting parameters with LOW TRANSACTION COSTS
    
    Cost scenarios:
    - ZERO:          commission=0,       slippage=0      -> 0bp RT
    - MINIMAL:       commission=0.00005, slippage=0      -> 1bp RT
    - LOW (DEFAULT): commission=0.00005, slippage=0.5   -> 2bp RT
    - MODERATE:      commission=0.0001,  slippage=0.5   -> 3bp RT
    - ORIGINAL:      commission=0.0002,  slippage=1.0   -> 6bp RT
    """
    initial_capital: float = 1_000_000
    
    # LOW TRANSACTION COSTS (institutional level)
    commission_rate: float = 0.00005     # 0.5bp per side (was 0.0002 = 2bp)
    slippage_bps: float = 0.5            # 0.5bp per side (was 1.0)
    # Total: ~2bp round trip (was ~6bp)
    
    execution_delay: int = 1
    risk_free_rate: float = 0.02
    target_sharpe: float = 2.0
    regime_threshold: float = 0.3
    benchmark_csv: str = str(PROJECT_ROOT / "OHLCV_SPY.csv")
    
    # Stop loss and profit target
    max_position_pct: float = 0.25
    stop_loss_pct: float = 0.03          # 3% hard stop
    trailing_stop_pct: float = 0.015     # 1.5% trailing
    trailing_activation_pct: float = 0.01
    profit_target_pct: float = 0.06


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
    """Master configuration - UPDATED with low costs and relaxed filtering"""
    def __init__(self):
        self.data = DataConfig() 
        self.jump_detection = JumpDetectionConfig() 
        self.hawkes = HawkesConfig() 
        self.mrjd = MRJDConfig() 
        self.trading = TradingConfig()
        self.backtest = BacktestConfig() 
        self.visualization = VisualizationConfig() 
        self.train_val = TrainValConfig()

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
    
    def print_cost_summary(self):
        """Print transaction cost summary"""
        commission_rt = self.backtest.commission_rate * 2 * 10000  # bp
        slippage_rt = self.backtest.slippage_bps * 2              # bp
        total_rt = commission_rt + slippage_rt
        
        print("\n" + "=" * 50)
        print("TRANSACTION COST CONFIGURATION")
        print("=" * 50)
        print(f"  Commission: {self.backtest.commission_rate*10000:.1f}bp per side")
        print(f"  Slippage:   {self.backtest.slippage_bps:.1f}bp per side")
        print(f"  TOTAL:      {total_rt:.1f}bp round trip")
        print("=" * 50)
    
    def print_filtering_summary(self):
        """Print filtering configuration summary"""
        print("\n" + "=" * 50)
        print("FILTERING CONFIGURATION")
        print("=" * 50)
        print(f"  Lambda decay required: {self.trading.min_lambda_decay_pct*100:.0f}%")
        print(f"  Skip decay in calm:    {self.trading.skip_decay_in_calm}")
        print(f"  Adaptive low-jump:     {self.trading.adaptive_for_low_jumps}")
        print(f"  Crisis block disabled: {self.trading.disable_crisis_block}")
        
        if self.trading.min_lambda_decay_pct == 0 and self.trading.disable_crisis_block:
            print("\n  MODE: MINIMAL FILTERING (maximum trades)")
        elif self.trading.min_lambda_decay_pct < 0.10:
            print("\n  MODE: MODERATE FILTERING")
        else:
            print("\n  MODE: AGGRESSIVE FILTERING")
        print("=" * 50)


# Preset configurations
def get_config_minimal_costs():
    """Zero transaction costs for gross edge analysis"""
    config = ConfigV2()
    config.backtest.commission_rate = 0.0
    config.backtest.slippage_bps = 0.0
    return config


def get_config_low_costs():
    """Low/institutional transaction costs (2bp RT) - DEFAULT"""
    config = ConfigV2()
    config.backtest.commission_rate = 0.00005  # 0.5bp/side
    config.backtest.slippage_bps = 0.5         # 0.5bp/side
    return config


def get_config_original_costs():
    """Original transaction costs (6bp RT) for comparison"""
    config = ConfigV2()
    config.backtest.commission_rate = 0.0002   # 2bp/side
    config.backtest.slippage_bps = 1.0         # 1bp/side
    return config


def diagnose_signal_generation(spread: pd.Series,
                                lambda_intensity: pd.Series,
                                z_score: pd.Series,
                                config: Optional[TradingConfig] = None) -> Dict:
    """
    Diagnose why signals aren't being generated.

    Returns statistics about threshold crossings and possible entries.
    """
    if config is None:
        config = TradingConfig()

    common_idx = spread.index.intersection(lambda_intensity.index).intersection(z_score.index)
    spread = spread.loc[common_idx]
    lambda_intensity = lambda_intensity.loc[common_idx]
    z_score = z_score.loc[common_idx]

    n = len(spread)
    if n == 0:
        return {
            'total_observations': 0,
            'z_score_stats': {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'pct_above_entry': 0.0,
                'pct_below_neg_entry': 0.0,
            },
            'lambda_stats': {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'pct_below_threshold': 0.0,
            },
            'entry_opportunities': {
                'long_signals_possible': 0,
                'short_signals_possible': 0,
                'total_opportunities': 0,
                'pct_of_data': 0.0,
            },
            'recommended_thresholds': {
                'z_entry': np.nan,
                'lambda_threshold': np.nan,
            }
        }

    z_extreme_high = (z_score > config.z_entry_threshold).sum()
    z_extreme_low = (z_score < -config.z_entry_threshold).sum()
    lambda_safe = (lambda_intensity < config.lambda_threshold).sum()

    long_opportunities = ((z_score < -config.z_entry_threshold) &
                          (lambda_intensity < config.lambda_threshold)).sum()
    short_opportunities = ((z_score > config.z_entry_threshold) &
                           (lambda_intensity < config.lambda_threshold)).sum()

    return {
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
            'z_entry': z_score.std() * 1.5,
            'lambda_threshold': lambda_intensity.quantile(0.8),
        }
    }


def print_diagnostics(diagnostics: Dict) -> None:
    """Pretty print signal generation diagnostic results."""
    print("\n" + "="*70)
    print("SIGNAL GENERATION DIAGNOSTICS")
    print("="*70)

    print(f"\nTotal observations: {diagnostics['total_observations']}")

    print("\nZ-Score Statistics:")
    for key, value in diagnostics['z_score_stats'].items():
        print(f"  {key}: {value:.4f}")

    print("\nLambda (Jump Intensity) Statistics:")
    for key, value in diagnostics['lambda_stats'].items():
        print(f"  {key}: {value:.6f}")

    print("\nEntry Opportunities:")
    for key, value in diagnostics['entry_opportunities'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nRecommended Thresholds (based on data):")
    for key, value in diagnostics['recommended_thresholds'].items():
        print(f"  {key}: {value:.4f}")

    print("="*70)


# Default configuration
config_v2 = ConfigV2()


if __name__ == "__main__":
    config = ConfigV2()
    config.print_cost_summary()
    config.print_filtering_summary()
