"""
Configuration for Self-Exciting Pairs 
"""

import numpy as np 
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

@dataclass
class DataConfig:
    """ Data acquisition and processing params"""
    #Equity pairs (NVDA/AMD - Energy Sector)
    asset_a_symbol: str = "NVDA" #Exxon Mobil
    asset_b_symbol: str = "AMD" #Chevron 

    #CSV file paths 
    asset_a_csv: str = str(PROJECT_ROOT / "OHLCV_NVDA.csv")
    asset_b_csv: str = str(PROJECT_ROOT / "OHLCV_AMD.csv")

    #Data params
    frequency: str = "1d" #Daily data
    date_columns: str = "ts_event" #column name for dates in the csv

    #Spread construction
    hedge_ratio_method: str = 'cointegration' # or regression
    lookback_period: int = 30 #Days for hedge ratio estimation. Changed to 30 from 60 to be more adaptive

@dataclass 
class JumpDetectionConfig:
    """ Jump detection algorithm params"""
    #Detection method
    method: str = 'bipower_variation' #or 'threshold', 'lee_mykland'

    #Bipower variation params (for daily data)
    window_size: int = 20 #1 month of trading days
    significance_level: float = 0.05 #Catches 2sigma+ jumps compared to 0.01 which only catches 3sigma+ jumps. Relaxes the jump detection parameters from before. 

    #Thrshold based detection
    threshold_sigma: float = 3.0 #standard deviations

    #Jump size filtering
    min_jump_size: float = 0.01 #Minimum relative jump size 

@dataclass
class HawkesConfig:
    """ Hawkes process calibration params"""
    #Model specifications
    kernel: str = "exponential" #decay kernel

    #Parameter bounds
    baseline_bounds: tuple = (0.001, 10.0) #λ_bar
    excitation_bounds: tuple = (0.1, 5.0) # α
    decay_bounds: tuple = (0.1, 10.0) # β_H

    #Estimation
    estimation_method: str = "MLE" #or "GMM"
    max_iterations: int = 1000
    tolerance: float = 1e-6

    #Validation
    test_fraction: float = 0.2

@dataclass
class MRJDConfig:
    """ Mean Reverting Jump Diffusion model params"""
    #OU params
    kappa_init: float = 0.5 #Mean reversion speed
    theta_init: float = 0.0 #Long-run equilibrium
    sigma_init: float = 0.1 #Diffusive volatility

    #Jump parameters
    jump_mean_init: float = 0.0
    jump_std_init: float = 0.05 

    #Estimation
    estimation_method: str = "MLE"
    dt: float = 1/252 #Daily observations (252 trading days per year)

    #Parameter bounds
    kappa_bounds: tuple = (0.01, 10.0)
    theta_bounds: tuple = (-1.0, 1.0)
    sigma_bounds: tuple = (0.001, 1.0)
    jump_mean_bounds: tuple = (-0.5, 0.5)
    jump_std_bounds: tuple = (0.001, 0.5)

@dataclass
class TradingConfig:
    """ Trading signal generation params"""
    #Entry conditions (for daily equity data)
    z_entry_threshold: float = 0.40 #Spread z-score threshold. 0.6 allows for more frequent entries compared to previous 2.0
    lambda_threshold: float = 0.75 #need to filter adequately. Previous value of 5.0 was way too much, didn't filter anything

    #Position sizing
    max_position_size: float = 1.0
    scaling_constant: float = 0.1 # c in w_t ∝ Z_t / (1 + c*λ_t)

    #Exit conditions
    z_exit_threshold: float = 0.3
    max_holding_period: int = 50 #trading days. Previously was 20, but that was too short. Want to match 1.5 x half-life roughly
    stop_loss_sigma: float = 3.0 #stop loss in std

    #Risk management
    max_drawdown_threshold: float = 0.15 #15% max drawdown
    position_limit: float = 2.0 #maximum leverage 

@dataclass
class BacktestConfig:
    """ Backtesting params"""
    #Initial conditions
    initial_capital: float = 1_000_000

    #Transaction costs
    commission_rate: float = 0.0002 #2 bps
    slippage_bps: float = 1.0

    #Execution
    execution_delay: int = 1 #bars delay for execution 

    #Performance metrics
    risk_free_rate: float = 0.02
    target_sharpe: float = 2.0

    #Regime analysis
    regime_threshold: float = 0.7 #jump intensity threshold for regime classifications

@dataclass
class VisualizationConfig:
    """ Visualization and reporting params"""
    #Plot settings
    figure_size: tuple = (14, 8)
    style: str = "seaborn-v0_8-darkgrid"

    #Colors
    long_color: str = "#2ecc71"
    short_color: str = "#e74c3c"
    neutral_color: str = "#95a5a6"
    jump_color: str = "#f39c12"

    ##xport
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300

class Config:
    """Master configuration class"""
    def __init__(self):
        self.data = DataConfig() 
        self.jump_detection = JumpDetectionConfig() 
        self.hawkes = HawkesConfig() 
        self.mrjd = MRJDConfig() 
        self.trading = TradingConfig() 
        self.backtest = BacktestConfig() 
        self.visualization = VisualizationConfig() 

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'data': self.data.__dict__,
            'jump_detection': self.jump_detection.__dict__,
            'hawkes': self.hawkes.__dict__,
            'mrjd': self.mrjd.__dict__,
            'trading': self.trading.__dict__,
            'backtest': self.backtest.__dict__,
            'visualization': self.visualization.__dict__
        }
    
#Default configuration instance
config = Config() 
