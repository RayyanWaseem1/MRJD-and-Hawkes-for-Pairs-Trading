""" Configuration for the self-exciting pairs trading study"""

from __future__ import annotations

from dataclasses import dataclass, field 
from pathlib import Path 
from typing import Dict, Optional, Tuple 

PROJECT_ROOT = Path(__file__).resolve().parent


# Pair registry -- the CLI iterates over these, so one command regenerates
# every artifact. Previously reproducing five pairs meant hand-editing 
# DataConfig five times and moving files by hand 

PAIRS: Dict[str, Tuple[str, str]] = {
    "CVX_XOM": ("CVX", "XOM"),
    "AMD_NVDA": ("AMD", "NVDA"),
    "SPY_IVV": ("SPY", "IVV"),
    "GS_MS": ("GS", "MS"),
    "GLD_GDX": ("GLD", "GDX"),
}

@dataclass
class DataConfig:
    """ Data loading and spread construction"""

    asset_a_symbol: str = "CVX"
    asset_b_symbol: str = "XOM"
    asset_a_csv: str = str(PROJECT_ROOT / "OHLCV_CVX.csv")
    asset_b_csv: str = str(PROJECT_ROOT / "OHLCV_XOM.csv")
    date_columns: str = "ts_event"

    # Hedge-ratio estimator: 'johansen' | 'engle_granger' | 'ols' | 'tls'
    hedge_ratio_method: str = "johansen"

    # 'static' (primary) | 'periodic' | 'rolling' (robustness only)
    hedge_mode: str = "static"

    # Rolling-mode window. Only used when hedge_mode == 'rolling'.
    lookback_period: int = 30 

    # Adjust for the corporate actions in corporate_actions.SPLIT_TABLE
    adjust_splits: bool = True 

    # Flag (never delete) single-day moves above this size 
    large_move_flag_threshold: float = 0.25

    def apply_pair(self, key: str) -> None:
        """ Point this config at one of the registered pairs"""
        if key not in PAIRS:
            raise KeyError(f"Unknown pair '{key}'. Known pairs: {sorted(PAIRS)}")
        a, b = PAIRS[key]
        self.asset_a_symbol, self.asset_b_symbol = a, b
        self.asset_a_csv = str(PROJECT_ROOT / f"OHLCV_{a}.csv")
        self.asset_b_csv = str(PROJECT_ROOT / f"OHLCV_{b}.csv")

@dataclass 
class JumpDetectionConfig:
    """ Jump detection."""

    # 'lee_mykland' (primary) | 'bipower' (robustness) | 'threshold'
    method: str = "lee_mykland"

    # local bipower volatility window, in trading days
    window_size: int = 20

    significance_level: float = 0.05 

    # Benjamini_Hochberg FDR control across all tested observations
    # At nominal alpha = 0.05 over ~1,950 tests, ~98 false positives are
    # expected by chance, so this is on by default 
    apply_fdr: bool = True 

    # Detection basis used to FIT the Hawkes layer when FDR leaves too few 
    # events. Recorded in the artifacts; never silently substituted
    min_jumps_for_hawkes: int = 10 
    fallback_to_nominal_for_hawkes: bool = True 

    # k-sigma for the naive detector in the comparison table 
    threshold_sigma: float = 4.0

@dataclass
class HawkesConfig: 
    """ Hawkes calibration. Every field here is now read by the estimator """

    kernel: str = "exponential"
    baseline_bounds: Tuple[float, float] = (1e-6, 10.0)
    excitation_bounds: Tuple[float, float] = (1e-6, 5.0)
    decay_bounds: Tuple[float, float] = (1e-4, 10.0)
    estimation_method: str = "MLE"
    max_iterations: int = 1000
    tolerance: float = 1e-8
    n_restarts: int = 4

    # Parametric bootstrap replications for the LR test against Poisson 
    # Needed because beta is unidentified under H0 (the Davies problem), so 
    # the LR statistic is not asymptotically chi-squared. 0 disables it
    lr_bootstrap_reps: int = 200 

@dataclass
class MRJDConfig:
    """ MRJD estimation. dt is in TRADING DAYS and must stay 1.0"""

    # 1.0 == one trading day. A dt of 1/252 makes kappa per-YEAR while
    # half-lives and holding periods are in days -- the factor-of-252 bug
    dt: float = 1.0

    estimation_method: str = "MLE"

    # Run the joint MLE over (kappa, theta, sigma, mu_J, sigma_J)
    joint_refinement: bool = False 

    # Raise (rather than only report) when the model and empirical half-lives
    # disagree by more than 50%
    raise_on_half_life_mismatch: bool = False

@dataclass
class TradingConfig:
    """ Signal generation """

    z_entry_threshold: float = 2.0
    z_exit_threshold: float = 0.5

    # 'empirical' (rolling z-score) | 'mrjd' (OU stationary z-score)
    z_score_basis: str = "empirical"
    z_score_lookback: int = 60

    max_position_size: float = 0.25 
    min_position_size: float = 0.10 

    # Hawkes regime machinery. When False the generator is the CONTROL ARM
    # a plain z-score strategy with no Hawkes layer at all 
    use_hawkes_regimes: bool = True
    use_jump_entries: bool = True

    # Block new entries when the training-window pair validation fails
    # (`is_tradeable` is False). Applies to both arms. False reproduces the
    # ungated behaviour, where failing pairs were traded anyway
    require_tradeable: bool = True

    # Regime cut-points on EXCESS intensity (lambda_t - lambda_bar) / lambda_bar
    # Percentile bucketing was degenerate: lambda(t) >= lambda_bar always, so 
    # p25 equalled lambda_bar exactly on two pairs and CALM never fired 
    regime_excess_calm: float = 0.05
    regime_excess_elevated: float = 1.0 
    regime_excess_crisis: float = 5.0 

    # only enter once a jump cascade is subsiding 
    lambda_decay_lookback: int = 5
    min_lambda_decay_pct: float = 0.15

    # Holding period as multiples of the spread half-life 
    min_hold_fraction: float = 0.5
    target_hold_fraction: float = 0.8
    max_hold_fraction: float = 1.5
    max_holding_period_cap: int = 120

    # Use the MRJD conditional distribution to set the holding period 
    use_mrjd_holding_period: bool = False 

    # Emergency exit when z moves this far against the entry level 
    emergency_z_move: float = 2.5 

@dataclass
class BacktestConfig:
    """ Backtest execution, costs, and reporting """

    initial_capital: float = 1_000_000.0

    # Fraction of notional per side: 0.0002 == 2bp. This was 0.002 (20bp)
    # under a "2bp" comment, which made every round trip cost 42bp, not 6bp
    commission_rate: float = 0.0002 #2bp per side
    slippage_bps: float = 1.0 #1bp per side 

    # Bars between signal generation and execution. Read by the engine
    execution_delay: int = 1 

    # 'next_open' | 'next_close'
    execution_price: str = "next_open"

    risk_free_rate: float = 0.02 

    # Credit idle cash at the risk free rate. Without this the engine charges
    # rf/252 every day while ~90% of the book sits in uncredited cash, which
    # mechanically produces an "alpha" of -rf for every pair 
    credit_idle_cash: bool = True 

    # Financing and borrow, annualized
    long_financing_rate: float = 0.02 
    short_rebate_rate: float = 0.015
    default_borrow_rate: float = 0.0030 
    borrow_rates: Dict[str, float] = field(
        default_factory = lambda: {
            "CVX": 0.0025, "XOM": 0.0025,
            "GS": 0.0030, "MS": 0.0030,
            "SPY": 0.0020, "IVV": 0.0025,
            "AMD": 0.0075, "NVDA": 0.0050,
            "GLD": 0.0030, "GDX": 0.0100,
        }
    )

    # Risk limits. 
    max_position_pct: float = 0.25 

    #: Stop sizing. 'fixed' uses the *_pct values below as fractions of gross
    #: notional. 'volatility' scales them by the spread's own daily standard
    #: deviation, which is the default because fixed percentages are in
    #: structural conflict with mean reversion:
    #:
    #:   a spread with a 50-70 day half-life routinely moves several percent
    #:   AGAINST the position before reverting -- that is what mean reversion
    #:   IS -- so a fixed 3% stop exits at systematically the worst moment.
    #:   Measured on the training window, the shipped 3% stop caused 85-100%
    #:   of trades to exit via stop at an average of 1.7-12 days against a
    #:   design intending 25-107, and removing stops entirely improved the
    #:   Sharpe on all five pairs (see diagnostics.stop_sensitivity).
    #:
    #: The multipliers are in units of the spread's STATIONARY standard
    #: deviation -- the strategy's own state variable -- not daily sigma.
    #: Daily sigma is the wrong scale: at a 60-day half-life, 6 daily sigma is
    #: still only ~3% and reproduces the defect.
    #:
    #: Entry is at z = 2. A stop at 4 stationary sigma therefore triggers only
    #: after the spread has moved a further 2 sigma against the position, which
    #: is a genuine tail event rather than ordinary mean-reverting behaviour.
    #: Normal risk control is the z-based emergency exit in signal_generation
    #: (entry_z +/- emergency_z_move), expressed in the same units; this price
     #: stop is a catastrophe backstop for a cointegration break.

    stop_mode: str = "volatility"
    stop_loss_sigma: float = 4.0
    trailing_stop_sigma: float = 3.0
    trailing_activation_sigma: float = 1.5
    profit_target_sigma: float = 5.0 

    # Used only when stop_mode == 'fixed.' Retained so the shipped 
    # configuration remains reproducible for comparison 
    stop_loss_pct: float = 0.03
    trailing_stop_pct: float = 0.015
    trailing_activation_pct: float = 0.01
    profit_target_pct: float = 0.06

    # Hard floor and cap on the resulting stop, as a fraction of notioanl
    # so a degenerate volatility estimate cannot produce an absurd level
    stop_floor_pct: float = 0.01
    stop_cap_pct: float = 0.50

    # check stops against the intraday High/Low rather than the close only 
    use_intraday_stops: bool = True 

    # Entry-lambda split for the regime performance breakdown
    regime_threshold: float = 0.3

    # Report metrics on capital at risk in addition to notional
    report_capital_at_risk: bool = True 
    # Optional volatility target for a scaled reporting variant. None disables
    target_volatility: Optional[float] = 0.10 

    benchmark_csv: str = str(PROJECT_ROOT / "OHLCV_SPY.csv")

@dataclass
class TrainValConfig:
    """ In-sample / out-of-sample split """

    train_start: str = "2018-05-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2024-12-31"

@dataclass
class WalkForwardConfig:
    """ Quarterly walk-forward"""

    min_train_days: int = 504

    # Grid-search entry/exit thresholds on each quarter's TRAINING window and
    # apply the winner to the next quarter. 
    tune_thresholds: bool = True 
    z_entry_grid: Tuple[float, ...] = (1.5, 2.0, 2.5)
    z_exit_grid: Tuple[float, ...] = (0.25, 0.5, 0.75)

    # Re-estimate the hedge ratio at each quarter boundary and hold it fixed
    # within the quarter 
    reestimate_hedge_each_quarter: bool = True

@dataclass
class StatisticsConfig:
    """ Inference settings"""

    bootstrap_reps: int = 1000
    bootstrap_mean_block: float = 20.0
    power_target_effect: float = 0.01
    seed: int = 42

@dataclass
class VisualizationConfig:
    figure_size: Tuple[float, float] = (14.0, 8.0)
    save_plots: bool = True 
    plot_format: str = "png"
    dpi: int = 150

class Config:
    """ Master configuration"""

    def __init__(self) -> None:
        self.data = DataConfig() 
        self.jump_detection = JumpDetectionConfig()
        self.hawkes = HawkesConfig()
        self.mrjd = MRJDConfig()
        self.trading = TradingConfig()
        self.backtest = BacktestConfig()
        self.train_val = TrainValConfig()
        self.walk_forward = WalkForwardConfig()
        self.statistics = StatisticsConfig()
        self.visualization = VisualizationConfig()

    def for_pair(self, key: str) -> "Config":
        self.data.apply_pair(key)
        return self 

    def as_control_arm(self) -> "Config":
        """ 
        Strip the Hawkes layer: plain rolling z-score, no regimes, no jump 
        entries, no lambda-decay filter 

        This is the CONTROL. Without it "hawkes doesn't add alpha" is not a 
        measurable claim, because there is nothing to compare against
        """

        self.trading.use_hawkes_regimes = False 
        self.trading.use_jump_entries = False 
        self.trading.z_score_basis = "empirical"
        return self 

    def to_dict(self) -> Dict:
        return {
            "data": dict(self.data.__dict__),
            "jump_detection": dict(self.jump_detection.__dict__),
            "hawkes": dict(self.hawkes.__dict__),
            "mrjd": dict(self.mrjd.__dict__),
            "trading": dict(self.trading.__dict__),
            "backtest": dict(self.backtest.__dict__),
            "train_val": dict(self.train_val.__dict__),
            "walk_forward": dict(self.walk_forward.__dict__),
            "statistics": dict(self.statistics.__dict__),
            "visualization": dict(self.visualization.__dict__),
        }

def set_seeds(seed: int = 42) -> None:
    """ Seed every stochastic path in the project."""
    import random 

    import numpy as np 

    random.seed(seed)
    np.random.seed(seed)
