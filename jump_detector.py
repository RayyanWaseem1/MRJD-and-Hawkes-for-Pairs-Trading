"""
Jump Detection for Calendar Spreads
Implements multiple jump detection algorithms:
    - Bipower Variation
    - Lee-Mykland Test
    - Threshold-based Detection
"""

import numpy as np 
import pandas as pd
from scipy import stats
from scipy.special import gamma as gamma_func
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class JumpDetector:
    """
    Detects jumps in spread time series through various methods
    """

    def __init__(self, significance_level: float = 0.01):
        """
        Initializing the jump detector 

        Params:
        - Significance_level: float
            - significance level for statistical tests
        """

        self.significance_level = significance_level
        self.jumps = None 
        self.jump_stats = {}

    def detect_jumps_bipower_variation(self, 
                                       returns: pd.Series,
                                       window: int = 78) -> pd.DataFrame:
        """
        Detecting jumps using Bipower Variation 

        This compares realized variance to bipower variation. 
        Under no jumps, they should be similar. Larger differences indicate jumps 

        Params:
        - returns: pd.Series
            - price returns
        - window: int
            - window size for computing statistics (78 for 5-min data in a day)

        Returns:
        - pd.DataFrame
            - DataFrame with jump indicators and statistics
        """

        print("Detecting jumps using Bipower Variation")

        n = len(returns)

        #Realized variance (sum of squared returns)
        RV = returns.rolling(window = window).apply(lambda x: np.sum(x**2), raw = True)

        #Bipower variation
        abs_returns = returns.abs() 
        BV = (np.pi / 2) * abs_returns.rolling(window = window).apply(
            lambda x: np.sum(x[:-1] * x[1:]), raw = True
        )

        #Tripower quarticity (for variance of test statistic)
        mu_1 = 2**(1/2) * gamma_func(1) / gamma_func(0.5) #E[|Z|]
        TP = n * (mu_1**(-3)) * abs_returns.rolling(window = window).apply(
            lambda x: np.sum(x[:-2]**(4/3) * x[1:-1]**(4/3) * x[2:]**(4/3)), raw = True
        )

        #Z-statistic for jump test
        max_ratio = np.maximum(1, TP / (BV**2))
        z_stat = (RV - BV) / np.sqrt(max_ratio * BV**2 / window)

        #critical value from standard normal
        critical_value = stats.norm.ppf(1 - self.significance_level)

        #jump indicator
        jump_indicator = (z_stat > critical_value).astype(int)

        #Jump contribution to variance
        jump_variation = np.maximum(0, RV - BV)

        result = pd.DataFrame({
            'returns': returns,
            'RV': RV,
            'BV': BV,
            'z_statistic': z_stat,
            'jump_indicator': jump_indicator,
            'jump_variation': jump_variation,
            'continous_variation': BV
        }, index = returns.index)

        n_jumps = jump_indicator.sum()
        print(f"Detected {n_jumps} jumps ({100 * n_jumps / n:.2f}% of observations)")

        self.jumps = result
        return result 
    
    def detect_jumps_lee_mykland(self,
                                 prices: pd.Series,
                                 window: int = 78) -> pd.DataFrame:
        """
        Lee-Mykland jump test
        Tests if log returns standarddized by local volatility exceed a certain threshold 

        Params:
        - prices: pd.Series
            - price series (not returns)
        - window: int
            - window for local volatility estimation 

        Returns:
        - pd.DataFrame
            - Jump detection results
        """

        print("Detecting jumps using Lee-Mykland Test")

        #log returns
        log_returns = pd.Series(
            np.log(prices / prices.shift(1)),
            index=prices.index,
            dtype=float
        )

        #Bipower variation for local volatility
        abs_returns = log_returns.abs() 
        sigma_t = np.sqrt(
            (np.pi / 2) * abs_returns.rolling(window = window).apply(
                lambda x: np.sum(x[:-1] * x[1:]) / (window - 1), raw = True
            )
        )

        #Test Statistics
        L = log_returns.abs() / sigma_t 

        #Critical values (Lee & Mykland 2008)
        n = window 
        c_n = (2 * np.log(n)) ** 0.5
        S_n = c_n - np.log(np.pi * np.log(n)) / (2 * c_n)

        beta = -np.log(-np.log(1 - self.significance_level)) #from Gumbel distribution
        threshold = (beta + S_n) / c_n 

        #jump indicator
        jump_indicator = (L > threshold).astype(int) 

        #jump size
        jump_size = log_returns * jump_indicator 

        result = pd.DataFrame({
            'prices': prices,
            'log_returns': log_returns,
            'local_volatility': sigma_t,
            'L_statistic': L,
            'threshold': threshold,
            'jump_indicator': jump_indicator,
            'jump_size': jump_size
        }, index = prices.index)

        n_jumps = jump_indicator.sum()
        print(f" Detected {n_jumps} jumps ({100 * n_jumps / len(prices):.2f}% of observations)")

        return result 
    
    def detect_jumps_threshold(self,
                               returns: pd.Series,
                               threshold_sigma: float = 4.0,
                               window: int = 60) -> pd.DataFrame:
        
        """
        Simple threshold-based jump detection

        Identifies returns that exceed threshold * a rolling standard deviation

        Params:
        - returns: pd.Series
            - Return series
        - threshold_sigma: float
            - number of standard deviations for threshold
        - window: int
            - window for computing rolling statistic

        Returns:
        - pd.DataFrame
            - jump detection results
        """

        print(f"Detecting jumps using {threshold_sigma} - sigma threshold")

        #Rolling statistics
        rolling_mean = returns.rolling(window = window).mean()
        rolling_std = returns.rolling(window = window).std() 

        #Standardized returns
        z_score = (returns - rolling_mean) / rolling_std 

        #jump indicator
        jump_indicator = (z_score.abs() > threshold_sigma).astype(int)

        #jump size
        jump_size = returns * jump_indicator 

        result = pd.DataFrame({
            'returns': returns,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'z_score': z_score,
            'jump_indicator': jump_indicator,
            'jump_size': jump_size
        }, index = returns.index)

        n_jumps = jump_indicator.sum()
        print(f" Detected {n_jumps} jumps ({100 * n_jumps / len(returns):.2f}% of observations)")

        return result 
    
    def extract_jump_times(self, jump_df: pd.DataFrame) -> np.ndarray:
        """
        Extract jump times from jump detection results

        Params:
        - jump_df: pd.DataFrame
            - Jump detection results with 'jump_indicator' column

        Returns:
        - np.ndarray
            - Array of jump times (as numeric timestamps)
        """

        jump_mask = jump_df['jump_indicator'] == 1
        jump_positions = np.flatnonzero(jump_mask.to_numpy())

        if len(jump_positions) == 0:
            return np.array([], dtype=float)

        index = jump_df.index

        #Convert to numeric time (days since start)
        if isinstance(index, pd.DatetimeIndex):
            start_time = index[0]
            jump_times = index[jump_positions]
            jump_times_numeric = (jump_times - start_time).total_seconds() / (24 * 3600)
        else:
            index_values = index.to_numpy()
            if np.issubdtype(index_values.dtype, np.number):
                jump_times_numeric = index_values[jump_positions].astype(float) - float(index_values[0])
            else:
                jump_times_numeric = jump_positions.astype(float)
        
        return np.asarray(jump_times_numeric, dtype=float)
    
    def extract_jump_sizes(self, jump_df: pd.DataFrame) -> np.ndarray:
        """
        Extract jump sizes from jump detection results 

        Params:
        - jump_df: pd.DataFrame
            - jump detection results 

        Returns:
        - np.ndarray
            - Array of jump sizes
        """

        if 'jump_size' in jump_df.columns:
            jump_sizes = jump_df.loc[jump_df['jump_indicator'] == 1, 'jump_size']
        elif 'jump_variation' in jump_df.columns:
            jump_sizes = jump_df.loc[jump_df['jump_indicator'] == 1, 'jump_variation']
        else:
            raise ValueError("No jump size information in this dataframe")
        
        return np.asarray(jump_sizes, dtype=float)
    
    def calculate_jump_statistics(self, jump_df: pd.DataFrame) -> Dict:
        """
        Calculating summary statistics for detected jumps

        Params:
        - jump_df: pd.DataFrame
            - jump detection results 

        Returns:
        - Dict
            - Dictionary of jump statistics
        """

        jump_indicator = jump_df['jump_indicator']
        n_jumps = jump_indicator.sum()
        n_obs = len(jump_df)

        #jump frequency
        jump_freq = n_jumps / n_obs

        #Average time between jumps
        jump_times = self.extract_jump_times(jump_df)
        if len(jump_times) > 1:
            inter_jump_times = np.diff(jump_times)
            mean_inter_jump = inter_jump_times.mean()
            std_inter_jump = inter_jump_times.std()
        else:
            mean_inter_jump = np.nan
            std_inter_jump = np.nan 

        #Jump sizes
        jump_sizes = self.extract_jump_sizes(jump_df)
        if len(jump_sizes) > 0:
            mean_jump_size = jump_sizes.mean()
            std_jump_size = jump_sizes.std()
            max_jump_size = jump_sizes.max() 
        else:
            mean_jump_size = np.nan
            std_jump_size = np.nan
            max_jump_size = np.nan 

        #Clustering coefficient (proportion of jumps followed by another jump within 5 periods)
        clustering_window = 5
        clustering_count = 0
        jump_indices = np.where(jump_indicator == 1)[0]

        for idx in jump_indices:
            if idx + clustering_window < len(jump_indicator):
                if jump_indicator.iloc[idx + 1: idx + clustering_window + 1].sum() > 0:
                    clustering_count += 1

        clustering_coef = clustering_count / n_jumps if n_jumps > 0 else 0

        stats = {
            'n_jumps': int(n_jumps),
            'n_observations': n_obs,
            'jump_frequency': jump_freq,
            'mean_inter_jump_time': mean_inter_jump,
            'std_inter_jump_time': std_inter_jump,
            'mean_jump_size': mean_jump_size,
            'std_jump_size': std_jump_size,
            'max_jump_size': max_jump_size,
            'clustering_coefficient': clustering_coef
        }

        self.jump_stats = stats
        return stats
    
    def plot_jumps(self, spread_df: pd.DataFrame, jump_df: pd.DataFrame,
                  save_path: Optional[str] = None): 
        """
        Visualizing the detected jumps on a spread series

        Params:
        - spread_df: pd.DataFrame
            - spread data
        - jump_df: pd.DataFrame
            - jump detection results with 'jump_indicator' column
        - save_path: str, optional
            - path to save the plot (if None, just show it)
        """

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize = (14,10))

        #Plot 1: Spread with jump markers
        axes[0].plot(spread_df.index, spread_df['spread'],
        label = 'Spread', color = 'blue', alpha = 0.7, linewidth = 0.8)

        jump_times = jump_df.index[jump_df['jump_indicator'] == 1]
        jump_values = spread_df.loc[jump_times, 'spread']

        axes[0].scatter(jump_times, jump_values,
                    color = 'red', s = 50, marker = 'x',
                    label = f"Jumps (n = {len(jump_times)})", zorder=5)
    
        axes[0].set_ylabel('Spread Value')
        axes[0].set_title('Calendar Spread with Detected Jumps')
        axes[0].legend()
        axes[0].grid(True, alpha = 0.3)

        #Plot 2: Jump intensity over time
        if 'z_statistic' in jump_df.columns:
            axes[1].plot(jump_df.index, jump_df['z_statistic'],
                     label = 'Z-statistic', color = 'green', alpha = 0.7, linewidth = 0.8)
            axes[1].axhline(y = stats.norm.ppf(1 - self.significance_level),
                        color = 'red', linestyle = '--', label = 'Critical Value')
            axes[1].set_ylabel('Z-statistic')
            axes[1].set_title('Jump Test Statistic')
            axes[1].legend()
            axes[1].grid(True, alpha = 0.3)

        #Plot 3: Cumulative jumps
        cumulative_jumps = jump_df['jump_indicator'].cumsum()
        axes[2].plot(jump_df.index, cumulative_jumps,
                 color = 'purple', linewidth = 1.5)
        axes[2].set_ylabel('Cumulative Jump Count')
        axes[2].set_xlabel('Date')
        axes[2].set_title('Cumulative Jump Count Over Time')
        axes[2].grid(True, alpha = 0.3)

        plt.tight_layout() 

        if save_path:
            plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
            print(f" Plot saved to {save_path}")

        plt.show() 

if __name__ == "__main__":
    """
    Test jump detection on XOM/CVX equity pairs
    """

    import sys 
    import os

    #Adding parent directory to path for imports 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

    from equity_pairs_loader import EquityPairsDataPipeline
    import matplotlib.pyplot as plt 

    print("=" * 70)
    print("Jump Detection Test - XOM/CVX Equity Paris")
    print("=" * 70)

    #Initializing data pipeline
    print("\n [1] Loading XOM/CVX data from CSV files")
    pipeline = EquityPairsDataPipeline()

    #Try to load CSV files from multiple possible locations
    csv_paths = [
        (os.path.join(current_dir, 'OHLCV_XOM.csv'),
         os.path.join(current_dir, 'OHLCV_CVX.csv')), #script directory
        ('OHLCV_XOM.csv', 'OHLCV_CVX.csv'), #current dir
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
        print("Could not find CSV files")
        print("\n Searched in:")
        for xom_path, cvx_path in csv_paths:
            print(f" - {os.path.abspath(xom_path)}")
        sys.exit(1)

    #Clean and construct spread
    print("\n [2] Constructing spread")
    cleaned_data = pipeline.clean_data()
    spread_df = pipeline.construct_spread(method = 'cointegration', lookback = 60)

    #Calculate returns for jump detection 
    print("\n [3] Calculating returns for jump detection")
    returns = spread_df['spread'].pct_change().dropna()
    print(f" Calculated {len(returns)} returns")
    print(f" Mean: {returns.mean():.6f}")
    print(f" Std: {returns.std():.6f}")
    print(" Min: {:.6f}".format(returns.min()))
    print(" Max: {:.6f}".format(returns.max()))

    #Initialize jump detector
    print("\n [4] Initializing jump detector")
    detector = JumpDetector(significance_level = 0.01)

    #Test 1: Bipower Variation
    print("\n" + "=" * 70)
    print("Test 1: Bipower Variation Method")
    print("=" * 70)

    bpv_jumps = detector.detect_jumps_bipower_variation(returns, window = 20)
    bpv_stats = detector.calculate_jump_statistics(bpv_jumps)

    print(f"\n Bipower Variation Results:")
    print(f" Total Jumps: {bpv_stats['n_jumps']}")
    print(f" Jump Frequency: {bpv_stats['jump_frequency']:.4f} ({bpv_stats['jump_frequency']*100:2f}%)")
    print(f" Mean Inter-jump Time: {bpv_stats['mean_inter_jump_time']:.2f} days")
    print(f" Clustering coefficient: {bpv_stats['clustering_coefficient']:.4f}")

    if bpv_stats['n_jumps'] > 0:
        print(f" Avg Jump Size: {bpv_stats['mean_jump_size']:.6f}")
        print(f" Jump Size Std: {bpv_stats['std_jump_size']:.6f}")

    #Test 2: Lee - Mykland Test
    print("\n" + "=" * 70)
    print("Test 2: Lee-Mykland Test")
    print("=" * 70)

    lm_jumps = detector.detect_jumps_lee_mykland(returns, window = 20)
    lm_stats = detector.calculate_jump_statistics(lm_jumps)

    print(f"\n Lee-Mykland Results:")
    print(f" Total Jumps: {lm_stats['n_jumps']}")
    print(f" Jump Frequency: {lm_stats['jump_frequency']:.4f} ({lm_stats['jump_frequency'] * 100:.2f}%)")
    print(f" Mean Inter-jump Time: {lm_stats['mean_inter_jump_time']:.2f} days")
    print(f" Clustering coefficient: {lm_stats['clustering_coefficient']:.4f}")

    if lm_stats['n_jumps'] > 0:
        print(f" Avg Jump Size: {lm_stats['mean_jump_size']:.6f}")
        print(f" Jump Size Std: {lm_stats['std_jump_size']:.6f}")

    #Test 3: Threshold-Based Detection
    print("\n" + "=" * 70)
    print("Test 3: Threshold-Based Detection")
    print("=" * 70)

    threshold_jumps = detector.detect_jumps_threshold(returns, threshold_sigma = 4.0)
    threshold_stats = detector.calculate_jump_statistics(threshold_jumps)

    print(f"\n Threshold Results (4-sigma):")
    print(f" Total Jumps: {threshold_stats['n_jumps']}")
    print(f" Jump Frequency: {threshold_stats['jump_frequency']:.4f} ({threshold_stats['jump_frequency'] * 100:.2f}%)")
    print(f" Mean Inter-jump Time: {threshold_stats['mean_inter_jump_time']:.2f} days")
    print(f" Clustering Coefficient: {threshold_stats['clustering_coefficient']:.4f}")

    if threshold_stats['n_jumps'] > 0:
        print(f" Avg Jump Size: {threshold_stats['mean_jump_size']:.6f}")
        print(f" Jump Size Std: {threshold_stats['std_jump_size']:.6f}")

    #Visualization 
    print("\n [5] Creating Visualization")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Plot 1: Spread
    axes[0].plot(spread_df.index, spread_df['spread'], linewidth=1, color='blue')
    axes[0].set_ylabel('Spread (log scale)')
    axes[0].set_title('XOM/CVX Spread (Log Prices)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Returns with Bipower Variation jumps
    axes[1].plot(returns.index, returns, linewidth=0.8, color='gray', alpha=0.6, label='Returns')
    jump_times_bv = bpv_jumps[bpv_jumps['jump_indicator'] == 1].index
    jump_returns_bv = returns.loc[jump_times_bv]
    axes[1].scatter(jump_times_bv, jump_returns_bv, color='red', s=100, 
                    marker='o', zorder=5, label=f'Jumps (Bipower, n={len(jump_times_bv)})')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Returns')
    axes[1].set_title('Returns with Jump Detection (Bipower Variation)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Returns with Lee-Mykland jumps
    axes[2].plot(returns.index, returns, linewidth=0.8, color='gray', alpha=0.6, label='Returns')
    jump_times_lm = lm_jumps[lm_jumps['jump_indicator'] == 1].index
    jump_returns_lm = returns.loc[jump_times_lm]
    axes[2].scatter(jump_times_lm, jump_returns_lm, color='orange', s=100,
                    marker='s', zorder=5, label=f'Jumps (Lee-Mykland, n={len(jump_times_lm)})')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Returns')
    axes[2].set_title('Returns with Jump Detection (Lee-Mykland)', fontsize=14, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Comparison of all methods
    axes[3].plot(returns.index, returns, linewidth=0.8, color='gray', alpha=0.4, label='Returns')
    if len(jump_times_bv) > 0:
        axes[3].scatter(jump_times_bv, jump_returns_bv, color='red', s=80, 
                       marker='o', alpha=0.7, label='Bipower')
    if len(jump_times_lm) > 0:
        axes[3].scatter(jump_times_lm, jump_returns_lm, color='orange', s=60,
                       marker='s', alpha=0.7, label='Lee-Mykland')
    jump_times_thresh = threshold_jumps[threshold_jumps['jump_indicator'] == 1].index
    if len(jump_times_thresh) > 0:
        jump_returns_thresh = returns.loc[jump_times_thresh]
        axes[3].scatter(jump_times_thresh, jump_returns_thresh, color='green', s=40,
                       marker='^', alpha=0.7, label='Threshold')
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Returns')
    axes[3].set_xlabel('Date')
    axes[3].set_title('Comparison of Jump Detection Methods', fontsize=14, fontweight='bold')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'jump_detection_test_xom_cvx.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nData: {len(returns)} daily returns (XOM/CVX spread)")
    print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"\nJump Detection Results:")
    print(f"  Bipower Variation: {bpv_stats['n_jumps']} jumps ({bpv_stats['jump_frequency']*100:.2f}%)")
    print(f"  Lee-Mykland:       {lm_stats['n_jumps']} jumps ({lm_stats['jump_frequency']*100:.2f}%)")
    print(f"  Threshold (4-sigma):    {threshold_stats['n_jumps']} jumps ({threshold_stats['jump_frequency']*100:.2f}%)")
    
    print(f"\nAll tests completed successfully!")
    print("="*70)
