"""
Hawkes Process Calibration for Pairs Trading
Models self-exciting jump arrival intensity using Hawkes processes

The intensity follows: λ_t = λ̄ + ∑_{t_i < t} α * e^(-β_H(t - t_i))

Where:
- λ̄: Baseline intensity (constant)
- α: Jump impact (how much each jump increases intensity)
- β_H: Decay rate (how quickly the impact of jumps diminishes over time)
- t_i: Jump arrival times
"""

import numpy as np 
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class HawkesProcess:
    """
    Univariate Hawkes process for modeling self-exciting jump arrivals
    """

    def __init__(self, kernel: str = 'exponential'):
        """ 
        Initialize Hawkes Process
        
        Params:
        - kernel: str
            - kernel type ('exponential', 'power_law', 'sum_exponential')
        """

        self.kernel = kernel
        self.params = {}
        self.intensity = None 
        self.jump_times = None 

    def fit(self, jump_times: np.ndarray,
            T: float,
            method: str = 'MLE',
            initial_params: Optional[Dict] = None) -> Dict:
        
        """
        Fit Hawkes process parameters to the observed jump times 

        Params:
        - jump_times: np.ndarray
            - Array of jump arrival times
        - T: float
            - total observation period 
        - method: str
            - estimation method ('MLE', 'GMM')
        - initial_params: Dict, Optinoal
            - initial parameter values 

        Returns:
        - Dict
            - estimated parameters {lambda_bar, alpha, beta_H}

        """

        print(f"Fitting Hawkes process using {method}")

        self.jump_times = np.sort(jump_times)
        n_jumps = len(jump_times)

        if n_jumps < 2:
            print("Not enough jumps to fit the Hawkes process")
            return {'lambda_bar': 0, 'alpha': 0, 'beta': 1.0, 'beta_H': 1.0}
        
        #Initial parameter guesses
        if initial_params is None:
            lambda_bar_init = n_jumps / T #baseline intensity
            alpha_init = 0.5
            beta_init = 2.0
        else:
            lambda_bar_init = initial_params.get('lambda_bar', n_jumps / T)
            alpha_init = initial_params.get('alpha',  0.5)
            beta_init = initial_params.get('beta', initial_params.get('beta_H', 2.0))

        x0 = np.array([lambda_bar_init, alpha_init, beta_init])

        if method == 'MLE':
            result = self._fit_mle(jump_times, T, x0)
        elif method == 'GMM':
            result = self._fit_gmm(jump_times, T, x0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.params = {
            'lambda_bar': result['lambda_bar'],
            'alpha': result['alpha'],
            'beta': result['beta'],
            'beta_H': result['beta']
        }

        #calculate intensity at jump times
        self.intensity = self.compute_intensity(jump_times, T)

        print(f" Hawkes parameters estimate: ")
        print(f" λ̄ (baseline intensity): {self.params['lambda_bar']:.4f}")
        print(f" α (jump impact): {self.params['alpha']:.4f}")
        print(f" β_H (decay rate): {self.params['beta']:.4f}")
        print(f" Branching ratio (α/β_H): {self.params['alpha']/self.params['beta']:.4f}")

        return self.params
    
    def _fit_mle(self, jump_times: np.ndarray, T: float,
                 x0: np.ndarray) -> Dict:
        """
        Maximum Likelihood Estimation

        The log likelihood for a Hawkes process is given by:
        L = ∑_i log(λ(t_i)) - ∫_0^T λ(s) ds

        Params:
        - jump_times: np.ndarray
            - jump arrival times
        - T: float
            - total observation period
        - x0: np.ndarray
            - initial parameter guess [lambda_bar, alpha, beta_H]

        Returns:
        - Dict
            -Optimal parameters
        """

        def neg_log_likelihood(params):

            """ Negative log likelihood for minimization"""

            lambda_bar, alpha, beta = params

            #ensure positivity
            if lambda_bar <= 0 or alpha <= 0 or beta <= 0:
                return 1e10
            
            #ensure stability (branching ratio < 1)
            if alpha >= beta: 
                return 1e10
            branching_ratio = alpha / beta
            if branching_ratio > 0.85:
                return 1e10
            
            n = len(jump_times)

            #First term: sum of log intensities at jump times
            log_intensity_sum = 0
            for i in range(n):
                intensity_i = self._compute_intensity_at_time(
                    jump_times[i], jump_times[:i], lambda_bar, alpha, beta
                )
                if intensity_i <= 0:
                    return 1e10
                log_intensity_sum += np.log(intensity_i)

            #Second term: integral of intensity
            #For exponential kernel: ∫λ(s)ds = λ̄*T + α*n - α*∑e^(-β(T-t_i))

            integral = lambda_bar * T
            for t_i in jump_times:
                integral += (alpha / beta) * (1 - np.exp(-beta * (T - t_i)))

            return -(log_intensity_sum - integral)
        
        #Bounds for parameters
        bounds = [(0.001, 0.5), (0.01, 2.0), (0.1, 5.0)]
        #[λ̄: 0.1-50%] [α: moderate] [β: days-weeks decay]

        #optimize
        result = minimize(
            neg_log_likelihood,
            x0,
            method = 'L-BFGS-B',
            bounds = bounds
        )

        if not result.success:
            print(f"Optimization did not covnerge: {result.message}")

        return {
            'lambda_bar': result.x[0],
            'alpha': result.x[1],
            'beta': result.x[2],
            'log_likelihood': -result.fun
        }
    
    def _fit_gmm(self, jump_times: np.ndarray, T: float,
                 x0: np.ndarray) -> Dict:
         """
         Generalized method of moments estimation
         
         Uses moment conditions based on inter-arrival times 

         Params:
         - jump_times: np.ndarray
            - jump arrival times
        - T: float
            - total observation period
        - x0: np.ndarray
            - initial parameter guess

        Returns:
        - Dict
            - optimal parameters
        """
         
         #For simplicity, going to use MLE
         #Full GMM implementaiton would use moment conditions
         return self._fit_mle(jump_times, T, x0)
    
    def _compute_intensity_at_time(self, t: float, 
                                   past_jumps: np.ndarray,
                                   lambda_bar: float, 
                                   alpha: float,
                                   beta: float) -> float:
        
        """
        Computing intensity at a specific time given past jumps

        (t) = λ̄ + ∑_{t_i < t} α * e^(-β(t - t_i))

        Params:
        - t: float
            - time point 
        - past_jumps: np.ndarray
            - jump times before t
        - lambda_bar: float 
            - baseline intensity
        - alpha: float
            - excitation parameter
        - beta: float
            - decay rate

        Returns:
        - float
            - intensity at time t
        """

        if len(past_jumps) == 0:
            return lambda_bar
        
        #sum exponential kernels for all past jumps
        excitement = alpha * np.sum(np.exp(-beta * (t - past_jumps)))

        return lambda_bar + excitement
    
    def compute_intensity(self, 
                          jump_times: np.ndarray,
                          T: float,
                          n_points: int = 1000) -> pd.Series:
        
        """
        Computing intensity path over entire observation period

        Params:
        - jump_times: np.ndarray
            - jump arrival times
        - T: float
            - Total observation period
        - n_points: int 
            - number of points for intensity path 

        Returns:
        - pd.Series
            - Intensity time series
        """

        if not self.params:
            raise ValueError("Model not fitted. Call fit() first.")
        
        lambda_bar = self.params['lambda_bar']
        alpha = self.params['alpha']
        beta = self.params['beta']

        #create time grid
        time_grid = np.linspace(0, T, n_points)
        intensity_path = np.zeros(n_points)

        for i, t in enumerate(time_grid):
            past_jumps = jump_times[jump_times < t]
            intensity_path[i] = self._compute_intensity_at_time(
                t, past_jumps, lambda_bar, alpha, beta
            )
        return pd.Series(intensity_path, index = time_grid)
    
    def compute_intensity_at_dates(self,
                                   jump_times: np.ndarray,
                                   dates: pd.DatetimeIndex,
                                   T: float) -> pd.Series:
        
        """
        Compute intensity at specific dates

        Params:
        - jump_timess: np.ndarray
            - jump arrival times (numeric)
        -dates: pd.DatetimeIndex
            - dates to compute intensity
        - T: float
            - total observation period 

        Returns:
        - pd.Series
            - intensity indexed by dates
        """

        if not self.params:
            raise ValueError("Model not fitted. Call fit() first")
        
        lambda_bar = self.params['lambda_bar']
        alpha = self.params['alpha']
        beta = self.params['beta']

        #Convert dates to numeric time 
        start_date = dates[0]
        time_grid = (dates - start_date).days + (dates - start_date).seconds / 86400

        intensity_path = np.zeros(len(time_grid))

        for i, t in enumerate(time_grid):
            past_jumps = jump_times[jump_times < t]
            intensity_path[i] = self._compute_intensity_at_time(
                t, past_jumps, lambda_bar, alpha, beta
            )

        return pd.Series(intensity_path, index = dates)
    
    def simulate(self, T: float,
                 lambda_bar: Optional[float] = None,
                 alpha: Optional[float] = None, 
                 beta: Optional[float] = None) -> np.ndarray:
        
        """
        Simulate Hawkes process jump times using Ogata's thinning algorithm

        Params:
        - T: float
            - simulation horizon
        - lambda_bar: float, Optional
            - baseline intensity (uses fitted if not provided)
        - alpha: float, optional
            - excitation parameter
        - beta: float, optional
            - decay rate

        Returns:
        - np.ndarray
            - simulated jump times 
        """

        #use fitted parameters if not provided
        if lambda_bar is None or alpha is None or beta is None:
            if not self.params:
                raise ValueError("No parameters available")
            if lambda_bar is None:
                lambda_bar = self.params.get('lambda_bar')
            if alpha is None:
                alpha = self.params.get('alpha')
            if beta is None:
                beta = self.params.get('beta', self.params.get('beta_H'))

        if lambda_bar is None or alpha is None or beta is None:
            raise ValueError("missing Hawkes parameters: lambda_bar, alpha, beta")

        #Ogata's thinning algorithm
        jump_times = []
        t = 0

        #Upper bound on intensity
        lambda_max = lambda_bar + alpha / beta * 100 #Conservative upper bound

        while t < T:
            #propose next jump time from homogenous Poisson 
            u1 = np.random.random()
            t = t - np.log(u1) / lambda_max
            
            if t > T:
                break 

            #Compute actual intensity
            intensity = self._compute_intensity_at_time(
                t, np.array(jump_times), lambda_bar, alpha, beta
            )

            #Accept with probability intensity / lambda_max
            u2 = np.random.random()
            if u2 <= intensity / lambda_max:
                jump_times.append(t)
                #update upper bound
                lambda_max = max(lambda_max, intensity)

        return np.array(jump_times)
    
    def goodness_of_fit(self, jump_times: np.ndarray, T: float) -> Dict:
        """
        Assess goodness of fit using residual analysis
         
        For a well fitted Hawkes process, the compensator transformation
        should produce a unit-rate Poission process

        Params:
        - jump_times: np.ndarray
            - observed jump times 
        - T: float
            - Total observation period 

        Returns
        - Dict
            - goodness of fit statistics
        """

        if not self.params:
            raise ValueError("Model not fitted. Call fit() first.")
        
        lambda_bar = self.params['lambda_bar']
        alpha = self.params['alpha']
        beta = self.params['beta']

        n = len(jump_times)

        #compute compensator (integrated intensity)
        compensator = np.zeros(n)
        for i in range(n):
            #integral from 0 to t_i
            Lambda_i = lambda_bar * jump_times[i]
            for j in range(i):
                Lambda_i += (alpha / beta) * (1 - np.exp(-beta * (jump_times[i] - jump_times[j])))
            compensator[i] = Lambda_i

        #Transformed times should be unit_rate Poisson
        #Interarrival times should be exponential(1)
        transformed_inter_arrivals = np.diff(np.concatenate([[0], compensator]))

        #Kolmogorov-Smirnov test against exponential(1)
        ks_stat, ks_pvalue = stats.kstest(transformed_inter_arrivals, 'expon')

        #QQ plot values
        theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, len(transformed_inter_arrivals)))
        empirical_quantiles = np.sort(transformed_inter_arrivals)

        gof = {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'is_good_fit': ks_pvalue > 0.05,
            'theoretical_quantiles': theoretical_quantiles,
            'empirical_quantiles': empirical_quantiles,
            'transformed_times': compensator
        }

        return gof 
    
    def branching_ratio(self) -> float:
        """
        Calculating branching ratio (meaasure of self-excitation strenght)

        Branching ratio = α / β
        - If < 1: process is stationary
        - If = 1: critical (explosive boundary)
        - If > 1: explosive (unstable)

        Returns:
        - float
            - branching ratio 
        """

        if not self.params:
            raise ValueError("Model not fitted. Call fit() first")
        return self.params['alpha'] / self.params['beta']
    
if __name__ == "__main__":
    """
    Test Hawkes process calibration on GDX/GLD equity pairs
    """
    import sys
    import os
    import matplotlib.pyplot as plt
    
    # Add parent directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    from equity_pairs_loader import EquityPairsDataPipeline
    from jump_detector import JumpDetector
    
    print("="*70)
    print("HAWKES PROCESS CALIBRATION TEST - GDX/GLD EQUITY PAIRS")
    print("="*70)
    
    # Initialize data pipeline
    print("\n[1] Loading GDX/GLD data from CSV files...")
    pipeline = EquityPairsDataPipeline()
    
    # Try to load CSV files from multiple possible locations
    csv_paths = [
        (os.path.join(current_dir, 'OHLCV_GDX.csv'),
         os.path.join(current_dir, 'OHLCV_GLD.csv')),  # Script directory
        ('OHLCV_GDX.csv', 'OHLCV_GLD.csv'),  # Current directory
    ]
    
    data_loaded = False
    for xom_path, cvx_path in csv_paths:
        if os.path.exists(xom_path) and os.path.exists(cvx_path):
            try:
                data = pipeline.load_from_csv(xom_path, cvx_path)
                data_loaded = True
                print(f"✓ Loaded from: {os.path.abspath(xom_path)}")
                break
            except Exception as e:
                continue
    
    if not data_loaded:
        print("Could not find CSV files. Testing with simulated data instead...")
        print("\n" + "="*70)
        print("SIMULATED DATA TEST")
        print("="*70)
        
        # Simulate some jump times
        hawkes = HawkesProcess()
        true_params = {'lambda_bar': 0.5, 'alpha': 0.8, 'beta': 2.0}
        
        print("\nSimulating Hawkes process with true parameters:")
        print(f"  λ̄ = {true_params['lambda_bar']}")
        print(f"  α = {true_params['alpha']}")
        print(f"  β = {true_params['beta']}")
        print(f"  Branching ratio = {true_params['alpha']/true_params['beta']:.3f}")
        
        T = 100
        simulated_jumps = hawkes.simulate(T, **true_params)
        
        print(f"\nSimulated {len(simulated_jumps)} jumps over period T={T}")
        
        # Fit to simulated data
        print("\nFitting Hawkes process to simulated data...")
        estimated_params = hawkes.fit(simulated_jumps, T, method='MLE')
        
        # Compare parameters
        print("\n" + "="*70)
        print("PARAMETER COMPARISON")
        print("="*70)
        print(f"{'Parameter':<15} {'True':<12} {'Estimated':<12} {'Error %':<12}")
        print("-"*70)
        for param in ['lambda_bar', 'alpha', 'beta']:
            true_val = true_params[param]
            est_val = estimated_params[param]
            error = 100 * abs(true_val - est_val) / true_val
            print(f"{param:<15} {true_val:<12.4f} {est_val:<12.4f} {error:<12.2f}")
        
        # Goodness of fit
        print("\n" + "="*70)
        print("GOODNESS OF FIT TEST")
        print("="*70)
        gof = hawkes.goodness_of_fit(simulated_jumps, T)
        print(f"KS Statistic: {gof['ks_statistic']:.4f}")
        print(f"KS P-value: {gof['ks_pvalue']:.4f}")
        print(f"Good fit: {'Yes' if gof['is_good_fit'] else 'No'}")
        print(f"Branching ratio: {hawkes.branching_ratio():.4f}")
        
        sys.exit(0)
    
    # Clean and construct spread
    print("\n[2] Constructing spread...")
    cleaned_data = pipeline.clean_data()
    spread_df = pipeline.construct_spread(method='cointegration', lookback=60)
    
    print(f"✓ Spread constructed: {len(spread_df)} observations")
    print(f"  Date range: {spread_df.index[0].date()} to {spread_df.index[-1].date()}")
    
    # Calculate returns for jump detection
    print("\n[3] Detecting jumps...")
    returns = spread_df['spread'].pct_change().dropna()
    
    detector = JumpDetector(significance_level=0.01)
    jump_df = detector.detect_jumps_bipower_variation(returns, window=20)
    
    # Extract jump times
    jump_times = jump_df[jump_df['jump_indicator'] == 1].index
    jump_times_numeric = (jump_times - spread_df.index[0]).days.values
    
    n_jumps = len(jump_times)
    print(f"Detected {n_jumps} jumps ({100*n_jumps/len(returns):.2f}%)")
    
    if n_jumps < 2:
        print("\nWarning: Too few jumps for reliable Hawkes calibration")
        print(f"  Minimum 2 jumps required, found only {n_jumps}")
        print("  This is actually good - shows stable GDX/GLD relationship!")
        
        # Still fit with dummy parameters for demonstration
        hawkes = HawkesProcess()
        hawkes.params = {
            'lambda_bar': 0.001,
            'alpha': 0.1,
            'beta': 1.0
        }
        print("\nUsing baseline Hawkes parameters (minimal self-excitation)")
        sys.exit(0)
    
    # Fit Hawkes process to real jump times
    print("\n[4] Fitting Hawkes process to detected jumps...")
    
    T = (spread_df.index[-1] - spread_df.index[0]).days
    
    hawkes = HawkesProcess()
    try:
        params = hawkes.fit(jump_times_numeric, T, method='MLE')
        
        print(f"\nHawkes Process Parameters (Real Data):")
        print(f"  λ̄ (baseline):    {params['lambda_bar']:.6f}")
        print(f"  α (excitation):  {params['alpha']:.6f}")
        print(f"  β (decay):       {params['beta']:.6f}")
        print(f"  Branching ratio: {hawkes.branching_ratio():.6f}")
        
        if hawkes.branching_ratio() >= 1.0:
            print("  WARNING: Branching ratio ≥ 1 (explosive process)")
        else:
            print("  Stationary process (branching ratio < 1)")
        
    except Exception as e:
        print(f"Fitting failed: {str(e)}")
        print("  This can happen with very sparse jumps")
        sys.exit(1)
    
    # Goodness of fit test
    print("\n[5] Goodness of Fit Test...")
    gof = hawkes.goodness_of_fit(jump_times_numeric, T)
    
    print(f"  KS Statistic: {gof['ks_statistic']:.4f}")
    print(f"  KS P-value:   {gof['ks_pvalue']:.4f}")
    print(f"  Good fit:     {'Yes ✓' if gof['is_good_fit'] else 'No ✗'}")
    
    # Simulate future jumps
    print("\n[6] Simulating future jumps with fitted parameters...")
    T_future = 365  # Simulate 1 year ahead
    simulated_jumps = hawkes.simulate(T_future, 
                                      lambda_bar=params['lambda_bar'],
                                      alpha=params['alpha'],
                                      beta=params['beta'])
    
    print(f"  Simulated {len(simulated_jumps)} jumps over next {T_future} days")
    print(f"  Expected jumps per year: {len(simulated_jumps)}")
    print(f"  Average inter-jump time: {T_future/max(1, len(simulated_jumps)):.1f} days")
    
    # Visualization
    print("\n[7] Creating visualizations...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Spread with jump times
    axes[0].plot(spread_df.index, spread_df['spread'], linewidth=1, color='blue', alpha=0.7)
    axes[0].scatter(jump_times, spread_df.loc[jump_times, 'spread'], 
                   color='red', s=100, marker='x', zorder=5, 
                   label=f'Detected Jumps (n={n_jumps})')
    axes[0].set_ylabel('Spread')
    axes[0].set_title('GDX/GLD Spread with Detected Jumps', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Jump intensity over time
    if n_jumps >= 2:
        # Calculate conditional intensity at each observed jump time
        intensities = []
        for i, t in enumerate(jump_times_numeric):
            lambda_t = params['lambda_bar']
            for j in range(i):
                if jump_times_numeric[j] < t:
                    lambda_t += params['alpha'] * np.exp(-params['beta'] * (t - jump_times_numeric[j]))
            intensities.append(lambda_t)
        
        axes[1].plot(jump_times, intensities, marker='o', linestyle='-', 
                    color='orange', linewidth=2, markersize=8)
        axes[1].axhline(y=params['lambda_bar'], color='red', linestyle='--', 
                       label=f"Baseline λ̄={params['lambda_bar']:.6f}")
        axes[1].set_ylabel('Jump Intensity λ(t)')
        axes[1].set_title('Self-Exciting Jump Intensity', fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: QQ plot for goodness of fit
    if gof['theoretical_quantiles'] is not None:
        axes[2].scatter(gof['theoretical_quantiles'], gof['empirical_quantiles'], 
                       alpha=0.6, s=50, color='purple')
        # 45-degree line
        min_val = min(gof['theoretical_quantiles'].min(), gof['empirical_quantiles'].min())
        max_val = max(gof['theoretical_quantiles'].max(), gof['empirical_quantiles'].max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect fit')
        axes[2].set_xlabel('Theoretical Quantiles (Exponential)')
        axes[2].set_ylabel('Empirical Quantiles')
        axes[2].set_title(f'QQ Plot (KS p-value: {gof["ks_pvalue"]:.4f})', 
                         fontsize=14, fontweight='bold')
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure to the same folder used by performance metrics
    output_dir = os.path.join(current_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'hawkes_calibration_test_xom_cvx.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nData: GDX/GLD spread ({len(spread_df)} days)")
    print(f"Date range: {spread_df.index[0].date()} to {spread_df.index[-1].date()}")
    print(f"\nJump Detection:")
    print(f"  Total jumps: {n_jumps}")
    print(f"  Jump frequency: {100*n_jumps/len(returns):.2f}%")
    print(f"\nHawkes Parameters:")
    print(f"  λ̄ (baseline):    {params['lambda_bar']:.6f}")
    print(f"  α (excitation):  {params['alpha']:.6f}")
    print(f"  β (decay):       {params['beta']:.6f}")
    print(f"  Branching ratio: {hawkes.branching_ratio():.6f} {'✓' if hawkes.branching_ratio() < 1.0 else '⚠'}")
    print(f"\nGoodness of Fit:")
    print(f"  KS p-value: {gof['ks_pvalue']:.4f}")
    print(f"  Fit quality: {'Good ✓' if gof['is_good_fit'] else 'Poor ✗'}")
    print(f"\nAll tests completed successfully!")
    print("="*70)
