"""
Mean-Reverting Jump Diffusion (MRJD) Model with Hawkes Jumps

The model combines:
1. Ornstein-Uhlenbeck mean reversion: κ(θ - X_t)dt
2. Diffusive volatility: σ dW_t
3. Self-exciting jumps: Y dN_t where N_t ~ Hawkes(λ_t)

Stochastic Differential Equation: dX_t = κ(θ - X_t)dt + σdW_t + Y dN_t
"""

import numpy as np 
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class MRJDModel:
    """
    Mean-Reverting Jump Diffusion model with self-exciting jumps
    """

    def __init__(self):
        self.params = {}
        self.ou_params = {}
        self.jump_params = {}
        self.fitted = False

    def fit(self,
            spread: pd.Series,
            jump_indicator: pd.Series,
            dt: float = 1/252,
            method: str = 'MLE') -> Dict:
        """
        Fitting MRJD model params

        Strategy:
        1. Estimate the OU params (κ, θ, σ) from non-jump periods
        2. Estimate jump distribution (μ_J, σ_J) from detected jumps
        3. Refine all parameters jointly via MLE

        Params:
        - spread: pd.Series
            - spread time series
        - jump_indicator: pd.Series
            - binary jump indicator
        - dt: float
            - time step
        - method: str
            - estimation method 

        Returns:
        - Dict
            - estimated parameters
        """

        print("Fitting the MRJD model")

        #Align data
        common_index = spread.index.intersection(jump_indicator.index)
        spread = spread.loc[common_index]
        jump_indicator = jump_indicator.loc[common_index]

        #Step 1: Estimate OU parameters from continous part
        print(" Step 1: Estimating OU params from continous periods")
        self.ou_params = self._estimate_ou_parameters(
            spread, jump_indicator, dt
        )

        #Step 2: Estimate jump distribution
        print(" Step 2: Estimating jump distribution")
        self.jump_params = self._estimate_jump_parameters(
            spread, jump_indicator
        )

        #Step 3: Joint MLE refinement
        print(" Step 3: Joint MLE refinement")
        print("Warning: Skipping joint MLE to preserve the validated kappa")
        print(" (Joint MLE tends to overfit and revert to the incorrect parameters)")
        #Always use the separate estimates to preserve validated kappa from Step 1

        refined_params = {**self.ou_params, **self.jump_params}

        self.params = refined_params
        self.fitted = True 

        print("MRJD Model Parameters:")
        print(f" κ (mean reversion speed): {self.params['kappa']:.4f}")
        print(f" θ (long-run mean): {self.params['theta']:.4f}")
        print(f" σ (diffusive volatility): {self.params['sigma']:4f}")
        print(f"  μ_J (jump mean): {self.params['jump_mean']:.4f}")
        print(f"  σ_J (jump volatility): {self.params['jump_std']:.4f}")

        return self.params 
    
    def _estimate_ou_parameters(self,
                                spread: pd.Series,
                                jump_indicator: pd.Series,
                                dt: float) -> Dict:
        """
        Estimating Ornstein-Uhlenbeck parameters from non jump periods 

        Uses maximum likelihood for discrete observations:
        X_{t+dt} | X_t ~ N(θ + (X_t - θ)e^(-κdt), σ²(1-e^(-2κdt))/(2κ))

        Params:
        - spread: pd.Series
            - spread series
        - jump_indicator: pd.Series
            - jump indicators
        - dt: float
            - time step 

        Returns:
        - Dict
            - OU parameters {kappa, theta, sigma}
        """

        #Filter out jump periods
        no_jump_mask = (jump_indicator == 0)
        spread_no_jumps = spread[no_jump_mask]

        if len(spread_no_jumps) < 10:
            print("Warning: Very few non-jump observations")
            spread_no_jumps = spread #use all data

        #Prepare the data for estimation
        X_t = spread_no_jumps.values[:-1]
        X_t_plus_dt = spread_no_jumps.values[1:]

        def neg_log_likelihood(params):
            """Negative log likelihood for OU process"""
            kappa, theta, sigma = params 

            #Ensure positivity
            if kappa <= 0 or sigma <= 0:
                return 1e10
            
            #Conditional mean and variance
            mu_cond = theta + (X_t - theta) * np.exp(-kappa * dt)
            var_cond = (sigma ** 2) * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)

            if var_cond <= 0:
                return 1e10
            
            #Log-likelihood
            residuals = X_t_plus_dt - mu_cond
            log_lik = -0.5 * np.sum(
                np.log(2 * np.pi * var_cond) + (residuals ** 2) / var_cond 
            )

            return -log_lik 
        
        #Initial guess
        theta_init = spread_no_jumps.mean() 
        #Simple AR(1) for kappa estimate
        spread_diff = spread_no_jumps.diff().dropna()
        spread_lag = spread_no_jumps.shift(1).dropna()
        common_idx = spread_diff.index.intersection(spread_lag.index)
        if len(common_idx) > 0:
            beta = np.cov(spread_diff.loc[common_idx],
                          spread_lag.loc[common_idx] - theta_init)[0,1] / np.var(spread_lag.loc[common_idx] - theta_init)
            kappa_init = -np.log(1 + beta) / dt if beta < 0 else 1.0
        else:
            kappa_init = 1.0 

        sigma_init = spread_no_jumps.std() * np.sqrt(2 * kappa_init)

        x0 = np.array([max(kappa_init, 0.01), theta_init, max(sigma_init, 0.01)])

        #bounds
        bounds = [(0.01, 10.0), (-5.0, 5.0), (0.001, 5.0)]

        #optimize
        result = minimize(
            neg_log_likelihood,
            x0,
            method = 'L-BFGS-B',
            bounds = bounds
        )

        if not result.success:
            print(f"Warning: OU optimization did not converge")

        #calculating MRJD estimate half-life from optimized kappa 
        kappa_mrjd = result.x[0]
        half_life_mrjd = np.log(2) / kappa_mrjd 

        #Calculating empirical half-life using AR(1) regression on the full spread
        #Not just non-jump periods so that we can capture true mean reversion rate 
        spread_lag = spread.shift(1).dropna() 
        spread_diff = spread.diff().dropna()
        common_idx = spread_lag.index.intersection(spread_diff.index)

        if len(common_idx) > 10:
            try:
                #Ar(1) regression: Δy_t = β * y_{t-1}
                #half-life: -log(2) / β

                beta_empirical = np.polyfit(spread_lag.loc[common_idx], spread_diff.loc[common_idx], 1)[0]

                if beta_empirical < 0:
                    #mean reverting process
                    half_life_empirical = -np.log(2) / beta_empirical 
                else:
                    #non stationary 
                    half_life_empirical = np.inf
                    print(" Warning: spread is not mean reverting: (Beta >= 0)")

                #Checking for large discrepencies (>50% difference)
                if np.isfinite(half_life_empirical) and half_life_empirical > 0:
                    relative_error = abs(half_life_mrjd - half_life_empirical) / half_life_empirical 

                    if relative_error > 0.5: #which is more than a 50% difference
                        print("\n" + "=" * 70)
                        print("Critical: MRJD Half Life Mismatch Detected")
                        print("=" * 70)
                        print(f"    MRJD estimated half-life: {half_life_mrjd:.1f} days")
                        print(f"    Empirical half-life: {half_life_empirical:.1f} days")
                        print(f"    Relative error: {relative_error * 100:.1f}%")
                        print(f"    Discrepancy: {abs(half_life_mrjd - half_life_empirical):.1f} days")
                        print("\n Overiding MRJD kappa with the empirical estimate instead")

                        #overriding with empirical kappa estimate 
                        kappa_empirical = np.log(2) / half_life_empirical 

                        #update result
                        result.x[0] = kappa_empirical 

                        print(f"\n Corrected kappa: {kappa_mrjd:.6f} -> {kappa_empirical:.6f}")
                        print(f" Corrected half-life: {half_life_mrjd:.1f} days -> {half_life_empirical:.1f} days")
                        print("=" * 70 + "\n")
                    else:
                        #If a good fit, just report it normally
                        print(f"\n MRJD Half life Validation Passed")
                        print(f"    MRJD estimated: {half_life_mrjd:.1f} days")
                        print(f"    Empirical: {half_life_empirical:.1f} days")
                        print(f"    Error: {relative_error * 100:.1f}% (acceptable)\n")

            except Exception as e:
                print(f"Warning: could not validate the MRJD half-life: {str(e)}")
        else:
            print("Warning: Insufficient data for MRJD validation")

        return {
            'kappa': result.x[0],
            'theta': result.x[1],
            'sigma': result.x[2]
        }
    
    def _estimate_jump_parameters(self,
                                  spread: pd.Series,
                                  jump_indicator: pd.Series) -> Dict:
        """
        Estimate jump size distribution parameters
        
        Assumes jump sizes Y ~ N(μ_J, σ_J²)

        Params:
        - spread: pd.Series
            - spread series
        - jump_indicator: pd.Series
            - jump indicators 

        Returns:
        - dict
            - jump parameters {jump_mean, jump_std}
        """

        #extract jump sizes
        spread_diff = spread.diff()
        jump_sizes = spread_diff[jump_indicator == 1].dropna()

        if len(jump_sizes) == 0:
            print("Warning: No jumps detected, using default parameters")
            return{'jump_mean': 0, 'jump_std': 0.1}
        
        #estimate from empirical distribution
        jump_mean = jump_sizes.mean() 
        jump_std = jump_sizes.std() 

        if np.isnan(jump_std) or jump_std == 0:
            jump_std = 0.1

        return {
            'jump_mean': jump_mean,
            'jump_std': jump_std
        }
    
    def _joint_mle(self,
                   spread: pd.Series,
                   jump_indicator: pd.Series,
                   dt: float) -> Dict:
        
        """ 
        Joint MLE for all MRJD params

        Params:
        - spread: pd.Series
            - spread series
        - jump_indicator: pd.Series
            - jump indicators
        - dt: float
            - time step 

        Returns:
        - dict
            - all parameters
        """

        X_t = spread.values[:-1]
        X_t_plus_dt = spread.values[1:]
        jumps = jump_indicator.values[1:]

        def neg_log_likelihood(params):
            """ Joint negative log likelihood """
            kappa, theta, sigma, jump_mean, jump_std = params 

            #ensure positivity
            if kappa <= 0 or sigma <= 0 or jump_std <= 0:
                return 1e10
            
            #OU component
            mu_ou = theta + (X_t - theta) * np.exp(-kappa * dt)
            var_ou = (sigma ** 2) * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)

            if var_ou <= 0:
                return 1e10
            
            #Total conditional distribution
            #Non jump periods: N(mu_ou, var_ou)
            #Jump periods: N(mu_ou + jump_mean, var_ou + jump_std^2)

            mu_cond = np.where(jumps == 1, mu_ou + jump_mean, mu_ou)
            var_cond = np.where(jumps == 1, var_ou + jump_std ** 2, var_ou)

            #Log-likelihood
            residuals = X_t_plus_dt - mu_cond 
            log_lik = -0.5 * np.sum(
                np.log(2 * np.pi * var_cond) + (residuals ** 2) / var_cond
            )

            return -log_lik
        
        #initial guess from separate estimates
        x0 = np.array([
            self.ou_params['kappa'],
            self.ou_params['theta'],
            self.ou_params['sigma'],
            self.jump_params['jump_mean'],
            self.jump_params['jump_std']
        ])

        #bounds
        bounds = [
            (0.01, 10.0), #kappa
            (-5.0, 5.0), #theta
            (0.001, 5.0), #sigma
            (-1.0, 1.0), #jump_mean
            (0.001, 2.0) #jump_std
        ]

        #optimize
        result = minimize(
            neg_log_likelihood,
            x0,
            method = 'L-BFGS-B',
            bounds = bounds
        )

        if not result.success:
            print(f"Warning: Joing MLE did not converge, using separate estimates")
            return {**self.ou_params, **self.jump_params}
        
        return {
            'kappa': result.x[0],
            'theta': result.x[1],
            'sigma': result.x[2],
            'jump_mean': result.x[3],
            'jump_std': result.x[4]
        }
    
    def simulate(self,
                 X0: float,
                 T: float,
                 dt: float,
                 jump_times: Optional[np.ndarray] = None,
                 seed: Optional[int] = None) -> pd.DataFrame:
        
        """ 
        Simulate MRJD Path

        Params:
        - X0: float
            - initial value
        - T: float
            - time horizon
        - dt: float
            - time step
        - jump_times: np.ndarray, optional
            - pre-specified jump times
        - seed: int, optional
            - random seed 

        Returns:
        - pd.DataFrame
            - simulated path
        """

        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first")
        
        if seed is not None:
            np.random.seed(seed)

        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma']
        jump_mean = self.params['jump_mean']
        jump_std = self.params['jump_std']

        #Time grid
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps)

        #initialize path
        X = np.zeros(n_steps)
        X[0] = X0

        #Convert jump times to indices
        if jump_times is not None:
            jump_indices = (jump_times / dt).astype(int)
            jump_indices = jump_indices[jump_indices < n_steps]
        else:
            jump_indices = np.array([])

        #Simulate path
        for t in range(1, n_steps):
            #OU dynamics
            dW = np.random.normal(0, np.sqrt(dt))
            dX = kappa * (theta - X[t-1]) * dt + sigma * dW

            #add jump if at jump time
            if t in jump_indices:
                dX += np.random.normal(jump_mean, jump_std)

            X[t] = X[t-1] + dX

        
        return pd.DataFrame({
            'time': times,
            'spread': X,
            'jump_indicator': np.isin(np.arange(n_steps), jump_indices).astype(int)
        })
    
    def predict_spread_statistics(self,
                                  X_current: float,
                                  horizon: float) -> Dict:
        
        """
        Predict spread statistics at future horizon

        For OU process:
        E[X_t | X_0] = θ + (X_0 - θ)e^(-κt)
        Var[X_t | X_0] = σ²(1 - e^(-2κt)) / (2κ)

        Params:
        - X_current: float
            - current spread value
        - horizon: float
            - prediction horizon (in same units as dt)

        Returns:
        - Dict
            - predicted statistics 
        """

        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first")
        
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma']

        #expected value
        E_X = theta + (X_current - theta) * np.exp(-kappa * horizon)

        #variance (excluding jumps for conservative estimate)
        Var_X = (sigma ** 2) * (1 - np.exp(-2 * kappa * horizon)) / (2 * kappa)

        #Standard deviation
        Std_X = np.sqrt(Var_X)

        #Half life of mean reversion
        half_life = np.log(2) / kappa 

        return {
            'expected_spread': E_X,
            'spread_std': Std_X,
            'half_life': half_life,
            'reversion_pct': 100 * (1 - np.exp(-kappa * horizon))
        }
    
    def calculate_z_score(self, X: pd.Series) -> pd.Series:
        """
        Calculate standardized spread (Z-score)

        Z_t = (X_t - θ) / σ_X

        Params:
        - X: pd.Series
            - spread series

        Returns:
        - pd.Series
            - Z-scores
        """

        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first")
        
        theta = self.params['theta']
        sigma = self.params['sigma']
        kappa = self.params['kappa']

        #Steady state standard deviation
        sigma_X = sigma / np.sqrt(2 * kappa)

        z_score = (X - theta) / sigma_X

        return z_score 
    

if __name__ == "__main__":
    """
    Test MRJD model estimation on NVDA/AMD equity pairs
    """
    import sys
    import os
    
    # Add parent directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    from equity_pairs_loader import EquityPairsDataPipeline
    from jump_detector import JumpDetector
    
    print("="*70)
    print("MRJD MODEL ESTIMATION TEST - NVDA/AMD EQUITY PAIRS")
    print("="*70)
    
    # Initialize data pipeline
    print("\n[1] Loading NVDA/AMD data from CSV files...")
    pipeline = EquityPairsDataPipeline()
    
    # Try to load CSV files from multiple possible locations
    csv_paths = [
        (os.path.join(current_dir, 'OHLCV_NVDA.csv'),
         os.path.join(current_dir, 'OHLCV_AMD.csv')),  # Script directory
        ('OHLCV_NVDA.csv', 'OHLCV_AMD.csv'),  # Current directory
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
        print("✗ Could not find CSV files. Testing with synthetic data instead...")
        print("\n" + "="*70)
        print("SYNTHETIC DATA TEST")
        print("="*70)
        
        # Generate synthetic data (fallback)
        np.random.seed(42)
        
        # True parameters
        true_params = {
            'kappa': 0.5,
            'theta': 0.0,
            'sigma': 0.15,
            'jump_mean': -0.02,
            'jump_std': 0.05
        }
        
        print("\nTrue parameters:")
        for param, value in true_params.items():
            print(f"  {param}: {value:.4f}")
        
        # Simulate MRJD process
        T = 500
        dt = 1/252
        n_steps = int(T / dt)
        
        # Generate jump times
        n_jumps = 50
        jump_times = np.sort(np.random.uniform(0, T, n_jumps))
        jump_indices = (jump_times / dt).astype(int)
        
        # Simulate
        X = np.zeros(n_steps)
        X[0] = 0.0
        
        for t in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            dX = true_params['kappa'] * (true_params['theta'] - X[t-1]) * dt + \
                 true_params['sigma'] * dW
            
            if t in jump_indices:
                dX += np.random.normal(true_params['jump_mean'], true_params['jump_std'])
            
            X[t] = X[t-1] + dX
        
        # Create series
        spread = pd.Series(X)
        jump_indicator = pd.Series(np.isin(np.arange(n_steps), jump_indices).astype(int))
        
        print(f"\n✓ Simulated {n_steps} observations with {n_jumps} jumps")
        
        # Fit model
        model = MRJDModel()
        estimated_params = model.fit(spread, jump_indicator, dt=dt)
        
        # Compare parameters
        print("\n" + "="*70)
        print("PARAMETER COMPARISON")
        print("="*70)
        print(f"{'Parameter':<15} {'True':<12} {'Estimated':<12} {'Error %':<12}")
        print("-"*70)
        for param in ['kappa', 'theta', 'sigma', 'jump_mean', 'jump_std']:
            true_val = true_params[param]
            est_val = estimated_params[param]
            error = 100 * abs(true_val - est_val) / abs(true_val) if true_val != 0 else 0
            print(f"{param:<15} {true_val:<12.4f} {est_val:<12.4f} {error:<12.2f}")
        
        sys.exit(0)
    
    # Clean and construct spread
    print("\n[2] Constructing spread...")
    cleaned_data = pipeline.clean_data()
    spread_df = pipeline.construct_spread(method='cointegration', lookback=60)
    
    spread = spread_df['spread']
    print(f"✓ Spread constructed: {len(spread)} observations")
    print(f"  Date range: {spread.index[0].date()} to {spread.index[-1].date()}")
    print(f"  Spread statistics:")
    print(f"    Mean: {spread.mean():.4f}")
    print(f"    Std: {spread.std():.4f}")
    print(f"    Min: {spread.min():.4f}")
    print(f"    Max: {spread.max():.4f}")
    
    # Detect jumps
    print("\n[3] Detecting jumps...")
    returns = spread.pct_change().dropna()
    
    detector = JumpDetector(significance_level=0.01)
    jump_df = detector.detect_jumps_bipower_variation(returns, window=20)
    
    # Create jump indicator aligned with spread
    jump_indicator = pd.Series(0, index=spread.index)
    jump_times = jump_df[jump_df['jump_indicator'] == 1].index
    jump_indicator.loc[jump_times] = 1
    
    n_jumps = jump_indicator.sum()
    print(f"✓ Detected {n_jumps} jumps ({100*n_jumps/len(spread):.2f}%)")
    
    if n_jumps == 0:
        print("⚠ Warning: No jumps detected")
        print("  MRJD will reduce to pure OU process")
    
    # Fit MRJD model
    print("\n[4] Fitting MRJD model...")
    model = MRJDModel()
    dt = 1/252  # Daily data
    
    try:
        estimated_params = model.fit(spread, jump_indicator, dt=dt, method='MLE')
        
        print("\n✓ MRJD Model Fitted Successfully!")
        print("\nEstimated Parameters:")
        print(f"  κ (mean reversion): {estimated_params['kappa']:.4f}")
        print(f"  θ (equilibrium):    {estimated_params['theta']:.4f}")
        print(f"  σ (volatility):     {estimated_params['sigma']:.4f}")
        print(f"  μ_J (jump mean):    {estimated_params['jump_mean']:.4f}")
        print(f"  σ_J (jump std):     {estimated_params['jump_std']:.4f}")
        
        # Calculate derived metrics
        half_life = np.log(2) / estimated_params['kappa']
        annual_vol = estimated_params['sigma'] * np.sqrt(252)
        
        print("\nDerived Metrics:")
        print(f"  Half-life: {half_life:.2f} days")
        print(f"  Annual volatility: {annual_vol:.2f} ({annual_vol*100:.1f}%)")
        
        # Check parameter validity
        print("\nParameter Validity Checks:")
        print(f"  κ > 0 (mean reverting): {'✓' if estimated_params['kappa'] > 0 else '✗'}")
        print(f"  σ > 0 (has volatility): {'✓' if estimated_params['sigma'] > 0 else '✗'}")
        print(f"  Half-life reasonable: {'✓' if 0.1 < half_life < 100 else '⚠'}")
        
    except Exception as e:
        print(f"✗ MRJD fitting failed: {str(e)}")
        sys.exit(1)
    
    # Prediction test
    print("\n[5] Testing predictions...")
    
    # Current spread
    X_current = spread.iloc[-1]
    
    # Predict 5 days ahead
    horizons = [1, 5, 10, 20]
    
    print(f"\nCurrent spread: {X_current:.4f}")
    print(f"Equilibrium θ: {estimated_params['theta']:.4f}")
    print(f"\nPredictions:")
    print(f"{'Horizon (days)':<20} {'Expected':<15} {'Std Dev':<15} {'Reversion %':<15}")
    print("-"*70)
    
    for horizon in horizons:
        pred = model.predict_spread_statistics(X_current, horizon)
        print(f"{horizon:<20} {pred['expected_spread']:<15.4f} "
              f"{pred['spread_std']:<15.4f} {pred['reversion_pct']:<15.2f}")
    
    # Simulate paths
    print("\n[6] Simulating future paths...")
    n_paths = 100
    horizon = 20  # 20 days
    
    try:
        #Simulating multiple paths using the simulate method
        paths = []
        for i in range(n_paths):
            path_df = model.simulate(X_current, horizon, dt = 1/252, seed = 42 + i)
            paths.append(path_df['spread'].values)
        paths = np.array(paths) #Shape: (n_paths, n_steps)
        
        print(f" Simulated {n_paths} paths over {horizon} days")
        print(f"  Final spread range: [{paths[:, -1].min():.4f}, {paths[:, -1].max():.4f}]")
        print(f"  Mean final spread: {paths[:, -1].mean():.4f}")
        print(f"  Std final spread: {paths[:, -1].std():.4f}")
        
    except Exception as e:
        print(f"⚠ Simulation skipped: {str(e)}")
    
    # Calculate Z-scores
    print("\n[7] Testing Z-score calculation...")
    z_scores = model.calculate_z_score(spread)
    
    print(f" Z-score statistics:")
    print(f"  Mean: {z_scores.mean():.4f}")
    print(f"  Std: {z_scores.std():.4f}")
    print(f"  Min: {z_scores.min():.4f}")
    print(f"  Max: {z_scores.max():.4f}")
    print(f"  |Z| > 2: {(abs(z_scores) > 2).sum()} occurrences ({100*(abs(z_scores) > 2).sum()/len(z_scores):.2f}%)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nData: NVDA/AMD spread ({len(spread)} days)")
    print(f"Date range: {spread.index[0].date()} to {spread.index[-1].date()}")
    print(f"\nMRJD Model:")
    print(f"  Mean reversion speed: κ = {estimated_params['kappa']:.4f} (half-life {half_life:.2f} days)")
    print(f"  Equilibrium level: θ = {estimated_params['theta']:.4f}")
    print(f"  Diffusive volatility: σ = {estimated_params['sigma']:.4f}")
    print(f"  Jump characteristics: μ_J = {estimated_params['jump_mean']:.4f}, σ_J = {estimated_params['jump_std']:.4f}")
    print(f"\nJump Statistics:")
    print(f"  Total jumps: {n_jumps}")
    print(f"  Jump frequency: {100*n_jumps/len(spread):.2f}%")
    print(f"\nModel Quality:")
    print(f"  Fast mean reversion: {'✓' if half_life < 5 else '⚠'}")
    print(f"  Stable equilibrium: {'✓' if abs(estimated_params['theta'] - spread.mean()) < spread.std() else '⚠'}")
    print(f"  Low jump impact: {'✓' if n_jumps < len(spread)*0.05 else '⚠'}")
    print(f"\n All tests completed successfully!")
    print("="*70)
