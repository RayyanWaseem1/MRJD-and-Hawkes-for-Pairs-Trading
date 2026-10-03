"""
Mean-Reverting Jump Diffusion (MRJD) estimation.

    dX_t = kappa (theta - X_t) dt + sigma dW_t + Y dN_t,   N_t ~ Hawkes

Time unit: TRADING DAYS. dt = 1.0

Changes from the prior version:
1. UNITS. `dt = 1/252` made kappa a per-YEAR rate, so `log(2)/kappa` was a 
    half-life in YEARS -- then compared against, and used to overwrite, an 
    empirical half-life measured in TRADING DAYS. The committed GS/MS bundle
    recorded `kappa = 0.01228` (ln2/kappa = 56.43 years, = 14,220 trading days)
    next to `half_life = 56.48` days and the pipeline reported them as agreeing
    to two decimal places. `dt` is now 1.0 and `assert_trading_day_units` makes
    a reintroduction fail loudly

2. REPARAMETERIZED OU. The old fit optimized (kappa, theta, sigma) directly.
    A 56-day half-life on daily data is an AR(1) coefficient of 0.988 -- nearly 
    a unit root -- where the likelihood is very flat in kappa, and the optimizer
    duly wandered: GS/MS produced `sigma = 3.656`, implying a stationary
    standard deviation sigma/sqrt(2 kappa) = 23.3 against an actual spread
    standard deviation of 1.49, a 15.6x mismatch; AMD/NVDA pinned sigma at its
    hardcoded upper bound of 5.0.

    The model is now fitted in the AR(1) parameterization

        X_{t+1} | X_t ~ N( theta + (X_t - theta) phi,  s^2 (1 - phi^2) )
        phi = exp(-kappa dt),   s^2 = sigma^2 / (2 kappa)   (stationary variance)

    which has an EXACT closed-form conditional MLE (it is a linear regression),
    no flat direction, and no optimizer to fail. Kappa and sigma are recovered 
    by inversion. 

3. FAILURES RAISE. `result.success` used to be checked only to print a
    warning. Estimation problems now raise `MRJDFitError`

4. NO SILENT OVERRIDE. The old code replaced the fitted kappa with 
    `ln(2)/empirical_half_life` whenever the two disagreed by >50%, printing a 
    reassuring "validation passed" message the rest of the time. The check is 
    retained as a DIAGNOSTIC -- it reports the discrepancy and can raise -- but
    it no longer silently overwrites a parameter 

5. THE DEAD CODE IS LIVE. `_joint_mle`, `simulate` and
    `predic_spread_statistics` were never called from anywhere. They are now
    used by the joint refinement path, the synthetic-recovery test, and the
    MRJD-based signal path respectively
"""

from __future__ import annotations

from typing import Dict, Optional 

import numpy as np
import pandas as pd
from scipy.optimize import minimize 

from time_units import DT, TIME_UNIT, assert_trading_day_units

__all__ = ["MRJDModel", "MRJDFitError"]

class MRJDFitError(RuntimeError):
    """ Raised when MRJD estimation fails or produces an unusable parameter set"""

class MRJDModel:
    """ Mean-reverting jump diffusion with separately estimated jump sizes"""

    def __init__(self, verbose: bool = True):
        self.params: Dict[str, float] = {}
        self.ou_params: Dict[str, float] = {}
        self.jump_params: Dict[str, float] = {}
        self.diagnostics: Dict = {}
        self.fitted = False 
        self.verbose = verbose 

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    # ------ #

    def fit(
        self,
        spread: pd.Series,
        jump_indicator: pd.Series,
        dt: float = DT,
        method: str = "MLE",
        joint_refinement: bool = False,
        empirical_half_life: Optional[float] = None,
        raise_on_half_life_mismatch: bool = False,
    ) -> Dict:
        """
        Fit the MRJD parameters

        Params:
        dt: float 
            Must be 1.0 (trading days). Guarded
        joint_refinement: bool 
            Run the joint MLE over (kappa, theta, sigma, mu_J, sigma_j) starting
            from the separate estimates. The audited pipeline printed 
            "Step 3: Joint MLE refinement" and then immediately skipped it. 
        empirical_half_life: float, optional
            AR(1) half-life in trading days, for the consistency diagnostic
        """
        assert_trading_day_units(dt, context = "MRJDModel.fit")

        self._log("Fitting MRJD model")

        idx = spread.index.intersection(jump_indicator.index)
        spread = spread.loc[idx].astype(float)
        jump_indicator = jump_indicator.loc[idx].astype(int)

        if len(spread) < 30:
            raise MRJDFitError(f"Only {len(spread)} observations; too few for MRJD")
        
        self._log(" [1/3] OU parameters (AR(1) closed-form conditional MLE)")
        self.ou_params = self._estimate_ou_parameters(spread, jump_indicator, dt)

        self._log(" [2/3] Jump size distribution")
        self.jump_params = self._estimate_jump_parameters(spread, jump_indicator)

        params = {**self.ou_params, **self.jump_params}

        if joint_refinement:
            self._log(" [3/3] Joint MLE refinement")
            params = self._joint_mle(spread, jump_indicator, dt, params)
        else:
            self._log(" [3/3] Joint MLE refinement: disabled by caller")

        self.params = params 
        self.fitted = True 

        # -- consistency diagnostics (report, never silently overwrite) -- #
        kappa = params["kappa"]
        model_half_life = float(np.log(2) / kappa) if kappa > 0 else float("inf")

        implied_sd = params["sigma"] / np.sqrt(2 * kappa) if kappa > 0 else float("inf")
        actual_sd = float(spread.std())
        sd_ratio = implied_sd / actual_sd if actual_sd > 0 else float("inf")

        self.diagnostics = {
            "model_half_life_days": model_half_life,
            "empirical_half_life_days": empirical_half_life,
            "implied_stationary_sd": float(implied_sd),
            "actual_spread_sd": actual_sd,
            "sd_ratio": float(sd_ratio),
            "time_units": TIME_UNIT,
            "dt": dt,
        }

        if empirical_half_life is not None and np.isfinite(empirical_half_life) and empirical_half_life > 0:
            rel = abs(model_half_life - empirical_half_life) / empirical_half_life
            self.diagnostics["half_life_relative_error"] = float(rel)
            if rel > 0.5:
                message = (
                    f"MRJD half-life {model_half_life:.1f}d vs empirical "
                    f"{empirical_half_life:.1f}d ({rel:.0%} apart). Reported, NOT "
                    "overwritten -- the audited code silently replaced kappa here."
                )
                self._log(f" DIAGNOSTIC: {message}")
                if raise_on_half_life_mismatch:
                    raise MRJDFitError(message)
            else:
                self._log(
                    f" half-life check: model {model_half_life:.1f}d vs "
                    f"empirical {empirical_half_life:.1f}d ({rel:.0%} apart)"
                )

        if not (0.5 <= sd_ratio <= 2.0):
            self._log(
                f" DIAGNOSTIC: implied stationary sd {implied_sd:.3f} vs actual "
                f"{actual_sd:.3f} ({sd_ratio:.1f}x). The OU fit does not describe "
                "the data's dispersion."
            )

        self._log(" MRJD parameters:")
        self._log(f" kappa = {params['kappa']:.6f} (half-life {model_half_life:.1f} {TIME_UNIT}s)")
        self._log(f" theta = {params['theta']:.4f}")
        self._log(f" sigma = {params['sigma']:.6f}")
        self._log(f" mu_J = {params['jump_mean']:.4f}")
        self._log(f" sigma_j = {params['jump_std']:.4f}")

        return self.params 

    # ---- # 

    def _estimate_ou_parameters(
            self, spread: pd.Series, jump_indicator: pd.Series, dt: float
    ) -> Dict[str, float]:
        """
        Exact conditional MLE in the AR(1) parameterization 
        
            X_{t+1} = c + phi X_t + e, e ~ N(0, v)
            theta = c / (1 - phi)
            s^2 = v / (1 - phi^2) (stationary variance)
            kappa = -log(phi) / dt
            sigma = s * sqrt(2 kappa)
            
        Transitions that END on a detected jump are excluded, so the diffusion
        parameters are estimated from the continous part only.
        """

        values = spread.to_numpy(dtype = float)
        jumps = jump_indicator.to_numpy(dtype = int)

        x_t = values[:-1]
        x_next = values[1:]
        jump_next = jumps[1:]

        keep = jump_next == 0
        if keep.sum() < 20:
            self._log(
                f" only {int(keep.sum())} non-jump transitions; using all transitions"
            )
            keep = np.ones_like(jump_next, dtype = bool)

        x0, x1 = x_t[keep], x_next[keep]

        design = np.column_stack([np.ones_like(x0), x0])
        coeffs, *_ = np.linalg.lstsq(design, x1, rcond = None)
        c ,phi = float(coeffs[0]), float(coeffs[1])

        residuals = x1 - (c + phi * x0)
        dof = max(len(x1) - 2, 1)
        v = float(residuals @ residuals / dof)

        if not np.isfinite(phi):
            raise MRJDFitError("AR(1) coefficient is not finite")

        if phi <= 0.0:
            raise MRJDFitError(
                f"AR(1) coefficient phi = {phi:.4f} <= 0: the spread is not "
                "mean-reverting in the OU sense, so kappa = -log(phi) is undefined."
            )

        if phi >= 1.0:
            raise MRJDFitError(
                f" AR(1) coefficient phi = {phi:.6f} >= 1: the spread has a unit root "
                "and no finite mean-reversion half-life. An OU model is the wrong "
                "specificaiton for this series."
            )

        kappa = float(-np.log(phi) / dt)
        theta = float(c / (1.0 - phi))
        stationary_var = v / (1.0 - phi ** 2)
        sigma = float(np.sqrt(max(stationary_var, 1e-18) * 2.0 * kappa))

        self._log(
            f" phi = {phi:.6f} -> kappa = {kappa:.6f}, "
            f"half-life = {np.log(2)/kappa:.1f} {TIME_UNIT}s"
        )

        return {
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma,
            "phi": phi, 
            "stationary_sd": float(np.sqrt(stationary_var)),
            "n_transitions_used": int(keep.sum()),
        }

    def _estimate_jump_parameters(
            self, spread: pd.Series, jump_indicator: pd.Series
    ) -> Dict[str, float]:
        """
        Jump sizes from the spread's first difference on detected jump days.

        With Lee-Mykland these are genuinely the jump days. Under the audited 
        BNS-on-a-rolling-window detector they were the LAST days of 20-day 
        windows containing a jump, i.e. near-arbitrary daily changes
        """
        diffs = spread.diff()
        sizes = diffs[jump_indicator == 1].dropna()

        if len(sizes) == 0:
            self._log(" no jumps detected; jump distribution set to degenerate")
            return {"jump_mean": 0.0, "jump_std": 0.0, "n_jumps_used":0}

        mean = float(sizes.mean())
        std = float(sizes.std(ddof = 1)) if len(sizes) > 1 else 0.0
        if not np.isfinite(std) or std <= 0:
            std = float(abs(mean)) if mean != 0 else float(spread.diff().std())

        return {"jump_mean": mean, "jump_std": std, "n_jumps_used": int(len(sizes))}

    def _joint_mle(
        self,
        spread: pd.Series,
        jump_indicator: pd.Series,
        dt: float,
        start: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Joint MLE over (phi, theta, s, mu_J, sigma_J), started from the separate
        estimates. Optimized in the same stable AR(1) parameterization
        """
        values = spread.to_numpy(dtype = float)
        jumps = jump_indicator.to_numpy(dtype = int)
        x_t, x_next, jump_next = values[:-1], values[1:], jumps[1:]

        def unpack(u):
            phi = 1.0 / (1.0 + np.exp(-u[0]))
            theta = u[1]
            s = np.exp(np.clip(u[2], -30, 30))
            mu_j = u[3]
            sd_j = np.exp(np.clip(u[4], -30, 30))
            return phi, theta, s, mu_j, sd_j


        def neg_ll(u):
            phi, theta, s, mu_j, sd_j = unpack(u)
            if not (0 < phi < 1) or s <= 0 or sd_j <= 0:
                return 1e12
            mean = theta + (x_t - theta) * phi 
            var = (s**2) * (1 - phi**2)
            if var <= 0:
                return 1e12
            mean = np.where(jump_next == 1, mean + mu_j, mean)
            var = np.where(jump_next == 1, var + sd_j**2, var)
            resid = x_next - mean 
            return float(0.5 * np.sum(np.log(2 * np.pi * var) + resid**2 / var))

        phi0 = float(start["phi"])
        phi0 = float(np.clip(phi0, 1e-6, 1 - 1e-9))
        s0 = max(float(start["stationary_sd"]), 1e-9)
        sdj0 = max(float(start["jump_std"]), 1e-9)

        u0 = np.array(
            [
                np.log(phi0 / (1 - phi0)),
                float(start["theta"]),
                np.log(s0),
                float(start["jump_mean"]),
                np.log(sdj0),
            ]
        )

        result = minimize(neg_ll, u0, method = "L-BFGS-B", options = {"maxiter":2000})
        if not result.success or not np.isfinite(result.fun):
            raise MRJDFitError(f"Joint MLE did not converge: {result.message}")

        phi, theta, s, mu_j, sd_j = unpack(result.x)
        kappa = float(-np.log(phi) / dt)

        return {
            "kappa": kappa,
            "theta": float(theta),
            "sigma": float(s * np.sqrt(2 * kappa)),
            "phi": float(phi),
            "stationary_sd": float(s),
            "jump_mean": float(mu_j),
            "jump_std": float(sd_j),
            "joint_log_likelihood": float(-result.fun),
            "n_jumps_used": int(start.get("n_jumps_used", 0)),
            "n_transitions_used": int(start.get("n_transitions_used", len(x_t))),
        }

    
    ### model outputs that are actually used ###

    def calculate_z_score(self, X: pd.Series) -> pd.Series:
        """
        Parametric z-score against the OU stationary distribution:

            Z_t = (X_t - theta) / (sigma / sqrt(2 kappa))
        """
        self._require_fit()
        theta, sigma, kappa = self.params["theta"], self.params["sigma"], self.params["kappa"]
        stationary_sd = sigma / np.sqrt(2 * kappa)
        if stationary_sd <= 0:
            raise MRJDFitError("Non-positive stationary standard deviation")
        return (X - theta) / stationary_sd

    def predict_spread_statistics(self, X_current: float, horizon: float) -> Dict:
        """
        Conditional OU forecast, used for the expected-reversion holding period

            E[X_h | X_0] = theta + (X_0 - theta) exp(-kappa h)
            Var[X_h | X_0] = sigma^2 (1 - exp(-2 kappa h)) / (2 kappa)
        """
        self._require_fit()
        kappa, theta, sigma = (
            self.params["kappa"],
            self.params["theta"],
            self.params["sigma"],
        )
        expected = theta + (X_current - theta) * np.exp(-kappa * horizon)
        variance = (sigma**2) * (1 - np.exp(-2 * kappa * horizon)) / (2 * kappa)
        return {
            "expected_spread": float(expected),
            "spread_std": float(np.sqrt(max(variance, 0.0))),
            "half-life": float(np.log(2) / kappa),
            "reversion_pct": float(100 * (1 - np.exp(-kappa * horizon))),
        }

    def expected_reversion_time(self, z_from: float, z_to: float) -> float:
        """
        Expected trading days for the OU mean to carry |z| from `z_from` to `z_to`.

            t = ln(z_from / z_to) / kappa 

        Used to set a model-based holding period instead of a fixed multiple of 
        the half-life
        """
        self._require_fit()
        if z_to <= 0 or z_from <= 0 or z_to >= z_from:
            return float(np.log(2) / self.params["kappa"])
        return float(np.log(z_from / z_to) / self.params["kappa"])

    def simulate(
        self,
        X0: float,
        T: float,
        dt: float = DT,
        jump_times: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """ Exact discretization OU simulation with optional jumps at given time"""
        self._require_fit()
        assert_trading_day_units(dt, context = "MRJDModel.simulate")

        rng = np.random.default_rng(seed)
        kappa, theta, sigma = (
            self.params["kappa"],
            self.params["theta"],
            self.params["sigma"],
        )
        mu_j, sd_j = self.params["jump_mean"], self.params["jump_std"]

        n_steps = int(T / dt)
        phi = np.exp(-kappa * dt)
        cond_sd = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

        jump_positions = set()
        if jump_times is not None:
            jump_positions = {
                int(round(t / dt)) for t in jump_times if 0 <= int(round(t / dt)) < n_steps
            }

        path = np.zeros(n_steps)
        path[0] = X0
        flags = np.zeros(n_steps, dtype = int)

        for i in range(1, n_steps):
            path[i] = theta + (path[i-1] - theta) * phi + rng.normal(0.0, cond_sd)
            if i in jump_positions:
                path[i] += rng.normal(mu_j, sd_j)
                flags[i] = 1

        return pd.DataFrame(
            {"time": np.arange(n_steps) * dt, "spread": path, "jump_indicator": flags}
        )

    def _require_fit(self) -> None:
        if not self.fitted:
            raise MRJDFitError("Model not fitted. Call fit() first")


if __name__ == "__main__":
    print("=" * 72)
    print("MRJD estimation -- synthetic recovery check")
    print("=" * 72)

    rng = np.random.default_rng(11)
    true_kappa, true_theta, true_sigma = 0.04, 1.5, 0.06
    n = 3000

    phi = np.exp(-true_kappa * DT)
    cond_sd = true_sigma * np.sqrt((1 - np.exp(-2 * true_kappa * DT)) / (2 * true_kappa))
    x = np.zeros(n)
    x[0] = true_theta 
    for i in range(1, n):
        x[i] = true_theta + (x[i - 1] - true_theta) * phi + rng.normal(0, cond_sd)

    idx = pd.RangeIndex(n)
    spread = pd.Series(x, index = idx)
    no_jumps = pd.Series(0, index = idx)

    model = MRJDModel()
    fitted = model.fit(spread, no_jumps, dt = DT, empirical_half_life=np.log(2) / true_kappa)

    print(f"\n{'param':<10}{'true':>10}{'fitted':>12}{'err %':>10}")
    for key, truth in (("kappa", true_kappa), ("theta", true_theta), ("sigma", true_sigma)):
        got = fitted[key]
        print(f"{key:<10}{truth:>10.4f}{got:>12.4}{100*abs(got-truth)/truth:>10.1}")

    print(f"\n half-life: true {np.log(2)/true_kappa:.1f}d, fitted "
          f"{np.log(2)/fitted['kappa']:.1f}d")
    print(f"implied stationary sd {model.diagnostics['implied_stationary_sd']:.4f} "
          f"vs actual {model.diagnostics['actual_spread_sd']:.4f} "
          f"({model.diagnostics['sd_ratio']:.2f}x)")
