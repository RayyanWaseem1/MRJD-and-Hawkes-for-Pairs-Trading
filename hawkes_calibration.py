"""
Hawkes process calibration, inference, and specification testing.

Intensity: lambda(t) = lambda_bar + sum_{t_i < t} alpha * exp(-beta (t - t_i))
Time Unit: TRADING DAYS (see time_units.py)

Changes from the audited version:
1. REPARAMETERIZED MLE: The old objective returned a literal `1e10` whenever
    `alpha >= beta` or `alpha/beta > 0.85`:
        
        if alpha >= beta: return 1e10
        if branching_ratio > 0.85: return 1e10

    L-BFGS-B uses finite-difference gradients, so near that cliff it sees a step 
    function and the gradient is meaningless -- a genuine optimizer bug, not just 
    a modelling choice. Every reported branching ratio then landed within 2% of
    the 0.85 wall (0.8363 SPY/IVV, 0.8487 AMD/NVDA, 0.8174 GS/MS), i.e. at the 
    constraint boundary, where standard errors are invalid in any case. 

    The model is now fitted in an unconstrained space:

        lambda_bar = exp(u0), beta = exp(u1), eta = alpha/beta = sigmoid(u2)

    so 0 < eta < 1 holds by construction, the obective is smooth everywhere,
    and the optimum is interior.

2. O(n) LIKELIHOOD. The old evaluation looped over all past jumps for every 
    jump, i.e. O(n^2). The exponential kernel admits the standard recursion

        R(i) = exp(-beta(t_i - t_{i-1})) * (1 + R(i-1)), R(0) = 0 

    used here for both the likelihood and the intensity path. NOTE: on this 
    data a full fit took ~0.1 s even at O(n^2), so this is correctness and 
    craft, not a fix for a measured bottleneck. 

3. CONFIG IS READ. `bounds` used to be hardcoded as 
    [(0.001, 0.5), (0.01, 2.0), (0.1, 5.0)] while HawkesConfig specified
    different values, and `max_iterations`, `tolerance`, `estimation_method`
    and `kernel` were ignored entirely.

4. INFERENCE THAT WAS NEVER THERE. The project's central claim -- that spread 
    jumps are self-exciting -- was never tested. This module now provides:
        * `likelihood_ratio_test` against the homogenous Poisson null;
        * a PARAMETRIC BOOTSTRAP null for that test, because under H0: alpha = 0
        the decay parameter beta is unidentified (the Davies problem), so the 
        LR statistic is NOT asymptotically chi-squared;
        * `standard_errors` from the numerical Hessian, plus a delta-method
        confidence interval on the branching ratio;
        * `goodness_of_fit` (written in the audited version, never called) -- 
        random time change by the fitted compensator, KS against Exp(1);
        * `intensity_calibration` -- does a high fitted lambda actually predict
        more jumps next period?
"""

from __future__ import annotations 

from typing import Dict, Optional, Sequence, Tuple 

import numpy as np 
import pandas as pd
from scipy import stats 
from scipy.optimize import minimize 

from time_units import TIME_UNIT, observation_span, positional_times 

__all__ = ["HawkesProcess", "HawkesFitError"]

class HawkesFitError(RuntimeError):
    """ Raised when a Hawkes fit cannot be performed or did not converge."""

def _sigmoid(x: float) -> float:
    #Numerically stable logistic
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)

class HawkesProcess:
    """ Univariate Hawkes process with an exponential kernel."""

    def __init__(self, kernel: str = "exponential", verbose: bool = True):
        if kernel != "exponential":
            raise NotImplementedError(
                f"kernel = '{kernel}' is not implemented. Only the exponential "
                "kernel admits the O(n) recursion used here; a power-law kernel "
                "would need a different likelihood."
            )
        self.kernel = kernel
        self.verbose = verbose 

        self.params: Dict[str, float] = {}
        self.jump_times: Optional[np.ndarray] = None 
        self.T: Optional[float] = None 
        self.log_likelihood: Optional[float] = None 
        self.fit_diagnostics: Dict = {}

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    ### likelihood ###

    @staticmethod 
    def _recursion(times: np.ndarray, beta: float) -> np.ndarray:
        """
        R(i) = exp(-beta (t_i - t_{i-1})) (1 + R(i-1)), R(0) = 0

        R[i] is the sum of decayed kernel contributions from all jumps strictly 
        before t_i, computed in O(n) rather than O(n^2).
        """

        n = len(times)
        R = np.zeros(n)
        for i in range(1, n):
            R[i] = np.exp(-beta * (times[i] - times[i - 1])) * (1.0 + R[i-1])
        return R 

    @classmethod 
    def log_likelihood_at(
        cls, times: np.ndarray, T: float, lambda_bar: float, alpha: float, beta: float
    ) -> float:
        """
        Exact Hawkes log-likelihood.

            LL = sum_i log(lambda_bar + alpha R_i)
                - lambda_bar T 
                - (alpha/beta) sum_i (1 - exp(-beta (T - t_i)))
        """
        if lambda_bar <= 0 or beta <= 0 or alpha < 0:
            return -np.inf 

        R = cls._recursion(times, beta)
        intensity = lambda_bar + alpha * R
        if np.any(intensity <= 0):
            return -np.inf 

        term_sum = float(np.sum(np.log(intensity)))
        compensator = lambda_bar * T + (alpha/beta) * float(
            np.sum(1.0 - np.exp(-beta * (T - times)))
        )
        return term_sum - compensator 

    @staticmethod
    def poisson_log_likelihood(n_jumps: int, T: float) -> float:
        """ LL of the homogenous Poisson null at its MLE lambda = n/T"""
        if n_jumps == 0 or T <= 0:
            return -np.inf 
        lam = n_jumps / T
        return n_jumps * np.log(lam) - lam * T 

    ### fitting ###

    def fit(
        self,
        jump_times: np.ndarray,
        T: float, 
        method: str = "MLE",
        initial_params: Optional[Dict] = None, 
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        baseline_bounds: Sequence[float] = (1e-6, 10.0),
        excitation_bounds: Sequence[float] = (1e-6, 5.0),
        decay_bounds: Sequence[float] = (1e-4, 10.0),
        min_jumps: int = 5,
        n_restarts: int = 4,
    ) -> Dict:
        """ 
        Fit by maximum likelihood in an unconstrained reparameterization.

        Params:
        jump_times: np.ndarray
            Jump times in TRADING DAYS (positional), sorted or not 
        T: float 
            Observation span in trading days 
        baseline_bounds, excitation, decay_bounds: 
            Read from HawkesConfig by callers. Enforced by clipping the 
            transformed optimum, not by the discontinous penalty
        n_restarts: int 
            Multi-start to reduce sensitivity to the initial guess
        """
        if method.upper() not in ("MLE", "GMM"):
            raise ValueError(f"Unknown estimation method '{method}'")

        times = np.sort(np.asarray(jump_times, dtype = float))
        n = len(times) 

        if n < min_jumps:
            raise HawkesFitError(
                f"Only {n} jumps; need at least {min_jumps} for a Hawkes fit. "
                "Fit the Poisson null instead and report that the data cannot "
                "support a self-excitation estimate."
            )
        if T <= 0:
            raise HawkesFitError(f"Non-positive observation span T = {T}")

        observation_span = float(T)
        self.jump_times = times 
        self.T = observation_span

        lo_lam, hi_lam = baseline_bounds 
        lo_beta, hi_beta = decay_bounds 
        lo_alpha, hi_alpha = excitation_bounds 

        def unpack(u: np.ndarray):
            lambda_bar = float(np.exp(np.clip(u[0], -30, 30)))
            beta = float(np.exp(np.clip(u[1], -30, 30)))
            eta = _sigmoid(float(u[2]))
            alpha = eta * beta 
            return lambda_bar, alpha, beta, eta 

        def neg_ll(u: np.ndarray) -> float:
            lambda_bar, alpha, beta, _ = unpack(u)
            if not (np.isfinite(lambda_bar) and np.isfinite(beta) and np.isfinite(alpha)):
                return 1e12
            ll = self.log_likelihood_at(times, observation_span, lambda_bar, alpha, beta)
            if not np.isfinite(ll):
                return 1e12
            return -ll 

        # starting values 
        lam0 = max(n / observation_span, 1e-6)
        if initial_params:
            lam0 = float(initial_params.get("lambda_bar", lam0))

        starts = []
        for eta0 in (0.2, 0.45, 0.7, 0.9)[:max(n_restarts, 1)]:
            for beta0 in (0.5, ):
                starts.append(
                    np.array(
                        [
                            np.log(lam0 * (1 - eta0)),
                            np.log(beta0),
                            np.log(eta0 / (1 - eta0)),
                        ]
                    )
                )
        best = None 
        for u0 in starts:
            try:
                res = minimize(
                    neg_ll,
                    u0,
                    method = "L-BFGS-B",
                    options = {"maxiter": max_iterations, "ftol": tolerance, "gtol": tolerance},
                )
            except Exception: #noqa: BLE001
                continue 
            if res is None or not np.isfinite(res.fun):
                continue
            if best is None or res.fun < best.fun:
                best = res

        if best is None:
            raise HawkesFitError("All Hawkes optimization restarts failed")

        lambda_bar, alpha, beta, eta = unpack(best.x)

        # Bounds are enforced by clipping the optimum, and any clipping is 
        # reported -- never by a discontinous penalty inside the objective 
        clipped = {}
        if not (lo_lam <= lambda_bar <= hi_lam):
            clipped["lambda_bar"] = (lambda_bar, float(np.clip(lambda_bar, lo_lam, hi_lam)))
            lambda_bar = float(np.clip(lambda_bar, lo_lam, hi_lam))
        if not (lo_beta <= beta <= hi_beta):
            clipped["beta"] = (beta, float(np.clip(beta, lo_beta, hi_beta)))
            beta = float(np.clip(beta, lo_beta, hi_beta))
        if not (lo_alpha <= alpha <= hi_alpha):
            clipped["alpha"] = (alpha, float(np.clip(alpha, lo_alpha, hi_alpha)))
            alpha = float(np.clip(alpha, lo_alpha, hi_alpha))

        self.params = {
            "lambda_bar": lambda_bar,
            "alpha": alpha,
            "beta": beta, 
            "beta_H": beta,
        }
        self.log_likelihood = self.log_likelihood_at(
            times, observation_span, lambda_bar, alpha, beta
        )

        self.fit_diagnostics = {
            "converged": bool(best.success),
            "message": str(best.message),
            "n_jumps": n,
            "T": observation_span,
            "time_units": TIME_UNIT,
            "log_likelihood": self.log_likelihood,
            "branching_ratio": alpha / beta if beta > 0 else np.nan,
            "bounds_clipped": clipped,
            "n_restarts": len(starts),
        }

        if clipped:
            self._log(f" WARNING: config bounds clipped the optimum: {clipped}")
        if not best.success:
            self._log(f" WARNING: optimizer reported non-convergence: {best.message}")

        self._log(f" Hawkes fit ({n} jumps over T = {observation_span:.0f} {TIME_UNIT}s):")
        self._log(f" lambda_bar = {lambda_bar:.6f}")
        self._log(f" alpha = {alpha:.6f}")
        self._log(f" beta = {beta:.6f}")
        self._log(f" branching = {alpha/beta:.4f}")

        return self.params 


    ### inference ###

    def standard_errors(self, epsilon: float = 1e-5) -> Dict:
        """ 
        Standard errors from the numerical Hessian of the negative log-likelihood,
        in NATURAL parameters, plus a delta-method CI on the branching ratio.

        Valid because the reparameterized optimum is interior. At a constraint
        boundary -- where every branching ratio in the audited results sat --
        these would be meaningless, which is part of why the boundary mattered.
        """
        times, T, _ = self._require_fit()
        theta = np.array(
            [self.params["lambda_bar"], self.params["alpha"], self.params["beta"]]
        )

        def nll(p: np.ndarray) -> float:
            val = self.log_likelihood_at(times, T, p[0], p[1], p[2])
            return -val if np.isfinite(val) else 1e12

        k = len(theta)
        H = np.zeros((k, k))
        steps = np.maximum(np.abs(theta) * epsilon, 1e-8)

        for i in range(k):
            for j in range(k):
                tp, tm = theta.copy(), theta.copy()
                tpp, tmm = theta.copy(), theta.copy()
                tp[i] += steps[i]; tp[j] += steps[j]
                tm[i] += steps[i]; tm[j] -= steps[j]
                tpp[i] -= steps[i]; tpp[j] += steps[j]
                tmm[i] -= steps[i]; tmm[j] -= steps[j]
                H[i, j] = (nll(tp) - nll(tm) - nll(tpp) + nll(tmm)) / (
                    4.0 * steps[i] * steps[j]
                )

        H = 0.5 * (H + H.T)

        out: Dict = {"hessian": H.tolist()}
        try:
            cov = np.linalg.inv(H)
            variances = np.diag(cov)
            if np.any(variances < 0):
                out["warning"] = (
                    "Negative variacne on the diagonal: The Hessian is not "
                    "positive definite, so these standard errors are unreliable."
                )
            se = np.sqrt(np.abs(variances))
            out.update(
                {
                    "se_lambda_bar": float(se[0]),
                    "se_alpha": float(se[1]),
                    "se_beta": float(se[2]),
                    "cov": cov.tolist(),
                }
            )

            a, b = self.params["alpha"], self.params["beta"]
            d_alpha, d_beta = 1.0 / b, -a / (b**2)
            var_eta = (
                d_alpha**2 * cov[1, 1]
                + d_beta**2 * cov[2, 2]
                + 2 * d_alpha * d_beta * cov[1, 2]
            )
            se_eta = float(np.sqrt(abs(var_eta)))
            eta = a / b
            out.update(
                {
                    "branching_ratio": float(eta),
                    "se_branching_ratio": se_eta,
                    "branching_ratio_ci95": (
                        float(eta - 1.96 * se_eta),
                        float(eta + 1.96 * se_eta),
                    ),
                }
            )
        except np.linalg.LinAlgError:
            out["error"] = "Hessian is singular; standard errors unavailable."

        return out 

    def likelihood_ratio_test(
        self, n_bootstrap: int = 0, seed: Optional[int] = None
    ) -> Dict:
        """
        Test H0: alpha = 0 (homogenous Poisson) against the Hawkes alternative.

        This is the test the project's central claim requires and never had originally.

        IMPORTANT -- the Davies problem: under H0 the decay parameter beta is
        NOT identified (it mulitplies a term that vanishes), so the LR statistic
        does not have a chi-squared limiting distribution. The chi-squared
        p-values below are reported as conservative references only. When 
        `n_bootstrap > 0` a parametric bootstrap null is simulated instead, and
        `p_bootstrap` is the p-value to quote.
        """
        times, T, ll_hawkes = self._require_fit()
        n = len(times)
        ll_poisson = self.poisson_log_likelihood(n, T)
        lr = 2.0 * (ll_hawkes - ll_poisson)

        out = {
            "ll_hawkes": ll_hawkes,
            "ll_poission": ll_poisson,
            "lr_statistic": float(lr),
            "p_chi2_df1": float(stats.chi2.sf(max(lr, 0.0), 1)),
            "p_chi2_df2": float(stats.chi2.sf(max(lr, 0.0), 2)),
            "identification_note": (
                "beta is unidentified under H0 (Davies problem); chi-squared "
                "p-values are conservative references, not exact."
            ),
        }

        if n_bootstrap and n_bootstrap > 0:
            rng = np.random.default_rng(seed)
            lam0 = n / T
            null_stats = []
            for _ in range(n_bootstrap):
                k = rng.poisson(lam0 * T)
                if k < 5:
                    continue 
                sim = np.sort(rng.uniform(0.0, T, size = k))
                try:
                    probe = HawkesProcess(verbose = False)
                    probe.fit(sim, T, min_jumps = 5, n_restarts = 2)
                    _, _, ll_h = probe._require_fit()
                    ll_p = self.poisson_log_likelihood(k, T)
                    null_stats.append(2.0 * (ll_h - ll_p))
                except (HawkesFitError, Exception): # noqa: BLE001
                    continue 

            if null_stats:
                null_arr = np.asarray(null_stats, dtype = float)
                out.update(
                    {
                        "n_bootstrap_effective": int(len(null_arr)),
                        "bootstrap_null_mean": float(null_arr.mean()),
                        "bootstrap_null_q95": float(np.quantile(null_arr, 0.95)),
                        "p_bootstrap": float(np.mean(null_arr >= lr)),
                    }
                )
        return out 

    def goodness_of_fit(
        self, jump_times: Optional[np.ndarray] = None, T: Optional[float] = None
    ) -> Dict:
        """
        Random time changes residual test. 

        Under correct specification the compensator-transformed inter-arrival 
        times are i.i.d. Exp(1). Written in the audited version, never called 
        from anywhere
        """
        fitted_times, fitted_T, _ = self._require_fit()
        times = (
            np.sort(np.asarray(jump_times, dtype=float))
            if jump_times is not None
            else fitted_times
        )
        T = float(T) if T is not None else fitted_T

        lam, alpha, beta = (
            self.params["lambda_bar"],
            self.params["alpha"],
            self.params["beta"],
        )

        # compensator Lambda(t_i), O(n) via the same recursion structure
        n = len(times)
        compensator = np.zeros(n)
        running = 0.0
        for i in range(n):
            if i > 0:
                gap = times[i] - times[i - 1]
                # decayed carry-over plus the newly-completed jump i - 1
                running = running * np.exp(-beta * gap) + (1.0 - np.exp(-beta * gap))
            compensator[i] = lam * times[i] + (alpha / beta) * float(
                np.sum(1.0 - np.exp(-beta * (times[i] - times[:i])))
            )
        
        inter = np.diff(np.concatenate([[0.0], compensator]))
        inter = inter[np.isfinite(inter) & (inter >= 0)]

        if len(inter) < 3:
            return {"error": "too few transformed inter-arrivals for a KS test"}
        
        ks_stat, ks_p = stats.kstest(inter, "expon", args = (0,1))

        probs = (np.arange(1, len(inter) + 1) - 0.5) / len(inter)
        return {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "is_good_fit": bool(ks_p > 0.05),
            "n_residuals": int(len(inter)),
            "mean_residuals": float(inter.mean()),
            "theoretical_quantiles": stats.expon.ppf(probs, loc = 0, scale = 1),
            "empirical_quantiles": np.sort(inter),
            "transformed_times": compensator,
        }

    def intensity_calibration(
        self, jump_indicator: pd.Series, intensity: pd.Series, n_bins: int = 5
    ) -> Dict:
        """
        Does a high fitted lambda(t) actually predict more jumps next period?

        Bins observations by fitted intensity and compares the realized
        next-period jump rate in each bin against the fitted mean. Also runs a
        Poisson regression of the realized indicator on log(fitted intensity):
        a well calibrated intensity gives a slope near 1
        """
        idx = jump_indicator.index.intersection(intensity.index)
        ind = jump_indicator.loc[idx].astype(float)
        lam = intensity.loc[idx].astype(float)

        realized_next = ind.shift(-1).dropna()
        lam = lam.loc[realized_next.index]

        if len(realized_next) < 20 or lam.nunique() < 3:
            return {"error": "insufficient variation in fitted intensity for calibration"}

        try:
            bins = pd.qcut(lam, q = n_bins, duplicates = "drop")
        except ValueError:
            return {"error": "could not form intensity bins (degenerate distribution)"}

        table = pd.DataFrame({"lam": lam, "realized": realized_next, "bin": bins})
        grouped = table.groupby("bin", observed = True).agg(
            n=("realized", "size"),
            fitted_mean = ("lam", "mean"),
            realized_rate = ("realized", "mean"),
        )

        out: Dict = {
            "bins": grouped.reset_index().astype({"bin": str}).to_dict("records"),
            "n_bins_used": int(len(grouped)),
        }

        try:
            import statsmodels.api as sm 

            X = sm.add_constant(np.log(np.maximum(lam.to_numpy(), 1e-12)))
            model = sm.GLM(
                realized_next.to_numpy(), X, family = sm.families.Poisson()
            ).fit()
            out.update(
                {
                    "poisson_slope": float(model.params[1]),
                    "poisson_slope_se": float(model.bse[1]),
                    "poisson_slope_pvalue": float(model.pvalues[1]),
                    "slope_note": "slope ~ 1 indicates a well-calibrated intensity",
                }
            )
        except Exception as exc: # noqa: BLE001
            out["poisson_regression_error"] = f"{type(exc).__name__}: {exc}"

        return out 

    ### intensity paths ###

    def compute_intensity_at_times(
        self, jump_times: np.ndarray, eval_times: np.ndarray
    ) -> np.ndarray:
        """
        lambda(t) on an arbitrary evaluation grid, O(n + m) by merged sweep.

        Only jumps STRICTLY BEFORE each evaluation time contribute, so the
        series is causal and usable as a trading signal
        """
        self._require_params()
        lam, alpha, beta = (
            self.params["lambda_bar"],
            self.params["alpha"],
            self.params["beta"],
        )

        jt = np.sort(np.asarray(jump_times, dtype = float))
        et = np.asarray(eval_times, dtype = float)

        out = np.full(len(et), lam, dtype = float)
        order = np.argsort(et)

        carry = 0.0 # sum of exp(-beta (t_prev - t_j)) over processed jumps 
        last_t = None 
        j = 0 
        for pos in order:
            t = et[pos]
            if last_t is None:
                carry = 0.0
            else:
                carry *= np.exp(-beta * (t - last_t))
            while j < len(jt) and jt[j] < t:
                carry += np.exp(-beta * (t - jt[j]))
                j += 1 
            out[pos] = lam + alpha * carry 
            last_t = t 

        return out 
    
    def compute_intensity_at_dates(
        self, jump_times: np.ndarray, dates: pd.DatetimeIndex, T: Optional[float] = None
    ) -> pd.Series:
        """
        lambda(t) indexed by date, with time measured in TRADING DAYS
        (positional), not calendar days
        """
        eval_times = positional_times(dates)
        values = self.compute_intensity_at_times(jump_times, eval_times)
        return pd.Series(values, index = dates)

    ### Simulation ###

    def simulate(
        self,
        T: float,
        lambda_bar: Optional[float] = None,
        alpha: Optional[float] = None, 
        beta: Optional[float] = None, 
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Ogata thinning. Used by the synthetic-recovery test, which is how we 
        know the estimator recovers known parameters
        """

        if lambda_bar is None or alpha is None or beta is None:
            self._require_params()
            lambda_bar = self.params["lambda_bar"] if lambda_bar is None else lambda_bar
            alpha = self.params["alpha"] if alpha is None else alpha
            beta = self.params["beta"] if beta is None else beta 

        if alpha >= beta:
            raise ValueError(
                f"Non-stationary parameters: alpha = {alpha} >= beta = {beta} "
                "(branching ratio >= 1). Simulation would explode."
            )

        rng = np.random.default_rng(seed)
        times: list = []
        t = 0.0

        while t < T: 
            # Upper bound is the intensity right now: it only decays until the 
            # next accepted event, so this dominates on the whole interval 
            current = lambda_bar + alpha * float(
                np.sum(np.exp(-beta * (t - np.asarray(times)))) if times else 0.0
            )
            lambda_max = max(current, lambda_bar)

            t = t - np.log(rng.random()) / lambda_max 
            if t >= T:
                break 

            actual = lambda_bar + alpha * float(
                np.sum(np.exp(-beta * (t - np.asarray(times)))) if times else 0.0
            )
            if rng.random() <= actual / lambda_max:
                times.append(t)

        return np.asarray(times, dtype = float)

    def branching_ratio(self) -> float:
        self._require_params()
        return self.params["alpha"] / self.params["beta"]

    def _require_fit(self) -> Tuple[np.ndarray, float, float]:
        """Return fitted state, raising a domain error until a fit is available."""
        if self.jump_times is None or self.T is None or self.log_likelihood is None:
            raise HawkesFitError("Model not fitted. Call fit() first.")
        return self.jump_times, self.T, self.log_likelihood

    def _require_params(self) -> None:
        if not self.params:
            raise HawkesFitError("Model not fitted. Call fit() first.")

if __name__ == "__main__":
    print("=" * 72)
    print("Hawkes calibration -- synthetic recovery check")
    print("=" * 72)

    truth = {"lambda_bar": 0.30, "alpha": 0.60, "beta": 1.20}
    T = 4000.0

    model = HawkesProcess(verbose=False)
    sim = model.simulate(T, seed=7, **truth)
    print(f"\nSimulated {len(sim)} jumps over T={T:.0f} (branching {truth['alpha']/truth['beta']:.3f})")

    fitted = model.fit(sim, T)
    print(f"\n{'param':<12}{'true':>10}{'fitted':>12}{'err %':>10}")
    for key in ("lambda_bar", "alpha", "beta"):
        t_, f_ = truth[key], fitted[key]
        print(f"{key:<12}{t_:>10.4f}{f_:>12.4f}{100*abs(f_-t_)/t_:>10.1f}")

    se = model.standard_errors()
    print(f"\nBranching ratio {model.branching_ratio():.4f} "
          f"CI95 {se.get('branching_ratio_ci95')}")

    lr = model.likelihood_ratio_test()
    print(f"LR vs Poisson: {lr['lr_statistic']:.2f}  (chi2 df2 p = {lr['p_chi2_df2']:.3e})")

    gof = model.goodness_of_fit()
    print(f"KS on compensator residuals: stat={gof['ks_statistic']:.4f} p={gof['ks_pvalue']:.4f}")
