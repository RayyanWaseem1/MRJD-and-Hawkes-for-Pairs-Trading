"""
Inference tools for evaluating a trading strategy's returns.

The audited repository reported point estimates with no uncertainty attached
anywhere: no standard error on any Sharpe ratio, no confidence interval on
mean trade P&L, no power calculation, and no adjustment for the number of
parameter configurations that had been tried. With 19 trades and ~1.5%
annualised volatility, the design has essentially no power to distinguish a 1%
annualised edge from zero -- which makes "we found no alpha" the wrong summary.
The right one is "the design cannot detect an effect of the size we are looking
for", and that requires the numbers below.

Contents
--------
    newey_west_mean_test   -- HAC t-test of H0: mean excess return = 0.
                              This replaces the CAPM-vs-SPY regression as the
                              headline test: for a market-neutral book beta is
                              ~0 by construction, so a CAPM alpha mostly
                              measures the accounting of the cash balance.
    sharpe_standard_error  -- Lo (2002) SE, with an IID and an HAC variant.
    stationary_bootstrap   -- Politis-Romano block bootstrap for serially
                              dependent returns.
    deflated_sharpe_ratio  -- Bailey & Lopez de Prado, adjusting for the number
                              of configurations tried.
    minimum_detectable_effect / power_statement -- what the sample can resolve.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy import stats

__all__ = [
    "pool_pair_returns",
    "newey_west_mean_test",
    "sharpe_standard_error",
    "stationary_bootstrap",
    "bootstrap_metric_ci",
    "deflated_sharpe_ratio",
    "minimum_detectable_effect",
    "power_statement",
    "TRADING_DAYS",
]

TRADING_DAYS = 252

# Daily return volatility below this is treated as zero. A book that never
# trades earns exactly rf/252 a day, and floating-point noise of ~1e-19 in
# that constant would otherwise produce an arbitrary "Sharpe ratio".
VOL_EPS = 1e-12


def _nw_variance(x: np.ndarray, maxlags: int) -> float:
    """Newey-West HAC long-run variance of the mean, Bartlett kernel."""
    n = len(x)
    demeaned = x - x.mean()
    gamma0 = float(demeaned @ demeaned / n)
    total = gamma0
    for lag in range(1, maxlags + 1):
        if lag >= n:
            break
        cov = float(demeaned[lag:] @ demeaned[:-lag] / n)
        weight = 1.0 - lag / (maxlags + 1.0)
        total += 2.0 * weight * cov
    return max(total, 1e-24)


def newey_west_mean_test(
    excess_returns: ArrayLike, maxlags: Optional[int] = None
) -> Dict:
    """
    Test H0: E[excess return] = 0 with Newey-West HAC standard errors.

    For a dollar-neutral pairs book this is the correct headline test. The
    CAPM regression against SPY should be reported only to DEMONSTRATE
    neutrality (beta ~ 0), not as the primary evidence -- in the audited
    results it produced an "alpha" of almost exactly -rf for every pair, with
    R-squared ~0.0005, because the strategy sat in uncredited cash ~90% of the
    time.
    """
    x = np.asarray(excess_returns, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 10:
        return {"n_obs": n, "error": "too few observations for a mean test"}

    if maxlags is None:
        maxlags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        maxlags = max(maxlags, 1)

    mean = float(x.mean())
    lrv = _nw_variance(x, maxlags)
    se = float(np.sqrt(lrv / n))

    tstat = mean / se if se > 0 else 0.0
    pvalue = float(2.0 * stats.t.sf(abs(tstat), df=max(n - 1, 1)))

    return {
        "n_obs": n,
        "mean_daily_excess": mean,
        "mean_annualized_excess_pct": mean * TRADING_DAYS * 100.0,
        "se_daily": se,
        "se_annualized_pct": se * TRADING_DAYS * 100.0,
        "t_statistic": float(tstat),
        "p_value": pvalue,
        "maxlags": int(maxlags),
        "ci95_annualized_pct": (
            (mean - 1.96 * se) * TRADING_DAYS * 100.0,
            (mean + 1.96 * se) * TRADING_DAYS * 100.0,
        ),
        "reject_at_5pct": bool(pvalue < 0.05),
    }


def sharpe_standard_error(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    use_hac: bool = True,
) -> Dict:
    """
    Annualised Sharpe with a standard error.

    IID approximation (Lo 2002):  SE(S) ~ sqrt((1 + S^2/2) / n)
    on the PER-PERIOD Sharpe, then annualised by sqrt(periods_per_year).

    The HAC variant rescales by the ratio of the Newey-West long-run variance
    to the IID variance, which matters because daily strategy returns are
    autocorrelated (positions are held for weeks).
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 10:
        return {"n_obs": n, "error": "too few observations for a Sharpe estimate"}

    excess = x - risk_free_rate / periods_per_year
    sd = float(excess.std(ddof=1))
    if sd <= VOL_EPS:
        return {"n_obs": n, "error": "zero return volatility; Sharpe undefined"}

    sharpe_period = float(excess.mean() / sd)
    sharpe_annual = sharpe_period * np.sqrt(periods_per_year)

    se_period = float(np.sqrt((1.0 + 0.5 * sharpe_period**2) / n))

    out = {
        "n_obs": n,
        "sharpe_annualized": sharpe_annual,
        "sharpe_per_period": sharpe_period,
        "se_annualized_iid": se_period * np.sqrt(periods_per_year),
        "t_statistic_iid": sharpe_period / se_period if se_period > 0 else 0.0,
    }

    if use_hac:
        maxlags = max(int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))), 1)
        lrv = _nw_variance(excess, maxlags)
        inflation = float(np.sqrt(max(lrv, 1e-24) / max(excess.var(ddof=1), 1e-24)))
        se_hac = se_period * inflation
        out.update(
            {
                "se_annualized_hac": se_hac * np.sqrt(periods_per_year),
                "hac_inflation_factor": inflation,
                "t_statistic_hac": sharpe_period / se_hac if se_hac > 0 else 0.0,
                "ci95_annualized": (
                    sharpe_annual - 1.96 * se_hac * np.sqrt(periods_per_year),
                    sharpe_annual + 1.96 * se_hac * np.sqrt(periods_per_year),
                ),
            }
        )

    return out


def stationary_bootstrap(
    x: np.ndarray, n_boot: int = 2000, mean_block: float = 20.0, seed: Optional[int] = None
) -> np.ndarray:
    """
    Politis-Romano stationary bootstrap: resample geometric-length blocks so
    serial dependence is preserved. Returns an (n_boot, n) array of resamples.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.empty((0, 0))

    rng = np.random.default_rng(seed)
    p = 1.0 / max(mean_block, 1.0)

    # Vectorised: a block restarts with probability p at each step, otherwise
    # the index advances by one (wrapping). Expressed without a Python loop by
    # locating each position's most recent restart and offsetting from it.
    restarts = rng.random((n_boot, n)) < p
    restarts[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))

    positions = np.arange(n)
    last_restart = np.maximum.accumulate(np.where(restarts, positions, 0), axis=1)
    offsets = positions - last_restart
    block_starts = np.take_along_axis(starts, last_restart, axis=1)

    return x[(block_starts + offsets) % n]


def bootstrap_metric_ci(
    returns: ArrayLike,
    metric: str = "sharpe",
    n_boot: int = 1000,
    mean_block: float = 20.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    seed: Optional[int] = 42,
) -> Dict:
    """Stationary-bootstrap confidence interval for a headline metric."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return {"error": "too few observations to bootstrap", "n_obs": len(x)}

    samples = stationary_bootstrap(x, n_boot=n_boot, mean_block=mean_block, seed=seed)

    def compute(sample: np.ndarray) -> float:
        if metric == "sharpe":
            excess = sample - risk_free_rate / periods_per_year
            sd = sample.std(ddof=1)
            return float(np.sqrt(periods_per_year) * excess.mean() / sd) if sd > VOL_EPS else np.nan
        if metric == "mean":
            return float(sample.mean() * periods_per_year)
        if metric == "total_return":
            return float(np.prod(1.0 + sample) - 1.0)
        if metric == "max_drawdown":
            curve = np.cumprod(1.0 + sample)
            peak = np.maximum.accumulate(curve)
            return float(np.min((curve - peak) / peak))
        raise ValueError(f"Unknown metric '{metric}'")

    values = np.array([compute(s) for s in samples], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"error": f"bootstrap produced no finite values for '{metric}'"}

    point = compute(x)
    return {
        "metric": metric,
        "point_estimate": point,
        "bootstrap_mean": float(values.mean()),
        "bootstrap_se": float(values.std(ddof=1)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "n_boot_effective": int(len(values)),
        "mean_block": mean_block,
    }


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    variance_of_trial_sharpes: Optional[float] = None,
) -> Dict:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Selecting the best of `n_trials` configurations inflates the winner's
    Sharpe even when no configuration has an edge. The DSR is the probability
    that the observed Sharpe exceeds the expected maximum under a null of no
    skill, given how many trials were run.

    All Sharpe inputs and outputs here are ANNUALISED; the internal
    calculation converts to per-observation units.
    """
    if n_trials < 1 or n_obs < 10:
        return {"error": "need n_trials >= 1 and n_obs >= 10"}

    sr = observed_sharpe / np.sqrt(TRADING_DAYS)

    if variance_of_trial_sharpes is None:
        # Benchmark spread across trials under the null, in per-period units.
        var_trials = 1.0 / n_obs
    else:
        var_trials = variance_of_trial_sharpes / TRADING_DAYS

    euler = 0.5772156649015329
    if n_trials > 1:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        expected_max = np.sqrt(var_trials) * ((1 - euler) * z1 + euler * z2)
    else:
        expected_max = 0.0

    denominator = np.sqrt(
        max(1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr**2, 1e-12)
    )
    numerator = (sr - expected_max) * np.sqrt(max(n_obs - 1, 1))
    dsr = float(stats.norm.cdf(numerator / denominator))

    return {
        "observed_sharpe_annualized": float(observed_sharpe),
        "n_trials": int(n_trials),
        "n_obs": int(n_obs),
        "expected_max_sharpe_annualized": float(expected_max * np.sqrt(TRADING_DAYS)),
        "deflated_sharpe_probability": dsr,
        "passes_at_95pct": bool(dsr > 0.95),
        "interpretation": (
            "Probability that the observed Sharpe reflects skill rather than "
            f"the best of {n_trials} trials under a no-skill null."
        ),
    }


def minimum_detectable_effect(
    n_obs: int,
    annual_volatility: float,
    power: float = 0.80,
    alpha: float = 0.05,
    periods_per_year: int = TRADING_DAYS,
) -> Dict:
    """
    Smallest annualised mean return this sample could detect at the given power.

        MDE = (z_{1-alpha/2} + z_{power}) * sigma_annual / sqrt(n_years)
    """
    if n_obs < 10 or annual_volatility <= 0:
        return {"error": "need n_obs >= 10 and positive volatility"}

    n_years = n_obs / periods_per_year
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_power = stats.norm.ppf(power)
    mde = (z_alpha + z_power) * annual_volatility / np.sqrt(max(n_years, 1e-9))

    return {
        "n_obs": int(n_obs),
        "n_years": float(n_years),
        "annual_volatility_pct": float(annual_volatility * 100),
        "power": power,
        "alpha": alpha,
        "mde_annualized_pct": float(mde * 100),
    }


def power_statement(
    n_obs: int,
    annual_volatility: float,
    target_effect_annual: float = 0.01,
    alpha: float = 0.05,
    periods_per_year: int = TRADING_DAYS,
) -> Dict[str, object]:
    """
    Power to detect an effect of `target_effect_annual`, plus a plain-English
    statement suitable for pasting into a README.
    """
    if n_obs < 10:
        return {"error": "need n_obs >= 10"}
    if annual_volatility <= 1e-9:
        # A flat return series means the strategy never took risk. Reporting
        # "100% power" here would be an artifact of dividing by ~zero.
        return {
            "error": "return volatility is ~zero (the strategy did not trade); "
                     "power is undefined rather than perfect"
        }

    n_years = n_obs / periods_per_year
    se_annual = annual_volatility / np.sqrt(max(n_years, 1e-9))
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    ncp = target_effect_annual / se_annual
    achieved = float(stats.norm.sf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp))

    mde = minimum_detectable_effect(n_obs, annual_volatility, 0.80, alpha, periods_per_year)

    return {
        "n_obs": int(n_obs),
        "n_years": float(n_years),
        "annual_volatility_pct": float(annual_volatility * 100),
        "target_effect_annual_pct": float(target_effect_annual * 100),
        "se_of_mean_annual_pct": float(se_annual * 100),
        "achieved_power": achieved,
        "mde_at_80pct_power_annual_pct": mde.get("mde_annualized_pct"),
        "statement": (
            f"Over {n_years:.1f} years at {annual_volatility * 100:.2f}% annualised "
            f"volatility, the standard error of the mean return is "
            f"{se_annual * 100:.2f}%/yr. Power to detect a "
            f"{target_effect_annual * 100:.1f}%/yr edge at alpha={alpha} is "
            f"{achieved:.1%}; the smallest effect detectable at 80% power is "
            f"{mde.get('mde_annualized_pct', float('nan')):.2f}%/yr. A null result "
            f"here means the design could not resolve an edge of the size sought, "
            f"not that no edge exists."
        ),
    }


def pool_pair_returns(
    returns_by_pair: Dict[str, "pd.Series"],
    weights: Optional[Dict[str, float]] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = TRADING_DAYS,
) -> Dict:
    """
    Pool several pairs into one equal-weight portfolio and test it as a whole.

    This is the statistical payoff of breadth, and the reason "more pairs" is
    the only route to power from daily data. A single pair with a 50-70 day
    half-life offers ~2 independent round trips a year; N imperfectly
    correlated pairs offer ~2N, and the standard error of the mean falls with
    the square root of that.

    The gain is real only to the extent the pairs are independent, so the
    average pairwise correlation is reported alongside the pooled test: with
    correlation rho, the effective breadth is roughly N / (1 + (N-1) rho), and
    the function reports that too rather than letting N stand unqualified.

    Params
    ------
    returns_by_pair : {pair_key: daily return Series}
    weights : optional {pair_key: weight}; defaults to equal weight.
    """
    import pandas as pd

    frame = pd.DataFrame(returns_by_pair).dropna(how="all")
    if frame.empty or frame.shape[1] == 0:
        return {"error": "no return series supplied"}

    frame = frame.fillna(0.0)
    n_pairs = frame.shape[1]

    if weights:
        w = np.array([weights.get(c, 0.0) for c in frame.columns], dtype=float)
        if w.sum() <= 0:
            return {"error": "weights sum to zero"}
        w = w / w.sum()
    else:
        w = np.full(n_pairs, 1.0 / n_pairs)

    portfolio = pd.Series(frame.to_numpy() @ w, index=frame.index)

    # Effective breadth: N independent bets only if the pairs are uncorrelated.
    if n_pairs > 1:
        corr = frame.corr().to_numpy()
        off_diagonal = corr[~np.eye(n_pairs, dtype=bool)]
        mean_corr = float(np.nanmean(off_diagonal))
        effective_n = n_pairs / (1.0 + (n_pairs - 1) * max(mean_corr, 0.0))
    else:
        mean_corr = 0.0
        effective_n = 1.0

    daily_rf = risk_free_rate / periods_per_year
    excess = portfolio.to_numpy(dtype=float) - daily_rf

    test = newey_west_mean_test(excess)
    sharpe = sharpe_standard_error(
        portfolio.to_numpy(dtype=float), risk_free_rate, periods_per_year
    )

    out = {
        "n_pairs": n_pairs,
        "mean_pairwise_correlation": mean_corr,
        "effective_independent_pairs": effective_n,
        "breadth_gain_vs_single": float(np.sqrt(effective_n)),
        "n_obs": len(portfolio),
    }
    for key in (
        "mean_annualized_excess_pct", "se_annualized_pct", "t_statistic", "p_value"
    ):
        if key in test:
            out[f"pooled_{key}"] = test[key]
    if "ci95_annualized_pct" in test:
        out["pooled_ci95_low_pct"] = test["ci95_annualized_pct"][0]
        out["pooled_ci95_high_pct"] = test["ci95_annualized_pct"][1]
    for key in ("sharpe_annualized", "se_annualized_hac", "t_statistic_hac"):
        if key in sharpe:
            out[f"pooled_{key}"] = sharpe[key]

    out["portfolio_returns"] = portfolio
    return out
