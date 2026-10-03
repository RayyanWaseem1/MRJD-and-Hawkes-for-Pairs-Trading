"""
Synthetic-recovery tests for the estimators.

The audited repository had NO tests at all. For a research codebase whose
central claim is a statistical one, the minimum bar is: can each estimator
recover parameters it is given? These tests exist so a future change that
breaks the Hawkes likelihood or the OU parameterisation fails loudly.
"""

import numpy as np
import pandas as pd
import pytest

from hawkes_calibration import HawkesFitError, HawkesProcess
from jump_detector import JumpDetector, benjamini_hochberg
from mrjd_estimation import MRJDFitError, MRJDModel
from time_units import DT


# --------------------------------------------------------------------- #
# Hawkes
# --------------------------------------------------------------------- #

def test_hawkes_recovers_known_parameters():
    """Simulate a Hawkes process with known (lambda_bar, alpha, beta), refit it."""
    truth = {"lambda_bar": 0.30, "alpha": 0.60, "beta": 1.20}
    T = 4000.0

    model = HawkesProcess(verbose=False)
    times = model.simulate(T, seed=7, **truth)
    assert len(times) > 500, "simulation produced too few events to identify parameters"

    fitted = model.fit(times, T)

    for key in ("lambda_bar", "alpha", "beta"):
        rel_error = abs(fitted[key] - truth[key]) / truth[key]
        assert rel_error < 0.25, f"{key}: recovered {fitted[key]:.4f} vs true {truth[key]}"

    true_branching = truth["alpha"] / truth["beta"]
    assert abs(model.branching_ratio() - true_branching) < 0.10


def test_hawkes_branching_ratio_ci_covers_truth():
    """The Hessian-based CI on the branching ratio should cover the true value."""
    truth = {"lambda_bar": 0.30, "alpha": 0.60, "beta": 1.20}
    T = 4000.0

    model = HawkesProcess(verbose=False)
    times = model.simulate(T, seed=11, **truth)
    model.fit(times, T)

    se = model.standard_errors()
    low, high = se["branching_ratio_ci95"]
    assert low < truth["alpha"] / truth["beta"] < high


def test_hawkes_optimum_is_interior_not_at_a_cliff():
    """
    A strongly self-exciting process must be allowed to fit above 0.85.

    The audited objective returned a literal 1e10 for any branching ratio
    above 0.85, so every fit landed within 2% of that wall. The
    reparameterisation removes the wall entirely.
    """
    truth = {"lambda_bar": 0.20, "alpha": 0.88, "beta": 1.00}  # branching 0.88
    T = 5000.0

    model = HawkesProcess(verbose=False)
    times = model.simulate(T, seed=3, **truth)
    model.fit(times, T)

    assert model.branching_ratio() > 0.80, (
        f"branching ratio {model.branching_ratio():.4f} -- a true value of 0.88 "
        "should be reachable, not clamped below 0.85"
    )
    assert not model.fit_diagnostics["bounds_clipped"]


def test_hawkes_lr_test_rejects_poisson_on_clustered_data():
    """Genuinely self-exciting data should reject the Poisson null."""
    model = HawkesProcess(verbose=False)
    times = model.simulate(4000.0, lambda_bar=0.3, alpha=0.6, beta=1.2, seed=5)
    model.fit(times, 4000.0)

    lr = model.likelihood_ratio_test()
    assert lr["lr_statistic"] > 0
    assert lr["p_chi2_df2"] < 0.01


def test_hawkes_lr_test_does_not_reject_on_poisson_data():
    """Homogeneous Poisson data must NOT produce a self-excitation finding."""
    rng = np.random.default_rng(17)
    T = 4000.0
    times = np.sort(rng.uniform(0, T, size=800))

    model = HawkesProcess(verbose=False)
    model.fit(times, T)

    lr = model.likelihood_ratio_test()
    assert lr["p_chi2_df2"] > 0.01, (
        "Poisson data produced a significant self-excitation result -- "
        "this is the failure mode the whole audit was about"
    )
    assert model.branching_ratio() < 0.5


def test_hawkes_on_data_matches_on_recursion_and_naive_likelihood():
    """The O(n) recursion must agree with a direct O(n^2) evaluation."""
    model = HawkesProcess(verbose=False)
    times = model.simulate(600.0, lambda_bar=0.4, alpha=0.5, beta=1.5, seed=2)
    lam, alpha, beta, T = 0.4, 0.5, 1.5, 600.0

    fast = HawkesProcess.log_likelihood_at(times, T, lam, alpha, beta)

    naive_sum = 0.0
    for i, t in enumerate(times):
        intensity = lam + alpha * np.sum(np.exp(-beta * (t - times[:i])))
        naive_sum += np.log(intensity)
    compensator = lam * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - times)))
    naive = naive_sum - compensator

    assert abs(fast - naive) < 1e-6


def test_hawkes_refuses_too_few_jumps():
    model = HawkesProcess(verbose=False)
    with pytest.raises(HawkesFitError, match="at least"):
        model.fit(np.array([1.0, 5.0]), 100.0)


def test_hawkes_simulate_rejects_explosive_parameters():
    model = HawkesProcess(verbose=False)
    with pytest.raises(ValueError, match="Non-stationary"):
        model.simulate(100.0, lambda_bar=0.1, alpha=2.0, beta=1.0)


# --------------------------------------------------------------------- #
# MRJD / OU
# --------------------------------------------------------------------- #

def _simulate_ou(kappa, theta, sigma, n, seed=11):
    rng = np.random.default_rng(seed)
    phi = np.exp(-kappa * DT)
    cond_sd = sigma * np.sqrt((1 - np.exp(-2 * kappa * DT)) / (2 * kappa))
    x = np.zeros(n)
    x[0] = theta
    for i in range(1, n):
        x[i] = theta + (x[i - 1] - theta) * phi + rng.normal(0, cond_sd)
    return pd.Series(x, index=pd.RangeIndex(n))


def test_mrjd_recovers_known_ou_parameters():
    true_kappa, true_theta, true_sigma = 0.04, 1.5, 0.06
    spread = _simulate_ou(true_kappa, true_theta, true_sigma, 3000)

    model = MRJDModel(verbose=False)
    fitted = model.fit(spread, pd.Series(0, index=spread.index), dt=DT)

    assert abs(fitted["kappa"] - true_kappa) / true_kappa < 0.20
    assert abs(fitted["theta"] - true_theta) < 0.15
    assert abs(fitted["sigma"] - true_sigma) / true_sigma < 0.15


def test_mrjd_implied_dispersion_matches_the_data():
    """
    sigma / sqrt(2 kappa) must match the realised spread standard deviation.

    The audited GS/MS fit implied 23.3 against an actual 1.49 -- a 15.6x
    mismatch produced by optimising in a flat direction of the likelihood.
    """
    spread = _simulate_ou(0.04, 1.5, 0.06, 3000)

    model = MRJDModel(verbose=False)
    model.fit(spread, pd.Series(0, index=spread.index), dt=DT)

    ratio = model.diagnostics["sd_ratio"]
    assert 0.7 < ratio < 1.4, f"implied/actual dispersion ratio {ratio:.2f}"


def test_mrjd_half_life_is_in_trading_days():
    """
    With dt = 1.0 the half-life is in TRADING DAYS.

    The audited config used dt = 1/252, making kappa per-YEAR, so
    log(2)/kappa was a half-life in years compared against days -- off by 252x.
    """
    true_kappa = 0.04
    expected_half_life = np.log(2) / true_kappa  # ~17.3 trading days
    spread = _simulate_ou(true_kappa, 1.5, 0.06, 3000)

    model = MRJDModel(verbose=False)
    fitted = model.fit(spread, pd.Series(0, index=spread.index), dt=DT)

    half_life = np.log(2) / fitted["kappa"]
    assert 0.7 * expected_half_life < half_life < 1.4 * expected_half_life


def test_mrjd_rejects_year_units():
    """dt = 1/252 must raise, not silently produce per-year parameters."""
    spread = _simulate_ou(0.04, 1.5, 0.06, 500)
    model = MRJDModel(verbose=False)
    with pytest.raises(ValueError, match="factor-of-252"):
        model.fit(spread, pd.Series(0, index=spread.index), dt=1 / 252)


def test_mrjd_on_a_random_walk_reports_an_untradeable_half_life():
    """
    A random walk has no real mean reversion. Finite-sample bias pulls the
    AR(1) coefficient just below 1, so the fit succeeds -- but the honest
    signal is a half-life far outside anything tradeable, which is what
    `validate_pair`'s half-life gate then rejects.
    """
    rng = np.random.default_rng(3)
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 2000)))

    model = MRJDModel(verbose=False)
    fitted = model.fit(walk, pd.Series(0, index=walk.index), dt=DT)

    assert fitted["phi"] > 0.99, "a random walk should look near-unit-root"
    half_life = np.log(2) / fitted["kappa"]
    assert half_life > 200, (
        f"half-life {half_life:.0f}d -- a random walk must not masquerade as a "
        "fast-reverting spread"
    )


def test_mrjd_raises_on_an_explosive_series():
    """phi >= 1 has no finite half-life and must fail rather than guess."""
    rng = np.random.default_rng(4)
    n = 500
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 1.02 * x[i - 1] + rng.normal(0, 0.01) + 0.001

    model = MRJDModel(verbose=False)
    with pytest.raises(MRJDFitError, match="unit root|mean-reverting"):
        model.fit(pd.Series(x), pd.Series(0, index=pd.RangeIndex(n)), dt=DT)


# --------------------------------------------------------------------- #
# Jump detection
# --------------------------------------------------------------------- #

def test_lee_mykland_dates_the_jump_correctly():
    """
    A jump injected on a known day must be flagged ON THAT DAY.

    This is finding 0.5: the audited detector computed a statistic over a
    20-day trailing window and marked the window's LAST day, so jump times fed
    to the Hawkes MLE were misdated by 0-19 days and one jump became a run of
    up to 20 "events".
    """
    rng = np.random.default_rng(4)
    n = 400
    series = pd.Series(rng.normal(0, 0.01, n))

    jump_position = 250
    series.iloc[jump_position] += 0.25  # ~25 sigma

    detector = JumpDetector(0.05, apply_fdr=False, verbose=False)
    result = detector.detect_jumps_lee_mykland(series, window=20)

    flagged = np.where(result["jump_indicator"].to_numpy() == 1)[0]
    assert jump_position in flagged, f"jump at {jump_position} not detected; got {flagged}"
    assert len(flagged) <= 3, f"one jump produced {len(flagged)} events: {flagged}"


def test_lee_mykland_does_not_flag_clean_gaussian_noise():
    """FDR control should keep the false-positive rate near zero on pure noise."""
    rng = np.random.default_rng(9)
    series = pd.Series(rng.normal(0, 0.01, 1500))

    detector = JumpDetector(0.05, apply_fdr=True, verbose=False)
    result = detector.detect_jumps_lee_mykland(series, window=20)

    n_flagged = int(result["jump_indicator"].sum())
    assert n_flagged <= 5, f"{n_flagged} false positives on clean noise"


def test_jump_times_are_trading_day_positions_not_calendar_days():
    """
    Jump times must be positional, so weekends do not open artificial gaps.

    The audited `extract_jump_times` used `(t - t0).days`, so the Hawkes kernel
    saw a 3-day gap every weekend on a series that only exists on trading days.
    """
    index = pd.bdate_range("2020-01-01", periods=30, tz="UTC")
    frame = pd.DataFrame({"jump_indicator": 0}, index=index)
    frame.iloc[[0, 5, 20], 0] = 1

    detector = JumpDetector(verbose=False)
    times = detector.extract_jump_times(frame)

    assert list(times) == [0.0, 5.0, 20.0]


def test_benjamini_hochberg_controls_discoveries():
    p_null = np.random.default_rng(1).uniform(0, 1, 1000)
    assert benjamini_hochberg(p_null, 0.05).sum() <= 5

    p_mixed = np.concatenate([np.full(20, 1e-8), np.random.default_rng(2).uniform(0, 1, 980)])
    rejected = benjamini_hochberg(p_mixed, 0.05)
    assert rejected[:20].all()


def test_bipower_attributes_to_the_largest_move_in_the_window():
    """BNS flags a window; the indicator must land on the biggest move in it."""
    rng = np.random.default_rng(6)
    n = 300
    series = pd.Series(rng.normal(0, 0.01, n))
    jump_position = 150
    series.iloc[jump_position] += 0.30

    detector = JumpDetector(0.05, apply_fdr=False, verbose=False)
    result = detector.detect_jumps_bipower_variation(series, window=20)

    flagged = np.where(result["jump_indicator"].to_numpy() == 1)[0]
    assert len(flagged) > 0
    assert min(abs(flagged - jump_position)) <= 1
