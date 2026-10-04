"""
Jump Detection for pair spreads.

Detectors:
* lee_mykland -- PRIMARY. A per-observation test: each return is 
                 standardized by *local* bipower volatility estimated from 
                 a strictly preceding window, so a rejection at time i means 
                 "there was a jump at i".
* bipower -- Barndorff-Nielsen-Shepard. ROBUSTNESS ONLY (see caveat)
* threshold -- Naive k-sigma rule, kept for the detector comparison 

Why the primary detector changed:
The audited pipeline used BNS with a 20-day trailing rolling window and then wrote

    jump_indicator = (z_stat > critical_value).astype(int) 

A significant statistic at time t means "there was a jump somewhere in the 
trailing 20 days", not "there was a jump today." So (a) the jump times fed to 
the Hawkes MLE were misdated by 0-19 days, and (b) one genuine jump kept the 
stastic elevated for up to 20 consecutive windows, manufacturing a run of up 
to 20 "events" from a single jump. A Hawkes process fit to runs of consecutive
integers reports strong self-excitation with a short decay constant -- which is 
exactly what the project reported. Lee-Mykland was previously implemented but never called. 

BNS caveat (kept, however demoted):
Bipower variation is a HIGH-FREQUENCY estimator; its asymptotics require
Delta -> 0, i.e. many intraday returns *within* the period being tested. This
module's own docstring used to say `window: int = 78 #78 for 5-min data in a 
day` while the config fed it 20 DAILY bars. There is no asymptotic regime in 
which the test is valid on daily data. It is retained only as a robustness
comparison, with its three implementation errors fixed:

    * the Barndorff-Nielsen-Shephard variance constant 
      theta = pi^2/4 + pi - 5 ~= 0.6090 was absent entirely, so that statistic
      was not asymptotically N(0,1) and `norm.ppf(0.95)` was not a 5% critical 
      value of anything;
    * tripower quarticity was scaled by `n = len(returns)` (the whole ~1,950
      observation series) instead of the window length, inflating TP ~100x;
    * mu_{4/3} used Gamma(5/6) where the definition
      mu_p = 2^{p/2} Gamma((p+1)/2) / Gamma(1/2) gives Gamma(7/6)

The level form RV - BV is replaced by the log-ratio form, which has far better
finite-sample behavior

Multiple testing:
At nominal alpha = 0.05 across ~1,950 tests roughly 98 false positives are
expected by chance. All detectors now return per-observation p-values and 
apply Benjamini-Hochberg FDR control by default.

Causality:
Two parts of the test depend on the whole sample: the Gumbel normalisers
C_n, S_n (through n) and the BH rejection cutoff (through every p-value).
Run over a sample that extends past the training window, a flag at day t
would then depend on later data. Callers that evaluate out of sample pass
`n_ref` (the training length) and `p_cutoff` (the training BH cutoff, see
`last_p_cutoff`), which reproduces the training flags exactly and makes every
later flag a function of the past only.
"""

from __future__ import annotations

from typing import Dict, List, Optional 

import numpy as np 
import pandas as pd 
from scipy import stats 
from scipy.special import gamma as gamma_func

from time_units import TIME_UNIT, observation_span, positional_times 

_all__ = ["JumpDetector", "benjamini_hochberg", "DETECTORS"]

DETECTORS = ("lee_mykland", "bipower", "threshold")

# E|Z| for a standard normal. Enters the Lee-Mykland critical values 
MU_1 = np.sqrt(2.0 / np.pi)

# mu_p = 2^{p/2} Gamma((p+1)/2) / Gamma(1/2), at p = 4/3
MU_4_3 = 2.0 ** (2.0 / 3.0) * gamma_func(7.0 / 6.0) / gamma_func(0.5)

# Barndorff-Nielsen-Shephard variance constant for the bipower test 
BNS_THETA = np.pi**2 / 4.0 + np.pi - 5.0 #~= 0.6090

def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR control.

    Returns a boolean array of rejections at false-discovery-rate `alpha`.
    NaN p-values are treated as non-rejections
    """

    p = np.asarray(pvalues, dtype = float)
    out = np.zeros(len(p), dtype = bool)

    finite = np.isfinite(p)
    if not finite.any():
        return out 

    idx = np.where(finite)[0]
    ordered = idx[np.argsort(p[idx])]
    m = len(ordered)

    thresholds = alpha * np.arange(1, m + 1) / m 
    passing = p[ordered] <= thresholds

    if not passing.any():
        return out 

    k = np.max(np.where(passing)[0])
    out[ordered[: k + 1]] = True 
    return out 

class JumpDetector:
    """ Detect jumps in a spread series"""

    def __init__(
        self,
        significance_level: float = 0.05,
        apply_fdr: bool = True,
        verbose: bool = True,
    ):
        """
        Params:
        significance_level : float 
            Nominal alpha, and the FDR level when `apply_fdr` is True
        apply_fdr : bool
            Apply Benjamini-Hochberg control across all tested observations
        """
        self.significance_level = significance_level
        self.apply_fdr = apply_fdr
        self.verbose = verbose 

        self.jumps: Optional[pd.DataFrame] = None 
        self.jump_stats: Dict = {}

        # BH p-value cutoff from the most recent FDR run, so it can be frozen
        self.last_p_cutoff: Optional[float] = None

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    ### dispatch ###

    def detect(
        self,
        spread_diff: pd.Series,
        method: str = "lee_mykland",
        window: int = 20,
        n_ref: Optional[int] = None,
        p_cutoff: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Run one detector by name.

        Params:
        spread_diff: pd.Series
            FIRST DIFFERENCE of the log spread. Not 'pct_change' -- the spread 
            is already in logs and crosses zero, so a percentage change divides
            by numbers arbitrarily close to zero (observed max |pct_change| of 
            758.9 on AMD/NVDA)
        n_ref: int, optional
            Sample size for the Lee-Mykland Gumbel normalisers. Defaults to
            len(spread_diff); pass the training length to freeze them.
        p_cutoff: float, optional
            Frozen FDR cutoff: flag p <= p_cutoff instead of re-running BH.
            Only used when `apply_fdr` is True.
        """
        if method == "lee_mykland":
            return self.detect_jumps_lee_mykland(
                spread_diff, window = window, n_ref = n_ref, p_cutoff = p_cutoff
            )
        if method in ("bipower", "bipower_variation"):
            return self.detect_jumps_bipower_variation(
                spread_diff, window = window, p_cutoff = p_cutoff
            )
        if method == "threshold":
            return self.detect_jumps_threshold(spread_diff, window = window)
        raise ValueError(f"Unknown detector '{method}'. Choose from {DETECTORS}.")

    ### Primary detector ###

    def detect_jumps_lee_mykland(
        self,
        spread_diff: pd.Series,
        window: int = 20,
        n_ref: Optional[int] = None,
        p_cutoff: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Lee - Mykland (2008) per-observation jump test. 
        
        The statistic for observation i is
        
                L_i = r_i / sigma_hat_i,
                sigma_hat_i^2 = (1/(K-2)) * sum_{j=i-K+2}^{i-1} |r_j| |r_{j-1}|
        
        where the local bipower window is STRICTLY PRECEDING i, so the tested
        return never contributes to its own volatility estimate.
        
        Under the null of no jump, max|L| follows a Gumbel law:
        
            (max|L| - C_n) / S_n  ->  Gumbel
            C_n = sqrt(2 log n)/c - (log pi + log log n) / (2 c sqrt(2 log n))
            S_n = 1 / (c sqrt(2 log n)),        c = E|Z| = sqrt(2/pi)
        
        which gives a per-observation p-value
        p_i = 1 - exp(-exp(-(L_i - C_n)/S_n)) that can be FDR-controlled.
        
        Params:
        spread_diff : pd.Series
            First difference of the log spread (see `detect`).
        window : int
            K, the local bipower volatility window in trading days.
        """

        self._log(f"Detecting jumps: Lee-Mykland (window = {window} {TIME_UNIT}s)")

        r = spread_diff.astype(float)
        n = len(r)
        if n < window + 3:
            raise ValueError(
                f"Lee-Mykland needs more than {window + 3} observations, got {n}"
            )

        abs_r = r.abs()
        # |r_j| * |r_{j-1}|, aligned so entry j uses j and j - 1
        bipower_products = abs_r * abs_r.shift(1)

        # Strictly preceding window: shift(1) before rolling 
        local_bp = (
            bipower_products.shift(1)
            .rolling(window = window, min_periods = max(window // 2, 3))
            .sum()
        )
        denom = max(window - 2, 1)
        sigma_hat = np.sqrt(local_bp / denom)

        with np.errstate(divide = "ignore", invalid = "ignore"):
            L = r.abs() / sigma_hat 
        L = L.replace([np.inf, -np.inf], np.nan)

        c = MU_1
        log_n = np.log(max(n_ref if n_ref is not None else n, 3))
        root = np.sqrt(2.0 * log_n)
        C_n = root / c - (np.log(np.pi) + np.log(log_n)) / (2.0 * c * root)
        S_n = 1.0 / (c * root)

        standardized = (L - C_n) / S_n
        # Gumbel survival: P(xi > x) = 1 - exp(-exp(-x))
        pvalues = 1.0 - np.exp(-np.exp(-standardized))
        pvalues = pd.Series(pvalues, index = r.index).clip(0.0, 1.0)

        beta_star = -np.log(-np.log(1.0 - self.significance_level))
        threshold = C_n + beta_star * S_n

        if self.apply_fdr:
            jump_indicator, rule = self._fdr_indicator(pvalues, p_cutoff)
        else:
            jump_indicator = (L > threshold).fillna(False).astype(int)
            rule = f"Gumbel critical value {threshold:.3f}"

        result = pd.DataFrame(
            {
                "returns": r,
                "local_volatility": sigma_hat,
                "L_statistic": L,
                "p_value": pvalues,
                "threshold": threshold,
                "jump_indicator": jump_indicator,
                "jump_size": r * jump_indicator,
            },
            index = r.index,
        )

        n_jumps = int(jump_indicator.sum())
        self._log(
            f" {n_jumps} jumps ({100 * n_jumps / n:.2f}% of {n} obs), rule: {rule}"
        )

        self.jumps = result 
        return result 

    ### robustness detector ###

    def detect_jumps_bipower_variation(
        self,
        spread_diff: pd.Series,
        window: int = 20,
        p_cutoff: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Barndorff-Nielsen-Shephard biipower test, log-ratio form.

        ROBUSTNESS ONLY -- the asymptotics reuqire Delta -> 0 and this is daily 
        data. See the module docstring. 

        The statistic is

            Z = [log RV - log BV] / 
                sqrt( (theta / m) * max(1, TP / BV^2) ),
            theta = pi^2/4 + pi - 5

        with RV, BV, and TP all computed over m-observation window, and TP
        scaled by m (not by the length of the whole series).

        NOTE ON ATTRIBUTION: a rejection at time t means "a jump occurred
        somewhere in the trailing `window` observations". The indicator is 
        therefor attributed to the LARGEST |return| inside the window rather
        than to the window's last day, which is what the audited code did
        """
        self._log(
            f"Detecting jumps: bipower / BNS (window = {window})"
            "--ROBUSTNESS ONLY, invalid asymptotics on daily data"
        )

        r = spread_diff.astype(float)
        n = len(r)
        m = float(window)

        abs_r = r.abs()

        RV = r.rolling(window = window).apply(lambda x: np.sum(x**2), raw = True)

        BV = (1.0 / MU_1**2) * (abs_r * abs_r.shift(1)).rolling(window = window).sum()

        power = abs_r ** (4.0 / 3.0)
        tri = power * power.shift(1) * power.shift(2)
        # Scale by the WINDOW length, not len(returns)
        TP = m * (MU_4_3 ** (-3.0)) * tri.rolling(window = window).sum()

        with np.errstate(divide = "ignore", invalid = "ignore"):
            ratio_guard = np.maximum(1.0, TP / (BV**2))
            variance = (BNS_THETA / m) * ratio_guard 
            z_stat = (np.log(RV) - np.log(BV)) / np.sqrt(variance)

        z_stat = z_stat.replace([np.inf, -np.inf], np.nan)
        pvalues = pd.Series(stats.norm.sf(z_stat), index = r.index)
        pvalues[z_stat.isna()] = np.nan 

        if self.apply_fdr:
            window_flag, rule = self._fdr_indicator(pvalues, p_cutoff)
        else:
            crit = stats.norm.ppf(1 - self.significance_level)
            window_flag = (z_stat > crit).fillna(False).astype(int)
            rule = f"normal critical value {crit:.3f}"

        ### correct attribution: blame the biggest move in the window ###
        jump_indicator = pd.Series(0, index = r.index, dtype = int)
        flagged_positions = np.where(window_flag.to_numpy() == 1)[0]
        for pos in flagged_positions:
            lo = max(0, pos - window + 1)
            local = abs_r.iloc[lo : pos + 1]
            if local.notna().any():
                jump_indicator.iloc[lo + int(np.nanargmax(local.to_numpy()))] = 1

        result = pd.DataFrame(
            {
                "returns": r,
                "RV": RV,
                "BV": BV,
                "TP": TP,
                "z_statistic": z_stat,
                "p_value": pvalues,
                "window_flag": window_flag,
                "jump_indicator": jump_indicator,
                "jump_size": r * jump_indicator,
                "jump_variation": np.maximum(0.0, RV - BV),
            },
            index = r.index,
        )

        n_flag = int(window_flag.sum())
        n_jumps = int(jump_indicator.sum())
        self._log(
            f" {n_flag} windows flagged -> {n_jumps} distinct jumps "
            f"({100 * n_jumps / n:.2f}% of obs), rule: {rule}"
        )

        self.jumps = result 
        return result 

    def _fdr_indicator(
        self, pvalues: pd.Series, p_cutoff: Optional[float]
    ) -> tuple:
        """
        FDR jump flags, either by running BH on these p-values or, when
        `p_cutoff` is given, by applying a cutoff frozen on the training window.

        The cutoff from a BH run is the largest rejected p-value (BH rejects
        every p-value at or below it), or the most stringent BH level alpha/m
        when nothing is rejected.
        """
        p = pvalues.to_numpy(dtype = float)
        if p_cutoff is not None:
            flags = np.isfinite(p) & (p <= p_cutoff)
            self.last_p_cutoff = float(p_cutoff)
            rule = f"frozen BH cutoff p <= {p_cutoff:.3g}"
        else:
            flags = benjamini_hochberg(p, self.significance_level)
            m = max(int(np.isfinite(p).sum()), 1)
            self.last_p_cutoff = (
                float(np.max(p[flags])) if flags.any() else self.significance_level / m
            )
            rule = f"BH-FDR at {self.significance_level:.0%}"
        return pd.Series(flags.astype(int), index = pvalues.index), rule

    ### naive detector ###

    def detect_jumps_threshold(
        self, spread_diff: pd.Series, threshold_sigma: float = 4.0, window: int = 60
    ) -> pd.DataFrame:
        """ Naive k-sigma rule against a strictly preceding rolling window"""
        self._log(f"Detecting jumps: {threshold_sigma}-sigma threshold (window = {window})")

        r = spread_diff.astype(float)

        rolling_mean = r.shift(1).rolling(window = window, min_periods = window // 2).mean()
        rolling_std = r.shift(1).rolling(window = window, min_periods = window //2).std()

        with np.errstate(divide = "ignore", invalid = "ignore"):
            z = (r - rolling_mean) / rolling_std 
        z = z.replace([np.inf, -np.inf], np.nan)

        jump_indicator = (z.abs() > threshold_sigma).fillna(False).astype(int)
        pvalues = pd.Series(2.0 * stats.norm.sf(z.abs()), index = r.index)

        result = pd.DataFrame(
            {
                "returns": r, 
                "rolling_mean": rolling_mean,
                "rolling_std": rolling_std,
                "z_score": z, 
                "p_value": pvalues,
                "jump_indicator": jump_indicator,
                "jump_size": r * jump_indicator,
            },
            index = r.index,
        )

        n_jumps = int(jump_indicator.sum())
        self._log(f" {n_jumps} jumps ({100 * n_jumps / len(r):.2f}% of obs)")
        return result 

    ### extraction ###

    def extract_jump_times(self, jump_df: pd.DataFrame) -> np.ndarray:
        """
        Jump times as TRADING-DAY positions within `jump_df`

        Previously this returned `(t - t0).days`, calendar days, so the Hawkes
        kernel saw an artificial 3-day gap every weekend on a series that only
        exists on trading days -- biasing beta-hat
        """
        mask = (jump_df["jump_indicator"] == 1).to_numpy()
        return positional_times(jump_df.index, mask)

    @staticmethod
    def observation_span(jump_df: pd.DataFrame) -> float:
        """ Total observation period T in trading days"""
        return observation_span(jump_df.index)

    def extract_jump_sizes(self, jump_df: pd.DataFrame) -> np.ndarray:
        if "jump_size" in jump_df.columns:
            sizes = jump_df.loc[jump_df["jump_indicator"] == 1, "jump_size"]
        elif "jump_variation" in jump_df.columns:
            sizes = jump_df.loc[jump_df["jump_indicator"] == 1, "jump_variation"]
        else:
            raise ValueError("No jump size information in this dataframe")
        return np.asarray(sizes, dtype = float) 

    ### statistics and comparison ###

    def calculate_jump_statistics(self, jump_df: pd.DataFrame) -> Dict:
        """ Summary statistics for a detector's output"""
        indicator = jump_df["jump_indicator"]
        n_jumps = int(indicator.sum())
        n_obs = len(jump_df)

        times = self.extract_jump_sizes(jump_df)
        if len(times) > 1:
            gaps = np.diff(times)
            mean_gap, std_gap = float(gaps.mean()), float(gaps.std())
        else:
            mean_gap = std_gap = float("nan")

        sizes = self.extract_jump_sizes(jump_df)
        if len(sizes) > 0:
            mean_size, std_size = float(np.mean(sizes)), float(np.std(sizes))
            max_size = float(np.max(np.abs(sizes)))
        else:
            mean_size = std_size = max_size = float("nan")

        # Fraction of jumps followed by another within 5 trading day s
        clustering_window = 5
        positions = np.where(indicator.to_numpy() == 1)[0]
        clustered = 0
        for pos in positions:
            nxt = positions[(positions > pos) & (positions <= pos + clustering_window)]
            if len(nxt) > 0:
                clustered += 1

        stats_out = {
            "n_jumps": n_jumps,
            "n_observations": n_obs,
            "jump_frequency": n_jumps / n_obs if n_obs else 0.0,
            "mean_inter_jump_time": mean_gap,
            "std_inter_jump_time": std_gap,
            "mean_jump_size": mean_size,
            "std_jump_size": std_size,
            "max_jump_size": max_size,
            "clustering_coefficient": clustered / n_jumps if n_jumps else 0.0,
            "time_units": TIME_UNIT,
        }
        self.jump_stats = stats_out
        return stats_out 

    def compare_detectors(
        self, spread_diff: pd.Series, window: int = 20
    ) -> pd.DataFrame:
        """
        Run every detector on the same series and report agreement.

        The module docstring has always advertised "multiple jump detection
        algorithms"; no comparison across them was ever run
        """
        results: Dict[str, pd.DataFrame] = {}
        for name in DETECTORS:
            try:
                results[name] = self.detect(spread_diff, method = name, window = window)
            except Exception as exc: # noqa: BLE001 - reported, not swallowed
                self._log(f" detector '{name}' failed: {type(exc).__name__}: {exc}")

        rows: List[dict] = []
        indicators = {k: v["jump_indicator"] for k, v in results.items()}

        for name, ind in indicators.items():
            st = self.calculate_jump_statistics(results[name])
            row = {
                "detector": name,
                "n_jumps": st["n_jumps"],
                "jump_frequency_pct": 100 * st["jump_frequency"],
                "mean_inter_jump_days": st["mean_inter_jump_time"],
                "clustering_coef": st["clustering_coefficient"],
            }
            for other, other_ind in indicators.items():
                if other == name:
                    continue
                both = int(((ind == 1) & (other_ind == 1)).sum())
                row[f"overlap_with_{other}"] = both 
            rows.append(row)

        return pd.DataFrame(rows)

if __name__ == "__main__":
    from pathlib import Path 

    from equity_pairs_loader import EquityPairsDataPipeline

    here = Path(__file__).resolve().parent 
    print("=" * 72)
    print("Jump detection -- CVX/XOM")
    print("=" * 72)

    pipeline = EquityPairsDataPipeline(
        asset_a_path = str(here / "OHLCV_CVX.csv"),
        asset_b_path = str(here / "OHLCV_XOM.csv"),
        asset_a_symbol = "CVX",
        asset_b_symbol = "XOM",
        verbose = False,
    )

    pipeline.load_from_csv(date_columns = "ts_event")
    pipeline.clean_data()
    spread_df = pipeline.construct_spread(method = "johansen", hedge_mode = "static")

    # .diff(), not .pct_change(): the spread is already in logs 
    spread_diff = spread_df["spread"].diff().dropna()

    detector = JumpDetector(significance_level=0.05, apply_fdr = True)
    comparison = detector.compare_detectors(spread_diff, window = 20)

    print("\n Detector comparison: ")
    print(comparison.to_string(index = False))
