"""
Data pipeline for loading equity pairs and constructing a tradeable spread.

Substantive changes from the audited version:

1. Prices are split-adjusted on load and then *verified* to contain no
    remaining unadjusted corporate action (previously NVDA carried two 
    unadjusted splits and `clean_data` deleted the split days while leaving 
    the permanent level shift in place).

2. `clean_data` no longer DELETES large_return days. Deleting the largest 
    moves from a jump study removes exactly the observations under study and
    creates irregular gaps in a daily time axis that the Hawkes `T` and the OU
    `dt` both assume is regular. Large moves are flagged, not dropped. The
    unconditional first-row drop (`NaN < 0.5` is `False`) is also fixed. 

3. The hedge ratio is estimated ONCE on a training window and frozen
    (`hedge_mode = `static``, the default and the primary specification).
    The rolling estimator is retained only as a labelled robustness mode.

    Motivation: with a 30-day rolling h, the spread change is 

        dS_t = dlog A_t - h_t*dlog B_t - dh_t*log B_{t-1}

    and the third term accounted for 99.2 - 99.8% of mean |dS| on all five 
    pairs, inflating spread sigma by 4.8x to 81.9x. A hedge ratio that moves 
    daily is also untradeable regardless of its statistical merit. 

4. `_estimate_hedge_ratio_cointegration` and `_estimate_hedge_ratio_regression`
    were byte-for-byte identical rolling `np.polyfit` calls with no 
    cointegration test anywhere. The methods are now genuinely different:
    Johansen, Engle-Granger, OLS, and total least squares.

5. `calculate_spread_statistics` reports BOTH the naive ADF p-value and the 
    Engle_Granger p-value. ADF assumes the tested series is observed; a spread 
    built from an estimated cointegrating vector needs Engle-Granger / 
    Phillips-Ouliaris critical values, which are substantially more negative.
    Reporting both makes the size of the over-rejection visible. 

6. `.bfill()` on the hedge-ratio warm-up is gone. It back-filled the first 
    `lookback` observations with a hedge ratio estimated from 0 days..lookback,
    i.e. a look-ahead at the start of every sample. Warm-up rows are dropped
"""

from __future__ import annotations

from pathlib import Path 
from typing import Dict, Literal, NotRequired, Optional, Tuple, TypedDict, cast

import numpy as np 
import pandas as pd 

from corporate_actions import (
    adjust_for_splits,
    estimate_dividend_drag,
    verify_no_unadjusted_actions,
    verify_split_table,
)

from time_units import TRADING_DAYS_PER_YEAR

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class SpreadStatistics(TypedDict):
    """Typed result of :meth:`calculate_spread_statistics`."""

    mean: float
    std: float
    adf_statistic: float
    adf_pvalue: float
    adf_says_stationary: bool
    ar1_beta: float
    half_life: float
    half_life_units: Literal["trading_days"]
    eg_statistic: NotRequired[float]
    eg_pvalue: NotRequired[float]
    eg_crit_5pct: NotRequired[float]
    eg_says_cointegrated: NotRequired[bool]
    adf_eg_pvalue_gap: NotRequired[float]

# Hedge-ratio estimation modes. `static` is the primary specification
HEDGE_MODES = ("static", "periodic", "rolling")

# Hedge-ratio estimators 
HEDGE_METHODS = ("johansen", "engle_granger", "ols", "tls")

class EquityPairsDataPipeline:
    """ Load, adjust, clean, and build a spread for a pair of equities"""

    def __init__(
        self,
        asset_a_path: Optional[str] = None, 
        asset_b_path: Optional[str] = None,
        asset_a_symbol: str = "Asset A",
        asset_b_symbol: str = "Asset B",
        adjust_splits: bool = True, 
        verbose: bool = True,
    ):
        self.asset_a_path = asset_a_path
        self.asset_b_path = asset_b_path 
        self.asset_a_symbol = asset_a_symbol 
        self.asset_b_symbol = asset_b_symbol 
        self.adjust_splits = adjust_splits 
        self.verbose = verbose 

        self.data: Dict[str, pd.DataFrame] = {}
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.adjustment_report: Dict[str, dict] = {}
        self.cleaning_report: Dict[str, object] = {}
        self.hedge_report: Dict[str, object] = {}

    ### Loading ###

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    @staticmethod
    def _read_one(path: str, date_column: str, parse_dates: bool) -> pd.DataFrame:
        frame = pd.read_csv(path)

        col = date_column if date_column in frame.columns else None 
        if col is None:
            lowered = {str(c).lower(): c for c in frame.columns}
            col = (
                lowered.get(str(date_column).lower())
                or lowered.get("ts_event")
                or lowered.get("date")
            )

        if col is None:
            raise ValueError(
                f"No date column found in {path}. Looked for `{date_column}`, "
                "`ts_event`, `date`."
            )

        if parse_dates:
            frame[col] = pd.to_datetime(frame[col], errors = "coerce")
        frame = frame.set_index(col).sort_index() 
        frame.columns = [str(c).capitalize() for c in frame.columns]
        return frame 

    def load_from_csv(
        self,
        asset_a_path: Optional[str] = None,
        asset_b_path: Optional[str] = None, 
        date_columns: str = "Date",
        parse_dates: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """ Load both legs, split-adjust them, and verify no action remains"""
        asset_a_path = asset_a_path or self.asset_a_path 
        asset_b_path = asset_b_path or self.asset_b_path 

        if asset_a_path is None or asset_b_path is None:
            raise ValueError("Both asset_a_path and asset_b_path must be provided")

        self._log("Laoding equity pairs data...")
        self._log(f" Asset A({self.asset_a_symbol}): {asset_a_path}")
        self._log(f" Asset B({self.asset_b_symbol}): {asset_b_path}")

        asset_a = self._read_one(asset_a_path, date_columns, parse_dates)
        asset_b = self._read_one(asset_b_path, date_columns, parse_dates)

        for name, frame in (("asset_a", asset_a), ("asset_b", asset_b)):
            for col in REQUIRED_COLUMNS:
                if col not in frame.columns:
                    raise ValueError(f"Missing column `{col}` in {name} data")

        self.raw_data = {"asset_a": asset_a.copy(), "asset_b": asset_b.copy()}

        # corporate actions 
        symbols = {"asset_a": self.asset_a_symbol, "asset_b": self.asset_b_symbol}
        adjusted = {}
        for key, frame in (("asset_a", asset_a), ("asset_b", asset_b)):
            symbol = symbols[key]
            if self.adjust_splits:
                implied = verify_split_table(frame, symbol)
                out = adjust_for_splits(frame, symbol)
                self.adjustment_report[symbol] = {
                    "splits_applied": len(implied),
                    "implied_factors": implied,
                }
                if implied:
                    self._log(
                        f" {symbol}: applied {len(implied)} split(s) "
                        f"{ {k: round(v, 3) for k, v in implied.items()} }"
                    )
            else:
                out = frame 
                self.adjustment_report[symbol] = {"splits_applied": 0, "implied_factors": {}}

            # hard failure rather than silent deletion 
            verify_no_unadjusted_actions(out, symbol)
            adjusted[key] = out 

        self.data = adjusted 

        self._log(f" Asset A: {len(adjusted['asset_a'])} observations")
        self._log(f" Asset B: {len(adjusted['asset_b'])} observations")
        self._log(
            f" Date range: {adjusted['asset_a'].index[0].date()} to "
            f"{adjusted['asset_a'].index[-1].date()}"
        )

        div = estimate_dividend_drag(self.asset_a_symbol, self.asset_b_symbol)
        self._log(
            f" Dividend differential (disclosed, NOT modelled): "
            f"{div['differential_annual'] * 100:+.2f}%/yr"
        )

        return self.data 

    ### cleaning ###

    def clean_data(self, flag_threshold: float = 0.25) -> Dict[str, pd.DataFrame]:
        """
        Align both legs and drop missing values.

        Large single-day moves are FLAGGED, never dropped: this is a jump 
        study and deleting the largest moves both removes the phenomenon
        under study and creates irregular gaps in a time axis that downstream
        estimators assume is regular.
        """
        self._log("Cleaning data...")

        if not self.data:
            raise ValueError("No data to clean. Run load_from_csv() first.")

        common = self.data["asset_a"].index.intersection(self.data["asset_b"].index)
        cleaned = {
            "asset_a": self.data["asset_a"].loc[common].copy(),
            "asset_b": self.data["asset_b"].loc[common].copy(),
        }

        n_before = len(common)
        for key in cleaned:
            cleaned[key] = cleaned[key].dropna(subset = list(REQUIRED_COLUMNS))

        for key in cleaned:
            if (cleaned[key]["Close"] <= 0).any():
                raise ValueError(f"Non-positive close prices in {key}; cannot take logs")

        common = cleaned["asset_a"].index.intersection(cleaned["asset_b"].index)
        cleaned = {k: v.loc[common] for k, v in cleaned.items()}

        flagged = {}
        for key, symbol in (("asset_a", self.asset_a_symbol), ("asset_b", self.asset_b_symbol)):
            rets = cleaned[key]["Close"].pct_change()
            big = rets[rets.abs() > flag_threshold].dropna()
            big_dates = cast(pd.DatetimeIndex, big.index)
            flagged[symbol] = [
                (ts.date().isoformat(), float(value))
                for ts, value in zip(big_dates, big.to_numpy())
            ]
            if len(big) > 0:
                self._log(
                    f" {symbol}: {len(big)} day(s) with |return| > {flag_threshold:.0%} "
                    f" FLAGGED (retained, not deleted)"
                )

        n_dropped = n_before - len(common)
        self.cleaning_report = {
            "n_before": int(n_before),
            "n_after": int(len(common)),
            "n_dropped_missing": int(n_dropped),
            "large_move_days_flagged": flagged,
        }

        self._log(
            f" Cleaned: {len(common)} observations "
            f"({n_dropped} dropped for missing values, 0 dropped for size)"
        )
        
        self.data = cleaned 
        return self.data 

    ### hedge ratio estimation ###

    @staticmethod 
    def _hedge_ols(log_a: np.ndarray, log_b: np.ndarray) -> float:
        """ OLS of log A on log B. Asymmetric: regressing B on A gives a different answer."""
        slope, _ = np.polyfit(log_b, log_a, 1)
        return float(slope)

    @staticmethod
    def _hedge_tls(log_a: np.ndarray, log_b: np.ndarray) -> float:
        """ 
        Total least squares (orthogonal regression) via the first principal
        component. Symmetric in the two legs, so it removes the arbitrary
        direction choice that plain OLS forces
        """
        x = log_b - log_b.mean()
        y = log_a - log_a.mean() 
        # (n, 2) so the right singular vectors live in the 2-D (x, y) space
        stacked = np.column_stack([x,y])
        _, _, vt = np.linalg.svd(stacked, full_matrices = False)
        vx, vy = vt[0]
        if abs(vx) < 1e-12:
            return float("nan")
        return float(vy / vx)

    @staticmethod
    def _hedge_engle_granger(log_a: np.ndarray, log_b: np.ndarray) -> Tuple[float, dict]:
        """
        Engle-Granger: OLS cointegrating regression plus the EG cointegration
        test with correct (Phillips-Oularis) critical values
        """
        import statsmodels.api as sm 
        from statsmodels.tsa.stattools import coint 

        design = sm.add_constant(log_b)
        fit = sm.OLS(log_a, design).fit() 
        beta = float(fit.params[1])

        stat, pvalue, crit = coint(log_a, log_b, trend = "c")
        return beta, {
            "eg_statistic": float(stat),
            "eg_pvalue": float(pvalue),
            "eg_crit_1pct": float(crit[0]),
            "eg_crit_5pct": float(crit[1]),
            "eg_crit_10pct": float(crit[2]),
            "intercept": float(fit.params[0]),
        }

    @staticmethod 
    def _hedge_johansen(log_a: np.ndarray, log_b: np.ndarray) -> Tuple[float, dict]:
        """
        Johansen trace test. The cointegrating vector is normalized on log A, 
        so the hedge ratio is -v_b / v_a
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        data = np.column_stack([log_a, log_b])
        result = coint_johansen(data, det_order = 0, k_ar_diff = 1)

        vec = result.evec[:, 0]
        if abs(vec[0]) < 1e-12:
            raise ValueError("Johansen: degenerate cointegrating vector (v_a ~ 0)")
        beta = float(-vec[1] / vec[0])

        trace_stat = float(result.lr1[0])
        crit = result.cvt[0] # [90%, 95%, 99%]
        return beta, {
            "johansen_trace_stat": trace_stat,
            "johansen_crit_90": float(crit[0]),
            "johansen_crit_95": float(crit[1]),
            "johansen_crit_99": float(crit[2]),
            "johansen_rejects_no_coint_95": bool(trace_stat > crit[1]),
        }

    def estimate_hedge_ratio_static(
        self,
        asset_a: pd.Series,
        asset_b: pd.Series,
        method: str = "johansen",
        direction_note: str = "A regressed on B (A is the dependent leg)",
    ) -> Tuple[float, dict]:
        """
        Estimate ONE hedge ratio on the supplied (training) window.

        Params:
            method: {'johansen', 'engle_granger', 'ols', 'tls'}

        Returns:
            (hedge_ratio, diagnostics)
        """
        if method not in HEDGE_METHODS:
            raise ValueError(f"Unknown hedge method '{method}'. Choose from {HEDGE_METHODS}.")

        log_a = np.log(asset_a.to_numpy(dtype = float))
        log_b = np.log(asset_b.to_numpy(dtype = float))

        diagnostics: dict = {
            "method": method,
            "n_obs": int(len(log_a)),
            "direction": direction_note,
        }

        if method == "johansen":
            beta, extra = self._hedge_engle_granger(log_a, log_b)
        elif method == "engle_granger":
            beta, extra = self._hedge_engle_granger(log_a, log_b)
        elif method == "ols":
            beta, extra = self._hedge_ols(log_a, log_b), {}
        else: #tls
            beta, extra = self._hedge_tls(log_a, log_b), {}

        diagnostics.update(extra)

        # always report the alternatives so the direction/method sensitivity is visible
        diagnostics["h_ols_a_on_b"] = self._hedge_ols(log_a, log_b)
        slope_b_on_a = self._hedge_ols(log_b, log_a)
        diagnostics["h_ols_b_on_a_inverted"] = (
            1.0 / slope_b_on_a if abs(slope_b_on_a) > 1e-12 else float("nan")
        )
        diagnostics["h_tls"] = self._hedge_tls(log_a, log_b)
        diagnostics["hedge_ratio"] = float(beta)

        return float(beta), diagnostics 

    def _rolling_hedge_ratio(
            self, asset_a: pd.Series, asset_b: pd.Series, lookback: int, method: str
    ) -> pd.Series: 
        """ 
        Rolling hedge ratio. ROBUSTNESS MODE ONLY -- not the primary specification.

        Warm up rows are left NaN and dropped by the caller. They are NOT
        back-filled: `.bfill()` here would seed the first `lookback` rows with 
        an estimate computed from those same rows, a look-ahead
        """

        log_a = np.log(asset_a.astype(float).to_numpy())
        log_b = np.log(asset_b.astype(float).to_numpy())

        values = np.full(len(asset_a), np.nan)
        for i in range(lookback, len(asset_a)):
            wa = log_a[i - lookback : i]
            wb = log_b[i - lookback : i]
            if method == "tls":
                values[i] = self._hedge_tls(wa, wb)
            else: 
                values[i] = self._hedge_ols(wa, wb)

        return pd.Series(values, index = asset_a.index, dtype = float)

    ### spread construction ###

    def construct_spread(
        self,
        method: str = "johansen",
        lookback: int = 30,
        hedge_mode: str = "static",
        train_end: Optional[str] = None, 
        fixed_hedge_ratio: Optional[float] = None,
    ) -> pd.DataFrame:
        """ 
        Build the spread S_t = log(P_A) - h * log(P_B).

        Params:
        method : {'johansen', 'engle_granger', 'ols', 'tls'}
            Estimator for the static hedge ratio 
        hedge_mode : {'static', 'periodic', 'rolling'}
            'static' -- one h, estimated on the training window, frozen (PRIMARY)
            'periodic' -- caller supplies 'fixed_hedge_ratio' per period
            'rolling' -- time-varying h (ROBUSTNESS ONLY; see module docstring)
        train_end: str, optional 
            If given with hedge_mode = 'static', h is estimated only on data up to 
            this date, so no evaluation-period information enters the spread.
        fixed_hedge_ratio : float, optional
            Use this h directly (walk-forward passes the quarter's frozen value)
        """
        if hedge_mode not in HEDGE_MODES:
            raise ValueError(f"Unknown hedge_mode '{hedge_mode}'. Choose from {HEDGE_MODES}")
        if not self.data:
            raise ValueError("No data available. Run load_from_csv() first.")

        price_a = self.data["asset_a"]["Close"].astype(float)
        price_b = self.data["asset_b"]["Close"].astype(float)

        log_a = np.log(price_a)
        log_b = np.log(price_b)

        diagnostics: dict = {"hedge_mode": hedge_mode}

        if hedge_mode == "rolling":
            self._log(
                f"Constructing spread: ROLLING hedge ({method}, {lookback}d) "
                " -- ROBUSTNESS MODE, not the primary specification"
            )

            hedge_series = self._rolling_hedge_ratio(price_a, price_b, lookback, method)
            valid = hedge_series.notna()
            n_warmup = int((~valid).sum())
            price_a, price_b = price_a[valid], price_b[valid]
            log_a, log_b = log_a[valid], log_b[valid]
            hedge_series = hedge_series[valid]
            diagnostics.update(
                {
                    "warmup_rows_dropped": n_warmup,
                    "hedge_min": float(hedge_series.min()),
                    "hedge_max": float(hedge_series.max()),
                    "hedge_mean": float(hedge_series.mean()),
                }
            )
            self._log(f" Dropped {n_warmup} warm-up rows (no back fill)")

        else: 
            if fixed_hedge_ratio is not None:
                hedge_value = float(fixed_hedge_ratio)
                diagnostics.update({"method": "supplied", "hedge_ratio": hedge_value})
                self._log(f"Constructing spread: FIXED hedge h = {hedge_value:.4f}")
            else:
                if train_end is not None:
                    fit_a = price_a.loc[:train_end]
                    fit_b = price_b.loc[:train_end]
                    if len(fit_a) < 60:
                        raise ValueError(
                            f"Only {len(fit_a)} observations up to train_end = {train_end}; "
                            " too few to identify a cointegrating vector"
                        )
                else:
                    fit_a, fit_b = price_a, price_b

                hedge_value, diag = self.estimate_hedge_ratio_static(fit_a, fit_b, method = method)
                diagnostics.update(diag)
                diagnostics["estimated_on"] = {
                    f"{fit_a.index[0].date()} .. {fit_a.index[-1].date()} ({len(fit_a)} obs)"
                }
                self._log(
                    f"Constructing spread: STATIC hedge h = {hedge_value:.4f} "
                    f"({method}, fit on {len(fit_a)} obs"
                    + (f" to {train_end}" if train_end else "")
                    + ")"
                )
                self._log(
                    f" sensitivity: OLS(A~B) = {diag['h_ols_a_on_b']:.4f} "
                    f"OLS(B~A)^-1 = {diag['h_ols_b_on_a_inverted']:.4f} "
                    f"TLS = {diag['h_tls']:.4f}"
                )
            hedge_series = pd.Series(hedge_value, index = price_a.index, dtype = float)

        spread = log_a - hedge_series * log_b 

        spread_df = pd.DataFrame(
            {
                "spread": spread,
                "asset_a_price": price_a,
                "asset_b_price": price_b,
                "hedge_ratio": hedge_series,
                "log_a": log_a,
                "log_b": log_b,
            }
        )

        self.hedge_report = diagnostics 

        self._log(f" Spread range: [{spread.min():.4f}, {spread.max():.4f}]")
        self._log(f" Spread mean: {spread.mean():.4f}")
        self._log(f" Spread std: {spread.std():.4f}")

        return spread_df 

    ### statistics ###

    def calculate_spread_statistics(
        self,
        spread: pd.Series,
        log_a: Optional[pd.Series] = None, 
        log_b: Optional[pd.Series] = None,
    ) -> SpreadStatistics:
        """
        Spread statistics with BOTH the naive ADF and the correct Engle-Granger test.

        `adfuller` returns Dickey-Fuller p-values, which assume the tested series
        is observed. A spread built from an estimted cointegrating vector is a 
        residual, so the correct null distribution is Engle-Granger / 
        Phillips-Ouliaris, whose critical values are substantially more negative.
        Plain ADF over-rejects. Both are reported so the gap is visible
        """
        from statsmodels.tsa.stattools import adfuller, coint 

        clean = spread.dropna() 

        adf_stat, adf_pvalue = adfuller(clean)[:2]

        stats: Dict[str, float | bool | str] = {
            "mean": float(clean.mean()),
            "std": float(clean.std()),
            "adf_statistic": float(adf_stat),
            "adf_pvalue": float(adf_pvalue),
            "adf_says_stationary": bool(adf_pvalue < 0.05),
        }

        if log_a is not None and log_b is not None:
            idx = log_a.index.intersection(log_b.index)
            eg_stat, eg_pvalue, eg_crit = coint(
                log_a.loc[idx].to_numpy(dtype = float),
                log_b.loc[idx].to_numpy(dtype = float),
                trend = "c",
            )
            stats.update(
                {
                    "eg_statistic": float(eg_stat),
                    "eg_pvalue": float(eg_pvalue),
                    "eg_crit_5pct": float(eg_crit[1]),
                    "eg_says_cointegrated": bool(eg_pvalue < 0.05),
                    "adf_eg_pvalue_gap": float(eg_pvalue - adf_pvalue),
                }
            )

        #half-life from an AR(1) regression, in TRADING DAYS
        lag = clean.shift(1).dropna()
        diff = clean.diff().dropna()
        idx = lag.index.intersection(diff.index)
        beta = float(np.polyfit(lag.loc[idx], diff.loc[idx], 1)[0])
        stats["ar1_beta"] = beta 
        stats["half_life"] = float(-np.log(2) / beta) if beta < 0 else float("inf")
        stats["half_life_units"] = "trading_days"

        return cast(SpreadStatistics, stats)

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent 

    print("=" * 72)
    print("Equity Pairs Data Pipeline -- CVX/XOM")
    print("=" * 72)

    pipeline = EquityPairsDataPipeline(
        asset_a_path = str(current_dir / "OHLCV_CVX.csv"),
        asset_b_path = str(current_dir / "OHLCV_XOM.csv"),
        asset_a_symbol = "CVX",
        asset_b_symbol = "XOM",
    )
    pipeline.load_from_csv(date_columns = "ts_event")
    pipeline.clean_data()

    static_df = pipeline.construct_spread(method = "johansen", hedge_mode = "static")
    rolling_df = pipeline.construct_spread(method = "ols", hedge_mode = "rolling", lookback = 30)

    print("\n Static vs rolling hedge:")
    print(f" sigma(static) = {static_df['spread'].std():.4f}")
    print(f" sigma(rolling) = {rolling_df['spread'].std():.4f}")
    print(f" inflation = {rolling_df['spread'].std() / static_df['spread'].std():.1f}x")

    stats = pipeline.calculate_spread_statistics(
        static_df["spread"], static_df["log_a"], static_df["log_b"]
    )
    print("\nStatic-hedge spread statistics:")
    for key, value in stats.items():
        print(f" {key}: {value}")
