"""
Cross-sectional pair screening

Why this exists:
The binding constraint on this study is not signal quality, it is BETS PER
YEAR. At a 50-70 day half-life a pair offers roughly two independent round 
trips annually, so a two-year evaluation window carries about four effective
observations. No parameters choice fixes that; only breadth does. 

Ten symbols give C(10, 2) = 45 candidate pairs. This module screens all of them 
on the TRAINING WINDOW ONLY against criteria fixed in advance, and returns the 
survivors. That is also the principled way to drop a pair: AMD/NVDA is not 
exluded because it performed badly out of sample -- it is excluded if and only 
if it fails a screen that was specified before looking at the outcome. 

Screening criteria (all evaluated on training data):

1. Engle-Granger cointegration p < COINT_P_MAX
    The spread is a residual from an estimated cointegrating vector, so 
    Engle-Granger critical values apply, not plain ADF.
2. Half-life within [MIN_HALF_LIFE, MAX_HALF_LIFE] trading days
    Too fast and costs dominate; too slow and there are no bets
3. Predictive t-statistic on forward spread change <= PREDICT_T_MAX
    Mean reversion must actually be present, with the right sign.
4. Edge per round trip >= EDGE_COST_MULTIPLE x round-trip cost
    An edge smaller than a few multiples of cost is not tradeable, which is
    what disqualifies near-identical ETF pairs
5. Hedge ratio within (0, MAX_HEDGE)
    A negative hedge ratio means the "spread" is long both legs, which is 
    economically incoherent -- the audit flagged exactly this when a rolling 
    estimator produced h in [-0.37, 2.10] for two oil majors. 

Nothing here looks at validation or walk-forward data.

MULTIPLICITY -- the point that decides the answer:
Screening 45 pairs at a nominal 5% level is itself 45 hypothesis tests, so 
roughly 2 spurious "cointegrated" pairs are expected from noise alone. The 
screen therefor reports Benjamini-Hochberg and Bonferroni-adjusted
cointegration decisions alongside the nominal one, exactly as the jump detector 
does. Selecting on the nominal p-value and trading the winner is the 
cross-sectional version of the mistake the audit found in the time series. 
"""

from __future__ import annotations 

import argparse
import itertools 
from pathlib import Path 
from typing import Dict, List, Optional, Tuple 

import numpy as np 
import pandas as pd 

from config import Config 
from equity_pairs_loader import EquityPairsDataPipeline

__all__ = [
    "SCREEN_CRITERIA",
    "available_symbols",
    "screen_pair",
    "screen_all",
    "selected_pairs",
]

PROJECT_ROOT = Path(__file__).resolve().parent 

# Fixed in advance
SCREEN_CRITERIA: Dict[str, float] = {
    "COINT_P_MAX": 0.05,
    "MIN_HALF_LIFE": 5.0,
    "MAX_HALF_LIFE": 120.0,
    "PREDICT_T_MAX": -1.5,
    "EDGE_COST_MULTIPLE": 5.0,
    "MAX_HEDGE": 5.0,
    "PREDICT_HORIZON": 20,
    #Family wise / false discovery level applied ACROSS the 45 candidates.
    "MULTIPLICITY_ALPHA": 0.05,
}

def available_symbols(root: Optional[Path] = None) -> List[str]:
    """ Symbols with a committed OHLCV file"""
    root = root or PROJECT_ROOT
    return sorted(p.stem.replace("OHLCV_", "") for p in root.glob("OHLCV_*.csv"))

def screen_pair(
    symbol_a: str, 
    symbol_b: str, 
    train_start: str, 
    train_end: str, 
    config: Optional[Config] = None,
) -> Dict:
    """ Screen one candidate pair on the training window. Never raises"""
    import statsmodels.api as sm 

    cfg = config or Config()
    criteria = SCREEN_CRITERIA
    result: Dict = {"pair": f"{symbol_a}_{symbol_b}", "symbol_a": symbol_a, "symbol_b": symbol_b}

    try:
        pipeline = EquityPairsDataPipeline(
            asset_a_path = str(PROJECT_ROOT / f"OHLCV_{symbol_a}.csv"),
            asset_b_path = str(PROJECT_ROOT / f"OHLCV_{symbol_b}.csv"),
            asset_a_symbol = symbol_a,
            asset_b_symbol = symbol_b,
            verbose = False,
        )
        pipeline.load_from_csv(date_columns = cfg.data.date_columns)
        pipeline.clean_data()

        spread_df = pipeline.construct_spread(
            method = cfg.data.hedge_ratio_method, hedge_mode = "static", train_end = train_end
        )
        train = spread_df.loc[train_start:train_end]
        if len(train) < 250:
            result["error"] = f"only {len(train)} training observations"
            return result 

        stats = pipeline.calculate_spread_statistics(
            train["spread"], train["log_a"], train["log_b"]
        )
        hedge = float(train["hedge_ratio"].iloc[0])

        half_life = stats["half_life"]
        coint_p = stats.get("eg_pvalue", stats["adf_pvalue"])
        spread_sd = stats["std"]

        # Predictive regression on the training window only
        horizon = int(criteria["PREDICT_HORIZON"])
        forward = train["spread"].shift(-horizon) - train["spread"]
        prior = train["spread"].shift(1)
        z = (
            (train["spread"] - prior.rolling(60, min_periods = 20).mean())
            / prior.rolling(60, min_periods = 20).std()
        )
        frame = pd.DataFrame({"z": z, "forward": forward}).dropna()

        if len(frame) < 100:
            predict_t = np.nan
            predict_r2 = np.nan 
        else:
            model = sm.OLS(
                frame["forward"].to_numpy(), sm.add_constant(frame["z"].to_numpy())
            ).fit(cov_type = "HAC", cov_kwds = {"maxlags": horizon * 2})
            predict_t = float(model.tvalues[1])
            predict_r2 = float(model.rsquared)

        # Tradeability: edge per round trip against round-trip cost 
        # abs(hedge) because gross notional is |leg A| + |leg B| regardless of 
        # sign; a negative hedge is caught by the hedge_sensible check instead 
        # of silently producitng a negative "edge".
        edge = (2.0 - 0.5) * spread_sd / (1.0 + abs(hedge))
        round_trip_cost = 2 * (
            cfg.backtest.commission_rate + cfg.backtest.slippage_bps / 1e4
        )
        borrow = cfg.backtest.borrow_rates.get(
            symbol_b, cfg.backtest.default_borrow_rate
        ) * (2 * min(half_life, 250) / 252)
        total_cost = round_trip_cost + borrow 
        edge_multiple = edge / total_cost if total_cost > 0 else np.inf

        checks = {
            "cointegrated": bool(coint_p < criteria["COINT_P_MAX"]),
            "half_life_ok": bool(
                criteria["MIN_HALF_LIFE"] <= half_life <= criteria["MAX_HALF_LIFE"]
            ),
            "predictive": bool(
                np.isfinite(predict_t) and predict_t <= criteria["PREDICT_T_MAX"]
            ),
            "edge_covers_cost": bool(edge_multiple >= criteria["EDGE_COST_MULTIPLE"]),
            "hedge_sensible": bool(0.0 < hedge < criteria["MAX_HEDGE"]),
        }

        result.update(
            {
                "hedge_ratio": hedge,
                "coint_pvalue": coint_p,
                "adf_pvalue": stats["adf_pvalue"],
                "half_life": half_life,
                "spread_sd": spread_sd,
                "predict_t": predict_t,
                "predict_r2": predict_r2,
                "edge_per_trip_pct": 100 * edge, 
                "cost_per_trip_pct": 100 * total_cost,
                "edge_cost_multiple": edge_multiple,
                "trips_per_year": 252.0 / (2 * half_life) if half_life > 0 else 0.0,
                "sharpe_ceiling": float(
                    np.sqrt(252.0 * (np.log(2) / half_life) / np.pi)
                )
                if half_life > 0
                else np.nan,
                **checks,
                "n_failed": sum(1 for v in checks.values() if not v),
                "selected": all(checks.values()),
                "n_train_obs": len(train),
            }
        )
    except Exception as exc: # noqa: BLE001 - a failed candidate is data, not a crash
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["selected"] = False

    return result 

def screen_all(
    symbols: Optional[List[str]] = None,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    config: Optional[Config] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """ screen every unordered symbol combination"""
    cfg = config or Config()
    symbols = symbols or available_symbols()
    train_start = train_start or cfg.train_val.train_start 
    train_end = train_end or cfg.train_val.train_end

    candidates = list(itertools.combinations(symbols,2))
    if verbose:
        print(
            f"Screening {len(candidates)} candidate pairs from {len(symbols)} symbols "
            f"on {train_start}..{train_end} (training window only)"
        )

    rows = []
    for i, (a,b) in enumerate(candidates, 1):
        row = screen_pair(a, b, train_start, train_end, cfg)
        rows.append(row)
        if verbose and i % 10 == 0:
            print(f" ...{i}/{len(candidates)}")

    frame = pd.DataFrame(rows)
    frame = _apply_multiplicity(frame)

    if "selected" in frame.columns:
        frame = frame.sort_values(
            ["selected_fdr", "selected", "predict_t"], ascending = [False, False, True]
        ).reset_index(drop = True)
    return frame 

def _apply_multiplicity(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Correct the cointegration decision for having tested every candidate.

    Without this, "we screened 45 pairs and found one" is indistinguishable
    from "we screened 45 pairs and found what noise produces".
    """
    from jump_detector import benjamini_hochberg

    if "coint_pvalue" not in frame.columns:
        return frame 

    alpha = SCREEN_CRITERIA["MULTIPLICITY_ALPHA"]
    pvalues = frame["coint_pvalue"].to_numpy(dtype = float)
    n_tests = int(np.isfinite(pvalues).sum())

    frame["n_candidates_tested"] = n_tests 
    frame["expected_false_positives"] = alpha * n_tests 
    frame["coint_bonferroni"] = pvalues < (alpha / max(n_tests, 1))
    frame["coint_fdr"] = benjamini_hochberg(pvalues, alpha)

    other_checks = ["half_life_ok", "predictive", "edge_covers_cost", "hedge_sensible"]
    have = [c for c in other_checks if c in frame.columns]
    passes_rest = frame[have].all(axis = 1) if have else True 

    frame["selected_fdr"] = frame["coint_fdr"] & passes_rest 
    frame["selected_bonferroni"] = frame["coint_bonferroni"] & passes_rest 
    return frame 

def selected_pairs(
        frame: pd.DataFrame, criterion: str = "selected_fdr"
) -> Dict[str, Tuple[str, str]]:
    """
    Survivors, in the {key: (symbol_a, symbol_b)} shape config.PAIRS uses.

    Defaults to the FDR-corrected decision. Passing criterion = 'selected' gives
    the uncorrected set, which is only useful for showing how much of it is 
    multiplicity.
    """
    if criterion not in frame.columns:
        criterion = "selected"
    chosen = frame[frame.get(criterion, False) == True] #noqa: E712
    return {
        row["pair"]: (row["symbol_a"], row["symbol_b"]) for _, row in chosen.iterrows()
    }

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Screen all candidate pairs")
    parser.add_argument("--out", default="outputs/pair_screen.csv")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = Config()
    frame = screen_all(config=cfg, verbose=not args.quiet)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    ok = frame[frame.get("selected", False) == True]  # noqa: E712

    print("\n" + "=" * 78)
    print("SCREEN CRITERIA (fixed in advance, training window only)")
    print("=" * 78)
    for key, value in SCREEN_CRITERIA.items():
        print(f"  {key:22s} {value}")

    n_tests = int(frame["coint_pvalue"].notna().sum())
    print("\n" + "=" * 78)
    print("MULTIPLICITY ACROSS THE SCREEN")
    print("=" * 78)
    print(f"  candidates tested                 {n_tests}")
    print(f"  expected false positives at 5%    {0.05 * n_tests:.1f}")
    print(f"  pass cointegration, nominal       {int(frame['cointegrated'].sum())}")
    print(f"  economically sensible hedge       {int(frame['hedge_sensible'].sum())}")
    print(f"  pass cointegration, BH-FDR        {int(frame['coint_fdr'].sum())}")
    print(f"  pass cointegration, Bonferroni    {int(frame['coint_bonferroni'].sum())}")
    print(f"  smallest cointegration p-value    {frame['coint_pvalue'].min():.4f}")
    print(f"\n  SELECTED after FDR correction     {int(frame['selected_fdr'].sum())}")
    print(f"  SELECTED uncorrected (unsafe)     {int(frame['selected'].sum())}")

    print("\n" + "=" * 78)
    print(f"UNCORRECTED SURVIVORS: {len(ok)} of {len(frame)} candidates")
    print("  (shown to demonstrate what multiplicity produces, NOT to trade)")
    print("=" * 78)
    cols = [
        "pair", "hedge_ratio", "coint_pvalue", "half_life", "predict_t",
        "predict_r2", "edge_cost_multiple", "trips_per_year", "sharpe_ceiling",
    ]
    if len(ok):
        print(ok[cols].round(4).to_string(index=False))
    else:
        print("  (none)")

    print("\n" + "=" * 78)
    print("ORIGINAL FIVE PAIRS -- how each fares under the screen")
    print("=" * 78)
    original = ["CVX_XOM", "AMD_NVDA", "SPY_IVV", "GS_MS", "GLD_GDX"]
    sub = frame[frame["pair"].isin(original)]
    show = [
        "pair", "coint_pvalue", "half_life", "predict_t", "edge_cost_multiple",
        "cointegrated", "half_life_ok", "predictive", "edge_covers_cost",
        "hedge_sensible", "selected",
    ]
    print(sub[[c for c in show if c in sub.columns]].round(4).to_string(index=False))

    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
