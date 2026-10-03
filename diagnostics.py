"""
Diagnostics: why is performance poor, and what could change it?

Answers the question "can this be improved, or is the method poor regardless?"
with evidence rather than opinion. Three analyses:

    1. ceiling_analysis     -- theoretical Sharpe ceiling per pair, independent
                               round trips per year, and cost as a share of edge
    2. predictability_test  -- does the z-score actually predict forward spread
                               change, in and out of sample?
    3. stop_sensitivity     -- are the fixed percentage stops destroying the edge?

Run:
    python diagnostics.py                # all three, all pairs
    python diagnostics.py --which stops  # one analysis

IMPORTANT -- these are DIAGNOSTICS, not results. `stop_sensitivity` in
particular evaluates several risk configurations on the TRAINING window. Its
output is evidence about a structural defect, not a tuned specification to
report. Re-tuning on these numbers and quoting the winner is precisely the
practice the audit criticised (finding 1.7).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import PAIRS, Config
from pipeline import PairPipeline
from results_io import ResultsWriter

PROJECT_ROOT = Path(__file__).resolve().parent

__all__ = ["ceiling_analysis", "predictability_test", "stop_sensitivity"]


def _require_spread_df(pipe: PairPipeline) -> pd.DataFrame:
    """Return the constructed spread, or fail clearly if acquisition did not finish."""
    spread_df = pipe.spread_df
    if spread_df is None:
        raise RuntimeError("PairPipeline has no spread data; call acquire_data() first")
    return spread_df


def _require_cleaned_data(pipe: PairPipeline) -> Dict[str, pd.DataFrame]:
    """Return cleaned legs, or fail clearly if acquisition did not finish."""
    cleaned_data = pipe.cleaned_data
    if cleaned_data is None:
        raise RuntimeError("PairPipeline has no cleaned data; call acquire_data() first")
    return cleaned_data


def _fit_once(key: str):
    """Load a pair and fit the training bundle. Shared by all three analyses."""
    cfg = Config().for_pair(key)
    tv = cfg.train_val
    pipe = PairPipeline(cfg, verbose=False)
    pipe.acquire_data(train_end=tv.train_end)
    spread_df = _require_spread_df(pipe)
    bundle = pipe.fit_models(spread_df.loc[tv.train_start : tv.train_end], train_end=tv.train_end)
    return cfg, pipe, bundle


# --------------------------------------------------------------------- #
# 1. ceiling
# --------------------------------------------------------------------- #

def ceiling_analysis(pairs: List[str]) -> pd.DataFrame:
    """
    Theoretical Sharpe ceiling for trading an OU spread.

    Sizing the position proportional to (theta - X), the daily Sharpe is
    kappa*|theta - X| / sigma. Averaging |theta - X| over the stationary law
    (E|theta - X| = sigma_X * sqrt(2/pi), sigma_X = sigma / sqrt(2 kappa)) gives

        daily Sharpe = sqrt(kappa / pi)
        annualised   = sqrt(252 * kappa / pi)

    This is an IDEALISED upper bound: perfect knowledge of (kappa, theta,
    sigma), continuous rebalancing, no costs, no position limits. A realised
    Sharpe far below it is expected; a realised Sharpe near it would be
    suspicious.

    Also reports independent round trips per year (~one per two half-lives),
    which is the quantity that determines how much evidence a sample can carry.
    """
    rows = []
    for key in pairs:
        cfg, pipe, bundle = _fit_once(key)

        kappa = bundle.mrjd_params["kappa"]
        sigma = bundle.mrjd_params["sigma"]
        half_life = np.log(2) / kappa
        stationary_sd = sigma / np.sqrt(2 * kappa)

        sharpe_ceiling = float(np.sqrt(252.0 * kappa / np.pi))
        trips_per_year = 252.0 / (2 * half_life)

        # Edge per round trip entering at 2 sd and exiting at 0.5 sd, expressed
        # as a fraction of gross notional.
        edge = (2.0 - 0.5) * stationary_sd / (1 + bundle.hedge_ratio)

        round_trip_cost = 2 * (
            cfg.backtest.commission_rate + cfg.backtest.slippage_bps / 1e4
        )
        borrow = cfg.backtest.borrow_rates.get(
            cfg.data.asset_b_symbol, cfg.backtest.default_borrow_rate
        ) * (2 * half_life / 252)

        rows.append(
            {
                "pair": key,
                "half_life_days": half_life,
                "kappa": kappa,
                "spread_sd": stationary_sd,
                "sharpe_ceiling": sharpe_ceiling,
                "independent_trips_per_yr": trips_per_year,
                "edge_per_trip_pct": 100 * edge,
                "round_trip_cost_pct": 100 * round_trip_cost,
                "borrow_pct": 100 * borrow,
                "cost_share_of_edge_pct": 100 * (round_trip_cost + borrow) / max(edge, 1e-12),
                "hedge_ratio": bundle.hedge_ratio,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# 2. predictability
# --------------------------------------------------------------------- #

def predictability_test(pairs: List[str], horizons=(5, 20, 60)) -> pd.DataFrame:
    """
    Regress forward spread change on the current z-score, with HAC errors.

        spread[t+h] - spread[t]  ~  z[t]

    Mean reversion implies a NEGATIVE slope: a high z should be followed by a
    fall. Run on the training and validation windows separately, so in-sample
    fit can be compared against out-of-sample stability.

    This is the test of the project's underlying premise. If the slope is not
    reliably negative, no amount of signal engineering on top will help.
    """
    import statsmodels.api as sm

    rows = []
    for key in pairs:
        cfg, pipe, bundle = _fit_once(key)
        tv = cfg.train_val
        spread_df = _require_spread_df(pipe)
        artifacts = pipe.compute_artifacts(spread_df, bundle)
        z_all = artifacts["z_score"]
        spread_all = spread_df["spread"]

        for label, lo, hi in (
            ("train", tv.train_start, tv.train_end),
            ("val", tv.val_start, tv.val_end),
        ):
            z = z_all.loc[lo:hi]
            spread = spread_all.loc[lo:hi]

            for h in horizons:
                forward = spread.shift(-h) - spread
                frame = pd.DataFrame({"z": z, "forward": forward}).dropna()
                if len(frame) < 60:
                    continue

                design = sm.add_constant(frame["z"].to_numpy())
                model = sm.OLS(frame["forward"].to_numpy(), design).fit(
                    cov_type="HAC", cov_kwds={"maxlags": h * 2}
                )
                rows.append(
                    {
                        "pair": key,
                        "window": label,
                        "horizon_days": h,
                        "beta": float(model.params[1]),
                        "t_stat": float(model.tvalues[1]),
                        "p_value": float(model.pvalues[1]),
                        "r_squared": float(model.rsquared),
                        "n_obs": len(frame),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# 3. stop sensitivity
# --------------------------------------------------------------------- #

def stop_sensitivity(pairs: List[str]) -> pd.DataFrame:
    """
    Are the fixed percentage stops destroying the edge?

    A mean-reverting spread with a 50-70 day half-life routinely moves several
    percent AGAINST the position before reverting -- that is what mean
    reversion is. A fixed 3% stop therefore exits at systematically the worst
    moment, and the design's half-life-aware holding period never gets used.

    Compares the shipped configuration against volatility-scaled stops and
    against no stops at all, on the TRAINING window only.

    Reading this table: if "stops off" dominates everywhere, the risk overlay
    is in structural conflict with the strategy's thesis, which is a defect
    rather than a parameter choice. It is still NOT licence to quote the best
    row as a result.
    """
    rows = []
    for key in pairs:
        base_cfg, pipe, bundle = _fit_once(key)
        tv = base_cfg.train_val
        spread_df = _require_spread_df(pipe)
        cleaned = _require_cleaned_data(pipe)
        artifacts = pipe.compute_artifacts(spread_df, bundle)

        # Daily standard deviation of the position return, per unit gross notional.
        daily_sd = float(
            spread_df["spread"].loc[tv.train_start : tv.train_end].diff().std()
            / (1 + bundle.hedge_ratio)
        )

        configurations = (
            ("as-shipped 3%", 0.03, 0.015, 0.06),
            ("2 daily sigma", 2 * daily_sd, 1.5 * daily_sd, 4 * daily_sd),
            ("4 daily sigma", 4 * daily_sd, 3.0 * daily_sd, 8 * daily_sd),
            ("stops off", 9.99, 9.99, 9.99),
        )

        for label, stop, trail, target in configurations:
            cfg = Config().for_pair(key)
            cfg.backtest.stop_loss_pct = stop
            cfg.backtest.trailing_stop_pct = trail
            cfg.backtest.profit_target_pct = target

            probe = PairPipeline(cfg, verbose=False)
            probe.loader, probe.spread_df, probe.cleaned_data = pipe.loader, spread_df, cleaned

            try:
                result = probe.evaluate_period(
                    spread_df, cleaned, artifacts, bundle,
                    tv.train_start, tv.train_end, label, light=True,
                )
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                rows.append({"pair": key, "config": label, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not result:
                continue

            metrics = result["metrics"]
            stopped = sum(
                v["count"] for k, v in result["exit_reasons"].items() if "stop" in k
            )
            rows.append(
                {
                    "pair": key,
                    "config": label,
                    "daily_sd_pct": 100 * daily_sd,
                    "trades": metrics["total_trades"],
                    "sharpe": metrics["sharpe_ratio"],
                    "ann_return_pct": metrics["annualized_return_pct"],
                    "avg_duration_days": metrics["avg_trade_duration_days"],
                    "pct_exits_via_stop": 100 * stopped / max(metrics["total_trades"], 1),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #

def main(argv=None) -> int:
    description = (__doc__ or "Diagnostics").strip().splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--pair", default="all")
    parser.add_argument(
        "--which", default="all", choices=["all", "ceiling", "predictability", "stops"]
    )
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "outputs" / "diagnostics"),
        help="directory for saved diagnostic CSV and JSON artifacts",
    )
    args = parser.parse_args(argv)

    pairs = sorted(PAIRS) if args.pair == "all" else [args.pair]
    unknown = sorted(set(pairs) - set(PAIRS))
    if unknown:
        parser.error(f"Unknown pair(s): {', '.join(unknown)}")

    output_root = Path(args.out)
    writers = {key: ResultsWriter(output_root / key) for key in pairs}
    for key, writer in writers.items():
        writer.write_json(
            {
                "pair": key,
                "analysis_requested": args.which,
                "command": f"diagnostics.py --pair {args.pair} --which {args.which}",
                "config": Config().for_pair(key).to_dict(),
                "interpretation": (
                    "Diagnostics describe structural properties and are not tuned "
                    "out-of-sample trading results."
                ),
            },
            "diagnostics_metadata.json",
        )
    pd.set_option("display.width", 220)

    if args.which in ("all", "ceiling"):
        print("=" * 78)
        print("1. THEORETICAL CEILING  (idealised: perfect parameters, no costs)")
        print("=" * 78)
        frame = ceiling_analysis(pairs)
        print(frame.round(3).to_string(index=False))
        for key, writer in writers.items():
            writer.write_frame(frame[frame["pair"] == key], "ceiling_analysis.csv", index=False)
        print(
            "\n  'independent_trips_per_yr' is the binding constraint: at ~2 per year,\n"
            "  a two-year window carries roughly four effective observations."
        )

    if args.which in ("all", "predictability"):
        print("\n" + "=" * 78)
        print("2. PREDICTABILITY  (spread[t+h] - spread[t] ~ z[t], HAC errors)")
        print("   Mean reversion => NEGATIVE beta")
        print("=" * 78)
        frame = predictability_test(pairs)
        for key, writer in writers.items():
            writer.write_frame(
                frame[frame["pair"] == key], "predictability_test.csv", index=False
            )
        if frame.empty:
            print("No windows contained enough observations for the predictability test.")
        else:
            for h in sorted(frame["horizon_days"].unique()):
                print(f"\n--- horizon {h} trading days ---")
                print(
                    frame[frame.horizon_days == h]
                    .pivot_table(index="pair", columns="window", values=["beta", "t_stat", "r_squared"])
                    .round(4)
                    .to_string()
                )

    if args.which in ("all", "stops"):
        print("\n" + "=" * 78)
        print("3. STOP SENSITIVITY  (TRAINING window -- diagnostic, NOT a result)")
        print("=" * 78)
        frame = stop_sensitivity(pairs)
        for key in pairs:
            sub = frame[frame["pair"] == key]
            writers[key].write_frame(sub, "stop_sensitivity.csv", index=False)
            if len(sub):
                print()
                print(sub.round(3).to_string(index=False))

    for writer in writers.values():
        writer.manifest()
    print(f"\nSaved diagnostics under {output_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
