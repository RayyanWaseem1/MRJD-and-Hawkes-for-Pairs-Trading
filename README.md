# MRJD and Hawkes Processes for Pairs Trading

This project investigates whether Hawkes-process jump intensity can improve a
mean-reverting pairs trading strategy. The core model combines a
Mean-Reverting Jump Diffusion (MRJD) spread process with a self-exciting Hawkes
process for jump arrivals, then evaluates the resulting signals across ETF,
energy, finance, technology, and gold-related pairs.

The current repository contains two major extensions beyond the original
single-sample backtest:

- A proper train/validation pipeline with models fit only on the training
  period and evaluated out of sample with frozen parameters.
- A quarterly walk-forward engine that refits models through each quarter-end,
  trades the next quarter, and stitches those out-of-sample periods into one
  continuous performance record.

The headline conclusion is unchanged but better supported: Hawkes intensity is
useful for describing jump clustering, but the strategy does not produce
statistically reliable positive alpha after accounting for realistic benchmark
evaluation and transaction costs.

## Research Question

The central question is:

> Can a pairs trading strategy that incorporates Hawkes-process jump intensity
> dynamics generate superior risk-adjusted returns compared with a simpler
> mean-reversion approach?

The initial hypothesis was that jump arrivals in pair spreads are
self-exciting. If a spread jump raises the probability of more jumps, then
Hawkes intensity might help avoid entering during active volatility cascades
and improve entry or exit timing.

The updated pipeline tests that hypothesis more rigorously by separating model
fitting from evaluation, freezing train-calibrated parameters, and measuring
strategy alpha against SPY using Newey-West adjusted t-statistics.

## Repository Layout

```text
.
|-- main.py                         # Train/validation pipeline and model bundle logic
|-- walk_forward.py                 # Quarterly walk-forward OOS engine
|-- config.py                       # Updated configuration with relaxed filters and low-cost presets
|-- config_old.py                   # V2 configuration used by the current V4 main/walk-forward scripts
|-- signal_generation.py            # Relaxed-filter signal generator variant
|-- signal_generation_old.py        # V4 signal generator used by current main.py imports
|-- backtest_engine.py              # Dollar-neutral backtester and alpha evaluator
|-- hawkes_calibration.py           # Hawkes MLE calibration
|-- mrjd_estimation.py              # MRJD estimation
|-- jump_detector.py                # Bipower variation jump detector
|-- equity_pairs_loader.py          # Data loading and spread construction
|-- outputs/<PAIR>/train_val/       # Train/validation CSV artifacts
|-- outputs/<PAIR>/walk_forward/    # Quarterly walk-forward CSV artifacts
|-- ETF Outputs/                    # Terminal logs for SPY/IVV runs
|-- Energy Outputs/                 # Terminal logs for CVX/XOM runs
|-- Finance Outputs/                # Terminal logs for GS/MS runs
|-- Tech Outputs/                   # Terminal logs for AMD/NVDA runs
`-- Gold Outputs/                   # Terminal logs for GLD/GDX runs
```

The canonical structured results are in `outputs/`. The sector folders contain
captured terminal output from the same experiments.

## Pairs Tested

| Segment | Pair | Role in the experiment |
|---|---:|---|
| ETF | SPY/IVV | Economically near-identical S&P 500 ETF control pair. |
| Energy | CVX/XOM | Integrated oil majors with common commodity exposure. |
| Finance | GS/MS | Investment banks with common capital markets exposure. |
| Technology | AMD/NVDA | Semiconductor pair with strong sector linkage and structural AI-cycle effects. |
| Gold | GLD/GDX | Gold ETF versus gold miners ETF, economically related but not identical. |

The CSV data span roughly May 2018 through February 2026, with the exact end
date depending on the pair.

## Model Framework

### Spread Construction

For assets A and B, the log spread is constructed as:

```text
S_t = log(P_A,t) - h log(P_B,t)
```

where `h` is the hedge ratio estimated through the cointegration-based spread
construction in `equity_pairs_loader.py`.

Pair validation now includes:

- Augmented Dickey-Fuller stationarity test on the spread.
- Empirical half-life test.
- Rolling mean stability.
- Spread range in standard-deviation units.
- Recent versus historical mean shift.

All five current pairs pass the train-period tradeability check.

### Jump Detection

Spread returns are tested for jumps using the bipower variation procedure in
`jump_detector.py`. The current configuration uses:

- `window_size = 20`
- `significance_level = 0.05`
- `min_jump_size = 0.01`

The significance level was relaxed from the earlier 1% setting to 5% to provide
more jump observations for Hawkes calibration.

### Hawkes Process

Jump arrivals are modeled as an exponential-kernel Hawkes process:

```text
lambda(t) = lambda_bar + alpha * sum(exp(-beta_H * (t - t_i)))
```

The branching ratio is:

```text
branching_ratio = alpha / beta_H
```

Values below one indicate a stationary, subcritical process. The updated
results still show meaningful jump clustering in several pairs, especially
SPY/IVV and AMD/NVDA, but that clustering does not translate into positive
alpha.

### MRJD Spread Dynamics

The spread is modeled as:

```text
dS_t = kappa(theta - S_t)dt + sigma dW_t + J_t dN_t
```

The implementation estimates MRJD parameters from the training spread and jump
indicator. A practical correction was added for cases where MRJD-implied
half-life disagrees materially with the validated empirical half-life. In that
case, `kappa` is overwritten with:

```text
kappa = log(2) / empirical_half_life
```

This prevents unstable or unrealistically fast MRJD mean-reversion estimates
from driving the trading horizon.

## Train/Validation Pipeline

The train/validation pipeline in `main.py` uses:

| Window | Dates |
|---|---:|
| Training | 2018-05-01 to 2022-12-31 |
| Validation | 2023-01-01 to 2024-12-31 |

The updated workflow is:

1. Load the full pair dataset.
2. Fit pair validation, jumps, Hawkes parameters, Hawkes suitability, and MRJD
   parameters on the training slice only.
3. Store those fitted values in a frozen `ModelBundle`.
4. Compute causal full-sample artifacts using the frozen bundle.
5. Evaluate the training period and validation period separately.
6. Save train/validation metrics, equity curves, trades, signals, pair
   validation results, Hawkes suitability, and the model bundle.

The frozen model bundle includes:

- Empirical spread half-life.
- Hawkes parameters: `lambda_bar`, `alpha`, `beta_H`.
- MRJD parameters: `kappa`, `theta`, `sigma`, `jump_mean`, `jump_std`.
- Training lambda percentiles: `p25`, `p50`, `p75`, `p90`, `p95`.
- Pair validation diagnostics.
- Hawkes suitability score and `use_hawkes_regimes` decision.

Freezing the lambda percentiles is important because the validation period
classifies regimes using the training intensity distribution rather than
future validation information.

## Walk-Forward Pipeline

The walk-forward engine in `walk_forward.py` performs quarterly out-of-sample
testing:

1. Use at least 504 observations, roughly two trading years, before the first
   evaluation.
2. At each quarter-end, refit the pair validation, jump detector, Hawkes model,
   Hawkes suitability score, MRJD parameters, and lambda percentiles using only
   data available through that date.
3. Freeze the fitted model bundle.
4. Trade the next quarter using the frozen parameters.
5. Stitch all quarterly equity curves into a continuous out-of-sample curve.

Each pair currently has 23 quarterly evaluation windows, beginning around
2020-07-01 and ending in February 2026.

## Signal Generation Changes

The signal generation layer now supports configurable Hawkes filtering:

- Empirical rolling z-score with `z_score_lookback = 60`.
- Base entry threshold `z_entry_threshold = 2.0`.
- Base exit threshold `z_exit_threshold = 0.5`.
- Half-life-aware minimum, target, and maximum holding periods.
- Regime-adjusted z-score thresholds.
- Jump-assisted entries at a reduced threshold.
- Dynamic position sizing based on z-score strength, lambda intensity, and
  regime.
- Frozen lambda percentiles via `set_lambda_percentiles()`.

Hawkes regimes are classified from the train-frozen lambda percentiles:

| Regime | Rule |
|---|---|
| Calm | `lambda < p25` |
| Normal | `p25 <= lambda < p75` |
| Elevated | `p75 <= lambda < p90` |
| Crisis | `lambda >= p90` |

The saved runs show no current entries blocked by crisis regimes or lambda
decay, partly because low-jump pairs are adaptively relaxed and partly because
the evaluated windows did not trigger those blockers. The newer `config.py`
also makes this behavior explicit through relaxed filtering defaults:

- `min_lambda_decay_pct = 0.0`, disabling the original 15% lambda-decay entry
  requirement.
- `skip_decay_in_calm = True`.
- `adaptive_for_low_jumps = True`.
- `disable_crisis_block = True`.

This change was made because the original aggressive Hawkes filters eliminated
too many trades and made the evaluation underpowered. Regimes are still used
for threshold adjustment and position sizing, but they no longer have to block
most entries when the relaxed configuration is used.

## Backtesting and Evaluators

The backtest engine uses:

- Initial capital: `$1,000,000`.
- Dollar-neutral positions with equal dollars per leg.
- Transaction costs from the active config. The older V4 config uses roughly
  6bp round trip, while `config.py` adds low-cost presets including roughly
  2bp round trip.
- Maximum position size: 25% of capital.
- Hard stop: 3%.
- Trailing stop: 1.5%, activated after a 1% favorable move.
- Profit target: 6%.
- Risk-free rate: 2%.
- Benchmark: SPY via `OHLCV_SPY.csv`.

The updated evaluator set includes:

- Total and annualized return.
- Annualized volatility.
- Sharpe ratio.
- Sortino ratio.
- Maximum drawdown.
- Calmar ratio.
- Total trades and win rate.
- Average win/loss.
- Profit factor.
- Expected value per trade.
- Average trade duration.
- Average maximum adverse and favorable excursion.
- CAPM alpha versus SPY.
- Alpha t-statistic.
- Alpha p-value.
- Beta, beta t-statistic, R-squared.
- Information ratio.

The alpha evaluator regresses strategy excess daily returns on SPY excess daily
returns and uses Newey-West HAC standard errors with `maxlags = 5`. This makes
the alpha t-stat more appropriate for daily strategy returns that may be
autocorrelated.

## Configuration Updates

The current configuration work added:

- Explicit train/validation windows.
- SPY benchmark CSV path for alpha evaluation.
- Lower-cost presets in `config.py`, including a 2bp round-trip institutional
  cost setting.
- Original-cost comparison preset, approximately 6bp round trip.
- Relaxed Hawkes filtering defaults.
- Adaptive low-jump handling.
- Wider MRJD sigma bounds.
- Empirical z-score as the default signal statistic.
- Half-life-aware holding periods.

Note: `main.py` and `walk_forward.py` currently import `ConfigV2` from
`config_old.py`, while `config.py` contains the newer low-cost and relaxed
filtering configuration. The saved result CSVs in `outputs/` are the source of
truth for the tables below.

## Pair Validation and Hawkes Suitability

These values are from `outputs/<PAIR>/train_val/`.

| Pair | ADF p-value | Half-life | Jump freq. | Branching ratio | Lambda CV | Hawkes score | Use Hawkes regimes |
|---|---:|---:|---:|---:|---:|---:|---|
| SPY/IVV | 2.34e-11 | 21.5d | 27.49% | 0.836 | 1.17 | 0.93 | Yes |
| CVX/XOM | 1.27e-10 | 50.7d | 3.74% | 0.683 | 2.76 | 0.80 | Yes |
| GS/MS | 1.41e-05 | 56.5d | 4.60% | 0.817 | 3.03 | 0.80 | Yes |
| AMD/NVDA | 9.87e-08 | 43.8d | 20.10% | 0.849 | 1.51 | 0.95 | Yes |
| GLD/GDX | 4.31e-04 | 44.4d | 0.26% | 0.100 | 0.00 | 0.45 | No |

Interpretation:

- SPY/IVV and AMD/NVDA show the strongest Hawkes suitability scores, with high
  jump frequency and strong self-excitation.
- CVX/XOM and GS/MS have low jump frequency but enough intensity variation to
  retain Hawkes regime logic in the current scoring system.
- GLD/GDX has almost no jump activity in the training period, so Hawkes regimes
  are disabled and the strategy effectively becomes a simpler z-score
  mean-reversion strategy.

## Train/Validation Results

Values are from `outputs/<PAIR>/train_val/performance_metrics.csv`. Alpha is
annualized versus SPY.

| Pair | Sample | Total Return | Ann. Return | Sharpe | Sortino | Max DD | Trades | Win Rate | Profit Factor | Alpha | Alpha t-stat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY/IVV | Train | -0.15% | -0.03% | -18.252 | -13.927 | -0.37% | 23 | 39.13% | 0.659 | -2.03% | -59.007 |
| SPY/IVV | Validation | -0.02% | -0.01% | -17.615 | -13.155 | -0.15% | 12 | 50.00% | 0.665 | -2.01% | -37.096 |
| CVX/XOM | Train | 2.06% | 0.44% | -1.727 | -1.477 | -1.16% | 18 | 50.00% | 2.572 | -1.57% | -3.993 |
| CVX/XOM | Validation | 2.20% | 1.10% | -0.951 | -0.796 | -0.87% | 5 | 60.00% | 7.293 | -0.84% | -1.341 |
| GS/MS | Train | -1.51% | -0.33% | -2.311 | -1.801 | -2.39% | 16 | 37.50% | 0.596 | -2.31% | -5.504 |
| GS/MS | Validation | -0.02% | -0.01% | -2.153 | -1.703 | -1.18% | 5 | 80.00% | 1.246 | -2.05% | -3.453 |
| AMD/NVDA | Train | 3.34% | 0.71% | -0.721 | -0.390 | -2.05% | 19 | 57.89% | 1.634 | -1.29% | -1.569 |
| AMD/NVDA | Validation | -0.75% | -0.38% | -1.587 | -0.645 | -1.72% | 7 | 42.86% | 0.617 | -2.26% | -2.281 |
| GLD/GDX | Train | -0.39% | -0.08% | -4.287 | -2.809 | -0.96% | 15 | 40.00% | 0.720 | -2.08% | -9.225 |
| GLD/GDX | Validation | -0.65% | -0.33% | -5.472 | -3.299 | -1.09% | 6 | 16.67% | 0.303 | -2.27% | -7.031 |

The validation results are mixed on raw returns but weak on risk-adjusted and
benchmark-adjusted performance. CVX/XOM is the strongest validation case, with
2.20% total return and a 60% win rate, but its alpha remains negative and
statistically insignificant at the 5% level. AMD/NVDA has the best in-sample
return but fails out of sample, which is consistent with overfitting or regime
instability in the semiconductor spread.

## Walk-Forward Results

Values are from `outputs/<PAIR>/walk_forward/walk_forward_metrics.csv`.

| Pair | Total Return | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD | Trades | Win Rate | Alpha | Alpha t-stat | Info Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY/IVV | 0.04% | 0.01% | 0.09% | -21.753 | -16.644 | -0.10% | 19 | 52.63% | -1.99% | -75.853 | -21.760 |
| CVX/XOM | 2.65% | 0.47% | 0.88% | -1.747 | -1.416 | -1.02% | 17 | 58.82% | -1.55% | -4.698 | -1.769 |
| GS/MS | 0.47% | 0.08% | 0.99% | -1.933 | -1.483 | -1.60% | 14 | 50.00% | -1.90% | -4.574 | -1.921 |
| AMD/NVDA | -2.50% | -0.45% | 1.94% | -1.254 | -0.499 | -3.26% | 19 | 36.84% | -2.42% | -3.197 | -1.247 |
| GLD/GDX | 1.13% | 0.20% | 0.50% | -3.574 | -2.522 | -1.02% | 16 | 62.50% | -1.77% | -8.722 | -3.527 |

Walk-forward testing confirms the central result. Some pairs produce positive
total returns, especially CVX/XOM and GLD/GDX, but none produce positive alpha
versus SPY. All walk-forward Sharpe ratios are negative because the strategy's
returns are too small relative to the 2% risk-free rate and its realized
volatility.

## Per-Pair Interpretation

### SPY/IVV

SPY/IVV has excellent Hawkes suitability because tracking-error jumps cluster
strongly. However, the economics are too tight. The pair is almost perfectly
arbitraged, so the strategy generates many small trades with negligible raw
return and deeply negative benchmark-adjusted alpha.

### CVX/XOM

CVX/XOM is the best overall pair in the updated results. It has positive
train, validation, and walk-forward total returns, and the validation period
has a strong profit factor. Even so, the walk-forward alpha is -1.55%
annualized with a t-stat of -4.698. The pair may contain tradable
mean-reversion episodes, but the realized edge is not large enough to beat the
benchmark or cash hurdle consistently.

### GS/MS

GS/MS passes pair validation and shows useful Hawkes diagnostics, but trading
performance remains weak. The validation win rate is high at 80%, but only
five validation trades were generated, so the sample is small. Walk-forward
results are close to flat in raw return and negative in alpha.

### AMD/NVDA

AMD/NVDA has strong jump clustering and the highest Hawkes suitability score,
but it is also the clearest example of regime instability. The model earns
3.34% in sample, then loses -0.75% in validation and -2.50% in walk-forward.
The semiconductor relationship is economically real, but the spread has been
affected by structural changes in NVIDIA's AI-driven growth.

### GLD/GDX

GLD/GDX has marginal Hawkes suitability because the training period contains
almost no detected jumps. Hawkes regimes are disabled for this pair. The
walk-forward raw return is positive, but risk-adjusted and alpha statistics
remain poor.

## Main Findings

1. Hawkes processes capture jump clustering, but clustering is not the same as
   directional edge.
2. Frozen train-calibrated parameters materially improve the validity of the
   experiment by reducing look-ahead bias.
3. The original aggressive lambda-decay and crisis filters were too
   restrictive, so the updated configuration supports relaxed filtering and
   adaptive low-jump behavior.
4. The strategy can generate positive raw returns in some pairs, especially
   CVX/XOM, but the returns are too small relative to the risk-free rate and
   SPY benchmark.
5. Alpha t-statistics are negative across the walk-forward tests.
6. The best use of Hawkes intensity may be risk management or volatility
   forecasting rather than standalone trade entry timing.

## How To Run

Run the train/validation pipeline for the configured pair:

```bash
python main.py
```

Run the quarterly walk-forward pipeline for the configured pair:

```bash
python walk_forward.py
```

To run a different pair, update the symbols and CSV paths in the configuration
object before calling the pipeline, then save results under the matching
`outputs/<PAIR>/` folder.

## Output Files

Each `outputs/<PAIR>/train_val/` directory contains:

- `performance_metrics.csv`
- `train_equity_curve.csv`
- `validation_equity_curve.csv`
- `train_trading_signals.csv`
- `validation_trading_signals.csv`
- `train_trade_summary.csv`
- `validation_trade_summary.csv`
- `pair_validation.csv`
- `hawkes_suitability.csv`
- `model_bundle.csv`
- `causal_artifacts.csv`
- `spread.csv`

Each `outputs/<PAIR>/walk_forward/` directory contains:

- `walk_forward_metrics.csv`
- `walk_forward_equity_curve.csv`
- `quarterly_metrics.csv`
- `quarterly_equity_curves.csv`
- `quarterly_trading_signals.csv`
- `quarterly_trade_summary.csv`
- `quarterly_model_bundles.csv`

## Conclusion

The expanded results make the negative finding more credible. The Hawkes model
does identify self-exciting jump behavior in several daily equity pair spreads,
but this information does not create reliable positive alpha when evaluated
out of sample. The model is statistically interesting and potentially useful
for regime awareness, but it is not sufficient as a standalone source of
profitable daily pairs-trading signals.

Future work should treat Hawkes intensity as a risk or sizing input, combine it
with independent directional predictors, or move to markets and frequencies
where jump timing contains more exploitable information.
