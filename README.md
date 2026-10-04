# Self-Exciting Jumps in Equity Pair Spreads: A Controlled Test of Hawkes-Driven Mean-Reverting Jump Diffusions for Statistical Arbitrage

> **Abstract.** We study whether the arrival of discontinuities ("jumps") in cointegrated equity pair spreads is *self-exciting*, and whether conditioning a mean-reversion strategy on the resulting jump intensity improves risk-adjusted performance. Each spread is modelled as a Mean-Reverting Jump Diffusion (MRJD) whose jump-counting process is a univariate Hawkes process with an exponential kernel. Jumps are dated with the Lee–Mykland (2008) per-observation test under Benjamini–Hochberg false-discovery control. The Hawkes process is estimated by exact maximum likelihood in an unconstrained reparameterisation, and self-excitation is tested against a homogeneous Poisson null with a parametric-bootstrap likelihood-ratio test, which handles the non-identification of the decay parameter under the null (the Davies problem). The trading layer maps the fitted intensity into volatility regimes that modulate entry thresholds, holding periods and position size. It is evaluated against a **matched control arm**: the identical strategy with the Hawkes layer removed. Evaluation uses a frozen-parameter train/validation split and a 23-quarter **continuous-book walk-forward** (July 2020 – February 2026) with in-loop threshold tuning, Newey–West inference on mean excess returns, Lo (2002) Sharpe standard errors, stationary-bootstrap intervals, the Deflated Sharpe Ratio, and explicit power analysis.
>
> **Findings.** (i) Once jumps are dated correctly and controlled for multiplicity, daily equity spreads yield only 0–13 jumps per pair over 4.7 training years. Self-excitation is not established on any pair: every branching-ratio 95% CI contains zero, and three of five pairs collapse to a Poisson process. (ii) The Hawkes arm does not outperform its control on any pair. A paired Newey–West test on daily return differences gives a pooled +0.23%/yr, *t* = 0.51. (iii) Neither arm earns statistically significant excess return over cash out of sample. The only significant result is negative: SPY/IVV loses 1.1–1.2%/yr to transaction costs (*t* ≈ −7). (iv) The design is underpowered. The minimum detectable effect at 80% power is 1.5–6.3%/yr per pair, so the null results should be read as *"the design cannot resolve an edge of the size sought"*, not as proof that no edge exists. The main methodological finding is a catalogue of failure modes that, left uncorrected, **manufacture** apparent self-excitation and spurious alpha. All of them were present in an earlier version of this study.

---

## Table of Contents

1. [Research Question and Hypotheses](#1-research-question-and-hypotheses)
2. [Intuition](#2-intuition)
3. [Contributions and Positioning](#3-contributions-and-positioning)
4. [Data](#4-data)
5. [Theoretical Framework](#5-theoretical-framework)
   - 5.1 [Spread construction and cointegration](#51-spread-construction-and-cointegration)
   - 5.2 [Mean-reverting jump diffusion](#52-mean-reverting-jump-diffusion-mrjd)
   - 5.3 [Jump detection](#53-jump-detection)
   - 5.4 [Hawkes process: estimation and inference](#54-hawkes-process-estimation-and-inference)
   - 5.5 [From intensity to trading decisions](#55-from-intensity-to-trading-decisions)
6. [Strategy and Execution Model](#6-strategy-and-execution-model)
7. [Experimental Design](#7-experimental-design)
8. [Statistical Evaluation](#8-statistical-evaluation)
9. [Results](#9-results)
10. [Discussion: Why There Is No Effect](#10-discussion-why-there-is-no-effect)
11. [Pitfalls, Limitations and Threats to Validity](#11-pitfalls-limitations-and-threats-to-validity)
12. [Implications and Future Work](#12-implications-and-future-work)
13. [Reproducibility](#13-reproducibility)
14. [Repository Structure](#14-repository-structure)
15. [References](#15-references)

---

## 1. Research Question and Hypotheses

> *Can a pairs-trading strategy that conditions on Hawkes-process jump intensity generate superior risk-adjusted returns relative to an otherwise identical mean-reversion strategy?*

The question splits into a **statistical** hypothesis about the data-generating process and an **economic** hypothesis about trading value. The economic claim only makes sense if the statistical one holds.

| | Null | Alternative | Test |
|---|---|---|---|
| **H1** (self-excitation) | Spread jump arrivals are homogeneous Poisson: $\alpha = 0$ | Hawkes with $\alpha > 0$, i.e. branching ratio $\eta = \alpha/\beta > 0$ | Likelihood ratio vs. Poisson with parametric-bootstrap null; Hessian CI on $\eta$; time-rescaling KS goodness-of-fit |
| **H2** (incremental value) | $\mathbb{E}[r^{\text{Hawkes}}_t - r^{\text{Control}}_t] = 0$ | Strictly positive | Paired Newey–West test on daily return differences; arm-by-arm comparison of Sharpe with HAC standard errors |
| **H3** (absolute value) | $\mathbb{E}[r_t - r_f] = 0$ for a dollar-neutral book | Strictly positive | Newey–West HAC *t*-test on daily excess returns; Deflated Sharpe Ratio for the threshold search |

H3 is reported for completeness. H2 is the question the project title asks: does the Hawkes layer add anything?

---

## 2. Intuition

A classical pairs trade bets that a stationary spread returns to its mean. The Ornstein–Uhlenbeck (OU) model behind it assumes continuous Gaussian innovations. Real spreads also contain **discontinuities**: earnings surprises, idiosyncratic news, index events, liquidity dislocations, ETF creation/redemption frictions. These jumps matter to a mean-reversion trader in three ways:

1. **They contaminate the signal.** A jump moves the z-score across an entry threshold without any "temporary mispricing" behind it. The spread has been displaced, not stretched.
2. **They contaminate the estimates.** Diffusion parameters fitted on a sample that contains jumps overstate $\sigma$ and understate mean-reversion speed.
3. **They may cluster.** If one jump raises the short-run probability of another, as is well documented for index returns and order flow, then entering a fade straight after a jump is "catching a falling knife". The better trade waits for the cascade to decay.

The Hawkes process is the canonical model for (3). Its conditional intensity $\lambda(t)$ rises by $\alpha$ at every event and decays at rate $\beta$. A fitted $\lambda(t)$ therefore gives a real-time, causal estimate of how "hot" the jump environment is. That leads to a natural trading hypothesis: **tighten or suspend entries when $\lambda(t)$ is elevated, relax them when it is at baseline, and size positions inversely to excess intensity**. The MRJD supplies the other half, a jump-robust estimate of mean-reversion speed and hence of the holding horizon.

The intuition is appealing, but it rests on an empirical premise: that there are enough jumps, and enough clustering among them, to estimate. Much of this study is about whether that premise holds at daily frequency.

---

## 3. Contributions and Positioning

Self-exciting jump processes are well established in finance (Hawkes 1971; Aït-Sahalia, Cacho-Diaz & Laeven 2015; Bacry, Mastromatteo & Muzy 2015). OU-based pairs trading is equally standard (Elliott, van der Hoek & Malcolm 2005; Gatev, Goetzmann & Rouwenhorst 2006; Avellaneda & Lee 2010; Bertram 2010). The project does not claim a new model class. Its contributions are:

1. **A joint MRJD–Hawkes specification for equity pair spreads, evaluated as a controlled experiment.** Every result is reported for a *treatment* arm (Hawkes layer on) and a *control* arm (Hawkes layer off) that share data, folds, costs, execution, thresholds search and seed. That makes "does the Hawkes layer add value?" a measurable quantity rather than an anecdote.
2. **A correct event-dating pipeline for point-process estimation on spreads.** Jumps are dated per observation (Lee–Mykland), FDR-controlled, and placed on a trading-day clock before the Hawkes likelihood ever sees them. Section 11.1 shows that a common alternative, a rolling-window bipower test whose rejection is attributed to the window's last day, **mechanically fabricates** self-excitation.
3. **Valid inference for self-excitation.** The Hawkes MLE is reparameterised so that stationarity holds by construction and the optimum is interior. The LR test against Poisson uses a parametric-bootstrap null because the decay parameter is unidentified under $H_0$, so the usual $\chi^2$ reference is invalid. Goodness of fit uses the time-rescaling theorem.
4. **An evaluation protocol designed against the usual backtest pathologies:** frozen training bundles, a continuous (not restarted-and-stitched) walk-forward book, threshold tuning inside the walk-forward loop with the number of trials recorded for the Deflated Sharpe Ratio, idle-cash credit so that a flat book does not "underperform" by exactly $r_f$, next-open execution, and gap-aware intraday stops.
5. **Explicit power accounting.** Each null result comes with its standard error and minimum detectable effect, and a synthetic study (`intraday.py`) quantifies how many events the Hawkes layer actually needs.
6. **A documented audit trail** of fifteen methodological errors from an earlier version of this study, each with its quantitative consequence and its fix (Section 11.1). For practitioners this is arguably the most transferable output.

---

## 4. Data

| Item | Detail |
|---|---|
| Source | Databento raw daily OHLCV bars (`ts_event`, open, high, low, close, volume); **unadjusted** |
| Universe | 10 US-listed symbols: AMD, CVX, GDX, GLD, GS, IVV, MS, NVDA, SPY, XOM |
| Span | 2018-05-01 → 2026-02-12/19 (1,959–1,963 bars per symbol) |
| Corporate actions | NVDA 4:1 (2021-07-20) and 10:1 (2024-06-10) splits back-adjusted from a hard-coded table. Each factor is **re-derived from the raw price jump and asserted** (`corporate_actions.verify_split_table`). Every other symbol is scanned and verified to have no single-day move above 40%. |
| Dividends | **Excluded** from both spread and P&L (disclosed; see Section 11.3). Indicative annual differentials: CVX/XOM ≈ 0.9%, GS/MS ≈ 0.8%, GLD/GDX ≈ 1.3%, SPY/IVV ≈ 0, AMD/NVDA ≈ 0. |
| Benchmark | SPY *price* series, used only for a market-neutrality regression |
| Cleaning | Inner join on common dates and drop missing rows. Large moves are **flagged, never deleted**: deleting them would remove the very phenomenon under study and leave irregular gaps in a clock that both the Hawkes compensator and the OU transition density assume is regular. |

**Registered pairs**, each with an ex-ante economic rationale:

| Segment | Pair (A/B) | Economic link | Role |
|---|---|---|---|
| ETF | SPY/IVV | Two S&P 500 trackers | Near-arbitrage control: maximal cointegration, minimal edge |
| Energy | CVX/XOM | Integrated oil majors | Common commodity factor |
| Financials | GS/MS | Investment banks | Common capital-markets factor |
| Semiconductors | AMD/NVDA | Sector peers | Linked but exposed to a structural break (AI cycle) |
| Gold | GLD/GDX | Bullion vs. miners | Related but levered/operationally distinct |

---

## 5. Theoretical Framework

### 5.1 Spread construction and cointegration

For prices $P^A_t, P^B_t$ the log-spread is

```math
S_t \;=\; \log P^A_t \;-\; h\,\log P^B_t ,
```

with hedge ratio $h$ estimated **once on the training window and then frozen** (`hedge_mode="static"`). In walk-forward it is re-estimated at each quarter boundary and held fixed within the quarter.

*Why not a rolling hedge ratio?* If $h_t$ varies, then

```math
\Delta S_t \;=\; \Delta\log P^A_t \;-\; h_t\,\Delta\log P^B_t \;-\; \Delta h_t\,\log P^B_{t-1}.
```

With a 30-day rolling OLS, the third term, an estimation artefact multiplied by a *price level*, accounted for 99.2–99.8% of $\mathbb{E}|\Delta S_t|$ across all five pairs and inflated spread volatility by 4.8× to 81.9×. A daily-moving hedge is also not a position anyone can hold. The rolling mode survives only as a labelled robustness option.

**Stationarity testing.** $S_t$ is a residual from an *estimated* cointegrating vector, so a standard ADF test, whose Dickey–Fuller critical values assume an observed series, over-rejects. Stationarity is therefore judged with the **Engle–Granger** test using Phillips–Ouliaris (1990) residual-based critical values. The naive ADF *p*-value is reported alongside so the size of the over-rejection is visible.

**Pair validation.** Five checks are computed on the training window: EG stationarity ($p<0.05$); AR(1) half-life in $[5, 120]$ trading days; rolling-mean stability ($\operatorname{sd}(\bar S^{(252)}_t)/\operatorname{sd}(S_t) < 0.5$); range below $10\,\operatorname{sd}$; and a recent-vs-full-sample mean shift below $1\,\operatorname{sd}$. All five feed `is_tradeable`. (See Section 11.2: the flag is recorded but does not currently gate trading.)

### 5.2 Mean-reverting jump diffusion (MRJD)

```math
dS_t \;=\; \kappa(\theta - S_t)\,dt \;+\; \sigma\,dW_t \;+\; Y\,dN_t,
\qquad Y \sim \mathcal N(\mu_J,\sigma_J^2),
\qquad N_t \sim \text{Hawkes}(\bar\lambda,\alpha,\beta).
```

**Time unit.** Everything is measured in **trading days**, with $\Delta t = 1$. Using $\Delta t = 1/252$ turns $\kappa$ into a per-year rate while half-lives and holding periods are read in days, a factor-of-252 error that an earlier version of the code had. `time_units.assert_trading_day_units` now makes any reintroduction fail loudly.

**Estimation by exact discretisation.** Between jumps the OU transition is exactly Gaussian:

```math
S_{t+1}\mid S_t \;\sim\; \mathcal N\!\Big(\theta + (S_t-\theta)\,\phi,\;\; s^2(1-\phi^2)\Big),
\qquad \phi = e^{-\kappa\Delta t},\quad s^2 = \frac{\sigma^2}{2\kappa}.
```

This is an AR(1) regression $S_{t+1} = c + \phi S_t + \varepsilon_t$, so the conditional MLE is closed-form OLS. It has no flat likelihood direction and no optimiser that can fail. Parameters are recovered by inversion:

```math
\kappa = -\frac{\ln\phi}{\Delta t},\qquad
\theta = \frac{c}{1-\phi},\qquad
\sigma = s\sqrt{2\kappa},\qquad
t_{1/2} = \frac{\ln 2}{\kappa},\qquad
\operatorname{sd}_\infty(S) = \frac{\sigma}{\sqrt{2\kappa}} .
```

Transitions that *end* on a detected jump are excluded, so $(\kappa,\theta,\sigma)$ describe the continuous component. $(\mu_J,\sigma_J)$ are the sample moments of $\Delta S_t$ on jump days. The estimator raises `MRJDFitError` if $\phi \le 0$ (not mean-reverting) or $\phi \ge 1$ (unit root).

Why reparameterise? A 50-day half-life implies $\phi \approx 0.986$, close to a unit root, where the likelihood in $(\kappa,\sigma)$ is nearly flat. Direct numerical optimisation in the earlier version produced an implied stationary s.d. 15.6× the empirical one (GS/MS), and AMD/NVDA pinned $\sigma$ at its upper bound. An optional joint MLE over $(\phi,\theta,s,\mu_J,\sigma_J)$, using a Gaussian mixture on jump days, is available (`MRJDConfig.joint_refinement`). The model's half-life is compared against the empirical one and reported, **never silently overwritten**.

The MRJD feeds the strategy through the half-life (which sets the holding period) and optionally through a parametric z-score $(S_t-\theta)/\operatorname{sd}_\infty$ and an expected reversion time $t = \ln(z_0/z_1)/\kappa$.

### 5.3 Jump detection

**Primary: Lee–Mykland (2008).** Each daily spread change $r_i = \Delta S_i$ is standardised by a local bipower volatility computed from a **strictly preceding** window of $K = 20$ days:

```math
\mathcal L_i = \frac{|r_i|}{\hat\sigma_i},\qquad
\hat\sigma_i^2 = \frac{1}{K-2}\sum_{j=i-K+2}^{i-1} |r_j|\,|r_{j-1}| .
```

Under the no-jump null the normalised maximum converges to a Gumbel law:

```math
\frac{\max_i \mathcal L_i - C_n}{S_n} \xrightarrow{d} \xi,\quad P(\xi\le x)=e^{-e^{-x}},\qquad
C_n = \frac{\sqrt{2\log n}}{c} - \frac{\log\pi + \log\log n}{2c\sqrt{2\log n}},\quad
S_n = \frac{1}{c\sqrt{2\log n}},\quad c=\sqrt{2/\pi}.
```

This gives a per-observation *p*-value $p_i = 1-\exp\{-e^{-(\mathcal L_i - C_n)/S_n}\}$. A rejection at $i$ means *"a jump occurred at $i$"*, which is exactly the event time a point-process likelihood requires.

**Multiplicity.** Roughly 1,950 tests at a nominal 5% would produce about 98 false positives. Jump flags are therefore Benjamini–Hochberg FDR-controlled at 5% by default ("FDR basis"). When FDR leaves fewer than 10 events, the Hawkes layer is fitted on the Gumbel-critical-value rule ("nominal basis"). This fallback is recorded in every artefact and never applied silently.

**Robustness: Barndorff-Nielsen–Shephard bipower test**, in log-ratio form (Huang & Tauchen 2005):

```math
Z = \frac{\log RV - \log BV}{\sqrt{\frac{\vartheta}{m}\max\!\big(1, TP/BV^2\big)}},\qquad \vartheta = \tfrac{\pi^2}{4}+\pi-5 \approx 0.609 .
```

BNS asymptotics require $\Delta\to 0$, meaning many *intraday* returns per tested period. On daily bars there is no asymptotic regime in which the test is valid, so it serves only as a robustness check (`intraday.Frequency.bipower_is_valid`). Because a rejection means "a jump somewhere in the trailing window", the flag is attributed to the largest $|r|$ inside that window. A naive $4\sigma$ threshold rule is kept for the detector comparison.

### 5.4 Hawkes process: estimation and inference

**Model.** On the trading-day clock, with event times $t_1<\dots<t_n$ in $[0,T]$:

```math
\lambda(t) \;=\; \bar\lambda \;+\; \sum_{t_i < t} \alpha\, e^{-\beta (t-t_i)} .
```

The **branching ratio** $\eta = \alpha/\beta$ is the expected number of direct "offspring" per event. In the cluster representation, each exogenous event spawns a cascade of expected total size $1/(1-\eta)$. The process is stationary iff $\eta<1$, with long-run mean intensity $\bar\lambda/(1-\eta)$. A value $\eta = 0$ is the homogeneous Poisson process.

**Exact log-likelihood** with the $O(n)$ recursion of Ozaki (1979):

```math
\ell(\bar\lambda,\alpha,\beta) = \sum_{i=1}^n \log\!\big(\bar\lambda + \alpha R_i\big) \;-\; \bar\lambda T \;-\; \frac{\alpha}{\beta}\sum_{i=1}^n\big(1-e^{-\beta(T-t_i)}\big),
\qquad R_i = e^{-\beta(t_i-t_{i-1})}(1+R_{i-1}),\; R_1 = 0 .
```

**Unconstrained reparameterisation.** The model is fitted in $u\in\mathbb R^3$ with

```math
\bar\lambda = e^{u_0},\qquad \beta = e^{u_1},\qquad \eta = \operatorname{logistic}(u_2),\qquad \alpha = \eta\beta ,
```

so $0<\eta<1$ holds *by construction*, the objective is smooth everywhere, and L-BFGS-B is run from four starting points. This replaces a penalty that returned the constant $10^{10}$ whenever $\eta > 0.85$. Finite-difference gradients see that as a cliff, and in the earlier version every reported branching ratio landed within 2% of the 0.85 wall, where standard errors are meaningless.

**Inference.**

| Quantity | Method |
|---|---|
| $\operatorname{se}(\bar\lambda,\alpha,\beta)$ | Inverse of the numerical Hessian of $-\ell$ in natural parameters |
| CI on $\eta$ | Delta method: $\nabla\eta = (1/\beta,\, -\alpha/\beta^2)$ |
| $H_0: \alpha = 0$ | $LR = 2(\ell_{\text{Hawkes}} - \ell_{\text{Poisson}})$, with $\ell_{\text{Poisson}} = n\log(n/T) - n$. Under $H_0$ the decay $\beta$ is **unidentified** (Davies 1977, 1987), so $LR \not\sim \chi^2$. The $\chi^2_2$ *p*-value is reported only as a reference; the quoted *p*-value comes from a **parametric bootstrap** (200 Poisson replications refitted with the full Hawkes MLE). |
| Goodness of fit | Time-rescaling theorem (Ogata 1988; Brown et al. 2002): $\tau_i = \Lambda(t_i) - \Lambda(t_{i-1}) \overset{iid}{\sim}\text{Exp}(1)$ under correct specification. Tested by KS and visualised by QQ plot. |
| Intensity calibration | Poisson GLM of next-day jump indicator on $\log\hat\lambda(t)$. A well-calibrated intensity has slope ≈ 1. |
| Estimator validity | Synthetic recovery via Ogata (1981) thinning in the test suite: known parameters recovered, LR rejects on clustered data and does not reject on Poisson data, CI covers truth. |

The intensity used for trading, `compute_intensity_at_times`, sums only over events **strictly before** each evaluation time, so it is causal.

### 5.5 From intensity to trading decisions

**Regimes on relative excess intensity.** Since $\lambda(t)\ge\bar\lambda$ always, percentile bucketing of a spike train is degenerate: the lower quantiles pile onto an atom at $\bar\lambda$, and in an earlier version "calm" was unreachable on two pairs. Regimes are therefore cut on

```math
e_t = \frac{\lambda(t)-\bar\lambda}{\bar\lambda},
```

where $\bar\lambda$ is frozen from the training bundle:

| Regime | Rule | Entry threshold | Exit threshold | Max hold | Size factor |
|---|---|---|---|---|---|
| CALM | $e_t < 0.05$ | $0.85\,z_{\text{in}}$ | $0.85\,z_{\text{out}}$ | $1.2\times$ | 1.2 |
| NORMAL | $0.05 \le e_t < 1$ | $z_{\text{in}}$ | $z_{\text{out}}$ | $1.0\times$ | 1.0 |
| ELEVATED | $1 \le e_t < 5$ | $1.25\,z_{\text{in}}$ | $1.2\,z_{\text{out}}$ | $0.75\times$ | 0.7 |
| CRISIS | $e_t \ge 5$ | $1.5\,z_{\text{in}}$, **entries blocked** | $1.3\,z_{\text{out}}$ | $0.5\times$ | 0.5 |

**Cascade-decay gate.** If $e_t \ge 0.05$, an entry also requires $\lambda(t)$ to have fallen at least 15% from its 5-day peak ("wait for the cascade to subside"). When no cascade is in progress the gate is open; otherwise a flat intensity would block every trade.

**Jump-assisted entries.** On a detected jump day the entry threshold is relaxed to $0.65\times$ the regime threshold and the position is sized down by 20%. These entries pass through the same CRISIS and decay gates.

**Position sizing.**

```math
w_t = \operatorname{clip}\Big(0.25 \cdot \min\!\big(\tfrac{|z_t|}{3},\,1.5\big)\cdot f_\lambda \cdot f_{\text{regime}},\;0.10,\;0.25\Big),
\qquad f_\lambda = \operatorname{clip}\big(1.5 - e_t,\;0.5,\;1.5\big).
```

**Control arm** (`Config.as_control_arm`): `use_hawkes_regimes=False`, `use_jump_entries=False`. Every observation is NORMAL, $f_\lambda = 1$, and there is no decay gate and no jump entries. Everything else is identical.

---

## 6. Strategy and Execution Model

**Signal.** The default is an empirical z-score with a 60-day *strictly lagged* window, $z_t = (S_t - \bar S_{t-60:t-1})/\operatorname{sd}(S_{t-60:t-1})$, so the current bar never enters its own normalisation. Long the spread when $z_t < -z_{\text{in}}$, short when $z_t > z_{\text{in}}$ (base $z_{\text{in}}=2.0$, $z_{\text{out}}=0.5$, tuned in walk-forward).

**Exits.** All conditions are evaluated **independently** each bar and resolved by a fixed priority. An earlier `elif` chain made three of the five exits unreachable, which every committed trade log confirmed.

| Priority | Exit | Condition |
|---|---|---|
| 1 | `emergency_stop` | $z$ moves 2.5 against the entry level |
| 2 | `regime_crisis` | Regime escalates to CRISIS after a non-CRISIS entry |
| 3 | `max_hold` | Holding period ≥ regime-adjusted maximum |
| 4 | `profit_target` | $z$ crosses through the mean past $z_{\text{out}}$ (after min hold) |
| 5 | `mean_reversion` | $|z| < z_{\text{out}}$ (after min hold) |

Holding periods scale with the training half-life: minimum $0.5\,t_{1/2}$, maximum capped at 120 days.

**Backtest engine** (`backtest_engine.py`), event-driven on daily bars:

| Component | Implementation |
|---|---|
| Sizing | Leg A gets $w\cdot\text{cash}/(1+|h|)$ dollars and leg B gets $h$ times that, opposite sign, so the book tracks $\log A - h\log B$. Gross notional is ≈ $w\cdot$cash. |
| Execution | Signals on bar $t$, fills at the **open of $t+1$** (`execution_delay=1`) |
| Costs | Commission + slippage per side on gross notional at entry and exit |
| Financing | Long financing 2%/yr, short rebate 1.5%/yr, per-symbol borrow 20–100 bp/yr, accrued daily |
| Idle cash | Credited at $r_f = 2\%$ on the undeployed share of equity. Without this a dollar-neutral book that is flat about 90% of the time shows a CAPM "alpha" of almost exactly $-r_f$, which is what the earlier version reported. |
| Stops | Volatility-scaled backstops in units of the spread's stationary s.d.: hard stop $4\,\operatorname{sd}$, profit target $5\,\operatorname{sd}$, trailing stop activated at $1.5\,\operatorname{sd}$. All are clamped to $[1\%, 50\%]$ of notional. Fixed 3% stops structurally conflict with mean reversion: in diagnostics they caused 85–100% of trades to exit via stop at 1.7–12 days. |
| Stop fills | Checked against the intraday High/Low. If the open already gaps through the level, the fill is at the open. |
| End of sample | Open positions are force-closed and **recorded** |
| Integrity | Raises if equity ≤ 0. The earlier `max(equity, 1)` denominator floor has been removed. |

---

## 7. Experimental Design

```mermaid
flowchart LR
    A["Raw OHLCV"] --> B["Split adjust + verify"]
    B --> C["Static hedge on train<br/>Engle–Granger"]
    C --> D["Pair validation"]
    C --> E["Lee–Mykland + BH-FDR"]
    E --> F["Hawkes MLE<br/>LR bootstrap, KS, CI"]
    E --> G["MRJD exact AR1 MLE"]
    D & F & G --> H["Frozen ModelBundle"]
    H --> I["Causal artefacts:<br/>intensity, z-score, jump flags"]
    I --> J["Hawkes arm"]
    I --> K["Control arm"]
    J & K --> L["Backtest, metrics, inference"]
```

**7.1 Train / validation.** Train 2018-05-01 → 2022-12-31 (1,177 obs); validation 2023-01-01 → 2024-12-31 (≈ 502 obs). The hedge ratio, pair validation, jump detection, Hawkes and MRJD parameters and $\bar\lambda$ are fitted on train only, frozen into a `ModelBundle`, and applied unchanged to validation.

**7.2 Walk-forward (headline).** After a minimum of 504 training observations, at each calendar quarter-end the procedure (a) re-estimates the hedge ratio on all data to date, (b) refits the full bundle, (c) grid-searches $z_{\text{in}}\in\{1.5,2.0,2.5\}\times z_{\text{out}}\in\{0.25,0.5,0.75\}$ **on the training window only**, maximising Sharpe, and (d) generates signals for the next quarter with frozen parameters. This yields 23 quarters (2020-07-01 → Feb 2026, about 1,414 trading days) and 9 × 23 = **207 configurations tried**, which feeds the Deflated Sharpe Ratio.

The book is **one continuous backtest**: parameters are swapped at quarter boundaries while cash and open positions carry through. The earlier engine restarted each quarter with fresh capital, silently liquidated trades that had not reached their minimum hold, and stitched the curves in a way that zeroed 23 genuine daily returns. It also wrote all-zero trade statistics that looked like measurements. Failed quarters are logged rather than dropped; there were 0 failures in the published runs.

**7.3 Robustness matrix** (pre-specified, one factor at a time; published for CVX/XOM): half and double costs; fixed thresholds (1.5/0.5 and 2.5/0.5, no tuning); bipower detector; static OLS hedge.

**7.4 Cross-pair portfolio.** An equal-weight portfolio of the five walk-forward return streams, per arm, reports effective breadth $N_{\text{eff}} = N/(1+(N-1)\bar\rho)$.

**7.5 Cross-sectional screen** (`pair_screen.py`). All $\binom{10}{2}=45$ pairs are screened on training data against criteria fixed in advance: EG $p<0.05$, half-life in $[5,120]$, predictive $t\le -1.5$ at 20 days, edge ≥ 5× cost, $0<h<5$. Results are reported with BH and Bonferroni corrections across the 45 tests.

**7.6 Diagnostics** (`diagnostics.py`): a theoretical Sharpe ceiling, z-score predictability regressions, and a stop sensitivity analysis.

**7.7 Synthetic power study** (`intraday.py`): how many events are needed to detect a known $\eta=0.5$?

---

## 8. Statistical Evaluation

| Statistic | Definition and rationale |
|---|---|
| **Headline test** | $H_0: \mathbb{E}[r_t - r_f]=0$ with a Newey–West (Bartlett) long-run variance, lag $\lfloor 4(n/100)^{2/9}\rfloor$ (Newey & West 1994). For a dollar-neutral book $\beta\approx 0$ by construction, so a CAPM intercept mostly measures cash accounting. The CAPM regression vs. SPY (HAC, 5 lags) is reported **only to demonstrate neutrality**. |
| Sharpe SE | Lo (2002): $\operatorname{se}(\widehat{SR}) \approx \sqrt{(1+\widehat{SR}^2/2)/n}$ per period, inflated by $\sqrt{\widehat{LRV}/\hat\sigma^2}$ to account for serial dependence from multi-week holding |
| Bootstrap CIs | Politis–Romano (1994) stationary bootstrap, mean block 20 days, 1,000 replications, for Sharpe, mean return and max drawdown |
| Deflated Sharpe | Bailey & López de Prado (2014), with expected maximum Sharpe under no skill after $N$ trials $E[\max SR] \approx \sqrt{V}\big[(1-\gamma)\Phi^{-1}(1-\tfrac1N) + \gamma\Phi^{-1}(1-\tfrac1{Ne})\big]$ |
| Power / MDE | $\text{MDE}_{80\%} = (z_{0.975}+z_{0.80})\,\sigma_{\text{ann}}/\sqrt{\text{years}}$; achieved power against a 1%/yr target |
| Capital-at-risk view | *Excess* return scaled by $1/\overline{\text{gross exposure}}$; the risk-free credit is not levered |
| Paired arm test (H2) | NW *t*-test on $r^{\text{Hawkes}}_t - r^{\text{Control}}_t$, computed from the published equity curves (Section 9.5) |

---

## 9. Results

All figures are read from the artefacts in `outputs/` produced by the current code. The authoritative files are those listed in each directory's `MANIFEST.json` (see Section 13). Returns are annualised percentages unless stated otherwise.

### 9.1 Spread properties and pair validation (training window, 1,177 obs)

| Pair | $h$ | EG *p* | ADF *p* | AR(1) $t_{1/2}$ (d) | Mean drift | Mean shift (sd) | Range (sd) | Tradeable | Failing checks |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| SPY/IVV | 1.004 | **0.009** | 0.002 | **1.9** | 0.44 | 0.18 | 8.96 | ✗ | half-life < 5 d |
| CVX/XOM | 0.720 | 0.266 | 0.109 | 49.5 | 0.63 | 1.20 | 6.10 | ✗ | EG, stable mean, regime shift |
| GS/MS | 0.795 | **0.042** | 0.011 | 46.3 | 0.48 | 0.45 | 4.41 | ✓ | — |
| AMD/NVDA | 0.908 | **0.016** | 0.003 | 69.8 | 0.45 | 0.15 | 5.44 | ✓ | — |
| GLD/GDX | 0.622 | 0.671 | 0.426 | 56.6 | 0.59 | 1.39 | 4.96 | ✗ | EG, stable mean, regime shift |

The ADF–EG gap is substantial (for example 0.109 vs. 0.266 on CVX/XOM), which confirms that naive ADF over-states cointegration. **Only two of the five registered pairs pass validation.** SPY/IVV is cointegrated but reverts with a ~2-day half-life, too fast to trade after costs.

### 9.2 MRJD estimates (training)

| Pair | $\kappa$ (1/day) | Model $t_{1/2}$ | $\sigma$ | $\operatorname{sd}_\infty$ implied / actual | $\mu_J$ | $\sigma_J$ | $|t^{\text{model}}_{1/2}/t^{\text{emp}}_{1/2}-1|$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPY/IVV | 0.4115 | 1.7 | 0.0015 | 0.94 | −0.0012 | 0.0061 | 11% |
| CVX/XOM | 0.0105 | 66.1 | 0.0118 | 1.12 | −0.0040 | 0.0485 | 33% |
| GS/MS | 0.0133 | 52.2 | 0.0092 | 1.00 | −0.0039 | 0.0448 | 13% |
| AMD/NVDA | 0.0100 | 69.4 | 0.0232 | 0.59 | −0.0083 | 0.1195 | 0.5% |
| GLD/GDX | 0.0068 | 102.0 | 0.0092 | 1.30 | −0.0083 | 0.0691 | 80% |

The reparameterised fit reproduces the empirical dispersion within roughly 0.6–1.3× on every pair, whereas the earlier direct optimisation was off by up to 15.6×. The large half-life disagreement on GLD/GDX is consistent with that pair failing the EG test: on a near-unit-root series, $\kappa$ is poorly identified.

### 9.3 Jump detection and the self-excitation test (H1)

| Pair | Jumps FDR / nominal | Basis | $\hat{\bar\lambda}$ | $\hat\alpha$ | $\hat\beta$ | $\hat\eta$ | 95% CI on $\eta$ | LR | $p_{\chi^2_2}$ | $p_{\text{boot}}$ (eff. reps) | KS *p* |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| SPY/IVV | 13 / 25 | FDR | 0.0095 | 0.046 | 0.324 | 0.142 | [−0.123, 0.407] | 2.08 | 0.353 | 0.086 (198) | 0.85 |
| CVX/XOM | 5 / 8 | nominal | 0.0068 | ≈0 | 0.500 | ≈0 | [−0.001, 0.001] | ≈0 | 1.000 | 0.706 (180) | 0.21 |
| GS/MS | 0 / 7 | nominal | 0.0060 | ≈0 | 0.500 | ≈0 | [−0.001, 0.001] | ≈0 | 1.000 | 0.649 (171) | 0.78 |
| AMD/NVDA | 2 / 7 | nominal | 0.0060 | ≈0 | 0.500 | ≈0 | [−0.001, 0.001] | ≈0 | 1.000 | 0.649 (171) | 0.48 |
| GLD/GDX | 0 / 5 | nominal | 0.0027 | 0.052 | 0.146 | 0.355 | [−0.201, 0.911] | 4.83 | 0.090 | 0.008 (122) | 0.46 |

<p align="center">
  <img src="outputs/CVX_XOM/train_val/jump_detection.png" width="85%" alt="CVX/XOM log spread with Lee–Mykland jumps (top) and the L statistic against its Gumbel critical value (bottom)"><br>
  <em>Figure 1. CVX/XOM: Lee–Mykland statistic vs. Gumbel critical value. Rejections are sparse and isolated; there are no visible bursts.</em>
</p>

**Verdict on H1: not supported.**

- Four of five pairs need the nominal fallback because FDR leaves 0–5 events. With so few events, three of the fits converge to the Poisson boundary $\alpha \to 0$.
- On those three pairs $\hat\beta = 0.500$ is exactly the optimiser's starting value. This is the Davies problem in action: once $\alpha=0$, $\beta$ has no influence on the likelihood and is not estimated at all.
- SPY/IVV, the only pair with a well-populated FDR basis, shows modest point estimates ($\eta = 0.14$), but neither the bootstrap LR ($p=0.086$) nor the CI rejects Poisson.
- GLD/GDX's bootstrap $p = 0.008$ rests on **five events** and only 122 usable bootstrap replications. The $\chi^2$ reference gives $p=0.09$ and the $\eta$ CI spans [−0.20, 0.91]. This is not credible evidence of excitation.
- Across the 23 walk-forward refits, the LR test rejects at 5% in 9/23 quarters for SPY/IVV, 10/23 for GLD/GDX, 1/23 for GS/MS and 0/23 for CVX/XOM and AMD/NVDA. Without correction across 23 refits this is weak and unstable evidence.
- On the α≈0 pairs the intensity-calibration GLM is degenerate (slopes of order $-10^5$), because there is no variation in $\hat\lambda(t)$ to calibrate.

<p align="center">
  <img src="outputs/SPY_IVV/train_val/hawkes_intensity.png" width="85%" alt="SPY/IVV fitted Hawkes intensity: isolated spikes decaying to baseline within days"><br>
  <em>Figure 2. SPY/IVV: the most active intensity in the study. Ten isolated spikes over eight years, each decaying to baseline within about a week.</em>
</p>

A direct consequence is that **the Hawkes arm is almost always in the CALM regime**. In validation it occupies CALM on 97% (SPY/IVV), 99.8% (CVX/XOM, GS/MS, AMD/NVDA) and 92% (GLD/GDX) of days, and on 60–94% of walk-forward days. In validation, the CRISIS block never binds (0 entries blocked on every pair), the decay gate binds once (SPY/IVV), and the jump-entry path fires at most once per pair.

### 9.4 Train / validation, both arms

| Pair | Window | Arm | Trades | Ann. ret | Ann. vol | Sharpe (HAC se) | Max DD | NW excess %/yr | NW *t* | *p* |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY/IVV | Train | Hawkes | 61 | −0.12 | 0.45 | −4.72 (0.63) | −2.00 | −2.12 | −7.70 | <0.001 |
| | | Control | 57 | 0.10 | 0.42 | −4.53 (0.63) | −0.89 | −1.90 | −7.30 | <0.001 |
| | Val | Hawkes | 25 | −0.09 | 0.41 | −5.08 (0.89) | −0.72 | −2.08 | −5.84 | <0.001 |
| | | Control | 19 | 0.63 | 0.31 | −4.35 (0.90) | −0.43 | −1.37 | −4.92 | <0.001 |
| CVX/XOM | Train | Hawkes | 21 | 3.10 | 1.85 | 0.58 (0.46) | −1.45 | 1.07 | 1.27 | 0.205 |
| | | Control | 17 | 2.18 | 1.52 | 0.11 (0.47) | −2.17 | 0.17 | 0.23 | 0.817 |
| | Val | Hawkes | 10 | 1.69 | 1.62 | −0.19 (0.74) | −1.86 | −0.30 | −0.25 | 0.800 |
| | | Control | 8 | 1.42 | 1.20 | −0.48 (0.73) | −1.06 | −0.58 | −0.65 | 0.513 |
| GS/MS | Train | Hawkes | 21 | 0.58 | 1.73 | −0.81 (0.46) | −2.94 | −1.40 | −1.79 | 0.074 |
| | | Control | 22 | 1.54 | 1.36 | −0.34 (0.48) | −1.60 | −0.46 | −0.71 | 0.475 |
| | Val | Hawkes | 10 | 1.76 | 1.80 | −0.13 (0.62) | −1.70 | −0.24 | −0.21 | 0.830 |
| | | Control | 8 | 1.46 | 1.38 | −0.39 (0.59) | −1.17 | −0.54 | −0.66 | 0.507 |
| AMD/NVDA | Train | Hawkes | 19 | −0.04 | 4.90 | −0.39 (0.49) | −15.76 | −1.92 | −0.80 | 0.426 |
| | | Control | 19 | 2.53 | 3.30 | 0.17 (0.47) | −8.05 | 0.55 | 0.36 | 0.721 |
| | Val | Hawkes | 7 | −6.00 | 5.32 | −1.51 (0.64) | −12.66 | −8.06 | −2.36 | 0.019 |
| | | Control | 7 | −2.28 | 3.72 | −1.14 (0.64) | −6.96 | −4.24 | −1.80 | 0.073 |
| GLD/GDX | Train | Hawkes | 21 | 1.39 | 1.58 | −0.38 (0.41) | −1.22 | −0.61 | −0.94 | 0.347 |
| | | Control | 21 | 1.76 | 1.54 | −0.16 (0.36) | −1.59 | −0.25 | −0.44 | 0.661 |
| | Val | Hawkes | 8 | 1.12 | 1.51 | −0.58 (0.64) | −1.03 | −0.88 | −0.91 | 0.362 |
| | | Control | 8 | 1.50 | 1.09 | −0.47 (0.61) | −0.73 | −0.51 | −0.76 | 0.445 |

*"Ann. ret" includes the risk-free credit on idle cash, so a positive return with a negative excess return means "earned less than cash". Beta vs. SPY lies in [−0.016, 0.031] throughout, confirming market neutrality.*

Validation samples are 7–25 trades. Most HAC Sharpe standard errors (0.6–0.9) are larger than the differences between arms. The CVX/XOM training Sharpe of 0.58 disappears out of sample, and AMD/NVDA is the clearest case of in-sample-to-out-of-sample decay (see Section 9.8).

### 9.5 Walk-forward out-of-sample (headline): 23 quarters, 2020-07-01 → Feb 2026

| Pair | Arm | Trades | Win % | Ann. ret | Ann. vol | Sharpe (HAC se) | Max DD | NW excess %/yr [95% CI] | NW *t* | *p* | DSR prob. | MDE₈₀ %/yr |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| SPY/IVV | Hawkes | 46 | 0 | 0.75 | 0.33 | −3.80 (0.55) | −0.55 | −1.25 [−1.60, −0.90] | −7.03 | <10⁻¹¹ | ≈0 | 0.39 |
| | Control | 39 | 0 | 0.92 | 0.31 | −3.55 (0.58) | −0.43 | −1.09 [−1.43, −0.74] | −6.17 | <10⁻⁹ | ≈0 | 0.36 |
| CVX/XOM | Hawkes | 35 | 54 | 1.76 | 1.69 | −0.14 (0.43) | −3.08 | −0.24 [−1.65, 1.18] | −0.33 | 0.740 | 0.001 | 2.00 |
| | Control | 29 | 55 | 1.53 | 1.30 | −0.36 (0.42) | −1.80 | −0.47 [−1.54, 0.60] | −0.86 | 0.388 | <0.001 | 1.54 |
| GS/MS | Hawkes | 21 | 62 | 1.94 | 1.60 | −0.04 (0.43) | −1.81 | −0.06 [−1.40, 1.27] | −0.09 | 0.926 | 0.002 | 1.89 |
| | Control | 18 | 44 | 1.99 | 1.40 | −0.01 (0.42) | −1.43 | −0.01 [−1.16, 1.13] | −0.02 | 0.980 | 0.003 | 1.65 |
| AMD/NVDA | Hawkes | 17 | 59 | 2.75 | 5.35 | 0.16 (0.38) | −10.31 | +0.86 [−3.16, 4.87] | 0.42 | 0.676 | 0.008 | 6.33 |
| | Control | 10 | 70 | 1.54 | 4.63 | −0.08 (0.41) | −13.93 | −0.37 [−4.09, 3.35] | −0.19 | 0.847 | 0.002 | 5.48 |
| GLD/GDX | Hawkes | 17 | 59 | 1.89 | 2.03 | −0.05 (0.36) | −2.06 | −0.11 [−1.56, 1.34] | −0.14 | 0.885 | 0.002 | 2.40 |
| | Control | 20 | 70 | 1.99 | 1.33 | −0.01 (0.35) | −0.96 | −0.02 [−0.92, 0.88] | −0.04 | 0.968 | 0.002 | 1.57 |

*Deflated Sharpe uses 207 trials; the expected maximum Sharpe under no skill is ≈ 1.17. MDE₈₀ is the smallest annual excess return detectable at 80% power and 5% size.*

**Paired test of H2** ($r^{\text{Hawkes}}_t - r^{\text{Control}}_t$, Newey–West; computed from `walk_forward/{hawkes,control}/walk_forward_equity_curve.csv`):

| Pair | Mean difference %/yr | 95% CI | *t* | *p* | Corr(arms) |
|---|---:|---|---:|---:|---:|
| SPY/IVV | −0.16 | [−0.41, 0.08] | −1.30 | 0.193 | 0.64 |
| CVX/XOM | +0.23 | [−0.87, 1.33] | 0.41 | 0.680 | 0.67 |
| GS/MS | −0.05 | [−0.90, 0.81] | −0.11 | 0.911 | 0.73 |
| AMD/NVDA | +1.22 | [−2.87, 5.32] | 0.59 | 0.559 | 0.48 |
| GLD/GDX | −0.09 | [−1.00, 0.83] | −0.19 | 0.850 | 0.73 |
| **Equal-weight pooled** | **+0.23** | **[−0.65, 1.11]** | **0.51** | **0.609** | — |

**Verdict on H2: not rejected on any pair.** **Verdict on H3: not rejected for any tradeable pair; significantly negative for SPY/IVV.**

The Hawkes arm runs higher gross exposure than the control on every pair (CVX/XOM 13.1% vs. 9.8%, GS/MS 13.6% vs. 11.4%, AMD/NVDA 20.6% vs. 16.3%, GLD/GDX 13.3% vs. 9.1%) and correspondingly higher volatility. Section 10 explains why.

### 9.6 Cross-pair portfolio (walk-forward, equal weight)

| Arm | Mean pairwise ρ | $N_{\text{eff}}$ | Excess %/yr | SE | *t* | *p* | 95% CI | Sharpe (HAC se) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Hawkes | −0.028 | 5.00 | −0.16 | 0.47 | −0.35 | 0.726 | [−1.08, 0.75] | −0.14 (0.40) |
| Control | +0.001 | 4.98 | −0.39 | 0.43 | −0.92 | 0.356 | [−1.23, 0.44] | −0.38 (0.41) |

The pair return streams are essentially uncorrelated, so breadth delivers its full $\sqrt{5}\approx 2.2\times$ reduction in standard error. Even so, the pooled confidence interval is ±0.9%/yr wide and contains zero.

### 9.7 Robustness (CVX/XOM walk-forward)

| Scenario | Hawkes: excess %/yr (*t*) | Control: excess %/yr (*t*) | Note |
|---|---:|---:|---|
| Baseline | −0.24 (−0.33) | −0.47 (−0.86) | |
| Half cost | +0.16 (+0.22) | −0.12 (−0.22) | The only scenario with a positive point estimate |
| Double cost | **−1.78 (−2.42)** | −0.39 (−0.61) | The higher-turnover Hawkes arm degrades fastest |
| Fixed 1.5 / 0.5 | −0.86 (−1.20) | −0.75 (−1.34) | No tuning |
| Fixed 2.5 / 0.5 | −1.35 (−2.11) | −0.96 (−1.26) | |
| Bipower detector | −0.48 (−0.71) | −0.47 (−0.86) | Control unchanged by construction |
| Static OLS hedge | −0.24 (−0.33) | −0.47 (−0.86) | Identical to baseline; see Section 11.2(a) |

No pre-specified perturbation produces a significantly positive result. Cost sensitivity dominates every other design choice.

### 9.8 Diagnostics: ceiling, predictability and costs

| Pair | $t_{1/2}$ (d) | Indep. trips/yr | Sharpe ceiling $\sqrt{252\kappa/\pi}$ | Edge/trip % | Cost/trip % (RT + borrow) | Cost share of edge | Predictive *t*, 20 d (train / val) | Predictive *t*, 60 d (train / val) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| SPY/IVV | 1.7 | 74.8 | 5.75 | 0.12 | 0.42 + 0.00 | **343%** | −18.8 / −15.5 | −14.2 / −11.5 |
| CVX/XOM | 66.1 | 1.9 | 0.92 | 7.12 | 0.42 + 0.13 | 7.7% | −2.59 / −0.91 | −2.14 / −3.16 |
| GS/MS | 52.2 | 2.4 | 1.03 | 4.71 | 0.42 + 0.12 | 11.6% | −1.38 / −1.50 | −2.64 / −0.47 |
| AMD/NVDA | 69.4 | 1.8 | 0.89 | 12.93 | 0.42 + 0.28 | 5.4% | −0.25 / +0.35 | −0.16 / −0.46 |
| GLD/GDX | 102.0 | 1.2 | 0.74 | 7.33 | 0.42 + 0.81 | 16.8% | −1.03 / −2.46 | −4.36 / −1.93 |

The ceiling assumes OU trading with position proportional to $(\theta - S_t)$: daily Sharpe $=\kappa\,\mathbb E|\theta-S|/\sigma = \sqrt{\kappa/\pi}$. It assumes perfect parameters, continuous rebalancing and zero cost, so it is an *upper bound*, not a forecast. "Edge/trip" is $(2.0-0.5)\operatorname{sd}_\infty/(1+h)$.

Three conclusions follow:

1. **SPY/IVV** is extremely predictable ($|t| > 10$) and economically worthless: the edge per round trip is about a third of the round-trip cost. The table shows the difference between statistical and economic significance in a single row.
2. **AMD/NVDA** has *no* predictability in either window ($|t| < 0.5$). The z-score does not forecast the spread, so no overlay can rescue it. The structural break in NVDA during the AI cycle is the obvious candidate cause.
3. For the other pairs, **independent round trips per year (1.2–2.4) are the binding constraint**. Over a two-year validation window that is roughly four effective observations, which is why every CI in Section 9.4 is wide.

### 9.9 Cross-sectional screen (45 candidate pairs, training window)

| | Count |
|---|---:|
| Candidates tested | 45 |
| Expected false positives at 5% | 2.25 |
| Pass EG cointegration (nominal) | 5 (IVV/SPY 0.009, AMD/NVDA 0.016, AMD/GLD 0.025, GS/MS 0.042, IVV/MS 0.043) |
| Pass after BH-FDR | **0** |
| Pass after Bonferroni | **0** |
| Selected (all criteria, uncorrected) | 1 (IVV/MS, which has no obvious economic rationale) |
| Selected after FDR | **0** |

The number of nominally cointegrated pairs (5) is barely above what noise alone produces (2.25). The original five pairs fail the screen for different reasons:

- CVX/XOM and GDX/GLD are not cointegrated.
- GS/MS and AMD/NVDA are not predictive.
- IVV/SPY fails on half-life and on edge versus cost (0.31×).

Selecting pairs on nominal *p*-values and trading the winner is the cross-sectional analogue of the time-series multiplicity error that FDR addresses in jump detection.

### 9.10 Per-pair interpretation

- **SPY/IVV.** Statistically the best-behaved pair: cointegrated, the most jumps, the only sensible Hawkes fit. Economically it is a pure cost drain: all 85 walk-forward trades across both arms lose money. The near-identical ETFs are kept in line by the creation/redemption arbitrage, so the residual tracking-error "spread" reverts within about two days, by amounts far below 42 bp round-trip cost.
- **CVX/XOM.** The best realised Sharpe in training (0.58), but not cointegrated by EG and with a mean that shifted during the 2022 energy shock. Walk-forward excess return is indistinguishable from zero in both arms. Of 35 walk-forward Hawkes-arm trades, 16 exit on `max_hold` with a net loss of about $45k, which offsets the gains from mean-reversion and profit-target exits.
- **GS/MS.** Passes validation and is marginally cointegrated, but the z-score has no reliable short-horizon predictive power. Excess returns are ≈ 0 in both arms.
- **AMD/NVDA.** Cointegrated in training, yet the spread is unpredictable and has the highest volatility and drawdown (−10% to −14%). The best walk-forward point estimate (+0.86%/yr, Hawkes arm) comes with an MDE of 6.3%/yr. It is noise.
- **GLD/GDX.** Not cointegrated, with a 100-day model half-life. It produces the study's most "significant" Hawkes fit, which rests on five events. Excess return ≈ 0.

---

## 10. Discussion: Why There Is No Effect

**1. The Hawkes layer has almost nothing to condition on.** At daily resolution, with correct dating and multiplicity control, the training samples contain 0–13 jumps per pair. The synthetic study in `intraday.demonstrate_power` (`python intraday.py`; 40 replications per row) simulates a process with a *true* $\eta = 0.5$ and measures how often the 95% CI on $\eta$ excludes zero:

| Events (sampling equivalent) | ≈12 (daily, this repo) | ≈40 (hourly) | ≈150 (15-min) | ≈600 (5-min) | ≈2,400 (1-min) |
|---|---:|---:|---:|---:|---:|
| Detection rate | 50% | 90% | 100% | 100% | 100% |
| Median CI width on $\eta$ | 0.86 | 0.52 | 0.27 | 0.14 | 0.07 |

Even for a strong effect, the daily event count gives coin-flip power and an uninformative interval. Weaker excitation, or the five to eight events available on four of the five pairs, is worse still. **The null on H1 is a statement about the estimator's power, not a demonstration that equity spread jumps are Poisson.**

**2. As implemented, the Hawkes arm is mostly a re-parameterised control.** Because $\hat\lambda(t) \approx \bar\lambda$ almost everywhere, the regime is CALM on 92–100% of validation days. In CALM the Hawkes arm enters at $0.85\,z_{\text{in}}$, exits at $0.85\,z_{\text{out}}$, holds up to 1.2× longer, and sizes at $f_\lambda f_{\text{regime}} = 1.5 \times 1.2 = 1.8$ times the control's multiplier, up to the 25% cap. The arm-level differences in Section 9.5 (more trades, higher exposure and volatility, faster degradation under double cost) are therefore **the result of looser bands and more leverage, not of information extracted from jump clustering**. A cleaner H2 design would hold thresholds and sizing fixed in CALM; see Section 12.

**3. The premise of mean reversion is weak out of sample.** The predictability regressions show that the z-score forecasts the spread reliably only for SPY/IVV, where the edge is too small to trade. Elsewhere $|t| \lesssim 2.5$ in training and often vanishes in validation. A risk overlay can reshape a return distribution, but it cannot create a first moment that the base signal lacks.

**4. Bets per year limit what the data can show.** With half-lives of 50–100 days, each pair supports about 1–2.5 independent round trips per year. Since $\text{MDE}_{80\%} \approx 2.8\,\sigma_{\text{ann}}/\sqrt{\text{years}}$, detecting a 1%/yr edge on a single pair at 1.3–2.0% annual volatility needs roughly 13–31 years of data. Breadth across five uncorrelated pairs helps (SE falls by √5), but 5.6 years of five pairs still leaves a ±0.9%/yr interval (Section 9.6). The fundamental law of active management, $IR \approx IC\sqrt{\text{breadth}}$, captures the same problem.

**5. Costs dominate small edges.** At the configured 21 bp per side, round-trip cost absorbs 5–17% of the theoretical edge per trip on the slow pairs before any estimation error, and 343% on SPY/IVV. On CVX/XOM, halving costs moves the point estimate from −0.24 to +0.16%/yr, and doubling them makes the Hawkes arm significantly negative.

**What the negative result does and does not say.** It does *not* say that jump clustering is absent from equity spreads, or that intensity-aware execution is useless. It says that **daily bars on a handful of pairs cannot identify self-excitation, and an overlay built on an unidentified intensity cannot add measurable value**. The rigour of the protocol is what makes this negative result credible. An earlier, less careful version of the same code reported strong self-excitation ($\eta \approx 0.82$–0.85 on three pairs) for reasons that turned out to be artefacts (Section 11.1).

---

## 11. Pitfalls, Limitations and Threats to Validity

### 11.1 Failure modes identified and corrected

Each of these was present in an earlier version of this repository. Several of them, individually, would have produced a publishable-looking false positive.

| # | Pitfall | Mechanism and consequence | Fix |
|---|---|---|---|
| 1 | **Window-attributed jump dating** | A rolling 20-day BNS test flags "a jump somewhere in the last 20 days" but was attributed to the window's last day. One true jump became a run of up to 20 consecutive "events", misdated by 0–19 days. A Hawkes fit to runs of consecutive integers reports strong excitation with fast decay, which is exactly what was reported. | Per-observation Lee–Mykland; BNS demoted and attributed to the largest move in the window |
| 2 | **Flawed BNS implementation** | Missing variance constant $\vartheta\approx0.609$; tripower quarticity scaled by full-sample $n$ (≈100× inflation); $\mu_{4/3}$ computed with $\Gamma(5/6)$ instead of $\Gamma(7/6)$ | Corrected, log-ratio form, robustness only |
| 3 | **No multiplicity control** | ≈98 expected false jumps at 5% over ≈1,950 tests | BH-FDR, with a disclosed nominal fallback |
| 4 | **Calendar-day event clock** | Weekends inserted artificial 3-day gaps into Hawkes inter-arrival times, biasing $\hat\beta$ | Positional trading-day times |
| 5 | **Penalty cliff in Hawkes MLE** | Objective returned $10^{10}$ if $\eta>0.85$. L-BFGS-B with finite differences cannot see past it, so every $\hat\eta$ sat within 2% of the wall, where SEs are invalid. | Logistic reparameterisation with an interior optimum |
| 6 | **Self-excitation never tested** | Clustering was asserted from point estimates | Bootstrap LR, Hessian CI, time-rescaling KS |
| 7 | **$\Delta t = 1/252$ unit error** | $\kappa$ per year compared with half-lives in days. A GS/MS $t_{1/2}$ of 56 *years* was reported as agreeing with 56 *days* | Trading-day units, guarded by an assertion |
| 8 | **Silent $\kappa$ override** | Fitted $\kappa$ replaced by $\ln 2/t^{\text{emp}}_{1/2}$ whenever they disagreed, while a "validation passed" message printed | Report only; optional raise |
| 9 | **Rolling hedge ratio** | The $\Delta h_t \log P^B$ term was 99%+ of spread variation; σ inflated 4.8–81.9×; `.bfill()` looked ahead | Static hedge on train; warm-up rows dropped |
| 10 | **Split days deleted, not adjusted** | NVDA's 4:1 and 10:1 splits left permanent level shifts of $h\log 4$ and $h\log 10$ in a spread with sd ≈ 0.7, so AMD/NVDA results measured two stock splits | Verified back-adjustment; hard failure on any residual action |
| 11 | **Percentile regimes on a spike train** | $\lambda\ge\bar\lambda$ makes lower quantiles collapse onto $\bar\lambda$, so CALM was unreachable on two pairs | Relative-excess cut-points |
| 12 | **`elif` exit chain** | `max_hold`, `regime_crisis` and `emergency_stop` were unreachable; confirmed by zero occurrences across all trade logs | Independent evaluation with an explicit priority |
| 13 | **Uncredited idle cash + CAPM headline** | A dollar-neutral book flat about 90% of the time "earned" an alpha of ≈ $-r_f$ on every pair with $R^2\approx 0.0005$ | Cash credit; NW mean-excess headline; CAPM for neutrality only |
| 14 | **Restarted-and-stitched walk-forward** | Fresh capital each quarter, positions silently liquidated, 23 returns zeroed, all-zero trade metrics, failed quarters dropped (survivorship) | One continuous book; failures logged |
| 15 | **Thresholds chosen on the full sample** | The "OOS" quarters reused $(2.0, 0.5)$ chosen with hindsight | Tuning inside the loop on train data; trial count fed to the DSR |

The general lesson for point-process work on financial data: **apparent self-excitation is easy to fabricate.** Misdating, temporal aggregation of tests, calendar-time clocks and boundary-constrained optimisers each bias $\hat\eta$ upward, and they compound.

### 11.2 Known implementation caveats in the current code

These are disclosed so that readers can judge their effect. None of them plausibly reverses the qualitative conclusions, but several affect specific numbers.

- **(a) "Johansen" hedge is Engle–Granger OLS.** In `estimate_hedge_ratio_static`, `method="johansen"` dispatches to `_hedge_engle_granger`, so the hedge ratio is the OLS slope of $\log A$ on $\log B$ with an intercept. `hedge_ratio_diagnostics.csv` shows `hedge_ratio` equal to `h_ols_a_on_b` to 14 digits, and the `static_ols_hedge` robustness scenario is identical to the baseline. `_hedge_johansen` is implemented but never called.
- **(b) Transaction costs are 10× the inline comment.** `BacktestConfig.commission_rate = 0.002` is annotated "2bp per side" but equals 20 bp. With 1 bp slippage the effective cost is 21 bp per side and 42 bp round trip, which is consistent with the 0.42% used in Sections 9.8–9.9. That is conservative for liquid large-cap equities and ETFs, where institutional all-in costs are typically single-digit bp. Given Section 9.7, this choice materially depresses every excess-return estimate. All published numbers use 42 bp RT.
- **(c) Maximum holding period is not wired from config.** `TradingConfig.max_hold_fraction = 1.5` is not passed to `TradingSignals`. In train/val it is passed as `target_hold_fraction`, and walk-forward passes neither. The effective maximum hold is therefore the class default of **0.8 × half-life** (0.96× in CALM), not 1.5×. This probably contributes to the large number of loss-making `max_hold` exits on CVX/XOM.
- **(d) Trailing-stop distance uses the activation multiplier.** In volatility mode the trailing distance is set to `trailing_activation_sigma` (1.5 sd), not `trailing_stop_sigma` (3 sd). In fixed mode, `__init__` similarly assigns the activation percentage to the trailing distance.
- **(e) The stop-sensitivity diagnostic is inert.** `diagnostics.stop_sensitivity` overrides the *fixed* stop percentages, but the default `stop_mode="volatility"` ignores them. All four configurations in `outputs/diagnostics/*/stop_sensitivity.csv` are therefore identical, and the claim that fixed 3% stops harmed performance rests on the earlier run described in the config docstring, not on the published table.
- **(f) The validation gate does not gate.** `is_tradeable` is computed and saved but never blocks trading. SPY/IVV, CVX/XOM and GLD/GDX fail validation on the training window yet are traded in both pipelines. Their results should be read as "what happens if the gate is ignored".
- **(g) Look-ahead in the risk overlay and sizing.** In train/val, stop levels are scaled by the *evaluation window's own* spread s.d. In walk-forward, the continuous book is sized with the **mean of all 23 quarterly hedge ratios**, including those estimated after the trade date, and stops are scaled by the s.d. of the full-sample-hedge spread over the whole OOS period. Signals are causal, but the P&L book tracks $\log A - \bar h\log B$ rather than each quarter's signalled spread. Stops rarely bind (Section 9.10), so the effect on returns is small, but these are leaks.
- **(h) Jump flags are not strictly causal.** `compute_artifacts` runs Lee–Mykland and BH over the *full* sample. The Gumbel normalisers $C_n, S_n$ depend on $n$, and the BH threshold depends on the full *p*-value distribution, so whether day $t$ is flagged depends weakly on later data. The intensity is causal given the flags.
- **(i) Lee–Mykland + BH is doubly conservative.** The Gumbel *p*-value is the probability that the *maximum* of $n$ null statistics exceeds $\mathcal L_i$, which is already a family-wise adjustment. Applying BH on top over-corrects, and even the "nominal" basis is a family-wise 5% rule. This pushes event counts down and compounds the scarcity in Section 10. A per-observation (non-maximal) calibration followed by BH would be the internally consistent alternative.
- **(j) Detector-comparison column.** `JumpDetector.calculate_jump_statistics` computes inter-jump times from jump *sizes*, so `mean_inter_jump_days` in `jump_detector_comparison.csv` is not meaningful. The other columns are unaffected.
- **(k) Deflated Sharpe trial count.** The 207 trials are nine configurations on each of 23 *different* expanding windows, not 207 strategies on the same series. Treating them as one search is conservative, and the DSR here should be read as indicative.
- **(l) Capital-at-risk extrapolation.** `car_*` metrics scale excess return linearly by 1/mean exposure. For SPY/IVV (about 2% mean exposure, 46× scale) this produces meaningless figures (−55%/yr) and should be ignored.

### 11.3 Data and design limitations

- **Dividends excluded.** The omitted dividend differential (up to about 1.3%/yr gross on the long/short legs, roughly 9–17 bp/yr on capital at observed exposures) is of the **same order as the effects being tested** for CVX/XOM, GS/MS and GLD/GDX. The SPY benchmark is a price series, which affects only the neutrality regression.
- **Ex-post pair selection.** The five registered pairs are well-known textbook pairs chosen with general hindsight. The FDR-corrected screen selects none of them.
- **Small universe.** Ten symbols and five pairs. Effective breadth is the main constraint on power.
- **Regime composition.** Training includes the 2020 COVID dislocation and the 2022 energy shock. Validation and walk-forward include the NVDA AI-driven re-rating. Stationarity of the cointegrating relation across these periods is doubtful for CVX/XOM, AMD/NVDA and GLD/GDX.
- **Daily frequency.** Jump detection, BNS asymptotics and Hawkes identification all favour intraday data. `intraday.py` provides frequency-aware infrastructure, but **no intraday data ships with this repository and no intraday P&L is claimed.**
- **Execution realism.** Open-auction fills with no market impact, no borrow recalls, a constant borrow table, and a single cost level for every symbol regardless of liquidity.
- **Univariate Hawkes.** Jumps in the two legs, in the market, and in spreads across pairs plausibly *cross*-excite. A univariate model on the spread cannot capture this.

---

## 12. Implications and Future Work

**For practitioners**

1. Do not infer self-excitation from point estimates. Use a bootstrap LR test against Poisson and a time-rescaling goodness-of-fit check, and look at how many events the estimate rests on. With about a dozen events, even a true branching ratio of 0.5 is detected only half the time; with five, the estimate mostly reflects the optimiser's starting point.
2. Statistical significance of mean reversion is not economic significance. SPY/IVV is the most predictable pair in the study and the only one that loses money with certainty.
3. When an overlay changes thresholds and leverage, you need a matched control. Otherwise "the overlay helped" cannot be separated from "more leverage helped".
4. Report MDEs alongside null results. A 1%/yr edge on a single daily pair is essentially undetectable over a decade.

**Research directions**

1. **Intraday data (5-minute or finer).** This increases events by one to two orders of magnitude, puts BNS in its valid asymptotic regime, and allows the Hawkes layer to be identified. This is the change most likely to make the title hypothesis testable.
2. **Multivariate / marked Hawkes.** Model cross-excitation between legs, the market and sector ETFs, with jump size as a mark (Aït-Sahalia et al. 2015).
3. **A cleaner H2 design.** Fix thresholds and sizing at control values in CALM and let the Hawkes layer act only when $e_t$ is elevated, so the treatment isolates information content. Alternatively, use $\lambda(t)$ purely as a *risk* input (volatility forecasting, position caps) rather than a signal.
4. **Breadth.** A sector-neutral universe of hundreds of pairs, screened with FDR control, in the spirit of Avellaneda & Lee (2010), to trade breadth for per-pair power.
5. **Fixes to Section 11.2**, in particular (a), (b), (c), (f) and (g), followed by a rerun of the published matrix.
6. **Model-based execution.** Use the MRJD conditional distribution for optimal entry and exit bands (Bertram 2010) instead of fixed z-thresholds.

---

## 13. Reproducibility

**Environment.** Python ≥ 3.11. Pinned versions are in `requirements.txt` (numpy, pandas ≥ 2.2 for the `QE` alias, scipy, statsmodels, matplotlib, pytest). All stochastic paths are seeded (default 42).

```bash
pip install -r requirements.txt

# Train/validation + walk-forward, both arms, all registered pairs
python main.py --pair all --mode all

# Single pair / single mode
python main.py --pair CVX_XOM --mode train_val
python main.py --pair GS_MS  --mode walk_forward --no-control

# Cross-pair equal-weight portfolio of walk-forward returns
python main.py --pair all --mode portfolio

# Pre-specified robustness matrix (published for CVX_XOM)
python main.py --pair CVX_XOM --mode robustness

# Options: --detector {lee_mykland,bipower,threshold}  --hedge-mode {static,periodic,rolling}
#          --no-fdr  --seed N  --quiet  --screened (trade FDR-surviving screen pairs only)

python pair_screen.py --out outputs/pair_screen.csv   # 45-pair screen
python diagnostics.py --pair all --which all          # ceiling / predictability / stops
python intraday.py                                    # synthetic Hawkes power study
python hawkes_calibration.py                          # synthetic recovery check
python mrjd_estimation.py                             # synthetic recovery check

pytest -q                                             # 62 tests (all passing)
```

**Test suite.** The 62 tests are regression tests for the failure modes in Section 11.1 and for estimator validity. They cover:

- Hawkes: parameter recovery, CI coverage, an interior optimum, and the LR test rejecting on clustered data but not on Poisson data.
- MRJD: recovery of OU parameters in day units, rejection of year units and explosive series.
- Lee–Mykland: correct jump dating and no false flags on Gaussian noise.
- BH-FDR control.
- Split detection and adjustment.
- Hedge-ratio variance reduction.
- Reachability of every exit.
- An exact risk-free return for a zero-trade book.
- Next-bar fills.
- Pooling and effective breadth.

**Artefacts.** Every file is written by `results_io.ResultsWriter`, and each output directory contains a `MANIFEST.json` listing exactly the files the current run produced. **Files present in `outputs/` but not listed in the adjacent manifest are legacy artefacts from earlier versions** and should not be cited. These include `train_val/performance_metrics.csv`, `train_val/hawkes_suitability.csv`, and the `walk_forward/quarterly_*` and `walk_forward/walk_forward_*` files at the root of `walk_forward/`. Current walk-forward results live in `walk_forward/hawkes/`, `walk_forward/control/` and `walk_forward/oos_arm_comparison.csv`.

```text
outputs/
├── <PAIR>/train_val/          bundle, pair validation, spread stats, MRJD diagnostics, Hawkes inference,
│                              causal artefacts, detector comparison, per-arm/window equity, signals,
│                              trades, metrics, exit reasons, arm_comparison.csv, figures
├── <PAIR>/walk_forward/
│   ├── hawkes/  control/      continuous-book equity, signals, trades, metrics, quarterly parameters, run_config.json
│   └── oos_arm_comparison.csv
├── CVX_XOM/robustness/<scenario>/   per-scenario walk-forward (both arms) + walk_forward_robustness.csv
├── portfolio/walk_forward/{hawkes,control}/   pooled returns, correlation, metrics
├── diagnostics/<PAIR>/        ceiling_analysis, predictability_test, stop_sensitivity
├── pair_screen.csv
└── summary_all_pairs*.csv     cross-pair summaries (+ captured terminal logs *_terminal.txt)
```

The paired H2 statistics in Section 9.5 can be reproduced with:

```python
import pandas as pd
from statistics_tools import newey_west_mean_test
for p in ["SPY_IVV", "CVX_XOM", "GS_MS", "AMD_NVDA", "GLD_GDX"]:
    h = pd.read_csv(f"outputs/{p}/walk_forward/hawkes/walk_forward_equity_curve.csv", index_col=0)["returns"]
    c = pd.read_csv(f"outputs/{p}/walk_forward/control/walk_forward_equity_curve.csv", index_col=0)["returns"]
    print(p, newey_west_mean_test((h - c).dropna().iloc[1:].values))
```

---

## 14. Repository Structure

| Module | Responsibility |
|---|---|
| `config.py` | Typed configuration dataclasses, pair registry, `as_control_arm()` |
| `time_units.py` | Single time unit (trading day), positional clocks, the units guard |
| `corporate_actions.py` | Verified split table and adjustment; dividend disclosure |
| `equity_pairs_loader.py` | Loading, cleaning (flag, not delete), hedge-ratio estimators, spread, ADF + EG statistics |
| `jump_detector.py` | Lee–Mykland (primary), BNS (robustness), threshold; BH-FDR; detector comparison |
| `hawkes_calibration.py` | Reparameterised MLE, O(n) likelihood, SEs, bootstrap LR, time-rescaling KS, calibration GLM, Ogata thinning |
| `mrjd_estimation.py` | Exact AR(1) OU MLE, jump moments, optional joint MLE, forecasts, simulation |
| `signal_generation.py` | Z-score, regimes, gates, sizing, independent exits |
| `backtest_engine.py` | Event-driven execution, costs, financing, stops, metrics, bootstrap, CAPM neutrality |
| `statistics_tools.py` | Newey–West test, Lo Sharpe SE, stationary bootstrap, DSR, power/MDE, pooling |
| `pipeline.py` | `ModelBundle`, pair validation, fit / artefacts / evaluate / tune |
| `walk_forward.py` | Continuous-book quarterly walk-forward |
| `main.py` | CLI: train/val, walk-forward, control arm, robustness, portfolio |
| `pair_screen.py` | 45-pair screen with multiplicity correction |
| `diagnostics.py` | Sharpe ceiling, predictability, stop sensitivity |
| `intraday.py` | Frequency abstraction, intraday loaders, synthetic Hawkes power study |
| `results_io.py` | All artefact and figure writing, manifests |
| `tests/` | 62 regression and estimator-validity tests |

---

## 15. References

- Aït-Sahalia, Y., Cacho-Diaz, J. & Laeven, R. (2015). Modeling financial contagion using mutually exciting jump processes. *Journal of Financial Economics*, 117(3), 585–606.
- Avellaneda, M. & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761–782.
- Bacry, E., Mastromatteo, I. & Muzy, J.-F. (2015). Hawkes processes in finance. *Market Microstructure and Liquidity*, 1(1).
- Bailey, D. H. & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.
- Barndorff-Nielsen, O. E. & Shephard, N. (2004). Power and bipower variation with stochastic volatility and jumps. *Journal of Financial Econometrics*, 2(1), 1–37.
- Barndorff-Nielsen, O. E. & Shephard, N. (2006). Econometrics of testing for jumps in financial economics using bipower variation. *Journal of Financial Econometrics*, 4(1), 1–30.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
- Bertram, W. K. (2010). Analytic solutions for optimal statistical arbitrage trading. *Physica A*, 389(11), 2234–2243.
- Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E. & Frank, L. M. (2002). The time-rescaling theorem and its application to neural spike train data analysis. *Neural Computation*, 14(2), 325–346.
- Davies, R. B. (1977; 1987). Hypothesis testing when a nuisance parameter is present only under the alternative. *Biometrika*, 64(2), 247–254; 74(1), 33–43.
- Elliott, R. J., van der Hoek, J. & Malcolm, W. P. (2005). Pairs trading. *Quantitative Finance*, 5(3), 271–276.
- Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error correction. *Econometrica*, 55(2), 251–276.
- Gatev, E., Goetzmann, W. N. & Rouwenhorst, K. G. (2006). Pairs trading: performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3), 797–827.
- Grinold, R. C. (1989). The fundamental law of active management. *Journal of Portfolio Management*, 15(3), 30–37.
- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83–90.
- Huang, X. & Tauchen, G. (2005). The relative contribution of jumps to total price variance. *Journal of Financial Econometrics*, 3(4), 456–499.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian VAR models. *Econometrica*, 59(6), 1551–1580.
- Lee, S. S. & Mykland, P. A. (2008). Jumps in financial markets: a new nonparametric test and jump dynamics. *Review of Financial Studies*, 21(6), 2535–2563.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36–52.
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
- Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Newey, W. K. & West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631–653.
- Ogata, Y. (1981). On Lewis' simulation method for point processes. *IEEE Transactions on Information Theory*, 27(1), 23–31.
- Ogata, Y. (1988). Statistical models for earthquake occurrences and residual analysis for point processes. *Journal of the American Statistical Association*, 83(401), 9–27.
- Ozaki, T. (1979). Maximum likelihood estimation of Hawkes' self-exciting point processes. *Annals of the Institute of Statistical Mathematics*, 31(1), 145–155.
- Phillips, P. C. B. & Ouliaris, S. (1990). Asymptotic properties of residual based tests for cointegration. *Econometrica*, 58(1), 165–193.
- Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.

---

<sub>This repository is a research study, not investment advice. All performance figures are simulated, gross of dividends, and subject to the limitations in Section 11.</sub>
