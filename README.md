# Introduction 

## Motivation and Research Question
Volatility clustering in financial markets has been extensively documented since the seminal work of Mandelbrot (1963) and formalezed through the ARCH/GARCH family of models introduced by Engle (1982) and Bollerslev (1986). In parallel, academics have examined the discrete nature of large price movements, also known as "jumps," which exhibit their own clustering behavior separate from the continous diffusion component of asset returns. I sought out to investigate whether the self-exciting nature of jump arrivals, in which the occurrence of one large price movement increases the probability of subsequent jumps, can be exploited to improve the timing and profitability of a pairs trading strategy.

My central research question is as follows: Could a pairs trading strategy that incorporates Hawkes process modelling of jump intensity dynamics generate a superior risk-adjusted return compared to the naive Poisson mean-reversion approach? I hypothesized that by entering positions only when the jump intensity began to decline (indicating the end of a volatility cascade) and exiting when the intensity began to escalate towards a crisis regime, I would be able to reduce adverse selection and hopefully improve the strategy's Sharpe ratio. 

The motivation for this project stemmed from observable market phenomena. During periods of market stress, it can be observed that large price movements tend to cluster together in time. For example, a single earnings surprise or macroeconomic shock can often precipitate a cascade of subsequent moves as market participants react, counterparties adjust hedges, and algorithmic strategies respond to ever changing conditions. This clustering behavior would suggest that jump arrivals are not independent Poisson events but rather exhibit positive feedback dynamics that can be captured through self-exciting point process models, as seen by the Hawkes process. 

## Scope
First, I developed a complete implementation framework for the combination of Mean Reversion Jump Diffusion (MRJD) spread dynamics with Hawkes intensity modeling, providing a detailed mathematical understanding of the calibration procedures and signal generation logic. Second, I conducted axtensive testing across five distinct equity pairs representing different sectors, correlation structures, and volatility regimes, documenting the successes and failures of each approach. Third, I provide an honest analysis as to why the strategy ultimately fails to generate any alpha, offering insights that could potentially guide future research in a more productive direction. 

The scope of this project is limited to daily frequency equity pairs using publicly available price data. I did not consider intraday implementaiton, derivatives, or alternative assset classes. 

## Theoretical Context
## Self-Exciting Point Processes in Finance
The Hawkes process has found many applications in financial econometrics over the last few decades. the key insight of the Hawkes process is that event arrivals are not independent but rather exhibit positive feedback: each event temporarily increases the probability of subsequent events. This self-excitation property makes Hawkes processes particularly well suited for modeling phenomena such as jump clustering in equity returns. 

The application of the Hawkes process to high-frequency trading and market microstructure have been particularly fruitful. Bacry et al (2015) provide a comprehensive analysis of Hawkes process in finance, demonstrating their utility for modeling the self-exciting dynamics of trades, quotes, and price changes at millisecond frequencies. Their work establishes that order flow exhibits strong self-excitation, with branching ratios often exceeding 0.9, indicating that the markets operate near criticality where small perturbations can lead to large movements. 

My project differs from this literature by applying Hawkes modeling to daily frequency pairs trading, where the relevant events are not individual trades but rather large jumps in the spread between cointegrated assets. This application raises the question of whether the self-excitation dynamics observed at high frequencies persist at the daily horizon and whether they can be exploited for trading profit. 

## Jump Detection and Mean-Reverting Jump Diffusion
The detection of jumps in asset returns has been extensively studied following the development of bipower variation and related realized volatility estimators. Bipower variation, introduced by Barndorff-Nielsen and Shephard (2006), exploits the fact that the product of adjacent absolute returns scales differently under continous diffusion versus jump components. This test has become a standard for separating jumps from continous price variation and forms the foundation of my jump detection methodology. 

Mean-reverting jump diffusion models extend the classical Ornstein-Uhlenbeck process to incorporate discrete jumps, providing a more realistic representation of spread dynamics in pairs trading. The general form equation of the MRJD process is given by:

$$
dS_t = \kappa(\theta - S_t)\dt + \sigma\dW_t + J_t\dN_t
$$

where $\kappa$ represents the speed of mean reversion, $\theta$ is the long-run equilibrium level, $\sigma$ captures diffusive volatility, $J_t$ is the random jump size, and $N_t$ is a counting process governing jump arrivals. The innovation in my approach is to model $N_t$ not as a simple Poisson process but as a Hawkes process with time-varying intensity.

## Pairs Trading and Statistical Arbitrage 
Pairs trading has a long history in quantitative finance. The application of cointegration methods to pairs trading provides a rigorous statistical foundation for identifying tradeable relationships. When two assets are cointegrated, their prices share a common stochastic trend, and any deviation from the equilibrium relationship is expected to be only temporary. However, as my results demonstrate, statistical cointegration is necessary, but not a sufficient condition for profitable trading. The edge must be large enough to survive the transaction costs and must not be arbitraged away by any competing strategies. 

# Mathematical Framework 
## The Hawkes Process
I began with a formal definition of the Hawkes self-exciting point process. Let $N_t$ denote a counting process representing the cumulative number of jumps up to time $t$. The Hawkes process is characterized by a conditional intensity function $\lambda(t)$ that depends on the history of the process:

$$
\lambda(t) = \lambda_0 + \sum_{t_i < t} \phi(t - t_i)
$$

where $\lambda_0 > 0$ is the baseline intensity representing the rate of "exogenous" events, and $\phi(\cdot)$ is the excitation kernel that captures how past events influence current intensity. I adopted the exponential kernel specification:

$$
\phi(s) = \alpha e^{-\beta s}, \quad s > 0
$$

where $\alpha > 0$ is the jump impact parameter representing the instantaneous increase in intensity following a jump, and $\beta > 0$ is the decay rate controlling how quickly the excitation effect dissipates.

The complete intensity function under the exponential kernel is therefore:

$$
\lambda(t) = \lambda_0 + \alpha \sum_{t_i < t} e^{-\beta(t - t_i)}
$$

This specification admits a convenient recursive representation. Defining the auxiliary process:

$$
R_t = \sum_{t_i < t} e^{-\beta(t - t_i)}
$$

Then $\lambda(t) = \lambda_0 + \alpha R_t$ and between jumps $R_t$ decays exponentially:

$$
dR_t = -\beta R_t\,dt
$$

At each jump time $t_i$, we have $R_{t_i^+} = R_{t_i^-} + 1$.

The branching ratio, defined as $n^* = \alpha/\beta$, plays a crucial role in the process dynamics. This quantity represents the expected number of "offspring" events triggered by each "parent" event. When $n^* < 1$, the process is subcritical and stationary, with finite expected intensity. When $n^* \geq 1$, the process becomes critical or supercritical, with explosive behavior. Empirically, I observed branching ratios in the range 0.7-0.9 in the equity pairs that I tested on, indicating strong but subcritical self-excitation.

## Maximum Likelihood Estimation
The parameters $\theta = (\lambda_0, \alpha, \beta)$ are estimated via maximum likelihood. Given a sequence of jump times $\{t_1, t_2, \ldots, t_n\}$ observed over the interval $[0, T]$, the log-likelihood function is:

$$
\ell(\theta) = \sum_{i=1}^{n} \log \lambda(t_i) - \int_0^T \lambda(s)\,ds
$$

The first term rewards high intensity at observed event times, while the second term penalizes high intensity in the absence of events. For the exponential kernel, the compensator integral admits a closed form:

$$
\int_0^T \lambda(s)\,ds = \lambda_0 T + \frac{\alpha}{\beta}\sum_{i=1}^{n}\left(1 - e^{-\beta(T - t_i)}\right)
$$

I optimized this likelihood numerically using the `L-BFGS-B` algorithm with bounds to ensure $\lambda_0, \alpha, \beta > 0$ and $\alpha < \beta$ (to ensure stationarity).

## Mean-Reverting Jump Diffusion
The spread between two cointegrated assets evolves according to a mean-reverting jump diffusion process. Let $S_t$ denote the log-spread, defined as:

$$
S_t = \log P_t^A - h \log P_t^B
$$

where $P_t^A$ and $P_t^B$ are the prices of the two assets and $h$ is the cointegration hedge ratio estimated via ordinary least squares on the log prices.

The dynamics of $S_t$ are modeled as:

$$
dS_t = \kappa(\theta - S_t)\dt + \sigma\dW_t + J_t\dN_t
$$

The first term represents mean reversion toward the equilibrium level $\theta$ at speed $\kappa$. The second term captures continous Brownian fluctuations with volatility $\sigma$. The third term introduces discrete jumps of random size $J_t \sim \mathcal{N}(\mu_J, \sigma_J^2)$, arriving according to the counting process $N_t$.

The half-life of mean reversion, a key quantity for calibrating trading horizons, is given by:

$$
t_{1/2} = \frac{\ln 2}{\kappa}
$$

This represents the expected time for the spread to move halfway from its current level back toward equilibrium. My empirical estimates yield half-lives ranging from 20 to 50 days across different pairs, which has direct implications for optimal holding periods. 

## Integration of Hawkes and MRJD
The innovation of my framework was to replace the standard Poisson assumption for $N_t$ with a Hawkes process. This would create a feedback mechanism: when the spread experiences a large jump, the intensity of future jumps increase temporarily, capturing the intuition that volatility begets volatility.

I tried exploiting this structure for trading by conditioning on the current intensity level. Defining the regime indicator as: 

$$
\text{Regime}(t)=
\begin{cases}
\text{CALM} & \text{if } \lambda(t) < \lambda_{25} \\
\text{NORMAL} & \text{if } \lambda_{25} \le \lambda(t) < \lambda_{75} \\
\text{ELEVATED} & \text{if } \lambda_{75} \le \lambda(t) < \lambda_{90} \\
\text{CRISIS} & \text{if } \lambda(t) \ge \lambda_{90}
\end{cases}
$$

where $\lambda_p$ denotes the $p$-th percentile of the empirical intensity distribution.

The trading logic incorporated this regime information in several ways. Entry thresholds are adjusted upward during elevated and crisis regimes, requiring larger deviations before initializing positions. I also implemented a "lambda decay" filter that permitted entry only when the intensity began declining. This was defined as:

$$
\frac{\lambda(t) - \max_{s \in [t-5, t]} \lambda(s)}{\max_{s \in [t-5, t]} \lambda(s)} < -0.15
$$

This condition ensured that I entered positions only after the intensity had falled at least 15% from its recent peak, avoiding entry during any active jump cascades. 

## Z-Score Signal Generation
Trading signals were generated based on the empirical z-score of the spread, defined as:

$$
z_t = \frac{S_t - \bar{S}_t(L)}{\hat{\sigma}_t(L)}
$$

where $\bar{S}_t(L)$ and $\hat{\sigma}_t(L)$ are the rolling mean and standard deviation computed over a lookback window of $L = 60$ trading days.

Entry signals are generated when $|z_t|$ exceeds a regime-adjusted threshold. Specifically, for a position to be initiated, it requires:

$$
|z_t| > z_{\text{entry}} \times \text{Regime Multiplier}
$$

where the regime multipliers are 0.85 for CALM, 1.0 for NORMAL, 1.25 for ELEVATED, and positions are prohibited in CRISIS regimes. 

Exit signals were generated when mean reversion was substantially completed according to:

$$
|z_t| < z_{\text{exit}}
$$

or when the position had been held beyond the regime-adjusted maximum holding period, computed as a multiple of the spread's half-life.

# Empirical Methodology
## Data Description
The empirical analysis deployed daily closing prices for five equity pairs representing diverse sectors and correlation structures. The sampling period extended from May 2018 through February 2026, providing approximately 1,960 trading days per pair. The selected pairs are as follows 

- The first pair, SPY/IVV, consisted of two exchange-traded funds that both track the S&P 500 index. The first pair served as a control case, as the two assets are economically identical and any apparent trading opportunities arise purely from tracking error noise. 

- The second pair, XOM/CVX, represented two integrated oil and gas majors with similar business models and exposure to global energy prices. This pair exhibited genuine economic cointegration drien by common exposure to crude oil and natural gas prices. 

- The third pair, GS/MS, consists of two major investment banks with similar business lines and regulatory environments. The cointegration relationship is driven by common exposure to capital markets activity and interest rate dynamics.

- The fourth pair, GDX/GLD, represents gold mining equities (GDX) versus physical gold (GLD). The mines provide operational leverage to gold prices, creating a theoretically cointegrated but more volatile spread

- The fifth pair, NVDA/AMD, represents two semiconductor companies competing in similar markets. While statistically cointegrated over portions of the sample, this pair exhibits sstructural breaks due to NVIDIA's dominant position in AI related chips. 

## Cointegration Testing and Spread Construction
For each pair, I verified cointegration using the Augmented Dickey-Fuller test of the log-spread. Let $p_t^A = \log P_t^A$ and $p_t^B = \log P_t^B$. The cointegration relationship was estimated by:

$$
p_t^A = c + h\ * p_t^B + \epsilon_t
$$

via ordinary least squares (OLS), and I tested the residuals $\hat{\epsilon}_t$ for stationarity. A rejection of the null hypothesis at the 5% level was required for the pair to be considered cointegrated.

The spread was then constructed as $S_t = p_t^A - \hat{h}\*p_t^B$, where $\hat{h}$ is the estimated hedge ratio. For improved stability, I employed a rolling window estimation of the hedge ratio with a 252-day lookback. Although, this had minimal impact on the results.

## Jump Detection Procedure
We detect jumps in the spread returns using the bipower variation test. Let $r_t = S_t - S_{t-1}$ denote the daily spread return. The realized variance and bipower variation over a rolling window of length $W = 20$ days are computed as:

$$
RV_t = \sum_{i=0}^{W-1} r_{t-i}^2
$$

$$
BV_t = \frac{\pi}{2} \sum_{i=0}^{W-2} |r_{t-i}|\*|r_{t-i-1}|
$$

Under the null hypothesis of no jumps, $RV_t - BV_t$ converges to zero in probability. Day $t$ was flagged as containing a jump if:

$$
\frac{RV_t - BV_t}{\sqrt{V_{qq}}} > \Phi^{-1}(1 - \alpha/2)
$$

where $V_{qq}$ is a consistent estimator of the asymptotic variance and $\alpha = 0.01$ is the significance level.

## Backtesting Protocol
The backtesting engine implemented a dollar-neutral position sizing, wherein each leg of the pairs trade receives equal dollar exposure regardless of the hedge ratio. For a position of total size $K$, I allocated $K/2$ dollars long to asset A and $K/2$ dollars short to asset B (or vice versa for short spread position).

The transaction costs were modeled as a combination of commission (1 basis point per transaction) and slippage (0.5 basis points per transaction).

Risk management incorporated three mechanisms. First, a hard stop-loss triggers exit when the position's unrealized loss exceeded 3% of the capital allocated. Second, a trailiing stop activates after the position achieves a 1% gain, thereafter locking in profits by exiting if the position's value falls more than 1.5% from its peak. Third, a time stop exits positions that have been held beyond 1.5 times the spread's half-life. 

# Empirical Results
## Summary Statistics
Table 1 presents a summary statistics for each pair's spread dynamics and jump characteristics. 

Table 1: Sprad and Jump Summary Statistics 
![alt text](<Screenshot 2026-02-20 at 8.04.08 PM.png>)

Substantial heterogeneity exists in jump frequencies and branching ratios. The SPY/IVV and XOM/CVX exhibited the highest jump frequencies at approximately 20%, while GDX/GLD shows only 1.4% jump frequency, rendering Hawkes modeling largely inapplicable for this pair

## Hawkes Parameter Estimates
Table 2 reports the maximum likelihood estimates of the Hawkes process parameters. 

Table 2: Hawkes Process Parameter Estimates
![alt text](<Screenshot 2026-02-20 at 8.17.55 PM.png>)

The branching ratios cluster around the 0.7 - 0.85 range, consistent with the subcritical self-excitation documented in past literature. However, it can be observed that pairs with low jump frequency (GS/MS and GDX/GLD) exhibit degenerate intensity distrubtions where the 90th percentile is close to or even sometimes below the mean. This indicates minimal variation in the intensity process. This severly limits the applicability of regime-based filtering for these pairs. 

## Trading Performance
Table 3 summarizes the backtest results across all pairs

Table 3: Backtest Performance Summary
![alt text](<Screenshot 2026-02-20 at 8.20.30 PM.png>)

The results are uniformly disappointing across the board from a risk-adjusted perspective. Even the best performing pair (XOM/CVX) generated an annualized return of only 0.11%, substantially below the risk-free rate of approximately 2% during the sample period. The Sharpe ratios are also uniformly negative across all pairs, indiated that the strategy destroys value relative to a risk-free investment. 

## Filter Effectiveness Analysis
Table 4 examines the impact of Hawkes-based filtering on signal generation

Table 4: Hawkes Filtering Impact
![alt text](<Screenshot 2026-02-20 at 8.25.48 PM.png>)

The $\lambda$ decay filter proved to be the most restrictive, blocking about 80-207 potential entries per pair. While this aggressive filtering was intended to avoid entry during volatility cascaces, it ultimately resulted in the elimination of nearly all trading opportunities for low-jump frequency pairs. The GDX/GLD pair is completely blocked from trading, while GS/MS retains only 5 trades over an 8-year period. 

# Analysis and Discussion
## Fundamental Disconnect: Statistical Predictability vs Economic Profitability
The central finding of this project is that the Hawkes process successfully captures the self-exciting dynamics of jump arrivals, but failed to generate tradeable alpha. This disconnect ends up warranting careful analysis, as it shines a light on the fundamental limitation of the research question.

The Hawkes model makes accurate predictions about the clustering of jumps in time. The branching ratios that were estimated, ranging from 0.70 to 0.85, indicate that each jump triggers on average 0.7 - 0.85 additional jumps in expectation. This is significant and an economically meaningful finding that aligns with my intuition about volatility cascades in financial markets. When a large move occurs, market participants react, hedgers adjust positions, and algorithmic strategies respond, all of which can precipitate further large moves. 

However, statistical predictability of jump timing does not imply predictability of jump direction or magnitude in a manner that is exploitable for profit. The Hawkes model tells us that if a jump occurred today, we should expect elevated probability of jumps tomorrow. It does not tell us whether those jumps will push the spread further from equilibrium or back toward it. The self-excitation is symmetric with respect to direction, as jumps can cluster in either direction. This was shown as the data exhibited approximately equal frequencies of positive and negative jumps during cascade periods. 

This observation suggests that the Hawkes framewor, while valuable for risk management and volatility forecasting, may not be the appropriate tool for generating trading signals. A more promising approach might combine the Hawkes intensity modeling with directional forecasts based on other signals, using the Hawkes output for position sizing and risk management rather than entry timing. 

## The Over-Filtering Problem
My implementation of regime-based filtering demonstrates a classic trade-off in systematic trading: filters that successfully avoid bad trades often also eliminate good trades. The $\lambda$ decay requirement, that intensity must have fallen from 15% from its recent peak before entry, was motivated by the intuition that we should avoid entering positions during active volatility cascacdes. In practice, this filter blocked the majority of trading opportunities, with blocking rates exceeding 80% for most pairs. 

The problem is particularly apparent for low-jump frequency pairs. When jumps are rare, the intensity process spends most of its time at or near the baseline level with minimal variation. In this regime, the $\lambda$ decay condition became almost impossible to satisfy, as there was no meaningful "peak" from which to decay. The GDX/GLD case is illustrative: with only 27 jumps over 1,961 trading days (1.4% frequency), the intensity distribution is essentially degenerate, and the Hawkes model provides no useful information beyond what a simple Poission assumption would have yielded. 

This finding led me to implement adaptive filtering, wherein Hawkes-based constraints are relaxed for pairs with jump frequencies below 5% or degenerate intensity distributions. While this adjustment allowed trades to be generated, it effectively concedes that the Hawkes framework was inapplicable for these pairs, reducing the strategy to a simple z-score mean reversion approach. 

## Transaction Costs and the Compression of Edge
Even for the pairs where the Hawkes framework was applicable and trades were generated, the strategy failed to overcome transaction costs. Consider the XOM/CVX pair, the best performer. Over 18 trades, the strategy generated a gross return of approximately 0.87%, implying an average gross return per trade of roughly 0.05%. Against this, round-trip transaction costs were occurred of approximately 0.06% per trade, leaving a net negatvie expected value per trade. 

This finding was actually consistent with the broader literature on the decay of statistical arbitrage profits. Do and Faff (2010, 2012) documented that pairs trading profitability has declined substantially since the strategy was first documented, with the decay being attributable to increased competition among quantitative strategies. My results suggest that this decay has continued to the point where simple cointegration-based pairs, even when enhanced with sophisticated jump modeling, no longer generates sufficient edge to cover the transaction costs. 

The implication of this is that any viable pairs trading strategy in the current environment must either operate at a higher frequency, exploit non-public information or alternative data sources, or just focus on less liquid markets where the competition is drastically reduced. 

# Conclusions and Future Directions
## Summary of Findings
This comprehensive analysis has presented an investigation of Hawkes process modeling for pairs trading, combining self-exciting jump dynamics with mean-reverting spread evolution. The key findings can be summarized as follows:

First, the Hawkes process successfully captures jump clustering in equity pair spreads, with branching ratios in the 0.7-0.85 range indicating strong but subcritical self-excitation. This confirms the applicability of self-exciting point process models to daily-frequency equity data. 

Second, despite accurate modeling of jump dynamics, the strategy failed to generate risk-adjusted returns exceeding the risk-free rate. The best-performing pair (XOM/CVX) achieves an annualized return of 0.11% against a risk-free rate of approximately 2%, yielding a Sharpe ratio of -2.95.

Third, the Hawkes-based filtering mechanism, while conceptually sound, proved to be considerably over-restrictive in practice. The $\lambda$ decay requirement blocks the majority of potential trades, and for low-jump-frequency pairs, the intensity distribution is degenerate, rendering the Hawkes modeling inapplicable. 

Fourth, transaction costs represent the binding constraint on strategy profitability. With round-trip costs of approximately 6 basis points and average gross returns pre trade below this level, the strategy destroys value in expectation. 

## Potential Extensions
Several extensions may yield more promising results. First, application to higher-frequency data might reveal exploitable patterns that are arbitraged away at the daily frequency. The market microstructure literature documents strong self-excitation in order flow at millisecond frequencies, and these dynamics may be more directly tradeable. 

Second, application to alternative asset classes, particularly cryptocurrencies, commodities, or emerging market equities, may identify markets where competition is less intense and edges remain. The crypto market in particular exhibits high volatility, strong jump clustering, and potentially weaker arbitrage efficiency. 

Third, integration of the Hawkes framework with other signal sources may prove to be fruitful. Rather than using jump intensity for entry timing, it might be more effective to use it for position sizing and risk management while generating directional signals from fundamental or alternative data sources. 

Fourth, machine learning methods might be employed to extract more predictive information from the intensity process. Neural network architectures designed for point process data could potentially identify non-linear patterns in the relationship between intensity and future returns that the linear filtering approach misses. 

## Closing remarks
This project began with an intellectually appealing hypothesis: that the self-exciting nature of market jumps could be exploited to time pairs trading entries and exits. The hypothesis was grounded in genuine market phenomena; that volatility does cluster, jumps do beget jumps, and market stress is contagious. The mathematical foundation was elegant and the implementation was rigorous. 

Despite all of this, the strategy failed. The market proved more efficient than my model had anticipated. The pattern that I identified was real but not profitable. The edge was either absent or too small to capture. 

The outcome, while disappointing from a profit-and-loss perspective, represents a successful investigation. I learned that Hawkes processes, while valuable for characterizing jump dynamics, do not provide tradeable alpha in equity paris at the daily frequency level. This negative result may guide future researchers toward a more promising direction. 
