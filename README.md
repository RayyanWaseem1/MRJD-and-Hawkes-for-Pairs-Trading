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

$dSt​=κ(θ−St​)dt+σdWt​+Jt​dNt​$

where κ\kappa
κ represents the speed of mean reversion, θ\theta
θ is the long-run equilibrium level, σ\sigma
σ captures diffusive volatility, JtJ_t
Jt​ is the random jump size, and NtN_t
Nt​ is a counting process governing jump arrivals. The innovation in my approach is to model NtN_t
Nt​ not as a simple Poisson process but as a Hawkes process with time-varying intensity.

## Pairs Trading and Statistical Arbitrage 
Pairs trading has a long history in quantitative finance. The application of cointegration methods to pairs trading provides a rigorous statistical foundation for identifying tradeable relationships. When two assets are cointegrated, their prices share a common stochastic trend, and any deviation from the equilibrium relationship is expected to be only temporary. However, as my results demonstrate, statistical cointegration is necessary, but not a sufficient condition for profitable trading. The edge must be large enough to survive the transaction costs and must not be arbitraged away by any competing strategies. 

# Mathematical Framework 
## The Hawkes Process
I began with a formal definition of the Hawkes self-exciting point process. Let *Nt​* denote a counting process representing the cumulative number of jumps up to time *t*. The Hawkes process is characterized by a conditional intensity function λ(t)\lambda(t)
*λ(t)* that depends on the history of the process: 

$λ(t)=λ0​+ti​<t∑​ϕ(t−ti​)$