# Part III — Statistics for Trading

Statistics is where most trading courses go superficial, and this one does not. The standard treatment hand-waves past the assumptions — returns are treated as normal and independent, t-statistics are quoted without checking what breaks them, and dozens of strategy variants are tested with no accounting for multiple comparisons. Those shortcuts are precisely how backtests that look brilliant on paper lose money in production. This part teaches the statistics a working quantitative researcher actually needs, with the failure modes front and center.

Every lesson analyzes real market data. Return distributions are fitted to actual index and futures returns, not simulated toy series; autocorrelation, volatility clustering, and cointegration are estimated on data you download and clean yourself; hypothesis tests are run on genuine strategy output where autocorrelation and fat tails are facts to be handled, not assumed away. The goal is not textbook fluency but judgment: knowing which test applies, which assumption is violated, and how much to trust a number.

The lessons here are applied by design. The formal foundations — probability spaces, distribution theory, estimators, limit theorems — live in the [Mathematical Prerequisites](../appendix/index.md) appendix, and each lesson links the exact pages it leans on. If a derivation in the appendix is unfamiliar, read it before the lesson; the lessons assume that material and spend their time on market data instead.

## Modules

| Module | Focus |
|---|---|
| [Probability and Random Variables](01-probability-and-random-variables.md) | Probability for trading, random variables, moments, and dependence |
| [Returns and Their Distributions](02-returns-and-distributions.md) | Simple vs log returns, stylized facts, and fitting fat-tailed distributions |
| [Time Series Analysis](03-time-series.md) | Stationarity, ACF/PACF, ARIMA, GARCH, and cointegration |
| [Hypothesis Testing and Multiple Testing](04-hypothesis-testing-and-multiple-testing.md) | Tests on strategy returns and the multiple-testing problem |
| [Bootstrap and Monte Carlo Methods](05-bootstrap-and-monte-carlo.md) | Resampled confidence intervals, strategy simulation, permutation tests |
| [Bayesian Methods and Hidden Markov Models](06-bayesian-methods-and-hmms.md) | Bayesian estimation, shrinkage, and regime detection with HMMs |
