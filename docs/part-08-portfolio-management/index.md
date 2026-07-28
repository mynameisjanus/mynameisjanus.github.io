# Part VIII — Portfolio Management

Strategies do not run in isolation. A signal that looks attractive on its own can add nothing — or worse, add concentrated risk — once it sits alongside everything else you trade. This part is about running a book: measuring the risk you actually hold, sizing positions so a bad month does not become a terminal one, allocating capital across strategies, and surviving the tails that the average backtest never shows you.

The treatment here is deliberately practitioner-first. We spend as much time on why textbook tools fail — mean-variance optimizers that maximize estimation error, correlations that converge to one exactly when diversification is needed — as on the tools themselves. Sizing is treated as a policy decision with explicit trade-offs (fractional Kelly, volatility targets, drawdown controls), not a formula to plug in.

By the end of this part you should be able to produce a daily risk picture of a multi-strategy book, defend a sizing and leverage policy in front of a risk committee, and stress the whole construction against the scenarios that actually break portfolios.

## Modules

| Module | Focus |
|---|---|
| [Risk Measurement](01-risk-measurement.md) | Volatility, VaR, expected shortfall, and decomposing the risk of a book versus its strategies |
| [Kelly, Volatility Targeting, and Leverage](02-kelly-vol-targeting-leverage.md) | Sizing policy: fractional Kelly, vol targeting, the mechanics and costs of leverage, drawdown-controlled sizing |
| [Risk Parity, Diversification, and Factors](03-risk-parity-diversification-factors.md) | Why correlation is the whole game, equal-risk allocation, and the factor exposures hiding in trading strategies |
| [Portfolio Optimization and Correlation](04-portfolio-optimization-and-correlation.md) | Mean-variance and its fragility, shrinkage estimators, correlation instability, robust optimization in practice |
| [Drawdowns, Tail Risk, and Stress Testing](05-drawdowns-tail-risk-stress-testing.md) | Drawdown statistics and psychology, tail measurement, scenario analysis, and hedging the left tail |
