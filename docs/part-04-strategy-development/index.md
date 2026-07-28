# Part IV — Strategy Development

This is the largest part of the course, and deliberately so: strategy development is where every earlier skill — the Python stack, the statistical toolkit — gets spent. It is also where most self-taught quants fail, not because their ideas are bad but because their process is. The difference between a professional research process and indicator tinkering is method: a hypothesis stated before the backtest, an economic rationale for why the effect should exist, and a pre-committed standard for what would falsify it.

Every strategy in this part is developed hypothesis-first. We state the market behavior we expect and why, translate it into signals and positions, and only then run the backtest — and every strategy must survive (or visibly fail) a validation gauntlet: sensitivity analysis, cost-aware evaluation, multiple-testing corrections, and the overfitting diagnostics of the capstone lesson. Failed strategies are kept in the course on purpose; watching an idea die honestly is more instructive than another curve-fit success story.

By the end of this part, students will have built multiple complete strategies — trend following, pairs trading, cross-sectional and volatility books, and seasonal systems — each with position sizing, portfolio construction, and transaction costs treated as first-class design decisions rather than afterthoughts. The capstone lesson on validation and overfitting supplies the gates every one of those strategies must pass before it earns the right to be taken further.

## Modules

| Module | Focus |
|---|---|
| [Momentum and Trend Following](01-momentum-and-trend-following.md) | Time-series momentum, trend filters, breakouts, and why momentum persists |
| [Mean Reversion and Pairs Trading](02-mean-reversion-and-pairs-trading.md) | Mean-reversion tests, Ornstein-Uhlenbeck intuition, cointegrated spreads |
| [Cross-Sectional and Volatility Strategies](03-cross-sectional-and-volatility-strategies.md) | Ranking-based long-short portfolios, volatility risk premium, term structure |
| [Seasonality and Calendar Effects](04-seasonality-and-calendar-effects.md) | Calendar effects, seasonality detection vs data mining, event-driven seasonals |
| [Feature and Signal Engineering](05-feature-and-signal-engineering.md) | Raw data to signals, information coefficient, decay, and combination |
| [Position Sizing and Risk Budgeting](06-position-sizing-and-risk-budgeting.md) | Signal-to-position mapping, volatility scaling, risk budgets, capacity |
| [Portfolio Construction and Transaction Costs](07-portfolio-construction-and-transaction-costs.md) | Combining strategies, netting, cost models, and cost-aware rebalancing |
| [Validation and Overfitting](08-validation-and-overfitting.md) | Walk-forward, purged CV, Reality Check, PBO, and the Deflated Sharpe Ratio |
