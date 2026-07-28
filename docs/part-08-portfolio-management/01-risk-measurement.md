# Risk Measurement

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Estimate ex-ante portfolio volatility using close-to-close, EWMA, and range-based estimators, and reconcile the estimates against realized outcomes
- Compute parametric, historical, and Monte Carlo VaR and expected shortfall for a book, and state the assumptions each method depends on
- Decompose book-level risk into gross/net, asset-class, and factor exposures and identify the dominant contributors
- Quantify the gap between a strategy's standalone risk and its marginal contribution to book risk using component VaR and marginal risk contributions

## Outline

1. Volatility estimation — close-to-close, EWMA, range-based estimators, choosing a lookback
2. Value at Risk — parametric, historical, and Monte Carlo approaches
3. Expected shortfall — why it is coherent, how to estimate it, backtesting the tail
4. Exposure decomposition — gross/net, sector, currency, and factor views of the book
5. Strategy risk versus book risk — marginal and component contributions
6. Aggregation pitfalls — correlation assumptions, horizon scaling, non-normality
7. The one-page daily risk report — what a PM actually reads each morning

## Prerequisites

- [Part III — Returns and Distributions](../part-03-statistics/02-returns-and-distributions.md)
