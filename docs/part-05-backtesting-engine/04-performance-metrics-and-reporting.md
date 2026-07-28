# Performance Metrics and Reporting

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Derive simple and log return series from an equity curve and annualize them correctly for the bar frequency of the backtest
- Implement Sharpe, Sortino, and Calmar ratios and state the assumptions and failure modes of each
- Compute drawdown depth, duration, and recovery statistics directly from the equity curve
- Generate a tearsheet report that combines turnover, exposure, hit rate, and risk-adjusted returns for any engine run

## Outline

1. From equity curve to return series — simple vs log returns
2. Annualization — frequency conventions and common mistakes
3. Risk-adjusted ratios — Sharpe, Sortino, Calmar
4. Drawdown statistics — depth, duration, and time to recovery
5. Turnover and exposure — gross, net, and time-in-market
6. Trade-level statistics — hit rate, payoff ratio, expectancy
7. The tearsheet — assembling a one-page report from the engine

## Prerequisites

- [Portfolio Accounting](02-portfolio-accounting.md)
