# Kelly, Volatility Targeting, and Leverage

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Derive the Kelly fraction for a strategy from its estimated return distribution, and justify a specific fractional-Kelly multiplier from estimation error and drawdown tolerance
- Implement a volatility-targeting overlay and quantify its effect on realized volatility, Sharpe ratio, and maximum drawdown
- Compute the all-in cost of leverage — financing spreads, securities borrow, margin requirements — for a given portfolio and broker arrangement
- Design a drawdown-controlled sizing rule with explicit cut and re-entry thresholds, and analyze its effect on recovery time

## Outline

1. The Kelly criterion — growth-optimal sizing, its derivation, and its assumptions
2. Fractional Kelly — parameter uncertainty, overbetting asymmetry, choosing the fraction
3. Volatility targeting — mechanics, estimator choice, rebalancing frequency
4. Leverage mechanics — margin, financing rates, securities borrow, portfolio margining
5. The cost of leverage — funding spreads, forced deleveraging, gap risk
6. Drawdown-controlled sizing — cut rules, ratchets, and re-entry logic
7. A sizing policy for the book — combining Kelly, vol targets, and drawdown controls

## Prerequisites

- [Risk Measurement](01-risk-measurement.md)
