# Portfolio Optimization and Correlation

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Implement mean-variance optimization and demonstrate empirically how estimation error in expected returns and covariances corrupts the resulting weights
- Apply shrinkage estimators (Ledoit–Wolf and structured targets) to a sample covariance matrix and measure the out-of-sample improvement
- Quantify correlation instability across market regimes and stress an optimized portfolio under alternative correlation assumptions
- Build a constrained optimization with position, turnover, and risk-budget limits, and justify each constraint as insurance against estimation error

## Outline

1. Mean-variance optimization — the formulation and the error-maximization problem
2. Where the errors bite — expected returns versus covariances
3. Shrinkage estimators — Ledoit–Wolf, structured targets, factor-model covariances
4. Correlation instability — regime shifts and correlations going to one in crises
5. Robust and constrained optimization — bounds, turnover penalties, risk budgets
6. Practical alternatives — inverse-volatility weighting, hierarchical risk parity
7. Rebalancing in practice — frequency, transaction costs, and drift bands

## Prerequisites

- [Risk Parity, Diversification, and Factors](03-risk-parity-diversification-factors.md)
