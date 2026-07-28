# Position Sizing and Risk Budgeting

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Map signal strength to position size using linear, stepped, and saturating schemes and articulate the trade-offs of each.
- Implement volatility scaling at the asset and portfolio level, including the choice of volatility estimator and its lag.
- Allocate risk budgets across strategies and assets, monitor realized versus allocated risk, and define rebalancing triggers.
- Estimate a strategy's capacity from liquidity, participation limits, and market-impact considerations.

## Outline

1. From signal to position — mapping functions, discretization, saturation
2. Volatility scaling — vol targeting, estimator choice, leverage implications
3. Risk budgets — per-asset and per-strategy allocation
4. Monitoring the budget — realized vs allocated risk, rebalancing triggers
5. Growth-optimal sizing — Kelly intuition and why full Kelly is a hazard
6. Capacity — liquidity, participation limits, impact-based capacity estimates
7. Sizing the course strategies — applying the framework to earlier lessons' systems

## Prerequisites

- [Feature and Signal Engineering](05-feature-and-signal-engineering.md)
