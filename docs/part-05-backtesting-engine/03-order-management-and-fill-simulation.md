# Order Management and Fill Simulation

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Implement an order state machine covering the full lifecycle — created, submitted, partially filled, filled, cancelled, rejected — with only legal transitions
- Simulate market, limit, and stop orders against bar data with fill assumptions you can defend
- Implement next-bar-open, mid-price, and spread-plus-impact fill models and quantify how the choice changes a strategy's backtest PnL
- Handle partial fills so that position, cash, and open-order state remain consistent at every step

## Outline

1. The order lifecycle — states, legal transitions, and the state machine
2. Order types — market, limit, stop, and stop-limit against bar data
3. Fill model I — next-bar open
4. Fill model II — mid-price with spread costs
5. Fill model III — spread plus market impact
6. Partial fills and residual order management
7. The simulated broker — assembling lifecycle, fill models, and rejections
8. Sensitivity analysis — how fill assumptions move the results

## Prerequisites

- [Portfolio Accounting](02-portfolio-accounting.md)
