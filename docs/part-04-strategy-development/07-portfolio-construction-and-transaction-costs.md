# Portfolio Construction and Transaction Costs

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Combine multiple strategies into a single portfolio and quantify the diversification and netting benefit versus running them separately.
- Specify realistic execution assumptions for a backtest — fill logic, latency, participation — and show how each assumption moves reported performance.
- Calibrate a transaction cost model covering spread, commissions, market impact, and slippage from available data.
- Implement cost-aware rebalancing using no-trade bands or turnover penalties and measure the net-of-cost improvement.

## Outline

1. Combining strategies — correlation across sleeves, allocation, aggregate risk
2. Netting — internal crossing, gross vs net exposure, the cost savings of a book
3. Execution assumptions — fills, latency, participation, and backtest realism
4. Transaction cost models — spread, commissions, market impact
5. Slippage — measurement, modeling, and calibrating the backtest to it
6. Cost-aware rebalancing — no-trade bands, turnover penalties, optimization
7. Net-of-cost evaluation — how costs reshape which strategies survive

## Prerequisites

- [Position Sizing and Risk Budgeting](06-position-sizing-and-risk-budgeting.md)
