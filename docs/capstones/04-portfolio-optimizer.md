# Capstone 4 — Portfolio Optimizer with Risk Constraints

!!! warning "Under development"
    Full project specifications, datasets, and grading rubrics are being written.
    The brief below defines the project's scope.

You will build a portfolio optimizer that combines multiple strategies — including the ones you built in Capstones 1–3 — into a single book under hard risk constraints: a volatility target, a maximum drawdown budget, and gross/net exposure limits. Single strategies do not run in isolation at real firms; capital allocation across them, done under estimation error, is where much of the practical difficulty lives. The optimizer must be a tested library, not a notebook, because Capstones 6 and 7 will call it in production.

## Objectives

- Estimate the covariance structure of strategy returns robustly (shrinkage or factor-based) and quantify the impact of estimation error on allocations
- Implement an optimizer that enforces a volatility target, a drawdown budget, and exposure limits as hard constraints
- Compare allocation schemes — equal weight, inverse volatility, risk parity, constrained mean-variance — out of sample
- Verify constraint enforcement under stress, including correlated drawdowns across strategies

## Deliverables

- An optimizer library with a documented API, explicit constraint handling, and unit tests for every constraint
- An out-of-sample backtest of the combined portfolio against naive baseline allocations
- A stress-test report showing constraint behavior in adverse scenarios (correlation spikes, single-strategy failure)
- A short design document explaining the chosen allocation scheme and its failure modes

## You will use

- [Part VIII](../part-08-portfolio-management/index.md)
