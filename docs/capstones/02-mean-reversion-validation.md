# Capstone 2 — Mean-Reversion Strategy with Statistical Validation

!!! warning "Under development"
    Full project specifications, datasets, and grading rubrics are being written.
    The brief below defines the project's scope.

You will build a mean-reversion strategy and then defend it with formal statistical validation: hypothesis tests on the performance claim, bootstrap confidence intervals on the key metrics, and explicit control for the multiple testing you did while developing it. The strategy itself is the easy half. The graded skill is distinguishing a real effect from noise — writing down your hypotheses before evaluation, quantifying uncertainty honestly, and reaching a defensible deploy-or-reject decision.

## Objectives

- Build a time-series or cross-sectional mean-reversion strategy with a documented economic rationale
- State the performance claim as a testable hypothesis and test it with appropriate methods for autocorrelated, non-normal returns
- Compute bootstrap confidence intervals for Sharpe ratio and drawdown statistics
- Account for every variant you tried: apply a multiple-testing correction (e.g., deflated Sharpe ratio) across the full search history

## Deliverables

- Strategy and backtest code with a log of every configuration evaluated during development
- A pre-registration document: hypotheses, evaluation metrics, and decision thresholds written before the final evaluation run
- A statistical validation report: test results, bootstrap confidence intervals, and multiple-testing-adjusted significance
- A one-page verdict memo: deploy, iterate, or reject — with the statistical reasoning

## You will use

- [Part III](../part-03-statistics/index.md)
- [Part IV](../part-04-strategy-development/index.md)
