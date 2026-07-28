# Capstone 6 — Paper Trading System Connected to a Broker API

!!! warning "Under development"
    Full project specifications, datasets, and grading rubrics are being written.
    The brief below defines the project's scope.

You will run a strategy live against a broker's paper-trading API as a scheduled, stateful, monitored service — not a script you run by hand. This is the first capstone where the market pushes back: sessions expire, orders partially fill, processes crash mid-run, and live behavior drifts from backtest assumptions. The project is graded on operational discipline — the system must survive restarts without losing state, and you must be able to explain every difference between what the backtest predicted and what the paper account did.

## Objectives

- Integrate a broker paper-trading API: authentication, market data, order submission, and position reconciliation
- Run the strategy on a schedule with state persistence, so that a process restart resumes cleanly with no duplicate orders
- Monitor the system: health checks, fill notifications, and alerts on errors or missed runs
- Reconcile live paper results against backtest expectations and account for every discrepancy

## Deliverables

- A running paper-trading service under a scheduler (cron, systemd timer, or equivalent) with documented deployment steps
- A state persistence layer recording positions, open orders, and run history, with demonstrated crash-recovery behavior
- Monitoring output: an alerting configuration and a basic system-health view
- A live-run report covering at least two weeks of operation, reconciling paper fills against simulated fills

## You will use

- [Part VI](../part-06-live-infrastructure/index.md)
