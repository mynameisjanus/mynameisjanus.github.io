# Resilience and Risk Controls

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Implement crash recovery that reconciles internal position and order state against broker records on every restart
- Make order submission idempotent with client order IDs so retries and replays can never duplicate an order
- Implement pre-trade risk limits and circuit breakers that halt trading on anomalous prices, PnL, or order flow
- Design a kill switch that freezes or flattens the book, triggerable both by a human and automatically

## Outline

1. Crash recovery — restarting without guessing at state
2. State reconciliation — trusting the broker's records over your own
3. Idempotent order handling — client order IDs, deduplication, and safe retries
4. Health checks and watchdogs — detecting a wedged component
5. Circuit breakers — halting on anomalous conditions
6. Pre-trade risk limits — size, notional, concentration, and rate checks
7. Kill switches — manual and automatic paths to flat

## Prerequisites

- [Monitoring, Logging, Alerting](04-monitoring-logging-alerting.md)
