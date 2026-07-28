# Seasonality and Calendar Effects

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Replicate the classic calendar-effect tests (day-of-week, turn-of-month, January) on current data and compare against the original published findings.
- Design a seasonality-detection procedure that controls for multiple comparisons and separates genuine seasonal structure from mined noise.
- Build and evaluate an event-driven seasonal strategy around a recurring known event such as an index rebalance or expiry cycle.
- Assess whether a detected seasonal survives out-of-sample data and transaction costs before committing capital to it.

## Outline

1. The calendar-effect literature — what was claimed and what survived replication
2. Testing classic effects — day-of-week, turn-of-month, January on modern data
3. Detecting seasonality — dummy-variable regressions and spectral views
4. Data-mining traps — multiple comparisons and post-publication decay
5. Event-driven seasonals — earnings cycles, index rebalances, option expiries
6. Building a seasonal strategy — filters, sizing, and combination with other signals

## Prerequisites

- [Part III — Hypothesis Testing](../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md)
