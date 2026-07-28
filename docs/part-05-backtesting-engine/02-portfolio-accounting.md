# Portfolio Accounting

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Implement FIFO position accounting that survives partial fills, position reversals, and short positions
- Maintain a cash ledger and mark-to-market equity curve that reconcile after every fill, and assert the accounting invariants that catch drift
- Apply stock splits and dividends to positions and history without distorting realized or unrealized PnL
- Mark a multi-currency book to market in a single base currency using per-instrument FX conversion

## Outline

1. The position object — quantity, direction, and open lots
2. Average cost vs FIFO lot accounting — where they diverge and why FIFO wins for trade analysis
3. The cash ledger — fills, commissions, and financing
4. Mark-to-market and the equity curve
5. Corporate actions — splits, dividends, and adjusting open positions
6. Multi-currency books — FX conversion and base-currency equity
7. Reconciliation — accounting invariants and self-checks after every event

## Prerequisites

- [Architecture and Event-Driven Design](01-architecture-and-event-driven-design.md)
