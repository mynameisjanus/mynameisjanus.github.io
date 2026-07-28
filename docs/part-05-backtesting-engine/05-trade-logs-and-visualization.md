# Trade Logs and Visualization

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Emit structured trade logs from the engine that allow every position and PnL figure to be reconstructed from fills alone
- Pair fills into round trips and analyze holding-period and per-trade PnL distributions
- Produce equity, drawdown, and exposure plots that tie out exactly to the tearsheet numbers
- Compare multiple backtest runs on aligned overlay plots and identify where their behavior diverges

## Outline

1. Structured trade logs — schema, serialization, and replayability
2. From fills to round trips — pairing logic and edge cases
3. Round-trip analysis — holding periods, PnL distributions, best and worst trades
4. Equity and drawdown plots
5. Exposure and position plots over time
6. Comparing runs — overlays, difference views, and run metadata

## Prerequisites

- [Performance Metrics and Reporting](04-performance-metrics-and-reporting.md)
- [Part II — Plotting](../part-02-python/06-plotting.md)
