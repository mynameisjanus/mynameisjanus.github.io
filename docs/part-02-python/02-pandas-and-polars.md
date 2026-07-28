# Pandas and Polars

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Index, slice, and align time-series DataFrames using a DatetimeIndex, including timezone handling and partial-date selection.
- Resample intraday data to arbitrary OHLCV bar frequencies and compute rolling-window statistics without introducing lookahead.
- Join price, reference, and fundamental datasets with correct as-of semantics and verify alignment.
- Translate a Pandas pipeline into a Polars lazy query and identify workloads where Polars outperforms Pandas.

## Outline

1. DataFrame and Series fundamentals — construction, dtypes, memory footprint
2. Time-series indexing — DatetimeIndex, timezones, date-based slicing
3. Resampling — OHLCV aggregation rules, bar boundaries, label alignment
4. Rolling and expanding windows — moving statistics, min_periods, lookahead traps
5. Joins and merges — inner/outer joins, merge_asof, alignment verification
6. Polars lazy frames — expressions, query plans, predicate pushdown
7. Pandas vs Polars — benchmarks, when each wins, migration patterns

## Prerequisites

- [NumPy and Vectorization](01-numpy-and-vectorization.md)
