# SQL and Data Storage

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Write SQL queries — joins, aggregations, and window functions — against a bar database to answer research questions.
- Design schemas for OHLCV and tick data with appropriate primary keys, indexes, and date partitioning.
- Decide between SQLite, PostgreSQL, and Parquet files for a given workload and migrate a research database from SQLite to PostgreSQL.
- Build point-in-time-correct tables and as-of queries that eliminate survivorship and restatement bias from historical research.

## Outline

1. SQL fundamentals — SELECT, joins, aggregation, window functions on price data
2. Schema design for OHLCV — keys, indexes, partitioning by date and symbol
3. Tick data storage — volume, granularity, and compression trade-offs
4. SQLite to PostgreSQL — when to graduate and how to migrate
5. Parquet and columnar storage — file layout, predicate pushdown, when files beat databases
6. Point-in-time correctness — as-of queries, restatements, survivorship bias

## Prerequisites

- [Pandas and Polars](02-pandas-and-polars.md)
