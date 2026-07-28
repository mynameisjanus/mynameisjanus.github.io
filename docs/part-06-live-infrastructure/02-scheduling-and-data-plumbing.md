# Scheduling and Data Plumbing

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Schedule trading jobs with cron and APScheduler that respect market calendars, half-days, holidays, and daylight-saving transitions
- Use Redis as the hot state store and pub/sub bus between live components, and justify what belongs in Redis versus PostgreSQL
- Design PostgreSQL schemas for durable storage of orders, fills, positions, and job runs
- Build a data pipeline that lands and validates vendor data in the live system before the trading job that depends on it fires

## Outline

1. Job scheduling — cron fundamentals and APScheduler in practice
2. Market calendars — sessions, holidays, half-days, and timezones
3. Redis as the hot state store — keys, TTLs, and what belongs there
4. Redis pub/sub — messaging between live components
5. PostgreSQL for durable storage — schemas for orders, fills, and positions
6. Data pipelines into the live system — ingestion, validation, and landing
7. Sequencing the trading day — job dependencies and late-data handling

## Prerequisites

- [Part II — SQL and Data Storage](../part-02-python/05-sql-and-data-storage.md)
