# Part II — Python for Quantitative Finance

This is not a general Python course. There are hundreds of those, and almost none of them prepare you for quantitative work. This part covers only what quantitative research and trading systems actually need: fast numerical computation, time-series manipulation at scale, disciplined code structure, asynchronous market data access, durable storage with point-in-time correctness, and the logging and configuration habits that make research reproducible. Anything a working quant does not routinely touch — web frameworks, GUI toolkits, generic scripting trivia — is deliberately absent.

The treatment is opinionated in the way professional desks are opinionated. Vectorization is not a performance nicety; it is the difference between a research iteration taking seconds or hours. Type hints and dataclasses are not bureaucracy; they are how research code survives contact with a second person, including your future self. Every lesson works with market data from the first example, so the idioms you learn are the ones you will use daily: aligning timestamps, resampling bars, handling NaNs deliberately, and storing data so that yesterday's query gives yesterday's answer.

By the end of this part you will have a working research stack — NumPy, Pandas/Polars, SQL, plotting, and a project structure with logging and configuration — that the statistics and strategy parts of the course build on directly. The mini-projects below stitch the individual lessons into an end-to-end pipeline.

## Modules

| Module | Focus |
|---|---|
| [NumPy and Vectorization](01-numpy-and-vectorization.md) | Arrays, broadcasting, vectorized computation, and numerical pitfalls |
| [Pandas and Polars](02-pandas-and-polars.md) | Time-series DataFrames, resampling, rolling windows, joins, and lazy frames |
| [Typing, Dataclasses, and Code Structure](03-typing-dataclasses-structure.md) | Type hints, dataclasses, enums, and structuring research code with mypy |
| [Async and Market Data APIs](04-async-and-apis.md) | async/await, REST vs streaming data, rate limits, and retries |
| [SQL and Data Storage](05-sql-and-data-storage.md) | Schema design for market data, SQLite to PostgreSQL, Parquet, point-in-time correctness |
| [Plotting for Research](06-plotting.md) | Equity curves, drawdowns, distributions, heatmaps, and IC-ready charts |
| [Logging, Configuration, and Reproducibility](07-logging-and-config.md) | Structured logging, config management, and reproducible runs |

## Mini-projects

- **Market data downloader** — download historical market data from a public API, with rate limiting and retry handling.
- **Vectorized indicators** — compute a set of standard indicators over a large universe using vectorized NumPy/Pandas, with no Python loops.
- **OHLCV database** — design a schema and store OHLCV data in SQL with appropriate keys and indexes.
- **Query and resample** — query historical data from the database and resample it to multiple bar frequencies without lookahead.
