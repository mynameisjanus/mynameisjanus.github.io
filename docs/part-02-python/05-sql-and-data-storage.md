# SQL and Data Storage

Every quant eventually re-downloads the same data because they stored it badly the first time. That is the cheap failure. The expensive one is subtler: most casually stored datasets silently rewrite history — a restated earnings number overwrites the original, a delisted symbol vanishes from the universe — so yesterday's query no longer returns yesterday's answer, and a backtest run on such data is fiction with confidence intervals. Storage is not plumbing; it is where research honesty is won or lost.

This lesson covers the storage stack a research operation actually uses — SQL for structured queries, SQLite growing into PostgreSQL, Parquet for columnar scans — and ends with the property that separates professional data from hobbyist data: point-in-time correctness.

## SQL on price data

SQL earns its keep in research the moment you meet **window functions**, which compute per-row statistics over ordered partitions — returns, moving averages, ranks — inside the database, before a single row crosses the wire into Python. The snippets here use the standard library's `sqlite3` with an in-memory database, so every query actually runs.

```python
import sqlite3
import numpy as np

rng = np.random.default_rng(42)
con = sqlite3.connect(":memory:")
con.execute("""CREATE TABLE bars (
    symbol TEXT NOT NULL, ts TEXT NOT NULL, close REAL NOT NULL,
    PRIMARY KEY (symbol, ts))""")

for sym in ["AAA", "BBB", "CCC"]:
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 25)))
    con.executemany(
        "INSERT INTO bars VALUES (?, ?, ?)",
        [(sym, f"2024-01-{d + 1:02d}", round(float(c), 2))
         for d, c in enumerate(closes)])

for row in con.execute("""
    SELECT ts, close,
           ROUND(close / LAG(close) OVER (PARTITION BY symbol ORDER BY ts) - 1,
                 5) AS ret,
           ROUND(AVG(close) OVER (PARTITION BY symbol ORDER BY ts
                                  ROWS 19 PRECEDING), 2) AS sma20
    FROM bars WHERE symbol = 'AAA' ORDER BY ts LIMIT 3"""):
    print(row)
# ('2024-01-01', 100.36, None, 100.36)
# ('2024-01-02', 99.37, -0.00986, 99.87)
# ('2024-01-03', 100.17, 0.00805, 99.97)
```

`LAG` reaches back one row within the symbol's partition — a return calculation — and the framed `AVG` is a 20-bar moving average. The first close, 100.36, is the same number the seeded generator produced in [NumPy and Vectorization](01-numpy-and-vectorization.md): same data, third representation. One semantic difference from pandas deserves attention: SQL happily averages a *partial* window (the first row's "sma20" is just its own close), where pandas' `min_periods` would give you NaN. If you require full windows, filter on a `ROW_NUMBER()` or count — the database will not volunteer the distinction.

Plain `GROUP BY` aggregation, joins against reference tables, and filtering push the same philosophy: do the data reduction where the data lives, and bring back only what research needs.

## Schema design for OHLCV

A bar is uniquely identified by what and when — so say exactly that, and let the database enforce it:

```sql
CREATE TABLE bars (
    symbol  TEXT             NOT NULL,
    ts      TIMESTAMPTZ      NOT NULL,
    open    DOUBLE PRECISION NOT NULL,
    high    DOUBLE PRECISION NOT NULL,
    low     DOUBLE PRECISION NOT NULL,
    close   DOUBLE PRECISION NOT NULL,
    volume  BIGINT           NOT NULL,
    PRIMARY KEY (symbol, ts)
) PARTITION BY RANGE (ts);

CREATE TABLE bars_2024_01 PARTITION OF bars
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- the primary key serves symbol-then-time; index the other direction too:
CREATE INDEX bars_ts_symbol ON bars (ts, symbol);
```

The composite primary key does double duty: it makes duplicate bars a constraint violation at insert time (your first line of defense against a vendor resending a day), and its index serves the most common research query. Index design is nothing more than listing your dominant query shapes and making sure each has a path:

| Query shape | Served by |
|---|---|
| One symbol, a range of dates (a price history) | `PRIMARY KEY (symbol, ts)` |
| All symbols on one date (a cross-section) | `INDEX (ts, symbol)` |
| Latest bar per symbol (a live snapshot) | either, via a window function |

Partitioning by month keeps any single physical table small, makes dropping or reloading a bad month an instant operation, and lets the planner skip whole partitions when a query names a date range. On price representation: `DOUBLE PRECISION` is the pragmatic default for research — you learned its 16-digit limits in lesson 01 — while systems that *account* for money (fills, positions, PnL) use exact types like `NUMERIC` or integer ticks; that distinction returns in the execution parts of the course.

## Tick data is a different animal

The schema above assumes bar-sized volumes. Ticks are three to six orders of magnitude past that, and intuition built on daily bars simply does not transfer:

| Data | Rows per symbol-day | 3,000 symbols × 1 year |
|---|---:|---:|
| Daily bars | 1 | ~750 thousand |
| 1-minute bars | 390 | ~300 million |
| Trades | tens of thousands | tens of billions |
| Quotes (top of book) | hundreds of thousands | hundreds of billions |
| Full order-book events | millions | trillions |

At that scale the design questions change. Granularity: do you need every quote, or one-second snapshots? Retention: full depth for the last quarter and trades-only history is a common compromise. And layout: a row-oriented database pays its per-row overhead billions of times, while columnar storage compresses beautifully — timestamps are nearly sequential, prices change by ticks, symbols repeat — which is precisely the case for the Parquet section below or a dedicated column store. The strategic advice for this course: know the numbers in this table, store ticks columnar when you must store them, and do not build order-book infrastructure before a strategy needs it — [Market Microstructure](../part-01-foundations/03-market-microstructure.md) tells you what the book would give you; most strategies in this course are built and tested on bars.

## SQLite to PostgreSQL

SQLite is a full SQL engine in a single file, ships inside Python, and is the right first database for a research project — everything in the first section ran on it. It has one writer at a time and lives on your disk. You graduate to PostgreSQL when a real limit bites: concurrent writers (a downloader appending while research reads), remote access from more than one machine, datasets pushing past memory, or the need for real types — SQLite's easygoing typing quietly accepts a string where a number belongs, and `TIMESTAMPTZ` alone prevents a class of timezone bugs you met in [Pandas and Polars](02-pandas-and-polars.md).

The migration itself is undramatic — dump, create the strict schema, bulk-load:

```text
sqlite3 research.db ".mode csv" ".once bars.csv" "SELECT * FROM bars;"
psql quant -c "\copy bars FROM 'bars.csv' WITH (FORMAT csv)"
```

Two practical notes. Use `COPY` (or `\copy`), not row-by-row `INSERT` — it is orders of magnitude faster for bulk loads, and it is also how your daily downloader should append. And treat the migration as a data-quality audit: the stricter types and constraints will reject rows SQLite accepted, and every rejection is a bug you already had.

## Parquet and columnar files

Not everything belongs in a database. **Parquet** stores a table column-by-column in compressed row groups, each carrying min/max statistics — so a reader can skip whole chunks without decompressing them (**predicate pushdown**) and read only the columns a query names (**projection**). This is the `scan_parquet` payoff promised in the Polars section of lesson 02:

```python
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

rng = np.random.default_rng(42)
root = Path(tempfile.mkdtemp())
pl.DataFrame({
    "symbol": ["AAA"] * 252 + ["BBB"] * 252 + ["CCC"] * 252,
    "day": list(range(252)) * 3,
    "close": 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 756))),
}).write_parquet(root / "bars.parquet")

lf = (pl.scan_parquet(root / "bars.parquet")
      .filter(pl.col("symbol") == "AAA")
      .select("day", "close"))
print(lf.explain())
# simple π 2/2 ["day", "close"]
#   Parquet SCAN [.../bars.parquet]
#   PROJECT 3/3 COLUMNS
#   SELECTION: col("symbol") == "AAA"
print(lf.collect().shape)  # => (252, 2)
```

The plan says it plainly: the symbol filter became part of the *scan*, not a post-processing step. At research scale the convention is a partitioned directory tree — `bars/symbol=AAA/date=2024-01/…` — where the path itself is an index and a reader touches only the files a query implicates. Files win when the workload is append-only and scan-heavy, which describes most research: write each day once, read whole histories often, version datasets by directory. Databases win when you need transactions, constraints, concurrent writers, or fast point lookups. Mature setups use both — Parquet as the bulk store researchers scan, SQL for reference data and anything that must be *correct by construction* — which is exactly where the final section aims.

## Point-in-time correctness

Here is the property that makes historical research trustworthy: **a query about the past must return what you would have known at the time — and keep returning it forever.** Two biases follow from violating it. Survivorship: building a universe from symbols that exist *today* silently excludes everything that delisted, and [Why Most Retail Strategies Fail](../part-01-foundations/10-why-most-retail-strategies-fail.md) shows how flattering the resulting backtests look. Restatement: overwriting a reported number with its corrected value hands your backtest information nobody had on the day.

The cure is structural, not procedural: never update, only append, and record *when you learned* each fact. That second timestamp — the knowledge time — turns "what is AAA's Q4 EPS?" into the answerable "what did we know about AAA's Q4 EPS as of date X?":

```python
import sqlite3

con = sqlite3.connect(":memory:")
con.execute("""CREATE TABLE eps (
    symbol TEXT, period TEXT, eps REAL, knowledge_time TEXT,
    PRIMARY KEY (symbol, period, knowledge_time))""")

con.executemany("INSERT INTO eps VALUES (?, ?, ?, ?)", [
    ("AAA", "2023Q4", 1.21, "2024-01-25"),   # the initial report
    ("AAA", "2023Q4", 0.94, "2024-03-08"),   # the restatement
])

def eps_asof(asof: str) -> float | None:
    row = con.execute("""
        SELECT eps FROM eps
        WHERE symbol = 'AAA' AND period = '2023Q4' AND knowledge_time <= ?
        ORDER BY knowledge_time DESC LIMIT 1""", (asof,)).fetchone()
    return row[0] if row else None

print(eps_asof("2024-01-10"))  # => None — not yet reported
print(eps_asof("2024-02-01"))  # => 1.21 — what the market knew then
print(eps_asof("2024-04-01"))  # => 0.94 — after the restatement
```

The same question gets three different answers depending on when it is asked — and every one of them is correct, because each reflects the information actually available on that date. This as-of query is the SQL twin of `merge_asof` from lesson 02: `direction="backward"` on knowledge time. The pattern generalizes to universe membership, index constituents, corporate actions, and vendor data corrections; a `valid_until` column or a current-flag view keeps the common "latest known value" query cheap.

!!! warning "If a query's answer can change retroactively, your backtests are fiction"
    Any table your research reads should be append-only with knowledge timestamps, or derived from one that is. The test is unforgiving: rerun last month's query today — byte-identical results or the dataset is not research-grade. Apply it to anything a vendor can restate, revise, or backfill, which in practice means almost everything except the raw prints themselves.

!!! abstract "Key takeaways"
    - Window functions (`LAG`, framed `AVG`) compute returns and rolling statistics inside the database — reduce data where it lives, and mind that SQL averages partial windows where pandas gives NaN.
    - `PRIMARY KEY (symbol, ts)` enforces uniqueness and serves history queries; add the `(ts, symbol)` index for cross-sections, and partition by date range.
    - Tick data is orders of magnitude beyond bars — know the scale table, store it columnar, and do not build book infrastructure before a strategy demands it.
    - Start on SQLite; graduate to PostgreSQL for concurrent writers, remote access, and strict types, and bulk-load with `COPY`.
    - Parquet's row-group statistics turn filters into skipped reads — the file format is the index; use files for append-only scan workloads and SQL where constraints must hold.
    - Point-in-time correctness means append-only tables with knowledge timestamps and as-of queries — the same answer to the same question, forever.

## Where this goes next

You can now compute, structure, fetch, and store market data trustworthily. What remains is looking at it — and being able to defend what you show. [Plotting for Research](06-plotting.md) treats charts as arguments rather than decoration.
