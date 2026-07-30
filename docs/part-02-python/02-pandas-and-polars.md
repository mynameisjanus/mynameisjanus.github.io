# Pandas and Polars

The hard part of market data is not the numbers — you handled those in [NumPy and Vectorization](01-numpy-and-vectorization.md). The hard part is the labels. A raw array does not know that 09:31 New York follows 09:30, that the market was closed for Thanksgiving, or that your two data vendors disagree about whether a bar's timestamp marks its open or its close. Misaligned labels are how lookahead sneaks into backtests, and lookahead is the most common way a beautiful research result dies on contact with live trading.

Pandas puts an index on top of NumPy's arrays and makes alignment the default behavior; Polars rebuilds the same idea on a modern parallel engine with a query optimizer. This lesson covers both — pandas as the daily driver, Polars for the workloads where pandas runs out of road.

## DataFrames as labeled arrays

A **Series** is an ndarray plus an index; a **DataFrame** is a dictionary of Series sharing one index. Everything from the previous lesson — dtypes, views, vectorized arithmetic — still applies, but operations now align on labels first. Here is the canonical object of this course: one trading day of one-minute OHLCV bars, built from the same seeded generator as before.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
idx = pd.date_range("2024-01-02 09:30", periods=390, freq="min",
                    tz="America/New_York")
close = 100.0 * np.exp(np.cumsum(rng.normal(0, 4e-4, 390)))
open_ = np.concatenate(([100.0], close[:-1]))
high = np.maximum(open_, close) + rng.uniform(0, 0.03, 390)
low = np.minimum(open_, close) - rng.uniform(0, 0.03, 390)
df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                   "volume": rng.integers(1_000, 50_000, 390)}, index=idx)

print(df.head(3).round(2))
#                              open    high     low   close  volume
# 2024-01-02 09:30:00-05:00  100.00  100.04  100.00  100.01    8951
# 2024-01-02 09:31:00-05:00  100.01  100.04   99.96   99.97    9591
# 2024-01-02 09:32:00-05:00   99.97  100.01   99.97  100.00    3152
```

Note that the frame is built in one shot from complete arrays — not row by row, which is the DataFrame equivalent of the Python loop you just learned to avoid. Dtypes deserve the same attention they got in NumPy, and the one addition that matters for market data is `category`: symbols, exchanges, and sector codes repeat endlessly, and storing each occurrence as a string wastes memory that a small integer code would not.

```python
import numpy as np
import pandas as pd

sym = pd.Series(np.repeat(["AAA", "BBB", "CCC"], 130))
print(sym.memory_usage(deep=True))                     # => 20412
print(sym.astype("category").memory_usage(deep=True))  # => 678
```

Thirty-to-one on a toy Series; on a universe of millions of rows the difference is whether the frame fits in memory at all.

## Time on the index

A `DatetimeIndex` is what turns a DataFrame into a time series, and the first decision it forces is timezones. The house rule for the rest of this course: **store timestamps timezone-aware, work in the exchange's timezone, convert to UTC at system boundaries**. The two operations people confuse are `tz_localize` — attach a zone to naive timestamps that were always implicitly in that zone — and `tz_convert`, which re-expresses an aware timestamp in another zone without changing the instant it names.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
idx = pd.date_range("2024-01-02 09:30", periods=390, freq="min",
                    tz="America/New_York")
close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 4e-4, 390))),
                  index=idx, name="close")

print(close.loc["2024-01-02 10"].shape)  # => (60,) — partial strings slice
print(close.between_time("09:30", "09:32").round(2))
# 2024-01-02 09:30:00-05:00    100.01
# 2024-01-02 09:31:00-05:00     99.97
# 2024-01-02 09:32:00-05:00    100.00
# Freq: min, Name: close, dtype: float64

print(close.index[0].tz_convert("UTC"))  # => 2024-01-02 14:30:00+00:00
naive = pd.Timestamp("2024-01-02 09:30")
print(naive.tz_localize("America/New_York"))  # => 2024-01-02 09:30:00-05:00
```

Partial-string indexing — `close.loc["2024-01-02 10"]` returning the whole ten o'clock hour — and `between_time` for session slicing are the idioms you will use daily. The payoff for keeping the index aware is that daylight-saving transitions, half days, and cross-market joins stop being sources of silent one-hour errors and become things the library refuses to get wrong on your behalf.

## Resampling to bars

Downsampling one-minute bars to five-minute bars is not one aggregation but five different ones, and getting any of them wrong produces bars that look plausible and are subtly fictional:

| Column | Rule |
|---|---|
| `open` | `first` |
| `high` | `max` |
| `low` | `min` |
| `close` | `last` |
| `volume` | `sum` |

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
idx = pd.date_range("2024-01-02 09:30", periods=390, freq="min",
                    tz="America/New_York")
close = 100.0 * np.exp(np.cumsum(rng.normal(0, 4e-4, 390)))
open_ = np.concatenate(([100.0], close[:-1]))
high = np.maximum(open_, close) + rng.uniform(0, 0.03, 390)
low = np.minimum(open_, close) - rng.uniform(0, 0.03, 390)
df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                   "volume": rng.integers(1_000, 50_000, 390)}, index=idx)

bars5 = df.resample("5min", label="left", closed="left").agg(
    {"open": "first", "high": "max", "low": "min",
     "close": "last", "volume": "sum"})
print(bars5.head(3).round(2))
#                              open    high    low   close  volume
# 2024-01-02 09:30:00-05:00  100.00  100.06  99.94   99.96   67021
# 2024-01-02 09:35:00-05:00   99.96   99.99  99.85   99.87  155867
# 2024-01-02 09:40:00-05:00   99.87  100.00  99.86  100.00  161183
```

`label` and `closed` decide which boundary names the bar and which side of the interval it owns — here the 09:30 bar covers 09:30:00 through 09:34:59. That convention is not cosmetic: it must match your execution assumption. If your strategy "trades on the 09:35 bar," you need to know whether the information in that bar was complete at 09:35:00 or only at 09:39:59, because the difference is five minutes of the future.

!!! note "Frequency aliases changed in pandas 2.2"
    The old offset aliases `"T"`, `"H"`, and `"M"` are deprecated in favor of `"min"`, `"h"`, and `"ME"`. Code you find in older books and answers will use the old spellings; write the new ones.

## Rolling windows without lookahead

Rolling statistics are the workhorse of signal construction, and `rolling` gets the mechanics right by default: the window ends at the current row and looks only backward. The two ways to break that guarantee are `center=True`, which openly centers the window on the future, and the subtler one — using the statistic computed *through* bar $t$ to make a decision *at* bar $t$, when in live trading it is only available after the bar closes. The fix is one `shift`.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
close = pd.Series(
    100.0 * np.exp(np.cumsum(rng.normal(0, 4e-4, 390))),
    index=pd.date_range("2024-01-02 09:30", periods=390, freq="min",
                        tz="America/New_York"), name="close")

mean20 = close.rolling(20, min_periods=20).mean()
std20 = close.rolling(20, min_periods=20).std()
z = (close - mean20.shift(1)) / std20.shift(1)  # stats known BEFORE bar t

print(mean20.isna().sum())     # => 19 — no statistic until a full window
print(z.iloc[20:23].round(2))
# 2024-01-02 09:50:00-05:00    0.31
# 2024-01-02 09:51:00-05:00   -0.26
# 2024-01-02 09:52:00-05:00    0.92
# Freq: min, Name: close, dtype: float64
```

The z-score here is $z_t = (P_t - \mu_{t-1}) / \sigma_{t-1}$: the price at $t$ measured against a mean and standard deviation that were fully computable one bar earlier. Insist on `min_periods` equal to the window — a "20-bar average" computed from 3 bars is a different, noisier statistic wearing the same name — and let the leading NaNs stand; they are the honest statement that no valid window exists yet. What these rolling moments do and do not summarize is treated properly in the appendix's [Descriptive Statistics](../appendix/part-10-statistics-foundations/02-descriptive-statistics.md).

## Joins, merges, and as-of alignment

Ordinary joins answer "same key, same row." Market data usually needs a different question: *what was the latest quote as of this trade?* Timestamps from two feeds almost never match exactly, so an exact-key join returns nearly nothing, and an approximate one — `merge_asof` — is the correct tool. It matches each left row to the most recent right row at or before it, optionally bounded by a `tolerance`.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
quotes = pd.DataFrame({
    "ts": pd.date_range("2024-01-02 09:30:00", periods=120, freq="s"),
    "bid": (100.0 + np.cumsum(rng.normal(0, 0.005, 120))).round(3),
})
quotes["ask"] = quotes["bid"] + 0.02

trades = pd.DataFrame({
    "ts": pd.to_datetime(["2024-01-02 09:29:59.100",
                          "2024-01-02 09:30:42.200",
                          "2024-01-02 09:31:33.900"]),
    "px": [100.01, 99.98, 100.05],
})

merged = pd.merge_asof(trades, quotes, on="ts", direction="backward",
                       tolerance=pd.Timedelta("5s"))
print(merged)
#                        ts      px      bid      ask
# 0 2024-01-02 09:29:59.100  100.01      NaN      NaN
# 1 2024-01-02 09:30:42.200   99.98  100.011  100.031
# 2 2024-01-02 09:31:33.900  100.05  100.005  100.025
print(merged["bid"].isna().sum())  # => 1 — no quote existed yet: honest NaN
```

`direction="backward"` is the only choice that never looks into the future, which makes it the default for anything feeding a backtest; the first trade, printed before any quote existed, correctly gets NaN rather than a quote from one second later. The same pattern joins slow-moving reference data — shares outstanding, index membership, fundamentals — onto prices. And regardless of which join you use, adopt the verification habit: after every merge, check the row count against what you expected and count the nulls in the joined columns. An outer join that silently doubled your rows, or an inner join that silently dropped a third of them, is telling you something about your data that you want to hear *now*, not after the backtest. Where those quotes come from and why the bid–ask spread exists at all is [Market Microstructure](../part-01-foundations/03-market-microstructure.md) territory.

## Polars and the lazy frame

Polars keeps the DataFrame idea but changes the execution model. You describe a computation as a chain of **expressions** on a `LazyFrame`; nothing runs until `collect()`, at which point a query optimizer reorders and prunes the plan — pushing filters down toward the data, reading only the columns the query touches, and running the rest on all cores. `explain()` shows you the optimized plan before you pay for it.

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(42)
ts = pl.datetime_range(
    pl.datetime(2024, 1, 2, 9, 30), pl.datetime(2024, 1, 2, 15, 59),
    interval="1m", time_zone="America/New_York", eager=True)

lf = (
    pl.LazyFrame({"ts": ts,
                  "price": 100.0 * np.exp(np.cumsum(rng.normal(0, 4e-4, 390))),
                  "volume": rng.integers(1_000, 50_000, 390)})
    .with_columns(pl.col("ts").set_sorted())
    .filter(pl.col("ts").dt.hour() < 11)
    .group_by_dynamic("ts", every="5m")
    .agg(open=pl.col("price").first(), high=pl.col("price").max(),
         low=pl.col("price").min(), close=pl.col("price").last(),
         volume=pl.col("volume").sum())
)
print(lf.explain())
# AGGREGATE[maintain_order: true]
#   [col("price").first().alias("open"), … ] BY []
#   FROM
#    …
#     FILTER (col("ts").dt.hour() < 11)
#     FROM
#       DF ["ts", "price", "volume"]; PROJECT */3 COLUMNS

out = lf.collect()
print(out.select("ts", "open", "close", "volume").head(3))
# shape: (3, 4)
# ┌────────────────────────────────┬────────────┬───────────┬────────┐
# │ ts                             ┆ open       ┆ close     ┆ volume │
# │ ---                            ┆ ---        ┆ ---       ┆ ---    │
# │ datetime[μs, America/New_York] ┆ f64        ┆ f64       ┆ i64    │
# ╞════════════════════════════════╪════════════╪═══════════╪════════╡
# │ 2024-01-02 09:30:00 EST        ┆ 100.012189 ┆ 99.960196 ┆ 151793 │
# │ 2024-01-02 09:35:00 EST        ┆ 99.908144  ┆ 99.865862 ┆ 105370 │
# │ 2024-01-02 09:40:00 EST        ┆ 99.900996  ┆ 99.99849  ┆ 127058 │
# └────────────────────────────────┴────────────┴───────────┴────────┘
```

The pipeline is the same one you just built in pandas — filter, five-minute bars, OHLCV aggregation — expressed as a plan instead of a sequence of materialized intermediates. On one day of bars the difference is invisible; on a few years of tick files it is the difference between a scan that reads 40 GB and one that reads the two columns and three months you asked about. The `scan_parquet` entry point that makes that concrete appears in [SQL and Data Storage](05-sql-and-data-storage.md).

When does each engine win? Honest benchmarks are workload-dependent, but the pattern is stable:

| Workload | Better fit |
|---|---|
| Interactive exploration, small-to-medium data | pandas — richest ecosystem, every tutorial speaks it |
| Scans over files bigger than memory | Polars — lazy execution, predicate and projection pushdown |
| Wide group-bys and joins on many cores | Polars — parallel engine, no interpreter bottleneck |
| Feeding statsmodels, scikit-learn, plotting | pandas — those libraries consume it natively |

Migration is not a rewrite decision. The practical pattern is to keep pandas as the interactive lingua franca, move the heavy loading and aggregation stages to Polars as data grows, and cross the boundary explicitly with `pl.from_pandas` and `.to_pandas()` — each conversion is a copy, so make the crossing once per pipeline stage, not per function call.

!!! abstract "Key takeaways"
    - A DataFrame is NumPy plus labels, and label alignment — not positional arithmetic — is the default; build frames from whole arrays, and use `category` dtype for repeating symbols.
    - Keep timestamps timezone-aware: `tz_localize` attaches a zone to naive stamps, `tz_convert` re-expresses an instant; store aware, think in exchange time.
    - Resampling OHLCV is five aggregations (`first/max/min/last/sum`), and the `label`/`closed` convention must match your execution assumption or you are trading on the future.
    - Rolling statistics look backward by default; keep them honest with `min_periods` equal to the window and a `shift(1)` between computing a statistic and acting on it.
    - `merge_asof` with `direction="backward"` is the join that respects time; verify every merge by checking row counts and null counts.
    - Polars expresses the same pipelines as lazy query plans that an optimizer prunes and parallelizes — adopt it at the loading/aggregation layer first and convert at explicit boundaries.

## Where this goes next

You can now compute almost anything about a price series — which raises the question of what the code computing it should look like once it is more than a notebook cell. [Typing, Dataclasses, and Code Structure](03-typing-dataclasses-structure.md) is about making research code survive contact with a second reader, including the one you will be in six months.
