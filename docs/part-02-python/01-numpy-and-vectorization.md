# NumPy and Vectorization

A modest backtest touches a decade of daily bars for a few thousand symbols — call it seven million rows — and a serious research process recomputes signals over that history many times a day. Pure Python treats every one of those numbers as a full object on the heap, with a type check and reference-count update on every addition. NumPy treats the whole array as one block of typed memory and hands the arithmetic to compiled loops. That difference is not a performance nicety; it decides whether a research iteration takes seconds or hours, and everything you will use for the rest of this course — pandas in the next lesson, the statistics of Part III, the backtesting engine of Part V — is NumPy underneath.

This lesson builds the mental model that makes NumPy predictable rather than magical: what an array actually is in memory, when an operation aliases and when it copies, how shapes combine without loops, and where floating-point arithmetic quietly betrays financial calculations.

## Arrays are how prices want to be stored

A Python list of floats is an array of pointers to boxed objects scattered across the heap. An **ndarray** is the opposite: one contiguous buffer of raw values, described by a dtype (what each element is), a shape (how the buffer is logically arranged), and strides (how many bytes to step per axis). Every fast thing NumPy does follows from that layout — the CPU streams through memory in order, and the per-element work happens in compiled code instead of the interpreter.

Throughout Part II we generate synthetic market data with a seeded generator, so every output you see is reproducible; the appendix page on [Random Number Generation](../appendix/part-09-monte-carlo-methods/01-random-number-generation.md) covers what lives inside `default_rng`.

```python
import numpy as np

rng = np.random.default_rng(42)
rets = rng.normal(0.0005, 0.01, 252)      # one year of daily log returns
prices = 100.0 * np.exp(np.cumsum(rets))  # a synthetic price path

print(prices.dtype, prices.shape)  # => float64 (252,)
print(prices.nbytes)               # => 2016 — 252 values, 8 bytes each, one block
print(prices[:3].round(2))         # => [100.36  99.37 100.17]
```

!!! note "Versions"
    Part II assumes Python 3.12+, NumPy 2.x, pandas 2.2+, and Polars 1.x; the examples were verified with NumPy 2.5, pandas 3.0, and Polars 1.43. If an idiom looks unfamiliar — `default_rng` instead of `np.random.seed`, `"min"` instead of `"T"` — that is deliberate: the older spellings are deprecated or removed.

The dtype is a decision, not a detail. For market data the defaults are:

| dtype | Size | Typical use |
|---|---|---|
| `float64` | 8 bytes | Prices, returns, signals — the default, ~16 significant digits |
| `float32` | 4 bytes | Large feature matrices where memory dominates and precision is negotiable |
| `int64` | 8 bytes | Share counts, volume, tick counts |
| `bool` | 1 byte | Masks — universe membership, halts, signal on/off |
| `datetime64[ns]` | 8 bytes | Timestamps (pandas builds its index on these) |

Stay in `float64` until you have measured a reason not to. The 16 digits comfortably hold any price; what they do not survive is careless arithmetic, which is the subject of the floating-point section below.

## Slicing, views, and the copy trap

The single most common NumPy bug in research code is not knowing whether you are holding a **view** — a new array object aliasing the same memory — or a **copy**. Basic slices are views: cheap, instant, and connected. Boolean masks and fancy (integer-array) indexing allocate fresh copies. The `.base` attribute tells you which one you have.

```python
import numpy as np

rng = np.random.default_rng(42)
rets = rng.normal(0.0005, 0.01, 252)

window = rets[20:40]        # a slice is a view: same memory, new bounds
print(window.base is rets)  # => True

window[:] = 0.0             # writes through to the original
print(rets[25])             # => 0.0

up_days = rets[rets > 0]      # a boolean mask allocates a fresh copy
print(up_days.base is None)   # => True

up_days[:] = 999.0            # goes nowhere near rets
print(rets.max() < 999)       # => True
```

Both behaviors are what you want, at different moments. Views are why slicing a 20-day lookback out of a ten-year history costs nothing. But if you winsorize or clip that window in place, you have silently edited history — and every later computation that touches those bars inherits the edit. The rule: mutate in place only when you own the array; when in doubt, make the copy explicit with `.copy()` rather than relying on indexing side effects.

## Broadcasting: aligning shapes without loops

Broadcasting is how NumPy combines arrays of different shapes without writing loops, and it has exactly two rules: compare shapes from the trailing axis backwards, and two axes are compatible when they are equal or one of them is 1 (a length-1 axis stretches to match). Everything else is an error.

The shape that matters in equity research is `(days, assets)`. Cross-sectional operations — demeaning a day's returns across the universe, standardizing each asset by its own volatility — are one-liners once the shapes are right:

```python
import numpy as np

rng = np.random.default_rng(42)
rets = rng.normal(0.0005, 0.01, (252, 3))  # a year of returns: AAA, BBB, CCC

daily_mean = rets.mean(axis=1, keepdims=True)  # (252, 1) cross-sectional mean
demeaned = rets - daily_mean                   # (252, 3) - (252, 1) broadcasts

vols = rets.std(axis=0)          # (3,) per-asset daily volatility
standardized = rets / vols       # (252, 3) / (3,) broadcasts

print(standardized.std(axis=0))  # => [1. 1. 1.]
```

| Left shape | Right shape | Result |
|---|---|---|
| `(252, 3)` | `(252, 1)` | `(252, 3)` — the length-1 axis stretches |
| `(252, 3)` | `(3,)` | `(252, 3)` — the vector aligns to the trailing axis |
| `(252, 3)` | `(252,)` | error — 252 vs 3 on the trailing axis |
| `(252, 3)` | scalar | `(252, 3)` |

The third row is the one that bites. A per-day quantity has shape `(252,)`, which does *not* align against `(252, 3)` — the trailing axes are 252 and 3. That is why `keepdims=True` exists: it preserves the reduced axis as length 1 so the result broadcasts back against its source. Reductions (`mean`, `std`, `sum`) take an `axis` argument with the same geometry — `axis=0` collapses days, `axis=1` collapses assets. If 2-D arrays as matrices feel rusty, the appendix [Linear Algebra Review](../appendix/part-01-mathematical-foundations/05-linear-algebra-review.md) rebuilds the intuition.

## Vectorizing a real signal, timed honestly

Here is the moving average everyone writes first, next to the one that belongs in a research library. The vectorized version uses a prefix sum: the mean of a window is the difference of two cumulative sums, divided by the window length — no window is ever re-added.

```python
import time
import numpy as np

rng = np.random.default_rng(42)
prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 100_000)))

def sma_loop(p, n=20):
    out = np.full(len(p), np.nan)
    for i in range(n - 1, len(p)):
        out[i] = sum(p[i - n + 1 : i + 1]) / n
    return out

def sma_vec(p, n=20):
    c = np.cumsum(p)
    out = np.full(len(p), np.nan)
    out[n - 1:] = (c[n - 1:] - np.concatenate(([0.0], c[:-n]))) / n
    return out

t0 = time.perf_counter()
slow = sma_loop(prices)
t1 = time.perf_counter()
fast = sma_vec(prices)
t2 = time.perf_counter()

print(np.allclose(slow[19:], fast[19:]))  # => True
print(f"{(t1 - t0) / (t2 - t1):.0f}x")    # => ~100x (machine-dependent)
```

Two habits make the comparison honest. First, verify equivalence before celebrating speed — `np.allclose`, not eyeballing. Second, time with `time.perf_counter` around the exact call, and report the ratio as approximate: it moves with array size, cache state, and hardware, and a claimed "347.2x speedup" is false precision. For windowed computations that do not reduce to prefix sums, `np.lib.stride_tricks.sliding_window_view` gives you a `(n_windows, window)` view — no copying — that you can reduce along the last axis.

Compounding deserves the same treatment. The obvious equity curve is `np.cumprod(1 + r)`; the professional habit is log space:

```python
import numpy as np

rng = np.random.default_rng(42)
simple = rng.normal(0.0005, 0.01, 252)      # one year of simple daily returns

curve = np.exp(np.cumsum(np.log1p(simple)))           # compound in log space
print(np.allclose(curve, np.cumprod(1.0 + simple)))   # => True
print(curve[-1].round(4))                             # => 0.9931 — a flat year
```

At daily scale the two agree to machine precision, as the `allclose` shows. The log-space form wins at the edges: sums are numerically friendlier than long products, log returns add across time (which Part III leans on constantly), and `log1p` keeps precision when returns are tiny — the last point in the next section.

## Floating-point reality

A `float64` carries about 16 significant digits, with relative representation error bounded by machine epsilon:

$$
\frac{\lvert \operatorname{fl}(x) - x \rvert}{\lvert x \rvert} \;\le\; \epsilon_{\text{mach}} \approx 2.2 \times 10^{-16}.
$$

Storage is essentially never the problem — no price needs 16 digits. *Operations* are the problem, and three failure modes account for nearly all of the damage in return calculations.

**Accumulation error.** Adding a small representation error a million times produces a visible one:

```python
import math
import numpy as np

total = 0.0
for _ in range(1_000_000):
    total += 0.01                 # a million penny increments
print(f"{total:.10f}")            # => 10000.0000001719

print(f"{math.fsum([0.01] * 1_000_000):.10f}")     # => 10000.0000000000
print(f"{np.sum(np.full(1_000_000, 0.01)):.10f}")  # => 10000.0000000000
```

`math.fsum` tracks the lost digits exactly; `np.sum` uses pairwise summation, which keeps error growth logarithmic instead of linear. The naive running total is the only one of the three you have to write deliberately — and it is the one most hand-rolled PnL accumulators use.

**Catastrophic cancellation.** Subtracting two nearly equal large numbers annihilates the digits you cared about. The textbook one-pass variance formula does exactly that on price-like data:

```python
import numpy as np

rng = np.random.default_rng(42)
x = 100.0 + rng.normal(0.0, 1e-7, 100_000)   # prices pinned near 100

naive = (x**2).mean() - x.mean()**2          # E[x²] − E[x]² in one pass
print(f"{naive:.3e}")                        # => -1.819e-12 — negative!
print(f"{((x - x.mean())**2).mean():.3e}")   # => 1.007e-14
print(f"{np.var(x):.3e}")                    # => 1.007e-14
```

Both terms are near 10⁴, the true variance is near 10⁻¹⁴, and the subtraction leaves pure rounding noise — here, a *negative* variance, which detonates the moment something downstream takes its square root for a volatility. `np.var` demeans first and never cancels. The lesson generalizes: prefer formulations that subtract small numbers from small numbers.

**Unsafe equality.** Computed floats almost never equal the literal you have in mind:

```python
import numpy as np

print(0.1 + 0.2 == 0.3)             # => False
print(0.1 + 0.2)                    # => 0.30000000000000004
print(np.isclose(0.1 + 0.2, 0.3))   # => True

r = 1e-9                            # a one-tick return on a large notional
print(np.log(1 + r))                # => 1.000000082240371e-09 — digits lost
print(np.log1p(r))                  # => 9.999999995e-10
```

The `log1p` pair shows the same disease in miniature: forming `1 + r` first rounds away most of `r`'s information; `np.log1p` (and its inverse `np.expm1`) work directly on the small quantity and keep it.

!!! warning "Never compare floats with =="
    Test computed values with `np.isclose` / `np.allclose` (or `math.isclose` for scalars), and pick tolerances consciously — the defaults assume quantities of order one. Exact `==` is defensible only for values you assigned and never computed, and for exact sentinels like `0.0` that you set yourself. This applies to tests, to `assert` guards in pipelines, and especially to "did the backtest change?" comparisons between runs.

## NaN is a signal, not a nuisance

`NaN` is floating point's honest answer to "there is no number here" — a delisted symbol, a trading halt, a vendor gap. Its defining behavior is propagation: any arithmetic involving NaN yields NaN, so a plain reduction over dirty data refuses to invent a result. That is a feature. The dangerous tools are the convenient ones:

```python
import numpy as np

rng = np.random.default_rng(42)
rets = rng.normal(0.0005, 0.01, 252)
rets[100:110] = np.nan             # a ten-day trading halt

print(rets.mean())                 # => nan — one NaN poisons the reduction
print(np.nanmean(rets).round(6))   # => 7.9e-05

mask = np.isnan(rets)
print(mask.sum())                  # => 10
clean = rets[~mask]
print(clean.size, clean.mean().round(6))  # => 242 7.9e-05
```

`np.nanmean` and its family silently change the denominator: the halted symbol's "annual" statistics are now computed from 242 days while every clean symbol uses 252, and nothing in the output says so. The explicit-mask version computes the same number but forces you to see the count — and counting what you dropped is the entire discipline. Choose a NaN policy per pipeline stage: propagate while loading (so gaps surface early), mask explicitly where you compute statistics, and log the drop counts. What you must never do is decide by default, which is what reaching for `nan*` functions everywhere amounts to.

!!! abstract "Key takeaways"
    - An ndarray is one contiguous block of typed memory; dtype, shape, and strides describe it, and compiled loops over that block are why NumPy is fast.
    - Slices are views that write through to their source; boolean masks and fancy indexing return copies — check `.base` when in doubt, and copy explicitly before mutating.
    - Broadcasting compares shapes from the trailing axis, stretching length-1 axes; `keepdims=True` is what lets a reduction broadcast back against its source.
    - Vectorize with prefix sums and window views, verify with `np.allclose`, and report speedups as approximate — honest timing is part of the result.
    - Floating point fails through accumulation, cancellation, and equality tests: prefer pairwise/`fsum` summation, two-pass formulas, `log1p`/`expm1`, and `np.isclose`.
    - NaN propagation is a feature; `nan*` reductions silently shrink the denominator, so mask explicitly and count what you dropped.

## Where this goes next

Arrays are fast, but they do not know what time it is — nothing in a `(252, 3)` block says which row is March 17 or that the market was closed on the 18th. [Pandas and Polars](02-pandas-and-polars.md) puts labels, timestamps, and alignment on top of exactly the machinery you just learned, which is where market data starts to feel native.
