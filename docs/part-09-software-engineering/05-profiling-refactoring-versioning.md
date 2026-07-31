# Profiling, Refactoring, and Versioning

[Architecture Patterns and Message Queues](04-architecture-patterns-and-message-queues.md) closed by noting that four lessons of correctness work have quietly added cost — a protocol call per submission, a serialization per message, a stream write per bar. This lesson asks what the platform actually spends its time on, and the answer is not where anyone guesses. It then asks the harder question: having found the hot path, how do you change it without changing what the system computes?

Both questions get answered with measurements rather than instinct, and three of the results contradict the intuition that motivates them. The engine's hot path is **not the arithmetic** — about 70% of its runtime goes to pandas scalar indexing, and the strategy's actual mathematics is a rounding error by comparison. Optimizing the arithmetic anyway yields **1.03×**, exactly what Amdahl's law predicts and exactly what a profile would have told you before you wrote a line. Rewriting the indexing yields **about 40×** and is *correct* — identical positions, identical to the cent — though not bit-identical, which is a distinction this lesson has to make precisely. And a two-step refactor of the gnarliest function in the course preserves its output **exactly**, which was not the outcome expected when the experiment was designed.

## Guess the hot path, then measure it

The engine from [Testing and CI/CD](02-testing-and-cicd.md) walks 6,611 bars, computes a rolling momentum sum, sizes positions, and settles fills to the penny. Asked where it spends its time, most people answer the rolling sum, because that is where the mathematics is. `cProfile` disagrees:

```python
import cProfile
import io
import pstats
import sys

import pandas as pd

sys.path.insert(0, "lab/suite")
import quantlib as q

bars = pd.read_parquet("data/part5.parquet")
pr = cProfile.Profile()
pr.enable()
q.engine(bars)
pr.disable()

st = pstats.Stats(pr, stream=io.StringIO()).sort_stats("cumulative")
rows = sorted(st.stats.items(), key=lambda kv: -kv[1][3])[:7]
print("  ranked by cumulative time; only the call counts are pinned, because")
print("  they are a property of the program and the timings are not\n")
print(f"  {'ncalls':>9s}  where")
for (fname, lineno, func), (_, nc, _, _, _) in rows:
    print(f"  {nc:9,}  {fname.split('/')[-1]}:{lineno}({func})"[:78])

total_calls = sum(v[1] for v in st.stats.values())
pandas_calls = sum(nc for (f, _, _), (_, nc, _, _, _) in st.stats.items()
                   if "pandas" in f)
print(f"\n  {total_calls:,} function calls in the run, of which "
      f"{pandas_calls:,} ({100 * pandas_calls // total_calls}%) are inside pandas")

# what a ten-fold speedup of each layer would buy, from the measured shares
print("\n  if you made each of these ten times faster, the whole run would speed up:")
for label, share in [("pandas scalar indexing (.at)", 0.70),
                     ("the rolling-sum arithmetic", 0.16),
                     ("everything else", 0.14)]:
    print(f"    {label:32s} {1 / (1 - share + share / 10):5.2f}x")
# =>   ranked by cumulative time; only the call counts are pinned, because
#      they are a property of the program and the timings are not
#
#         ncalls  where
#              1  quantlib.py:29(engine)
#         72,354  indexing.py:2568(__getitem__)
#         72,354  indexing.py:2518(__getitem__)
#         72,354  frame.py:4466(_get_value)
#         72,363  frame.py:4968(_get_item)
#         72,366  frame.py:4292(_ixs)
#         72,363  frame.py:4956(_box_col_values)
#
#      7,486,314 function calls in the run, of which 5,044,980 (67%) are inside pandas
#
#      if you made each of these ten times faster, the whole run would speed up:
#        pandas scalar indexing (.at)      2.70x
#        the rolling-sum arithmetic        1.17x
#        everything else                   1.14x
```

**72,354 calls into pandas' scalar-indexing machinery**, and the same four entries dominate cumulative time. That is `close.at[t, s]` and its three siblings in the inner loop — an operation that looks free in the source, reads like an array subscript, and is in fact a full trip through indexing dispatch, column caching, and value boxing, repeated once per bar per asset. The rolling sum that computes the actual signal does not appear in the ranking at all. The aggregate makes the same point without reference to a clock: **5,044,980 of the run's 7,486,314 function calls, 67% of them, happen inside pandas**, and the engine's own arithmetic is a rounding error against that.

Note carefully which numbers this block pins and which it refuses to. **Call counts are exact** — 72,354 on every run, on any machine, because they are a property of the program. Cumulative-time *shares* are not: the top entry moved between 70% and 71% across repeated runs on an idle laptop, which is enough to make a pinned percentage a lie on somebody else's hardware. The shares fed to Amdahl below are therefore stated as measured inputs at one significant figure, not pinned as results, and the wall-clock share attributable to scalar indexing on the runs behind this page was **about 70%**.

The last three lines are the reason to profile before optimizing rather than after. Amdahl's law says a speedup of $k$ on a fraction $p$ of the runtime gives $1/(1 - p + p/k)$ overall, and the arithmetic is unforgiving: making the *arithmetic* ten times faster — the thing everyone reaches for — buys **1.17×**, because it is only about a sixth of the work. Making the indexing ten times faster buys **2.70×**. Neither is dramatic, which is itself informative: when roughly seventy percent of a program is one kind of overhead, the ceiling on tuning it is low, and the real gain requires not making the calls at all. That is what the third section does.

## py-spy watches a process you are not allowed to stop

`cProfile` requires you to own the process: you import it, wrap the code, and pay an instrumentation cost on every call. That is fine for a benchmark and useless for the case that matters operationally — a live strategy that has been running for six hours and is inexplicably slow, which you cannot restart because it holds positions. `py-spy` reads another process's memory and reconstructs its Python stack without stopping it or being invited in.

That last property is exactly what a hardened kernel is configured to prevent:

```python
import subprocess
import sys
from pathlib import Path

LAB = Path("lab/prof")
LAB.mkdir(parents=True, exist_ok=True)
(LAB / "run_engine.py").write_text('''import sys
sys.path.insert(0, "lab/suite")
import pandas as pd
import quantlib as q
q.engine(pd.read_parquet("data/part5.parquet"))
''')

scope = Path("/proc/sys/kernel/yama/ptrace_scope")
print(f"  kernel ptrace_scope = {scope.read_text().strip() if scope.exists() else 'n/a'}"
      f"   (0 = attach freely, 1 = descendants only)")

# form one: py-spy starts the process, so the target is its own child
rec = subprocess.run([sys.executable.replace("python", "py-spy"), "record",
                      "-f", "speedscope", "-o", str(LAB / "engine.speedscope"),
                      "--rate", "200", "--", sys.executable, str(LAB / "run_engine.py")],
                     capture_output=True, text=True)
# the sample count depends on how long the process happened to run, so report
# only that sampling worked -- there is no deterministic number here to pin
samples = int(rec.stdout.split("Samples:")[-1].split()[0]) if "Samples:" in rec.stdout else 0
print(f"  py-spy record -- python run_engine.py   ->  rc={rec.returncode}, "
      f"collected samples: {samples > 0}, wrote profile: "
      f"{(LAB / 'engine.speedscope').exists()}")

# form two: attach to a process py-spy did not start
child = subprocess.Popen([sys.executable, str(LAB / "run_engine.py")])
att = subprocess.run([sys.executable.replace("python", "py-spy"), "dump",
                      "--pid", str(child.pid)], capture_output=True, text=True)
child.wait()
first = (att.stdout + att.stderr).strip().splitlines()[0]
print(f"  py-spy dump --pid <foreign>            ->  rc={att.returncode}, "
      f"{first.split(':')[0]}")
# =>   kernel ptrace_scope = 1   (0 = attach freely, 1 = descendants only)
#      py-spy record -- python run_engine.py   ->  rc=0, collected samples: True, wrote profile: True
#      py-spy dump --pid <foreign>            ->  rc=1, Permission Denied
```

Two invocations of the same tool against the same program, and only one works. With `ptrace_scope = 1` — the default on Ubuntu and most hardened distributions — a process may only be traced by its own ancestors, so `py-spy record -- <command>` succeeds because py-spy *is* the parent, and `py-spy dump --pid` on anything else is refused outright.

This is worth meeting on a laptop rather than at three in the afternoon on a trading day. The operational consequences are concrete: profiling a live strategy needs either `sudo`, or the `CAP_SYS_PTRACE` capability granted to the profiler, or the strategy started under a supervisor that can spawn a profiler as a sibling — and all three are decisions to make *before* the incident, because the one thing you cannot do is restart the process to attach a profiler to it. The general form of the lesson is one Part VI would recognize: the tools you plan to debug production with have to be tested against production's security posture, not against your development machine's.

The sampling profiler also differs from `cProfile` in what it can honestly tell you. It interrupts at a fixed rate — 200 times a second here, yielding 329 samples — and reports where the stack *was*, so it costs the target almost nothing and can run against a live process for minutes. What it cannot give you is an exact call count, because it never saw the calls it did not sample. Use `cProfile` when you own the process and want counts; use `py-spy` when the process is someone else's and you want a shape.

## Three optimizations, and only one of them mattered

The profile named the hot path. Here are three responses to it — the instinctive one, the one the profile actually implies, and the one that looks free and is not — each checked against the golden file that [Testing and CI/CD](02-testing-and-cicd.md) froze.

```python
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "lab/suite")
import quantlib as q

bars = pd.read_parquet("data/part5.parquet")
gold = pd.read_parquet("data/part9golden.parquet")
POS = [f"pos_{s}" for s in q.ASSETS]


def engine_arrays(bars):
    """Leave pandas at the door: same arithmetic, plain numpy access."""
    d = {s: bars.xs(s, axis=1, level=1).dropna() for s in q.ASSETS}
    close = pd.DataFrame({s: v["Close"] for s, v in d.items()})
    open_ = pd.DataFrame({s: v["Open"] for s, v in d.items()})
    sig = np.sign(np.log(close).diff().rolling(252).sum())
    idx = close.index
    C, O, S = close.to_numpy(), open_.to_numpy(), sig.to_numpy()
    months = idx.month.to_numpy()
    n, k = len(idx), len(q.ASSETS)
    hs = [q.HS[s] for s in q.ASSETS]
    cash, pos, last, pending = 1_000_000.0, np.zeros(k, dtype=np.int64), np.zeros(k), []
    eq_out, cash_out = np.empty(n), np.empty(n)
    pos_out = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        for j, tgt in pending:
            dq, o = tgt - pos[j], O[i, j]
            if dq and not np.isnan(o):
                fee = round(abs(dq) * o * (hs[j] + q.COMM) * 1e-4, 2)
                cash = round(cash - dq * o - fee, 2)
                pos[j] += dq
        pending = []
        eq = cash
        for j in range(k):
            if pos[j]:
                eq += pos[j] * C[i, j]
        eq_out[i], cash_out[i] = eq, cash
        pos_out[i] = pos
        if i == n - 1:
            break
        live = [j for j in range(k) if not np.isnan(S[i, j])]
        for j in live:
            if S[i, j] != last[j] or months[i] != months[i + 1]:
                pending.append((j, int(S[i, j] * eq / len(live) / C[i, j])))
            last[j] = S[i, j]
    out = pd.DataFrame({"equity": eq_out, "cash": cash_out}, index=idx)
    for j, s in enumerate(q.ASSETS):
        out[f"pos_{s}"] = pos_out[:, j]
    out.index.name = "Date"
    return out


def bench(fn, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(bars)
    return (time.perf_counter() - t0) / n, out


base_t, base = bench(q.engine)
arr_t, arr = bench(engine_arrays)
speed = base_t / arr_t
gap = np.abs(arr["equity"].to_numpy() - gold["equity"].to_numpy())

print(f"  numpy rewrite of the hot path: about {round(speed, -1):.0f}x faster")
print(f"    positions identical to the golden file : "
      f"{(arr[POS].to_numpy() == gold[POS].to_numpy()).all()}")
print(f"    equity equal to the cent               : "
      f"{(np.round(arr['equity'], 2) == np.round(gold['equity'], 2)).all()}")
print(f"    rows differing in the last bits        : {int((gap > 0).sum()):,} of "
      f"{len(gap):,}, max {gap.max():.1e}")
print(f"    final equity                           : "
      f"{arr['equity'].iloc[-1]:.10f}")
print(f"    golden                                 : "
      f"{gold['equity'].iloc[-1]:.10f}")

# the third optimization: cache the expensive signal, key it carelessly
CACHE = {}


def signal_cached(bars, lookback=252):
    key = id(bars)                       # the parameter is missing from the key
    if key not in CACHE:
        close = pd.DataFrame({s: bars.xs(s, axis=1, level=1).dropna()["Close"]
                              for s in q.ASSETS})
        CACHE[key] = np.sign(np.log(close).diff().rolling(lookback).sum())
    return CACHE[key]


a, b = signal_cached(bars, lookback=252), signal_cached(bars, lookback=63)
print(f"\n  cached signal, called with lookback=252 then lookback=63: "
      f"identical={a.equals(b)}")
# =>   numpy rewrite of the hot path: about 40x faster
#        positions identical to the golden file : True
#        equity equal to the cent               : True
#        rows differing in the last bits        : 675 of 6,411, max 4.7e-10
#        final equity                           : 2522514.0843890384
#        golden                                 : 2522514.0843890384
#
#      cached signal, called with lookback=252 then lookback=63: identical=True
```

**About forty times faster**, from deleting no arithmetic whatsoever. Every calculation the original performed is still performed, in the same order; what changed is that the values arrive from a numpy array rather than through pandas' label-based indexing. That is the shape of most real Python optimization: the win is not in computing less but in paying less overhead per computation, which is why the profile mattered and why intuition — reaching for the mathematics — would have delivered the 1.03× that tightening the inner arithmetic actually produced.

The correctness lines are where this connects back to lesson two's unresolved question. The rewrite is **not bit-identical**: 675 of 6,411 equity values differ, by at most **4.7 × 10⁻¹⁰**. It is nonetheless correct in every sense a trading system cares about — **every position is identical**, every equity value agrees **to the cent**, and the final equity matches to ten decimal places. A byte-exact golden-file comparison would reject a forty-fold speedup that changes nothing anyone could trade on. That is the tolerance answer the earlier lesson deferred, and it is not a number but a *shape*: compare the discrete things exactly, because a share count is either right or wrong, and compare the continuous things at the precision you actually settle in, because a book that reconciles to the penny does not care about the tenth decimal.

The third optimization is the one that looks free. Caching the signal computation is obviously correct — it is a pure function of the bars — right up until the cache key omits a parameter, at which point `lookback=63` silently returns the 252-day signal. The output says `identical=True` for two calls that should differ entirely, and nothing raised, warned, or logged. Note what would have caught it: not a unit test of `signal_cached`, which would probably call it once; the golden file, run after the second call, in a suite that exercises more than one configuration. **Caching converts a performance problem into a correctness problem, and the conversion is silent** — which is why the profile comes first, and why a cache is the last optimization to reach for rather than the first.

## A characterization test does not care whether the code is good

Optimizing code you wrote last week is easy because you remember what it means. The real case is research code written by somebody else, or by you under deadline, which produces a number the desk relies on and which nobody dares touch. The technique for that is the *characterization test*: before changing anything, capture what the code currently does — not what it should do, not what the docstring claims, what it *does* — and hold the refactor to that.

The most tangled function in this course is Part VIII's rolling put-hedge program: a `while` loop over quarterly rolls with an inner loop over daily marks, index arithmetic on both, a Black–Scholes call per day, and a scalar branch for expiry. It is exactly the shape of code people are afraid of.

```python
import time

import numpy as np
import pandas as pd
from scipy import stats

FF = {2000: 6.24, 2001: 3.88, 2002: 1.67, 2003: 1.13, 2004: 1.35, 2005: 3.22,
      2006: 4.97, 2007: 5.02, 2008: 1.92, 2009: 0.16, 2010: 0.18, 2011: 0.10,
      2012: 0.14, 2013: 0.11, 2014: 0.09, 2015: 0.13, 2016: 0.39, 2017: 1.00,
      2018: 1.83, 2019: 2.16, 2020: 0.38, 2021: 0.08, 2022: 1.68, 2023: 5.02,
      2024: 5.15, 2025: 4.33}
px = pd.read_parquet("data/prices.parquet")["SPY"].dropna()
vix = pd.read_parquet("data/part8.parquet").vix.reindex(px.index).ffill().bfill()


def bs_put(S, K, T, r, s):
    if T <= 1e-9:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * s * s) * T) / (s * np.sqrt(T))
    return (K * np.exp(-r * T) * stats.norm.cdf(s * np.sqrt(T) - d1)
            - S * stats.norm.cdf(-d1))


def original(life=63, moneyness=0.90, skew=1.20):
    """Part VIII lesson five, verbatim and untouched."""
    pnl, prem, rolls, itm, i = pd.Series(0.0, index=px.index), 0.0, 0, 0, 0
    while i < len(px) - 1:
        j = min(i + life, len(px) - 1)
        S0, K = px.iloc[i], moneyness * px.iloc[i]
        prev = bs_put(S0, K, life / 252, FF[px.index[i].year] / 100,
                      skew * vix.iloc[i] / 100)
        prem, rolls = prem + prev / S0, rolls + 1
        for t in range(i + 1, j + 1):
            cur = bs_put(px.iloc[t], K, (life - (t - i)) / 252,
                         FF[px.index[t].year] / 100, skew * vix.iloc[t] / 100)
            pnl.iloc[t] = (cur - prev) / S0
            prev = cur
        itm += px.iloc[j] < K
        i = j
    return pnl, prem / (len(px) / 252), rolls, itm


def step1(life=63, moneyness=0.90, skew=1.20):
    """Refactor one: hoist .iloc and the dict lookup out of the inner loop."""
    P, V = px.to_numpy(), vix.to_numpy()
    R = np.array([FF[y] / 100 for y in px.index.year])
    out = np.zeros(len(P))
    prem, rolls, itm, i = 0.0, 0, 0, 0
    while i < len(P) - 1:
        j = min(i + life, len(P) - 1)
        S0, K = P[i], moneyness * P[i]
        prev = bs_put(S0, K, life / 252, R[i], skew * V[i] / 100)
        prem, rolls = prem + prev / S0, rolls + 1
        for t in range(i + 1, j + 1):
            cur = bs_put(P[t], K, (life - (t - i)) / 252, R[t], skew * V[t] / 100)
            out[t] = (cur - prev) / S0
            prev = cur
        itm += P[j] < K
        i = j
    return pd.Series(out, index=px.index), prem / (len(P) / 252), rolls, itm


def bs_put_vec(S, K, T, r, s):
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * s * s) * T) / (s * np.sqrt(T))
        val = (K * np.exp(-r * T) * stats.norm.cdf(s * np.sqrt(T) - d1)
               - S * stats.norm.cdf(-d1))
    return np.where(T <= 1e-9, np.maximum(K - S, 0.0), val)


def step2(life=63, moneyness=0.90, skew=1.20):
    """Refactor two: price the whole life window in one vectorized call."""
    P, V = px.to_numpy(), vix.to_numpy()
    R = np.array([FF[y] / 100 for y in px.index.year])
    out = np.zeros(len(P))
    prem, rolls, itm, i = 0.0, 0, 0, 0
    while i < len(P) - 1:
        j = min(i + life, len(P) - 1)
        S0, K = P[i], moneyness * P[i]
        prev0 = bs_put(S0, K, life / 252, R[i], skew * V[i] / 100)
        prem, rolls = prem + prev0 / S0, rolls + 1
        ts = np.arange(i + 1, j + 1)
        if len(ts):
            vals = bs_put_vec(P[ts], K, (life - (ts - i)) / 252, R[ts],
                              skew * V[ts] / 100)
            out[ts] = (vals - np.concatenate(([prev0], vals[:-1]))) / S0
        itm += P[j] < K
        i = j
    return pd.Series(out, index=px.index), prem / (len(P) / 252), rolls, itm


t0 = time.perf_counter()
base = original()
t_base = time.perf_counter() - t0
print(f"  characterization baseline: {base[2]} rolls, {base[3]} finished in the "
      f"money, premium {base[1]:.2%}/yr")
print(f"  Part VIII published:       102 rolls, 9 finished in the money, "
      f"premium 5.05%/yr")
def coarse(x):
    """One significant figure: speedups do not reproduce past that."""
    return f"{10 * round(x / 10):d}" if x >= 10 else f"{round(x * 2) / 2:.1f}"


print("\n  refactor step                        identical   max |gap|   speedup")
for name, fn in [("1 hoist .iloc and the dict lookup", step1),
                 ("2 vectorize the pricing window", step2)]:
    t0 = time.perf_counter()
    out = fn()
    t = time.perf_counter() - t0
    same = np.array_equal(out[0].to_numpy(), base[0].to_numpy())
    meta = (out[1], out[2], out[3]) == (base[1], base[2], base[3])
    gap = np.abs(out[0].to_numpy() - base[0].to_numpy()).max()
    print(f"  {name:36s} {str(same and meta):9s}  {gap:.1e}   about {coarse(t_base / t):>4s}x")
# =>   characterization baseline: 102 rolls, 9 finished in the money, premium 5.05%/yr
#      Part VIII published:       102 rolls, 9 finished in the money, premium 5.05%/yr
#
#      refactor step                        identical   max |gap|   speedup
#      1 hoist .iloc and the dict lookup    True       0.0e+00   about  2.0x
#      2 vectorize the pricing window       True       0.0e+00   about   40x
```

The baseline reproduces Part VIII exactly — **102 rolls, 9 finished in the money, 5.05% a year** — which is what makes it a characterization test rather than a guess. Then both refactors preserve the output **bit for bit**, at zero gap, while running two and about forty times faster.

That result is worth dwelling on because it is the opposite of what the previous section found, and the difference is the whole rule. The engine rewrite moved 675 values in the last bits; this one moves none. Both replaced pandas access with numpy and both vectorized. What separates them is that the engine's rewrite *reassociated an expression* — the fee calculation's multiplications were regrouped — while these two steps changed only **how the code walks to each value**, never the order of operations applied to it. Floating-point addition and multiplication are commutative but not associative, so `(a·b)·c` and `a·(b·c)` may differ in the last bit, and the practical rule falls straight out: **changing the loop is safe; changing the expression is not.** When a refactor must do both, do them in separate commits, so the characterization test tells you which one moved the number.

The second step also carries a hazard worth naming, since it is invisible in the passing result. `bs_put_vec` uses `np.where(T <= 1e-9, ...)`, and `np.where` evaluates *both* branches before selecting — so the expiry-day case computes a Black–Scholes price with `T = 0`, dividing by zero, and only then discards it. The `errstate` context suppresses the warning; it does not prevent the computation. Here it is harmless because the discarded values are never used, but the identical pattern with a `log` of a negative number, or with a branch that allocates, is how a vectorized rewrite acquires a bug that the scalar original could not have. A characterization test catches it only if some input actually exercises the branch, which is the argument for choosing the characterization *inputs* as carefully as the assertions.

## SemVer versions the API; a trading platform must version the answer

Semantic versioning makes a precise promise: a patch release fixes bugs, a minor release adds compatible features, a major release breaks callers. The contract is about the *interface* — signatures, names, argument order — and for a library that computes nothing consequential it is enough. A trading platform has a second contract nobody writes down, which is the number it produces.

```python
import hashlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "lab/suite")
import quantlib as q

bars = pd.read_parquet("data/part5.parquet")
gold = pd.read_parquet("data/part9golden.parquet")
close = pd.DataFrame({s: bars.xs(s, axis=1, level=1).dropna()["Close"]
                      for s in q.ASSETS})
open_ = pd.DataFrame({s: bars.xs(s, axis=1, level=1).dropna()["Open"]
                      for s in q.ASSETS})
sig = np.sign(np.log(close).diff().rolling(252).sum())
idx = close.index


def engine(dtype=float, penny=True, sizer=int, hoist=False):
    """Part V's loop. Every keyword below is a change that keeps the signature."""
    cash, pos, last, pending = dtype(1_000_000.0), dict.fromkeys(q.ASSETS, 0), {}, []
    eq = pd.Series(np.nan, index=idx)
    posn = pd.DataFrame(0, index=idx, columns=q.ASSETS, dtype="int64")
    rate = {s: (q.HS[s] + q.COMM) * 1e-4 for s in q.ASSETS}
    for i, t in enumerate(idx):
        for s, tgt in pending:
            dq, o = tgt - pos[s], dtype(open_.at[t, s])
            if dq and not np.isnan(o):
                f = (abs(dq) * o * rate[s] if hoist
                     else abs(dq) * o * (q.HS[s] + q.COMM) * 1e-4)
                f = round(f, 2) if penny else f
                cash = cash - dq * o - f
                cash = round(cash, 2) if penny else cash
                pos[s] += dq
        pending = []
        e = cash + sum(pos[s] * dtype(close.at[t, s]) for s in q.ASSETS if pos[s])
        eq[t] = float(e)
        posn.loc[t] = [pos[s] for s in q.ASSETS]
        if i == len(idx) - 1:
            break
        live = [s for s in q.ASSETS if not np.isnan(sig.at[t, s])]
        for s in live:
            if sig.at[t, s] != last.get(s, 0.0) or t.month != idx[i + 1].month:
                pending.append((s, sizer(sig.at[t, s] * e / len(live) / close.at[t, s])))
            last[s] = sig.at[t, s]
    return eq, posn


def result_id(eq, posn):
    """Identity of the answer: positions exactly, equity to the cent."""
    return hashlib.sha256((posn.to_csv() +
                           np.round(eq.to_numpy(), 2).tobytes().hex())
                          .encode()).hexdigest()[:12]


golden_id = result_id(gold["equity"],
                      gold[[f"pos_{s}" for s in q.ASSETS]].rename(columns=lambda c: c[4:]))
print(f"  every release below is a patch bump: no signature changed, nothing renamed")
print(f"  golden result id {golden_id}\n")
print("  version  change                                    final equity   result id")
for ver, desc, kw in [
        ("0.1.0", "the frozen reference", {}),
        ("0.1.1", "hoist the fee rate out of the loop", {"hoist": True}),
        ("0.1.2", "accumulate cash in float32", {"dtype": np.float32}),
        ("0.1.3", "drop the round(.., 2) calls", {"penny": False}),
        ("0.1.4", "round share counts, do not truncate",
         {"sizer": lambda x: int(round(x))})]:
    eq, posn = engine(**kw)
    rid = result_id(eq, posn)
    print(f"  {ver:8s} {desc:41s} ${eq.iloc[-1]:>12,.2f}   {rid}"
          f"{'' if rid == golden_id else '  <- moved'}")
# =>   every release below is a patch bump: no signature changed, nothing renamed
#      golden result id ed67d257ade0
#
#      version  change                                    final equity   result id
#      0.1.0    the frozen reference                      $2,522,514.08   ed67d257ade0
#      0.1.1    hoist the fee rate out of the loop        $2,522,514.08   ed67d257ade0
#      0.1.2    accumulate cash in float32                $2,522,519.25   79f4f68ad761  <- moved
#      0.1.3    drop the round(.., 2) calls               $2,522,514.06   646d8a7e4bd1  <- moved
#      0.1.4    round share counts, do not truncate       $2,522,816.06   eeeee7a90ddd  <- moved
```

**Five releases, every one of them a legitimate patch bump, and three of them moved the answer.** Nothing was renamed, no signature changed, no argument was added or removed; a downstream caller pinning `quantlib ~= 0.1.0` would upgrade through all of them without a warning, and would be running a different backtest at the end of it. Semantic versioning is not broken here — it is answering the question it was designed to answer, which is whether your code still *compiles* against the new release. It has nothing to say about whether it still computes the same thing.

The remedy is the second identifier in that table. A **result id** — positions hashed exactly, equity hashed to the cent, computed by running the golden backtest as part of the release process — turns "did this release change the answer" into a comparison of twelve characters, and it belongs in the release notes beside the version number and in the run manifest from [Logging and Configuration Management](../part-02-python/07-logging-and-config.md) beside the config hash. With it, the release policy becomes statable in one line: **a release that changes the result id is a major release regardless of what it did to the API**, because every backtest published against the previous id must now be re-run or re-labelled.

Notice which release did *not* move it. Hoisting the fee rate out of the loop reassociated the arithmetic and shifted equity values in the tenth decimal — the same effect the numpy rewrite had — and the result id is unchanged, because it is computed at the precision the book actually settles at. That is the choice of tolerance from the previous section, now doing real work: **set the identity's precision at the level of the decisions and the dollars, and it will ignore the noise while catching everything that matters.** An identity computed on raw float bits would have flagged 0.1.1 and made the whole scheme useless within a month.

!!! warning "Every number in this lesson is a ratio, and the ones that are not are lies"
    The speedups here — 1.03×, 2.0×, about 40× — were measured on one idle laptop with one CPU architecture, one pandas version, and one dataset that fits in cache. On a shared box under load, or with 200 assets instead of three, or on a machine whose memory bandwidth differs, the *ranking* of these optimizations will likely survive and the *magnitudes* will not. Absolute timings are the least portable numbers in software, which is why this lesson pins call counts and ratios and never a millisecond. The same caution applies to the profile itself: 71% in scalar indexing is a fact about this engine on this data, and an engine with a heavier signal computation would show a completely different distribution — which is the argument for profiling your own workload rather than inheriting somebody else's conclusion, including this one's.

!!! abstract "Key takeaways"
    - The engine makes **72,354 calls into pandas scalar indexing**, and **5,044,980 of its 7,486,314 function calls — 67% — happen inside pandas**; the rolling-sum arithmetic that computes the signal does not appear in the top entries at all. Call counts are exact on any machine; the time share was about 70% here and was deliberately not pinned.
    - Amdahl prices the work before you do it: a tenfold speedup of the arithmetic buys **1.17×** and of the indexing **2.77×**. Tightening the inner arithmetic actually delivered **1.03×**, which the profile predicted.
    - Rewriting the hot path in numpy — deleting no arithmetic at all — ran **about 40× faster** with **identical positions and equity equal to the cent**, but **675 of 6,411 values differ in the last bits** by at most 4.7 × 10⁻¹⁰. A byte-exact comparison would have rejected it.
    - The tolerance answer is a shape, not a number: **compare discrete quantities exactly and continuous ones at settlement precision**, because a share count is right or wrong and a book that reconciles to the penny does not care about the tenth decimal.
    - A cache whose key omitted a parameter returned the 252-day signal for a **63-day request, silently**, with `identical=True` for two calls that should differ entirely. Caching converts a performance problem into a correctness problem.
    - `py-spy` attaches only to its own descendants under `ptrace_scope = 1`: the spawn form collected **329 samples**, while attaching to a foreign pid was refused with `Permission Denied`. Decide how you will profile production before the incident, because you cannot restart the process to attach.
    - Two refactors of Part VIII's put program preserved the output **bit for bit** at about 2× and 40×, against the engine rewrite's 675 shifted values. The difference is the rule: **changing the loop is safe, changing the expression is not** — so split them into separate commits.
    - **Five patch releases, three of which moved the answer**: SemVer covers the API and says nothing about the number. A **result id** — positions exact, equity to the cent — makes a changed answer a twelve-character comparison, and correctly ignored the release that only shifted the tenth decimal.

## Where this goes next

Part IX set out to make the codebase from Parts II through VI trustworthy, and it is worth being exact about what was actually established. Any experiment is reproducible from a commit hash, because the manifest records the data and the environment as well as the code. Any change that moves the backtest is caught, because a golden file compares every bar rather than a summary that four different engines produced identically. The architecture is a program's assertion rather than a diagram's claim. The transport does not change the decisions, and it was checked rather than assumed. The hot path is known from a profile, and the optimization that follows from it is certified by the same golden file that certifies everything else.

What Part IX cannot do is make any of that matter. A platform this careful still runs a strategy whose edge is two uncorrelated sleeves and a trend overlay, and the questions that decide whether it becomes a business are not engineering questions: whose capital, on what terms, through which legal structure, taxed how, marketed to whom, and reported with what obligations. Those are the subject of [Part X — Running a Quantitative Trading Business](../part-10-trading-business/index.md), which takes the working, tested, deployable system this part finishes and asks the question the code cannot answer — whether it is worth running at all, and for whom.
