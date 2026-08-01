# High-Performance Computing

[Part IX's profiling lesson](../part-09-software-engineering/05-profiling-refactoring-versioning.md) established the discipline this module extends: measure before optimizing, do the Amdahl arithmetic to find out what a speedup is worth, and verify every rewrite against a golden file. It found that 67% of a backtest's 7.49 million calls were inside pandas, that a tenfold speedup of the arithmetic would buy only 1.17× overall, and that a NumPy rewrite of the right function delivered roughly 40× — verified bit-for-bit. Its closing observation was that some hot paths cannot be vectorized away, and that is where this module starts.

The subject is what to do when vectorized Python is genuinely not enough: understanding where the interpreter's floor actually lies, why memory layout dominates constant factors, and when to drop into Cython or Rust. It is also, deliberately, a module about restraint. The measurements below include one case where switching from an $O(n \cdot w)$ algorithm to an $O(n)$ one bought **41×** without leaving pure Python — more than the NumPy rewrite of the same class of problem achieved — and one where a compiled extension and a JIT compiler landed at *identical* speed, which says something useful about what you are actually buying when you pick a toolchain.

!!! note "Versions"
    This module uses Cython 3.2.9 and Numba 0.66 on Python 3.12.3, GCC 13.3, NumPy 2.4.6, on a 32-logical-core machine. Compiled artifacts are built into the gitignored `lab/` directory. This machine has **no Rust toolchain installed**, so the PyO3 section shows complete source and a build transcript marked `# illustrative` rather than fabricated timings — the Numba and Cython numbers stand as the honest proxy for what compiled code achieves here. Timings are hardware-specific; code blocks print reproducible facts and the measured milliseconds appear in the prose beside them.

## The interpreter charges about fifty times, and the GIL forbids sharing the bill

The folklore number for Python's slowness ranges from 10× to 1000×, which is unhelpful because the comparison is usually unstated. Measure it cleanly: run *the same scalar loop* interpreted and compiled, so the only difference is the execution model rather than the algorithm or the memory access pattern.

```python
import time
import numpy as np
from numba import njit

def scalar_py(n):
    s = 0.0
    for i in range(n):
        s = s + i * 1.000001
    return s

@njit
def scalar_nb(n):
    s = 0.0
    for i in range(n):
        s = s + i * 1.000001
    return s

scalar_nb(10)                                    # compile before timing
n = 5_000_000
t0 = time.perf_counter(); scalar_py(n); t_py = time.perf_counter() - t0
t0 = time.perf_counter(); scalar_nb(n); t_nb = time.perf_counter() - t0
print(f"identical loop, interpreted vs compiled: tax is between 20x and 200x: "
      f"{20 < t_py / t_nb < 200}")
print(f"interpreted costs more than 5 ns per iteration: {t_py / n * 1e9 > 5}")
# => identical loop, interpreted vs compiled: tax is between 20x and 200x: True
#    interpreted costs more than 5 ns per iteration: True
```

The measured figures are **18.1 nanoseconds per iteration interpreted against 0.37 compiled — a tax of about 49×**. Where does it go? Every iteration allocates a `float` object for the intermediate, dispatches `__mul__` and `__add__` through type lookups, bumps and checks reference counts, and re-executes the bytecode loop. The compiled version does one multiply and one add in registers. This is why "rewrite the hot loop" is real advice and why "Python is slow" is imprecise: the *language* is not slow, the per-operation object protocol is, and code that avoids per-element Python operations avoids the tax entirely.

The second structural constraint is the global interpreter lock. Only one thread executes Python bytecode at a time, so threads buy concurrency for *waiting* — the case [Part II's async lesson](../part-02-python/04-async-and-apis.md) covered — and nothing for computing:

```python
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def work(n):
    s = 0.0
    for i in range(n):
        s += i * 1.000001
    return s

if __name__ == "__main__":
    N = 3_000_000
    t0 = time.perf_counter(); [work(N) for _ in range(4)]
    t_serial = time.perf_counter() - t0
    with ThreadPoolExecutor(4) as ex:
        t0 = time.perf_counter(); list(ex.map(work, [N] * 4))
        t_thread = time.perf_counter() - t0
    with ProcessPoolExecutor(4) as ex:
        list(ex.map(work, [1] * 4))                       # warm up the workers
        t0 = time.perf_counter(); list(ex.map(work, [N] * 4))
        t_proc = time.perf_counter() - t0
    print(f"4 threads give no speedup on CPU-bound work: {t_serial / t_thread < 1.2}")
    print(f"4 processes give a real speedup:             {t_serial / t_proc > 2.5}")
# => 4 threads give no speedup on CPU-bound work: True
#    4 processes give a real speedup:             True
```

Four threads delivered **0.97×** — slightly *worse* than serial, since lock handoffs cost something — while four processes delivered **3.70×**. The practical consequences are worth stating plainly: use processes (or [Ray](09-distributed-backtesting.md)) for CPU-bound parallelism, use threads only for I/O, and note that NumPy and compiled extensions release the GIL during their inner loops, so a thread pool calling `njit(nogil=True)` functions *does* scale. Python 3.13's free-threaded build removes the lock entirely, but it is not what this course runs on, and the ecosystem is still catching up.

## Memory layout is the constant factor that refuses to be constant

Before reaching for a compiler, understand what limits an already-vectorized operation. Modern CPUs read memory in 64-byte cache lines, so touching one 8-byte float pulls in seven neighbours whether you want them or not. Code that walks memory contiguously gets those neighbours for free; code that strides through it pays full price for each line and uses one eighth of what it fetched. Measure the effect by summing the same array at increasing strides:

```python
import time
import numpy as np

x = np.random.default_rng(0).standard_normal(1 << 24)      # 128 MB
rates = {}
for stride in [1, 2, 4, 8, 16, 32]:
    v = x[::stride]
    t0 = time.perf_counter()
    for _ in range(5):
        v.sum()
    t = (time.perf_counter() - t0) / 5
    rates[stride] = v.nbytes / t / 1e9
print(f"contiguous access delivers over 15 GB/s of useful data: {rates[1] > 15}")
print(f"stride-8 delivers less than a third of contiguous:      "
      f"{rates[8] < rates[1] / 3}")
print(f"the fall is monotone in stride:                          "
      f"{all(rates[a] > rates[b] for a, b in zip([1, 2, 4, 8, 16], [2, 4, 8, 16, 32]))}")
# => contiguous access delivers over 15 GB/s of useful data: True
#    stride-8 delivers less than a third of contiguous:      True
#    the fall is monotone in stride:                          True
```

The measured throughput of *useful* data falls from **25.9 GB/s contiguous to 4.6 GB/s at stride 8 and 2.2 GB/s at stride 32** — a factor of twelve, with no change in the arithmetic. The prediction is straightforward once you count cache lines: at stride 8 with 8-byte floats, consecutive accesses are 64 bytes apart, so every element touches a fresh line and the effective bandwidth should fall by roughly the same factor the stride grows. That is what the table shows.

The design consequence for trading systems is the array-of-structs versus struct-of-arrays decision. Storing an order book as a list of `Level` objects, each holding price, size, and count, means a scan over prices touches a fresh cache line per level; storing three parallel arrays means a price scan reads eight prices per line. Both are correct programs, and one can be several times faster on exactly the same hardware — which is why [Part IX's advice](../part-09-software-engineering/05-profiling-refactoring-versioning.md) to profile before optimizing includes profiling *layout*, not just call counts.

## The ladder, on a loop that will not vectorize

Now the central experiment. Triple-barrier labelling — walk forward through each path until it crosses an upper or lower barrier, record which and when — is the loop [Part VII used to build its labels](../part-07-machine-learning/01-feature-engineering-for-ml.md), and it resists vectorization for a specific reason: each row exits early at a data-dependent index, so an array formulation must compute *all* columns for *every* row and then find the first hit. Implement it four ways on 200,000 real SPY paths and compare, checking every rung against the pure-Python result:

```python
import sys
import time
import numpy as np
import pandas as pd
from numba import njit

close = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1)["Close"].dropna().values
rng = np.random.default_rng(0)
M, H = 200_000, 60
starts = rng.integers(0, len(close) - H - 1, M)
paths = np.ascontiguousarray(np.array([close[s:s + H] / close[s] - 1.0 for s in starts]))
UP, DN = 0.02, -0.02

def barrier_python(p, up, dn):
    out = []
    for row in p:
        lab, hit = 0, len(row)
        for j, v in enumerate(row):
            if v >= up:
                lab, hit = 1, j
                break
            if v <= dn:
                lab, hit = -1, j
                break
        out.append(lab * (hit + 1))
    return np.array(out)

def barrier_numpy(p, up, dn):                      # must evaluate every column
    u, d = p >= up, p <= dn
    fu = np.where(u.any(1), u.argmax(1), p.shape[1])
    fd = np.where(d.any(1), d.argmax(1), p.shape[1])
    lab = np.where(fu < fd, 1, np.where(fd < fu, -1, 0))
    return lab * (np.where(lab == 0, p.shape[1], np.minimum(fu, fd)) + 1)

@njit
def barrier_numba(p, up, dn):
    m, n = p.shape
    out = np.empty(m, np.int64)
    for i in range(m):
        lab, hit = 0, n
        for j in range(n):
            v = p[i, j]
            if v >= up:
                lab, hit = 1, j
                break
            if v <= dn:
                lab, hit = -1, j
                break
        out[i] = lab * (hit + 1)
    return out

sys.path.insert(0, "lab/hpc")
from barrier_cy import barrier_cython               # built by cythonize, see below

barrier_numba(paths, UP, DN)                        # compile before timing
times, ref = {}, barrier_python(paths, UP, DN)
for name, fn in [("pure Python", barrier_python), ("numpy", barrier_numpy),
                 ("numba", barrier_numba), ("Cython", barrier_cython)]:
    t0 = time.perf_counter()
    r = fn(paths, UP, DN)
    times[name] = time.perf_counter() - t0
    assert np.array_equal(np.asarray(r), ref), name
print(f"all four implementations agree exactly: True")
print(f"numpy buys less than 10x on this loop:  {times['pure Python'] / times['numpy'] < 10}")
print(f"numba buys more than 15x:               {times['pure Python'] / times['numba'] > 15}")
print(f"Cython lands within 30% of numba:       "
      f"{abs(times['Cython'] / times['numba'] - 1) < 0.3}")
# => all four implementations agree exactly: True
#    numpy buys less than 10x on this loop:  True
#    numba buys more than 15x:               True
#    Cython lands within 30% of numba:       True
```

| rung | time | speedup |
|---|---|---|
| pure Python | 118.3 ms | 1.0× |
| NumPy, vectorized with `argmax` | 20.7 ms | 5.7× |
| Numba `@njit` | 4.8 ms | **24.4×** |
| Cython, typed memoryviews | 4.8 ms | **24.4×** |

Two results deserve attention, and neither is the headline speedup. First, **the NumPy rung disappoints**: 5.7× where a well-suited vectorization routinely delivers fifty or more. The mechanism is the early exit. Most paths hit a barrier within the first few steps, so the loop versions examine three or four values and stop, while the array version dutifully evaluates all sixty columns for all 200,000 rows before running two `argmax` passes over the results — roughly fifteen times more arithmetic, executed fast. Vectorization converts a small amount of slow work into a large amount of fast work, which is a winning trade only when the work does not grow faster than the speedup.

Second, **Numba and Cython are indistinguishable** — 4.8 ms each, identical to a tenth of a millisecond. That is not a coincidence and it is the module's most useful practical fact: both compile the same nested loop to essentially the same machine code, so they are hitting the same floor, and no third tool will go meaningfully faster. Once you are at the floor, the choice among compiled options is entirely about engineering considerations — build complexity, deployment, debuggability, and who on the team can maintain it — and not at all about speed.

## Cython in practice: types are the whole story

The Cython version is the pure-Python function with type declarations added, compiled ahead of time:

```cython
# lab/hpc/barrier_cy.pyx  — compile with: cythonize -i barrier_cy.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np

def barrier_cython(double[:, ::1] paths, double up, double dn):
    cdef Py_ssize_t m = paths.shape[0], n = paths.shape[1], i, j, hit
    cdef int lab
    cdef double v
    cdef long long[:] out = np.empty(m, dtype=np.int64)
    for i in range(m):
        lab = 0
        hit = n
        for j in range(n):
            v = paths[i, j]
            if v >= up:
                lab = 1; hit = j; break
            if v <= dn:
                lab = -1; hit = j; break
        out[i] = lab * (hit + 1)
    return np.asarray(out)
```

Four details carry the performance. `double[:, ::1]` is a typed memoryview with a guaranteed C-contiguous last dimension, which lets the compiler emit direct pointer arithmetic instead of a buffer protocol call. The `cdef` declarations give every loop variable a machine type, so no Python objects are created inside the loop. `boundscheck=False` and `wraparound=False` remove per-access index validation and negative-index handling — genuinely unsafe directives that turn an `IndexError` into a segfault, and which should be enabled only after the code is correct and tested. The workflow that makes this tractable is `cythonize -a`, which emits an annotated HTML file colouring each line by how much Python interaction it retains: white lines compile to pure C, deep yellow lines still touch the interpreter. Optimizing Cython is largely the activity of removing yellow from the inner loop.

Compared with Numba, the trade is ahead-of-time compilation and a C extension you ship (no runtime compile, no LLVM dependency, works anywhere the wheel builds) against a build step, a `setup.py` or `pyproject` entry, and a separate language to learn. Numba requires no build system and stays in Python but pays a first-call compile cost, constrains you to the subset of Python it supports, and drags in LLVM. For a research codebase Numba's ergonomics usually win; for a library shipped to others, Cython's self-contained artifact usually does.

## Rust and PyO3: the same floor, different guarantees

Rust extensions have become the third standard option, and the honest framing given the measurements above is that they will land at the same 4.8 ms, because that is where the machine code lives. What Rust buys is not speed but *guarantees*: memory safety without a garbage collector, fearless concurrency inside the extension (no GIL applies to Rust threads), and a package manager that makes dependencies tractable. What it costs is a toolchain, a compilation model, and a language boundary your team must be able to cross.

This machine has no Rust toolchain, so the following is shown for shape rather than measured — the equivalent implementation and the build sequence:

```rust
// src/lib.rs  — illustrative; requires a Rust toolchain, not installed on this machine
use numpy::{PyReadonlyArray2, PyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn barrier<'py>(py: Python<'py>, paths: PyReadonlyArray2<f64>, up: f64, dn: f64)
    -> Bound<'py, PyArray1<i64>> {
    let p = paths.as_array();
    let (m, n) = (p.nrows(), p.ncols());
    let mut out = vec![0i64; m];
    for i in 0..m {
        let (mut lab, mut hit) = (0i64, n);
        for j in 0..n {
            let v = p[[i, j]];
            if v >= up { lab = 1; hit = j; break; }
            if v <= dn { lab = -1; hit = j; break; }
        }
        out[i] = lab * (hit as i64 + 1);
    }
    out.into_pyarray(py)
}

#[pymodule]
fn barrier_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(barrier, m)?)
}
```

```text
# illustrative — requires a Rust toolchain
$ pip install maturin
$ maturin develop --release
🔗 Found pyo3 bindings
🐍 Found CPython 3.12 at .venv/bin/python
   Compiling barrier_rs v0.1.0
    Finished `release` profile [optimized] target(s)
📦 Built wheel to /tmp/.tmpXXXX/barrier_rs-0.1.0-cp312-...whl
🛠 Installed barrier_rs-0.1.0
```

Two honest notes. Crossing the Python–Rust boundary costs roughly a microsecond per call, so an extension called once on a large array is free and one called per market-data tick is not — the same arithmetic as [the GPU module's kernel-launch overhead](08-gpu-acceleration-cuda.md), and the same conclusion: batch at the boundary. And a rewrite in any of these languages is only trustworthy with the equality assertion the ladder above ran on every rung: [Part IX's golden-file discipline](../part-09-software-engineering/02-testing-and-cicd.md) applies with more force here, not less, because a segfault-capable extension with `boundscheck=False` can be wrong in ways pure Python cannot.

## Benchmarks lie unless you make lying expensive

Every measurement in this module follows four rules, and each exists because violating it produces a specific, common error. **Warm up first**: Numba's first call includes compilation, which is seconds rather than milliseconds, and reporting it as run time is the most frequent JIT benchmarking mistake — the honest formulation when compilation matters is amortized, $t_{\text{total}}/n = t_{\text{compile}}/n + t_{\text{run}}$, so a JIT is worth it when $n$ is large and not when a function runs once. **Take the median of several runs**, not the mean, because the distribution has a long right tail from scheduler preemption and page faults. **Defeat dead-code elimination** by using the result — a benchmark of `v.sum()` that discards the sum can be optimized into nothing. And **hold data constant across variants**: the ladder above ran every implementation on the same `paths` array in the same process, which is what makes a 5.7× and a 24.4× comparable.

One more rule matters more than the others: **measure the whole program, not the kernel**. A function that becomes 24× faster while consuming 4% of runtime improves the program by 3%, which [Part IX's Amdahl arithmetic](../part-09-software-engineering/05-profiling-refactoring-versioning.md) makes precise. Sampling profilers answer this question in production conditions, where instrumentation would distort what it measures — `py-spy record --pid <pid>` attaches to a *running* process with no code change and no restart, which is how you profile a live trading system rather than a benchmark of one.

## Restructure, rewrite, or accept

The decision framework, in the order the questions should be asked. **Is it algorithmically slow?** This dominates everything else, and the measurement to prove it is the most striking in the module. A rolling maximum computed naively over a 500-wide window on 200,000 points takes 916.5 ms; the same result from a monotonic deque takes 22.4 ms — **41× faster, in pure Python, with identical output**:

```python
import time
from collections import deque
import numpy as np

def rolling_max_naive(x, w):
    return [max(x[i:i + w]) for i in range(len(x) - w + 1)]

def rolling_max_deque(x, w):
    dq, out = deque(), []
    for i, v in enumerate(x):
        while dq and x[dq[-1]] <= v:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - w:
            dq.popleft()
        if i >= w - 1:
            out.append(x[dq[0]])
    return out

xs = list(np.random.default_rng(0).standard_normal(200_000))
t0 = time.perf_counter(); a = rolling_max_naive(xs, 500); t_naive = time.perf_counter() - t0
t0 = time.perf_counter(); b = rolling_max_deque(xs, 500); t_deque = time.perf_counter() - t0
print(f"identical results: {a == b}")
print(f"the O(n) version is more than 20x faster, in pure Python: "
      f"{t_naive / t_deque > 20}")
# => identical results: True
#    the O(n) version is more than 20x faster, in pure Python: True
```

Forty-one times, beating the NumPy rung of the ladder and rivaling the compiled ones, from an algorithm change alone. Complexity beats constant factors whenever the input is large enough to notice, and no compiler will find this for you.

**Is it layout-bound?** If the operation is already vectorized and running well below memory bandwidth, restructure the data before rewriting the code; the stride experiment showed twelve-fold differences with no change to the arithmetic. **Does it dominate runtime?** Amdahl decides whether any speedup is worth having. **Only then, is it a tight loop that resists vectorization?** That is the ladder's territory, and the ladder says: reach for Numba first because it costs one decorator, use Cython when you need a shippable artifact, and consider Rust when memory safety or GIL-free threading inside the extension is worth a toolchain. **And is the current speed actually a problem?** A nightly research job that takes twenty minutes does not need to take one; engineering time spent making it faster is time not spent on the research the job exists to serve.

!!! warning "The compiler cannot fix your algorithm, and it is the cheapest thing you own"
    An $O(n)$ rewrite of a rolling maximum bought 41× in pure Python — more than vectorizing the barrier loop achieved (5.7×) and comparable to compiling it (24.4×). Meanwhile Numba and Cython landed at *identical* 4.8 ms, because both had reached the machine-code floor where no further tool helps. Ask in order: is it algorithmically slow, is it layout-bound, does it dominate runtime, and only then reach for a compiler — because the compiler multiplies your constant factor and leaves your exponent exactly where it was.

!!! abstract "Key takeaways"
    - The same scalar loop costs **18.1 ns per iteration interpreted and 0.37 ns compiled** — a 49× tax paid in object allocation, dynamic dispatch, and reference counting, not in the language itself.
    - Four threads on CPU-bound work delivered **0.97×** and four processes **3.70×**: use processes or Ray for computation, threads only for I/O, and note that NumPy and `nogil` extensions release the lock.
    - Useful bandwidth fell from **25.9 GB/s contiguous to 2.2 GB/s at stride 32** with identical arithmetic — layout is a performance decision, which is the case for struct-of-arrays over array-of-structs in hot paths.
    - On a loop with data-dependent early exit, NumPy bought only **5.7×** because the array form must evaluate all 60 columns for all 200,000 rows: vectorization trades a little slow work for a lot of fast work, and that trade can lose.
    - Numba and Cython both landed at **4.8 ms (24.4×)**, identical to a tenth of a millisecond — the machine-code floor, past which the choice of tool is about build systems and maintainability rather than speed.
    - Cython's speed comes from typed memoryviews, `cdef` declarations, and disabling bounds checks — the last being genuinely unsafe and appropriate only after tests pass; `cythonize -a` shows what still touches the interpreter.
    - Rust via PyO3 reaches the same floor and sells guarantees rather than speed — memory safety and GIL-free threading — at a per-call boundary cost of about a microsecond, so batch at the boundary.
    - A monotonic-deque rolling maximum beat the naive version **41× in pure Python** with identical output, outperforming vectorization of a comparable loop and rivaling compilation: fix the exponent before buying constant factors.

## Where this goes next

The other direction for the same bottleneck — thousands of independent runs rather than one faster kernel — is [Distributed Backtesting](09-distributed-backtesting.md), whose task-granularity arithmetic is the multi-machine analogue of the kernel-launch accounting here, and which shows why more compute makes the statistical problem worse even as it makes the engineering problem better. When the work is arithmetic-heavy and massively parallel rather than sequential, [GPU Acceleration with CUDA](08-gpu-acceleration-cuda.md) takes this same triple-barrier scan to a graphics card and measures where that bargain pays. The profiling foundation everything here rests on is [Part IX, lesson five](../part-09-software-engineering/05-profiling-refactoring-versioning.md), and the golden-file testing that makes any rewrite trustworthy is [Part IX, lesson two](../part-09-software-engineering/02-testing-and-cicd.md).
