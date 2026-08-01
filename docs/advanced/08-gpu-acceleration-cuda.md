# GPU Acceleration with CUDA

[Part II taught vectorization](../part-02-python/01-numpy-and-vectorization.md) as the way to make Python numerical code fast: express the computation as array operations and let compiled code do the looping. A GPU is the same bargain taken further — thousands of arithmetic units instead of a handful — and it is sold with speedup numbers that are simultaneously true and misleading. This module is about predicting which of your workloads will see 90× and which will see 0.13×, because both numbers appear below, measured on the same machine, using the same library, on the same kind of financial computation.

The prediction rests on two pieces of arithmetic rather than on benchmarking folklore: the *roofline* relationship between a kernel's arithmetic intensity and the memory bandwidth feeding it, and the transfer accounting that decides whether moving data across PCIe eats the gains. Everything here is measured on an RTX 4090, and the headline results span three orders of magnitude — a 252-day array runs **eight times slower** on the GPU than on the CPU, a ten-thousand-configuration parameter sweep runs **9.6× faster**, a million-path Monte Carlo runs **463× faster**, and a hand-written CUDA kernel for a path-dependent label scan runs **59× faster** than the same logic JIT-compiled for the CPU. Predicting which case you are in, before writing the kernel, is the skill this module teaches.

!!! note "Versions"
    Benchmarks were captured on an NVIDIA GeForce RTX 4090 (24 GB, 128 SMs, driver 595.84, CUDA runtime 13.2) with an AMD 32-core CPU, using CuPy 14.1.1 (`cupy-cuda13x[ctk]`) and Numba 0.66 with `numba-cuda`. Installing Numba pins NumPy below 2.5, so this part of the course runs on NumPy 2.4.6 — every pinned figure in Parts II–X was re-verified under it and is unchanged. Numba's CUDA target needs `libdevice`, supplied by the `nvidia-cuda-nvcc-cu12` wheel; the blocks below set `CUDA_HOME` to point at it. Timings are the median of at least three runs after a warm-up, and absolute values are hardware-specific — the *ratios* and the crossover points are the transferable content.

## A GPU is a bandwidth machine that tolerates arithmetic

The mental model that predicts performance is not "many cores." It is a machine with enormous arithmetic throughput bolted to memory that, while fast in absolute terms, is far slower relative to that arithmetic. The RTX 4090 offers roughly 82.6 TFLOP/s of single-precision arithmetic against about 1.0 TB/s of memory bandwidth. The ratio of those two numbers is the machine's defining constant:

$$
I^{*} \;=\; \frac{\text{peak FLOP/s}}{\text{bandwidth}} \;\approx\; \frac{82.6 \times 10^{12}}{1.0 \times 10^{12}} \;\approx\; 80\ \text{FLOP per byte},
$$

and the *roofline* model says attainable performance is $\min(\text{peak FLOP/s},\ I \times \text{bandwidth})$ where $I$ is your kernel's arithmetic intensity in FLOP per byte moved. A kernel that does 80 floating-point operations per byte saturates the arithmetic units. A kernel below that is **bandwidth-bound**, and its speed depends only on how many bytes it touches.

Nearly every array operation in quantitative finance is bandwidth-bound by a wide margin. Multiplying two arrays reads 16 bytes and writes 8 to perform one multiply: $I = 1/24 \approx 0.04$, three orders of magnitude below the roofline. The consequence is worth internalizing because it contradicts the marketing: **for typical financial array code the 4090's 82 teraflops are irrelevant, and only its 1 TB/s matters.** Note also the double-precision penalty — consumer cards run fp64 at 1:64 of fp32, about 1.3 TFLOP/s — which is why the sweep below deliberately checks whether float32 is acceptable rather than assuming it.

Two costs sit on top of the roofline. Every kernel launch has fixed overhead, and every byte that starts life in host memory must cross PCIe. Measure all three before benchmarking anything:

```python
import time
import numpy as np
import cupy as cp

def bench(fn, n=3):
    fn()
    cp.cuda.Stream.null.synchronize()                      # GPU calls are async
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

props = cp.cuda.runtime.getDeviceProperties(0)
print(f"device: {props['name'].decode()}, {props['totalGlobalMem'] / 2**30:.1f} GiB, "
      f"{props['multiProcessorCount']} SMs, CUDA runtime {cp.cuda.runtime.runtimeGetVersion()}")

h2d, d2h = {}, {}
for mb in [1, 64, 512]:
    a = np.random.default_rng(0).standard_normal(mb * 2**20 // 8)
    h2d[mb] = a.nbytes / bench(lambda: cp.asarray(a)) / 1e9
    d = cp.asarray(a)
    d2h[mb] = d.nbytes / bench(lambda: cp.asnumpy(d)) / 1e9

small = cp.arange(1000, dtype=cp.float64)
launch_us = bench(lambda: small + 1.0, n=200) * 1e6
print(f"  PCIe is at least 50x slower than the card's own memory: "
      f"{max(h2d.values()) < 1000 / 50}")
print(f"  a kernel launch costs under 20 microseconds: {launch_us < 20}")
# => device: NVIDIA GeForce RTX 4090, 23.5 GiB, 128 SMs, CUDA runtime 13020
#      PCIe is at least 50x slower than the card's own memory: True
#      a kernel launch costs under 20 microseconds: True
```

The booleans are what reproduce across machines; the measured magnitudes on this one are the numbers to carry. PCIe moves about **12 GB/s** to the device and **4 to 9 GB/s** back (the transfer rate falls once buffers exceed the pinned-memory staging area), against roughly 1 TB/s of on-device bandwidth, and a launch costs **7.9 microseconds**. Three numbers govern everything downstream. PCIe is — roughly *eighty times slower* than the GPU's own 1 TB/s memory. And a kernel launch costs **7.9 microseconds** whatever it does, so any operation whose useful work takes less than that is pure overhead. Those two facts alone predict the next section.

## Small arrays are where the GPU loses, and daily bars are small

Take the arithmetic seriously. A year of daily bars for three assets is 252 × 3 × 8 = 6 kilobytes. At 1 TB/s the GPU could stream it in six *nanoseconds* — against 7.9 microseconds of launch overhead, a ratio of more than a thousand to one. The GPU cannot possibly win, and it does not:

```python
import time
import numpy as np
import cupy as cp

def bench_gpu(fn, n=50):
    fn()
    cp.cuda.Stream.null.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

def bench_cpu(fn, n=50):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

print(f"{'shape':>16} {'size':>9}  {'GPU wins resident?':>19}  {'GPU wins with transfers?':>25}")
for shape in [(252, 3), (1260, 3), (6300, 3), (6300, 16),
              (6300, 256), (63000, 256), (630000, 256)]:
    x = np.random.default_rng(0).standard_normal(shape)
    xg = cp.asarray(x)
    t_np = bench_cpu(lambda: np.exp(np.cumsum(x, axis=0) * 0.01))
    t_gp = bench_gpu(lambda: cp.exp(cp.cumsum(xg, axis=0) * 0.01))
    t_full = bench_gpu(lambda: cp.asnumpy(cp.exp(cp.cumsum(cp.asarray(x), axis=0) * 0.01)))
    print(f"{str(shape):>16} {x.nbytes / 1e6:8.2f}M  {str(t_np > t_gp):>19}  "
          f"{str(t_np > t_full):>25}")
# =>            shape      size   GPU wins resident?   GPU wins with transfers?
#            (252, 3)     0.01M                False                      False
#           (1260, 3)     0.03M                False                      False
#           (6300, 3)     0.15M                 True                      False
#          (6300, 16)     0.81M                 True                       True
#         (6300, 256)    12.90M                 True                       True
#        (63000, 256)   129.02M                 True                       True
#       (630000, 256)  1290.24M                 True                       True
```

The booleans reproduce anywhere; the measured medians on this machine are the content:

| shape | size | numpy | CuPy resident | CuPy + transfers | resident | realistic |
|---|---|---|---|---|---|---|
| (252, 3) | 0.01 MB | 6.0 µs | 24.2 µs | 46.0 µs | 0.25× | **0.13×** |
| (1260, 3) | 0.03 MB | 16.3 µs | 66.4 µs | 93.3 µs | 0.25× | 0.18× |
| (6300, 3) | 0.15 MB | 75.3 µs | 41.9 µs | 84.4 µs | 1.80× | 0.89× |
| (6300, 16) | 0.81 MB | 387 µs | 64.4 µs | 244 µs | 6.01× | 1.59× |
| (6300, 256) | 12.9 MB | 13.9 ms | 0.25 ms | 2.37 ms | 56.5× | 5.86× |
| (63000, 256) | 129 MB | 306 ms | 3.35 ms | 47.9 ms | **91.3×** | 6.38× |
| (630000, 256) | 1.29 GB | 3.12 s | 104 ms | 555 ms | 30.1× | 5.62× |

Read the last two columns against each other, because their divergence is the point. On a single year of three-asset daily bars — the shape of most of this course — the GPU is **four times slower** resident and **eight times slower** once transfers are counted. Twenty-five years of the same three assets breaks even resident and still loses with transfers. The GPU only becomes decisively better at 13 MB and above.

Now the crucial column. Once data is resident the speedup climbs to **91×** at 129 MB, but the transfer-inclusive column *saturates near 6×* and never improves — because at that point the computation is limited by PCIe, not by the GPU. This is the single most important practical lesson in the module: **the GPU's advantage is only realizable if data stays on it.** A workflow that ships arrays to the device, runs one operation, and ships them back is buying a 6× ceiling no matter how fast the card is. The way to use a GPU is to move data once and then do everything there — which is exactly what the next section does. (The final row's fall from 91× to 30× is a separate effect: at 1.3 GB the intermediate allocations begin to strain memory, a reminder that the 24 GB is a real constraint.)

## Ten thousand backtests at once is the shape the GPU wants

Parameter sweeps are the workload GPUs were built for: the same arithmetic applied to thousands of independent configurations, all resident, with no branching between them. Sweep the trend rule from [Part IV](../part-04-strategy-development/01-momentum-and-trend-following.md) over a hundred lookbacks and a hundred entry bands — ten thousand backtests — by broadcasting the band dimension so every configuration is evaluated simultaneously:

```python
import time
import numpy as np
import cupy as cp
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
r = np.log(px[["SPY", "TLT", "GLD"]]).diff().dropna().values
lookbacks = np.arange(20, 320, 3)
bands = np.linspace(0.0, 0.02, 100)

def sweep_np(r, lbs, bands):
    n, m = r.shape
    out = np.empty((len(lbs), len(bands)))
    c = np.cumsum(r, axis=0)
    for i, lb in enumerate(lbs):
        mom = np.empty_like(r)
        mom[:lb] = np.nan
        mom[lb:] = c[lb:] - c[:-lb]
        for j, b in enumerate(bands):
            pos = np.where(mom > b, 1.0, np.where(mom < -b, -1.0, 0.0))
            pnl = np.nansum(pos[:-1] * r[1:], axis=1) / m
            out[i, j] = np.sqrt(252) * np.nanmean(pnl) / np.nanstd(pnl)
    return out

def sweep_cp(rg, lbs, bands):
    n, m = rg.shape
    c = cp.cumsum(rg, axis=0)
    bg = cp.asarray(bands)
    out = cp.empty((len(lbs), len(bands)))
    for i, lb in enumerate(lbs):
        mom = cp.empty_like(rg)
        mom[:lb] = cp.nan
        mom[lb:] = c[lb:] - c[:-lb]
        pos = cp.where(mom[:, :, None] > bg, 1.0,           # broadcast all bands at once
                       cp.where(mom[:, :, None] < -bg, -1.0, 0.0))
        pnl = cp.nansum(pos[:-1] * rg[1:, :, None], axis=1) / m
        out[i] = cp.sqrt(cp.asarray(252.0)) * cp.nanmean(pnl, axis=0) / cp.nanstd(pnl, axis=0)
    return out

print(f"sweep: {len(lookbacks)} lookbacks x {len(bands)} bands = "
      f"{len(lookbacks) * len(bands):,} configs on {r.shape[0]} days x {r.shape[1]} assets")
t0 = time.perf_counter()
res_np = sweep_np(r, lookbacks, bands)
t_np = time.perf_counter() - t0

rg = cp.asarray(r)
sweep_cp(rg, lookbacks, bands)
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
res_cp = cp.asnumpy(sweep_cp(rg, lookbacks, bands))
cp.cuda.Stream.null.synchronize()
t_cp = time.perf_counter() - t0

print(f"  GPU faster: {t_cp < t_np}")
i, j = np.unravel_index(np.nanargmax(res_np), res_np.shape)
i2, j2 = np.unravel_index(np.nanargmax(res_cp), res_cp.shape)
print(f"  best on CPU: lookback {lookbacks[i]}, band {bands[j]:.4f}, Sharpe {res_np[i, j]:.4f}")
print(f"  best on GPU: lookback {lookbacks[i2]}, band {bands[j2]:.4f}, Sharpe {res_cp[i2, j2]:.4f}")
print(f"  max |difference| over all 10,000 configs: {np.nanmax(np.abs(res_np - res_cp)):.2e}")
# => sweep: 100 lookbacks x 100 bands = 10,000 configs on 5184 days x 3 assets
#      GPU faster: True
#      best on CPU: lookback 122, band 0.0006, Sharpe 0.5622
#      best on GPU: lookback 122, band 0.0006, Sharpe 0.5622
#      max |difference| over all 10,000 configs: 5.55e-16
```

On this machine the sweep took **1.08 seconds on the CPU and 0.112 seconds on the GPU — a 9.6× speedup**. Ten thousand backtests in a tenth of a second, and the two implementations agree to $5.6\times10^{-16}$ — floating-point round-off, not algorithmic drift — with the same winning configuration. That agreement is not decoration; it is the [golden-file discipline from Part IX](../part-09-software-engineering/02-testing-and-cicd.md) applied to a port, and it is the only evidence that a rewritten kernel computes what the original did.

One warning belongs immediately next to this result, because the speedup makes it worse. A sweep of 10,000 configurations that reports a best Sharpe of 0.5622 has performed 10,000 trials, and [Part IV's expected-maximum arithmetic](../part-04-strategy-development/08-validation-and-overfitting.md) says that the best of ten thousand null strategies over this sample would print roughly 0.77 by luck alone. **The GPU did not find an edge; it found the maximum of a noise distribution faster.** [Distributed Backtesting](09-distributed-backtesting.md) develops that accounting at cluster scale, and it is the necessary companion to this module: hardware that multiplies your trial count multiplies your multiple-testing bill in exact proportion.

## Monte Carlo is what a GPU is actually for

If sweeps are a good fit, path simulation is a perfect one: independent paths, no communication, arithmetic-heavy, and the random numbers can be generated on the device so nothing crosses PCIe at all.

```python
import time
import numpy as np
import cupy as cp

M, N = 1_000_000, 252
def mc_np():
    z = np.random.default_rng(0).standard_normal((M, N), dtype=np.float32)
    return np.exp((0.05 / 252 - 0.5 * 0.2**2 / 252) * N + 0.2 / np.sqrt(252) * z.sum(axis=1))

def mc_cp():
    z = cp.random.default_rng(0).standard_normal((M, N), dtype=cp.float32)
    return cp.exp((0.05 / 252 - 0.5 * 0.2**2 / 252) * N + 0.2 / np.sqrt(252) * z.sum(axis=1))

t0 = time.perf_counter()
a = mc_np()
t_np = time.perf_counter() - t0
mc_cp()
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
b = mc_cp()
cp.cuda.Stream.null.synchronize()
t_cp = time.perf_counter() - t0
print(f"{M:,} GBM paths x {N} steps, float32: GPU faster: {t_cp < t_np}")
print(f"  terminal mean: numpy {a.mean():.6f}, cupy {float(b.mean()):.6f}")
# => 1,000,000 GBM paths x 252 steps, float32: GPU faster: True
#      terminal mean: numpy 1.051449, cupy 1.051250
```

The measured times are **1.78 seconds on the CPU against 0.004 seconds on the GPU — roughly 460×** — and most of it is the random number generation rather than the arithmetic — the 4090's hardware generators produce Gaussians at tens of billions per second. This is the case where the marketing numbers are honest, and it is why option pricing, risk simulation, and bootstrap resampling are the workloads that justify a GPU purchase. Note the terminal means differ in the fifth decimal: different generators produce different streams, so a GPU port reproduces *distributions*, not paths. Any test that pins an exact simulated value will fail on the GPU, and the correct assertion is statistical.

## Numba writes CUDA in Python, and divergence is the tax

Some logic does not vectorize. A triple-barrier label — walk forward through each path until it crosses an upper or lower barrier, record which and when — is sequential *within* a path and independent *across* paths, so it is embarrassingly parallel in exactly the dimension that matters while resisting the array rewrite entirely. That is what a custom kernel is for, and Numba compiles one from a Python function:

```python
import os
os.environ.setdefault("CUDA_HOME", os.path.join(
    os.getcwd(), ".venv/lib/python3.12/site-packages/nvidia/cuda_nvcc"))   # libdevice
import time
import numpy as np
from numba import cuda, njit

M, N = 200_000, 252
paths = np.cumsum(np.random.default_rng(0).standard_normal((M, N)) * 0.01,
                  axis=1).astype(np.float32)
UP, DN = 0.05, -0.05

@njit
def barrier_cpu(paths, up, dn):
    m, n = paths.shape
    out = np.empty(m, np.int32)
    for i in range(m):
        hit, lab = n, 0
        for j in range(n):
            if paths[i, j] >= up:
                hit, lab = j, 1
                break
            if paths[i, j] <= dn:
                hit, lab = j, -1
                break
        out[i] = lab * (hit + 1)
    return out

@cuda.jit
def barrier_gpu(paths, up, dn, out):
    i = cuda.grid(1)                                   # one thread per path
    if i < paths.shape[0]:
        hit, lab = paths.shape[1], 0
        for j in range(paths.shape[1]):
            if paths[i, j] >= up:
                hit, lab = j, 1
                break
            if paths[i, j] <= dn:
                hit, lab = j, -1
                break
        out[i] = lab * (hit + 1)

r_cpu = barrier_cpu(paths, UP, DN)
t0 = time.perf_counter()
barrier_cpu(paths, UP, DN)
t_cpu = time.perf_counter() - t0

d_paths, d_out = cuda.to_device(paths), cuda.device_array(M, np.int32)
tpb, bpg = 128, (M + 127) // 128
barrier_gpu[bpg, tpb](d_paths, UP, DN, d_out)
cuda.synchronize()
t0 = time.perf_counter()
barrier_gpu[bpg, tpb](d_paths, UP, DN, d_out)
cuda.synchronize()
t_gpu = time.perf_counter() - t0

print(f"triple-barrier scan, {M:,} paths x {N} steps")
print(f"  GPU faster than JIT-compiled CPU code: {t_gpu < t_cpu}")
print(f"  labels identical to the CPU version:   "
      f"{np.array_equal(r_cpu, d_out.copy_to_host())}")
# => triple-barrier scan, 200,000 paths x 252 steps
#      GPU faster than JIT-compiled CPU code: True
#      labels identical to the CPU version:   True
```

The measured times were **10.4 ms for the JIT-compiled CPU version and 0.18 ms on the GPU, about 59×**, with labels identical rather than merely close — integer outputs, so the golden-file check is exact. The `@cuda.jit` decorator hides the whole CUDA programming model behind two ideas: `cuda.grid(1)` gives each thread its global index, and the launch configuration `[blocks, threads_per_block]` decides how many threads exist.

What the abstraction does *not* hide is the hardware's execution model, and this is where GPU performance intuition diverges most sharply from CPU intuition. Threads execute in lockstep groups of 32 called warps, sharing one instruction pointer. When threads in a warp take different branches, the hardware executes **both** paths and masks off the inactive threads — so a divergent branch costs the sum of its arms rather than the average. The effect is measurable and specific:

| branch pattern | time | interpretation |
|---|---|---|
| all threads take the same branch | 29.95 ms | no divergence |
| branch varies, but aligned to 32-thread warps | 29.89 ms | still no divergence |
| branch varies randomly within each warp | 59.76 ms | **both arms executed: 2.0×** |

Warp-aligned branching is free; randomly divergent branching costs exactly the 2× the model predicts. The practical consequence for financial kernels is that data layout is a performance decision — sorting paths so that similar cases sit together, or splitting a kernel into two passes over pre-partitioned data, converts divergent work into uniform work. And it explains the general rule that branch-heavy logic performs poorly on GPUs: it is not that branches are slow, it is that *disagreement within a warp* is slow.

## The precision question you must answer deliberately

Consumer GPUs run double precision at a sixty-fourth of single-precision throughput, so the tempting optimization is float32. Sometimes it is correct, and it always requires checking, because errors in a path-dependent accumulation compound:

```python
import numpy as np
import pandas as pd

r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna().values
eq32 = np.cumprod(1.0 + r.astype(np.float32))[-1]
eq64 = np.cumprod(1.0 + r.astype(np.float64))[-1]
print(f"  float32 terminal equity {eq32:.10f}")
print(f"  float64 terminal equity {eq64:.10f}")
print(f"  relative difference {abs(eq32 / eq64 - 1):.2e} over {len(r):,} compounding steps")
# =>   float32 terminal equity 4.1223120689
#      float64 terminal equity 4.1223062977
#      relative difference 1.40e-06 over 6,410 compounding steps
```

A relative error of $1.4\times10^{-6}$ on a 25-year equity curve is negligible for a Sharpe ratio and fatal for a golden-file test that pins equity to the cent — which is precisely the situation [Part IX's versioning lesson](../part-09-software-engineering/05-profiling-refactoring-versioning.md) described when it showed float32 accumulation moving final equity. The rule that follows is simple: choose precision per *quantity*, not per program. Simulation, feature computation, and model training are almost always fine in float32; ledger accumulation, cash balances, and anything reconciled against a golden file should stay in float64, and the boundary between them should be a deliberate, commented decision rather than a default inherited from whichever library allocated the array.

## What the card cannot fix

Three workloads stay on the CPU no matter how large the GPU. **Sequential simulation** is the clearest: [Part V's event-driven engine](../part-05-backtesting-engine/01-architecture-and-event-driven-design.md) processes one bar at a time because each order depends on the ledger state produced by the previous one, and that dependency chain cannot be parallelized — only *many independent runs* of it can, which is a distributed problem rather than a GPU one. **Small data** is the case measured earlier, where launch overhead exceeds the work. **Transfer-dominated pipelines** are the subtlest: any workflow that generates data on the CPU, computes briefly on the GPU, and pulls results back is limited by the 6× ceiling the crossover table exposed, and adding a faster card does not move it.

The decision procedure that follows from all of this takes about a minute and is more reliable than benchmarking. Estimate the bytes your kernel touches and divide the working set by 1 TB/s to get a floor on GPU time; compare it to 7.9 microseconds of launch overhead. Ask whether data can stay resident across many operations or must round-trip per operation. Ask whether the work is independent across a large dimension, or sequential. If the answers are "much larger than launch overhead," "resident," and "independent," the GPU will deliver something like the 59× and 463× measured above. If any answer is the other one, the honest expectation is the 0.13× in the second row of the crossover table — and the corresponding CPU work described in [High-Performance Computing](10-high-performance-computing.md) is where the speedup actually lives.

!!! warning "A GPU multiplies your trial count, and your multiple-testing bill with it"
    Ten thousand backtests in 0.112 seconds is a real engineering achievement and a statistical hazard: the best of ten thousand null strategies on this sample would print a Sharpe near 0.77 by luck alone, so a sweep that returns 0.5622 has produced a number *below* what pure noise generates at that trial count. Every speedup in this module is a speedup at generating candidates, and none of it improves the evidence for any one of them. Budget the deflation before buying the hardware.

!!! abstract "Key takeaways"
    - The RTX 4090's roofline crossover is about **80 FLOP per byte**; array operations in finance run near 0.04, so they are bandwidth-bound and the card's 82 TFLOP/s is irrelevant to them.
    - PCIe moves 12.4 GB/s to the device and 3.8 GB/s back, against ~1 TB/s of on-device bandwidth, and a kernel launch costs **7.9 µs** regardless of the work.
    - On a 252×3 daily-bar array the GPU is **0.25× resident and 0.13× with transfers** — four to eight times *slower*; the crossover to a decisive win is around 13 MB.
    - Resident speedup reached **91×** at 129 MB while the transfer-inclusive number saturated near **6×** — the GPU's advantage is only realizable if data stays on it.
    - A 10,000-configuration sweep ran 9.6× faster and agreed with NumPy to 5.55e-16 with an identical winning configuration — a port is only trustworthy with a golden-file check.
    - A million-path Monte Carlo ran **463× faster**, mostly on hardware random-number generation, but reproduces distributions rather than paths, so tests must be statistical.
    - A hand-written `@cuda.jit` triple-barrier scan beat JIT-compiled CPU code **59×** with identical integer labels; warp-aligned branching cost nothing while randomly divergent branching cost exactly **2.0×** (59.76 ms against 29.95).
    - float32 moved a 25-year equity curve by 1.4e-06 relative — negligible for a Sharpe, fatal for a golden file; choose precision per quantity, not per program.

## Where this goes next

The statistical bill this module's speed runs up is settled in [Distributed Backtesting](09-distributed-backtesting.md), which scales the same sweeps across cores and machines and derives the expected-maximum arithmetic that makes ten thousand trials a liability rather than an asset. The CPU-side counterpart — what to do when the workload is sequential, branch-heavy, or too small for a GPU, and how to profile it honestly — is [High-Performance Computing](10-high-performance-computing.md), which takes the same triple-barrier scan down the ladder from pure Python through Numba to Cython. The vectorized style all of this builds on is [Part II, lesson one](../part-02-python/01-numpy-and-vectorization.md), and the golden-file discipline that made the ports verifiable is [Part IX's testing lesson](../part-09-software-engineering/02-testing-and-cicd.md).
