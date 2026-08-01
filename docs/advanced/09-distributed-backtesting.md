# Distributed Backtesting

[Part V built an event-driven engine](../part-05-backtesting-engine/index.md) that runs one strategy over one history, and every part since has wanted to run it many times — across parameter grids, across bootstrap replications, across walk-forward folds. That workload is embarrassingly parallel, so scaling it across thirty-two cores or three hundred is an engineering problem with well-understood answers, and the first half of this module gives them: task granularity, data locality through an object store, determinism as a protocol rather than a hope, and resumable sweeps.

The second half is the reason the module sits beside [Part IV's validation lesson](../part-04-strategy-development/08-validation-and-overfitting.md) rather than in the software-engineering part. A cluster is a machine for producing candidate strategies, and candidates are exactly what a research process must be *skeptical* of. The arithmetic is stark and it can be computed before the cluster is switched on: with twenty-five years of daily data, the best of ten thousand pure-noise strategies has an expected Sharpe of **0.77**. An overnight sweep therefore ships something that looks like a viable strategy whether or not any edge exists, and the correction has to scale with the core count or the speed is worse than useless. This module ends by running that accounting on a real ten-thousand-configuration sweep, where the answer turns out to be subtler than either "it is all noise" or "we found something."

!!! note "Versions"
    This module adds Ray 2.56.1 and Dask 2026.7.1 to the stack, measured on a 32-logical-core machine (16 physical cores with simultaneous multithreading) with 62 GB of RAM, on NumPy 2.4.6 and pandas 3.0.5. Wall-clock timings are hardware-specific; the *shapes* — where efficiency falls off, how the null grows with trial count — are what transfers. Code blocks print reproducible facts, and measured times appear in the prose tables beside them.

## A cluster is a multiple-testing machine with an electricity bill

Start with the statistics, because they determine whether the engineering is worth doing. A Sharpe ratio estimated over $T$ years has a standard error of roughly $1/\sqrt{T}$, so twenty-five years of daily data gives $\sigma_{S} = 0.20$ even for a strategy with no edge whatsoever. Draw $N$ such strategies and the expected maximum follows the standard extreme-value approximation for Gaussians,

$$
\mathbb{E}\bigl[\max_N\bigr] \;\approx\; \sigma_S\left[(1 - \gamma_E)\,\Phi^{-1}\!\left(1 - \frac{1}{N}\right) \;+\; \gamma_E\,\Phi^{-1}\!\left(1 - \frac{1}{N e}\right)\right],
$$

where $\gamma_E \approx 0.5772$ is the Euler–Mascheroni constant. This is the same machinery behind [the deflated Sharpe ratio](../part-04-strategy-development/08-validation-and-overfitting.md); what changes at cluster scale is only $N$, and $N$ is now a function of your hardware budget:

```python
import numpy as np
from scipy import stats

def emax(n):                                     # E[max of n standard normals]
    return ((1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / n)
            + np.euler_gamma * stats.norm.ppf(1 - 1 / (n * np.e)))

sigma_s = 1 / np.sqrt(25)                        # 25 years of daily data
print(f"standard error of a Sharpe over 25 years: {sigma_s:.2f}")
for n in [10, 100, 1_000, 10_000, 100_000]:
    print(f"  best of {n:>7,} strategies with no edge: E[max Sharpe] = {sigma_s * emax(n):.2f}")

rng = np.random.default_rng(0)
T = 25 * 252
for n in [1_000, 10_000]:
    best = []
    for _ in range(20):
        r = rng.standard_normal((n, T)) * 0.01     # pure noise, by construction
        sh = np.sqrt(252) * r.mean(axis=1) / r.std(axis=1)
        best.append(sh.max())
    print(f"  simulated n = {n:>6,}: E[max] {np.mean(best):.3f} (theory {sigma_s * emax(n):.3f})")
# => standard error of a Sharpe over 25 years: 0.20
#      best of      10 strategies with no edge: E[max Sharpe] = 0.31
#      best of     100 strategies with no edge: E[max Sharpe] = 0.51
#      best of   1,000 strategies with no edge: E[max Sharpe] = 0.65
#      best of  10,000 strategies with no edge: E[max Sharpe] = 0.77
#      best of 100,000 strategies with no edge: E[max Sharpe] = 0.88
#      simulated n =  1,000: E[max] 0.660 (theory 0.651)
#      simulated n = 10,000: E[max] 0.754 (theory 0.772)
```

Simulation confirms the formula to within its own sampling error, and the table is the exchange rate between compute and self-deception. Ten strategies buy a 0.31 by luck; ten thousand buy **0.77** — a number most investors would fund. Note the shape: because the growth is like $\sqrt{2\ln N}$, going from ten thousand to one hundred thousand adds only 0.11, so a tenfold larger cluster does not make the problem tenfold worse. It makes it *slightly* worse and much faster, which is the genuinely dangerous combination — the marginal cost of another thousand trials feels like nothing, and the accounting is what has to hold the line.

## Sweeps live in Gustafson's regime, not Amdahl's

[Part IX's profiling lesson](../part-09-software-engineering/05-profiling-refactoring-versioning.md) applied Amdahl's law: with a fixed workload and a serial fraction $1-f$, speedup saturates at $1/(1-f)$ no matter how many cores you add. That is the correct model for making *one* backtest faster, and it is pessimistic. Sweeps are a different regime. Gustafson's observation is that in practice we do not hold the work fixed and add cores — we hold the *time* fixed and use cores to do more work, so scaled speedup is

$$
S(p) \;=\; p - \alpha\,(p - 1),
$$

with $\alpha$ the serial fraction of the scaled workload. Nobody buys a cluster to run yesterday's grid faster; they run a bigger grid in the same overnight window, and that is why the multiple-testing bill grows with the hardware.

Between the theory and the measurement sits task granularity. Every distributed task has fixed overhead — scheduling, serialization, result transport — so efficiency is roughly

$$
e \;=\; \frac{t_{\text{task}}}{t_{\text{task}} + t_{\text{overhead}}},
$$

and the practical rule follows immediately: measure your framework's overhead, then batch until each task runs at least a hundred times longer than it. Ray's overhead on this machine is small but not zero, and the sweep from [the GPU module](08-gpu-acceleration-cuda.md) is instructive precisely because it is *too small* to distribute well — 10,000 configurations that take 0.4 seconds in total leave nothing for thirty-two workers to do.

## Ray in one page: tasks, the object store, and locality

Ray's model is three ideas. A function decorated with `@ray.remote` becomes a task submitted to a scheduler; calling it returns a future immediately and `ray.get` blocks for results. Large shared data goes into the object store once via `ray.put`, and workers access it by reference without re-serializing — the difference between shipping a 100 MB price panel to every task and shipping a pointer. And results come back in completion order, which is why determinism needs a protocol.

Give the workers real work — for each configuration, a Sharpe ratio plus fifty block-bootstrap replications, which is what an honest sweep computes anyway — and measure the scaling:

```python
import time
import numpy as np
import pandas as pd
import ray

ray.init(num_cpus=32, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)

px = pd.read_parquet("data/prices.parquet")
R = np.log(px[["SPY", "TLT", "GLD"]]).diff().dropna().values
R_ref = ray.put(R)                                     # one copy, shared by reference
CONFIGS = [(lb, round(b, 5)) for lb in range(20, 320, 3)
           for b in np.linspace(0.0, 0.02, 100)]

@ray.remote
def evaluate(r, batch, n_boot, seed):
    rng = np.random.default_rng(seed)                  # seed passed in, never derived from index
    c = np.cumsum(r, axis=0)
    out = []
    for lb, band in batch:
        mom = np.empty_like(r)
        mom[:lb] = np.nan
        mom[lb:] = c[lb:] - c[:-lb]
        pos = np.where(mom > band, 1.0, np.where(mom < -band, -1.0, 0.0))
        pnl = np.nansum(pos[:-1] * r[1:], axis=1) / r.shape[1]
        pnl = pnl[~np.isnan(pnl)]
        sh = np.sqrt(252) * pnl.mean() / pnl.std()
        blk, m = 21, len(pnl)
        boot = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, m - blk, m // blk)
            samp = np.concatenate([pnl[j:j + blk] for j in idx])
            boot[i] = np.sqrt(252) * samp.mean() / samp.std()
        out.append(((lb, band), float(sh), float((boot >= sh).mean())))
    return out

def batches(items, k):
    size = (len(items) + k - 1) // k
    return [items[i:i + size] for i in range(0, len(items), size)]

@ray.remote
def noop():
    return 0
ray.get([noop.remote() for _ in range(100)])           # warm up
t0 = time.perf_counter()
ray.get([noop.remote() for _ in range(2000)])
overhead_ms = (time.perf_counter() - t0) / 2000 * 1e3
print(f"Ray per-task overhead is under 1 ms: {overhead_ms < 1.0}")

times = {}
for p in [1, 2, 4, 8, 16, 32]:
    bs = batches(CONFIGS, p)
    sd = np.random.SeedSequence(0).spawn(len(bs))
    t0 = time.perf_counter()
    ray.get([evaluate.remote(R_ref, b, 50, s) for b, s in zip(bs, sd)])
    times[p] = time.perf_counter() - t0
print(f"32 workers beat 1 worker by more than 8x: {times[1] / times[32] > 8}")
print(f"efficiency falls below 60% by 16 workers: {times[1] / times[16] / 16 < 0.6}")
ray.shutdown()
# => Ray per-task overhead is under 1 ms: True
#    32 workers beat 1 worker by more than 8x: True
#    efficiency falls below 60% by 16 workers: True
```

Ray's per-task overhead measures **0.05 ms** here, so a task doing 10 ms of work runs at a 99% efficiency ceiling and a task doing 0.05 ms runs at 50%. The measured scaling on the properly-sized workload:

| workers | wall clock | speedup | efficiency |
|---|---|---|---|
| 1 | 32.73 s | 1.00× | 100% |
| 2 | 16.47 s | 1.99× | 99.4% |
| 4 | 8.89 s | 3.68× | 92.0% |
| 8 | 5.05 s | 6.48× | 81.0% |
| 16 | 4.17 s | 7.85× | 49.0% |
| 32 | 3.44 s | 9.51× | 29.7% |

Near-perfect scaling to four workers, good scaling to eight, and then a collapse in efficiency that has a specific and unglamorous cause: this machine has 32 *logical* cores but 16 *physical* ones, so beyond sixteen the workers are sharing execution units rather than owning them. Fitting the universal scalability law,

$$
X(p) \;=\; \frac{\lambda\,p}{1 + \sigma\,(p - 1) + \kappa\,p\,(p - 1)},
$$

gives contention $\sigma = 0.0448$ and coherency $\kappa = 0.00103$, which predicts peak throughput at about **30 workers** — beyond that, adding workers makes the sweep slower. That the fit lands so close to the physical core count is the point of running it: the USL is a way to discover your machine's real ceiling from measurements rather than from the number in the specification sheet.

## Determinism is a protocol, not a hope

Distributed results arrive in completion order, which varies run to run. Any aggregation that depends on arrival order — a running maximum, an accumulator, a "first result wins" rule — produces different answers on identical inputs, and the resulting irreproducibility is indistinguishable from a bug. Three rules make a sweep bit-reproducible, and all three appear in the code above.

**Seed by spawning, not by arithmetic.** `np.random.SeedSequence(0).spawn(k)` produces $k$ statistically independent streams; deriving seeds as `seed = base + task_index` produces streams that can be correlated, which silently degrades bootstrap estimates. **Return the key with the value.** Every task emits `(config, result)` pairs so nothing depends on which worker finished when. **Sort before reducing.** Aggregate only after sorting by key, and hash the sorted table to prove it:

```python
import hashlib
import json
import time
import numpy as np
import pandas as pd
import ray

ray.init(num_cpus=32, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
px = pd.read_parquet("data/prices.parquet")
R = np.log(px[["SPY", "TLT", "GLD"]]).diff().dropna().values
R_ref = ray.put(R)
CONFIGS = [(lb, round(b, 5)) for lb in range(20, 320, 3)
           for b in np.linspace(0.0, 0.02, 100)]

@ray.remote
def sharpe_batch(r, batch):
    c = np.cumsum(r, axis=0)
    out = []
    for lb, band in batch:
        mom = np.empty_like(r)
        mom[:lb] = np.nan
        mom[lb:] = c[lb:] - c[:-lb]
        pos = np.where(mom > band, 1.0, np.where(mom < -band, -1.0, 0.0))
        pnl = np.nansum(pos[:-1] * r[1:], axis=1) / r.shape[1]
        out.append(((lb, band), float(np.sqrt(252) * np.nanmean(pnl) / np.nanstd(pnl))))
    return out

def batches(items, k):
    size = (len(items) + k - 1) // k
    return [items[i:i + size] for i in range(0, len(items), size)]

def run():
    res = ray.get([sharpe_batch.remote(R_ref, b) for b in batches(CONFIGS, 128)])
    return [kv for r in res for kv in r]

def table_hash(rows):
    blob = json.dumps([[list(k), round(v, 10)] for k, v in sorted(rows)]).encode()
    return hashlib.sha256(blob).hexdigest()[:16]

r1, r2 = run(), run()
print(f"two independent runs produce the same sorted-table hash: "
      f"{table_hash(r1) == table_hash(r2)}")
best_cfg, best_sh = max(r1, key=lambda kv: kv[1])
print(f"best of {len(CONFIGS):,} configs: lookback {best_cfg[0]}, band {best_cfg[1]:.5f}, "
      f"Sharpe {best_sh:.4f}")
ray.shutdown()
# => two independent runs produce the same sorted-table hash: True
#    best of 10,000 configs: lookback 122, band 0.00061, Sharpe 0.5627
```

Identical hashes (`848e81b2e32d58b0` on both runs), and a best configuration that agrees with [the GPU module's independent implementation](08-gpu-acceleration-cuda.md) — lookback 122, Sharpe 0.5627 against 0.5622 there, the difference being band rounding. A sweep that cannot reproduce its own hash cannot be debugged, and this check costs three lines.

Two operational corollaries follow. Sweeps fail partway — a worker is preempted, a node runs out of memory, a spot instance is reclaimed — so results should be written **keyed by a hash of the configuration** as they complete, letting a restart skip finished work by reading the store rather than recomputing. And because the store is keyed by configuration hash, a resumed sweep is bit-identical to an uninterrupted one, which is the property that makes checkpointing safe rather than merely convenient.

## Ray or Dask is mostly a question of what your work looks like

Both frameworks run this workload. Dask's `LocalCluster` and `client.map` express the same fan-out, and for array and dataframe workloads Dask is often the better fit because it provides *collections* — a distributed dataframe that mirrors the pandas API and a distributed array that mirrors NumPy's — so an existing pandas pipeline can sometimes be scaled by changing the import and adding a `.compute()`. Its scheduler is built around task graphs known in advance, which suits ETL and dataframe operations.

Ray's model is lower-level and more dynamic: arbitrary Python tasks, actors that hold mutable state, and a scheduler comfortable with tasks that spawn tasks. That fits parameter sweeps whose next batch depends on the last, reinforcement-learning workloads, and anything where the work is a Python function rather than a dataframe operation. The honest summary for a backtesting sweep specifically: **the choice rarely matters, and the granularity decision matters much more than which framework makes it.** Pick the one your team already runs; if neither, `multiprocessing.Pool` handles single-machine sweeps with no new dependency at all, and is what [Part VI used for process isolation](../part-06-live-infrastructure/01-system-architecture.md) — for a different reason, since there the goal was crash domains rather than throughput.

## Ten thousand trials, seventy-nine of them independent

Now the accounting the module exists for. The sweep's best configuration prints a Sharpe of 0.5627. Section one says that ten thousand *independent* null strategies would produce 0.77 by luck, which would make 0.5627 unremarkable — worse than noise, in fact. But the sweep's configurations are not independent: a 122-day lookback and a 125-day lookback trade almost identically. The right denominator is the number of *effective* trials, estimated from the eigenvalue spectrum of the strategy-return correlation matrix via the participation ratio,

$$
N_{\mathrm{eff}} \;=\; \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2},
$$

which equals $N$ when the strategies are uncorrelated and 1 when they are identical:

```python
import numpy as np
import pandas as pd
import ray
from scipy import stats

ray.init(num_cpus=32, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
px = pd.read_parquet("data/prices.parquet")
R = np.log(px[["SPY", "TLT", "GLD"]]).diff().dropna().values
R_ref = ray.put(R)
CONFIGS = [(lb, round(b, 5)) for lb in range(20, 320, 3)
           for b in np.linspace(0.0, 0.02, 100)]

@ray.remote
def returns_for(r, cfgs):
    c = np.cumsum(r, axis=0)
    out = []
    for lb, band in cfgs:
        mom = np.empty_like(r)
        mom[:lb] = np.nan
        mom[lb:] = c[lb:] - c[:-lb]
        pos = np.where(mom > band, 1.0, np.where(mom < -band, -1.0, 0.0))
        out.append(np.nansum(pos[:-1] * r[1:], axis=1) / r.shape[1])
    return np.array(out)

def batches(items, k):
    size = (len(items) + k - 1) // k
    return [items[i:i + size] for i in range(0, len(items), size)]

sample = CONFIGS[::37]
M = np.vstack(ray.get([returns_for.remote(R_ref, b) for b in batches(sample, 32)]))
C = np.corrcoef(M)
ev = np.linalg.eigvalsh(C)
ev = ev[ev > 0]
n_eff_sample = (ev.sum() ** 2) / (ev ** 2).sum()
n_eff_full = n_eff_sample * len(CONFIGS) / len(sample)

def emax(n):
    return ((1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / n)
            + np.euler_gamma * stats.norm.ppf(1 - 1 / (n * np.e)))

print(f"mean pairwise correlation among {len(sample)} sampled configs: "
      f"{C[np.triu_indices_from(C, 1)].mean():.3f}")
print(f"effective trials in the full sweep: about {n_eff_full:.0f} of {len(CONFIGS):,}")
print(f"null at the naive count:     E[max] = {emax(len(CONFIGS)) / np.sqrt(25):.3f}")
print(f"null at the effective count: E[max] = {emax(n_eff_full) / np.sqrt(25):.3f}")
print(f"realized best 0.5627 clears the effective null: "
      f"{0.5627 > emax(n_eff_full) / np.sqrt(25)}")
ray.shutdown()
# => mean pairwise correlation among 271 sampled configs: 0.658
#    effective trials in the full sweep: about 79 of 10,000
#    null at the naive count:     E[max] = 0.772
#    null at the effective count: E[max] = 0.489
#    realized best 0.5627 clears the effective null: True
```

The configurations correlate 0.658 on average, and ten thousand of them contain roughly **seventy-nine independent bets**. That single number reframes everything. Judged against the naive trial count, the sweep's champion (0.5627) falls *below* the 0.772 that pure noise would produce — an apparently damning verdict. Judged against the effective count, the null is 0.489 and the champion clears it by 0.073.

Neither reading is a licence to trade. The honest conclusion is a modest one: there is a little more here than noise, in a strategy family [Part IV already established](../part-04-strategy-development/01-momentum-and-trend-following.md) has a faint real edge, and the sweep has added almost nothing to that knowledge while consuming a machine for an hour. The methodological point is sharper and general. **Reporting the raw trial count overstates the penalty; reporting no trial count at all understates it catastrophically.** Both numbers belong in the write-up, along with the correlation that separates them, and a research process that logs its configurations can compute all three for free — which is precisely why [Part IV made keeping the trial log a ritual](../part-04-strategy-development/05-feature-and-signal-engineering.md).

## Knowing when not to run the sweep

The cost side deserves the same arithmetic as the statistics. Cloud spot instances are typically 60–90% cheaper than on-demand and can be reclaimed with two minutes' notice, which is affordable exactly when the sweep checkpoints by configuration hash — the design from the determinism section is also the design that makes cheap compute usable. Aim tasks at a few seconds each: long enough to amortize scheduling, short enough that a reclaimed instance loses little.

The deeper question is whether to run at all, and three cases argue against. When the parameter surface is smooth, a hundred well-spaced configurations locate the plateau as well as ten thousand, and [Bayesian Optimization](01-bayesian-optimization.md) does it in fewer still — with the same warning attached, since a sample-efficient optimizer climbs noise efficiently too. When the effective trial count is low, as it was here, the extra configurations are re-measuring one bet with different labels. And when the hypothesis was not stated in advance, the sweep is a search for a hypothesis rather than a test of one, and no amount of deflation repairs that — a pre-registered strategy scored a deflated Sharpe of 0.93 in Part IV where the identical returns scored 0.64 as a grid selection, and the difference was made before either backtest ran.

!!! warning "The cluster's output is candidates, and candidates are the cheap part"
    Ten thousand backtests in three and a half seconds is real engineering, and it changes nothing about the evidence for any one of them. The best of ten thousand independent null strategies prints 0.77 on twenty-five years of data; this sweep's ten thousand configurations contained about seventy-nine independent bets, so its honest null was 0.489 and its champion cleared it by 0.073 — a modest result that a hundred configurations would have found. Compute the effective trial count, publish it next to the raw one, and treat any sweep that cannot reproduce its own hash as unreviewable.

!!! abstract "Key takeaways"
    - With a 0.20 standard error on a 25-year Sharpe, the best of $N$ null strategies is expected to print 0.31 at $N = 10$, **0.77 at $N = 10{,}000$**, and 0.88 at $N = 100{,}000$; simulation matched the formula (0.754 against 0.772 at ten thousand).
    - Growth like $\sqrt{2\ln N}$ means a tenfold larger cluster adds only ~0.11 to the null — the danger is not the size of the penalty but how cheap the marginal trial feels.
    - Sweeps are Gustafson's regime, not Amdahl's: nobody runs yesterday's grid faster, they run a bigger grid, which is why the multiple-testing bill scales with the hardware.
    - Ray's per-task overhead measured 0.05 ms, so efficiency $t/(t + t_{\text{overhead}})$ demands batching until tasks run in the seconds; the too-small version of this sweep scaled only 3.9× on 32 workers.
    - Properly batched, the sweep scaled 9.51× on 32 workers with efficiency falling from 99.4% at two to 29.7% at thirty-two — the collapse tracking 16 physical cores, and a USL fit ($\sigma = 0.0448$, $\kappa = 0.00103$) predicting peak throughput at about 30.
    - Determinism needs three rules: spawn seeds with `SeedSequence`, return `(key, value)` from every task, and sort before reducing — verified by identical sorted-table hashes across runs, which also makes checkpointed resumption bit-identical.
    - The 10,000 configurations correlated 0.658 pairwise and contained about **79 effective trials**; the champion's 0.5627 falls below the naive null of 0.772 and clears the effective null of 0.489 by 0.073.
    - Report both trial counts and the correlation between them: the raw count overstates the penalty, no count at all understates it catastrophically, and a logged sweep computes all three for free.

## Where this goes next

The other axis of the same problem — one machine, one kernel, and the arithmetic that decides whether the work belongs on a GPU at all — is [GPU Acceleration with CUDA](08-gpu-acceleration-cuda.md), which ran this very sweep on a graphics card and hit the same statistical wall from the opposite direction. When the bottleneck is a single hot function rather than the number of runs, [High-Performance Computing](10-high-performance-computing.md) covers profiling, Numba, and Cython, and it argues for restructuring before rewriting. The statistical machinery this module scaled up is [Part IV, lesson eight](../part-04-strategy-development/08-validation-and-overfitting.md), whose deflated Sharpe and probability-of-backtest-overfitting are the tools that turn a trial count into a verdict, and the search-history accounting for adaptive rather than exhaustive search is in [Bayesian Optimization](01-bayesian-optimization.md).
