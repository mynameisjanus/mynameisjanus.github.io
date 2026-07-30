# Random Number Generation

There is no randomness inside a computer. Every "random" number used in simulation is produced by a **pseudorandom number generator (PRNG)**: a deterministic algorithm whose output sequence is designed to be statistically indistinguishable from independent uniform draws. The determinism is not a defect — it is the property that makes simulation-based research reproducible, and the course leans on it constantly: the seeded generators in [NumPy and Vectorization](../../part-02-python/01-numpy-and-vectorization.md) and the seed discipline of [Logging, Configuration, and Reproducibility](../../part-02-python/07-logging-and-config.md) both rest on the machinery described here.

## Pseudorandom Generators

A PRNG consists of a finite **state space** $\mathcal{S}$, a **transition function** $f$, and an **output function** $g$:

$$
s_{k+1} = f(s_k), \qquad u_k = g(s_k) \in [0, 1).
$$

The initial state $s_0$ is derived from the **seed**. Because $f$ is deterministic and $\mathcal{S}$ is finite, the sequence is fully determined by $s_0$ and must eventually revisit a state, after which it repeats. The **period** $p$ is the length of that cycle — the smallest $p$ such that $s_{k+p} = s_k$ for all sufficiently large $k$. Everything about PRNG design is a trade-off between period, state size, speed, and how well the outputs mimic independence.

### Linear Congruential Generators

The classical construction, and still the clearest illustration, is the **linear congruential generator (LCG)**:

$$
x_{k+1} = (a\,x_k + c) \bmod m, \qquad u_k = \frac{x_k}{m},
$$

with multiplier $a$, increment $c$, and modulus $m$. A toy example with $a = 5$, $c = 3$, $m = 16$, and seed $x_0 = 7$ produces

$$
7,\ 6,\ 1,\ 8,\ 11,\ 10,\ 5,\ 12,\ 15,\ 14,\ 9,\ 0,\ 3,\ 2,\ 13,\ 4,\ 7,\ \ldots
$$

— all sixteen residues exactly once, then the cycle restarts. This generator achieves the maximum possible period $p = m$ because its parameters satisfy the Hull–Dobell conditions ($c$ coprime to $m$; $a - 1$ divisible by every prime factor of $m$, and by 4 when $m$ is).

??? note "The Hull–Dobell theorem"
    An LCG with $c \neq 0$ has full period $m$ for every seed if and only if:

    1. $\gcd(c, m) = 1$,
    2. $a \equiv 1 \pmod{q}$ for every prime $q$ dividing $m$,
    3. $a \equiv 1 \pmod{4}$ if $4 \mid m$.

    For $m = 16 = 2^4$: condition 1 holds ($c = 3$ is odd), the only prime factor is 2 and $a - 1 = 4$ is even, and $4 \mid 16$ with $a = 5 \equiv 1 \pmod 4$. Hence the full period observed above.

A long period is necessary but nowhere near sufficient. The famous failure is lattice structure: successive $k$-tuples $(u_i, u_{i+1}, \ldots, u_{i+k-1})$ from any LCG do not fill the unit hypercube but lie on a family of parallel hyperplanes — at most $(k!\,m)^{1/k}$ of them, by Marsaglia's theorem. IBM's **RANDU** ($a = 65539$, $c = 0$, $m = 2^{31}$), distributed throughout the 1960s and used in published research for years, places all of its 3-tuples on just **15 planes** in the unit cube. Any simulation that consumed RANDU draws three at a time — a 3-D random walk, a triple integral — was sampling 15 slices of the space and calling it uniform. The lesson generalizes: a generator's defects are invisible in histograms of single draws and lethal in the joint behavior your simulation actually depends on.

### Modern Generators

| Generator | State | Period | Notes |
|---|---|---|---|
| LCG (e.g. MINSTD, $a = 7^5$, $m = 2^{31}-1$) | 4–8 bytes | $\le m$ | Historic; lattice structure makes it unfit for serious work |
| Mersenne Twister (MT19937) | 2.5 KB | $2^{19937}-1$ | Legacy default in NumPy's `np.random.*` module; equidistributed to 623 dimensions, but fails linear-complexity tests and is predictable from 624 outputs |
| PCG64 | 16 bytes | $2^{128}$ | NumPy's `default_rng` since 1.17: an LCG core whose state is scrambled by a permutation output function; passes TestU01 BigCrush and PractRand |

The design idea behind PCG64 is worth one sentence, because it explains why the LCG story above still matters: a 128-bit LCG supplies the long-period state evolution cheaply, and a nonlinear output permutation destroys the lattice structure before anything reaches the user. The weak classical generator is still inside — it is just no longer allowed to talk to you directly.

## Uniformity Is the Raw Material

A PRNG manufactures approximations to independent draws of $U \sim \mathrm{Unif}(0,1)$, with

$$
\mathbb{E}[U] = \frac{1}{2}, \qquad \mathrm{var}(U) = \frac{1}{12}.
$$

??? note "Proof"
    $$\begin{align}
    \mathbb{E}[U] &= \int_0^1 u \, du = \frac{1}{2},\\[4pt]
    \mathbb{E}[U^2] &= \int_0^1 u^2 \, du = \frac{1}{3},\\[4pt]
    \mathrm{var}(U) &= \mathbb{E}[U^2] - (\mathbb{E}[U])^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}.
    \end{align}$$

Every other distribution a simulation needs — normal returns, exponential waiting times, Poisson arrival counts — is obtained by *transforming* uniforms; the standard recipes (inverse transform, Box–Muller, and friends) are the subject of [Sampling Methods](02-sampling-methods.md), and what you can estimate with the resulting samples, at what error rate, is [Monte Carlo Simulation](03-monte-carlo-simulation.md). This layering is why generator quality is foundational: a defect in the uniforms propagates into every distribution built on top of them.

## Seeding

The seed is the only input a PRNG takes, and fixing it fixes the entire output sequence — a seeded generator is a deterministic function of its seed. Modern libraries do not use the seed as the state directly: NumPy passes it through a `SeedSequence`, which hashes the user's low-entropy integer (42, a date, a student ID) into well-mixed bits so that nearby seeds produce unrelated streams. Two consequences follow. First, no seed value is "better" than another — 0, 42, and 20240102 all yield full-quality streams. Second, reproducibility is exact, not approximate:

```python
import numpy as np

rng = np.random.default_rng(42)          # PCG64 underneath
print(rng.random(3))                     # => [0.77395605 0.43887844 0.85859792]

again = np.random.default_rng(42)
print(np.allclose(again.random(3),
                  [0.77395605, 0.43887844, 0.85859792]))  # => True
```

Left unseeded, `default_rng()` draws its initial state from operating-system entropy — correct for one-off use, and exactly wrong for research, where the run cannot then be repeated. The working discipline (argued at length in [Logging, Configuration, and Reproducibility](../../part-02-python/07-logging-and-config.md)) is one seeded generator created at the top of a run and passed explicitly to everything that needs randomness.

Two classic seeding mistakes are worth naming. Re-seeding inside a loop — `default_rng(int(time.time()))` per iteration — can hand identical seeds to iterations that run within the same clock tick, silently duplicating "independent" samples. And re-seeding per function call turns what should be one long high-quality stream into many short ones whose relationships are governed by the seeding scheme rather than the generator.

### Parallel Streams

Running simulations on many workers raises a sharper version of the same problem: each worker needs its own stream, and the streams must not overlap or correlate. Hand-crafting worker seeds (`seed + worker_id`) was historically dangerous for exactly the reasons above; the modern solution is built in. `SeedSequence.spawn` derives child generators that are statistically independent by construction:

```python
import numpy as np

children = np.random.default_rng(42).spawn(3)   # one per worker
print([round(float(c.random()), 3) for c in children])
# => [0.917, 0.467, 0.071]
```

The children are themselves reproducible — the same parent seed always spawns the same family — so a parallel Monte Carlo run remains a deterministic function of a single recorded seed.

## Statistical Quality

"Looks random" is an operational claim: the sequence should pass the statistical tests a real i.i.d. uniform sequence would pass. The standard batteries — Diehard historically, **TestU01** (SmallCrush through BigCrush) and **PractRand** today — probe equidistribution in many dimensions, serial correlation, runs and gaps, rank of random binary matrices, and dozens of subtler properties. Passing them is the entry ticket for a modern generator; RANDU fails them instantly, and even Mersenne Twister fails the linear-complexity family, which is part of why defaults moved on.

Period imposes a separate, cruder constraint: a common rule of thumb is to draw no more than roughly $\sqrt{p}$ numbers from a generator of period $p$, beyond which the deterministic cycle starts to become statistically detectable. For PCG64's $p = 2^{128}$ this permits about $2^{64} \approx 10^{19}$ draws — not a binding constraint for any backtest or bootstrap you will run. It was binding for the 32-bit LCGs of the past, which is one more reason nothing serious uses them now.

## Pseudorandom versus Cryptographic

Statistical quality and unpredictability are different properties. Mersenne Twister passes equidistribution tests yet is completely predictable: observing 624 consecutive outputs determines its entire state, and with it every future draw. That is irrelevant for Monte Carlo and disqualifying for security. Session tokens, API keys, order identifiers that must not be guessable — anything adversarial — requires a **cryptographically secure** generator, which in Python means the `secrets` module or `os.urandom`, never `numpy.random` or the `random` module. The converse also holds: cryptographic generators are slower and offer no reproducibility, so they have no place inside a simulation. Use each kind for what it is for.
