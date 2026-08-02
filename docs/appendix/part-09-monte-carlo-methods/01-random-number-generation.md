# Random Number Generation

There is no randomness inside a computer, and the absence is the point rather than the compromise. Every draw a simulation consumes is the output of a deterministic function of a number somebody wrote down, which is what makes a simulated result an object other people can check. The engineering question is not how to obtain randomness — that is impossible — but how to build a deterministic sequence whose failures are far enough from the questions being asked that they never surface. That last clause is where the subject lives, because a generator's defects are never visible in the statistic anyone thinks to compute, and they are total in the one the simulation actually depends on.

This page covers the generator as a finite state machine and the period that finiteness forces, the lattice structure that makes a defect invisible in one dimension and fatal in three, the difference between a seed and a state and why adjacent seeds are not adjacent streams, the spawning discipline that makes parallel streams independent, and the test batteries that decide whether a generator is fit for use. It does not derive the transformations that turn uniforms into other laws, which is [Sampling Methods](02-sampling-methods.md); it does not estimate anything with the resulting draws or put an error bar on it, which is [Monte Carlo Simulation](03-monte-carlo-simulation.md); it constructs no proposal distribution, which are [Importance Sampling](04-importance-sampling.md) and [Rejection Sampling](05-rejection-sampling.md); it resamples no data, which is [Bootstrap Methods](07-bootstrap-methods.md); it proves no limit theorem about the resulting averages, which is [Part VII](../part-07-asymptotic-theory/index.md); and it builds nothing an adversary is meant to fail to guess.

The trading stake is a promise the course makes about its own numbers. [Logging, Configuration, and Reproducibility](../../part-02-python/07-logging-and-config.md) names this page and rests a working discipline on it — "a seeded generator is a deterministic function, so the same seed *is* the same data" — and [Architecture and Event-Driven Design](../../part-05-backtesting-engine/01-architecture-and-event-driven-design.md) turns the same property into an engine invariant, that "identical inputs produce an identical processing sequence, bit for bit, every run". Both sentences are true, and neither is free. The third and fourth sections price them: the discipline fails quietly under the most natural way of parallelizing a simulation, and it fails not by crashing but by printing an error bar four times narrower than the truth.

## A Generator Is a State Machine and the Seed Is Its Only Freedom

A **pseudorandom number generator** is a finite state space $\mathcal{S}$, a transition function $f:\mathcal{S}\to\mathcal{S}$, and an output function $g:\mathcal{S}\to[0,1)$, run as

$$s_{k+1}=f(s_k),\qquad u_k=g(s_k).$$

The initial state $s_0$ is derived from the **seed**, and that is the whole of the user's input. Because $\mathcal{S}$ is finite and $f$ is a function, the orbit of $s_0$ must revisit a state, after which it repeats forever; the length of that cycle is the **period**. Everything in generator design trades period, state size, speed and output quality against one another, and the first three are easy to measure while the fourth is the only one that matters.

The classical construction is the **linear congruential generator**,

$$x_{k+1}=(a\,x_k+c)\bmod m,\qquad u_k=x_k/m,$$

with multiplier $a$, increment $c$ and modulus $m$. It is worth understanding not because anyone should use one — nobody should — but because it is still inside the generators everybody does use, and because its one famous failure is the cleanest available illustration of how a generator breaks.

??? note "Proof that the Hull–Dobell conditions are exactly what buys the full period, and that the full period buys nothing else"
    An LCG with $c\neq0$ attains period $m$ from every seed if and only if $\gcd(c,m)=1$, $a\equiv1$ modulo every prime dividing $m$, and $a\equiv1\pmod 4$ whenever $4\mid m$. The necessity of the first is immediate: if a prime $q$ divides both $c$ and $m$ then $x_{k+1}\equiv a x_k \pmod q$, so the residue class of $x_k$ modulo $q$ never visits every value and the orbit is confined. The other two conditions are the statement that the map $x\mapsto ax+c$ generates the full additive group modulo each prime power in $m$, which is checked prime power by prime power and assembled by the Chinese remainder theorem.

    Take $a=5$, $c=3$, $m=16$, $x_0=7$. Then $\gcd(3,16)=1$; the only prime dividing $16$ is $2$ and $a-1=4$ is even; and $4\mid16$ with $5\equiv1\pmod4$. The orbit is $7,6,1,8,11,10,5,12,15,14,9,0,3,2,13,4$ and then $7$ again — all sixteen residues, each exactly once.

    The load-bearing observation is what the theorem does not say. A full period guarantees that the generator visits every representable value exactly once per cycle, which is a statement about the *marginal* distribution of a single draw and nothing else. It says nothing about the order in which the values arrive, and order is the whole content of independence. A counter that emits $0,1/m,2/m,\dots$ also has full period and perfect uniformity, and no one would call it random. **Every generator failure worth knowing about is a failure of joint behaviour on a sequence whose marginal behaviour is exactly right.**

## Lattice Structure Is Invisible in One Dimension and Fatal in Three

Because the recurrence is linear, consecutive outputs of an LCG are not merely correlated — they are confined to a lattice. Marsaglia's theorem states that the $k$-tuples $(u_i,u_{i+1},\dots,u_{i+k-1})$ from any LCG of modulus $m$ lie on at most $(k!\,m)^{1/k}$ parallel hyperplanes in the unit cube. For $m=2^{31}$ and $k=3$ that bound is about $2{,}344$ planes, which is already a defect and is not usually a disaster. A badly chosen multiplier does very much worse than the bound.

??? note "Proof that RANDU's three-tuples lie on fifteen planes, from one line of arithmetic"
    RANDU is the multiplicative generator $a=65539$, $c=0$, $m=2^{31}$, shipped by IBM through the 1960s and consumed by a decade of published simulation. Write $a=2^{16}+3$. Then

    $$a^{2}=2^{32}+6\cdot2^{16}+9=2^{32}+6(2^{16}+3)-9\equiv 6a-9 \pmod{2^{31}},$$

    because $2^{32}\equiv0\pmod{2^{31}}$. Since $x_{k+2}=a^{2}x_k \bmod 2^{31}$, the sequence satisfies the exact three-term recurrence $x_{k+2}=6x_{k+1}-9x_k \pmod{2^{31}}$, so $9x_k-6x_{k+1}+x_{k+2}$ is an exact multiple of $2^{31}$ and

    $$9u_k-6u_{k+1}+u_{k+2}\in\mathbb{Z}.$$

    Each $u$ lies in $[0,1)$, so the left side lies in $(-6,10)$ and can therefore take exactly fifteen integer values. Every three-tuple RANDU will ever produce lies on one of fifteen parallel planes, and the whole derivation is the observation that $65539$ is three more than a power of two.

    The load-bearing feature is that the defect is a property of the *multiplier* and not of the construction. Marsaglia's bound permits $2{,}344$ planes here and the generator uses fifteen, so the failure is a hundred and fifty times worse than the guarantee, and no amount of testing single draws could have found it. The rule this generalizes to is uncomfortable and exact: a generator is only ever certified against the tests that were run, and the dimension in which it was tested is a parameter of that certification.

```python
import numpy as np
from scipy.stats import kstest

rng = np.random.default_rng(9011)
n = 1_000_000


def randu(x0, k):                                              # a = 65539, c = 0, m = 2**31
    x = np.empty(k, dtype=np.int64)
    x[0] = x0
    for i in range(1, k):
        x[i] = (65539 * x[i - 1]) % 2**31
    return x / 2**31


print(f"  RANDU against PCG64 on {n} draws, read in one dimension and in three")
print("   generator    KS p-value    lag-1 corr    distinct values of 9u-6u'+u''")
for name, u in (("RANDU", randu(1, n)), ("PCG64", rng.random(n))):
    plane = 9 * u[:-2] - 6 * u[1:-1] + u[2:]                   # an exact integer for RANDU
    rho = np.corrcoef(u[:-1], u[1:])[0, 1]
    print(f"  {name:>9} {kstest(u, 'uniform').pvalue:13.4f} {rho:13.5f}"
          f" {np.unique(np.round(plane, 9)).size:31d}")
# =>   RANDU against PCG64 on 1000000 draws, read in one dimension and in three
#       generator    KS p-value    lag-1 corr    distinct values of 9u-6u'+u''
#          RANDU        0.5438      -0.00049                              15
#          PCG64        0.3020      -0.00101                          999958
```

The first two columns are the diagnostics anyone actually runs, and RANDU passes both cleanly. A Kolmogorov–Smirnov test against the uniform on a million draws returns $p=0.5438$ — a more comfortable result than the $0.3020$ the modern generator gets, and if the two rows had been handed over unlabelled, the naive reading would prefer the broken one. The lag-one autocorrelation is $-0.00049$ against $-0.00101$, both indistinguishable from zero on a million observations. Nothing in a histogram, a Q–Q plot, or a correlogram of this sequence is wrong.

The third column is the same million draws read three at a time. RANDU's million three-tuples occupy **fifteen** distinct values of the plane statistic; PCG64's occupy $999{,}958$, which is a million minus the handful of coincidental ties floating-point rounding produces. The unit cube is not being sampled by RANDU at all. It is being sampled on fifteen slices, and the volume between them — which is nearly all of it — is never visited by any triple in the sequence's entire period.

**A generator's failures do not announce themselves in the marginal distribution, because the marginal distribution is the one thing every generator gets right by construction.** Any simulation consuming RANDU three values at a time — a three-dimensional random walk, a triple integral, a three-asset return vector — was integrating over fifteen planes and reporting the answer as if it had integrated over a cube. The results were published. The failure was found by plotting the tuples and turning the plot, which is not a test anyone runs on a schedule.

## The Seed Is Not the State, and Adjacent Seeds Are Not Adjacent Streams

The naive way to seed is to load the seed into the state. For an LCG that is exactly what the formula invites, and it is why the historical practice of deriving worker seeds arithmetically — `seed + worker_id`, `seed * 1000 + fold` — was not merely inelegant but wrong. If the seed is the state, then two seeds related by a simple arithmetic relation produce two streams related by the same relation pushed through $f$, and $f$ was chosen for speed rather than for destroying structure.

Modern libraries interpose a hash. NumPy passes the user's integer through a `SeedSequence`, which mixes a low-entropy input — $42$, a date, a ticket number — into a well-distributed 128-bit state, so that the relationship between two seeds tells you nothing about the relationship between two streams. The consequence practitioners quote is that no seed is better than another, and $0$, $42$ and $20240102$ are equally good. The consequence that matters more is the one below.

```python
import numpy as np

rng = np.random.default_rng(9013)
workers, m, a = 8_192, 2**31 - 1, 16_807                       # MINSTD, the textbook LCG
ids = np.arange(1, workers + 1)

lcg = (a * ids) % m / m                                        # worker id seeds the state directly
hashed = np.array([np.random.default_rng(i).random() for i in ids])
spawned = np.array([c.random() for c in rng.spawn(workers)])
one = (a ** np.arange(1, 9) % m) / m                           # eight draws from seed 1
two = (2 * (a ** np.arange(1, 9) % m) % m) / m                 # eight draws from seed 2

print(f"  first draw of {workers} workers, three ways of turning a worker id into a seed")
print("   scheme                      corr with worker id    pooled mean    stream 2 = 2*stream 1")
for name, first, tied in (("MINSTD, seed = id", lcg, np.allclose(two, (2 * one) % 1)),
                          ("PCG64, seed = id", hashed, False),
                          ("PCG64, parent.spawn(id)", spawned, False)):
    print(f"  {name:<25} {np.corrcoef(ids, first)[0, 1]:21.5f} {first.mean():14.5f}"
          f" {str(tied):>24}")
print(f"  a fair generator puts that pooled mean within {np.sqrt(1 / 12 / workers) * 1.96:.5f} of 0.5")
# =>   first draw of 8192 workers, three ways of turning a worker id into a seed
#       scheme                      corr with worker id    pooled mean    stream 2 = 2*stream 1
#      MINSTD, seed = id                       1.00000        0.03206                     True
#      PCG64, seed = id                       -0.01494        0.49604                    False
#      PCG64, parent.spawn(id)                 0.00077        0.49866                    False
#      a fair generator puts that pooled mean within 0.00625 of 0.5
```

The first row is not an approximation and not a small effect. MINSTD is the multiplicative generator $a=7^5=16{,}807$, $m=2^{31}-1$, which was the recommended portable generator for two decades and is still the default in more than one language. Seeded with the worker id, its first output is $16807\,i \bmod m$, and for $i$ up to $8{,}192$ the product never reaches the modulus, so the first draw is a *linear function of the worker index*. The correlation is $1.00000$ exactly, and the pooled mean of eight thousand workers' first draws is $0.03206$ where a fair generator would land within $0.00625$ of $0.5$ — off by more than seventy standard errors.

The last column is worse in a way that is easy to miss. Because the recurrence is multiplicative, the stream from seed $2$ is the stream from seed $1$ doubled modulo one, term by term, forever: `np.allclose(two, (2 * one) % 1)` returns `True`. These are not two independent streams that happen to correlate. They are one stream and a deterministic function of it, and no amount of averaging over workers recovers the information that was never generated.

Both PCG64 rows behave. Correlations of $-0.01494$ and $0.00077$ are within the $\pm0.011$ a sample of this size produces by chance, and both pooled means sit inside the interval. The hash is doing the work, and the working rule follows from the first row rather than from taste: **the seed you write down and the state the generator runs on must be separated by something designed to destroy structure, and "designed" excludes anything you can do in your head.**

!!! note "The advice to never re-seed inside a loop is the same theorem in working clothes"
    Two habits cause the failure above without ever writing `seed + i`. Re-seeding per iteration from the clock — `default_rng(int(time.time()))` inside a loop — hands identical seeds to every iteration that runs inside one clock tick, so a run that reports ten thousand independent replications may contain a few hundred distinct ones. Re-seeding per function call turns one long stream, whose quality the generator's designers certified, into many short ones whose *relationships* are governed by the seeding scheme, which nobody certified. In both cases the arithmetic downstream is correct and the sample size is a fiction. The discipline that avoids all of it is the one [Logging, Configuration, and Reproducibility](../../part-02-python/07-logging-and-config.md) states: create exactly one generator at the top of a run and pass it explicitly into everything that draws.

## Independent Streams Are a Spawning Problem, Not an Arithmetic One

A simulation that runs on many workers needs many streams, and needs them to behave as though they came from one. `SeedSequence.spawn` is the construction that supplies it: a parent sequence derives children by hashing its entropy together with a child index, so the children are statistically independent by design and reproducible as a family from the single recorded parent seed. This is what [Distributed Backtesting](../../advanced/09-distributed-backtesting.md) means by "seed by spawning, not by arithmetic", and the reason it phrases the alternative as something that "silently degrades bootstrap estimates" is measurable.

```python
import numpy as np

rng = np.random.default_rng(9017)
workers, per, reps, m, a = 512, 32, 4_000, 2**31 - 1, 16_807
nominal = np.sqrt((1 / 12) / (workers * per))                  # sd of a mean of iid uniforms


def farm_lcg(base):                                            # seed = base + worker id
    s = (base + np.arange(1, workers + 1)) % m
    out = np.empty((workers, per))
    for k in range(per):
        s = (a * s) % m
        out[:, k] = s / m
    return out


def farm_spawn(base):                                          # one parent, spawned children
    return np.array([c.random(per) for c in np.random.default_rng(base).spawn(workers)])


print(f"  {reps} runs of a {workers}-worker farm, {per} draws each, estimating E[U] = 0.5")
print("   seeding scheme      mean estimate    reported se    actual sd    actual/reported")
for name, farm in (("seed = base + id", farm_lcg), ("parent.spawn(id)", farm_spawn)):
    est = np.empty(reps)
    rep_se = np.empty(reps)
    for r in range(reps):
        u = farm(int(rng.integers(1, 2**31 - 1)))
        est[r] = u.mean()
        rep_se[r] = u.std(ddof=1) / np.sqrt(u.size)            # the error bar the run would print
    print(f"  {name:<20} {est.mean():13.5f} {rep_se.mean():14.6f} {est.std(ddof=1):12.6f}"
          f" {est.std(ddof=1) / rep_se.mean():18.2f}")
print(f"  nominal se for {workers * per} independent draws: {nominal:.6f}")
# =>   4000 runs of a 512-worker farm, 32 draws each, estimating E[U] = 0.5
#       seeding scheme      mean estimate    reported se    actual sd    actual/reported
#      seed = base + id           0.50006       0.002254     0.009066               4.02
#      parent.spawn(id)           0.50000       0.002255     0.002225               0.99
#      nominal se for 16384 independent draws: 0.002255
```

Read the second column first, because it is what makes the failure survive review. Both schemes estimate the right answer: $0.50006$ and $0.50000$ against a truth of $0.5$. Arithmetic seeding is not biased, and any check that compares the estimate against a known value passes.

The third and fourth columns are the whole story. Both farms print a standard error of $0.00225$, computed the ordinary way from the pooled sample's own spread, and both printed error bars agree with the nominal $0.002255$ that $16{,}384$ independent draws would deserve. But the actual run-to-run standard deviation of the arithmetic scheme's estimate is $0.009066$ — **four times what it reports** — while the spawned farm's is $0.002225$, a ratio of $0.99$. The arithmetic farm's nominal $95\%$ interval covers the truth about a third of the time.

Nothing inside a single run reveals this. The pooled sample from the bad farm has the right mean, the right variance, and passes a one-dimensional uniformity test, because the dependence lives *across workers at matched positions in the stream* and every diagnostic anyone runs pools the workers together first. **The estimate is right, the error bar is wrong by a factor of four, and the only way to see it is to run the whole farm four thousand times, which is precisely what nobody does.** This is the shape the rest of Part IX keeps finding: a procedure that reports its own precision from the same draws that produced its estimate cannot detect a defect that afflicts both.

!!! warning "A generator that passes every test you ran is not a generator that passes the test your simulation is"
    The two failures on this page are the same failure. RANDU is uniform in one dimension and degenerate in three; arithmetic seeding is uniform in the pooled sample and degenerate across workers. In both cases the certification and the use are in different spaces, and the gap between them is invisible from inside the run. The defensive practice is not to test the generator, which somebody with better tools has already done, but to know which joint structure your simulation consumes — how many draws per unit of state, whether the units are compared or pooled, whether workers are aggregated at matched offsets — and to confirm the generator was certified in *that* space. For anything using `default_rng` with spawned children and no arithmetic anywhere, the answer is yes and no further thought is required. For anything that reaches for the seed by hand, the answer is unknown, and unknown here has historically meant wrong.

## Statistical Quality Is a Battery, and Unpredictability Is a Different Property

"Looks random" is an operational claim about what a sequence survives. The standard batteries — Diehard historically, **TestU01** from SmallCrush to BigCrush and **PractRand** today — probe equidistribution in many dimensions, serial correlation, runs and gaps, the rank of random binary matrices, birthday spacings, and dozens of properties chosen precisely because they are joint rather than marginal. Passing them is the entry ticket. RANDU fails within seconds. The Mersenne Twister, NumPy's legacy default, is equidistributed to $623$ dimensions and still fails the linear-complexity family, because its state evolves by a linear recurrence over $\mathbb{F}_2$ and enough consecutive outputs let you solve for it. PCG64, the current default, puts a $128$-bit linear congruential core underneath a nonlinear output permutation, which is the design idea worth one sentence: the weak classical generator is still in there, and it is simply no longer allowed to speak to the user directly.

Period imposes a separate and cruder constraint. A common rule of thumb is to consume no more than about $\sqrt{p}$ draws from a generator of period $p$, beyond which the cycle becomes statistically detectable in the output. For PCG64's $p=2^{128}$ that permits roughly $2^{64}\approx1.8\times10^{19}$ draws, which no backtest, bootstrap or option-pricing run will approach. For the 32-bit generators of the past it permitted about $46{,}000$, and a single afternoon's simulation exceeded it.

Statistical quality and unpredictability are unrelated properties, and conflating them causes errors in both directions. The Mersenne Twister passes equidistribution tests and is completely predictable: $624$ consecutive outputs determine the state and hence every future draw. That is irrelevant for a Monte Carlo integral and disqualifying for anything adversarial — session tokens, API keys, order identifiers that must not be guessable — which in Python means `secrets` or `os.urandom` and never `numpy.random`. The converse also holds and is the more common mistake in research code: cryptographic generators are slower, cannot be seeded, and cannot be replayed, so using one inside a simulation buys nothing and destroys the property the whole enterprise rests on.

## Determinism Is the Only Thing That Makes a Number Checkable

The uniform stream this page produces is the raw material for everything else in Part IX, and its quality is foundational in the literal sense that a defect propagates. [Sampling Methods](02-sampling-methods.md) turns uniforms into every other law by transformation, and a transformation cannot repair structure in its input — RANDU pushed through an inverse normal CDF gives normals confined to fifteen surfaces. [Monte Carlo Simulation](03-monte-carlo-simulation.md) averages the results and quotes a standard error derived from an independence assumption about the draws, which is an assumption about the generator and about the seeding. [Bootstrap Methods](07-bootstrap-methods.md) draws indices rather than values, billions of them, and the index stream is exactly where a short period or a lattice would surface.

What the determinism buys, in exchange, is the only property that distinguishes a simulated result from an assertion. A seeded run can be re-executed by a stranger and compared byte for byte, which is why every Python block in this appendix pins its output in the source and why a discrepancy between the pinned numbers and a fresh run is a bug report rather than a statistical question. The same property is what makes the engine of [Part V](../../part-05-backtesting-engine/index.md) testable at all: a comparison is an experiment only when every input but one is held fixed, and a run whose randomness cannot be replayed has no fixed inputs to hold.

The two failures shown here are worth carrying forward as a single sentence, because the rest of the part is variations on it. In both cases the estimate was correct, the marginal distribution was correct, every diagnostic that was run passed, and the quantity that was wrong — the joint structure in one case, the standard error in the other — was one that the procedure computes from the same draws it used for everything else. That is the structural weakness of estimation by simulation, and it is not fixed by better generators. It is fixed by knowing what the estimator's error bar is actually a statement about, which is [Monte Carlo Simulation](03-monte-carlo-simulation.md), and by the transformations that stand between a uniform and the law you meant to sample, which come first.
