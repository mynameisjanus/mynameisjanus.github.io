# Sequences and Infinite Series

Every asymptotic result in this appendix is a statement about a limit. "The sample mean converges to the true mean" ([Weak Law of Large Numbers](../part-07-asymptotic-theory/01-weak-law-of-large-numbers.md)), "the standardized sum becomes normal" ([Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md)), "the chain forgets its starting state" ([Markov Chains](../part-08-stochastic-processes/05-markov-chains.md)) — each is a sequence with a limit, and the probabilistic content is *which* notion of convergence applies. The deterministic notion comes first, and it is this page.

Infinite series matter for a more immediate reason: they are how a decaying weight scheme is summed. An exponentially weighted moving average, a discounted stream of future rewards, and the mean of a geometric distribution are all the same geometric series with different labels, and the closed form is what turns them from an infinite computation into one line of arithmetic.

## Sequences

A **sequence** is an ordered list of elements indexed by the natural numbers, written $\{a_i\}$ or $a_1, a_2, a_3,\ldots$ Formally it is a function $f:\mathbb{N}\to S$; the sequence notation is just $f(i)$ written as $a_i$, and thinking of it as a function is what makes "sequence of vectors", "sequence of matrices", or "sequence of estimators" obviously legitimate. Three that recur in this book:

- $a_n = 1/n$ — the archetypal decay to zero.
- $\hat\mu_n = \frac{1}{n}\sum_{i=1}^{n} X_i$ — the running sample mean, a sequence of *random* variables, and the object the laws of large numbers are about.
- $s_n = \sum_{i=1}^{n} a_i$ — the partial sums of another sequence, which is how series are defined.

### Convergence

A sequence $\{a_i\}$ **converges** to a limit $a$, written $\lim_{i\to\infty} a_i = a$ or $a_i\to a$, if for every $\epsilon>0$ there exists an $N$ such that

$$i > N \implies \lvert a_i - a\rvert < \epsilon.$$

The definition is a game with a specific structure, and reading it in the right order is most of understanding it. An adversary picks a tolerance $\epsilon$ — as small as they like, after seeing the sequence. You must then produce a cutoff $N$ beyond which *every* term is within $\epsilon$ of the limit. Convergence means you can always win, for every $\epsilon$, with $N$ allowed to depend on $\epsilon$.

Geometrically: draw a horizontal band of half-width $\epsilon$ around the limit. The sequence converges if, no matter how thin the band, only finitely many terms lie outside it. Terms may leave the band early and often; what is forbidden is leaving it infinitely often.

```python
# a_n = 1 + (-1)^n / n  ->  1.  For each tolerance, the first N that works.
def first_N(eps, limit=1.0, max_n=10_000_000):
    n = 1
    while n < max_n:
        if all(abs(1 + (-1) ** k / k - limit) < eps for k in range(n + 1, n + 51)):
            return n
        n += 1
    return None

for eps in (0.5, 0.1, 0.01, 0.001):
    print(f"eps={eps:<6} N={first_N(eps)}")
# => eps=0.5    N=2
#    eps=0.1    N=10
#    eps=0.01   N=100
#    eps=0.001  N=999
```

The pattern $N \approx 1/\epsilon$ is the *rate* of convergence, and rates are what separate a useful limit from a useless one. The sample mean converges at $N\approx 1/\epsilon^2$ — to halve the error you need four times the data — which is the practical content of the $O(n^{-1/2})$ standard error and the reason a decade of returns still leaves a Sharpe ratio uncertain.

A sequence that does not converge is **divergent**. Divergence has flavors: $a_n = n$ grows without bound, $a_n=(-1)^n$ oscillates forever between two values, and $a_n = \sin(n)$ does something less tidy than either. All three fail the definition, and only the first is usefully written $a_n\to\infty$.

### Boundedness and Monotone Convergence

A sequence is **bounded** if there is an $M$ with $\lvert a_i\rvert\le M$ for all $i$. Every convergent sequence is bounded — beyond some $N$ the terms sit inside a band, and the finitely many earlier terms have a largest absolute value. The converse fails: $(-1)^n$ is bounded and divergent.

Adding monotonicity closes the gap.

!!! note "Monotone convergence theorem"
    A sequence that is increasing and bounded above converges, and its limit is the supremum of its values. Likewise for decreasing and bounded below.

    This is the workhorse existence result: it establishes that a limit *exists* without producing it. Partial sums of non-negative terms are increasing, so such a series either converges or diverges to $+\infty$ — never oscillates. That dichotomy is what makes the comparison test below legitimate.

**Subsequences** are obtained by discarding terms while keeping order: $a_{n_1}, a_{n_2},\ldots$ with $n_1<n_2<\cdots$. If $a_n\to a$ then every subsequence converges to the same $a$, which gives the standard way to *disprove* convergence: exhibit two subsequences with different limits, as the even and odd terms of $(-1)^n$ do. The **Bolzano–Weierstrass theorem** supplies the partial converse — every bounded sequence has *some* convergent subsequence — which is the compactness fact underlying most proofs that an optimum or a limit point exists.

### Limit Laws

If $a_i\to a$ and $b_i\to b$, then

$$a_i + b_i\to a+b,\qquad a_ib_i\to ab,\qquad \frac{a_i}{b_i}\to\frac{a}{b}\ \ (b\neq 0),$$

and for any continuous $g$,

$$g(a_i)\to g(a).$$

??? note "Proof of the sum rule"
    Let $\epsilon>0$. Since $a_i\to a$ there is $N_1$ with $\lvert a_i-a\rvert<\epsilon/2$ for $i>N_1$, and since $b_i\to b$ there is $N_2$ with $\lvert b_i-b\rvert<\epsilon/2$ for $i>N_2$. For $i>\max(N_1,N_2)$ the triangle inequality gives

    $$\lvert (a_i+b_i)-(a+b)\rvert \le \lvert a_i-a\rvert + \lvert b_i - b\rvert < \tfrac{\epsilon}{2}+\tfrac{\epsilon}{2} = \epsilon.$$

    Splitting the tolerance in half and taking the later of the two cutoffs is the standard move; the product rule follows the same pattern with an extra step to bound $\lvert a_i\rvert$ using convergence-implies-boundedness.

The last law — continuity preserves limits — has probabilistic descendants that look far more sophisticated than they are. The [Continuous Mapping Theorem](../part-07-asymptotic-theory/06-continuous-mapping-theorem.md) says exactly this for random sequences: if $X_n\to X$ in an appropriate sense and $g$ is continuous, then $g(X_n)\to g(X)$. That is why a converging estimator of variance yields a converging estimator of standard deviation for free, and why [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md) can combine limits of different types. The deterministic version is the ancestor; the probabilistic one adds bookkeeping about *how* convergence is measured, not a new idea.

### limsup and liminf

Some bounded sequences never settle, and it is useful to describe their eventual range anyway. The **limit superior** and **limit inferior** are

$$\limsup_{n\to\infty} a_n = \lim_{n\to\infty}\ \sup_{k\ge n} a_k,\qquad \liminf_{n\to\infty} a_n = \lim_{n\to\infty}\ \inf_{k\ge n} a_k.$$

Both always exist for a bounded sequence (each is a monotone limit), $\liminf\le\limsup$, and the sequence converges precisely when the two coincide. For $a_n = (-1)^n$, $\limsup = 1$ and $\liminf = -1$.

The vocabulary is needed as soon as convergence is claimed "almost surely". The [Strong Law of Large Numbers](../part-07-asymptotic-theory/02-strong-law-of-large-numbers.md) asserts that the sample-mean sequence converges for almost every realization — a statement about $\limsup$ and $\liminf$ of a random sequence coinciding with probability one, which is genuinely stronger than the weak law's claim about each fixed $n$.

## Infinite Series

Given $\{a_i\}$, form the **partial sums** $s_n = \sum_{i=1}^{n}a_i$. The **infinite series** is the limit of that sequence:

$$\sum_{i=1}^{\infty}a_i = \lim_{n\to\infty} s_n,$$

when the limit exists; otherwise the series diverges. A series is not an infinite addition — it is a limit of finite additions, and every property it has is inherited from the sequence $\{s_n\}$.

**The $n$-th term test.** If $\sum a_i$ converges then $a_i\to 0$. The contrapositive is the only cheap divergence check available: if the terms do not vanish, the series cannot converge. The converse is false, and its most important counterexample is next.

**The harmonic series diverges.**

$$\sum_{i=1}^{\infty}\frac{1}{i} = \infty.$$

??? note "Proof by grouping"
    Group the terms in blocks of doubling length:

    $$\underbrace{\frac{1}{1}}_{\ge\,1} + \underbrace{\frac{1}{2}}_{\ge\,1/2} + \underbrace{\frac{1}{3}+\frac{1}{4}}_{\ge\,2\cdot\frac14 = 1/2} + \underbrace{\frac{1}{5}+\cdots+\frac{1}{8}}_{\ge\,4\cdot\frac18=1/2}+\cdots$$

    Each block contributes at least $1/2$, and there are infinitely many blocks, so the partial sums exceed any bound. They do so extremely slowly — $s_n\approx\ln n + 0.5772$, so reaching 20 takes about $2.7\times10^{8}$ terms — which is why numerical evidence alone would never have settled the question.

More generally the **$p$-series** $\sum_{i\ge1} i^{-p}$ converges if and only if $p>1$. The boundary at $p=1$ is sharp: $\sum 1/i$ diverges while $\sum 1/i^{1.01}$ converges.

### Convergence Tests

| Test | Statement | Use when |
|---|---|---|
| Comparison | If $0\le a_i\le b_i$ and $\sum b_i$ converges, so does $\sum a_i$ | The terms resemble a known series |
| Ratio | If $\lvert a_{i+1}/a_i\rvert\to L$, converges for $L<1$, diverges for $L>1$ | Factorials or powers appear |
| Root | If $\lvert a_i\rvert^{1/i}\to L$, same conclusion | Terms are $i$-th powers |
| Integral | $\sum_{i\ge1} f(i)$ and $\int_1^\infty f$ converge together, for positive decreasing $f$ | The terms come from a smooth function |

The ratio test is the one that gets used, because the series this book cares about are geometric or factorial. Both tests are inconclusive at $L=1$, which is exactly where the $p$-series family lives — the boundary always requires a sharper argument.

### Absolute and Conditional Convergence

A series **converges absolutely** if $\sum\lvert a_i\rvert$ converges, and **conditionally** if it converges while $\sum\lvert a_i\rvert$ does not. The alternating harmonic series is the standard example of the latter:

$$\sum_{i=1}^{\infty}\frac{(-1)^{i+1}}{i} = 1 - \frac12 + \frac13 - \frac14 + \cdots = \ln 2.$$

The distinction is not bookkeeping. Absolutely convergent series can be rearranged and regrouped freely — every ordering gives the same sum. Conditionally convergent ones cannot: **Riemann's rearrangement theorem** says the terms of a conditionally convergent series can be reordered to converge to *any* prescribed real number, or to diverge. Taking two positive terms for every negative one is enough to shift the sum:

```python
import math

def alternating_harmonic(n_terms):
    return sum((-1) ** (k + 1) / k for k in range(1, n_terms + 1))

def rearranged(n_blocks):
    """Two positive terms, then one negative, repeatedly."""
    total, pos, neg = 0.0, 1, 2
    for _ in range(n_blocks):
        total += 1 / pos + 1 / (pos + 2)
        pos += 4
        total -= 1 / neg
        neg += 2
    return total

print(f"{alternating_harmonic(2_000_000):.6f}  vs  ln 2      = {math.log(2):.6f}")
print(f"{rearranged(500_000):.6f}  vs  1.5*ln 2  = {1.5 * math.log(2):.6f}")
# => 0.693147  vs  ln 2      = 0.693147
#    1.039720  vs  1.5*ln 2  = 1.039721
```

Same terms, same multiset of numbers, different sum. The practical warning is about the infinite sums that appear in probability: an expectation $\mathbb{E}[X]=\sum_x x\,p(x)$ over a countably infinite support is only well defined when the sum converges absolutely, which is why [Expected Value](../part-04-expectation-and-moments/01-expected-value.md) requires $\mathbb{E}\lvert X\rvert<\infty$ before speaking of $\mathbb{E}[X]$. Distributions that violate it — the Cauchy is the canonical case — have no mean, not merely an inconvenient one.

## The Geometric Series

The one series worth knowing cold. For $\lvert\alpha\rvert<1$,

$$S = \sum_{i=0}^{\infty}\alpha^i = 1 + \alpha + \alpha^2 + \cdots = \frac{1}{1-\alpha}.$$

??? note "Proof"
    The finite sum comes first. Let $s_n = \sum_{i=0}^{n-1}\alpha^i$. Then

    $$s_n - \alpha s_n = (1 + \alpha + \cdots + \alpha^{n-1}) - (\alpha + \cdots + \alpha^{n}) = 1 - \alpha^n,$$

    the interior terms telescoping, so for $\alpha\neq1$

    $$s_n = \frac{1-\alpha^n}{1-\alpha}.$$

    If $\lvert\alpha\rvert<1$ then $\alpha^n\to0$, and the limit is $1/(1-\alpha)$. If $\lvert\alpha\rvert\ge1$ the terms do not vanish and the $n$-th term test kills it.

The finite form $\sum_{i=0}^{n-1}\alpha^i = (1-\alpha^n)/(1-\alpha)$ is used as often as the infinite one, and shifting the start index is just factoring: $\sum_{i=k}^{\infty}\alpha^i = \alpha^k/(1-\alpha)$.

Differentiating the identity with respect to $\alpha$ — legitimate inside the radius of convergence — produces the two weighted sums that moment calculations need:

$$\sum_{i=1}^{\infty} i\,\alpha^{i} = \frac{\alpha}{(1-\alpha)^2},\qquad \sum_{i=1}^{\infty} i^2\,\alpha^{i} = \frac{\alpha(1+\alpha)}{(1-\alpha)^3}.$$

These are precisely the sums that give the [geometric distribution](../part-05-common-distributions/03-geometric-distribution.md) its mean $1/p$ and variance $(1-p)/p^2$: the expectation $\sum_k k(1-p)^{k-1}p$ is the first identity with $\alpha = 1-p$, and the second moment is the second. The same sums reappear as discounted reward streams in [Reinforcement Learning and Meta-Labeling](../../part-07-machine-learning/04-reinforcement-learning-and-meta-labeling.md), where $\sum_t \gamma^t r_t$ is finite for $\gamma<1$ for exactly this reason and the discount factor's *effective horizon* is $1/(1-\gamma)$.

### Two More Series Worth Recognizing

$$e^x = \sum_{k=0}^{\infty}\frac{x^k}{k!},\qquad \log(1+x) = \sum_{k=1}^{\infty}\frac{(-1)^{k+1}x^k}{k}\ \ \text{for } \lvert x\rvert<1.$$

The first converges for every $x$ (ratio test: the ratio is $x/(k+1)\to0$) and is the reason the [Poisson](../part-05-common-distributions/06-poisson-distribution.md) probabilities $e^{-\lambda}\lambda^k/k!$ sum to one. The second is the series behind $\log(1+r)\approx r - r^2/2$, the approximation that governs the gap between simple and log returns in [Exponentials, Logarithms, and Growth](07-exponentials-logarithms-growth.md). Setting $x=1$ in it, at the edge of the interval, recovers the alternating harmonic sum $\ln 2$ used above. The same integral-and-series machinery gives $\Gamma(a)=\int_0^\infty t^{a-1}e^{-t}\,\mathrm{d}t$ its recursion and its half-integer values, which is what lets [Gamma Distribution](../part-05-common-distributions/13-gamma-distribution.md) carry a shape parameter that is not a whole number.

## Where This Shows Up: Exponential Weighting

An exponentially weighted moving average assigns weight $(1-\lambda)\lambda^{k}$ to the observation $k$ periods ago:

$$\hat\sigma^2_t = (1-\lambda)\sum_{k=0}^{\infty}\lambda^{k}\,r_{t-k}^2.$$

Three facts about it are geometric series in disguise.

**The weights sum to one.** $(1-\lambda)\sum_{k\ge0}\lambda^k = (1-\lambda)\cdot\frac{1}{1-\lambda} = 1$, so the estimator is a genuine weighted average and not an arbitrarily scaled one.

**The effective window is $\lambda/(1-\lambda)$ periods.** The mean lag is $(1-\lambda)\sum_{k\ge0}k\lambda^k = \lambda/(1-\lambda)$, using the first differentiated identity. RiskMetrics' $\lambda = 0.94$ therefore has a centre of mass about 15.7 days back — comparable to a 30-day equally weighted window, which is the honest way to compare the two schemes.

**The half-life is $\log(1/2)/\log\lambda$.** The weight $k$ periods back has fallen to half its initial value when $\lambda^k = 1/2$.

```python
import math

lam = 0.94
w = [(1 - lam) * lam ** k for k in range(2000)]

print(f"weights sum to        {sum(w):.10f}")
print(f"effective window      {sum(k * wk for k, wk in enumerate(w)):.4f}"
      f"   closed form {lam / (1 - lam):.4f}")
print(f"half-life (periods)   {math.log(0.5) / math.log(lam):.4f}")
print(f"mass in first 30 obs  {sum(w[:30]):.4f}")
# => weights sum to        1.0000000000
#    effective window      15.6667   closed form 15.6667
#    half-life (periods)   11.2023
#    mass in first 30 obs  0.8437
```

Eighty-four percent of an EWMA's weight sits in its most recent 30 observations, which is why the estimator reacts quickly to a volatility spike and why it forgets a crisis about as fast. The parameter choices in [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md) and the decay constants in [Time Series](../../part-03-statistics/03-time-series.md) are choices about where on this trade-off to sit, and the closed forms above are how to compare a $\lambda$ to a window length without guessing.

## Rates, Not Just Limits

Two convergent quantities can behave completely differently in practice, and the difference is the rate. Compare a geometric tail with a $p$-series tail:

```python
import math

lam = 0.94
targets = (0.90, 0.99, 0.999)

for tgt in targets:
    # observations needed for an EWMA to accumulate `tgt` of its weight
    k_geo = math.ceil(math.log(1 - tgt) / math.log(lam))
    # terms needed for sum 1/i^2 to reach `tgt` of its limit pi^2/6
    limit, s, k_pow = math.pi ** 2 / 6, 0.0, 0
    while s < tgt * limit:
        k_pow += 1
        s += 1 / k_pow ** 2
    print(f"{tgt:6.3f}   geometric {k_geo:5d}   1/i^2 {k_pow:7d}")
# =>  0.900   geometric    38   1/i^2       6
#     0.990   geometric    75   1/i^2      61
#     0.999   geometric   112   1/i^2     608
```

Geometric decay costs a constant number of extra terms per decimal place; the $p$-series costs a factor of ten. Both converge, and only one of them converges fast enough to truncate casually. The same distinction decides whether a Monte Carlo estimator is usable at $10^4$ paths or needs $10^8$ ([Monte Carlo Simulation](../part-09-monte-carlo-methods/03-monte-carlo-simulation.md)), and whether a Markov chain's forgetting of its initial state is a detail or a modelling problem ([Markov Chains](../part-08-stochastic-processes/05-markov-chains.md)) — where convergence to the stationary distribution is geometric, at a rate set by the second-largest eigenvalue of the transition matrix.
