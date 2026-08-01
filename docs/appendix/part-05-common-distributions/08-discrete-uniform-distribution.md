# Discrete Uniform Distribution

A uniform law on a finite set is what is left when every structural claim has been withdrawn, which makes it useless as a description of anything and indispensable as a reference. Nothing in a market is uniform. The value of the family is that it says, exactly and with no free parameters, what "nothing is going on here" would look like — and because it has no parameters, the smallest departure from it is detectable with an embarrassingly small amount of data.

This page covers the mass function, the mean and the variance derived from a telescoping identity rather than quoted, the maximum-entropy property that explains why this is the law ignorance selects, and the one place in trading where the uniform is a genuine and testable null: the last digit of a price on a tick grid. It does not cover the flat density on an interval, which is [Continuous Uniform Distribution](09-continuous-uniform-distribution.md); it does not build the statistic that tests the null across all cells at once, which is [Chi-Square Distribution](15-chi-square-distribution.md); and it does not treat the vector of cell counts, which is [Multinomial Distribution](07-multinomial-distribution.md).

The trading stake is that prices are not real numbers. They live on a grid of ticks, and the last digit of a traded price is a draw from a ten-point support that would be uniform if nothing but arithmetic determined it. It is not uniform: human quoting habits pile trades onto round numbers, and [Market Microstructure](../../part-01-foundations/03-market-microstructure.md) describes the mechanisms. The fourth section shows how little data it takes to prove it.

## Equally Likely Outcomes on a Finite Range

Let $X$ be uniform on the integers $a,a+1,\ldots,b$, and write $n=b-a+1$ for the number of them. Every outcome carries the same mass, and the masses must sum to one, so

$$p_X(k)=\frac{1}{n},\qquad k=a,a+1,\ldots,b.$$

There is no parameter beyond the endpoints, and the endpoints are usually fixed by the problem rather than estimated — a die has six faces, a tick grid has ten digits, a month has a known number of trading days. That is the family's defining feature: with the support given, the law is completely determined, and there is nothing left to fit.

The moments are

$$\mathbb{E}[X]=\frac{a+b}{2},\qquad \mathrm{var}(X)=\frac{n^{2}-1}{12}.$$

The mean is the midpoint by symmetry — pair $a$ with $b$, $a+1$ with $b-1$, and every pair averages to $(a+b)/2$. The variance depends only on how many points there are and not on where they sit, which is the shift-invariance of [Variance](../part-04-expectation-and-moments/02-variance.md) restated for this family.

??? note "Proof of the variance from a telescoping sum of cubes"
    Shift to $Y=X-a$, uniform on $\{0,1,\ldots,n-1\}$; the variance is unchanged. We need $\sum_{k=0}^{n-1}k^{2}$, and the cleanest route is a telescoping identity rather than an induction. Expanding,

    $$k^{3}-(k-1)^{3}=3k^{2}-3k+1.$$

    Sum both sides from $k=1$ to $m$. The left telescopes to $m^{3}$, since every interior term cancels against its neighbour. The right gives $3\sum k^{2}-3\sum k+m$, and $\sum_{k=1}^{m}k=m(m+1)/2$ is the same telescoping trick one order down. Solving,

    $$\sum_{k=1}^{m}k^{2}=\frac{1}{3}\Big(m^{3}+\frac{3m(m+1)}{2}-m\Big)=\frac{m(m+1)(2m+1)}{6}.$$

    Now with $m=n-1$: $\mathbb{E}[Y]=(n-1)/2$ and $\mathbb{E}[Y^{2}]=\frac{(n-1)n(2n-1)}{6n}=\frac{(n-1)(2n-1)}{6}$. Subtracting,

    $$\mathrm{var}(Y)=\frac{(n-1)(2n-1)}{6}-\frac{(n-1)^{2}}{4}=\frac{(n-1)(n+1)}{12}=\frac{n^{2}-1}{12}.$$

    The telescoping is worth more than the answer. It needs no guess to verify and no induction hypothesis, and the same manoeuvre one order up gives $\sum k^{3}$ — the general pattern being that a sum of $k^{p}$ falls out of the expansion of $k^{p+1}-(k-1)^{p+1}$ once all lower sums are known. The $1/12$ that appears here is the same $1/12$ that appears in the continuous uniform's variance $(b-a)^{2}/12$, and the next page explains why: as $n$ grows, $(n^2-1)/12\to n^2/12$, and $n$ is the width of the range in grid units.

## Maximum Entropy, and What That Justifies

The uniform is the distribution on a finite support that maximises the entropy $H(p)=-\sum_k p_k\log p_k$. That is the formal content of calling it the law of maximum ignorance, and it is a one-line consequence of Jensen's inequality.

??? note "Proof that the uniform maximises entropy on a finite support"
    Let $p$ be any distribution on $n$ points and $u_k=1/n$ the uniform. Consider the difference from the maximum,

    $$\log n-H(p)=\sum_k p_k\log\frac{p_k}{u_k},$$

    which is the Kullback–Leibler divergence of $p$ from $u$. Applying Jensen's inequality to the convex function $t\log t$, or equivalently applying it to $-\log$ and the ratios $u_k/p_k$, gives

    $$\sum_k p_k\log\frac{p_k}{u_k}=-\sum_k p_k\log\frac{u_k}{p_k}\ \ge\ -\log\sum_k p_k\frac{u_k}{p_k}=-\log 1=0,$$

    with equality exactly when $u_k/p_k$ is constant, that is when $p=u$. So $H(p)\le\log n$ always, uniquely attained by the uniform. The convexity argument is the one set out in [Expected Value](../part-04-expectation-and-moments/01-expected-value.md), used here on a finite sum.

    What the theorem licenses is narrower than it is usually taken to be. It says the uniform is the least committed distribution *given that the support is the one specified* — and the choice of support is itself a strong modelling claim that the theorem is silent about. Declaring the last digit uniform on ten values already asserts that all ten are reachable and that nothing outside them is, which on a market with a five-cent minimum increment would be false before any data arrived.

```python
import numpy as np

rng = np.random.default_rng(53)
for a, b in ((1, 6), (0, 9), (0, 251)):
    n = b - a + 1
    x = rng.integers(a, b + 1, 2_000_000)
    print(f"  Unif({a}, {b}): n = {n:4d}   mean {x.mean():9.4f} (exact {(a + b) / 2:8.4f})"
          f"   var {x.var():11.4f} (exact {(n ** 2 - 1) / 12:11.4f})")
print("  entropy against the uniform maximum, on ten points")
for name, p in (("uniform      ", np.full(10, 0.10)),
                ("mild rounding", np.array([.14, .08, .09, .09, .09, .12, .09, .09, .09, .12])),
                ("heavy        ", np.array([.30, .05, .05, .05, .05, .25, .05, .05, .05, .10]))):
    print(f"    {name}  H = {-(p * np.log(p)).sum():.5f}   log 10 = {np.log(10):.5f}"
          f"   shortfall {np.log(10) + (p * np.log(p)).sum():.5f}")
# =>   Unif(1, 6): n =    6   mean    3.4985 (exact   3.5000)   var      2.9170 (exact      2.9167)
#      Unif(0, 9): n =   10   mean    4.4990 (exact   4.5000)   var      8.2569 (exact      8.2500)
#      Unif(0, 251): n =  252   mean  125.5269 (exact 125.5000)   var   5291.4695 (exact   5291.9167)
#      entropy against the uniform maximum, on ten points
#        uniform        H = 2.30259   log 10 = 2.30259   shortfall 0.00000
#        mild rounding  H = 2.28647   log 10 = 2.30259   shortfall 0.01612
#        heavy          H = 1.98653   log 10 = 2.30259   shortfall 0.31605
```

The three uniforms reproduce both closed forms exactly, including the $n^2/12$ growth that makes the $251$-point case have a variance over five thousand. The entropy rows show the maximum being approached from below and never reached: a mild rounding pattern costs $0.016$ nats out of $2.303$, and a heavy one costs $0.32$. Those shortfalls look small, and the next section is about how easy they are to detect anyway.

## A Tick Grid Is Not Uniform

The last digit of a traded price has a ten-point support and, absent human preferences, no reason to favour any of them. The null is therefore parameter-free, which is what makes it powerful.

```python
import numpy as np
from scipy.stats import chi2

rng = np.random.default_rng(59)
p = np.array([.14, .08, .09, .09, .09, .12, .09, .09, .09, .12])  # a mild rounding pattern
print("  power of a goodness-of-fit test against the uniform-digit null")
print("      trades      mean chi2      5% crit     power     median p-value")
for n in (200, 1000, 6410, 25_000):
    counts = rng.multinomial(n, p, 4000)
    stat = ((counts - n / 10) ** 2 / (n / 10)).sum(axis=1)
    crit = chi2.ppf(0.95, 9)
    print(f"  {n:9d} {stat.mean():13.2f} {crit:12.2f} {(stat > crit).mean():9.3f}"
          f" {np.median(chi2.sf(stat, 9)):15.2e}")
null = ((rng.multinomial(6410, np.full(10, 0.1), 4000) - 641.0) ** 2 / 641.0).sum(axis=1)
print(f"  control — the same test on genuinely uniform digits, n = 6410:"
      f" rejects {(null > chi2.ppf(0.95, 9)).mean():.3f}")
# =>   power of a goodness-of-fit test against the uniform-digit null
#          trades      mean chi2      5% crit     power     median p-value
#            200         15.88        16.92     0.390        9.09e-02
#           1000         42.93        16.92     0.989        3.71e-06
#           6410        227.31        16.92     1.000        9.06e-44
#          25000        859.08        16.92     1.000       7.56e-179
#      control — the same test on genuinely uniform digits, n = 6410: rejects 0.042
```

The deviation being tested is small — a fourteen-percent share on the digit zero instead of ten, and twelve on the fives — and it is not the kind of thing that would be visible in a histogram of a few hundred prices. It does not need to be. At two hundred trades the test is weak, firing $39\%$ of the time. At a thousand it fires $98.9\%$ of the time, and by the twenty-five thousand a liquid name generates in an afternoon the median $p$-value is $10^{-179}$ — a number with no interpretation beyond *decided*. The last line is the control: run on genuinely uniform digits the same test rejects $4.2\%$ of the time, so the power above belongs to the deviation and not to a broken statistic.

!!! note "A parameter-free null is the sharpest kind, and the discrete uniform is the only parameter-free law in this part"
    Every other family here has at least one parameter that must be estimated from the same data used to test it, which costs degrees of freedom and blunts the test. The uniform-digit null costs none: the support is known, the probabilities are $1/10$ by construction, and all nine degrees of freedom are available for detecting the deviation. That is why this particular market fact — price clustering on round numbers — is one of the most robustly established in microstructure, and why it can be demonstrated on an afternoon of data rather than a decade of it.

## The Uniform as a Null Rather Than a Model

There is a temptation, having established that the digits are not uniform, to reach for a better-fitting ten-point law and report its parameters. That is almost always the wrong move, and the reason is the entropy theorem read backwards.

A fitted ten-point distribution has nine free parameters and will fit any digit histogram perfectly, which means it explains nothing — it is the histogram, relabelled. The uniform's value came entirely from having no parameters and therefore being refutable, and a model flexible enough to accommodate the data has given that up. What is worth extracting from the rejection is not a better distribution but the *mechanism*: which digits are favoured, by how much, and whether the pattern is stable across venues, times of day, and price levels. Those are questions about market structure, and the uniform's job was only to establish that there was something to ask about.

So this is the one family in this part whose usefulness is inversely proportional to its accuracy. It is never true, it is cheap to reject, and the rejection is informative precisely because the null was so specific. The practical rule is to reach for the discrete uniform whenever a quantity has a known finite support and no obvious reason to prefer one value — not because the answer will be yes, but because the answer is worth having and costs almost nothing to obtain.
