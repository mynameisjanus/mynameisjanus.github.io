# Negative Binomial Distribution

This family arrives by two routes that have nothing obvious to do with each other. One is a waiting time — how many trials until the $r$-th success — and it is the reason for the name. The other is a count of events whose rate is itself uncertain, and it is the reason the family is worth knowing. They produce the same distribution, and almost every use of it in finance is the second one wearing the first one's clothes.

This page covers the waiting-time mass function and the conventions attached to it, the decomposition into $r$ independent geometrics that supplies both moments with no new summation, the gamma mixture of Poissons that produces the identical law from an unrelated premise, and overdispersion — the property that gives the family its only real job. It does not cover the case $r=1$, which is [Geometric Distribution](03-geometric-distribution.md); it does not cover the equidispersed limit the family collapses to when the rate stops moving, which is [Poisson Distribution](06-poisson-distribution.md); and it does not build the mixing law, which is [Gamma Distribution](13-gamma-distribution.md).

The trading stake is that a Poisson is what a risk system reaches for when it counts breaches, stop-outs, or fills, and a Poisson has no parameter for clustering. [Binomial Distribution](02-binomial-distribution.md) showed a pairwise correlation of $0.05$ between trades inflating a count's variance thirteenfold while leaving its mean untouched. The negative binomial is the smallest family that can carry that inflation, and the fourth section below prices what ignoring it costs in the tail.

## Waiting for the r-th Success

Run independent trials with success probability $p$ and let $X$ be the trial on which the $r$-th success occurs. For $\{X=k\}$ the $k$-th trial must be a success, and exactly $r-1$ of the preceding $k-1$ trials must have been successes, with no constraint on their order. That gives one binomial coefficient and the usual product of per-trial probabilities,

$$p_X(k)=\binom{k-1}{r-1}p^{r}(1-p)^{k-r},\qquad k=r,r+1,\ldots$$

The coefficient counts arrangements of the *prefix* only, which is the single structural difference from the binomial: there, all $n$ positions were free; here the last one is pinned to a success by the definition of "the $r$-th". Setting $r=1$ collapses the coefficient to $1$ and recovers the geometric.

As with the geometric there are two conventions, worth stating once rather than debugging later.

| Convention | Support | Mean | Variance |
|---|---|---|---|
| Trials until the $r$-th success | $k\ge r$ | $r/p$ | $r(1-p)/p^2$ |
| Failures before the $r$-th success | $k\ge0$ | $r(1-p)/p$ | $r(1-p)/p^2$ |

`scipy.stats.nbinom` and `numpy`'s `negative_binomial` both use the second. This page states results in the first and converts where the code needs it.

## A Sum of r Geometrics

The waiting time to the $r$-th success decomposes with no algebra at all. Wait for the first success, then start counting afresh — memorylessness, proved in [Geometric Distribution](03-geometric-distribution.md), guarantees the next wait is an independent copy of the first — and repeat $r$ times. So

$$X=G_1+G_2+\cdots+G_r,\qquad G_i\ \text{independent}\ \mathrm{Geom}(p).$$

??? note "Proof that both moments follow from the decomposition with no new summation"
    Linearity of expectation gives $\mathbb{E}[X]=\sum_{i=1}^{r}\mathbb{E}[G_i]=r/p$ immediately, and as [Binomial Distribution](02-binomial-distribution.md) emphasised, that step needs nothing whatever about the joint law of the $G_i$.

    The variance is where the independence is spent: $\mathrm{var}(X)=\sum_i\mathrm{var}(G_i)=r(1-p)/p^2$ holds because the cross-covariances vanish, and they vanish because memorylessness makes each wait genuinely independent of its predecessors. Strip that property away and the decomposition still delivers the mean and delivers nothing else.

    Notice what the argument did not require: no differentiated series, no manipulation of $\binom{k-1}{r-1}$, and no summation over the support at all. The direct route — writing $\sum_k k\binom{k-1}{r-1}p^r(1-p)^{k-r}$ and absorbing the $k$ into the coefficient — works and is several lines longer. The decomposition is available because the family is closed under addition in $r$, the same structural fact that makes two independent negative binomials on a common $p$ add to a third.

## The Same Law from a Random Poisson Rate

Now forget waiting times entirely. Suppose events arrive as a Poisson process, but the rate is neither known nor constant — it is itself drawn from a gamma law before the counting starts. That is the position of a desk whose daily fill count depends on a volatility regime it cannot observe. The resulting count is negative binomial.

??? note "Proof that a Poisson whose rate is gamma-distributed is negative binomial"
    Let $\Lambda\sim\mathrm{Gamma}(r,\theta)$ with density $f(\lambda)=\lambda^{r-1}e^{-\lambda/\theta}/(\Gamma(r)\theta^{r})$, and let $N\mid\Lambda=\lambda$ be Poisson with mean $\lambda$. Marginalising over the rate,

    $$\mathbf{P}(N=k)=\int_{0}^{\infty}\frac{e^{-\lambda}\lambda^{k}}{k!}\cdot\frac{\lambda^{r-1}e^{-\lambda/\theta}}{\Gamma(r)\theta^{r}}\,\mathrm{d}\lambda=\frac{1}{k!\,\Gamma(r)\theta^{r}}\int_{0}^{\infty}\lambda^{k+r-1}e^{-\lambda(1+1/\theta)}\,\mathrm{d}\lambda.$$

    The remaining integral is a gamma integral, equal to $\Gamma(k+r)\big(\tfrac{\theta}{1+\theta}\big)^{k+r}$. Substituting and writing $p=1/(1+\theta)$ collapses the constants to

    $$\mathbf{P}(N=k)=\frac{\Gamma(k+r)}{k!\,\Gamma(r)}\,p^{r}(1-p)^{k},$$

    which is the failures-before-the-$r$-th-success mass function, since $\Gamma(k+r)/(k!\,\Gamma(r))=\binom{k+r-1}{k}$ whenever $r$ is an integer.

    Two things are worth extracting. First, $r$ need not be an integer here — the gamma shape is any positive real, and the law is perfectly well defined even though "the $r$-th success" is then meaningless. The mixture representation is therefore the *larger* of the two, and the waiting-time story is the special case that happens to admit a combinatorial reading. Second, the gamma was not chosen for realism but because it is the conjugate mixing law that keeps the integral in closed form — the property developed in [Gamma Distribution](13-gamma-distribution.md) and used again in [Beta Distribution](14-beta-distribution.md).

Under this parameterisation the moments are $\mathbb{E}[N]=r\theta$ and $\mathrm{var}(N)=r\theta(1+\theta)$, so the ratio of variance to mean is $1+\theta$ — one plus the scale of the rate's own uncertainty.

```python
import numpy as np

rng = np.random.default_rng(19)
r, theta, n = 2.5, 2.0, 2_000_000                              # mean 5, var/mean = 3
p = 1 / (1 + theta)
wait = rng.negative_binomial(r, p, n)                          # failures before the r-th success
mix = rng.poisson(rng.gamma(r, theta, n))                      # Poisson with a random rate
print(f"  NegBin(r = {r}, theta = {theta})  ->  p = {p:.4f}")
print("      route                  mean       var    var/mean   P(N >= 15)")
for name, s in (("waiting time ", wait), ("gamma-Poisson", mix)):
    print(f"    {name} {s.mean():9.4f} {s.var():9.4f} {s.var() / s.mean():9.4f}"
          f" {(s >= 15).mean():12.5f}")
print(f"    exact         {r * theta:9.4f} {r * theta * (1 + theta):9.4f}"
      f" {1 + theta:9.4f}")
# =>   NegBin(r = 2.5, theta = 2.0)  ->  p = 0.3333
#          route                  mean       var    var/mean   P(N >= 15)
#        waiting time     4.9992   15.0149    3.0035      0.02609
#        gamma-Poisson    4.9942   14.9864    3.0007      0.02589
#        exact            5.0000   15.0000    3.0000
```

The two rows are the same distribution reached from premises that share no vocabulary. One counted trials until a target number of successes; the other never mentioned a success at all, and drew a rate before counting anything. They agree on the mean, on the variance, and on the probability of a count three times the mean, to the third decimal place in each case — because these are not two models that happen to sit close together. They are one law with two derivations.

## Overdispersion Is the Whole Point

A Poisson has one parameter, so its mean and its variance are the same number and cannot be set apart. The negative binomial has two, and the second buys precisely the freedom the Poisson lacks: the variance may exceed the mean by any factor. It may not fall below it, so the family covers overdispersion and not underdispersion — the right asymmetry for financial counts, where events cluster far more often than they space themselves out.

```python
import numpy as np
from scipy.stats import poisson, nbinom

mean = 5.0                                                     # expected breaches per year
print("  P(N >= k) for a count with mean 5, under three dispersion assumptions")
print("      var/mean       k=10       k=15       k=20    99th pct")
for ratio in (1.0, 2.0, 3.0):
    if ratio == 1.0:
        tail = [poisson.sf(k - 1, mean) for k in (10, 15, 20)]
        q99 = poisson.ppf(0.99, mean)
    else:
        theta = ratio - 1.0
        r, q = mean / theta, 1 / (1 + theta)
        tail = [nbinom.sf(k - 1, r, q) for k in (10, 15, 20)]
        q99 = nbinom.ppf(0.99, r, q)
    print(f"    {ratio:9.1f} {tail[0]:10.2e} {tail[1]:10.2e} {tail[2]:10.2e}"
          f" {q99:10.0f}")
print("  how far the tail at k = 20 is understated by assuming a Poisson")
for ratio in (2.0, 3.0):
    theta = ratio - 1.0
    print(f"    var/mean {ratio:.0f}:  "
          f"{nbinom.sf(19, mean / theta, 1 / (1 + theta)) / poisson.sf(19, mean):9.1f}x")
# =>   P(N >= k) for a count with mean 5, under three dispersion assumptions
#          var/mean       k=10       k=15       k=20    99th pct
#              1.0   3.18e-02   2.26e-04   3.45e-07         11
#              2.0   8.98e-02   9.61e-03   7.72e-04         14
#              3.0   1.23e-01   2.59e-02   4.90e-03         17
#      how far the tail at k = 20 is understated by assuming a Poisson
#        var/mean 2:     2236.1x
#        var/mean 3:    14193.5x
```

The mean is five in every row, so a report quoting only the expected number of breaches cannot tell these three situations apart. The tails are not close, and they separate faster the further out one looks. At ten breaches the three probabilities span a factor of four; at fifteen, a factor of a hundred; at twenty, a factor of fourteen thousand. The ninety-ninth percentile moves from $11$ to $14$ to $17$ — a buffer sized on the Poisson is more than a third too small at a dispersion of three, from a change that leaves the average exactly where it was. A capital buffer set at a Poisson $99$th percentile is therefore breached a good deal more often than one year in a hundred if the true rate wanders at all.

!!! warning "Assuming a Poisson count is assuming the rate is known, and it is the tail rather than the average that pays when it is not"
    Equidispersion is not a mild regularity condition. It is the strong claim that nothing about the arrival rate is uncertain or time-varying, and volatility regimes, liquidity droughts, and signals that fire in bursts each violate it — every one of them acting on the count exactly as a random $\Lambda$ does. The diagnostic is a single number, the ratio of sample variance to sample mean, and it costs nothing; [Poisson Distribution](06-poisson-distribution.md) runs it on clustered arrivals and finds the ratio far above one. The repair is this page's family, and its second parameter is not a nuisance to be profiled away — it is the estimate of how badly the rate moves.

## Which Derivation You Used Decides What You Can Estimate

The two routes give the same law and are not the same model, and the difference surfaces the moment anything is inferred from data.

Read $r$ as a count of successes and it is an integer fixed by the design of the experiment; the only unknown is $p$, and the data are a waiting time. Read $r$ as a gamma shape and it is a positive real to be estimated, the data are counts, and $r$ is measuring something physical — the stability of the rate, with $r\to\infty$ at fixed mean recovering the Poisson and small $r$ describing violent clustering. The same symbol is a design constant in one reading and the answer in the other.

That ambiguity is where the family gets misused. A fitted $r$ of $2.5$ is often reported as though it were a number of trials, when what it actually says is that the arrival rate has a coefficient of variation of $1/\sqrt{r}\approx0.63$ — that the rate moves by nearly two thirds of its own size. So the practical rule is to name the derivation before quoting the parameter, and when counting anything in a market, to assume the mixture reading is the operative one. The rate is almost never known, and the negative binomial's second parameter is the honest place to put that ignorance rather than the first place to optimise it away.
