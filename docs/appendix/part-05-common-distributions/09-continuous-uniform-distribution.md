# Continuous Uniform Distribution

The flat density is the least interesting law in this part and by a wide margin the most used. Every random number a computer generates starts here, every other distribution in the continuous half of this part is reachable from it by one transformation, and every $p$-value ever computed is a claim that some quantity is a draw from $\mathrm{Unif}(0,1)$. That last role is the one worth taking seriously, because it turns a statement about a test into a statement about a distribution — and distributions can be checked.

This page covers the flat density and the two moments, the probability integral transform that carries any continuous law onto this one, inverse-transform sampling that carries it back, and the reading of a $p$-value as a uniform variate together with the standard way a backtest destroys that property. It does not derive the transform, which is [Change of Variables](../part-03-random-variables/09-change-of-variables.md); it does not cover the finite-support case, which is [Discrete Uniform Distribution](08-discrete-uniform-distribution.md); and it does not develop simulation technique, which is [Part IX](../part-09-monte-carlo-methods/index.md).

The trading stake is that a $p$-value is only uniform when the null used to compute it is the null that generated the data. Returns are autocorrelated and backtest windows overlap, and both violations act the same way — they leave the test statistic's formula untouched and change its distribution. The last section runs a correctly specified test and two mildly misspecified ones side by side: the nominal $5\%$ test rejects at $5.2\%$, then $9.0\%$, then $19.8\%$, with no change to a single line of the testing code.

## The Flat Density

$X$ is uniform on $[a,b]$ when its density is constant there and zero elsewhere. Constancy plus integration to one fixes the value,

$$f_X(x)=\frac{1}{b-a}\ \ \text{for}\ x\in[a,b],\qquad F_X(x)=\frac{x-a}{b-a}\ \ \text{for}\ x\in[a,b].$$

The density is a rate rather than a probability, as [Probability Density Functions](../part-03-random-variables/04-probability-density-functions.md) insists, and this family makes the distinction unusually visible: on $[0,0.1]$ the uniform density is $10$, a number that could not be a probability and is not trying to be. The distribution function is the more useful object here because it is a straight line, and a straight line is what makes the next two sections work.

The moments are

$$\mathbb{E}[X]=\frac{a+b}{2},\qquad \mathrm{var}(X)=\frac{(b-a)^{2}}{12}.$$

Both follow from integrating $x$ and $x^2$ against a constant. The variance is the continuous limit of the discrete uniform's $(n^{2}-1)/12$, with $n$ read as the width in grid units — the $-1$ being the correction for the grid's granularity, which vanishes as the spacing goes to zero.

## The Probability Integral Transform

If $X$ has a continuous distribution function $F$, then $F(X)\sim\mathrm{Unif}(0,1)$. This holds whatever $F$ is: normal, Student's $t$, a fitted mixture, an empirical distribution smoothed. The uniform is where every continuous law goes when it is measured against its own quantiles.

Running it backwards is the more constructive direction. Given $U\sim\mathrm{Unif}(0,1)$, the variable $F^{-1}(U)$ has distribution function $F$ — which is inverse-transform sampling, and is why a single uniform generator suffices to produce every other law.

??? note "Proof that the quantile function applied to a uniform recovers the law, with no continuity assumed"
    Define the quantile function as the generalised inverse $F^{-1}(u)=\inf\{x:F(x)\ge u\}$, which exists for every distribution function including step functions. The claim is that $\mathbf{P}(F^{-1}(U)\le x)=F(x)$ for all $x$.

    The key is the equivalence $F^{-1}(u)\le x \iff u\le F(x)$. Going right: if $u\le F(x)$ then $x$ belongs to the set whose infimum defines $F^{-1}(u)$, so the infimum is at most $x$. Going left: if $F^{-1}(u)\le x$ then by right-continuity of $F$ the infimum is attained, so $F(F^{-1}(u))\ge u$, and monotonicity gives $F(x)\ge F(F^{-1}(u))\ge u$. Hence

    $$\mathbf{P}\big(F^{-1}(U)\le x\big)=\mathbf{P}\big(U\le F(x)\big)=F(x),$$

    the last step being the definition of a uniform on $[0,1]$.

    Right-continuity of $F$ is the load-bearing hypothesis and it is available for every distribution function, which is why this direction needs no assumptions while the forward direction does. The forward transform $F(X)\sim\mathrm{Unif}(0,1)$ genuinely requires $F$ continuous: if $F$ jumps, $F(X)$ has an atom and cannot be uniform. That asymmetry matters in practice, because a strategy that is sometimes flat has an atom at zero return, and its $F(X)$ is not uniform no matter how correct the model is.

```python
import numpy as np
from scipy.stats import norm, t, expon, kstest

rng = np.random.default_rng(61)
n = 500_000
print("  F(X) is uniform whatever F was")
print("      law                sample mean    sample var     KS vs Unif(0,1)   p")
for name, dist, draw in (("normal(0.0004, 0.011)", norm(0.0004, 0.011),
                          rng.normal(0.0004, 0.011, n)),
                         ("t(2.65)              ", t(2.65), rng.standard_t(2.65, n)),
                         ("exponential(1/50)    ", expon(scale=50), rng.exponential(50, n))):
    u = dist.cdf(draw)
    ks = kstest(u, "uniform")
    print(f"    {name} {u.mean():13.5f} {u.var():13.6f} {ks.statistic:16.5f} {ks.pvalue:7.3f}")
print(f"    exact for Unif(0,1)   {0.5:13.5f} {1 / 12:13.6f}")
# =>   F(X) is uniform whatever F was
#          law                sample mean    sample var     KS vs Unif(0,1)   p
#        normal(0.0004, 0.011)       0.49968      0.083090          0.00139   0.285
#        t(2.65)                     0.50080      0.083257          0.00177   0.087
#        exponential(1/50)           0.50000      0.083375          0.00064   0.986
#        exact for Unif(0,1)         0.50000      0.083333
```

Three laws with nothing in common — a near-symmetric normal, a Student's $t$ so heavy it has no variance to speak of, and a one-sided exponential with a mean of fifty — all land on the same uniform, with mean $0.5$ and variance $1/12$ and a Kolmogorov–Smirnov test that finds nothing to report. The transform has erased every distinguishing feature, which is exactly its purpose: what survives $F(X)$ is not the shape of the law but whether the data were really drawn from the $F$ that was applied.

## Inverse-Transform Sampling

The reverse direction is the reason this page sits where it does in the reading order. Everything in the rest of this part can be generated from uniforms:

| Target | Construction from $U$ or from uniforms |
|---|---|
| $\mathrm{Exp}(\lambda)$ | $-\log(U)/\lambda$, since $F^{-1}(u)=-\log(1-u)/\lambda$ and $1-U$ is uniform too |
| $\mathrm{Unif}(a,b)$ | $a+(b-a)U$ |
| Any discrete law | the smallest $k$ with $F(k)\ge U$ |
| $\mathcal{N}(0,1)$ | no closed-form $F^{-1}$; use a pair of uniforms through a polar transform |
| $\mathrm{Gamma}$, $\mathrm{Beta}$, $t$ | built from the above by sums and ratios, as their own pages show |

The normal's absence from the closed-form column is not an oversight and is worth registering. Its distribution function has no elementary inverse, which is why sampling it needs a different device and why $\Phi$ and $\Phi^{-1}$ are tabulated functions rather than formulas — a fact [The Gaussian Distribution](11-gaussian-distribution.md) has to work around rather than solve.

## A p-Value Is a Uniform Random Variable, Until It Isn't

A $p$-value is $1-F_0(T)$ for an observed statistic $T$ and its null distribution $F_0$. By the transform above it is $\mathrm{Unif}(0,1)$ under the null — *provided $F_0$ is the distribution the data actually followed*. That proviso is the entire content of the section, and it is where backtests fail.

```python
import numpy as np
from scipy.stats import ttest_1samp, kstest

rng = np.random.default_rng(67)
reps, n = 20_000, 252                                          # a year of days
print("  a nominal 5% t-test on the mean, under three nulls that all have mean zero")
print("      data              rejects at 5%    KS vs uniform    mean p    median p")
for name, ar in (("iid              ", 0.0), ("AR(1), phi = 0.15", 0.15),
                 ("AR(1), phi = 0.40", 0.40)):
    e = rng.standard_normal((reps, n))
    x = e.copy()
    if ar:
        for i in range(1, n):                                  # same variance, some memory
            x[:, i] = ar * x[:, i - 1] + e[:, i] * np.sqrt(1 - ar ** 2)
    p = ttest_1samp(x, 0.0, axis=1).pvalue
    ks = kstest(p, "uniform")
    print(f"    {name} {(p < 0.05).mean():13.4f} {ks.statistic:16.5f}"
          f" {p.mean():10.4f} {np.median(p):10.4f}")
# =>   a nominal 5% t-test on the mean, under three nulls that all have mean zero
#          data              rejects at 5%    KS vs uniform    mean p    median p
#        iid                      0.0519          0.00407     0.4993     0.4976
#        AR(1), phi = 0.15        0.0899          0.07438     0.4517     0.4310
#        AR(1), phi = 0.40        0.1983          0.19944     0.3703     0.3055
```

All three rows have a true mean of exactly zero, all three have unit variance, and all three are tested by identical code. The iid row behaves: it rejects at $5.2\%$ against a nominal $5\%$, and its Kolmogorov–Smirnov statistic is the small number a correct test produces. The autocorrelated rows reject at $9.0\%$ and $19.8\%$ — the second is four times the advertised rate, from a persistence of $0.40$ that is unremarkable in daily data. The diagnosis, though, is in the KS column rather than the rejection column: the statistic climbs from $0.004$ to $0.074$ to $0.199$, which says the $p$-values are not uniform, so they are not $p$-values, and the $5\%$ printed on the report is a label rather than a property.

The mean and median columns show why summary statistics are the wrong instrument for catching this. They do drift — $0.499$, $0.452$, $0.370$ — but a drift of that size is entirely unremarkable across a research programme where the alternative is often true and $p$-values are *supposed* to be small. Nothing about a mean $p$-value of $0.45$ announces a broken null, whereas the KS statistic is comparing the whole shape against a fixed reference and has nowhere to hide.

!!! warning "A p-value that is not uniform under the null is not a p-value, and the failure is invisible to every check except a distributional one"
    The test statistic, its formula, its degrees of freedom, and the code computing it are all correct in the second row above. What is wrong is $F_0$: the $t$ distribution assumed by `ttest_1samp` is the null for *independent* observations, and the data have memory. Overlapping windows do the same thing more severely — a signal evaluated on rolling $252$-day windows produces adjacent statistics sharing $251$ of their observations, which is autocorrelation of about $0.996$. The repair is not a different threshold but a different null: block bootstrap, an explicit autocorrelation correction, or a statistic built on non-overlapping data. [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) uses the first, and [Independence](../part-02-probability-foundations/05-independence.md) is where the assumption being violated is defined.

So the flat density's importance is entirely instrumental, and both of its jobs are the same job seen from opposite ends. Going forward, $F(X)$ collapses every continuous law onto the uniform, which is what makes a $p$-value comparable across tests that have nothing else in common. Going backward, $F^{-1}(U)$ builds every law from the uniform, which is what makes simulation possible at all.

The practical rule follows from the first of those. Because uniformity is the common currency, it is also the common diagnostic: whenever a procedure emits something that ought to be uniform — $p$-values, PIT residuals from a fitted density, rank transforms of a forecast — plot it, or run one line of Kolmogorov–Smirnov against it. It is the cheapest specification test available, it requires no alternative hypothesis, and it fails loudly in exactly the cases where the model is wrong in a way that matters.
