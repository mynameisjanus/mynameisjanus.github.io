# Hypergeometric Distribution

Sampling without replacement gives a count with exactly the binomial's mean and a strictly smaller variance. The usual reading of that is a warning about finite populations, but the shrinkage is better understood as a measurement: not being allowed to draw the same item twice is information, and the correction says how much of it there is. The awkward part, and the one this page spends its last section on, is that the same count can be described two ways round, the two descriptions disagree about how large the correction is, and only one of them is telling you whether it matters.

This page covers the counting derivation, the mean by exchangeable indicators, the finite-population correction and where it comes from, the limit in which the correction disappears and the binomial returns, and the family's real job: supplying the exact null distribution for the question of whether a signal caught more than its share of the days that mattered. It does not cover sampling with replacement, which is [Binomial Distribution](02-binomial-distribution.md); it does not cover more than two categories, which is [Multinomial Distribution](07-multinomial-distribution.md); and it does not develop permutation testing in general, which is [Part XII](../part-12-hypothesis-testing/index.md).

The trading stake is a question every trend follower's tearsheet invites. The book's sleeve is long on $71\%$ of days, and twenty-five years contain $6{,}410$ of them. Of the fifty best days in that history, how many should a signal with no skill whatever have been long for — and how far above that number does a real signal have to land before the excess means anything? [Performance Metrics and Reporting](../../part-05-backtesting-engine/04-performance-metrics-and-reporting.md) supplies both inputs; the last section answers the question exactly, and finds that the binomial everyone reaches for instead is very nearly right — for a reason that is not the one usually given.

## Counting the Draws

Take a population of $N$ items of which $K$ are of interest, and draw $n$ of them without replacement. Every unordered sample of size $n$ is equally likely, so the probability of getting exactly $k$ items of interest is a ratio of counts: choose which $k$ of the $K$ interesting items appear, choose which $n-k$ of the $N-K$ others fill out the sample, and divide by the number of samples,

$$p_X(k)=\frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}},\qquad \max(0,\,n-N+K)\le k\le\min(n,K).$$

Every coefficient here is an unordered draw without replacement, straight from [Counting Principles](../part-01-mathematical-foundations/02-counting-principles.md). The support bounds are worth reading rather than skipping: $k$ cannot exceed the number of interesting items available or the number of draws taken, and it cannot fall below $n-(N-K)$, because a sample larger than the uninteresting part of the population is forced to contain some interesting items. The binomial has no such constraints, and their presence is the first sign that the two laws are genuinely different objects rather than one being an approximation of the other.

The symmetry $\binom{K}{k}\binom{N-K}{n-k}/\binom{N}{n}$ in $K\leftrightarrow n$ is not an accident: it says that drawing $n$ items and counting how many are interesting is the same experiment as labelling $K$ items interesting and counting how many were drawn. That interchangeability is what makes the family a permutation null.

## The Mean Is the Binomial's, and Needs No Independence

Write $p=K/N$ for the population fraction. Then

$$\mathbb{E}[X]=np,$$

which is precisely the binomial mean, despite the draws being dependent in a way that visibly matters — after a success the next draw is less likely to succeed.

??? note "Proof that the mean is np by exchangeability, with no independence used anywhere"
    Let $X_i$ indicate that the $i$-th draw is an interesting item, so $X=X_1+\cdots+X_n$. Consider the $i$-th draw in isolation. Before anything is observed, every item in the population is equally likely to be the one that lands in position $i$ — the draws are exchangeable, and no position is privileged — so $\mathbf{P}(X_i=1)=K/N=p$ for every $i$, including the last.

    Linearity of expectation then gives $\mathbb{E}[X]=\sum_i\mathbb{E}[X_i]=np$, and as in [Binomial Distribution](02-binomial-distribution.md), that step is indifferent to the joint law. This is the cleanest possible illustration of the point: the indicators here are strongly and obviously dependent, the sum is not remotely binomial, and the mean is the binomial's anyway.

    What the argument uses is exchangeability of the draws, which is weaker than independence and is exactly what an unordered uniform sample provides. Remove it — sample the interesting items preferentially, as any real selection procedure does — and the marginal $\mathbf{P}(X_i=1)$ is no longer $K/N$, the first equality fails, and the mean moves. So the fragile hypothesis is that the sample is uniform, not that the draws are independent.

## The Finite-Population Correction

The variance does feel the dependence, and it feels it through a single multiplicative factor:

$$\mathrm{var}(X)=np(1-p)\cdot\frac{N-n}{N-1}.$$

The first factor is the binomial variance; the second is the finite-population correction. It equals $1$ when $n=1$, falls as the sample takes in more of the population, and hits $0$ when $n=N$ — at which point the whole population has been drawn, $X=K$ with certainty, and a variance of zero is the only possible answer.

??? note "Proof of the correction from the covariance of two draws"
    Two distinct draws are negatively correlated, and the covariance is computable directly. By exchangeability, $\mathbf{P}(X_i=1,X_j=1)$ for $i\ne j$ is the probability that two specified positions both hold interesting items, which is $\frac{K}{N}\cdot\frac{K-1}{N-1}$. Hence

    $$\mathrm{cov}(X_i,X_j)=\frac{K(K-1)}{N(N-1)}-\Big(\frac{K}{N}\Big)^{2}=-\frac{p(1-p)}{N-1}.$$

    The sign is forced: an interesting item consumed by draw $i$ is unavailable to draw $j$, so one success makes the other less likely, always. Summing the $n$ diagonal terms and the $n(n-1)$ off-diagonal ones,

    $$\mathrm{var}(X)=np(1-p)-\frac{n(n-1)p(1-p)}{N-1}=np(1-p)\Big[1-\frac{n-1}{N-1}\Big]=np(1-p)\frac{N-n}{N-1}.$$

    This is the same double sum that [Binomial Distribution](02-binomial-distribution.md) used to show how correlation inflates a count's variance, run with the sign reversed. There, an average pairwise correlation of $+0.05$ multiplied the variance by $13$; here the induced correlation is $-1/(N-1)$, tiny per pair, but there are $n(n-1)$ pairs and the product $n\bar\rho$ is again what survives. The load-bearing quantity in both cases is the sampling fraction times the correlation, not either one alone.

```python
import numpy as np

rng = np.random.default_rng(31)
N, K = 6410, 50                                                # 25 years of days, the best 50
print("  hypergeometric variance against the binomial, as the sample grows")
print("      n      n/N     mean    sd(hyper)   sd(binom)    fpc    sd ratio")
for n in (64, 641, 3205, 4551, 6089):
    x = rng.hypergeometric(K, N - K, n, 400_000)
    p = K / N
    fpc = (N - n) / (N - 1)
    print(f"  {n:6d} {n / N:8.3f} {x.mean():8.3f} {x.std():11.4f}"
          f" {np.sqrt(n * p * (1 - p)):11.4f} {fpc:8.4f} {np.sqrt(fpc):9.4f}")
# =>   hypergeometric variance against the binomial, as the sample grows
#          n      n/N     mean    sd(hyper)   sd(binom)    fpc    sd ratio
#          64    0.010    0.499      0.7002      0.7038   0.9902    0.9951
#         641    0.100    5.004      2.1167      2.2273   0.9001    0.9488
#        3205    0.500   25.011      3.5199      4.9805   0.5001    0.7072
#        4551    0.710   35.501      3.1943      5.9348   0.2901    0.5386
#        6089    0.950   47.496      1.5368      6.8648   0.0501    0.2238
```

The mean column tracks $np$ exactly at every sampling fraction, which is the first proof printed. The two standard-deviation columns separate steadily, and the last column says by how much: the ratio is $\sqrt{(N-n)/(N-1)}$ and nothing else. At $n=6{,}089$ — drawing $95\%$ of the population — the true spread is under a quarter of what a binomial would claim, because almost nothing is left to be uncertain about.

One caution about reading that table, which the last section turns into the main point. The comparison is against $\mathrm{Binom}(n,K/N)$, the binomial that matches this orientation of the problem. The hypergeometric variance is symmetric under swapping $K$ and $n$, but the two factors it splits into are not, so "the finite-population correction" is only defined once an orientation is chosen, and the two orientations disagree about it while agreeing about the answer.

## The Binomial Limit

Hold $n$ fixed and let $N$ grow. The correction $(N-n)/(N-1)\to1$, the mass function converges to $\binom{n}{k}p^k(1-p)^{n-k}$, and the two families coincide. The usual rule of thumb is that the binomial is adequate when $n/N<0.05$, and the table above is the reason it is stated as a fraction rather than as a sample size: what governs the discrepancy is how much of the population was consumed, not how many items were drawn.

The direction of the error is fixed and worth carrying. Because the correction is always at most $1$, a binomial approximation always *overstates* the variance of a without-replacement count. It is therefore conservative for a significance test — too wide, not too narrow — and this is one of the few places in this part where the convenient approximation errs in the safe direction rather than the dangerous one.

The subtlety is which fraction the rule of thumb applies to. Because the law is symmetric in $K$ and $n$, the same count can be described as $n$ draws from a population containing $K$ interesting items or as $K$ draws from a population containing $n$ of them, and the correction is negligible as soon as the *smaller* of $n/N$ and $K/N$ is small. A problem that looks hopeless in one orientation can be routine in the other, which is exactly what happens next.

## The Exact Null for "Did the Signal Catch the Big Days?"

Now the trading stake. There are $N=6{,}410$ trading days and $K=50$ of them are the best by return. A signal that is long on $n=4{,}551$ of the days — the sleeve's $71\%$ — and has no ability whatever to tell one day from another is exactly a uniform sample of size $n$ from the $N$ days. The number of big days it happens to be long for is hypergeometric, and nothing else needs to be assumed.

```python
import numpy as np
from scipy.stats import hypergeom, binom

N, K, n = 6410, 50, 4551                                       # days, best days, days long
null = hypergeom(N, K, n)
print(f"  no-skill signal long {n / N:.0%} of days catches {null.mean():.1f}"
      f" of the top {K} on average, sd {null.std():.2f}")
for lbl, draws, good in (("draw 4551 days, 50 are big  ", n, K),
                         ("draw 50 big days, 4551 long ", K, n)):
    p = good / N
    print(f"    {lbl} np(1-p) {draws * p * (1 - p):7.3f}"
          f"   fpc {(N - draws) / (N - 1):7.4f}   var {draws * p * (1 - p) * (N - draws) / (N - 1):7.3f}")
print("      caught   hypergeom p    binomial p    ratio")
for caught in (40, 43, 45, 47):
    ph, pb = null.sf(caught - 1), binom.sf(caught - 1, K, n / N)
    print(f"    {caught:8d} {ph:13.5f} {pb:13.5f} {pb / ph:8.2f}")
print(f"  5% critical value:  hypergeometric {null.isf(0.05):.0f}"
      f"   binomial {binom.isf(0.05, K, n / N):.0f}")
# =>   no-skill signal long 71% of days catches 35.5 of the top 50 on average, sd 3.20
#        draw 4551 days, 50 are big   np(1-p)  35.222   fpc  0.2901   var  10.217
#        draw 50 big days, 4551 long  np(1-p)  10.295   fpc  0.9924   var  10.217
#          caught   hypergeom p    binomial p    ratio
#              40       0.10256       0.10346     1.01
#              43       0.01055       0.01079     1.02
#              45       0.00113       0.00117     1.04
#              47       0.00005       0.00006     1.05
#      5% critical value:  hypergeometric 41   binomial 41
```

A signal with no skill is long for about $35.5$ of the fifty best days, give or take $3.2$, and needs $41$ to clear a one-sided $5\%$ bar. Catching $40$ proves nothing; catching $45$ is decisive at better than one in eight hundred.

The two middle rows are the point of the section. They compute the same variance, $10.217$, from factors that share no digits — one splits it as $35.222\times0.2901$ and the other as $10.295\times0.9924$ — because the law does not care which side of the problem is called the sample. And in the second orientation the finite-population correction is $0.9924$, so the binomial is nearly exact: its $p$-values run $1$ to $5\%$ high, its critical value is identical, and no conclusion anywhere in the table would change.

!!! note "The sampling fraction that governs a finite-population correction is the smaller of the two, and a problem that looks badly non-binomial can be routinely binomial once it is turned around"
    Read as $4{,}551$ days drawn from $6{,}410$, this looks like a case where the binomial must fail: the sampling fraction is $71\%$ and the correction is $0.29$. Read as $50$ big days drawn from $6{,}410$, the fraction is $0.8\%$ and the correction is $0.99$. Both readings are correct and they give the same variance, so the second one settles it — the effective sample size here is fifty, not four and a half thousand, and fifty draws from six thousand items barely notice that the draws are not replaced. The general statement is that the correction depends on $\min(n,K)/N$, and the rule of thumb should be applied to that.

    This cuts the other way too. Had the question been about a rare regime — the twenty days of a liquidity crisis, say, against a signal active on twenty-five of them — both fractions would be small and the binomial would still be fine. It is only when *both* $n$ and $K$ are comparable to $N$ that the binomial genuinely breaks, and that is a narrower circumstance than the phrase "sampling without replacement" suggests.

!!! warning "The hypergeometric null tests whether the signal was long on the big days, not whether being long on them made money"
    The count discards the magnitudes, exactly as [Bernoulli Distribution](01-bernoulli-distribution.md) warned, and here the discarded quantity is the entire economics: the fifty best days are best by very different amounts, and a signal that caught the $50$th-best day and missed the best one scores identically to its opposite. The test also answers a question nobody quite asked — a strategy is not paid for being present on good days but for its return net of costs — so a decisive $p$-value here is evidence of selection ability and not of profitability. What it is genuinely good for is falsification: a signal that cannot beat this null has no timing ability at all, whatever its equity curve looks like.

So the family's value is not as a model of anything. It is as an exact null, available in closed form, for the very common situation where a procedure carves a fixed-size subset out of a finite record — days in a regime, trades in a bucket, names in a screen. Its worth is that it is exact and costs one line to evaluate, not that the binomial is dangerous; on the problem this page actually posed, the binomial was fine, and the section above had to work to establish that rather than assume it.

That is the habit the page is arguing for. The reflex on meeting a finite population is to reach for a correction, and the reflex on meeting a convenient approximation is to distrust it; both are cheaper than checking, and both were wrong here in opposite directions. The practical rule is to compute $\min(n,K)/N$ before deciding anything. Below a few percent the two laws agree and the argument is bookkeeping. Above a third in both orientations the binomial is not approximating the right question so much as answering a different one — and in between, the exact null is one function call away, so there is no reason to be estimating which side of the line the problem falls on.
