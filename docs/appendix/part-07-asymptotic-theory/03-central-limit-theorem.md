# The Central Limit Theorem

Add up enough independent shocks, standardize the sum, and the answer is normal no matter what the shocks looked like. That is the most useful theorem in applied statistics and the most abused, because it converges at different speeds in different parts of the distribution: the middle arrives early, the tail arrives late, and every risk number worth computing lives in the tail. A model can be simultaneously well within the theorem's reach for the purpose of testing whether a mean is positive and hopelessly outside it for the purpose of sizing a position.

This page covers the statement for sums and for means, its proof by transform, the repeated-convolution mechanism that makes it happen, the finite-sample rate and the Berry–Esseen bound that quantifies it, the multivariate version and the Cramér–Wold device that reduces it to the scalar case, and the finite-variance hypothesis that is the whole price of admission. It does not prove the laws of large numbers, which are [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md) and [The Strong Law of Large Numbers](02-strong-law-of-large-numbers.md); it does not build the normal distribution or its moment generating function, which is [The Gaussian Distribution](../part-05-common-distributions/11-gaussian-distribution.md); it does not construct the multivariate normal it converges to, which is [Multivariate Gaussian Distribution](../part-06-multivariate-probability/05-multivariate-gaussian.md); it does not describe what happens when the variance is infinite beyond naming the limit, which is [Heavy-Tailed Returns](../part-18-quant-finance-applications/12-heavy-tailed-returns.md); and it constructs no test, which is [Part XII](../part-12-hypothesis-testing/index.md).

The trading stake is a two-row table this course already published. [Hypothesis Testing and Multiple Testing](../../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) audits the assumptions behind a t-test on twenty-four years of daily returns and rules that normality is "violated — kurtosis 12" but "nearly harmless: at $n\approx6{,}000$ the CLT makes the *mean* nearly normal anyway", while finite variance is "the CLT's non-negotiable entry fee." Both rulings are correct. The fourth section shows they are correct about the $5\%$ level and wrong about the $0.1\%$ level at the same sample size, and the sixth shows that the same data has been fitted with a law under which the entry fee is unpaid.

## The Statement Is About the Sum, and the Mean Is a Corollary

Let $X_1,X_2,\dots$ be independent and identically distributed with mean $\mu$ and **finite** variance $\sigma^{2}>0$, and write $S_n=\sum_{i=1}^{n}X_i$. The central limit theorem says

$$\frac{S_n-n\mu}{\sigma\sqrt{n}}\ \Longrightarrow\ \mathcal{N}(0,1),$$

where $\Longrightarrow$ denotes **convergence in distribution**: $F_n(x)\to F(x)$ at every $x$ where the limiting $F$ is continuous. That last qualification is not decorative — it is the third and weakest of this part's three modes, it is defined through the cumulative distribution functions of [Cumulative Distribution Functions](../part-03-random-variables/02-cumulative-distribution-functions.md), and the reason for restricting to continuity points is exactly the one [Continuous Mapping Theorem](06-continuous-mapping-theorem.md) makes into a section. Convergence in distribution is implied by convergence in probability and implies neither of the other two modes; it is a statement about the *shape* of the ensemble and about nothing that happens on any path.

Dividing through by $n$ gives the version used on estimates,

$$\sqrt{n}\,\big(\bar X_n-\mu\big)\ \Longrightarrow\ \mathcal{N}(0,\sigma^{2}),$$

and the two forms are the same theorem. Which one to reach for is a matter of what is accumulating. A t-statistic accumulates an average and wants the second. A multi-period return accumulates a sum, and it is the first that [Exponentials, Logarithms, and Growth](../part-01-mathematical-foundations/07-exponentials-logarithms-growth.md) invokes when it argues that a sum of $T$ iid log returns is approximately normal and therefore that the price is approximately lognormal — a derivation that needs the sum form and would not follow from a statement about means.

The pairing of the two forms is also where the $\sqrt{T}$ scaling comes from. The sum's dispersion grows like $\sigma\sqrt{n}$ and the mean's shrinks like $\sigma/\sqrt{n}$, both from the same $\mathrm{var}(\bar X_n)=\sigma^{2}/n$ that [Variance](../part-04-expectation-and-moments/02-variance.md) derived without any limit at all. The central limit theorem adds no new scaling. It adds the shape.

## The Proof Is One Taylor Expansion Inside a Transform

The theorem's difficulty is that it is a statement about a distribution, and distributions do not combine easily under addition. The proof's whole idea is to move to a representation where addition becomes multiplication, do the easy thing there, and move back.

??? note "Proof that the standardized sum converges in distribution to the standard normal"
    Assume $\mu=0$ without loss of generality, since replacing $X_i$ by $X_i-\mu$ changes nothing on the left. Suppose first that the moment generating function $M(t)=\mathbb{E}[e^{tX}]$ is finite in a neighbourhood of the origin — the object [The Gaussian Distribution](../part-05-common-distributions/11-gaussian-distribution.md) builds and uses to prove that normals are closed under addition. Differentiating under the expectation gives $M(0)=1$, $M'(0)=\mathbb{E}[X]=0$ and $M''(0)=\mathbb{E}[X^{2}]=\sigma^{2}$, so Taylor's theorem from [Calculus Essentials](../part-01-mathematical-foundations/06-calculus-essentials.md) supplies

    $$M(s)=1+\tfrac{\sigma^{2}}{2}s^{2}+o(s^{2})\qquad\text{as } s\to0.$$

    Write $Z_n=S_n/(\sigma\sqrt{n})$. Because the $X_i$ are independent, the transform of a sum is the product of the transforms, and because they are identically distributed all $n$ factors are equal:

    $$M_{Z_n}(t)=\Big[M\!\left(\tfrac{t}{\sigma\sqrt{n}}\right)\Big]^{n}=\Big[1+\frac{t^{2}}{2n}+o\!\left(\tfrac1n\right)\Big]^{n}.$$

    Taking logarithms and using $\log(1+u)=u+O(u^{2})$ gives $n\log M_{Z_n}$ tending to $t^{2}/2$, so $M_{Z_n}(t)\to e^{t^{2}/2}$ — the moment generating function of a standard normal. The continuity theorem for transforms, which says that convergence of the transforms at every $t$ implies convergence in distribution, closes the argument.

    Two repairs make this the real proof rather than a sketch. The moment generating function need not exist — a $t$ distribution has no finite $\mathbb{E}[e^{tX}]$ for any $t\neq0$ — so the argument is run with the **characteristic function** $\varphi(t)=\mathbb{E}[e^{itX}]$ instead, which exists for every distribution because $\lvert e^{itx}\rvert=1$. The expansion becomes $\varphi(s)=1-\tfrac{\sigma^{2}}{2}s^{2}+o(s^{2})$, the limit becomes $e^{-t^{2}/2}$, and Lévy's continuity theorem replaces the one used above. Nothing else changes, and this is why the theorem applies to the heavy-tailed laws of the fourth section even though their moment generating functions do not exist.

    The load-bearing hypothesis is finite variance, and the proof shows exactly where it is spent: it is the existence of $M''(0)$, the coefficient on $s^{2}$. If $\mathbb{E}[X^{2}]=\infty$ that term is not there to be extracted, the expansion has a different leading behaviour — $\varphi(s)=1-c\lvert s\rvert^{\alpha}+o(\lvert s\rvert^{\alpha})$ for some $\alpha<2$ — and carrying it through the same algebra produces $e^{-c\lvert t\rvert^{\alpha}}$, the characteristic function of an $\alpha$-stable law rather than a normal. So the theorem does not merely fail without finite variance; it is replaced by a different theorem with a different limit and, critically, a different scaling, since the normalization becomes $n^{1/\alpha}$ rather than $n^{1/2}$. Nothing in the arithmetic of a backtest notices which regime it is in.

## Repeated Convolution Is the Mechanism, and It Is Visible in Six Steps

The transform proof is efficient and hides what is happening. [Functions of Random Variables](../part-03-random-variables/08-functions-of-random-variables.md) supplies the concrete picture: the density of a sum of independent variables is the convolution of their densities, and "iterating it is what drives a sum of many independent variables toward the bell shape". Iterating it is something a computer can do exactly, with no sampling error at all.

```python
import numpy as np
from scipy.stats import norm

pmf = np.array([0.70, 0.25, 0.00, 0.00, 0.00, 0.05])           # P(-1)=.70, P(0)=.25, P(4)=.05
lo = -1
print("  self-convolving a 3-point law: P(-1) = 0.70, P(0) = 0.25, P(4) = 0.05")
print("      terms   support        skew   excess kurt   sup|F - Phi|   x sqrt(k)   largest atom")
for k in (1, 2, 4, 8, 16, 32, 64):
    x = lo + np.arange(len(pmf))
    m = pmf @ x
    c = x - m
    v = pmf @ c ** 2
    s, ek = pmf @ c ** 3 / v ** 1.5, pmf @ c ** 4 / v ** 2 - 3
    gap = np.abs(np.cumsum(pmf) - norm.cdf((x - m + 0.5) / np.sqrt(v))).max()
    print(f"  {k:9d} {len(pmf):9d} {s:11.4f} {ek:13.4f} {gap:14.4f} {gap * np.sqrt(k):11.4f}"
          f" {pmf.max():14.4f}")
    pmf, lo = np.convolve(pmf, pmf), 2 * lo                    # double the number of terms
# =>   self-convolving a 3-point law: P(-1) = 0.70, P(0) = 0.25, P(4) = 0.05
#          terms   support        skew   excess kurt   sup|F - Phi|   x sqrt(k)   largest atom
#              1         6      3.2199       10.1600         0.2000      0.2000         0.7000
#              2        11      2.2768        5.0800         0.2159      0.3053         0.4900
#              4        21      1.6100        2.5400         0.1784      0.3568         0.3430
#              8        41      1.1384        1.2700         0.1381      0.3907         0.2059
#             16        81      0.8050        0.6350         0.0564      0.2257         0.0984
#             32       161      0.5692        0.3175         0.0385      0.2175         0.0638
#             64       321      0.4025        0.1587         0.0271      0.2170         0.0449
```

No randomness is involved; these are exact convolutions of an exact probability mass function, so every digit is a theorem rather than an estimate. The starting law is about as far from normal as a three-point law can be — seventy percent of its mass on one value, a five-percent chance of a jump five times the size of the common move, skewness $3.2199$ and excess kurtosis $10.1600$, which is very close to the kurtosis of daily SPY returns.

The two shape columns decay at exactly the rates the theory predicts. Skewness falls $3.2199\to2.2768\to1.6100\to\cdots\to0.4025$, which is division by $\sqrt{2}$ at every doubling: $3.2199/\sqrt{64}=0.4025$ to four decimals. Excess kurtosis falls $10.1600\to5.0800\to\cdots\to0.1587$, division by $2$ at every doubling, which is a factor of $64$ across the six steps shown. **Skewness dies like $1/\sqrt{k}$ and excess kurtosis like $1/k$**, so a law's asymmetry outlives its fat tails under aggregation, and asymmetry is what the next section shows doing the damage.

The distance to the normal itself is the fourth column, and it tells a less tidy story: $0.2000$, $0.2159$, $0.1784$, $0.1381$, $0.0564$, $0.0385$, $0.0271$. It rises before it falls, because at small $k$ the sum is still a coarse lattice — the last column shows a single atom holding $49\%$ of the mass at $k=2$ — and no lattice distribution can be close to a continuous one until its atoms are individually small. Once past that, $\sqrt{k}$ times the gap settles onto $0.2257$, $0.2175$, $0.2170$: the $1/\sqrt{n}$ rate again, arrived at from a third direction.

## The Limit Arrives Long Before the Tail Does

Convergence in distribution is convergence of the whole CDF, so it is tempting to treat "the CLT applies" as a single fact. It is not. The rate depends on where in the distribution the question is asked, and the gap between the centre and the tail is the difference between a t-test and a VaR.

```python
import numpy as np

rng = np.random.default_rng(7321)
z05, z01, z001 = 1.959963984540054, 2.3263478740408408, 3.090232306167813
laws = {                                                       # each standardized to variance 1
    "normal": (lambda s: rng.standard_normal(s), 1.5958),
    "t(5)": (lambda s: rng.standard_t(5, s) / np.sqrt(5 / 3), 3.4919),
    "t(2.65)": (lambda s: rng.standard_t(2.65, s) / np.sqrt(2.65 / 0.65), np.inf),
    "lognormal": (lambda s: (np.exp(rng.standard_normal(s)) - np.exp(0.5))
                  / np.sqrt(np.e * (np.e - 1)), 12.0655),
}
print("  actual size of a z-test on the mean, against three nominal levels")
print("            law        n   two-sided 5%   one-sided 1%   one-sided 0.1%   Berry-Esseen")
for name, (draw, rho) in laws.items():
    for n, reps in ((30, 400_000), (252, 200_000), (6_300, 40_000)):
        z = draw((reps, n)).mean(axis=1) * np.sqrt(n)
        be = 0.4748 * rho / np.sqrt(n)
        b = f"{be:15.4f}" if np.isfinite(be) else f"{'none exists':>15s}"
        print(f"  {name:>13s} {n:8d} {np.mean(np.abs(z) > z05):14.4f}"
              f" {np.mean(z < -z01):14.4f} {np.mean(z < -z001):16.5f}{b}")
# =>   actual size of a z-test on the mean, against three nominal levels
#                law        n   two-sided 5%   one-sided 1%   one-sided 0.1%   Berry-Esseen
#             normal       30         0.0496         0.0100          0.00098         0.1383
#             normal      252         0.0502         0.0102          0.00093         0.0477
#             normal     6300         0.0485         0.0104          0.00093         0.0095
#               t(5)       30         0.0503         0.0108          0.00148         0.3027
#               t(5)      252         0.0494         0.0100          0.00102         0.1044
#               t(5)     6300         0.0514         0.0095          0.00118         0.0209
#            t(2.65)       30         0.0416         0.0113          0.00374    none exists
#            t(2.65)      252         0.0444         0.0104          0.00249    none exists
#            t(2.65)     6300         0.0481         0.0098          0.00168    none exists
#          lognormal       30         0.0429         0.0001          0.00000         1.0459
#          lognormal      252         0.0478         0.0040          0.00013         0.3609
#          lognormal     6300         0.0491         0.0089          0.00040         0.0722
```

The first numeric column vindicates the published ruling completely. A nominal $5\%$ test has an actual size between $0.0416$ and $0.0514$ for every law and every sample size in the table, including thirty observations of a $t_{2.65}$ whose kurtosis is infinite. At $n=6{,}300$ — the sample this course works with — the four laws print $0.0485$, $0.0514$, $0.0481$, $0.0491$ against a nominal $0.0500$. Normality of the *data* is irrelevant to a $5\%$ test on the *mean*, exactly as claimed, and the reason is on the previous page: the skewness that governs the leading error term has been divided by $\sqrt{6300}\approx79$.

The last numeric column is the same theorem answering a different question, and it is wrong everywhere. At the $0.1\%$ level the $t_{2.65}$ over-rejects by $274\%$ at $n=30$, by $149\%$ at $n=252$, and still by $68\%$ at $n=6{,}300$ — a quarter century of daily data and the extreme quantile is not converged. The lognormal errs in the opposite direction and worse: $0.00000$ at $n=30$ against a nominal $0.001$, meaning the event *never happened* in four hundred thousand trials, and $0.00040$ at $n=6{,}300$, still two and a half times too small. Skewed and heavy-tailed laws do not merely converge slowly at the extremes; they converge slowly in opposite directions, so there is no safe adjustment.

The final column is what theory can certify, and it is a short list. Berry–Esseen bounds $\sup_x\lvert F_n(x)-\Phi(x)\rvert$ by $0.4748\,\rho/(\sigma^{3}\sqrt n)$ where $\rho=\mathbb{E}\lvert X-\mu\rvert^{3}$, and at $n=30$ that bound is $1.0459$ for the lognormal — larger than one, and therefore true but vacuous, since every CDF pair is within $1$ of each other. For the $t_{2.65}$ the entry reads *none exists*: a $t$ has moments only below its degrees of freedom, so with $\nu=2.65<3$ the third absolute moment is infinite and the bound has nothing to evaluate. The published fit of $\nu\approx2.6$ therefore leaves daily returns in the regime where the theorem holds and no finite-sample guarantee accompanies it, which is precisely what [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) means by saying that "df ≈ 2.6 strains the CLT machinery beneath the bootstrap."

!!! warning "The sample size that makes a mean test valid is not the sample size that makes a tail estimate valid, and only the first one is ever checked"
    Both columns above are the same estimator on the same data under the same theorem. A backtest that reports a t-statistic and a $99.9\%$ VaR is relying on the CLT twice, at two levels whose convergence rates differ by a factor of thousands in sample size, and reporting one significance discussion for both. The rule that follows is narrow and cheap: the CLT licenses inference about the centre of a distribution at the sample sizes finance has, and licenses nothing about the extreme quantiles at any sample size finance has. Tail quantities need a method built for tails — the peaks-over-threshold fit of [Extreme Value Theory](../part-18-quant-finance-applications/13-extreme-value-theory.md) — rather than a normal approximation extrapolated past where it was ever checked.

## Every Multivariate Convergence Claim Is the Scalar One Read Along a Direction

[Part VI](../part-06-multivariate-probability/index.md) builds mean vectors and covariance matrices and defers "every convergence claim" to here. The deferral costs almost nothing, because the multivariate central limit theorem is the scalar one applied one direction at a time.

For iid random vectors $X_1,X_2,\dots$ in $\mathbb{R}^{d}$ with mean $\mu$ and finite covariance matrix $\Sigma$,

$$\sqrt{n}\,\big(\bar X_n-\mu\big)\ \Longrightarrow\ \mathcal{N}(0,\Sigma),$$

the multivariate normal of [Multivariate Gaussian Distribution](../part-06-multivariate-probability/05-multivariate-gaussian.md).

??? note "Proof that convergence along every direction is convergence of the vector, and what that gives for free"
    The Cramér–Wold device states that $X_n\Longrightarrow X$ in $\mathbb{R}^{d}$ if and only if $a^{\top}X_n\Longrightarrow a^{\top}X$ in $\mathbb{R}$ for every fixed $a\in\mathbb{R}^{d}$. One direction is immediate from the continuous mapping theorem. The other is a one-line consequence of the transform representation: the characteristic function of the vector, evaluated at $a$, is

    $$\varphi_{X_n}(a)=\mathbb{E}\big[e^{ia^{\top}X_n}\big]=\varphi_{a^{\top}X_n}(1),$$

    so if every projection converges in distribution then every projection's characteristic function converges at the argument $1$, which is exactly the statement that $\varphi_{X_n}(a)\to\varphi_X(a)$ for every $a$. Lévy's continuity theorem then delivers convergence of the vector.

    Applied to the multivariate CLT the device does all the work. Fix $a$; the scalars $a^{\top}X_i$ are iid with mean $a^{\top}\mu$ and variance $a^{\top}\Sigma a$ — the sandwich formula of [Linear Transformations](../part-06-multivariate-probability/04-linear-transformations.md), and finite because $\Sigma$ is. The scalar theorem gives $\sqrt n\,a^{\top}(\bar X_n-\mu)\Longrightarrow\mathcal{N}(0,a^{\top}\Sigma a)$, which is precisely the projection of $\mathcal{N}(0,\Sigma)$ onto $a$ by the linear-combination definition of the multivariate normal. Every direction converges to the right thing, so the vector does.

    The load-bearing hypothesis is the same finite variance, now required in every direction at once: $a^{\top}\Sigma a<\infty$ for all $a$. The device's practical warning is the converse of its convenience. Because the conclusion is assembled from directions, its **failures are also directional**: one coordinate with an infinite or merely enormous third moment ruins the projections that load on it and leaves the rest untouched. There is no such thing as a portfolio being "approximately normal"; there are only particular weight vectors that are and particular weight vectors that are not, and a covariance matrix cannot distinguish them because it is a second-moment object and the rate is governed by the third.

```python
import numpy as np

rng = np.random.default_rng(7333)
z01, reps = 2.3263478740408408, 200_000
sd = np.array([1.0] * 8 + [0.5])                               # the ninth sleeve is the quiet one
w = {"equal weight": np.full(9, 1 / 9),
     "minimum variance": (1 / sd ** 2) / (1 / sd ** 2).sum(),
     "the quiet sleeve alone": np.eye(9)[8]}
print(f"  eight normal sleeves and one skewed heavy-tailed sleeve at half the volatility")
print("                 weights   w on sleeve 9   its share of var   port sd     n=252     n=6300")
for name, wt in w.items():
    v = (wt * sd) ** 2
    s = np.sqrt(v.sum())
    out = []
    for n in (252, 6_300):
        g = rng.standard_normal(reps) * np.sqrt(v[:8].sum() * n)     # the eight normals, exactly
        x = -(np.exp(rng.standard_normal((reps, n))) - np.exp(0.5)) / np.sqrt(np.e * (np.e - 1))
        out.append(np.mean((g + wt[8] * sd[8] * x.sum(axis=1)) / (s * np.sqrt(n)) < -z01))
    print(f"  {name:>23s} {wt[8]:15.4f} {v[8] / v.sum():18.4f} {s:9.4f}"
          f" {out[0]:9.4f} {out[1]:10.4f}")
print(f"  nominal size is 0.0100 in both columns")
# =>   eight normal sleeves and one skewed heavy-tailed sleeve at half the volatility
#                     weights   w on sleeve 9   its share of var   port sd     n=252     n=6300
#                 equal weight          0.1111             0.0303    0.3191    0.0098     0.0097
#             minimum variance          0.3333             0.3333    0.2887    0.0119     0.0102
#       the quiet sleeve alone          1.0000             1.0000    0.5000    0.0172     0.0112
#      nominal size is 0.0100 in both columns
```

Eight of the nine sleeves are normal and the ninth is skewed and heavy-tailed at half the volatility of the others — the profile of a short-volatility or carry sleeve, which is quiet until it is not. Every one of the nine has a finite variance, so the multivariate theorem applies in full and all three directions converge. The question is how fast, and the answer differs by direction.

Along the equal-weight direction the bad sleeve contributes $3.03\%$ of portfolio variance and the nominal $1\%$ test has an actual size of $0.0098$ at $n=252$: converged, for practical purposes, in a single year. Along the sleeve itself the actual size is $0.0172$, seventy-two percent too large after a year and still twelve percent too large after twenty-five. Same theorem, same data, same sample size, two directions, and one of them is usable while the other is not.

The middle row is the reason this matters. A minimum-variance optimizer sees only the covariance matrix, and the covariance matrix says the quiet sleeve is the good one — so it raises the weight from $0.1111$ to $0.3333$ and cuts portfolio standard deviation from $0.3191$ to $0.2887$, a genuine improvement in the objective it was given. It simultaneously raises the bad sleeve's share of variance from $3\%$ to $33\%$ and the test's actual size from $0.0098$ to $0.0119$. **The optimizer is selecting, blind, toward the direction along which its own inputs are least trustworthy**, and no amount of care with $\Sigma$ can detect it, because the quantity that governs the rate is a third moment and $\Sigma$ contains only second moments.

## Finite Variance Is a Claim About the Market, Not About the Method

The entry fee is one number and it is not observable. Everything on this page holds if $\sigma^{2}<\infty$ and none of it holds otherwise, and no finite sample can settle which case it is in, because the question is about the behaviour of the tail beyond the largest observation ever recorded.

What the data does offer is fits, and this course has published three of them on the same series. [Returns and Their Distributions](../../part-03-statistics/02-returns-and-distributions.md) fits a Student's $t$ with $\nu=2.65$ — finite variance, infinite kurtosis, entry fee paid with nothing to spare, and no Berry–Esseen bound to certify anything. It separately fits a stable law with $\alpha=1.53$, which "asserts infinite variance" and puts the series outside this theorem altogether, with a limit that is stable rather than normal and a scaling of $n^{1/1.53}$ rather than $n^{1/2}$. And [Drawdowns, Tail Risk, and Stress Testing](../../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md) fits an extreme-value shape parameter of $\xi=+0.327$, so "moments exist only up to order $1/0.327=3.1$" — variance finite, third moment finite by a margin of one-tenth, fourth moment not. Three estimates, three different verdicts about whether the theorem on this page applies and at what rate, from one dataset that everybody has.

That disagreement is the honest state of the question and it will not be resolved by more data, so the practical response is to stop treating the CLT as a background fact and start treating it as an assumption with a cost. Two habits do most of the work. Report the sample size *and the level* together, because the fourth section shows they are not separable — "significant at $5\%$ on six thousand observations" is a defensible sentence and "the $99.9\%$ VaR from the same fit" is not. And when a result depends on the normal approximation in a way that matters, check it the way the fourth section did: simulate from a law with the tail you actually fitted, run the same procedure, and read the actual size against the nominal one. That takes twenty lines and it is the only thing on this page that produces a number rather than a limit.
