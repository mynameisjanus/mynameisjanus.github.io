# Chi-Square Distribution

This is the law of a sum of squares, which makes it the law of every variance estimate ever computed. A volatility is a square root of an average of squares, so its sampling distribution is a chi-square wearing a square root — and that is what turns "the annualised volatility is $19.5\%$" from a number into a number with an interval attached.

This page covers the definition as a sum of squared standard normals, the identification with a gamma, the two moments, the reason a sample variance has $n-1$ degrees of freedom rather than $n$, and the exact confidence interval a volatility carries. It does not build the gamma family, which is [Gamma Distribution](13-gamma-distribution.md); it does not cover ratios of two of these, which are [F Distribution](17-f-distribution.md); it does not cover the normal-over-root-chi-square construction, which is [Student's t Distribution](16-students-t-distribution.md); and it does not develop testing procedure, which is [Part XII](../part-12-hypothesis-testing/index.md).

The trading stake is a pair of numbers this book has already published. [Expected Value](../part-04-expectation-and-moments/01-expected-value.md) reports that annualised volatility is pinned to within $4.5\%$ of itself on one year of daily data and $0.9\%$ on twenty-five years, and uses the contrast with the mean's $260\%$ and $52\%$ to argue that a quantitative business is built around estimating risk and hoping about return. Those two volatility figures come from this distribution, and the last section derives them exactly rather than asymptotically.

## A Sum of Squared Normals

If $Z_1,\ldots,Z_k$ are independent standard normals, then

$$Q=Z_1^{2}+\cdots+Z_k^{2}\ \sim\ \chi^{2}_{k},$$

with $k$ the degrees of freedom. The support is the positive half-line because squares are non-negative, and the family is additive in $k$ for the obvious reason: concatenating $j$ squares with $k$ more gives $j+k$ squares.

The same sum written in $n$ dimensions is the squared Mahalanobis distance of [Multivariate Gaussian Distribution](../part-06-multivariate-probability/05-multivariate-gaussian.md), where standardizing a correlated vector by $\Sigma^{-1/2}$ produces exactly $n$ independent squared normals. That is where every multivariate outlier screen gets its reference distribution, and the striking part is what the reference does not depend on: not the volatilities, not the correlations, only $n$.

??? note "Proof that the square of a standard normal is Gamma(1/2, 2), and hence that the chi-square is a gamma"
    Let $Y=Z^{2}$ with $Z$ standard normal. The map is two-to-one, so for $y>0$ the distribution function is

    $$F_Y(y)=\mathbf{P}(-\sqrt{y}\le Z\le\sqrt{y})=2\Phi(\sqrt{y})-1,$$

    and differentiating with the chain rule gives $f_Y(y)=\phi(\sqrt{y})\,y^{-1/2}$, that is

    $$f_Y(y)=\frac{1}{\sqrt{2\pi}}y^{-1/2}e^{-y/2}.$$

    Compare with the gamma density $x^{a-1}e^{-x/\theta}/(\Gamma(a)\theta^{a})$ at $a=1/2$ and $\theta=2$: the kernel is $y^{-1/2}e^{-y/2}$, matching exactly, and the constant is $1/(\Gamma(1/2)\sqrt{2})=1/\sqrt{2\pi}$ because $\Gamma(1/2)=\sqrt{\pi}$. So $Z^{2}\sim\mathrm{Gamma}(1/2,2)$.

    The additivity of [Gamma Distribution](13-gamma-distribution.md) in the shape now finishes it: summing $k$ independent copies gives $\mathrm{Gamma}(k/2,2)$, which is the definition of $\chi^{2}_{k}$. Two things follow that would be laborious to prove directly. Every chi-square moment is a gamma moment, so nothing needs separate derivation; and $\chi^{2}_{2}$ is $\mathrm{Gamma}(1,2)$, an exponential with mean $2$ — which is why the exponential turns up in the [Exponential Distribution](10-exponential-distribution.md) family tree as a chi-square with two degrees of freedom, a connection with no evident geometric meaning that falls straight out of the algebra.

## Mean k, Variance 2k

Reading the gamma moments at $a=k/2$, $\theta=2$:

$$\mathbb{E}[Q]=k,\qquad \mathrm{var}(Q)=2k,\qquad \text{skewness}=\sqrt{8/k}.$$

The mean is immediate without the gamma: $\mathbb{E}[Z^{2}]=1$ for each term, and linearity does the rest with no independence needed. The variance does need independence, and $\mathrm{var}(Z^{2})=\mathbb{E}[Z^{4}]-1=3-1=2$ uses the fourth moment of a normal from [The Gaussian Distribution](11-gaussian-distribution.md) — the same $3\sigma^{4}$ that gives excess kurtosis its $-3$.

The skewness $\sqrt{8/k}$ decays like the gamma's $2/\sqrt{a}$, so a chi-square with many degrees of freedom is approximately normal and one with few is badly skewed. That asymmetry is the reason the last section's exact interval is worth having: at large $k$ the symmetric approximation is fine, and at small $k$ it is not.

```python
import numpy as np
from scipy.stats import chi2, gamma

rng = np.random.default_rng(103)
print("  a sum of k squared standard normals, three ways")
print("      k       sample mean    var      chi2 says     gamma(k/2, 2) says     skew")
for k in (1, 2, 5, 30, 252):
    q = (rng.standard_normal((400_000, k)) ** 2).sum(axis=1)
    print(f"    {k:3d} {q.mean():14.4f} {q.var():9.2f}"
          f"   {chi2.mean(k):6.1f}, {chi2.var(k):7.1f}"
          f"   {gamma.mean(k / 2, scale=2):6.1f}, {gamma.var(k / 2, scale=2):7.1f}"
          f" {((q - q.mean()) ** 3).mean() / q.std() ** 3:9.4f}   exact {np.sqrt(8 / k):.4f}")
# =>   a sum of k squared standard normals, three ways
#          k       sample mean    var      chi2 says     gamma(k/2, 2) says     skew
#          1         1.0001      1.99      1.0,     2.0      1.0,     2.0    2.7993   exact 2.8284
#          2         1.9988      3.99      2.0,     4.0      2.0,     4.0    2.0130   exact 2.0000
#          5         4.9955      9.94      5.0,    10.0      5.0,    10.0    1.2531   exact 1.2649
#         30        30.0054     60.01     30.0,    60.0     30.0,    60.0    0.5157   exact 0.5164
#        252       251.9984    506.19    252.0,   504.0    252.0,   504.0    0.1777   exact 0.1782
```

The three descriptions agree in every row, which is the proof above printed. The skew column decays as $\sqrt{8/k}$: at one degree of freedom the law is violently lopsided, and by $252$ — one year of daily observations — it is down to $0.18$ and close enough to normal that the symmetric approximation starts to be defensible.

## Why One Degree of Freedom Goes Missing

The result actually used in practice is not about known-mean squares but about a sample variance, and it carries $n-1$ rather than $n$:

$$\frac{(n-1)s^{2}}{\sigma^{2}}\sim\chi^{2}_{n-1},\qquad s^{2}=\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\bar X)^{2}.$$

??? note "Proof that the sample variance carries n-1 degrees of freedom, and where the missing one went"
    Decompose each deviation around the true mean into a part explained by the sample mean and a part left over. Writing $\mu$ for the true mean,

    $$\sum_{i}(X_i-\mu)^{2}=\sum_{i}(X_i-\bar X)^{2}+n(\bar X-\mu)^{2}.$$

    The cross term vanishes because $\sum_i(X_i-\bar X)=0$ identically — that identity is the constraint, and it is the whole story. Dividing by $\sigma^{2}$, the left side is $\chi^{2}_{n}$ and the last term is $\big(\sqrt{n}(\bar X-\mu)/\sigma\big)^{2}$, which is $\chi^{2}_{1}$. So a $\chi^{2}_{n}$ has been split into the sample-variance piece plus a $\chi^{2}_{1}$.

    For the split to give $\chi^{2}_{n-1}$ the two pieces must be independent, and they are: $\bar X$ and the vector of deviations $(X_i-\bar X)$ are uncorrelated, and under normality uncorrelated implies independent. Geometrically, the deviation vector lies in the $(n-1)$-dimensional subspace orthogonal to the all-ones direction, and $\bar X$ lives entirely along that direction — so the sum of squares has been split along orthogonal axes, and the degrees of freedom split with it.

    The orthogonality is the same object as in [Conditional Expectation](../part-04-expectation-and-moments/06-conditional-expectation.md), where the residual of a best predictor is orthogonal to everything the predictor could use. Here the "predictor" is the sample mean, and the missing degree of freedom is the direction it consumed. Normality is genuinely required for the independence, not merely convenient: on a non-normal law the deviations and the mean are uncorrelated but dependent, and the chi-square result fails even though the $-1$ in the denominator still delivers an unbiased $s^{2}$.

## The Confidence Interval a Volatility Actually Has

Inverting the pivot gives an exact interval for $\sigma$ with no asymptotics anywhere:

$$\mathbf{P}\left(\sqrt{\frac{(n-1)s^{2}}{\chi^{2}_{n-1,\,1-\alpha/2}}}\ \le\ \sigma\ \le\ \sqrt{\frac{(n-1)s^{2}}{\chi^{2}_{n-1,\,\alpha/2}}}\right)=1-\alpha.$$

```python
import numpy as np
from scipy.stats import chi2

sd_a = 0.195                                                   # published annualized volatility
print(f"  exact interval on an annualized volatility of {sd_a}, from daily data")
print("     years      n      exact 68% interval      +/- %    asymptotic 1/sqrt(2n)")
for yrs in (1, 5, 25, 100):
    n = 252 * yrs
    lo, hi = np.sqrt((n - 1) / chi2.ppf([0.8413, 0.1587], n - 1)) * sd_a
    print(f"    {yrs:6d} {n:6d}   [{lo:.5f}, {hi:.5f}] {100 * (hi - lo) / 2 / sd_a:8.2f}"
          f" {100 / np.sqrt(2 * n):20.2f}")
print("  and the mean, for contrast, at the same horizons")
for yrs in (1, 5, 25):
    print(f"    {yrs:6d}y   mean 0.075 +/- {sd_a / np.sqrt(yrs):.3f}"
          f"  ({100 * sd_a / np.sqrt(yrs) / 0.075:6.1f}% of itself)")
# =>   exact interval on an annualized volatility of 0.195, from daily data
#         years      n      exact 68% interval      +/- %    asymptotic 1/sqrt(2n)
#             1    252   [0.18685, 0.20432]     4.48                 4.45
#             5   1260   [0.19123, 0.19900]     1.99                 1.99
#            25   6300   [0.19329, 0.19676]     0.89                 0.89
#           100  25200   [0.19414, 0.19587]     0.45                 0.45
#      and the mean, for contrast, at the same horizons
#             1y   mean 0.075 +/- 0.195  ( 260.0% of itself)
#             5y   mean 0.075 +/- 0.087  ( 116.3% of itself)
#            25y   mean 0.075 +/- 0.039  (  52.0% of itself)
```

The exact interval reproduces the published figures: $4.48\%$ at one year and $0.89\%$ at twenty-five, against the asymptotic $4.45\%$ and $0.89\%$. The two columns agree to within three hundredths of a percentage point at one year and are indistinguishable beyond it, which is the previous block's skewness column cashing out: by $k=252$ the chi-square is symmetric enough that the simple formula is adequate. Below about thirty observations they diverge and only the exact interval is usable — and note that the exact interval is not symmetric about the point estimate, running from $-4.2\%$ to $+4.8\%$ at one year, because the underlying law is not.

The second half of the output is the contrast that motivated the whole exercise. On one year of data the volatility is known to within $4.5\%$ of itself and the mean to within $260\%$ of itself — the mean's interval comfortably contains zero and both signs — and even at twenty-five years the gap is a factor of nearly sixty. Both numbers come from the same file and the same daily observations.

!!! note "The volatility is measured well because squaring destroys the sign, and the mean is measured badly for the same reason"
    The asymmetry is not an accident of these particular series. A variance estimate averages $n$ squared deviations, each of which carries information about magnitude regardless of direction, so its relative error falls as $1/\sqrt{2n}$ with no dependence on the parameter's own size. A mean estimate averages $n$ signed deviations that largely cancel, and its relative error is $\sigma/(\mu\sqrt{n})$ — carrying a factor $\sigma/\mu$ that for daily equity returns is about $40$. That factor is why the two quantities in the output are separated by an order of magnitude that no amount of additional data will close, since both improve at the same $\sqrt{n}$ rate from starting points forty times apart.

So the chi-square is the distribution that makes risk measurable. Every claim in this book that a volatility, a covariance, or a tracking error is known to some stated precision is an application of the pivot above, and every one of them holds only under the normality that the independence step required — a hypothesis that [The Gaussian Distribution](11-gaussian-distribution.md) showed is wrong in the tails, which is where the squares are largest.

The practical rule follows from that caveat rather than from the theorem. Use the exact interval whenever $n$ is below a few hundred, because the skewness is real and the symmetric approximation understates the upper end. And treat the resulting interval as optimistic on real returns, because a heavy-tailed law puts more weight on the large squared deviations than a normal does, which inflates the true variance of $s^{2}$ above the $2\sigma^{4}/(n-1)$ the chi-square promises. The interval is exact for the model and approximate for the market, and the direction of the error is the familiar one.
