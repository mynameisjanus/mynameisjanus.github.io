# The Delta Method

An estimate has a standard error; the thing you actually report is usually a function of that estimate, and it needs one too. The delta method supplies it in one line — multiply by the slope of the function — and it is how a confidence interval for a variance becomes one for a volatility, and how a Sharpe ratio acquires the error bar that makes it interpretable. It is also a first-order Taylor expansion, which means it is exact for straight lines, good for gentle curves, and silently wrong wherever the slope is small enough that the second term should have been kept.

This page covers the delta method's statement and its proof as a Taylor expansion with an $o_p(1)$ remainder discarded, the transformation of a variance interval into a volatility interval, the derivation of the Sharpe ratio's standard error and the correction that heavy tails and skew demand, the second-order case where the first derivative vanishes and the limit stops being normal, and the multivariate form. It does not prove the central limit theorem that supplies its input, which is [The Central Limit Theorem](03-central-limit-theorem.md); it does not prove the two theorems its own proof uses, which are [Slutsky's Theorem](05-slutskys-theorem.md) and [Continuous Mapping Theorem](06-continuous-mapping-theorem.md); it does not construct confidence intervals in general, which is [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md); and it does not resample anything, which is [Bootstrap Confidence Intervals](../part-11-parameter-estimation/08-bootstrap-confidence-intervals.md).

The trading stake is the course's most quoted figure. [Hypothesis Testing and Multiple Testing](../../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) reports a strategy's performance over twenty-four years as "Sharpe 0.30 +/- 0.20, 95% CI [-0.09, 0.70]", and calls it "the most sobering print in Part III." The $\pm0.20$ came out of a formula, the formula came out of this page, and the third section derives it and reproduces the number to three decimals. It then shows the same formula understating the error by forty percent on a strategy with a realistic negative skew.

## It Is a Taylor Expansion With a Variance Taken

Suppose an estimator satisfies $\sqrt{n}\,(\hat\theta-\theta)\Longrightarrow\mathcal{N}(0,\sigma^{2})$, which is what [The Central Limit Theorem](03-central-limit-theorem.md) delivers for a sample mean and what maximum likelihood delivers more generally. Let $g$ be differentiable at $\theta$ with $g'(\theta)\neq0$. Then

$$\sqrt{n}\,\big(g(\hat\theta)-g(\theta)\big)\ \Longrightarrow\ \mathcal{N}\!\big(0,\ g'(\theta)^{2}\sigma^{2}\big),$$

which in the form [Calculus Essentials](../part-01-mathematical-foundations/06-calculus-essentials.md) states it is $\mathrm{var}\big(g(\hat\theta)\big)\approx\big(g'(\theta)\big)^{2}\sigma^{2}/n$. The slope of the transform scales the standard error and nothing else happens.

??? note "Proof that a smooth transform inherits the limiting normal, scaled by the slope"
    Taylor's theorem with the mean-value form of the remainder gives, for some $\tilde\theta$ between $\hat\theta$ and $\theta$,

    $$g(\hat\theta)-g(\theta)=g'(\theta)\,(\hat\theta-\theta)+\big(g'(\tilde\theta)-g'(\theta)\big)(\hat\theta-\theta).$$

    Multiply by $\sqrt n$. The first term is $g'(\theta)$ times $\sqrt n(\hat\theta-\theta)$, which converges in distribution to $\mathcal{N}(0,g'(\theta)^{2}\sigma^{2})$ — a fixed constant times a converging sequence, which is [Slutsky's Theorem](05-slutskys-theorem.md) in its simplest use.

    The second term is where the notation of [Mathematical Notation](../part-01-mathematical-foundations/03-mathematical-notation.md) earns its keep. Since $\sqrt n(\hat\theta-\theta)$ converges in distribution it is bounded in probability, $\hat\theta-\theta=O_p(n^{-1/2})$, and in particular $\hat\theta\xrightarrow{\ p\ }\theta$. Then $\tilde\theta$, trapped between them, also converges in probability to $\theta$, so if $g'$ is continuous at $\theta$ the factor $g'(\tilde\theta)-g'(\theta)$ is $o_p(1)$ by [Continuous Mapping Theorem](06-continuous-mapping-theorem.md). The whole second term is therefore

    $$\underbrace{\big(g'(\tilde\theta)-g'(\theta)\big)}_{o_p(1)}\cdot\underbrace{\sqrt n\,(\hat\theta-\theta)}_{O_p(1)}=o_p(1),$$

    because $o_p(1)\cdot O_p(1)=o_p(1)$. A remainder that vanishes in probability can be added to a sequence converging in distribution without changing the limit — Slutsky again — so the sum converges to the first term's limit and the proof is done. This is the argument [Mathematical Notation](../part-01-mathematical-foundations/03-mathematical-notation.md) describes when it says the point is usually that some remainder is $o_p(1)$ and can be discarded.

    The load-bearing hypothesis is $g'(\theta)\neq0$, and it is easy to miss because it appears only as a non-degeneracy condition. When the slope vanishes the first term is identically zero, the limit is a point mass, and the discarded remainder — which was negligible only *relative to* a first term that no longer exists — becomes the entire answer. The fourth section computes what it becomes. Differentiability at $\theta$ is the other hypothesis and it is genuinely weaker than it looks: $g$ may be wild elsewhere, since the argument only ever evaluates it in a shrinking neighbourhood of $\theta$.

## A Confidence Interval for a Variance Becomes One for a Volatility

The canonical use is the one every risk report performs. Variance is what estimators produce and what has clean sampling theory; volatility is what gets quoted. With $g(v)=\sqrt v$ and $g'(v)=1/(2\sqrt v)$, and with $\mathrm{var}(\hat v)\approx2\sigma^{4}/n$ for normal data,

$$\mathrm{sd}(\hat\sigma)\approx\frac{1}{2\sigma}\cdot\sigma^{2}\sqrt{\frac{2}{n}}=\frac{\sigma}{\sqrt{2n}},$$

so a volatility estimate carries a relative standard error of $1/\sqrt{2n}$ — about $4.5\%$ on a year of daily data, and about $15\%$ on a month of it. That single expression is why volatility is estimable and, as [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md) computes, the mean is not: the mean's relative error depends on the Sharpe ratio and the calendar, while volatility's depends only on the observation count.

```python
import numpy as np
from scipy.stats import chi2

rng = np.random.default_rng(7411)
sigma, reps = 0.20, 200_000
print(f"  95% intervals for a volatility of {sigma}, from the sample variance, {reps} samples")
print("        n   delta cover   delta width   exact cover   exact width   exact arm ratio")
for n in (5, 21, 63, 252, 1_260):
    v = sigma ** 2 * rng.chisquare(n - 1, reps) / (n - 1)      # the sampling law of s-squared
    s = np.sqrt(v)
    d = 1.959963984540054 * s / np.sqrt(2 * n)                 # delta-method standard error
    lo_d, hi_d = s - d, s + d
    lo_e = s * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1))
    hi_e = s * np.sqrt((n - 1) / chi2.ppf(0.025, n - 1))
    print(f"  {n:9d} {np.mean((lo_d < sigma) & (sigma < hi_d)):13.4f} {(hi_d - lo_d).mean():13.4f}"
          f" {np.mean((lo_e < sigma) & (sigma < hi_e)):13.4f} {(hi_e - lo_e).mean():13.4f}"
          f" {np.mean((hi_e - s) / (s - lo_e)):17.4f}")
# =>   95% intervals for a volatility of 0.2, from the sample variance, 200000 samples
#            n   delta cover   delta width   exact cover   exact width   exact arm ratio
#              5        0.8215        0.2328        0.9505        0.4272            4.6738
#             21        0.9186        0.1194        0.9496        0.1341            1.8901
#             63        0.9400        0.0696        0.9502        0.0722            1.4286
#            252        0.9477        0.0349        0.9501        0.0352            1.1930
#           1260        0.9489        0.0156        0.9493        0.0156            1.0819
```

The exact interval is built by transforming the *endpoints* of the chi-square interval for the variance, which is legitimate at every sample size because $\sqrt{\cdot}$ is increasing and therefore preserves the ordering that defines a quantile. It covers at $0.9505$, $0.9496$, $0.9502$, $0.9501$, $0.9493$ — nominal on every row, as it must.

The delta-method interval transforms the *estimate* and attaches a symmetric band. At a year of daily data it covers at $0.9477$ and its width of $0.0349$ is within one percent of the exact $0.0352$; at five years the two are indistinguishable. At twenty-one observations it covers at $0.9186$, and at five it covers at $0.8215$ — a nominal $95\%$ interval that misses one time in six.

The last column explains the failure and is the more useful diagnostic. It reports how much longer the exact interval's upper arm is than its lower one: $4.67$ at $n=5$, $1.89$ at $n=21$, and $1.08$ at $n=1{,}260$. The sampling distribution of a volatility estimate is right-skewed, badly so at small $n$, and a symmetric interval cannot represent that no matter what half-width it is given. The delta method has no mechanism for asymmetry, because it kept one term of a Taylor expansion and asymmetry is in the next one.

!!! note "Transform the endpoints, not the estimate, whenever the exact interval is available"
    The two constructions above cost the same to compute and differ only in the order of two operations. Transforming the endpoints inherits the exact interval's coverage for free, for any monotone $g$ — the same trick turns a variance interval into a volatility interval, a log-return interval into a return interval, and an odds interval into a probability interval, in each case with no approximation and no small-sample penalty. The delta method is for the cases where no exact interval exists, which is most of them, and where the transform is not monotone. Reaching for it when the endpoint transform is available is a habit worth breaking, and the $n=21$ row prices it at three and a half points of coverage.

## The Sharpe Ratio's Standard Error Is Two Applications of the Same Line

A Sharpe ratio is a smooth function of two estimated moments, so it needs the two-argument version. Let $m=\mathbb{E}[R]$ and $v=\mathrm{var}(R)$ with sample counterparts $\hat m,\hat v$, and let $g(m,v)=m/\sqrt v$ so $SR=g(m,v)$.

??? note "Derivation of the standard error of a Sharpe ratio, and the terms Lo's formula drops"
    For iid returns with central moments $\mu_3$ and $\mu_4$, the pair $(\hat m,\hat v)$ is jointly asymptotically normal by the multivariate central limit theorem of [The Central Limit Theorem](03-central-limit-theorem.md), with asymptotic covariance matrix

    $$\Sigma=\begin{pmatrix}v&\mu_3\\\mu_3&\mu_4-v^{2}\end{pmatrix}.$$

    The gradient of $g$ is $\nabla g=\big(v^{-1/2},\ -m/(2v^{3/2})\big)$, and the multivariate delta method of the fifth section gives the asymptotic variance as the sandwich $\nabla g^{\top}\Sigma\,\nabla g$, evaluated with $\gamma_3=\mu_3/v^{3/2}$ the skewness and $\gamma_4=\mu_4/v^{2}$ the raw kurtosis:

    $$n\cdot\mathrm{var}\big(\widehat{SR}\big)\ \longrightarrow\ 1+\frac{\gamma_4-1}{4}SR^{2}-\gamma_3\,SR.$$

    Writing $\gamma_4=3+\kappa$ for the *excess* kurtosis puts it in the form used below,

    $$\mathrm{SE}\big(\widehat{SR}\big)\approx\sqrt{\frac{1+\tfrac12SR^{2}+\tfrac14\kappa\,SR^{2}-\gamma_3\,SR}{n}},$$

    and setting $\kappa=\gamma_3=0$ — assuming the returns are normal — collapses it to Lo's $\sqrt{(1+SR^{2}/2)/n}$, the formula the course prints.

    The load-bearing hypothesis here is not normality but independence, and it is the one nobody checks. Every entry of $\Sigma$ above assumed iid draws; with autocorrelated returns the covariance matrix picks up cross-period terms and the whole expression changes by a factor that has nothing to do with $g$. [Slutsky's Theorem](05-slutskys-theorem.md) is where that correction is made and where its size is measured, and it is larger than every correction on this page.

```python
import numpy as np

rng = np.random.default_rng(7423)
reps, sl = 200_000, 0.5
c = np.exp(sl ** 2)
g3, g4 = -(c + 2) * np.sqrt(c - 1), c ** 4 + 2 * c ** 3 + 3 * c ** 2 - 6   # skew, excess kurtosis
print(f"  standard error of an annualized Sharpe ratio; the skewed law has"
      f" skew {g3:.2f}, excess kurtosis {g4:.2f}")
print("     periods/yr        n   true SR      Lo formula   normal draws   skewed draws   Mertens")
for per, n, sr in ((252, 6_158, 0.30), (12, 294, 0.30), (12, 294, 1.50)):
    d = sr / np.sqrt(per)                                      # Sharpe per period
    lo = np.sqrt((1 + d * d / 2) / n) * np.sqrt(per)
    me = np.sqrt((1 + d * d / 2 - g3 * d + g4 * d * d / 4) / n) * np.sqrt(per)
    out = []
    for skewed in (False, True):
        if skewed:
            x = -(np.exp(sl * rng.standard_normal((reps, n))) - np.exp(sl ** 2 / 2))
            x = x / np.sqrt(c * (c - 1))
        else:
            x = rng.standard_normal((reps, n))
        r = d + x
        out.append((np.sqrt(per) * r.mean(axis=1) / r.std(axis=1, ddof=1)).std())
    print(f"  {per:15d} {n:8d} {sr:9.2f} {lo:15.4f} {out[0]:14.4f} {out[1]:14.4f} {me:9.4f}")
# =>   standard error of an annualized Sharpe ratio; the skewed law has skew -1.75, excess kurtosis 5.90
#         periods/yr        n   true SR      Lo formula   normal draws   skewed draws   Mertens
#                  252     6158      0.30          0.2023         0.2025         0.2061    0.2057
#                   12      294      0.30          0.2024         0.2020         0.2204    0.2182
#                   12      294      1.50          0.2113         0.2118         0.2954    0.2947
```

The first row is the published print, rebuilt. At $6{,}158$ daily observations and a true annualized Sharpe of $0.30$, the formula returns $0.2023$ and the simulated standard deviation of the estimate across two hundred thousand histories is $0.2025$ — agreement to three decimals, and the source of the course's $\pm0.20$. The $95\%$ interval that follows, $0.30\pm1.96\times0.2023=[-0.097,0.697]$, is the published $[-0.09,0.70]$.

There is a second confirmation available that used no formula at all. [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) resamples the same returns ten thousand times and obtains $[-0.09,0.71]$, noting that "when a formula exists, the bootstrap reproduces it." Two methods with nothing in common — one a Taylor expansion around a limiting normal, the other a plug-in argument licensed by [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md) — agreeing to the second decimal is what a correct standard error looks like.

The bottom row is where the assumption bites. At a monthly frequency and a true Sharpe of $1.50$, on returns with skew $-1.75$ and excess kurtosis $5.90$ — a mild version of what a short-volatility or credit strategy actually produces — Lo's formula reports $0.2113$ and the truth is $0.2954$. **The error bar is forty percent too narrow**, so an interval that looks like $[1.09,1.91]$ is really $[0.92,2.08]$. The skew-corrected expression from the derivation gives $0.2947$, which recovers the truth to three decimals and shows the gap is entirely the two dropped terms rather than any failure of the method. Nearly all of it is the $-\gamma_3 SR$ term: negative skew and a positive Sharpe make the estimate *less* precise, and the effect scales with the Sharpe, so it is invisible on a daily-frequency $0.30$ and dominant on a monthly $1.50$.

## Where the Slope Is Zero the Formula Returns Zero and Means Nothing

The proof's non-degeneracy condition $g'(\theta)\neq0$ is the only hypothesis on this page with no diagnostic attached, because the formula does not fail at $g'(\theta)=0$. It succeeds, and returns zero.

??? note "Proof that when the first derivative vanishes the limit is chi-square rather than normal"
    Suppose $g'(\theta)=0$ and $g''(\theta)\neq0$. Taking the Taylor expansion one term further,

    $$g(\hat\theta)-g(\theta)=\tfrac12 g''(\theta)\,(\hat\theta-\theta)^{2}+o_p\!\big((\hat\theta-\theta)^{2}\big),$$

    and the natural scaling is no longer $\sqrt n$ but $n$. Multiplying through,

    $$n\,\big(g(\hat\theta)-g(\theta)\big)=\tfrac12 g''(\theta)\Big(\sqrt n\,(\hat\theta-\theta)\Big)^{2}+o_p(1)\ \Longrightarrow\ \tfrac12 g''(\theta)\,\sigma^{2}\chi^{2}_{1},$$

    where the last step is the continuous mapping theorem applied to the square, since $x\mapsto x^{2}$ is continuous and the square of a $\mathcal{N}(0,\sigma^2)$ is $\sigma^{2}$ times a chi-square with one degree of freedom, the distribution of [Chi-Square Distribution](../part-05-common-distributions/15-chi-square-distribution.md).

    Three things change at once and each is worth naming. The **rate** improves from $n^{-1/2}$ to $n^{-1}$, so the estimate is more precise than the delta method would have predicted had it predicted anything. The **shape** stops being normal and becomes chi-square, which is bounded on one side and has skewness $\sqrt8\approx2.828$. And the **sign** is determined: $g(\hat\theta)-g(\theta)$ has the sign of $g''(\theta)$ with probability one, so the estimator is biased in a fixed direction and no symmetric interval can be right. Any confidence interval, test, or standard error built on the first-order formula is not merely imprecise here; it is describing a different distribution.

```python
import numpy as np

rng = np.random.default_rng(7437)
sigma, reps = 1.0, 400_000
print(f"  the delta method for g(theta) = theta^2, near and at the point where g'(theta) = 0")
print("     true theta        n   g'(theta)   delta SE   simulated sd   ratio   simulated skew")
for th in (0.50, 0.05, 0.00):
    for n in (252, 6_300):
        t = th + sigma / np.sqrt(n) * rng.standard_normal(reps)
        g = t * t
        d = abs(2 * th) * sigma / np.sqrt(n)
        c = g - g.mean()
        sk = (c ** 3).mean() / g.std() ** 3
        print(f"  {th:14.2f} {n:8d} {2 * th:11.2f} {d:10.6f} {g.std():14.6f}"
              f" {g.std() / d if d else float('inf'):7.2f} {sk:16.4f}")
# =>   the delta method for g(theta) = theta^2, near and at the point where g'(theta) = 0
#         true theta        n   g'(theta)   delta SE   simulated sd   ratio   simulated skew
#                0.50      252        1.00   0.062994       0.063248    1.00           0.3758
#                0.50     6300        1.00   0.012599       0.012588    1.00           0.0690
#                0.05      252        0.10   0.006299       0.008430    1.34           2.4054
#                0.05     6300        0.10   0.001260       0.001281    1.02           0.7354
#                0.00      252        0.00   0.000000       0.005603     inf           2.8397
#                0.00     6300        0.00   0.000000       0.000225     inf           2.7903
```

The top two rows are the method working. At $\theta=0.5$ the slope is $1.00$, the delta-method standard error is $0.062994$ against a simulated $0.063248$, and the ratio is $1.00$ at both sample sizes. Nothing on this page is in doubt away from the degenerate point.

The bottom two rows are the failure in its pure form. At $\theta=0$ the formula returns a standard error of exactly $0.000000$ — not small, not approximate, but zero to every printed digit — while the estimator's actual standard deviation is $0.005603$ at $n=252$. The simulated skewness is $2.8397$ and $2.7903$ against the chi-square's $\sqrt8=2.8284$, confirming the shape from the proof. And the actual standard deviation falls from $0.005603$ to $0.000225$, a factor of $24.9$ for a $25$-fold increase in $n$: the $1/n$ rate, not $1/\sqrt n$. The delta method is not wrong about the magnitude in a way that more data fixes. It is answering a question about a normal distribution that is not there.

The middle rows are the part that matters in practice, because exact zeros do not occur in data. At $\theta=0.05$ — a slope of $0.10$, small but not zero — the formula understates the standard error by $34\%$ at $n=252$ and by only $2\%$ at $n=6{,}300$. The degeneracy is not a point; it is a neighbourhood whose radius shrinks with $\sqrt n$, and whether a given $\theta$ is inside it depends on the sample size. The condition to check is not $g'(\theta)=0$ but $\lvert g'(\theta)\rvert\gg\lvert g''(\theta)\rvert\sigma/\sqrt n$, which at $\theta=0.05$, $n=252$ reads $0.10$ against $0.126$ and correctly predicts trouble.

!!! warning "Any statistic evaluated at an optimum, a crossing, or a symmetry point has a vanishing first derivative by construction, and the delta method will report a standard error of zero for it"
    The degenerate case is not exotic; it is where optimizers put things. The variance of an estimate whose true mean is zero, a hedge ratio at the point where two exposures exactly offset, a parameter at the boundary of its identified region, the height of a fitted curve at its peak, the value of a likelihood at its maximum — all of these have $g'=0$ as a defining property rather than an accident. In each case the reported standard error will be zero or near it, which reads as extraordinary precision, and the correct statement is that the distribution is chi-square and one-sided. The cheap check is to perturb: recompute the standard error at $\theta$ and at $\theta\pm\sigma/\sqrt n$, and if the three answers differ by more than a few percent the expansion point is inside the degenerate neighbourhood and the first-order formula should not be used.

## In More Than One Dimension the Slope Is a Gradient and the Sandwich Returns

The Sharpe derivation already used the general form, which is the scalar statement with the derivative replaced by a gradient. If $\sqrt n(\hat\theta-\theta)\Longrightarrow\mathcal{N}(0,\Sigma)$ for $\hat\theta\in\mathbb{R}^{d}$ and $g:\mathbb{R}^{d}\to\mathbb{R}$ is differentiable at $\theta$ with $\nabla g(\theta)\neq0$, then

$$\sqrt n\,\big(g(\hat\theta)-g(\theta)\big)\ \Longrightarrow\ \mathcal{N}\!\big(0,\ \nabla g(\theta)^{\top}\Sigma\,\nabla g(\theta)\big).$$

The proof is unchanged: the same Taylor expansion, now multivariate, with the same $o_p(1)$ remainder discarded by the same argument. What is worth pausing on is that the asymptotic variance is a quadratic form in $\Sigma$ — the sandwich $a^{\top}\Sigma a$ of [Linear Transformations](../part-06-multivariate-probability/04-linear-transformations.md), with the gradient playing the role of the weight vector. Every risk number in this book is that expression, and the delta method is the observation that a nonlinear function is, asymptotically, a linear one with weights $\nabla g$.

Two consequences follow immediately. The off-diagonal entries of $\Sigma$ matter and are routinely omitted: the Sharpe's derivation needed $\mathrm{cov}(\hat m,\hat v)=\mu_3/n$, and dropping it is exactly the $-\gamma_3 SR$ term whose absence cost forty percent of the error bar. And a vector-valued $g$ replaces the gradient with a Jacobian $J$ and the scalar sandwich with $J\Sigma J^{\top}$, which is how a covariance matrix for several derived quantities — a set of factor loadings, a yield curve's transformed parameters — comes out of the covariance matrix for the raw ones.

## A Standard Error Inherits Every Assumption Its Estimate Made

The delta method is a piece of bookkeeping, and it is honest about it: it converts a standard error you already have into one for a transformed quantity, contributing nothing except a derivative. Everything questionable about the output was already questionable about the input, and the transformation neither introduces nor repairs an assumption.

That makes the reading order clear whenever one of these formulas is quoted. First ask whether the input's limiting normal is real — the iid hypothesis behind $\Sigma$, the finite variance behind the central limit theorem, the sample size against the level being tested, all of which [The Central Limit Theorem](03-central-limit-theorem.md) prices. Then ask whether the transform's slope is comfortably away from zero at the estimate, which the fourth section reduces to comparing $\lvert g'\rvert$ against $\lvert g''\rvert\sigma/\sqrt n$. Then, and only then, is the number meaningful, and it is meaningful as a first-order approximation whose next term the fourth column of the second table measures at $4.67$ to $1.08$ across sample sizes.

The published $0.30\pm0.20$ survives all three questions, which is why the course keeps returning to it. What it does not survive is dependence — every entry of the covariance matrix in the Sharpe derivation assumed independent draws, and the next page shows what happens to the same statistic when that assumption is dropped instead.
