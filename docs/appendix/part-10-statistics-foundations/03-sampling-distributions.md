# Sampling Distributions

The previous page produced numbers and read them as descriptions. Every one of them is also a draw. Recompute a sample mean on a second twenty-five years and it is a different number, and the object governing how different is the **sampling distribution** of the statistic — the law of $\hat\theta$ induced by the law of the data. [Mathematical Notation](../part-01-mathematical-foundations/03-mathematical-notation.md) puts the whole page in one sentence: $\bar X=\frac1n\sum_i X_i$ "is itself a random variable — a fact that is easy to nod past". It is easy to nod past because a sampling distribution is never observed, only derived, and the derivations available are exact for one family, robust for one statistic, and quietly wrong for the number a research process actually reports.

This page covers a statistic as a random variable and the standard error as the summary of its law, the structural reason a drift's precision is governed by calendar span while a volatility's is governed by observation count, the single orthogonal rotation that delivers the $\chi^2$ law and the independence of $\bar X$ and $s^2$, which of the derived laws survives a heavy tail and which does not, and the sampling distribution of a maximum as the one nobody computes. It proves no limit theorem and takes the central limit theorem as given, which is [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md); it derives no density for the named laws, which is [Part V](../part-05-common-distributions/index.md); it linearizes no nonlinear statistic, which is [The Delta Method](../part-07-asymptotic-theory/04-delta-method.md); it constructs no interval as a procedure, which is [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md); it builds no test and names no critical region, which is [Part XII](../part-12-hypothesis-testing/index.md); it simulates no sampling distribution where the formula runs out, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it corrects nothing for the size of a search, which is [Part XV](../part-15-multiple-testing/index.md); and it assumes throughout that the sample is what the first page said it usually is not.

The trading stake is Part III's most-repeated rule and the arithmetic underneath it. [Probability and Random Variables](../../part-03-statistics/01-probability-and-random-variables.md) states that "an estimate without a standard error is a rumor" and then applies it to the number the industry is built on: after twenty-five years of daily data the equity premium estimate is $7.5\%$ with a standard error of $3.9\%$, pinned as `ann mean 0.075 +/- 0.039`, a $t$-statistic of about $1.9$, and the lesson's verdict is that "**Vol is estimable; the mean barely is.**" The second section shows that asymmetry is structural rather than unlucky — it reproduces the $3.9\%$ four times over at observation counts spanning a factor of a hundred and thirty — and the fifth section derives the $\pm0.20$ that [Hypothesis Testing and Multiple Testing](../../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) attaches to a Sharpe of $0.30$.

## A Statistic Is a Random Variable and Its Law Is What Inference Consumes

A **statistic** is any function of the data alone. Because the data is random, the statistic is random, and its distribution across repetitions of the sampling process is its **sampling distribution**. The **standard error** is the standard deviation of that distribution — not of the data.

Three quantities are routinely conflated and separating them is most of the battle. There is $\sigma$, the spread of the population, a property of the world. There is $s$, the spread of the sample, an estimate of $\sigma$ and a property of the data. And there is $s/\sqrt n$, the estimated spread of $\bar X$, which is a property of neither — it describes a distribution over samples that were never drawn. Only the first two would appear in a histogram of the data, which is why the third is the one that gets dropped.

What makes the subject genuinely hard is that the sampling distribution is a statement about counterfactual samples. Nothing in the observed data is a direct observation of it. Every route to it is therefore an argument rather than a measurement: derive it from an assumed family, approximate it by a limit theorem, or manufacture replicates by resampling. The first is this page, the second is [Part VII](../part-07-asymptotic-theory/index.md), and the third is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md).

## A Drift's Precision Is Fixed by Calendar Span and a Volatility's by Observation Count

The course's asymmetry — vol estimable, mean barely — is usually explained by saying returns are noisy. That is true and it is not the explanation, because it does not say why more data helps one and not the other.

??? note "Proof that the variance of a drift estimate depends on the calendar span alone and that of a volatility estimate on the number of observations alone"
    Let $X_t=\mu t+\sigma W_t$ be observed on $[0,T]$ at $m$ equally spaced times $t_k=kT/m$. The increments $\Delta X_k=X_{t_k}-X_{t_{k-1}}$ are independent and $\mathcal{N}(\mu T/m,\ \sigma^2 T/m)$. The natural drift estimator averages them,

    $$\hat\mu=\frac{1}{T}\sum_{k=1}^{m}\Delta X_k=\frac{X_T-X_0}{T},$$

    because the sum telescopes. Its variance is $\mathrm{var}(\hat\mu)=\sigma^{2}T/T^{2}=\sigma^{2}/T$, in which $m$ does not appear at all: the estimator depends on the first and last observation and discards everything between them. The realized-variance estimator is

    $$\hat\sigma^{2}=\frac{1}{T}\sum_{k=1}^{m}(\Delta X_k)^{2},\qquad \mathrm{var}(\hat\sigma^{2})=\frac{2\sigma^{4}}{m},$$

    since each squared increment is $(\sigma^2T/m)$ times a $\chi^2_1$, whose variance is $2$. Its relative standard error is $\sqrt{2/m}$, which goes to zero as the grid refines with $T$ held fixed.

    The load-bearing distinction is between a boundary functional and an interior one. The drift is determined by the endpoints of the path, so refining the grid adds observations that carry no information about it; the volatility is determined by the path's roughness, which is exactly what refining the grid reveals. **Sampling a fixed history more finely is free information about the second moment and precisely zero information about the first**, which is why the equity premium's error bar is a fact about how long equities have existed rather than about the data vendor, and why no sampling frequency rescues it.

```python
import numpy as np

rng = np.random.default_rng(10031)
paths, mu, sig = 20_000, 0.075, 0.195                          # the course's measured SPY numbers

print(f"  {paths} histories of the same law, sampled at different frequencies and spans")
print("   years    per year         n    sd(mu_hat)    sigma/sqrt(T)    rel sd(sig_hat)    1/sqrt(2n)")
for years, per in ((25, 12), (25, 52), (25, 252), (25, 1_560), (100, 252)):
    n = years * per
    r = mu / per + (sig / np.sqrt(per)) * rng.standard_normal((paths, n))
    mh = per * r.mean(axis=1)
    sh = np.sqrt(per) * r.std(axis=1, ddof=1)
    print(f"  {years:7d} {per:11d} {n:9d} {mh.std(ddof=1):13.4f} {sig / np.sqrt(years):16.4f}"
          f" {sh.std(ddof=1) / sh.mean():18.4f} {1 / np.sqrt(2 * n):13.4f}")
# =>   20000 histories of the same law, sampled at different frequencies and spans
#       years    per year         n    sd(mu_hat)    sigma/sqrt(T)    rel sd(sig_hat)    1/sqrt(2n)
#           25          12       300        0.0391           0.0390             0.0407        0.0408
#           25          52      1300        0.0387           0.0390             0.0197        0.0196
#           25         252      6300        0.0388           0.0390             0.0089        0.0089
#           25        1560     39000        0.0393           0.0390             0.0036        0.0036
#          100         252     25200        0.0195           0.0195             0.0045        0.0045
```

The fourth column is the finding and it does not move. The standard deviation of the drift estimate reads $0.0391$, $0.0387$, $0.0388$ and $0.0393$ across the first four rows, which run from three hundred monthly observations to thirty-nine thousand ten-minute ones — a hundred-and-thirty-fold increase in data for no improvement whatever. The fifth column says why: $\sigma/\sqrt{T}$ is $0.0390$ in all four cases because $T$ is twenty-five years in all four cases. This is the course's pinned $\pm0.039$, reproduced four separate times from four different sampling schemes.

The sixth column is the other half of the asymmetry. The relative standard error of the volatility estimate falls $0.0407$, $0.0197$, $0.0089$, $0.0036$ over the same rows, tracking $1/\sqrt{2n}$ exactly, so the identical data that bought nothing for the mean bought an elevenfold improvement for the volatility. The two estimators are computed from the same numbers and consume them in completely different ways.

The last row is the only intervention that works, and it is not available. Holding the sampling frequency at daily and extending the span from twenty-five years to a hundred halves the drift's standard error, from $0.0388$ to $0.0195$ — a factor of two for four times the history. **The equity premium's $t$-statistic of $1.9$ is not a data-collection problem and cannot be fixed by a faster feed; buying a hundred years of history to halve the error bar is the only remedy, and it is a remedy nobody can purchase.**

## One Rotation Delivers the Chi-Square Law and an Independence That Exists Nowhere Else

The $t$ statistic divides a mean by an estimate of its own standard error, which puts a random quantity in the denominator. That it works at all — that the ratio has a distribution free of $\sigma$ — rests on a fact that is exclusive to the normal family and is usually asserted rather than shown.

??? note "Proof that $\bar X$ and $s^2$ are independent and $(n-1)s^2/\sigma^2$ is $\chi^2_{n-1}$, by a single rotation"
    Let $X\sim\mathcal{N}(\mu\mathbf 1,\sigma^2 I)$ in $\mathbb{R}^n$ and let $H$ be any orthogonal matrix whose first row is $(1/\sqrt n,\dots,1/\sqrt n)$; the Helmert matrix is the standard choice. Set $Y=HX$. An orthogonal transformation of a spherical normal is a spherical normal, so the coordinates $Y_1,\dots,Y_n$ are independent with common variance $\sigma^2$, and only $Y_1$ has a non-zero mean. By construction

    $$Y_1=\sqrt n\,\bar X,\qquad \sum_{k=2}^{n}Y_k^{2}=\|Y\|^{2}-Y_1^{2}=\|X\|^{2}-n\bar X^{2}=\sum_{i=1}^{n}(X_i-\bar X)^{2}=(n-1)s^{2},$$

    using $\|Y\|=\|X\|$. So $\bar X$ is a function of $Y_1$ alone, $s^2$ is a function of $Y_2,\dots,Y_n$ alone, and the two are independent because the coordinates are. The $n-1$ remaining coordinates are independent $\mathcal{N}(0,\sigma^2)$, so $(n-1)s^2/\sigma^2\sim\chi^{2}_{n-1}$, and

    $$\frac{\bar X-\mu}{s/\sqrt n}=\frac{(\bar X-\mu)/(\sigma/\sqrt n)}{\sqrt{\big[(n-1)s^{2}/\sigma^{2}\big]/(n-1)}}$$

    is a standard normal over the square root of an independent $\chi^2_{n-1}$ divided by its degrees of freedom, which is the definition of $t_{n-1}$. The $F$ law is the same construction with two independent $\chi^2$ terms.

    The load-bearing hypothesis is rotational invariance, and it is a property of the iid normal and of nothing else — Lukacs's theorem states that the independence of $\bar X$ and $s^2$ *characterizes* the normal family. In general $\mathrm{cov}(\bar X,s^{2})=\mu_3/n$, so the independence fails at first order in the skewness, and the course measures `skewness -0.20` on SPY. **The independence is not an approximation that improves with $n$; it is exactly true under normality, exactly false otherwise, and it is the single hypothesis every $t$ and every $F$ on a desk is resting on.**

## Only One of the Three Derived Laws Survives a Fat Tail

The previous proof produced three laws from one construction, which invites the assumption that they degrade together when the construction's hypothesis fails. They do not, and the gap between them is the widest in this part.

```python
import numpy as np
from scipy.stats import chi2, t as tdist

rng = np.random.default_rng(10033)
reps = 40_000                                                  # all three laws standardized to var 1

print("   law           n    t-test size    t CI cover    chi2 var CI cover    mean excess kurt")
for name, nu in (("normal", None), ("t(5)", 5.0), ("t(3.4)", 3.4)):
    for n in (60, 252, 1_260):
        if nu is None:
            x = rng.standard_normal((reps, n))
        else:
            x = tdist.rvs(nu, size=(reps, n), random_state=rng) / np.sqrt(nu / (nu - 2))
        m, s2 = x.mean(axis=1), x.var(axis=1, ddof=1)
        tstat, crit = m / np.sqrt(s2 / n), tdist.ppf(0.975, n - 1)
        lo = (n - 1) * s2 / chi2.ppf(0.975, n - 1)             # the textbook variance interval
        hi = (n - 1) * s2 / chi2.ppf(0.025, n - 1)
        z = x - m[:, None]
        k = (z ** 4).mean(axis=1) / ((z ** 2).mean(axis=1)) ** 2 - 3
        print(f"  {name:<10} {n:5d} {np.mean(np.abs(tstat) > crit):14.4f}"
              f" {np.mean(np.abs(tstat) <= crit):13.4f} {np.mean((lo <= 1) & (1 <= hi)):20.4f}"
              f" {k.mean():19.2f}")
# =>    law           n    t-test size    t CI cover    chi2 var CI cover    mean excess kurt
#      normal        60         0.0498        0.9502               0.9497               -0.10
#      normal       252         0.0492        0.9508               0.9493               -0.02
#      normal      1260         0.0515        0.9486               0.9491               -0.00
#      t(5)          60         0.0493        0.9506               0.7989                1.87
#      t(5)         252         0.0505        0.9496               0.7604                3.38
#      t(5)        1260         0.0488        0.9513               0.7333                4.50
#      t(3.4)        60         0.0483        0.9517               0.6409                3.89
#      t(3.4)       252         0.0486        0.9514               0.5461                9.25
#      t(3.4)      1260         0.0493        0.9507               0.4566               19.15
```

The $t$ columns are the good news and they are unambiguous. The size of a nominal $5\%$ test never leaves $[0.0483,0.0515]$ and the coverage of a nominal $95\%$ interval never leaves $[0.9486,0.9517]$ — across a normal, a $t(5)$ and a $t(3.4)$, at sixty, two hundred and fifty-two and one thousand two hundred and sixty observations. The central limit theorem is doing precisely what it advertises, and it protects the $t$ statistic because the numerator is an average of the observations.

The $\chi^2$ column is the failure. Under normality it reads $0.9497$, $0.9493$, $0.9491$, exactly as derived. Under a $t(5)$ it reads $0.7989$, $0.7604$, $0.7333$. Under a $t(3.4)$ — a tail no heavier than the one the course fits to SPY — it reads $0.6409$, $0.5461$ and $0.4566$. **A nominal $95\%$ confidence interval for a variance is a $46\%$ interval**, and it is printed by the same statistical package, on the same line, from the same three lines of the same proof as the $t$ interval sitting beside it at $0.9507$.

The direction is the part that should alarm. Coverage *deteriorates* as the sample grows — $0.6409$ to $0.4566$ as $n$ goes from sixty to one thousand two hundred and sixty — because the interval narrows like $1/\sqrt n$ around a centre whose own sampling distribution is governed by a fourth moment that does not exist. The last column shows the mechanism climbing in step, from $3.89$ to $19.15$, which is the ceiling effect of the previous page arriving here as a coverage failure.

!!! note "The $t$ interval and the $\chi^2$ interval come out of one proof and only one of them inherits the central limit theorem's protection, which is the difference between averaging the data and averaging its square"
    The asymmetry is structural rather than empirical. $\bar X$ is an average of the observations, so its limiting law requires a second moment and the central limit theorem supplies normality regardless of the shape of $F$ — the $t$ interval is asymptotically valid under nothing more than finite variance. The quantity $s^2$ is an average of *squares*, so its limiting law requires a **fourth** moment, and at the tail index the course fits that moment is infinite: there is no limiting normal for $s^2$ to be, and the $\chi^2$ interval is not a slightly-wrong approximation but an appeal to a hypothesis that has failed outright. The practical consequences are worth naming, because they are not confined to variance intervals: any $F$ test comparing two volatilities inherits the same defect, as does any $\chi^2$ test of a covariance structure, and the correct instruments are the ones that make no fourth-moment claim — a bootstrap interval, or a test built on ranks. The rule of thumb that survives is short: **inference about a mean is robust to the tail and inference about a spread is not.**

## The Sampling Distribution of the Number You Chose to Report Is Not the One You Derived

Everything so far derived the law of a statistic fixed in advance. Research does not work that way. A statistic is computed for fifty variants and the largest is reported, and the largest of fifty draws does not have the distribution of one draw.

??? note "Proof that the expected maximum of $N$ independent nulls grows like $\sigma\sqrt{2\log N}$, and that the asymptotic form overstates it badly at the $N$ a research programme runs"
    Let $Z_1,\dots,Z_N$ be independent standard normals and $M_N=\max_k Z_k$. From $\mathbf{P}(M_N\le z)=\Phi(z)^N$ and the tail bound $1-\Phi(z)\le\varphi(z)/z$, the median of $M_N$ solves $\Phi(z)^N=1/2$, giving $z\approx\Phi^{-1}(2^{-1/N})$, and the standard extreme-value argument yields

    $$\mathbb{E}[M_N]=\sqrt{2\log N}-\frac{\log\log N+\log 4\pi}{2\sqrt{2\log N}}+o\!\left(\frac{1}{\sqrt{\log N}}\right).$$

    The leading term alone is a poor approximation at moderate $N$ because the correction is of order $\log\log N/\sqrt{\log N}$, which is not small until $N$ is enormous. Blom's approximation $\mathbb{E}[M_N]\approx\Phi^{-1}\!\big((N-0.375)/(N+0.25)\big)$ is accurate to about one part in a thousand from $N=5$ upward, and at $N=50$ it gives $2.249$ against $\sqrt{2\log 50}=2.797$.

    The load-bearing hypothesis is independence, and it fails in the direction people guess wrong: a family of correlated variants has a *smaller* expected maximum than an independent one, because correlated draws explore less. **The expected best of fifty nulls is a two-line calculation from a single standard error, and it was available before any of the fifty strategies was coded**, which is what makes reporting a winner without it a choice rather than an oversight.

```python
import numpy as np
from scipy.stats import t as tdist

rng = np.random.default_rng(10037)
paths, se, nobs, sr, nu = 20_000, 0.20, 6_048, 0.30, 4.2       # twenty-four years of daily returns

x = tdist.rvs(nu, size=(paths, nobs), random_state=rng) / np.sqrt(nu / (nu - 2))
r = x * (0.195 / np.sqrt(252)) + sr * 0.195 / 252
shp = np.sqrt(252) * r.mean(axis=1) / r.std(axis=1, ddof=1)
lo, hi = np.quantile(shp, [0.025, 0.975])
print(f"  a Sharpe of {sr} measured on {nobs} daily returns, {paths} independent histories")
print(f"    mean {shp.mean():.4f}   empirical sd {shp.std(ddof=1):.4f}   Lo formula"
      f" {np.sqrt(252 * (1 + sr ** 2 / (2 * 252)) / nobs):.4f}"
      f"   95% interval [{lo:.4f}, {hi:.4f}]")

print(f"  the best of N independent nulls, each with standard error {se}")
print("        N    E[max]    sd(max)    se*sqrt(2 ln N)    cover of the winner 95% CI")
for N in (1, 8, 20, 50, 200):
    g = se * rng.standard_normal((paths, N))
    mx = g.max(axis=1)
    print(f"  {N:9d} {mx.mean():9.4f} {mx.std(ddof=1):10.4f}"
          f" {se * np.sqrt(2 * np.log(N)) if N > 1 else 0.0:18.4f}"
          f" {np.mean(np.abs(mx) <= 1.96 * se):29.4f}")
# =>   a Sharpe of 0.3 measured on 6048 daily returns, 20000 independent histories
#        mean 0.2985   empirical sd 0.2064   Lo formula 0.2041   95% interval [-0.1050, 0.6988]
#      the best of N independent nulls, each with standard error 0.2
#            N    E[max]    sd(max)    se*sqrt(2 ln N)    cover of the winner 95% CI
#              1    0.0008     0.1992             0.0000                        0.9495
#              8    0.2843     0.1218             0.4079                        0.8189
#             20    0.3728     0.1046             0.4895                        0.6039
#             50    0.4506     0.0924             0.5594                        0.2802
#            200    0.5489     0.0801             0.6510                        0.0069
```

The first panel is the derivation working. Twenty thousand independent twenty-four-year histories of a strategy whose true annualized Sharpe is exactly $0.30$ produce estimates with mean $0.2985$ and standard deviation $0.2064$, against Lo's formula $\sqrt{252(1+SR^2/2k)/n}=0.2041$ — agreement to within about one percent, on innovations with a tail heavy enough to break the variance interval of the previous section, and the small excess is that tail rather than an error in the formula. The factor is $(1+SR^2/2k)$ with $k=252$ periods a year, not the $(1+SR^2/2)$ that Lo writes for a *per-period* Sharpe, and at an annualized $0.30$ the difference between the two is the difference between $0.2041$ and $0.2087$; it grows without limit as the Sharpe does, which is why [Statistical Power](../part-12-hypothesis-testing/05-statistical-power.md) states the annualized form explicitly before inverting it for a sample size. The empirical $95\%$ interval is $[-0.1050,0.6988]$ against the course's pinned `Sharpe 0.30 +/- 0.20, 95% CI [-0.09, 0.70]`. Twenty-four years of daily data, and the interval runs from slightly money-losing to genuinely good.

The second panel is the derivation not applying. Each of the $N$ nulls has a true value of exactly zero and an unbiased estimator with standard error $0.20$. The expected maximum climbs $0.0008$, $0.2843$, $0.3728$, $0.4506$, $0.5489$. At $N=50$ it is $0.4506$, which reproduces the course's `expected best of 50 nulls 0.45` and sits above the $+0.43$ that its fifty coin-flip strategies actually achieved — the observed champion did not even clear its own null.

The last two columns close it. The asymptotic $\sqrt{2\log N}$ form reads $0.5594$ at $N=50$ against a truth of $0.4506$, overstating by a quarter, so the crude version of this correction is not merely conservative but wrong enough to be worth replacing. And the winner's own nominal $95\%$ interval — the one a paper prints, computed correctly from the winner's own data — covers the truth $0.9495$ of the time at $N=1$ and $0.2802$ of the time at $N=50$. **The interval is not miscalculated; it is a correct interval for a statistic that was not the one selected, and selection is an operation no line of the estimator's derivation knows about.**

!!! warning "A standard error attached to the statistic you decided to report describes a statistic chosen before the data was seen, and the coverage it advertises is wrong by a factor rather than by a few percent"
    The three routine forms are the best of a parameter grid, the specification that survived a robustness sweep, and the metric that got reported because it looked best — and all three are the second panel above, whatever the research notebook calls them. The free diagnostic is available before any of the work is done: **compute $\text{SE}\times\Phi^{-1}\!\big((N-0.375)/(N+0.25)\big)$ for the $N$ you intend to try, write it down, and treat it as the number a winner has to beat.** At a Sharpe standard error of $0.20$ and fifty variants that threshold is $0.45$; a champion printing $0.43$ has produced no evidence at all, which is exactly what the course's fifty-coin-flip exhibit demonstrates. Two things make this cheap. It requires no correction to any $p$-value and no change to the estimator, only one pre-registered number. And it degrades gracefully when the variants are correlated, since correlation lowers the expected maximum, so the independent calculation is conservative in the safe direction. The full machinery for correlated families is [Part XV](../part-15-multiple-testing/index.md); the one number above is what stops a research programme from needing it.

## An Estimate Without Its Law Is a Rumor, and Some of the Laws Are Wrong

Three things were established and they sit at different distances from usable. The standard error of a mean is exact, and it revealed that the equity premium's precision is a fact about the calendar rather than about the data, unimprovable by any sampling scheme and halved only by three-quarters of a century nobody has. The $t$ law is derived under normality and turns out to be robust far beyond it, while the $\chi^2$ law derived in the same three lines of the same proof collapses to $46\%$ coverage under a realistic tail and gets worse with more data. And the law of a selected maximum is not the law of the statistic at all, so the interval a research process reports around its winner covers the truth barely a quarter of the time.

The symmetry worth carrying is with the part that preceded this one. Deriving a sampling distribution and resampling one are the two available routes to the same object, and they fail on opposite sides. The derivation is exact under a hypothesis nobody checks, which is how a variance interval reaches $0.4566$ while looking identical to one at $0.9507$. The resampling of [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) assumes almost nothing about shape and inherits whatever the sample failed to contain, which is why it is honest about a mean and inconsistent for a maximum. Neither route can rescue the other, and the practical rule is to use the one whose failure mode you have checked rather than the one whose assumptions you prefer.

Every law on this page required naming the family the data came from — normality for the rotation, a finite fourth moment for the variance interval, independence for the maximum. Naming families, saying what it costs to name them wrongly, and finding the ones that cannot be pinned down at all is [Statistical Models](04-statistical-models.md).

**A standard error is a claim about samples that were never drawn, and it is only ever as good as the description of the process that would have drawn them.**
