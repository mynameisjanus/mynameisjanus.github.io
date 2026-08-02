# Bootstrap Methods

Everything earlier in this part assumed a law: a distribution written down, sampled from, and reweighted or stratified as convenient. Strategy evaluation never has one. It has a few thousand rows of history, a statistic nobody has derived a standard error for, and a decision that has to be made anyway. The bootstrap answers by making one substitution — use the sample's own distribution in place of the unknown one — and then running the simulation machinery of the previous pages against it. What that buys is an error bar for any statistic whatever, with no derivation. What it cannot buy is information the sample does not contain, and almost every way the method fails is a version of being asked for exactly that.

This page covers the plug-in principle and the theorem that licenses it, the four standard interval constructions and where they stop agreeing, the dependence assumption hidden in independent resampling and which kind of dependence actually breaks it, the non-smooth statistics for which the bootstrap is not merely inaccurate but inconsistent, and resampling under a null as a distinct procedure carrying a Monte Carlo error of its own. It does not build the interval taxonomy in full, which is [Part XI](../part-11-parameter-estimation/index.md); it does not develop permutation tests as tests, which is [Part XII](../part-12-hypothesis-testing/index.md); it does not correct for multiplicity or construct a reality check, which is [Part XV](../part-15-multiple-testing/index.md); it does not use leave-one-out resampling, which is [Jackknife Methods](08-jackknife-methods.md); it proves none of the limit theorems it leans on, which is [Part VII](../part-07-asymptotic-theory/index.md); and it fits no model to the dependence it resamples around, which is [Time Series](../../part-03-statistics/03-time-series.md).

The trading stake is the most-cited promise in the appendix. [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) names this page for the general theory and then makes a claim on its behalf: "the same twenty lines of code produce an error bar for *any* statistic — Sharpe, drawdown, hit rate, skew of the worst decade — with no derivation, ever." That is true and the second section prices it. On five years of daily returns all four standard interval constructions land within half a point of their nominal coverage and the choice between them does not matter; on the thirty-six monthly observations that same lesson warns produce "beautiful intervals", three of the four cover a nominal ninety-five percent about ninety-one or ninety-two percent of the time, and the interval they produce for a Sharpe ratio of $0.30$ is more than two units wide.

## The Plug-In Principle Is One Substitution

Let $\theta=T(F)$ be a functional of the unknown distribution $F$ — a mean, a Sharpe ratio, a maximum drawdown, a difference of hit rates. The **empirical distribution function** $\hat F_n$ puts mass $1/n$ on each observed value, and the plug-in estimator is $\hat\theta=T(\hat F_n)$. The bootstrap's single idea is to apply the substitution twice: not only estimate $\theta$ by $T(\hat F_n)$, but estimate the *sampling distribution* of $T(\hat F_n)$ under $F$ by the sampling distribution of $T(\hat F_n^{\ast})$ under $\hat F_n$, where $\hat F_n^{\ast}$ is the empirical distribution of $n$ draws taken with replacement from the data.

The second substitution is the one that needs justifying, and its licence is a uniform convergence result the appendix has already stated. [Cumulative Distribution Functions](../part-03-random-variables/02-cumulative-distribution-functions.md) records that the empirical distribution function of a continuous law "has ten atoms and no density at all" and is "not a slightly noisy version of the truth" but "a different kind of object", and then supplies the rescue: $\sup_x|\hat F_n(x)-F(x)|\to0$ with probability one, naming this as "why resampling from it works at all".

??? note "Proof that the plug-in works for a smooth functional, and exactly which word is load-bearing"
    Suppose $T$ is Hadamard-differentiable at $F$ with influence function $\psi_F$, which for the statistics used in practice means $T$ can be written to first order as an average:

    $$\sqrt n\big(T(\hat F_n)-T(F)\big)=\frac{1}{\sqrt n}\sum_{i=1}^{n}\psi_F(X_i)+o_p(1).$$

    The Glivenko–Cantelli theorem gives $\hat F_n\to F$ uniformly, so $\psi_{\hat F_n}\to\psi_F$, and the same expansion applied to a resample from $\hat F_n$ gives

    $$\sqrt n\big(T(\hat F_n^{\ast})-T(\hat F_n)\big)=\frac{1}{\sqrt n}\sum_{i=1}^{n}\psi_{\hat F_n}(X_i^{\ast})+o_p(1).$$

    Both right-hand sides are standardized sums of independent draws with the same limiting variance $\mathrm{var}(\psi_F(X))$, so by the central limit theorem the two distributions converge to the same normal law. The bootstrap distribution of $T(\hat F_n^{\ast})-T(\hat F_n)$ therefore approximates the sampling distribution of $T(\hat F_n)-T(F)$, which is the entire claim.

    The load-bearing word is **smooth**. The argument requires the functional to be locally approximable by an average, which is what converts a statement about $\hat F_n$ being close to $F$ into a statement about $T(\hat F_n)$ being close to $T(F)$ *at the right rate and with the right shape*. A mean, a variance, a correlation, a regression coefficient, a Sharpe ratio and a quantile at a point of positive density all qualify. A sample maximum, a mode, the number of distinct values and a quantile at the edge of the support do not, and for those the bootstrap does not merely lose accuracy — the fourth section shows it converges to the wrong distribution and stays there.

## Four Intervals, and They Stop Agreeing Exactly When It Matters

Given the bootstrap replicates $\hat\theta^{\ast}_1,\dots,\hat\theta^{\ast}_B$ there are four standard ways to turn them into an interval, and they are not the same construction described four ways. The **normal** interval takes $\hat\theta\pm1.96\,\mathrm{sd}(\hat\theta^{\ast})$, using the replicates only for a standard error. The **percentile** interval reads the $2.5$th and $97.5$th percentiles of the replicates directly. The **basic** or pivotal interval reflects them through the estimate, $\big(2\hat\theta-\hat\theta^{\ast}_{(0.975)},\,2\hat\theta-\hat\theta^{\ast}_{(0.025)}\big)$, on the reasoning that it is $\hat\theta-\theta$ whose distribution was estimated. And **BCa** adjusts the percentile levels for median bias and for skewness, using a bias constant read off the replicates and an acceleration constant computed by leave-one-out.

```python
import numpy as np
from scipy.special import ndtri
from scipy.stats import norm
from scipy.stats import t as tdist

rng = np.random.default_rng(9071)
nu, sr = 2.6, 0.30                                             # the tail Part III fits, and its Sharpe


def sharpe(x, per, axis=-1):
    return np.sqrt(per) * x.mean(axis=axis) / x.std(axis=axis, ddof=1)


def intervals(x, per, b):
    n, hat = x.size, sharpe(x, per)
    star = sharpe(x[rng.integers(n, size=(b, n))], per, axis=1)
    s, q = x.sum(), (x ** 2).sum()
    m = (s - x) / (n - 1)                                      # leave-one-out, in closed form
    loo = np.sqrt(per) * m / np.sqrt((q - x ** 2 - (n - 1) * m ** 2) / (n - 2))
    d = loo.mean() - loo
    a = (d ** 3).sum() / (6 * (d ** 2).sum() ** 1.5)
    z0 = ndtri(np.clip((star < hat).mean(), 1e-6, 1 - 1e-6))
    lo, hi = (norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))) for z in (-1.96, 1.96))
    sd = star.std(ddof=1)
    return {"normal": (hat - 1.96 * sd, hat + 1.96 * sd),
            "basic": (2 * hat - np.quantile(star, 0.975), 2 * hat - np.quantile(star, 0.025)),
            "percentile": tuple(np.quantile(star, [0.025, 0.975])),
            "BCa": tuple(np.quantile(star, [lo, hi]))}


print(f"  coverage of nominal 95% intervals for an annualized Sharpe of {sr}, t({nu}) returns")
print("   sample            reps    normal    basic    percentile      BCa    median width")
for label, n, per, reps, b in (("36 months", 36, 12, 3_000, 2_000),
                               ("120 months", 120, 12, 3_000, 2_000),
                               ("360 months", 360, 12, 3_000, 2_000),
                               ("1260 days", 1_260, 252, 600, 1_000)):
    hit = dict.fromkeys(("normal", "basic", "percentile", "BCa"), 0)
    width = []
    for _ in range(reps):
        x = tdist.rvs(nu, size=n, random_state=rng) / np.sqrt(nu / (nu - 2)) + sr / np.sqrt(per)
        ci = intervals(x, per, b)
        for k, (lo, hi) in ci.items():
            hit[k] += lo <= sr <= hi
        width.append(ci["percentile"][1] - ci["percentile"][0])
    print(f"  {label:<14} {reps:7d} {hit['normal'] / reps:9.3f} {hit['basic'] / reps:8.3f}"
          f" {hit['percentile'] / reps:13.3f} {hit['BCa'] / reps:8.3f} {np.median(width):15.2f}")
# =>   coverage of nominal 95% intervals for an annualized Sharpe of 0.3, t(2.6) returns
#       sample            reps    normal    basic    percentile      BCa    median width
#      36 months         3000     0.920    0.908         0.917    0.945            2.31
#      120 months        3000     0.928    0.914         0.937    0.945            1.24
#      360 months        3000     0.927    0.911         0.929    0.940            0.72
#      1260 days          600     0.940    0.935         0.945    0.947            1.74
```

The BCa column is the one that behaves. It reads $0.945$, $0.945$, $0.940$ and $0.947$ against a nominal $0.95$, essentially correct at every sample size including thirty-six observations. The other three do not get there: the basic interval is the worst throughout at $0.908$ to $0.935$, and the normal and percentile intervals sit between $0.917$ and $0.945$. The ordering is the theory's — BCa is second-order accurate and the others are first-order — and the size of the gap is the fat tail, since a $t$ with $2.6$ degrees of freedom makes the Sharpe's sampling distribution skewed, which is precisely the defect BCa's two corrections are built to absorb.

What does not happen is equally worth noting: the shortfall does not close as the sample grows. At three hundred and sixty monthly observations — thirty years, longer than most funds exist — the percentile interval still covers $0.929$ of the time. It is the daily row that recovers, at $0.945$, because $1{,}260$ daily observations of a heavy-tailed law carry more information about its shape than $360$ monthly ones do. **Sampling frequency and sample length are not interchangeable when the tail is what limits you.**

The width column is where the practitioner's version of this lives. At thirty-six monthly observations the median percentile interval is $2.31$ wide, for a quantity whose true value is $0.30$ — an interval running from roughly $-0.85$ to $+1.45$, which excludes essentially nothing anybody would consider. This is the arithmetic behind the lesson's warning about "beautiful intervals from 36 monthly observations": the interval is beautiful in the sense of being smooth, symmetric and easy to plot, and it is $92\%$ accurate about a number it cannot locate to within a factor of five.

!!! note "The four intervals are not four opinions, and the two that disagree are disagreeing about which quantity was resampled"
    The percentile and basic intervals are reflections of each other through $\hat\theta$, so they coincide exactly when the bootstrap distribution is symmetric and diverge in proportion to its skew — and they diverge in *opposite* directions, which is why seeing them agree is a genuine check and seeing them differ tells you which way the statistic is skewed. Their logic differs: the basic interval assumes $\hat\theta-\theta$ has a distribution not depending on $\theta$, and inverts it; the percentile interval assumes there is some monotone transformation on which the bootstrap distribution is symmetric, and lets the quantiles find it. Neither assumption is checkable and BCa's contribution is to estimate the two ways they fail — median bias through $z_0$ and skew through the acceleration $a$ — rather than assume them away. The acceleration constant is where the next page enters: it is computed from the $n$ leave-one-out values of the statistic, so **every BCa interval in the course, including the one [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) obtains with `method="BCa"`, has a jackknife running inside it**, which is [Jackknife Methods](08-jackknife-methods.md).

## Independent Days Are the Assumption, and Which Dependence Matters Is Not Obvious

Drawing rows with replacement destroys their order, so the ordinary bootstrap assumes exchangeability — the assumption [Independence](../part-02-probability-foundations/05-independence.md) names when it says the bootstrap "resamples as if it held". Financial series are not exchangeable, and the standard repairs resample *blocks* instead of rows: the **moving block** bootstrap draws fixed-length runs from random start points, and the **stationary** bootstrap draws geometrically-distributed lengths around a mean, which avoids the seam artifacts of a fixed grid. Both use a block length near $n^{1/3}$.

What is less often said is which dependence the repair is for.

```python
import numpy as np

rng = np.random.default_rng(9073)
n, block, reps, b, mu = 500, 8, 1_000, 500, 0.05              # block length is about n^(1/3)


def iid(m):
    return mu + rng.standard_normal(m)


def garch(m):                                                  # uncorrelated level, clustered vol
    z = rng.standard_normal(m)
    s2 = np.empty(m)
    s2[0] = 1.0
    for t in range(1, m):
        s2[t] = 0.05 + 0.10 * (s2[t - 1] * z[t - 1] ** 2) + 0.85 * s2[t - 1]
    return mu + np.sqrt(s2) * z


def ar1(m, rho=0.5):                                           # correlated level
    e = rng.standard_normal(m) * np.sqrt(1 - rho ** 2)
    x = np.empty(m)
    x[0] = e[0]
    for t in range(1, m):
        x[t] = rho * x[t - 1] + e[t]
    return mu + x


def idx_iid():
    return rng.integers(n, size=(b, n))


def idx_block():
    nb = -(-n // block)
    st = rng.integers(n - block + 1, size=(b, nb))
    return (st[:, :, None] + np.arange(block)).reshape(b, -1)[:, :n]


def idx_stat():                                                # geometric lengths, circular
    out = np.empty((b, n), dtype=np.int64)
    out[:, 0] = rng.integers(n, size=b)
    jump = rng.random((b, n)) < 1.0 / block
    fresh = rng.integers(n, size=(b, n))
    for t in range(1, n):
        out[:, t] = np.where(jump[:, t], fresh[:, t], (out[:, t - 1] + 1) % n)
    return out


print(f"  coverage of a nominal 95% percentile interval for the mean, n = {n},"
      f" {reps} samples, {b} resamples")
print("   process                iid boot    moving block    stationary    mean width, iid"
      "    block")
for name, gen in (("iid normal", iid), ("GARCH(1,1) in the square", garch),
                  ("AR(1) in the level, 0.5", ar1)):
    hit = {"iid": 0, "block": 0, "stat": 0}
    wid = {"iid": [], "block": []}
    for _ in range(reps):
        x = gen(n)
        for key, f in (("iid", idx_iid), ("block", idx_block), ("stat", idx_stat)):
            lo, hi = np.quantile(x[f()].mean(axis=1), [0.025, 0.975])
            hit[key] += lo <= mu <= hi
            if key in wid:
                wid[key].append(hi - lo)
    print(f"  {name:<25} {hit['iid'] / reps:8.3f} {hit['block'] / reps:15.3f}"
          f" {hit['stat'] / reps:13.3f} {np.mean(wid['iid']):18.4f} {np.mean(wid['block']):8.4f}")
# =>   coverage of a nominal 95% percentile interval for the mean, n = 500, 1000 samples, 500 resamples
#       process                iid boot    moving block    stationary    mean width, iid    block
#      iid normal                   0.949           0.945         0.941             0.1733   0.1718
#      GARCH(1,1) in the square     0.946           0.942         0.944             0.1719   0.1709
#      AR(1) in the level, 0.5      0.755           0.920         0.926             0.1733   0.2695
```

The third row is the failure everyone expects. With a lag-one autocorrelation of $0.5$ in the level, the independent bootstrap covers a nominal $95\%$ interval **$75.5\%$** of the time, because it resamples days as if each carried a full observation's worth of information when in fact the effective sample size is a fraction of $n$. Both block schemes repair it, to $0.920$ and $0.926$, and the repair is visible in the width: the block interval averages $0.2695$ against the independent one's $0.1733$, so it is $56\%$ wider and the extra width is what the coverage was missing.

The second row is the one that resolves something in the course. A GARCH process has strong dependence — volatility clusters, squared returns are autocorrelated for months — and the independent bootstrap covers $0.946$, entirely correctly, while the block interval comes out at $0.1709$ against $0.1719$, which is very slightly *narrower*. That is not a defect. The sampling variance of a mean depends on the autocovariances of the *level* and on nothing else, and a GARCH series has none; the dependence lives in the square, where the mean cannot see it.

This is exactly what [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) observes and declines to explain away, reporting that "the block interval is a hair *narrower* than the iid one (SE 0.19 vs 0.20), not wider" and adding that the direction of the correction "is the data's decision, not a rule". The rows above are that decision made three times with the answer known in advance. Its series has a lag-one autocorrelation of $-0.061$ — essentially the first row — and pronounced volatility clustering, which is the second. **The block bootstrap is a repair for dependence in the statistic, not for dependence in the data, and volatility clustering is invisible to a mean while being the dominant feature of the series.** For a statistic that does read the second moment — a volatility estimate, a drawdown, a Sharpe over a short window — the same clustering widens the interval substantially, and the choice of scheme starts to matter for the same series.

## The Bootstrap of a Maximum Is Not the Maximum's Distribution

Smoothness was the load-bearing word in the first proof, and its failure is not a matter of degree.

??? note "Proof that the bootstrap of the sample maximum is inconsistent, with the atom that causes it"
    Let $X_{(n)}$ be the sample maximum and $X^{\ast}_{(n)}$ the maximum of a resample. The resample misses the observed maximum entirely only if none of its $n$ draws select that index, which has probability $(1-1/n)^{n}$. Hence

    $$\mathbf{P}\big(X^{\ast}_{(n)}=X_{(n)}\big)=1-\left(1-\frac1n\right)^{n}\longrightarrow 1-e^{-1}=0.6321.$$

    The bootstrap distribution of the maximum therefore places an atom of mass $0.632$ at the observed maximum, for every $n$, no matter how large. It also has support bounded above by $X_{(n)}$, so it assigns probability zero to the event that the true maximum exceeds the observed one — an event whose probability under $F$ is one whenever $F$ has unbounded support. The true sampling distribution of $X_{(n)}$, suitably normalized, converges to one of the three extreme-value laws and is continuous; the bootstrap's version has a point mass of $0.632$ and never converges to it.

    The load-bearing feature is that the statistic depends on the sample through a single observation rather than through an average. Nothing about $n$ repairs it, which distinguishes this from the ordinary small-sample inaccuracies of the previous sections: the percentile interval at thirty-six observations is $92\%$ accurate instead of $95\%$, and the bootstrap of a maximum is wrong by a fixed amount forever. The two standard repairs both work by resampling *fewer* points — the $m$-out-of-$n$ bootstrap with $m/n\to0$, and subsampling without replacement — which restores consistency by making the atom vanish. A statistic that depends on a handful of extreme observations, which includes a maximum drawdown computed on a short history and any tail quantile beyond the data's reach, sits somewhere between the two regimes, and where exactly is a question the bootstrap cannot answer about itself.

## Resampling Under a Null Is a Different Procedure With Its Own Error Bar

An interval and a test are not the same operation. A confidence interval resamples the data as it is; a test has to resample under a *null*, which means constructing a resampling scheme in which the effect being tested is absent by construction and everything else is preserved. [Probability Axioms](../part-02-probability-foundations/02-probability-axioms.md) makes the point that a uniform law over such a scheme "is what [Permutation Tests] and [Bootstrap Methods] *impose on purpose*, by generating the sample space themselves rather than inheriting it from the world", and that this is what makes the resulting $p$-values exact rather than asymptotic.

Two constructions do most of the work. Demeaning each series before resampling imposes a zero-mean null while leaving the dependence structure intact, which is what [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) means by "demeaning each variant imposes the null". And circularly shifting a position series against returns destroys the timing while preserving the long-short mix, the autocorrelation and the exposure, which is the "shift, don't shuffle" rule. The $p$-value is then the statistic's rank in the resampled null,

$$\hat p=\frac{1+\#\{S^{\ast}_b\geq S\}}{1+B},$$

where the $+1$ in both places is not cosmetic: it counts the observed data as one of its own permutations, which is what keeps the test exact — without it, $\hat p$ can be zero, and a test that can report a $p$-value of zero has a size larger than its nominal level.

```python
import numpy as np

rng = np.random.default_rng(9077)
n, look, phi, reps = 2_520, 2, 0.08, 4_000
e = rng.standard_normal(n) * 0.0122
ret = e + phi * np.roll(e, 1)                                  # a faint, real momentum edge
c = np.cumsum(ret)
pos = np.concatenate([np.zeros(look),
                      np.sign(c[look - 1:-1] - np.concatenate([[0.0], c[:-look - 1]]))])


def sr(p):
    x = p * ret
    return np.sqrt(252) * x.mean() / x.std(ddof=1)


actual = sr(pos)
shifted = np.array([sr(np.roll(pos, k)) for k in range(n)])    # the entire permutation null
exact = (shifted >= actual).mean()
print(f"  circular-shift permutation test, Sharpe {actual:.3f}, exact p = {exact:.4f}"
      f" over all {n} shifts")
print("        B    mean p-hat    sd of p-hat    sqrt(p(1-p)/B)    granularity"
      "    P(crosses 0.05)")
for b in (100, 500, 1_000, 5_000, 20_000):
    k = rng.integers(1, n, size=(reps, b))
    p_hat = (1 + (shifted[k] >= actual).sum(axis=1)) / (1 + b)
    print(f"  {b:9d} {p_hat.mean():13.4f} {p_hat.std(ddof=1):14.4f}"
          f" {np.sqrt(exact * (1 - exact) / b):17.4f} {1 / (1 + b):14.5f}"
          f" {np.mean((p_hat < 0.05) != (exact < 0.05)):17.3f}")
# =>   circular-shift permutation test, Sharpe 0.486, exact p = 0.0615 over all 2520 shifts
#            B    mean p-hat    sd of p-hat    sqrt(p(1-p)/B)    granularity    P(crosses 0.05)
#            100        0.0708         0.0242            0.0240        0.00990             0.262
#            500        0.0630         0.0107            0.0107        0.00200             0.127
#           1000        0.0622         0.0074            0.0076        0.00100             0.055
#           5000        0.0612         0.0034            0.0034        0.00020             0.000
#          20000        0.0612         0.0017            0.0017        0.00005             0.000
```

The construction is chosen so the answer is knowable. A circular shift of a series of length $n$ has exactly $n$ distinct outcomes, so the permutation null can be *enumerated* rather than sampled, and the exact $p$-value here is $0.0615$ for a strategy with an annualized Sharpe of $0.486$ — deliberately close to the family champion [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) reports at Sharpe $0.46$ and $p=0.06$. Any run using $B$ random shifts is estimating a number that could have been computed exactly, so its entire error is Monte Carlo error.

The third and fourth columns confirm what that error is. The measured standard deviation of $\hat p$ across four thousand independent runs matches $\sqrt{p(1-p)/B}$ to the printed precision at every $B$ — $0.0242$ against $0.0240$, $0.0107$ against $0.0107$, $0.0034$ against $0.0034$. A bootstrap $p$-value is a sample proportion and has the standard error of one, which is obvious once written down and almost never reported.

The last column is why it matters. That lesson's Reality Check uses $B=500$ resamples and prints $p=0.06$. At $B=500$ the standard error of a $p$-value near $0.06$ is $0.0107$, so the honest statement is $0.06\pm0.02$ — and **a run that uses five hundred resamples on a truly-$0.0615$ effect lands on the other side of the conventional $0.05$ threshold $12.7\%$ of the time.** At $B=100$ it is $26.2\%$, which is to say a coin flip dressed as evidence. The remedy is arithmetic rather than judgement: resolving $p=0.06$ against a $0.05$ threshold to one standard error needs $B$ of about five thousand, which the last two rows deliver at $0.0034$ and $0.0017$, and five thousand resamples of a strategy family is minutes of compute.

!!! warning "A bootstrap interval computed on inputs that were chosen by looking at the data is a confidence interval for a decision that was already made"
    Everything on this page conditions on the sample, and the sample was in the room when the strategy was designed. Three consequences follow and none is repaired by more resamples. A bootstrap interval around the best of fifty variants prices the sampling noise in that variant and not the fifty-way maximum that selected it, which is the correlated-family effect [Monte Carlo Simulation](03-monte-carlo-simulation.md) measures and what the Reality Check exists to charge for. A block length chosen because it produced a narrower interval has been fitted to the data, and the interval no longer has its nominal coverage. And a statistic whose functional form was picked after seeing which one looked significant has no null at all, resampled or otherwise. The bootstrap is honest about estimation noise and silent about selection, and the two are routinely reported as though the first covered the second — which is what the source lesson means when it says a bootstrap "cannot launder multiple testing".

## The Bootstrap Resamples Your History, Not Your Future

The promise this page was asked to justify holds, with the boundaries now visible. Twenty lines of resampling do produce an error bar for any smooth statistic with no derivation, and on a long daily sample all four standard constructions agree closely enough that the choice does not matter. On a short sample the choice matters and BCa is the answer, at the cost of a leave-one-out pass. On a dependent series the scheme matters, and *which* dependence matters depends on the statistic rather than on the data — volatility clustering is invisible to a mean and dominant for a volatility. On a non-smooth statistic the method is not inaccurate but inconsistent, with a point mass of $0.632$ that no sample size removes. And a resampled $p$-value is a sample proportion whose own standard error is usually larger than the precision it is quoted to.

Underneath all of that is one limitation, and it is the same one every technique in this part has in a different costume. The bootstrap draws from $\hat F_n$, so it can redistribute the information in the sample honestly and cannot add any. A tail it never saw has probability zero under $\hat F_n$; a regime absent from the history has probability zero; a correlation that only appears in a crisis is not in the resample if the crisis is not in the data. The source lesson states the consequence exactly — "the moment a bootstrap interval is presented as 'the range of outcomes we expect going forward,' it has been promoted beyond its competence" — and the promotion is easy precisely because the method feels assumption-free. It is not assumption-free. Its assumption is that the future resembles a reshuffling of the past, which is the strongest assumption anywhere in this part and the only one stated by omission.

What remains is the deterministic sibling of everything here. Resampling with replacement is a Monte Carlo procedure and carries a Monte Carlo error — the $0.0107$ on a $p$-value, the resample-to-resample wobble in an interval endpoint — which is an error nobody needs to accept, because there exists a resampling scheme with only $n$ possible outcomes that can be enumerated exactly. It is weaker, older, cheaper, and it is already running inside every BCa interval computed above. That is [Jackknife Methods](08-jackknife-methods.md).
