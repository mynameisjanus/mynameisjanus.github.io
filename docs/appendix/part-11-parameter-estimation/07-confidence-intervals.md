# Confidence Intervals

A confidence interval is the most widely reported and most widely misread object in applied statistics, and both facts have the same cause: the sentence that defines it is about a procedure and the sentence everybody hears is about a number. "Ninety-five percent" is a property of a machine that turns samples into intervals, measured over samples that were never drawn, and it says nothing whatever about the particular pair of endpoints on your screen. That distinction sounds like pedantry until it is priced, and this page prices it three ways: an interval whose coverage gets *worse* as the sample grows, an interval whose coverage depends entirely on which coordinate somebody built it in, and an interval that is a perfectly correct probability statement about the parameter and covers the truth zero percent of the time.

This page covers coverage as a property of a procedure that cannot be conditioned on the sample it produced, the construction of every interval by inverting a pivot, the coverage of an asymptotic interval as a function of the parameter and its non-monotonicity in $n$, the dependence of coverage on the scale the interval was built on, and the difference between a credible interval and a confidence interval. It derives no sampling distribution, which is [Sampling Distributions](../part-10-statistics-foundations/03-sampling-distributions.md); it proves no central limit theorem and linearizes no statistic, which are [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md) and [The Delta Method](../part-07-asymptotic-theory/04-delta-method.md); it resamples nothing, which are [Bootstrap Confidence Intervals](08-bootstrap-confidence-intervals.md) and [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it tests no hypothesis and draws no critical region, so the duality is spent here and proved in [The Hypothesis Testing Framework](../part-12-hypothesis-testing/01-hypothesis-testing-framework.md); it constructs no posterior, which are [Bayesian Estimation](05-bayesian-estimation.md) and [Part XVI](../part-16-bayesian-statistics/index.md); it adjusts nothing for the number of intervals computed, which is [Part XV](../part-15-multiple-testing/index.md); it builds no prediction interval for a future observation, which is [Part XIII](../part-13-regression/index.md); and it never claims a $95\%$ interval has a $95\%$ chance of containing the number inside it.

The trading stake is an agreement that gets read as a validation. [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) computes a momentum strategy's hit rate as `hit rate 0.5406  (3329 of 6158)`, quotes a `95% credible [0.5281, 0.5530]` beside a `Wald 95% CI  [0.5282, 0.5530]`, and concludes that "with a flat prior and six thousand observations, the credible interval and the frequentist interval agree to the third decimal — the data has drowned the prior, as it should." The agreement is real and the third section shows what it is and is not evidence of: the same Wald formula, at the same nominal level, covers $0.9496$ there and $0.8714$ at a sample size of five hundred with a rarer event — worse than at two hundred and fifty.

## Coverage Is a Property of the Procedure and Cannot Be Conditioned on the Sample It Produced

A **confidence interval at level $1-\alpha$** is a pair of statistics $L(X)\le U(X)$ satisfying

$$\mathbf{P}_\theta\big(L(X)\le\theta\le U(X)\big)\ge1-\alpha\quad\text{for every }\theta.$$

Read the quantifiers. The probability is over $X$ at a fixed $\theta$; the random objects are the endpoints and the fixed object is the parameter. Once a sample is in hand, $L$ and $U$ are numbers and $\theta$ is a number, so the event either happened or it did not, and there is no probability left to report. **The interval is a draw from a factory whose long-run defect rate is $\alpha$, and nothing about the particular unit is known.**

That is not a philosophical scruple. It has a concrete consequence: coverage cannot generally be conditioned on features of the sample you actually observed, and there are cases where the sample tells you with certainty whether the interval is right.

??? note "Proof that inverting a pivot gives exact coverage, and that the coverage statement is unconditional and can be worthless once a sample is in hand"
    A **pivot** is a function $Q(X,\theta)$ whose distribution does not depend on $\theta$. If $q_{\alpha/2}$ and $q_{1-\alpha/2}$ are its quantiles then $\mathbf{P}_\theta\big(q_{\alpha/2}\le Q(X,\theta)\le q_{1-\alpha/2}\big)=1-\alpha$ at every $\theta$ by construction, and solving the inequalities for $\theta$ produces an interval with exactly that coverage. For $X_i\sim\mathcal{N}(\mu,\sigma^{2})$ the quantity $(\bar X-\mu)/(s/\sqrt n)$ is a pivot with a $t_{n-1}$ law — the one [Sampling Distributions](../part-10-statistics-foundations/03-sampling-distributions.md) derives — and inverting it gives the interval every package returns.

    Now the conditioning failure, in its sharpest form. Let $\theta$ be unknown and let $X_1,X_2$ be independent, each taking the value $\theta-1$ or $\theta+1$ with probability $\tfrac12$. Define the interval to be the single point $\{(X_1+X_2)/2\}$ if $X_1\ne X_2$ and the single point $\{X_1-1\}$ if $X_1=X_2$. If the two observations differ, their average *is* $\theta$; if they agree, the reported point is $X_1-1$, which equals $\theta$ exactly when both came out high — half the time, given agreement. Unconditionally the procedure covers

    $$\tfrac12\cdot1+\tfrac12\cdot\tfrac12=0.75,$$

    so it is a valid $75\%$ interval. But the data always says which case you are in: when the observations differ you know the interval is right, and when they agree you know it is right with probability one-half. **Reporting $75\%$ in either case is correct about the procedure and wrong about the sample**, and no amount of additional data changes the structure.

    The load-bearing word is *unconditional*. Coverage averages over the entire sample space, including samples that would have been visibly uninformative and samples that would have settled the question outright. **Coverage describes a factory and not a number, and "there is a ninety-five percent chance the hit rate is between $0.5281$ and $0.5530$" is the one sentence the interval does not license** — that sentence needs a distribution over $\theta$, which is the fifth section.

## Every Interval Is a Pivot Inverted, and the Only Question Is Whether the Pivot Is Exact

The proof gives the recipe and the recipe organizes everything that follows. Find a quantity depending on the data and the parameter whose distribution is free of the parameter, look up its quantiles, and solve. The $t$ interval for a normal mean is exact because the pivot's law is exactly $t_{n-1}$. The $\chi^{2}$ interval for a normal variance is exact for the same reason, inverting $(n-1)s^{2}/\sigma^{2}\sim\chi^{2}_{n-1}$. Fisher's $z$ for a correlation is nearly exact because $\tfrac12\log\frac{1+\hat\rho}{1-\hat\rho}$ has a variance that does not depend on $\rho$, which is exactly what makes it a better pivot than $\hat\rho$ itself.

Most intervals in practice are not exact, because most estimators have no exact pivot. The standard substitute is the **Wald interval**, $\hat\theta\pm z_{1-\alpha/2}\,\hat{\mathrm{se}}(\hat\theta)$, built on the asymptotic normality of [Properties of Estimators](02-properties-of-estimators.md) and the observed information of [Maximum Likelihood Estimation](03-maximum-likelihood-estimation.md). It rests on two approximations rather than one: that $(\hat\theta-\theta)/\mathrm{se}$ is approximately normal, and that $\hat{\mathrm{se}}$ may be substituted for the unknown $\mathrm{se}$. The second is where it breaks, and it breaks in a way that does not go away.

A third route exists and it is the reason [Part XII](../part-12-hypothesis-testing/index.md) and this page are two views of one construction: the set of $\theta_0$ that a level-$\alpha$ test would fail to reject is a $1-\alpha$ confidence interval, and conversely. That duality is worth knowing and is proved there; here it is spent rather than derived.

## An Asymptotic Interval's Coverage Is a Function of the Parameter and It Oscillates With $n$

Coverage is a function $\theta\mapsto\mathbf{P}_\theta(\text{covered})$, not a number, and for a discrete observation it is a jagged function of both $\theta$ and $n$.

??? note "Proof that the Wald interval for a proportion has coverage that oscillates in $n$ and does not converge monotonically, and that the cause is a lattice meeting a continuous approximation"
    For $k\sim\mathrm{Binomial}(n,p)$ the coverage of any interval rule is computable exactly, with no simulation:

    $$C(n,p)=\sum_{k=0}^{n}\binom{n}{k}p^{k}(1-p)^{n-k}\,\mathbf 1\big\{L(k)\le p\le U(k)\big\},$$

    a finite sum of $n+1$ terms. As $p$ moves continuously, individual indicators switch on and off at the values of $p$ where an endpoint crosses $p$, so $C$ is a step function in $p$ with $n+1$ jumps — it cannot be smooth, and its value at any given $p$ depends on where that $p$ happens to fall relative to the lattice of achievable $\hat p=k/n$. Increasing $n$ by one relocates every jump, so $C(n,p)$ is not monotone in $n$ and a larger sample can land $p$ in a worse position than a smaller one did.

    The Wald rule makes this far worse than it needs to be, for a structural reason. It uses $\hat p(1-\hat p)/n$ in place of $p(1-p)/n$, so it is not the inversion of a pivot but the inversion of an estimate of one. When $k=0$ the estimated standard error is exactly zero and the interval is the single point $\{0\}$, which covers no positive $p$ at all — so for small $np$ the coverage is bounded above by $1-(1-p)^{n}$ no matter what the nominal level says. The **Wilson interval**, obtained by solving $|\hat p-p|\le z\sqrt{p(1-p)/n}$ for $p$ as a quadratic, inverts the genuine pivot and has no such collapse; the **Clopper–Pearson** interval inverts the exact binomial law and is conservative by construction.

    The load-bearing step is substituting $\hat p$ for $p$ inside the standard error, which converts an inverted pivot into an inverted approximation to one. **The Wald interval is the only construction on this page that is not an inverted pivot, and it is the one every package returns by default.**

```python
import numpy as np
from scipy.stats import beta, binom, norm

rng = np.random.default_rng(11071)
z, reps = norm.ppf(0.975), 200_000

def bounds(k, n, kind):
    p = k / n
    if kind == "wald":
        h = z * np.sqrt(p * (1 - p) / n)
        return p - h, p + h
    if kind == "wilson":
        d, c = 1 + z ** 2 / n, p + z ** 2 / (2 * n)
        h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
        return (c - h) / d, (c + h) / d
    if kind == "cp":
        return (beta.ppf(0.025, k, n - k + 1), beta.ppf(0.975, k + 1, n - k))
    return beta.ppf(0.025, k + 0.5, n - k + 0.5), beta.ppf(0.975, k + 0.5, n - k + 0.5)

w, n0 = 3_329, 6_158                                           # the course's hit-rate sample
print(f"  the lesson's numbers: hit rate {w / n0:.4f}, Wald"
      f" [{bounds(w, n0, 'wald')[0]:.4f}, {bounds(w, n0, 'wald')[1]:.4f}], Jeffreys"
      f" [{bounds(w, n0, 'jeff')[0]:.4f}, {bounds(w, n0, 'jeff')[1]:.4f}]")
print("        n         p    exact Wald    Wilson    Clopper-Pearson    Jeffreys"
      "    simulated Wald")
for p in (0.5406, 0.01):
    for n in (30, 100, 250, 500, 1000, 6158):
        k = np.arange(n + 1)
        pm = binom.pmf(k, n, p)
        cov = []
        for kind in ("wald", "wilson", "cp", "jeff"):
            lo, hi = bounds(np.clip(k, 1e-12, None), n, kind)
            cov.append(float(pm[(np.nan_to_num(lo) <= p) & (p <= np.nan_to_num(hi, nan=0.0))].sum()))
        s = rng.binomial(n, p, reps)
        lo, hi = bounds(s, n, "wald")
        print(f"  {n:9d} {p:9.4f} {cov[0]:13.4f} {cov[1]:9.4f} {cov[2]:18.4f} {cov[3]:11.4f}"
              f" {((lo <= p) & (p <= hi)).mean():17.4f}")
# =>   the lesson's numbers: hit rate 0.5406, Wald [0.5282, 0.5530], Jeffreys [0.5281, 0.5530]
#            n         p    exact Wald    Wilson    Clopper-Pearson    Jeffreys    simulated Wald
#             30    0.5406        0.9333    0.9574             0.9574      0.9574            0.9331
#            100    0.5406        0.9439    0.9439             0.9654      0.9439            0.9442
#            250    0.5406        0.9510    0.9510             0.9577      0.9510            0.9511
#            500    0.5406        0.9463    0.9517             0.9566      0.9517            0.9460
#           1000    0.5406        0.9509    0.9509             0.9509      0.9509            0.9514
#           6158    0.5406        0.9496    0.9496             0.9525      0.9496            0.9494
#             30    0.0100        0.2601    0.9639             0.9967      0.9639            0.2599
#            100    0.0100        0.6334    0.9206             0.9816      0.9816            0.6317
#            250    0.0100        0.9149    0.9588             0.9863      0.8778            0.9152
#            500    0.0100        0.8714    0.9623             0.9802      0.9291            0.8711
#           1000    0.0100        0.9270    0.9635             0.9761      0.9449            0.9269
#           6158    0.0100        0.9392    0.9457             0.9532      0.9532            0.9394
```

The header line reproduces the lesson exactly. The Wald interval on $3{,}329$ wins out of $6{,}158$ is $[0.5282,\ 0.5530]$ and the flat-prior interval is $[0.5281,\ 0.5530]$, agreeing to the third decimal as the lesson says. The last column exists to license the first: simulated Wald coverage matches the exact computation to three decimals at every row, so the other columns are sums over the binomial law rather than Monte Carlo estimates of it.

The top block says why the agreement happened. At $p=0.5406$ the coverage is near nominal at every sample size — $0.9333$, $0.9439$, $0.9510$, $0.9463$, $0.9509$, $0.9496$ — because a proportion near one-half on thousands of observations is the single most favourable case a Wald interval has. **The lesson's agreement is a fact about $p\approx\tfrac12$ and $n\approx6000$, not a certificate on the method**, and it is already visibly non-monotone in $n$: coverage falls from $0.9510$ at $n=250$ to $0.9463$ at $n=500$ and rises again.

The bottom block is what the same formula does on a rarer event, which is what a tail probability, a default rate or a stop-out frequency actually is. At $p=0.01$ the exact Wald coverage reads $0.2601$, $0.6334$, $0.9149$, $\mathbf{0.8714}$, $0.9270$, $0.9392$ — it improves, then *falls by more than four points when the sample doubles from two hundred and fifty to five hundred*, then improves again, and never reaches nominal even at six thousand observations. Wilson holds $0.9206$ to $0.9639$ throughout and Clopper–Pearson is conservative at $0.9761$ and above. **More data made the interval worse, the effect is exact rather than sampling noise, and the repair is a quadratic solve that has been in the literature since 1927.**

## The Scale an Interval Is Built On Decides Its Coverage, So There Is No Interval for a Parameter

The Wald construction has a second degree of freedom nobody records: which function of the parameter to build the interval on. Because $g(\hat\theta)\pm z|g'|\hat{\mathrm{se}}$ and $g(\hat\theta\pm z\hat{\mathrm{se}})$ are different intervals, the answer depends on whether you work in a variance or a volatility, a correlation or its Fisher transform, an odds or a log-odds.

??? note "Proof that the delta-method interval for a transform is not the transform of the interval, and that variance stabilization is the choice of the scale on which the pivot is closest to its limit"
    Let $\hat\theta$ be asymptotically normal with standard error $\mathrm{se}$ and let $g$ be smooth. [The Delta Method](../part-07-asymptotic-theory/04-delta-method.md) gives $g(\hat\theta)\approx\mathcal{N}\big(g(\theta),\ g'(\theta)^{2}\mathrm{se}^{2}\big)$, so the interval built on the transformed scale is $g(\hat\theta)\pm z|g'(\hat\theta)|\hat{\mathrm{se}}$. Transforming the untransformed interval instead gives the endpoints $g(\hat\theta\pm z\hat{\mathrm{se}})$, and a Taylor expansion of those endpoints is

    $$g(\hat\theta)\pm z\,g'(\hat\theta)\hat{\mathrm{se}}+\tfrac12 g''(\hat\theta)z^{2}\hat{\mathrm{se}}^{2}+\cdots,$$

    so the two agree to first order and differ at second order by a term proportional to $g''$. That second-order term is not a nuisance: it is exactly the asymmetry the normal approximation cannot represent, so the transformed interval is asymmetric about the point estimate precisely when the sampling distribution is skewed.

    This is what **variance stabilization** exploits. Choose $g$ with $g'(\theta)\propto1/\mathrm{se}(\theta)$ and the transformed estimator has constant variance, making $(g(\hat\theta)-g(\theta))$ a pivot to a better order. For a correlation this gives Fisher's $z=\tfrac12\log\frac{1+\rho}{1-\rho}$ with variance $1/(n-3)$ free of $\rho$; for a Poisson rate it gives the square root; for a volatility it gives the logarithm, since $\mathrm{se}(\hat\sigma)\propto\sigma$. In every case the improvement comes from picking the coordinate in which the normal approximation is least strained, and the transformation is chosen for that and nothing else.

    The load-bearing quantity is $g''$, the curvature that separates the two constructions and vanishes only when $g$ is affine. **There is no interval for a parameter; there is an interval for a parameterization, and the good ones are picked so that the pivot is closest to normal** — which means the "same" nominal level can mean materially different coverage depending on a choice that is usually made by whichever quantity the code happened to store.

The case that matters most is a volatility, because [Sampling Distributions](../part-10-statistics-foundations/03-sampling-distributions.md) already measured what happens when the standard interval meets a realistic tail.

```python
import numpy as np
from scipy.stats import chi2, norm, t as tdist

rng = np.random.default_rng(11073)
reps, z = 40_000, norm.ppf(0.975)

print("  law      n    chi2 interval    log-scale    robust log    median width: chi2    robust")
for law in ("normal", "t(5)", "t(3.4)"):
    for n in (63, 252, 1260):
        if law == "normal":
            x = rng.standard_normal((reps, n))
        else:
            v = float(law[2:-1])
            x = tdist.rvs(v, size=(reps, n), random_state=rng) / np.sqrt(v / (v - 2))
        d = x - x.mean(axis=1, keepdims=True)
        s2 = (d ** 2).sum(axis=1) / (n - 1)
        a = np.sqrt((n - 1) * s2 / chi2.ppf(0.975, n - 1))     # normal-theory interval for sigma
        b = np.sqrt((n - 1) * s2 / chi2.ppf(0.025, n - 1))
        g = np.log(np.sqrt(s2))
        c, e = np.exp(g - z / np.sqrt(2 * n)), np.exp(g + z / np.sqrt(2 * n))
        k = (d ** 4).mean(axis=1) / ((d ** 2).mean(axis=1) ** 2) - 3
        h = z * np.sqrt(np.maximum(k + 2, 0.1) / (4 * n))
        f, i = np.exp(g - h), np.exp(g + h)
        print(f"  {law:<8} {n:4d} {((a <= 1) & (1 <= b)).mean():14.4f}"
              f" {((c <= 1) & (1 <= e)).mean():12.4f} {((f <= 1) & (1 <= i)).mean():13.4f}"
              f" {np.median(b - a):21.4f} {np.median(i - f):9.4f}")
# =>   law      n    chi2 interval    log-scale    robust log    median width: chi2    robust
#      normal     63         0.9505       0.9459        0.9304                0.3605    0.3321
#      normal    252         0.9495       0.9480        0.9437                0.1759    0.1721
#      normal   1260         0.9513       0.9510        0.9497                0.0782    0.0779
#      t(5)       63         0.7996       0.7822        0.8750                0.3514    0.4186
#      t(5)      252         0.7614       0.7560        0.9141                0.1743    0.2461
#      t(5)     1260         0.7344       0.7333        0.9328                0.0779    0.1236
#      t(3.4)     63         0.6391       0.6105        0.7927                0.3362    0.4687
#      t(3.4)    252         0.5447       0.5352        0.8299                0.1693    0.3082
#      t(3.4)   1260         0.4592       0.4558        0.8630                0.0768    0.1784
```

The normal rows are the control and all three constructions sit at nominal, $0.9505$ down to $0.9304$ at the shortest window and within a few thousandths of each other by $n=1260$. Under the model they were derived for, the choice of scale is a matter of taste.

The $t$ rows are the failure and they reproduce the landmark. At $t(3.4)$ and $n=1260$ the $\chi^{2}$ interval covers $0.4592$, against the $0.4566$ that [Sampling Distributions](../part-10-statistics-foundations/03-sampling-distributions.md) published for the same construction under a realistic tail — a nominal ninety-five percent statement that is right slightly less than half the time. And the coverage *falls* as the sample grows, $0.6391$ to $0.5447$ to $0.4592$, because the interval's width shrinks like $1/\sqrt n$ while the estimator's true dispersion shrinks more slowly under a tail whose fourth moment is barely there. Building the same interval on the log scale changes nothing — $0.4558$ against $0.4592$ — which is the point: the log transform fixes skewness and does not fix a wrong standard error.

The robust column is the repair and the width columns are its price. Using $\sqrt{(\hat\kappa+2)/4n}$ as the standard error of $\log\hat\sigma$ in place of $\sqrt{1/2n}$ lifts coverage to $0.9328$ at $t(5)$ and $0.8630$ at $t(3.4)$, and unlike the others it *improves* with $n$ rather than deteriorating. It costs width: $0.1784$ against $0.0768$ at the last row, so an honest interval on that data is two and a third times wider than the one every risk report prints. **The narrow interval was not more precise; it was reporting a precision it did not have, and the entire difference is one estimated fourth moment.**

!!! warning "An interval computed on the statistic that was selected, on the scale that was convenient, at the sample size that happened to be available has three independent coverage defects and reports none of them"
    The three compound and none is visible in the output. Selection: if the statistic was chosen because it looked good among many, its interval is centred on a maximum and its coverage is far below nominal for reasons [Part XV](../part-15-multiple-testing/index.md) exists to quantify. Scale: the previous table moves coverage from $0.4592$ to $0.8630$ on identical data by changing a standard error nobody inspects. Sample size: the third section moves it from $0.9149$ to $0.8714$ by *adding* observations. A report showing an estimate and a bracket carries no trace of any of the three, and the bracket looks the same whether its true coverage is $0.95$ or $0.26$. The free diagnostic is to make the procedure grade itself: **draw ten thousand samples of your own $n$ from a law whose parameter you set to your own point estimate, run the identical interval code on each, and count how often the interval you built contains the value you set — if the count is not near nominal, the number in brackets is a label rather than a coverage.** It costs four lines, it needs no theory, and it catches all three defects at once, because the simulation inherits the same statistic, the same scale and the same $n$.

## A Credible Interval Answers the Question People Ask and a Confidence Interval Answers the One That Was Posed

The sentence the first section forbade — a ninety-five percent chance that the parameter lies in this interval — is available, at a price. A **credible interval** is a set containing $1-\alpha$ of the posterior mass, and because a posterior is a distribution over $\theta$, the probability statement is about $\theta$ and is exactly what people want. What it is not is a coverage guarantee.

```python
import numpy as np
from scipy.stats import beta, binom, norm

rng = np.random.default_rng(11077)
z, reps = norm.ppf(0.975), 200_000

def cred(k, n, a, b):
    return beta.ppf(0.025, a + k, b + n - k), beta.ppf(0.975, a + k, b + n - k)

print("  every credible interval below is a correct probability statement about theta")
print("        n         p    flat Beta(1,1)    skeptical Beta(50,50)    simulated Beta(50,50)"
      "    Wald    posterior mass inside the Wald interval")
for n, p in ((6158, 0.5406), (1000, 0.5406), (30, 0.5406), (250, 0.01), (100, 0.05)):
    k = np.arange(n + 1)
    pm = binom.pmf(k, n, p)
    cov = []
    for a, b in ((1, 1), (50, 50)):
        lo, hi = cred(k, n, a, b)
        cov.append(float(pm[(lo <= p) & (p <= hi)].sum()))
    s = rng.binomial(n, p, reps)
    lo, hi = cred(s, n, 50, 50)
    ph = k / n
    h = z * np.sqrt(ph * (1 - ph) / n)
    w = float(pm[(ph - h <= p) & (p <= ph + h)].sum())
    inside = beta.cdf(ph + h, 1 + k, 1 + n - k) - beta.cdf(ph - h, 1 + k, 1 + n - k)
    print(f"  {n:9d} {p:9.4f} {cov[0]:17.4f} {cov[1]:24.4f}"
          f" {((lo <= p) & (p <= hi)).mean():24.4f} {w:7.4f} {float((pm * inside).sum()):42.4f}")
# =>   every credible interval below is a correct probability statement about theta
#            n         p    flat Beta(1,1)    skeptical Beta(50,50)    simulated Beta(50,50)    Wald    posterior mass inside the Wald interval
#           6158    0.5406            0.9496                   0.9513                   0.9502  0.9496                                     0.9501
#           1000    0.5406            0.9509                   0.9514                   0.9524  0.9509                                     0.9504
#             30    0.5406            0.9574                   0.9933                   0.9934  0.9333                                     0.9608
#            250    0.0100            0.9588                   0.0000                   0.0000  0.9149                                     0.7933
#            100    0.0500            0.9659                   0.0000                   0.0000  0.8775                                     0.9014
```

The first two rows are the lesson's situation and everything agrees. At $n=6158$ the flat-prior credible interval covers $0.9496$, the skeptical one $0.9513$ and the Wald one $0.9496$, and the posterior mass sitting inside the Wald interval is $0.9501$. With thousands of observations near $p=\tfrac12$ the two frameworks are numerically interchangeable, which is exactly what the lesson claims and no more.

The last two rows are where the frameworks separate and the separation is total. A $\mathrm{Beta}(50,50)$ prior is a genuine belief — it says the rate is probably near one-half — and the credible interval it produces is a correct statement about the posterior at every row of this table. Its frequentist coverage when the truth is $p=0.01$ is $0.0000$: the interval never contains the parameter, not rarely, never, because the prior is worth a hundred observations and pulls the interval away from a truth it considers implausible. The simulated column confirms the exact computation to four decimals. **Both sentences are true simultaneously — the probability statement about $\theta$ is right and the interval is never right — and they are true of the same numbers.**

The flat-prior column is the useful middle. It covers $0.9588$ and $0.9659$ where the Wald interval covers $0.9149$ and $0.8775$, so on the case a Wald interval handles worst, a flat-prior credible interval is the better *frequentist* procedure — a fact with no Bayesian content, since a $\mathrm{Beta}(1,1)$ posterior interval is simply a well-behaved inversion that happens to have been derived by a different route. And the last column shows the correspondence dissolving as it goes: the posterior mass inside the Wald interval falls to $0.7933$, so at $(250,\ 0.01)$ the two constructions are not approximating each other in any sense.

!!! note "A confidence interval, a credible interval, a prediction interval and a tolerance interval are four different objects that print as two numbers in square brackets, and only one of them is about the next observation"
    A **confidence interval** is a random set covering a fixed parameter with a stated long-run frequency; the randomness is in the endpoints. A **credible interval** is a fixed set containing a stated fraction of a posterior's mass; the randomness is in the parameter, and the table above shows the two can disagree completely. A **prediction interval** covers a *future observation* rather than a parameter, so its width does not shrink to zero as $n$ grows — it retains the irreducible dispersion of the thing being predicted, which is why a ninety-five percent prediction interval for tomorrow's return is enormously wider than a ninety-five percent confidence interval for the mean return, and why quoting the second when a risk manager asked for the first understates the range by a factor of $\sqrt n$. A **tolerance interval** covers a stated proportion of the population with a stated confidence, so it has two levels rather than one. The operational tell is what happens as data accumulates: a confidence and a credible interval shrink toward a point, a prediction interval converges to a fixed positive width, and a tolerance interval converges to a population quantile. **If somebody's interval gets arbitrarily narrow with more data, it is not answering a question about what might happen next.**

## A Sentence About Repetition, Read as a Sentence About This Number

This page established that coverage is an unconditional property of a procedure, provable with a two-point example where the data reveals whether the interval is right while the stated level stays valid; that every interval is a pivot inverted and the Wald interval is the one construction that inverts an estimate of a pivot instead, which is why it collapses when $\hat{\mathrm{se}}$ is degenerate; that the exact coverage of a proportion interval is a step function in both arguments, reading $0.9496$ at the lesson's six thousand observations and $0.8714$ at five hundred with a rarer event, worse than at two hundred and fifty; that the scale an interval is built on decides its coverage, with a $\chi^{2}$ volatility interval falling to $0.4592$ under a realistic tail while a robust log interval holds $0.8630$ at two and a third times the width; and that a credible interval can be a perfectly correct probability statement about the parameter with a frequentist coverage of exactly zero.

One idea runs through all four failures and it is the idea the first section proved. An interval is a machine, its stated level is a property of the machine's output over inputs it did not receive, and every way of building the machine — the pivot, the scale, the standard error, the prior — is a modelling decision that changes the level without changing the printed brackets. The brackets look identical at $0.95$ coverage and at $0.00$. That is why the diagnostic in the warning is not a refinement but the minimum: a procedure that cannot be made to grade itself against a truth somebody set has not been checked at all, and running it costs less than reading this page.

There is a construction that takes that diagnostic seriously enough to build the interval out of it. Rather than deriving a pivot analytically and hoping the derivation survives the data, resample the data, watch the statistic move, and read the interval off the movement. That trades an assumption about the sampling distribution for a computation, and it comes with an accuracy ordering that is derivable in advance — including the discovery that the construction the theory ranks first is broken by the only standard error the industry has for the statistic every desk reports. That is [Bootstrap Confidence Intervals](08-bootstrap-confidence-intervals.md).

**Ninety-five percent is a property of the factory, and the interval on your screen came off the line without a serial number.**
