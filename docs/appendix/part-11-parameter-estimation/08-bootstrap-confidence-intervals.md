# Bootstrap Confidence Intervals

The bootstrap replaces an analytical pivot with a computation, and the replacement is usually presented as a menu — percentile, basic, studentized, bias-corrected — from which a practitioner picks by taste or by whichever one the library defaults to. It is not a menu. The constructions differ by the power of $n$ in their coverage error, the ordering is derivable before a single resample is drawn, and it says which one to reach for and why. What the ordering does not survive is contact with the standard errors that actually exist for the statistics a desk reports. The construction the theory ranks first needs a consistent standard error inside the resampling loop, and for the single most-reported statistic in this course the standard error the industry uses is not good enough — so the theoretically best interval is measurably the worst one on the data that motivated it.

This page covers the accuracy ordering and what it is an ordering of, the studentized interval and its dependence on the standard error it divides by, the percentile interval's exactness on an unknown scale together with BCa's two corrections, the dependence a resampling scheme preserves and the fact that which dependence matters is a property of the statistic rather than of the data, and the separation between Monte Carlo error and sampling error. It proves no plug-in theorem and builds no resampling scheme from scratch, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it develops no leave-one-out estimator in its own right, which is [Jackknife Methods](../part-09-monte-carlo-methods/08-jackknife-methods.md); it inverts no analytic pivot, which is [Confidence Intervals](07-confidence-intervals.md); it derives no standard error by linearization, which is [The Delta Method](../part-07-asymptotic-theory/04-delta-method.md); it resamples nothing under a null and reports no $p$-value, which are [Permutation Tests](../part-12-hypothesis-testing/09-permutation-tests.md) and [Bootstrap Tests](../part-12-hypothesis-testing/10-bootstrap-tests.md); it reduces no Monte Carlo variance, which is [Variance Reduction](../part-09-monte-carlo-methods/06-variance-reduction.md); it corrects no interval for the family of strategies that produced the statistic, which is [Part XV](../part-15-multiple-testing/index.md); and it never manufactures a tail the sample did not contain.

The trading stake is an agreement the course reports and reads correctly, and a disagreement it never had to face. [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) computes `Sharpe 0.30, iid bootstrap 95% CI [-0.09, 0.71]` and `BCa 95% CI [-0.10, 0.70]` against Lo's analytic $[-0.09,\ 0.70]$, and concludes: "When a formula exists, the bootstrap reproduces it — that agreement is the sanity check, and the bootstrap's value is everywhere the formula does not exist." The first two sections price that. Three constructions agreeing to two decimals is a diagnosis of the statistic rather than a certificate on the method, and the one that should win when they disagree loses on exactly this statistic, because it is built on exactly that formula.

## Six Constructions, One Ordering, and the Ordering Is a Statement About the Power of $n$ in the Coverage Error

Every bootstrap interval is built from the same object: the empirical distribution of $\hat\theta^{\ast}$ across resamples. They differ in what they do with it, and the differences fall into a strict hierarchy.

The **normal interval** is $\hat\theta\pm z\,\mathrm{sd}(\hat\theta^{\ast})$, which uses the resampling only to estimate a standard error and then assumes normality anyway. The **basic** interval reflects the bootstrap distribution about the estimate, $[2\hat\theta-\hat\theta^{\ast}_{(1-\alpha/2)},\ 2\hat\theta-\hat\theta^{\ast}_{(\alpha/2)}]$, on the theory that the law of $\hat\theta^{\ast}-\hat\theta$ approximates that of $\hat\theta-\theta$. The **percentile** interval reads the resampled quantiles directly. **BCa** shifts and stretches those quantiles by two estimated constants. The **studentized** interval resamples a pivot rather than the estimate itself. And a **calibrated** interval adjusts the nominal level until the realized coverage is right.

Coverage error is the gap between nominal and actual, and the ordering is in its rate: normal and basic are $O(n^{-1/2})$, percentile is $O(n^{-1/2})$ but **transformation-respecting**, BCa and studentized are $O(n^{-1})$, and a double bootstrap is $O(n^{-3/2})$. Doubling the accuracy order is worth far more than any refinement inside an order, which is why the hierarchy is worth knowing rather than experimenting with. [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) races the first four and finds them agreeing at long samples and separating at short ones; this page starts where the second-order constructions enter.

## The Studentized Bootstrap Is Second-Order Accurate Because It Resamples a Pivot, and Only as Accurate as the Standard Error It Divides By

The studentized construction resamples $T^{\ast}=(\hat\theta^{\ast}-\hat\theta)/\hat{\mathrm{se}}^{\ast}$ rather than $\hat\theta^{\ast}$, uses its quantiles as the quantiles of $T=(\hat\theta-\theta)/\hat{\mathrm{se}}$, and inverts. The extra order of accuracy is bought at the price of computing a standard error inside every resample.

??? note "Proof that a studentized bootstrap interval has coverage error $O(n^{-1})$ where a percentile interval has $O(n^{-1/2})$, and that the extra order is bought by resampling a quantity whose law is free of the parameter"
    For a smooth statistic the sampling law of the studentized quantity admits an Edgeworth expansion

    $$\mathbf{P}(T\le x)=\Phi(x)+n^{-1/2}q_1(x)\varphi(x)+O(n^{-1}),$$

    where $q_1$ is a polynomial whose coefficients involve the skewness of the underlying influence function. The bootstrap world has the same expansion with $\hat q_1$ in place of $q_1$, and $\hat q_1=q_1+O_p(n^{-1/2})$ because it is a smooth function of sample moments. Subtracting,

    $$\mathbf{P}^{\ast}(T^{\ast}\le x)-\mathbf{P}(T\le x)=n^{-1/2}\big(\hat q_1(x)-q_1(x)\big)\varphi(x)+O(n^{-1})=O_p(n^{-1}),$$

    so the $n^{-1/2}$ terms cancel and the studentized interval inherits an $O(n^{-1})$ coverage error.

    For the un-studentized quantity $\hat\theta^{\ast}-\hat\theta$ the corresponding polynomial depends on the parameter through the scale, so the leading $n^{-1/2}$ term does *not* cancel and the percentile interval retains $O(n^{-1/2})$. The distinction is exactly the pivotality of [Confidence Intervals](07-confidence-intervals.md): the studentized quantity is asymptotically a pivot and the raw one is not.

    The cancellation has a precondition that the notation hides. It requires $\hat{\mathrm{se}}^{\ast}$ to be a consistent estimate of the resample's standard error, computed on each resample, with the same functional form as $\hat{\mathrm{se}}$. If $\hat{\mathrm{se}}$ is itself derived under an assumption the data violates, then $\hat q_1$ is estimating the wrong polynomial and the cancellation does not occur.

    The load-bearing quantity is $q_1$, the skewness polynomial, together with the fact that it is the *same* polynomial in the bootstrap world as in the real one. **Second-order accuracy costs one extra standard-error computation inside the resampling loop and is the highest-return line of code in this part — provided the standard error it divides by is consistent, which for a Sharpe ratio under realistic tails it is not.**

The Sharpe ratio is the natural test, because it is both the statistic the course reports and the one with a famous analytic standard error to divide by.

```python
import numpy as np
from scipy.special import ndtri
from scipy.stats import norm, t as tdist

rng = np.random.default_rng(11081)
per, sr, B = 252, 0.30, 600                                    # the course's momentum Sharpe

def sharpe(x, ax=-1):
    return np.sqrt(per) * x.mean(axis=ax) / x.std(axis=ax, ddof=1)

def lo_se(x, ax=-1):                                           # Lo's analytic standard error
    return np.sqrt(per * (1 + sharpe(x, ax) ** 2 / (2 * per)) / x.shape[ax])

def jack_se(x):
    n = x.shape[-1]
    s, q = x.sum(-1, keepdims=True), (x ** 2).sum(-1, keepdims=True)
    m = (s - x) / (n - 1)
    loo = np.sqrt(per) * m / np.sqrt((q - x ** 2 - (n - 1) * m ** 2) / (n - 2))
    return np.sqrt((n - 1) / n * ((loo - loo.mean(-1, keepdims=True)) ** 2).sum(-1)), loo

print("     law      n    percentile    BCa    stud (Lo se)    stud (jackknife se)    Lo analytic"
      "    median width")
for nu in (8.0, 2.6):
    for n, reps in ((252, 400), (1512, 300)):
        hit = np.zeros(5)
        wid = []
        for _ in range(reps):
            x = (0.195 / np.sqrt(per)) * tdist.rvs(nu, size=n, random_state=rng) \
                / np.sqrt(nu / (nu - 2)) + sr * 0.195 / per
            hat, se = sharpe(x), lo_se(x)
            sj, loo = jack_se(x)
            xs = x[rng.integers(n, size=(B, n))]
            st = sharpe(xs, 1)
            t1, t2 = (st - hat) / lo_se(xs, 1), (st - hat) / jack_se(xs)[0]
            d = loo.mean() - loo
            a = (d ** 3).sum() / (6 * (d ** 2).sum() ** 1.5)
            z0 = ndtri(np.clip((st < hat).mean(), 1e-6, 1 - 1e-6))
            ql, qh = (norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))) for z in (-1.96, 1.96))
            ci = [np.quantile(st, [0.025, 0.975]), np.quantile(st, [ql, qh]),
                  [hat - np.quantile(t1, 0.975) * se, hat - np.quantile(t1, 0.025) * se],
                  [hat - np.quantile(t2, 0.975) * sj, hat - np.quantile(t2, 0.025) * sj],
                  [hat - 1.96 * se, hat + 1.96 * se]]
            hit += [lo <= sr <= hi for lo, hi in ci]
            wid.append(ci[4][1] - ci[4][0])
        h = hit / reps
        print(f"  t({nu:.1f}) {n:6d} {h[0]:13.3f} {h[1]:6.3f} {h[2]:15.3f} {h[3]:22.3f}"
              f" {h[4]:14.3f} {np.median(wid):15.3f}")
# =>      law      n    percentile    BCa    stud (Lo se)    stud (jackknife se)    Lo analytic    median width
#      t(8.0)    252         0.960  0.970           0.955                  0.960          0.970           3.922
#      t(8.0)   1512         0.957  0.950           0.947                  0.947          0.950           1.601
#      t(2.6)    252         0.943  0.955           0.905                  0.940          0.948           3.922
#      t(2.6)   1512         0.930  0.930           0.907                  0.927          0.937           1.601
```

The $t(8)$ rows are the course's situation and they say why the agreement it reports was inevitable. Every construction sits between $0.947$ and $0.970$, and at $n=1512$ they span $0.947$ to $0.950$ — a range of three thousandths. **When five constructions agree, the finding is that the statistic is nearly pivotal on the raw scale, and no construction has been validated by the agreement.** The lesson's reading is exactly right: the agreement is a sanity check on the arithmetic, not evidence about the method.

The $t(2.6)$ rows are the tail the course itself fits to daily returns, and they separate the constructions in the order the theory says — with one inversion that matters. At $n=252$ the studentized interval built on Lo's analytic standard error covers $0.905$, the worst of the five, against a percentile interval's $0.943$ and a BCa's $0.955$. At $n=1512$ it is still last at $0.907$. **The construction that is second-order accurate in theory is last in practice**, because Lo's formula assumes a fourth moment the fitted $\nu=2.6$ does not have, so the quantity being studentized is divided by an inconsistent scale and the Edgeworth cancellation never happens.

The repair is in the next column and it is one function call. Replacing Lo's formula with a jackknife standard error — the leave-one-out machinery of [Jackknife Methods](../part-09-monte-carlo-methods/08-jackknife-methods.md), computed inside the loop — lifts the same construction from $0.905$ to $0.940$ and from $0.907$ to $0.927$. The remaining gap is the second finding: at $n=1512$, six years of daily data, the best of the five constructions covers $0.937$ and none reaches $0.95$. **The taxonomy is real and it is second-order; the tail is a first-order problem, and no choice of construction repairs a sample that is short relative to how heavy its tail is.**

## The Percentile Interval Is Exact on a Scale Nobody Has to Find, and BCa's Two Constants Are the Two Ways That Scale Fails

The percentile interval looks naive — read off the $2.5$th and $97.5$th percentiles and stop — and its justification is the most interesting in the taxonomy, because it turns on a transformation that never has to be computed.

??? note "Proof that the percentile interval is exact whenever some monotone transformation makes the estimator normal and unbiased, and that BCa's two constants are the first two ways that assumption fails"
    Suppose there exists a monotone increasing $g$ such that $\hat\phi=g(\hat\theta)$ satisfies $\hat\phi\sim\mathcal{N}(\phi,\tau^{2})$ with $\phi=g(\theta)$ and $\tau$ constant. On that scale the exact interval is $\hat\phi\pm z\tau$, and because $g$ is monotone, applying $g^{-1}$ to the endpoints gives an exact interval for $\theta$. Now note that the bootstrap distribution of $\hat\phi^{\ast}$ is $\mathcal{N}(\hat\phi,\tau^{2})$, so its $\alpha/2$ and $1-\alpha/2$ quantiles are exactly $\hat\phi\mp z\tau$ — and since quantiles commute with monotone maps, the quantiles of $\hat\theta^{\ast}$ are exactly $g^{-1}(\hat\phi\mp z\tau)$. The percentile interval *is* the exact interval, and $g$ never appears in the computation.

    That is a strong result resting on a strong hypothesis, and BCa relaxes it in the two directions it usually fails. Allow a median bias, $\hat\phi\sim\mathcal{N}(\phi-z_0\tau(\phi),\ \tau(\phi)^{2})$, and allow the standard deviation to drift linearly with the parameter, $\tau(\phi)=1+a\phi$. Carrying those through gives adjusted quantile levels

    $$\alpha_{\text{lo}}=\Phi\Big(z_0+\frac{z_0-z}{1-a(z_0-z)}\Big),\qquad \alpha_{\text{hi}}=\Phi\Big(z_0+\frac{z_0+z}{1-a(z_0+z)}\Big),$$

    with $z_0$ estimated by the fraction of resamples below the point estimate and $a$ by the skewness of the jackknife influence values. Both are computed from quantities already available; neither requires knowing $g$.

    The load-bearing feature is that the transformation **never appears in the formula**. Every other construction on this page requires the analyst to pick a scale, and [Confidence Intervals](07-confidence-intervals.md) measured what that choice costs — a $\chi^{2}$ volatility interval covering $0.4592$ where a log-scale one on a robust standard error holds $0.8630$. **BCa's contribution is not accuracy for its own sake but the deletion of a modelling decision: the scale that changes coverage is chosen by the data instead of by whoever wrote the reporting code.**

That explains the table above. BCa scores $0.955$ and $0.930$ under the tail where the studentized construction on the wrong standard error scores $0.905$ and $0.907$, and it does so without being told anything about the statistic — the $z_0$ and $a$ it estimated absorbed the skewness that Lo's formula was supposed to handle and did not.

## An Interval Inherits the Dependence Its Resampling Scheme Preserves, and Which Dependence Matters Is a Property of the Statistic

Every construction above resamples observations independently, which asserts that the observations were independent. They are not. The standard repair is to resample contiguous blocks so that short-range dependence survives inside a block, and [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) measures the repair working on an autocorrelated level and — surprisingly — not being needed for a volatility-clustered one. That page closes the finding with a claim it does not measure: "For a statistic that does read the second moment — a volatility estimate, a drawdown, a Sharpe over a short window — the same clustering widens the interval substantially, and the choice of scheme starts to matter for the same series." This section measures it.

```python
import numpy as np

rng = np.random.default_rng(11083)
n, block, reps, b, mu = 500, 8, 600, 500, 0.05                 # block length is about n^(1/3)

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

stats = (("mean", lambda a: a.mean(-1), mu),
         ("std dev", lambda a: a.std(-1, ddof=1), 1.0),
         ("Sharpe", lambda a: a.mean(-1) / a.std(-1, ddof=1), mu))
print("  process    statistic    iid cover    moving block    stationary    width iid    width block")
for name, gen in (("iid", iid), ("GARCH", garch), ("AR(1) 0.5", ar1)):
    hit = np.zeros((3, 3))
    wid = np.zeros((3, 2))
    for _ in range(reps):
        x = gen(n)
        draws = [x[f()] for f in (idx_iid, idx_block, idx_stat)]
        for i, (_, fn, truth) in enumerate(stats):
            for j, d in enumerate(draws):
                lo, hi = np.quantile(fn(d), [0.025, 0.975])
                hit[i, j] += lo <= truth <= hi
                if j < 2:
                    wid[i, j] += hi - lo
    for i, (sname, _, _) in enumerate(stats):
        h, w = hit[i] / reps, wid[i] / reps
        print(f"  {name:<10} {sname:<12} {h[0]:9.3f} {h[1]:15.3f} {h[2]:13.3f}"
              f" {w[0]:12.4f} {w[1]:14.4f}")
# =>   process    statistic    iid cover    moving block    stationary    width iid    width block
#      iid        mean             0.950           0.952         0.950       0.1739         0.1717
#      iid        std dev          0.932           0.918         0.917       0.1220         0.1205
#      iid        Sharpe           0.953           0.950         0.948       0.1751         0.1729
#      GARCH      mean             0.952           0.937         0.935       0.1736         0.1720
#      GARCH      std dev          0.537           0.653         0.712       0.1355         0.1850
#      GARCH      Sharpe           0.935           0.930         0.935       0.1748         0.1730
#      AR(1) 0.5  mean             0.748           0.923         0.913       0.1737         0.2720
#      AR(1) 0.5  std dev          0.868           0.927         0.928       0.1221         0.1505
#      AR(1) 0.5  Sharpe           0.750           0.925         0.915       0.1748         0.2739
```

The mean rows reproduce the earlier page on a different seed and are the control. Under an autocorrelated level the independent bootstrap covers $0.748$ against that page's $0.755$, both block schemes repair it to $0.923$ and $0.913$ against its $0.920$ and $0.926$, and the block width is $0.2720$ against $0.1737$, matching its $0.2695$ against $0.1733$. Under GARCH the independent interval covers $0.952$ and the block interval is a hair *narrower*, $0.1720$ against $0.1736$ — the result that page reports and explains.

The `GARCH / std dev` row is the promise discharged and it is not a marginal effect. The independent bootstrap covers **$0.537$**, barely half its nominal level, on a series whose *level* has no autocorrelation at all. The block schemes lift it to $0.653$ and $0.712$ and widen the interval from $0.1355$ to $0.1850$, thirty-six percent wider — the substantial widening the earlier page asserted, in the direction it predicted, on the same process where the mean saw nothing. **The choice of resampling scheme is not a property of the data; it is a property of the pairing of a statistic with the data, and the same series demands independent resampling for one column of a report and block resampling for the next.**

The Sharpe rows carry the surprise. Under GARCH the Sharpe's independent interval covers $0.935$ — nearly nominal, and far better than the standard deviation's $0.537$ — because a Sharpe divides one quantity the clustering inflates by another, and the inflation partly cancels. Under an autocorrelated level, where the numerator is the whole story, the Sharpe fails exactly as the mean does, $0.750$ against $0.748$. **A ratio can be immune to a dependence that destroys both of its parts**, so a rule of thumb about which statistics need block resampling is not available; the pairing has to be measured.

!!! warning "Four bootstrap intervals agreeing to two decimal places have told you the statistic is nearly normal on the raw scale and have told you nothing whatever about whether the history is long enough"
    Agreement across constructions is the most persuasive-looking diagnostic in the toolkit and it tests only the shape of one distribution, not the adequacy of the sample it was computed from. The first table is the demonstration: at $t(8)$ five constructions land within three thousandths of each other and all of them are right, while at $t(2.6)$ they land within three thousandths of each other at $n=1512$ and all of them are wrong, covering between $0.907$ and $0.937$ against a nominal $0.95$. Every construction is resampling the same history, so a feature that history does not contain — a regime, a liquidity event, a correlation break — is absent from all of them identically, and their agreement is a measure of that shared blindness. The free diagnostic addresses what agreement cannot: **run the identical interval code on the first half of the history and on the second half and compare the two intervals to each other, and if they overlap by less than half their width, the full-sample interval is describing a parameter that did not stay still.** No construction in the taxonomy repairs that, because every one of them assumes a single fixed $\theta$ generated the whole sample — an assumption [Population vs Sample](../part-10-statistics-foundations/01-population-vs-sample.md) argues is the one the data can never check.

## More Resamples Buy Monte Carlo Precision and No Coverage At All

Two distinct errors sit inside every number above and the practical failure is treating them as one. The **Monte Carlo error** comes from using $B$ resamples instead of all $n^{n}$ of them; it is a property of the computation and it vanishes as $B$ grows. The **statistical error** comes from having $n$ observations instead of the population; it is a property of the data and $B$ does not touch it.

```python
import numpy as np
from scipy.stats import t as tdist

rng = np.random.default_rng(11087)
per, sr, n, reps, runs = 252, 0.30, 250, 300, 4                # four bootstrap runs per sample

def sharpe(x, ax=-1):
    return np.sqrt(per) * x.mean(axis=ax) / x.std(axis=ax, ddof=1)

grid = (50, 200, 800, 3200)
mc = {B: [] for B in grid}
across = {B: [] for B in grid}
hit = {B: 0 for B in grid}
wid = {B: [] for B in grid}
for _ in range(reps):
    x = (0.195 / np.sqrt(per)) * tdist.rvs(2.6, size=n, random_state=rng) / np.sqrt(2.6 / 0.6) \
        + sr * 0.195 / per
    for B in grid:
        ends = np.array([np.quantile(sharpe(x[rng.integers(n, size=(B, n))], 1), [0.025, 0.975])
                         for _ in range(runs)])
        mc[B].append(ends[:, 0].std(ddof=1))
        across[B].append(ends[0, 0])
        hit[B] += ends[0, 0] <= sr <= ends[0, 1]
        wid[B].append(ends[0, 1] - ends[0, 0])

print(f"  percentile interval for a Sharpe, t(2.6), n = {n}, {reps} samples, {runs} runs each")
print("       B    coverage    median width    Monte Carlo sd of the lower end"
      "    sample-to-sample sd    ratio")
for B in grid:
    a, b = np.mean(mc[B]), np.std(across[B], ddof=1)
    print(f"  {B:6d} {hit[B] / reps:11.3f} {np.median(wid[B]):15.3f} {a:34.4f}"
          f" {b:22.4f} {a / b:8.4f}")
# =>   percentile interval for a Sharpe, t(2.6), n = 250, 300 samples, 4 runs each
#           B    coverage    median width    Monte Carlo sd of the lower end    sample-to-sample sd    ratio
#          50       0.927           3.530                             0.2835                 0.9741   0.2911
#         200       0.960           3.826                             0.1594                 0.9365   0.1702
#         800       0.960           3.868                             0.0850                 0.9429   0.0901
#        3200       0.970           3.885                             0.0406                 0.9354   0.0434
```

The fourth column is the Monte Carlo error, measured directly by running the same bootstrap four times on the *same* sample and taking the standard deviation of the endpoint it returns. It falls $0.2835$, $0.1594$, $0.0850$, $0.0406$ across a sixty-four-fold increase in $B$ — a factor of $6.98$ against the $\sqrt{64}=8$ the rate predicts, and it goes to zero with enough compute like any Monte Carlo average.

The fifth column is the statistical error, the standard deviation of that same endpoint *across different samples*, and it does not move: $0.9741$, $0.9365$, $0.9429$, $0.9354$. It is a property of having two hundred and fifty observations and no amount of resampling touches it. The ratio column is the whole message — Monte Carlo noise is $29\%$ of sampling noise at $B=50$ and $4\%$ at $B=3200$, so beyond a few hundred resamples the compute is buying digits that the data cannot support.

Coverage confirms it from the other side. It moves from $0.927$ to $0.960$ between $B=50$ and $B=200$, because fifty resamples cannot locate a $2.5$th percentile and the interval comes out too narrow — median width $3.530$ against $3.885$ — and then it *stops improving*, sitting at $0.960$, $0.960$, $0.970$ while $B$ rises sixteenfold. **Enough resamples to stabilize the quantiles is a few hundred, and every resample after that is a computation about the sample rather than about the world.**

!!! note "A bootstrap confidence interval and a bootstrap standard error are two products of one resampling loop, and $\hat\theta \pm 1.96\,\mathrm{sd}(\hat\theta^{\ast})$ is the first and least accurate member of the taxonomy rather than the bootstrap answer"
    The confusion is common enough to have a shape. A resampling loop produces a distribution, and there are two things to do with it: take its standard deviation and plug that into a normal interval, or read its quantiles. The first is the **normal bootstrap interval**, which discards everything the resampling learned about shape and keeps only the scale, and it sits at the bottom of the accuracy ordering with a coverage error of $O(n^{-1/2})$ and an explicit normality assumption the resampling was performed to avoid. Reporting "we bootstrapped the standard error" is therefore not a statement that a bootstrap interval was used; it is a statement that the least accurate one was. The second distinction is the one the fifth section measures: $B$ is a compute parameter controlling Monte Carlo error and $n$ is a data parameter controlling statistical error, and the usual reporting convention — quoting $B$ prominently and $n$ in a footnote — has them exactly backwards. A useful habit is to report both alongside the interval, as [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) does when it prints "(stationary bootstrap, 10,000 resamples, mean block 18d)", and to remember that only one of those three numbers is about the market.

## The Interval Is Only as Long as the History and the History Is Only as Long as It Is

This page established that the bootstrap interval constructions form a strict accuracy ordering in the power of $n$ rather than a menu; that the studentized interval's second-order accuracy requires a consistent standard error inside the loop, so on the course's own fitted tail it covers $0.905$ built on Lo's formula and $0.940$ built on a jackknife, last and mid-table for the identical construction; that the percentile interval is exact whenever some monotone transformation normalizes the estimator and BCa estimates the two ways that assumption fails without ever computing the transformation; that a GARCH series destroys the independent bootstrap's interval for a standard deviation, $0.537$ against a nominal $0.95$, while leaving its interval for a mean untouched at $0.952$, so the resampling scheme is chosen by the statistic rather than by the data; and that raising $B$ from $50$ to $3200$ cuts Monte Carlo error by a factor of seven and leaves coverage at $0.960$.

The ordering is real and the ordering is not the constraint. Every construction on this page is a different way of extracting a sampling distribution from one sample, and all of them are extracting it from the same sample. When the tail is heavy relative to $n$, they fail together — $0.907$ to $0.937$ at six years of daily data, with the spread between constructions an order of magnitude smaller than the gap to nominal. When the dependence is in a moment the statistic reads, they fail together until the resampling scheme is changed, which is a decision made outside the taxonomy. And when the parameter did not stay still across the history, they fail together with no diagnostic at all, because the assumption that one $\theta$ generated the sample is shared by every construction and by the analytic intervals of [Confidence Intervals](07-confidence-intervals.md) equally.

That is where this part ends, and it ends where [Point Estimation](01-point-estimation.md) began. Eight pages have built estimators and error bars for them, and every guarantee any of them carries — an information bound, an efficiency, a coverage, a posterior probability — is conditional on a description of the world that was asserted before the calculation and never checked afterward. The estimators are correct. The arithmetic reproduces. What the next part does is stop asking how precisely a number can be estimated and start asking whether it differs from something, which turns out to be the same machinery with the quantifiers rearranged and a new way to be wrong: [Part XII](../part-12-hypothesis-testing/index.md) inverts the intervals of these last two pages into decisions, and the decisions inherit every defect measured here plus one of their own.

**Every construction in the taxonomy is an argument about the shape of a distribution nobody has seen, and the sample is the only witness any of them can call.**
