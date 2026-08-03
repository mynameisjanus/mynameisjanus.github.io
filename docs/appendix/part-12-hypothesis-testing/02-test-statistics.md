# Test Statistics

The previous page treated a rejection region as primitive and then never built one, which is the honest order to do things in but leaves the whole construction hanging on a step nobody describes. In practice a region is built by reducing the sample to a single number and taking a tail of it, and the number is chosen from a short list of functions someone has already implemented. That choice is usually made on grounds of familiarity, and it decides something the analyst never intended to decide: which departures from the null the test is physically capable of noticing. A statistic that does not move when the world moves produces a test whose power equals its size, and the test reports nothing unusual, because from its point of view nothing is.

This page covers what qualifies a function of the data as a test statistic, the pivotal property that makes a critical value a constant rather than a function of things nobody knows, the monotone likelihood ratio condition that makes one statistic uniformly best against every one-sided alternative, the blindness a choice of statistic buys, and the three places a null distribution can come from. It does not define the rejection region, the size or the level, which is [The Hypothesis Testing Framework](01-hypothesis-testing-framework.md); it does not turn a statistic's tail area into a reported number, which is [p-values](03-p-values.md); it computes no power function against a named alternative, which is [Statistical Power](05-statistical-power.md); it proves neither the Neyman–Pearson lemma nor Wilks' theorem, which are [Likelihood Ratio Tests](06-likelihood-ratio-tests.md); it defines no sufficiency and proves no factorization, which is [Statistics and Sufficiency](../part-10-statistics-foundations/05-statistics-and-sufficiency.md); it builds no resampling scheme, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); and it never claims a statistic is neutral.

The trading stake is a verdict the course reaches after aiming four instruments at one pair of assets and getting four answers. [Time Series](../../part-03-statistics/03-time-series.md) reports `SPY-IVV: hedge ratio 1.002, EG stat -2.14, p = 0.46` — the most mechanically tethered pair available fails a cointegration test at a p-value statistically indistinguishable from `SPY-GLD: EG p = 0.49` — while the same spread shows `spread std 23 bp, AR(1) rho 0.82, half-life 3.4 days`, and [Mean Reversion and Pairs Trading](../../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md) adds `VR(5) 0.21 (p 0.00), Hurst 0.04, ADF p 0.000` on that same spread. The lesson's conclusion is that "tests see the horizon their lag structure selects, not *the* answer." Sections 3 and 4 explain why that is a theorem rather than a disappointment.

## A Test Statistic Needs a Computable Null Law and a Stochastic Ordering Under the Alternative, and Nothing Else

A **test statistic** is any function $T=T(X)$ of the data used to order samples by how much they embarrass the null. Two requirements make it usable and there are no others. First, the law of $T$ under the null must be computable, exactly or approximately or by simulation, because without it there is no critical value and the region cannot be drawn. Second, $T$ must be **stochastically larger under the alternative** — $\mathbf{P}_{\theta_1}(T>c)>\mathbf{P}_{\theta_0}(T>c)$ for the relevant $c$ — because a rejection region in the upper tail of $T$ detects a departure only if the departure pushes $T$ up.

The second requirement is the one that gets skipped, and skipping it is not punished by any error message. A statistic with a beautiful null law and no ordering under the alternative yields a perfectly valid test that is also perfectly useless: if $T$ has the *same* law under $\theta_0$ and $\theta_1$, then $\mathbf{P}_{\theta_1}(T>c)=\mathbf{P}_{\theta_0}(T>c)=\alpha$, so the power is identically the size, and the test rejects a true null and a false one at the same rate. Nothing in the output distinguishes that test from a good one. It reports a p-value, the p-value is uniform, and the uniformity is exactly what a well-behaved test produces under a true null.

Neither requirement mentions the data being summarized well. A test statistic is not a summary and need not resemble the quantity of interest — [Likelihood Ratio Tests](06-likelihood-ratio-tests.md) shows the optimal statistic for a simple pair of hypotheses is a ratio of densities, which is not an estimate of anything. Conversely a good estimator can make a bad test statistic, which is the content of the failure in section 3.

## A Pivot Has a Null Law Free of the Nuisance Parameters, Which Is Why Its Critical Value Is a Constant

Almost every null of practical interest is composite because it leaves a scale free. "The two books have the same mean return" says nothing about their volatility, and the previous page showed the size of a test is a supremum over everything the null leaves unspecified. A statistic is **pivotal** when its null law does not depend on those free parameters at all: then the supremum is over a set on which the rejection probability is constant, size equals level exactly, and one number from one table serves every desk.

??? note "Proof that dividing by an estimate of its own standard error frees a statistic's null law from the scale, exactly under normality and only asymptotically otherwise"

    Let $X_1,\dots,X_n$ be independent $\mathcal{N}(\mu,\sigma^2)$ and consider testing $\mu=0$ with $\sigma$ unknown. The raw statistic $\bar X$ has null law $\mathcal{N}(0,\sigma^2/n)$, whose quantiles are proportional to $\sigma$: a critical value computed at one $\sigma$ is wrong at another by the ratio of the two. Write $\bar X=\sigma\bar Z$ and $s=\sigma s_Z$ where $Z_i=X_i/\sigma$ is standard normal under the null. Then
    $$\frac{\bar X}{s/\sqrt n}=\frac{\sigma\bar Z}{\sigma s_Z/\sqrt n}=\frac{\bar Z}{s_Z/\sqrt n},$$
    and $\sigma$ has cancelled identically, not approximately. The right-hand side is a function of standard normals alone, so its law is the same for every $\sigma$ — it is $t_{n-1}$, by the independence of $\bar Z$ and $s_Z$ and the $\chi^2_{n-1}$ law of $(n-1)s_Z^2$ that [Chi-Square Distribution](../part-05-common-distributions/15-chi-square-distribution.md) supplies.

    The cancellation used two facts: that $\sigma$ enters the model as a pure scale, and that the denominator is an estimate of the numerator's standard deviation built from the *same* scale. Drop the first — let the family be anything other than a location-scale family — and the ratio no longer loses its dependence on the nuisance parameter exactly. What survives is asymptotic: $s\to\sigma$ in probability, so by [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md) the studentized statistic has the same limit law as the one dividing by the true $\sigma$, and that limit is free of the nuisance. Exactness becomes approximation, and the approximation's quality is a finite-sample question the limit theorem does not answer.

    The load-bearing quantity is the denominator's scale: it must carry the same $\sigma$ as the numerator for the ratio to be free of it. **Studentizing is not a variance correction, it is a change of coordinates chosen so the answer stops depending on the coordinate.**

What pivotality is worth can be measured by taking it away. A desk that calibrates a critical value once — on its own data, at its own volatility — and then applies the same number elsewhere is running the raw statistic, and the following block prices that habit. Both tests are calibrated once at a daily volatility of $1\%$ and then run at volatilities they were not calibrated on, with the mean difference true and equal to zero throughout, so every rejection is a false one:

```python
import numpy as np

rng = np.random.default_rng(12023)
n, reps = 63, 40_000
sd_cal = 0.010                                     # the volatility the desk calibrated on

def draw(sd):
    a, b = rng.normal(0, sd, (reps, n)), rng.normal(0, sd, (reps, n))
    raw = a.mean(1) - b.mean(1)
    pooled = np.sqrt((a.var(1, ddof=1) + b.var(1, ddof=1)) / n)
    return np.abs(raw), np.abs(raw / pooled)

raw_c, piv_c = (np.quantile(v, 0.95) for v in draw(sd_cal))
print(f"  critical values fixed once at sigma={sd_cal:.3f}: "
      f"raw {raw_c:.6f}, studentized {piv_c:.4f}")
print("  then the same two tests are run at volatilities the desk did not calibrate on")
print("     sigma   raw 95th pct   raw size   studentized 95th pct   studentized size")
for sd in (0.005, 0.010, 0.020, 0.040):
    r, p = draw(sd)
    print(f"    {sd:.3f}   {np.quantile(r, 0.95):12.6f}   {(r > raw_c).mean():8.4f}   "
          f"{np.quantile(p, 0.95):20.4f}   {(p > piv_c).mean():16.4f}")
# =>   critical values fixed once at sigma=0.010: raw 0.003526, studentized 1.9941
#      then the same two tests are run at volatilities the desk did not calibrate on
#         sigma   raw 95th pct   raw size   studentized 95th pct   studentized size
#        0.005       0.001746     0.0001                 1.9817             0.0484
#        0.010       0.003459     0.0461                 1.9667             0.0471
#        0.020       0.006975     0.3232                 1.9883             0.0494
#        0.040       0.013966     0.6184                 1.9807             0.0487
```

The raw statistic's own $95$th percentile moves from $0.001746$ to $0.013966$ across the four volatilities, a factor of eight, because it is a difference of means measured in the units of the thing being measured. Freezing one critical value across that range therefore does not produce a slightly miscalibrated test; it produces four different tests. At half the calibration volatility the raw test's size is $0.0001$ — it has effectively stopped rejecting, and a desk running it would conclude, correctly at that level and for entirely the wrong reason, that nothing is ever significant. At twice the calibration volatility the size is $0.3232$ and at four times it is $0.6184$: on data where the two means are *identical by construction*, a nominal $5\%$ test fires on nearly two-thirds of samples.

The studentized column is the control and its job is to be boring. Its $95$th percentile sits between $1.9667$ and $1.9883$ across a sixteen-fold range of variance, and its size stays within $0.0484$ to $0.0494$ of nominal everywhere. Nothing was corrected and no variance was adjusted; the statistic was divided by an estimate carrying the same units, and the parameter that wrecked the first column left the problem entirely.

**A critical value is a constant only for a pivotal statistic, and for every other statistic it is a function of something the null did not pin down.**

## A Monotone Likelihood Ratio Makes One Statistic Uniformly Most Powerful, and Without It the Natural Statistic Can Be Worthless

Pivotality makes a statistic *usable*. It says nothing about whether the statistic is *good*, and the condition that does is a property of the family rather than of the statistic. A family has **monotone likelihood ratio** in $T$ when, for every $\theta_1>\theta_0$, the ratio $p_{\theta_1}(x)/p_{\theta_0}(x)$ is a non-decreasing function of $T(x)$. Where it holds, one statistic is optimal against an entire one-sided alternative at once.

??? note "Proof that a family with monotone likelihood ratio in $T$ has a uniformly most powerful one-sided test rejecting for large $T$, and that no such test exists two-sided"

    Fix $\theta_1>\theta_0$ and consider the simple-versus-simple problem $\theta_0$ against $\theta_1$. The Neyman–Pearson lemma, proved in [Likelihood Ratio Tests](06-likelihood-ratio-tests.md), says the most powerful level-$\alpha$ test rejects when $p_{\theta_1}/p_{\theta_0}>k$ for the $k$ making the size $\alpha$. Under monotone likelihood ratio that ratio is a non-decreasing function of $T$, so the event $\{p_{\theta_1}/p_{\theta_0}>k\}$ is the event $\{T>c\}$ for a corresponding $c$ — the two regions are the same set. The critical value $c$ is fixed by $\mathbf{P}_{\theta_0}(T>c)=\alpha$, which does not mention $\theta_1$.

    That last observation is the whole theorem. The optimal region against $\theta_1$ turned out not to depend on which $\theta_1$ was chosen, so the single test $\{T>c\}$ is simultaneously most powerful against every $\theta_1>\theta_0$: it is uniformly most powerful for $H_1\!:\theta>\theta_0$. Monotonicity of $\mathbf{P}_\theta(T>c)$ in $\theta$ additionally extends validity from the point $\theta_0$ to the composite null $\theta\le\theta_0$ at the same $c$.

    Two-sided, the argument breaks at its first step and cannot be repaired. Against $\theta_1>\theta_0$ the optimal region is an upper tail of $T$; against $\theta_1<\theta_0$ the same lemma gives a lower tail. These are different sets, no test is both, and so no uniformly most powerful test exists — which is why two-sided testing requires an extra restriction such as unbiasedness before "best" recovers a referent, exactly as it did for the volatility test in [The Hypothesis Testing Framework](01-hypothesis-testing-framework.md).

    The load-bearing step is that $c$ is determined by the null alone; monotone likelihood ratio is precisely the condition making the optimal region's *shape* independent of the alternative. **Uniform optimality is not a property of a good statistic, it is a property of a family that happens to admit one.**

When the condition fails, the statistic everyone reaches for can fail with it, and fail in a way no amount of data repairs. The Cauchy location family is the standard counterexample: the sample mean of $n$ Cauchy variables is Cauchy with the *same* scale, so averaging accomplishes nothing. Below, all three tests of $\theta=0$ against $\theta=1$ are calibrated by simulation to an exact $5\%$ size, so the comparison is of power alone:

```python
import numpy as np

rng = np.random.default_rng(12027)
reps = 40_000

def stats_at(n, theta):                            # Cauchy location family
    x = theta + rng.standard_cauchy((reps, n))
    lr = np.log1p(x**2).sum(1) - np.log1p((x - 1) ** 2).sum(1)
    return x.mean(1), np.median(x, axis=1), lr

print("  Cauchy location: H0 theta=0 against H1 theta=1, one-sided")
print("  every test calibrated by simulation to an exact 5% size")
print("      n   mean power   median power   LR power")
for n in (10, 100, 1_000, 10_000):
    m0, d0, l0 = stats_at(n, 0.0)
    m1, d1, l1 = stats_at(n, 1.0)
    row = []
    for a, b in ((m0, m1), (d0, d1), (l0, l1)):
        row.append((b > np.quantile(a, 0.95)).mean())
    print(f"  {n:5d}   {row[0]:10.4f}   {row[1]:12.4f}   {row[2]:8.4f}")

m0, _, _ = stats_at(10, 0.0)
m1, _, _ = stats_at(10_000, 0.0)
print(f"  the sample mean's null law does not move: 95th pct {np.quantile(m0, 0.95):.3f} "
      f"at n=10 and {np.quantile(m1, 0.95):.3f} at n=10,000")
# =>   Cauchy location: H0 theta=0 against H1 theta=1, one-sided
#      every test calibrated by simulation to an exact 5% size
#          n   mean power   median power   LR power
#         10       0.0624         0.5729     0.6847
#        100       0.0602         1.0000     1.0000
#       1000       0.0564         1.0000     1.0000
#      10000       0.0557         1.0000     1.0000
#      the sample mean's null law does not move: 95th pct 6.430 at n=10 and 6.176 at n=10,000
```

The mean column is the failure and it is total. Its power is $0.0624$ at ten observations and $0.0557$ at ten thousand — against a size of $0.05$, so the test is barely distinguishable from one that ignores the data and rejects at random, and a thousandfold increase in sample size moves it by less than a percentage point in the wrong direction. The last line names the mechanism: the null law of $\bar X$ is the same Cauchy at every sample size, with a $95$th percentile of $6.430$ at $n=10$ and $6.176$ at $n=10{,}000$. The critical value never shrinks, because the statistic never concentrates, so a location shift of one unit stays buried in a distribution whose spread is unchanged.

The other two columns are not exotic repairs. The median reaches $0.5729$ at $n=10$ and is at $1.0000$ by $n=100$; the likelihood ratio, which for this family is a sum of $\log(1+x^2)$ terms and looks nothing like a location estimate, reaches $0.6847$ at $n=10$. Same data, same level, same hypotheses, and the difference between a test that is asymptotically useless and one that is certain by a hundred observations is entirely which function of the sample was fed to the machinery.

**The Cauchy mean has a perfectly computable null law and no stochastic ordering worth the name, which is the first section's two requirements failing one at a time.**

!!! note "A test statistic, a summary statistic, a sufficient statistic and a pivotal quantity are four different requirements on one object, and only the first two can be read off a tearsheet"
    A **summary** compresses the sample for a human and is judged by interpretability — the descriptive moments of [Descriptive Statistics](../part-10-statistics-foundations/02-descriptive-statistics.md). A **sufficient** statistic loses nothing the model could have used, which is a statement about the likelihood factorizing and is [Statistics and Sufficiency](../part-10-statistics-foundations/05-statistics-and-sufficiency.md). A **pivotal** quantity has a null law free of the nuisance parameters, which is the previous section and is what [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md) inverts. A **test** statistic needs only a computable null law and an ordering under the alternative. The four coincide often enough in the normal model to hide the distinctions, and come apart immediately outside it: the Cauchy mean above is a summary and is not a usable test statistic, while the likelihood ratio that beats it is a fine test statistic and a meaningless summary.

## The Statistic Decides Which Alternatives the Test Can See, and It Is Blind to the Rest by Construction

A test does not ask whether the null is true. It asks whether one number is far into one tail, and every departure from the null that leaves that number alone is invisible to it — not weakly detected, invisible. This is the formal content of the course's verdict that tests see the horizon their lag structure selects: the Engle–Granger statistic on SPY–IVV and the variance ratio on the same spread are not in conflict, because they are functions of different features of the same data and each is silent about the other's.

The blindness is measurable. Below, four standard statistics are calibrated to an exact $5\%$ size on an independent-normal null, then each is run against four departures from that null — a mean shift, mild negative autocorrelation, volatility clustering at the persistence [Time Series](../../part-03-statistics/03-time-series.md) fits on SPY, and independent fat tails with no clustering at all:

```python
import numpy as np

rng = np.random.default_rng(12021)
n, reps = 1260, 4000
sd = 0.012

def paths(kind, k):                                # k independent series of length n
    z = rng.standard_normal((k, n))
    if kind == "iid null":
        return sd * z
    if kind == "mean +2.5 se":
        return sd * z + 2.5 * sd / np.sqrt(n)
    if kind == "AR(1) -0.10":
        x = np.empty((k, n))
        x[:, 0] = sd * z[:, 0]
        for t in range(1, n):
            x[:, t] = -0.10 * x[:, t - 1] + sd * z[:, t]
        return x
    if kind == "GARCH":                            # the course's fitted persistence
        a, b = 0.126, 0.856
        x, v = np.empty((k, n)), np.full(k, sd**2)
        for t in range(n):
            x[:, t] = np.sqrt(v) * z[:, t]
            v = sd**2 * (1 - a - b) + a * x[:, t] ** 2 + b * v
        return x
    return sd * rng.standard_t(4, (k, n)) / np.sqrt(2)   # iid t(4), same variance

def stat(name, x):
    if name == "t on mean":
        return np.abs(x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n)))
    if name == "VR(5)":
        c = np.cumsum(x, 1)[:, 4::5]
        return np.abs(np.var(np.diff(c, axis=1), axis=1) / (5 * np.var(x, axis=1)) - 1)
    if name == "LB(10) on sq":
        y = x**2 - (x**2).mean(1, keepdims=True)
        d = (y * y).sum(1)
        return sum((y[:, k:] * y[:, :-k]).sum(1) ** 2 / d**2 / (n - k) for k in range(1, 11))
    return np.abs((x**4).mean(1) / (x**2).mean(1) ** 2 - 3)      # excess kurtosis

names = ["t on mean", "VR(5)", "LB(10) on sq", "excess kurt"]
truths = ["iid null", "mean +2.5 se", "AR(1) -0.10", "GARCH", "iid t(4)"]
crit = {s: np.quantile(stat(s, paths("iid null", reps)), 0.95) for s in names}

print("  rejection rate at a simulated exact 5% level, n=1260, 4000 reps")
print(f"  {'truth':14s}" + "".join(f"{s:>15s}" for s in names))
for tr in truths:
    x = paths(tr, reps)
    row = "".join(f"{(stat(s, x) > crit[s]).mean():15.4f}" for s in names)
    print(f"  {tr:14s}{row}")
# =>   rejection rate at a simulated exact 5% level, n=1260, 4000 reps
#      truth               t on mean          VR(5)   LB(10) on sq    excess kurt
#      iid null               0.0542         0.0537         0.0515         0.0515
#      mean +2.5 se           0.7270         0.0530         0.0583         0.0575
#      AR(1) -0.10            0.0328         0.4890         0.0600         0.0522
#      GARCH                  0.0517         0.1700         1.0000         0.9830
#      iid t(4)               0.0570         0.0442         0.0835         1.0000
```

The first row is the calibration check and reads $0.0542$, $0.0537$, $0.0515$, $0.0515$ — four statistics at their nominal size, as designed. Every other row should be read across rather than down, because each row is one world and the four numbers in it are what four reasonable analysts would independently conclude about that world. Under a mean shift of two and a half standard errors the $t$-statistic rejects $72.70\%$ of the time and the other three read $0.0530$, $0.0583$ and $0.0575$ — their own size, to within simulation noise. Under negative autocorrelation the variance ratio rejects $48.90\%$ of the time while the $t$-statistic rejects $3.28\%$, *below* its own size, which is the same conservatism the course finds when a HAC correction on negatively autocorrelated returns raises the effective sample size above $n$.

The last two rows separate two statistics that are usually treated as interchangeable diagnostics for "non-normality". Under GARCH, Ljung–Box on squared returns rejects every single time and excess kurtosis rejects $98.30\%$ of the time; under independent $t(4)$ draws with the same variance, excess kurtosis again rejects every time while Ljung–Box on squares reads $0.0835$, close to its size. Fat tails with clustering and fat tails without it are different worlds, one statistic tells them apart and the other cannot, and both are routinely described as tests of the same thing. Meanwhile the $t$-statistic reads $0.0517$ and $0.0570$ in those rows: the two most dramatic departures from the null in the table are completely invisible to the test the same analyst is running on the same series to decide whether the strategy makes money.

**Twenty numbers, sixteen of which are $0.05$, and each of them is a test correctly reporting that it has found nothing while something is plainly there.**

## The Null Distribution Is the Entire Difficulty, and There Are Exactly Three Places to Get One

Given a statistic, everything else is bookkeeping except one step: obtaining the law of $T$ under the null. There are three sources and no fourth. The first is **exact** calculation, available when the null pins the law down completely and the statistic is tractable — the $t_{n-1}$ law above, the binomial enumeration of the previous page, the rank distributions of [Nonparametric Tests](08-nonparametric-tests.md). The second is **asymptotic** approximation, where a limit theorem supplies a law that the finite sample only resembles; this is where most reported p-values come from and where [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md) and [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md) do their work, at the cost that the approximation's quality is itself unknown without checking. The third is **simulation**, generating the null directly — the route every block on this page took, and the one [Permutation Tests](09-permutation-tests.md) and [Bootstrap Tests](10-bootstrap-tests.md) industrialize.

The three are not ranked by respectability, and reaching for the first when the second is honest is a common error. An exact law is exact about the model it assumed; if the model is wrong, exactness buys nothing and can cost a great deal, because it encourages a confidence the approximation would not have. The simulated route has the opposite property: it is never exact about anything except the null that was actually generated, which forces the analyst to write that null down in code, where it can be read.

!!! warning "A statistic chosen because a library exposes it has silently chosen the alternative hypothesis, and the choice is invisible in every line of the output"
    The grid above has no column for "the statistic was wrong". Each test reports a p-value drawn from its own correct null law, and a p-value of $0.4$ from a blind statistic is indistinguishable from a p-value of $0.4$ from a sighted one that genuinely found nothing. This is why the failure survives review: there is no residual to plot, no assumption to check, no diagnostic that fires. The alternative hypothesis a test is actually powered against is determined the moment `ttest_1samp` or `acorr_ljungbox` is typed, and it is never written down anywhere in the analysis. **The free diagnostic is to write down the one departure from the null you are trying to catch, simulate five thousand datasets containing it at the magnitude you would trade, and run your exact testing code on them — if the rejection rate comes back near $\alpha$, your statistic cannot see that departure, no sample size will fix it, and the test you have been running was answering a different question.**

## Choosing the Statistic Is Choosing the Question, and the Choice Is Usually Made by Whatever Was Already Imported

This page established that a test statistic needs only a computable null law and a stochastic ordering under the alternative, and that a statistic satisfying the first and failing the second yields a valid test whose power equals its size; that pivotality is what makes a critical value a constant, so a raw difference of means calibrated at one volatility has size $0.0001$ at half that volatility and $0.6184$ at four times it while its studentized twin holds between $0.0484$ and $0.0494$ throughout; that monotone likelihood ratio, a property of the family rather than the statistic, is what makes one test uniformly best one-sided, and that where it fails the Cauchy sample mean has power $0.0624$ at $n=10$ and $0.0557$ at $n=10{,}000$ against a median reaching $1.0000$ by $n=100$; and that a four-by-five grid of standard statistics against standard departures is mostly $0.05$, with Ljung–Box on squares and excess kurtosis agreeing under GARCH and disagreeing completely under independent fat tails.

The through-line is that a test is a much narrower instrument than the sentence used to describe it. "We tested whether the spread is mean-reverting" names no statistic, and the four instruments the course aims at SPY–IVV return $-2.14$, $0.21$, $0.04$ and $0.000$ because they are four different questions wearing one sentence. None is wrong. The mistake available here is not a miscalculation but a substitution: reading a statement about a statistic as a statement about the world, when the statistic was chosen by whoever wrote the import line and its blindness was fixed before the data was loaded.

What the next page adds is the number the whole apparatus finally reports. A statistic and its null law determine a tail area, that area is quoted to three decimals, and it is asked to carry more meaning than any of the objects on this page can supply — including, routinely, the probability that the null is true. That is [p-values](03-p-values.md).

**A test statistic is a question asked of the data, and the reason so many answers are uninformative is that the question was selected by autocomplete.**
