# Bootstrap Tests

A permutation test needs a group, and for most questions worth asking there isn't one. "Does this series have a positive mean" admits no set of transformations that preserves the law while forcing the mean to zero, so the previous page's exactness argument has nothing to act on. The bootstrap fills the gap by resampling from the empirical distribution instead of transforming the data, and it pays for the generality twice: the guarantee drops from exact to asymptotic, and the resampled data no longer satisfies the null unless the analyst makes it. That second cost is the one implementations skip, and the resulting object is widely reported as a p-value while being something else.

This page covers the modification a resampling scheme needs before it can test rather than estimate, the near-equivalence of the three constructions called a bootstrap p-value and the skewness all of them inherit, the extra order of accuracy studentizing buys, the dependence a resampling scheme preserves or destroys, and the question the bootstrap cannot answer at all. It does not build the resampling scheme or prove the plug-in principle, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it does not develop the taxonomy of interval constructions or their accuracy ordering, which is [Bootstrap Confidence Intervals](../part-11-parameter-estimation/08-bootstrap-confidence-intervals.md); it does not treat the bootstrap's inconsistency for a maximum or the $m$-out-of-$n$ repair, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it needs no group and proves no finite-sample exactness, which is [Permutation Tests](09-permutation-tests.md); it corrects nothing for the number of candidates examined, which is [Part XV](../part-15-multiple-testing/index.md); and it never manufactures information the sample did not contain.

The trading stake is a caveat the course puts in a table and then acts on. [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) reports `Sharpe 0.30, iid bootstrap 95% CI [-0.09, 0.71]`, `BCa 95% CI [-0.10, 0.70]` and `block bootstrap 95% CI [-0.07, 0.68]  (SE 0.19)`, notes that agreement with Lo's analytic interval "is the sanity check", and then names the limit: "the bootstrap quantifies sampling noise, not selection bias". Sections 4 and 5 price both halves — what the block bootstrap is buying, and what no resampling scheme can buy at all.

## A Test Needs a Null the Data Does Not Satisfy, So the Resampling Scheme Has to Be Modified and the Interval's Is Not

An interval and a test consume different objects. An interval needs the sampling distribution of $\hat\theta-\theta$, and resampling from the empirical distribution $\hat F$ estimates it directly, because the resampled world's true parameter is the observed $\hat\theta$. A test needs the distribution of $\hat\theta-\theta_0$ *when $\theta_0$ is true* — and $\theta_0$ is generally not true in $\hat F$, since $\hat F$ has parameter $\hat\theta$. Resampling from $\hat F$ unmodified therefore simulates the wrong world.

??? note "Proof that resampling from the unmodified empirical distribution estimates the law of $\hat\theta^{\ast}-\hat\theta$ rather than of $\hat\theta-\theta_0$, and that the raw tail proportion is the inverted percentile interval in disguise"

    Let $\hat F$ be the empirical distribution of the sample and $\hat\theta=T(\hat F)$. A bootstrap resample is drawn from $\hat F$, so within the resampled world the parameter's true value is $T(\hat F)=\hat\theta$, not $\theta_0$. The bootstrap principle asserts
    $$\hat\theta^{\ast}-\hat\theta \ \ \overset{d}{\approx}\ \ \hat\theta-\theta ,$$
    an approximation to the law of the *error*, which is exactly what an interval needs and is what [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) establishes. Nothing in it describes the law of $\hat\theta-\theta_0$ under the null, because no null was imposed anywhere.

    Now consider the raw tail proportion $\hat p_{\text{raw}}=2\min\big(\mathbf{P}^{\ast}(\hat\theta^{\ast}\ge\theta_0),\,\mathbf{P}^{\ast}(\hat\theta^{\ast}\le\theta_0)\big)$, the quantity most code returns. Writing $G$ for the bootstrap distribution function of $\hat\theta^{\ast}$, $\hat p_{\text{raw}}\le\alpha$ holds exactly when $\theta_0$ lies outside $\big[G^{-1}(\alpha/2),\,G^{-1}(1-\alpha/2)\big]$ — which is the percentile confidence interval. So the raw tail proportion is not a separate procedure at all: it is the percentile interval inverted, and it inherits whatever coverage error that interval has, no better and no worse.

    Imposing the null is a different operation. For a mean it means shifting, $x_i^{\ast}\mapsto x_i^{\ast}-\hat\theta+\theta_0$, so the resampled world genuinely has mean $\theta_0$; for a variance it means rescaling. The resulting reference is the law of the statistic under a world where the null holds, which is what a test was always supposed to consume.

    The load-bearing distinction is between the law of an error and the law of a statistic under a hypothesis. **The bootstrap answers the interval's question by default, and the two-sided test built from it is the interval read backwards — which is fine, and is not the additional thing the word "test" was promising.**

## The Three Constructions Called a Bootstrap p-Value Are Almost One Test, and All of Them Inherit the Statistic's Skewness

Three procedures circulate under the name. The **raw tail proportion** reads where $\theta_0$ falls in the unmodified bootstrap distribution. The **recentred null resample** shifts or rescales so the null holds in the resampled world, then compares. The **inverted interval** builds a percentile interval and rejects if it excludes $\theta_0$. The previous proof already collapses the first and third into one procedure; what remains is to find out whether the second is the meaningful improvement its derivation suggests, and whether any of them delivers the promised level on a statistic that is not a mean. Both questions are settled by running all three against nulls that are true by construction, so that every rejection counts as an error:

```python
import numpy as np

rng = np.random.default_rng(12101)
n, reps, B = 252, 2_000, 399
hits = np.zeros((2, 3))

for _ in range(reps):
    x = rng.standard_t(5, n) / np.sqrt(5 / 3)      # mean 0 and sd 1, both nulls TRUE
    idx = rng.integers(0, n, (B, n))
    xb = x[idx]
    for j, (obs, star, null) in enumerate((
            (x.mean(), xb.mean(1), 0.0),                       # H0: mu = 0
            (x.var(ddof=1), xb.var(1, ddof=1), 1.0))):         # H0: sigma^2 = 1
        raw = 2 * min((star >= null).mean(), (star <= null).mean())
        shift = star - obs + null if j == 0 else star * null / obs
        recentred = (np.abs(shift - null) >= abs(obs - null)).mean()
        lo, hi = np.quantile(star, [0.025, 0.975])
        hits[j] += [raw <= 0.05, recentred <= 0.05, not (lo <= null <= hi)]

print(f"  bootstrap tests of two TRUE nulls on t(5) data, n={n}, B={B}, nominal 0.05")
print("    hypothesis          raw tail   resampled under H0   inverted interval")
for j, name in enumerate(("H0: mean = 0", "H0: variance = 1")):
    r = hits[j] / reps
    print(f"    {name:18s}  {r[0]:8.4f}   {r[1]:18.4f}   {r[2]:17.4f}")
# =>   bootstrap tests of two TRUE nulls on t(5) data, n=252, B=399, nominal 0.05
#        hypothesis          raw tail   resampled under H0   inverted interval
#        H0: mean = 0          0.0600               0.0575              0.0635
#        H0: variance = 1      0.1025               0.1005              0.1100
```

Read across the rows first. For each hypothesis the three columns agree to within simulation noise — $0.0600$, $0.0575$, $0.0635$ for the mean, and $0.1025$, $0.1005$, $0.1100$ for the variance. The proof predicted the first and third columns would coincide, and they do; what the table adds is that the recentred version is barely distinguishable from either. For a two-sided test the modification that the previous section called non-optional turns out to change almost nothing, because a two-sided comparison of $|\hat\theta-\theta_0|$ against a spread is insensitive to where that spread is centred. The recentring matters for one-sided tests and for statistics whose bootstrap distribution is not centred on the observed value; for the common case it is bookkeeping.

Reading down the rows is where the real defect is. Every construction is near nominal for the mean and roughly twice nominal for the variance — $0.1005$ against a promised $0.05$, on data where the null is true by construction. The mean of $t(5)$ data has a nearly symmetric sampling distribution and the percentile machinery handles it; the *variance* of the same data is badly skewed, its bootstrap distribution is skewed the same way, and reading percentiles off a skewed distribution to build a symmetric-in-intent test misses in both tails at once. Nothing was recentred wrongly. The statistic simply is not close enough to pivotal for a percentile construction, which is the defect [Bootstrap Confidence Intervals](../part-11-parameter-estimation/08-bootstrap-confidence-intervals.md) orders the constructions by, arriving here as a size rather than as a coverage.

**The three things called a bootstrap p-value are one test wearing three names, and what separates a good one from a bad one is not which name it wears but whether the statistic was studentized first.**

## Studentizing Buys an Order of Accuracy That Shows Up as Size Exactly Where the Sample Is Short

The repair is the one [Bootstrap Confidence Intervals](../part-11-parameter-estimation/08-bootstrap-confidence-intervals.md) ranks first: resample a quantity whose law is free of the parameter. Dividing by an estimated standard error computed within each resample gives a bootstrap-$t$, whose distribution is approximated one order more accurately.

??? note "Proof that a studentized bootstrap test has size error $O(n^{-1})$ where a percentile test has $O(n^{-1/2})$, and that the extra order is bought by resampling a pivot"

    The Edgeworth machinery is developed in [Bootstrap Confidence Intervals](../part-11-parameter-estimation/08-bootstrap-confidence-intervals.md) and is spent rather than rebuilt here. Write the studentized statistic $T=(\hat\theta-\theta)/\hat\sigma$ and expand its distribution function in powers of $n^{-1/2}$:
    $$\mathbf{P}(T\le z)=\Phi(z)+n^{-1/2}q_1(z)\phi(z)+O(n^{-1}),$$
    where $q_1$ is a polynomial whose coefficients involve the skewness. The bootstrap version satisfies the same expansion with the population cumulants replaced by sample ones, which differ by $O_p(n^{-1/2})$. Subtracting, the $n^{-1/2}$ terms **cancel**, because both expansions carry the same $q_1$ up to an error of that order, leaving a discrepancy of $O(n^{-1})$.

    For the percentile construction the statistic being resampled is $\hat\theta^{\ast}-\hat\theta$, which is not pivotal: its law still depends on the unknown scale. Its expansion carries a leading polynomial that does **not** match the one governing $\hat\theta-\theta$, the $n^{-1/2}$ terms fail to cancel, and the error stays $O(n^{-1/2})$.

    The load-bearing property is pivotality, exactly as in [Test Statistics](02-test-statistics.md): dividing by an estimate of a statistic's own dispersion removes the nuisance scale from the leading term, and it is that removal — not the resampling — that buys the order. **The bootstrap does not confer accuracy; it inherits whatever accuracy the quantity being resampled already had.**

An order of $n^{-1/2}$ against $n^{-1}$ is invisible at $n=1260$ and decisive at $n=21$, which is where a new signal always begins:

```python
import numpy as np

rng = np.random.default_rng(12103)
reps, B = 2_000, 999

print("  size of two bootstrap tests of H0: mean = 0 on t(5) data, nominal 0.05")
print("        n   percentile   studentized")
for n in (21, 63, 126, 252):
    pc = st = 0
    for _ in range(reps):
        x = rng.standard_t(5, n) / np.sqrt(5 / 3)
        obs = x.mean()
        xb = x[rng.integers(0, n, (B, n))]
        mb, sb = xb.mean(1), xb.std(1, ddof=1)
        lo, hi = np.quantile(mb, [0.025, 0.975])   # percentile interval, inverted
        pc += not (lo <= 0.0 <= hi)
        t = (mb - obs) / (sb / np.sqrt(n))         # bootstrap-t, recentred by construction
        q = np.quantile(t, [0.025, 0.975])
        se = x.std(ddof=1) / np.sqrt(n)
        st += not (obs - q[1] * se <= 0.0 <= obs - q[0] * se)
    print(f"    {n:5d}   {pc / reps:10.4f}   {st / reps:11.4f}")
# =>   size of two bootstrap tests of H0: mean = 0 on t(5) data, nominal 0.05
#            n   percentile   studentized
#           21       0.0795        0.0675
#           63       0.0670        0.0635
#          126       0.0535        0.0570
#          252       0.0565        0.0540
```

At twenty-one observations the percentile test's size is $0.0795$ — more than half again its nominal level — against the studentized test's $0.0675$. At sixty-three the gap narrows to $0.0670$ against $0.0635$, and from a hundred and twenty-six onward the two are indistinguishable at these replication counts, both sitting a little above nominal. That is the theorem's shape: an $O(n^{-1/2})$ error and an $O(n^{-1})$ error are both small once $n$ is large, and their ratio is what the small-$n$ column exposes.

The practical reading is about when the distinction is worth the extra code. A desk evaluating a signal on twenty years of daily data can use whichever construction the library defaults to and the difference will be inside the noise. A desk evaluating a new strategy on a quarter of data, or a monthly series with two years of history, or a single sleeve's twenty-one trades, is squarely in the regime where the default is wrong by half its own level — and that is precisely the regime where the decision is least reversible and most likely to be made on the number alone.

**The construction that theory ranks first is worth nothing where samples are long and matters exactly where samples are short, which is the opposite of when anyone bothers.**

## A Bootstrap Test Inherits the Dependence Its Scheme Preserves, and a Block Length Has Two Ways to Be Wrong

Every scheme so far resamples observations independently, which asserts that the observations *are* independent. When they are not, the resampled world has less memory than the real one and the reference distribution is the wrong width — the same mechanism [Permutation Tests](09-permutation-tests.md) proved for shuffling, now with replacement.

??? note "Proof that the iid bootstrap is consistent for the mean of a martingale difference sequence and inconsistent under linear serial correlation, with the long-run variance as the exact discrepancy"

    For a stationary series with autocovariances $\gamma(k)$, the variance of the sample mean is
    $$\mathrm{var}(\bar X)=\frac{1}{n}\sum_{|k|<n}\left(1-\frac{|k|}{n}\right)\gamma(k)\ \longrightarrow\ \frac{1}{n}\sum_{k=-\infty}^{\infty}\gamma(k)=\frac{\sigma_{\mathrm{LR}}^{2}}{n},$$
    the **long-run variance**. The iid bootstrap draws observations independently from $\hat F$, so within the resampled world all autocovariances are zero and the resampled mean has variance $\hat\gamma(0)/n$. The ratio of what the bootstrap believes to what is true is therefore $\gamma(0)/\sigma^{2}_{\mathrm{LR}}$, which equals one exactly when $\sum_{k\neq0}\gamma(k)=0$.

    That condition holds for a **martingale difference sequence** — including a GARCH process, whose returns are serially uncorrelated even though their squares are not — so the iid bootstrap is consistent for the mean of a GARCH series despite the obvious dependence. It fails for any series with $\sum_{k\neq0}\gamma(k)\neq0$: for an AR(1) with coefficient $\phi$, $\sigma^{2}_{\mathrm{LR}}/\gamma(0)=(1+\phi)/(1-\phi)$, which at $\phi=0.30$ is $1.857$, so the iid reference is too narrow by a factor of $\sqrt{1.857}=1.363$.

    A block scheme repairs it by resampling contiguous runs, which preserves autocovariances at lags shorter than the block. The block length is then a bias–variance choice with two distinct failure modes: too short and the long-run variance is underestimated exactly as in the iid case, too long and few independent blocks remain, so the reference distribution itself becomes noisy.

    The load-bearing quantity is $\sum_{k\neq0}\gamma(k)$, and it is zero for the dependence that dominates financial returns at the daily horizon. **Whether the iid bootstrap is wrong depends not on whether the series is dependent but on whether its dependence shows up in the first autocovariance, and volatility clustering does not.**

Both cases can be run side by side, on series with identical marginals and a true mean of zero:

```python
import numpy as np

rng = np.random.default_rng(12107)
reps, n, B = 500, 504, 299

def series(kind):
    e = rng.standard_normal(n)
    if kind == "AR(1) 0.30":
        x, prev = np.empty(n), 0.0
        for t in range(n):
            x[t] = 0.30 * prev + e[t] * np.sqrt(1 - 0.09)
            prev = x[t]
        return x
    x, v = np.empty(n), 1.0                        # GARCH(1,1) at the course's fit
    for t in range(n):
        x[t] = np.sqrt(v) * e[t]
        v = (1 - 0.126 - 0.856) + 0.126 * x[t] ** 2 + 0.856 * v
    return x

def resample(x, block):
    if block == 1:
        return x[rng.integers(0, n, (B, n))]
    pos = np.empty((B, n), dtype=int)              # stationary bootstrap, Politis-Romano
    pos[:, 0] = rng.integers(0, n, B)
    for t in range(1, n):
        fresh = rng.random(B) < 1 / block
        pos[:, t] = np.where(fresh, rng.integers(0, n, B), (pos[:, t - 1] + 1) % n)
    return x[pos]

def size(kind, block):
    hit = 0
    for _ in range(reps):
        x = series(kind)
        obs = x.mean()
        mb = resample(x, block).mean(1) - obs       # recentred, so the null holds in the resample
        hit += (np.abs(mb) >= abs(obs)).mean() <= 0.05
    return hit / reps

print(f"  bootstrap test of H0: mean = 0, TRUE in both series; n={n}, B={B}, nominal 0.05")
print("    scheme                 GARCH   AR(1) 0.30")
for lab, b in (("iid bootstrap", 1), ("stationary, mean 18", 18), ("stationary, mean 3", 3)):
    print(f"    {lab:20s}  {size('GARCH', b):6.4f}   {size('AR(1) 0.30', b):10.4f}")
# =>   bootstrap test of H0: mean = 0, TRUE in both series; n=504, B=299, nominal 0.05
#        scheme                 GARCH   AR(1) 0.30
#        iid bootstrap         0.0600       0.1520
#        stationary, mean 18   0.0640       0.0520
#        stationary, mean 3    0.0480       0.0800
```

The GARCH column is the counterintuitive one and it confirms the proof. A GARCH series is strongly dependent — its volatility clusters, its squared returns are autocorrelated at $+0.294$ in the course's data — and the iid bootstrap nonetheless holds at $0.0600$, because none of that dependence appears in $\sum_{k\neq0}\gamma(k)$ for the returns themselves. Resampling observations independently is legitimate here, and using a block scheme instead costs nothing but buys nothing.

The AR(1) column is where it breaks. At $\phi=0.30$ the iid bootstrap's size is $0.1520$, three times nominal, and the proof gives the number: the reference is too narrow by $\sqrt{1.857}=1.363$, and a normal reference $36\%$ too narrow rejects a true null about three times too often. The stationary bootstrap at mean block $18$ restores $0.0520$. The last row is the other failure mode — a mean block length of $3$ is too short to capture the persistence, and it recovers only part of the damage at $0.0800$, still above nominal.

Between the two columns is the practical lesson. A block scheme is insurance whose premium is small and whose necessity depends on a property of the series that is checkable in one line: the sum of the autocorrelations of the quantity being averaged. Volatility clustering, the most visible dependence in financial data, does not require it for a test about a mean. Momentum in the mean, which is far less visible, does.

**The scheme has to preserve the dependence the statistic is sensitive to, and which dependence that is, is a question about the statistic rather than about how dependent the data looks.**

!!! note "The bootstrap p-value, the bootstrap interval and the bootstrap standard error are three outputs of one computation, and only the first two are interchangeable"
    The resampled distribution supports all three, and the relationships are worth keeping straight. The **standard error** is its standard deviation, and it is the only one of the three that survives a badly skewed statistic without further correction. The **percentile interval** reads its quantiles, and the **raw tail p-value** is that interval inverted, as section 1 proves — so quoting both a percentile interval and a bootstrap p-value is quoting one number twice. The **studentized** versions of the interval and the test are a genuinely different computation and the one the accuracy ordering prefers. Meanwhile "bootstrap" also names two things this page has kept separate: a scheme for resampling *observations*, and a scheme for resampling *residuals* from a fitted model, which belongs with [Part XIII](../part-13-regression/index.md).

## What the Bootstrap Cannot Do Is Notice That the Question Was Chosen After the Data

Every failure on this page has been a mis-specified reference distribution, and every one has had a repair inside the resampling loop. There is one failure with no repair available there at all, and the course names it in a single clause: "the bootstrap quantifies sampling noise, not selection bias".

The reason is structural. A resampling scheme takes the sample and the statistic as given and asks what would have happened under repeated sampling *of that statistic*. If the statistic was chosen by looking at the data — the best of fifty lookback windows, the sleeve that survived a screen, the horizon at which the effect appeared — then the quantity whose sampling variability is being estimated is not the one that was actually selected, and no amount of resampling recovers the selection. The bootstrap will faithfully report the sampling noise of the winner as though the winner had been named in advance, and it will do so with a tight interval, because the winner was chosen for having a large statistic and a large statistic is usually a stable one.

This is not a defect the resampling can be modified to fix, because the information required — how many candidates were examined, and how correlated they were — is not in the sample. It has to be supplied from outside, by an accounting of the search, which is [Data Snooping Bias](../part-15-multiple-testing/04-data-snooping-bias.md). The course is explicit that its own corrections "still apply first". Every number on this page is conditional on the hypothesis having been fixed before the data was seen, and that condition is the one most often violated and least often recorded.

!!! warning "A bootstrap test on a serially correlated series reports the wrong size in whichever direction the autocorrelation points, and the check is one line"
    The scheme cannot detect that it is wrong. Every diagnostic the resampling produces — the spread of the resampled statistics, the smoothness of the bootstrap distribution, the agreement between percentile and BCa intervals — is computed *inside* the assumed world, so a reference that is $36\%$ too narrow looks exactly as convincing as a correct one, and the two agree with each other. The course's own sanity check has the same limit: its iid and block intervals agree closely on the momentum Sharpe, which is reassuring precisely because that statistic's dependence is mild, and would have been equally reassuring if it were not. **The free diagnostic is to compute the sum of the autocorrelations of the series you are averaging — for the mean, the long-run variance ratio is $1+2\sum_{k\ge1}\rho_k$, one line of numpy, and it is the exact factor by which an iid reference is too narrow; if it is materially above one, use a block scheme with a mean block length several times the lag at which $\rho_k$ dies, and if you want the whole thing checked rather than reasoned about, fit your series' own model, simulate a few hundred zero-mean paths from it, and run your exact bootstrap testing code on each.**

## The Resample Is a Second Opinion From the Same Witness

This page established that resampling from the unmodified empirical distribution estimates the law of an error rather than of a statistic under a null, so the raw tail proportion is the percentile interval inverted — confirmed at $0.0600$ against $0.0635$ for a mean and $0.1025$ against $0.1100$ for a variance, where all three constructions sit at twice nominal because the statistic is skewed rather than because the null was imposed wrongly; that studentizing buys an order of accuracy visible as a size of $0.0795$ against $0.0675$ at $n=21$ and invisible by $n=126$; and that the iid bootstrap is legitimate for the mean of a GARCH series at $0.0600$, wrong by three times for an AR(1) at $0.1520$, repaired to $0.0520$ by a stationary bootstrap at mean block $18$ and only half-repaired at $0.0800$ by a block length of $3$.

Which closes the part. [The Hypothesis Testing Framework](01-hypothesis-testing-framework.md) found that a level is a promise and a size is what is delivered, with a pooled proportion test verified at $p=0.500$ having a true size of $0.0834$; [Test Statistics](02-test-statistics.md) found the Cauchy sample mean's power flat at $0.0557$ from ten observations to ten thousand; [p-values](03-p-values.md) found a genuine Sharpe-$0.30$ edge producing p-values from $0.00176$ to $0.8555$ across replications of one study; [Type I and Type II Errors](04-type-i-and-type-ii-errors.md) found that repairing a size from $0.1985$ to $0.0500$ silently discards two-thirds of the apparent power; [Statistical Power](05-statistical-power.md) found the course's own weekday tests carrying $8.20\%$ power against the effect they were hunting; [Likelihood Ratio Tests](06-likelihood-ratio-tests.md) found the Wald, score and likelihood-ratio statistics disagreeing about the verdict at $250$ observations and a boundary null putting $60.52\%$ of its mass at exactly zero; [Parametric Tests](07-parametric-tests.md) found the $F$-test firing on $44\%$ of samples with identical variances at the course's measured kurtosis; [Nonparametric Tests](08-nonparametric-tests.md) found the sign test rejecting at $7.09\times10^{-9}$ and reporting that a profitable strategy loses money; and [Permutation Tests](09-permutation-tests.md) found a shuffle building a null $18\%$ too narrow where a shift built the right one.

The single shape across all ten is that the arithmetic was never wrong. Every statistic on every page was computed correctly, every reference distribution was the correct distribution for *some* problem, and every p-value was a valid answer to a question. What varied was whether that question was the one being asked, and no output on any of these pages carried a field for the difference. The one repair that appeared on nine of the ten pages is the same instruction each time: generate the null you actually face and count, rather than looking one up. It works because it forces the analyst to write the hypothesis in code, where it can be read, instead of inheriting it from a table where it cannot.

That leaves the failure this last page could not fix and no page in this part can. Every level, every power, every coverage and every size here has been computed for a hypothesis fixed before the data arrived. Research does not work that way, and the arithmetic of what happens when a hypothesis is chosen after looking — how many candidates were examined, how correlated they were, and what the survivor's statistic is worth once the search is priced — is [White's Reality Check](../part-15-multiple-testing/05-whites-reality-check.md).

**A bootstrap asks the sample what else it might have said, and the sample answers honestly about everything except how it came to be the sample you were looking at.**
