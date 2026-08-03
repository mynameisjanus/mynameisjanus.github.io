# Permutation Tests

Every page so far has ended at the same recommendation: stop looking the null distribution up and generate it. A permutation test is that recommendation made rigorous. Instead of assuming a family and inheriting its tables, the analyst names a set of transformations that leave the data's law unchanged when the null is true, applies all of them, and reads off where the observed statistic falls. The resulting p-value is exact at every sample size — not asymptotically, not approximately — and the exactness costs no distributional assumption whatsoever. What it costs instead is a claim about which transformations are harmless, and on a time series that claim is where all the difficulty moves.

This page covers exchangeability as the condition that makes a reference distribution computable without a model, the group-invariance argument that delivers finite-sample exactness, the correspondence between what gets permuted and which null is being tested, the difference between equality of distributions and equality of means, and the failure of an ordinary shuffle on dependent data. It does not define the level, the size or the p-value, which are [The Hypothesis Testing Framework](01-hypothesis-testing-framework.md) and [p-values](03-p-values.md); it does not derive the $(1+\#)/(1+B)$ estimator, explain the $+1$, or measure the Monte Carlo error of a resampled p-value, all of which are [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it builds no rank statistic, which is [Nonparametric Tests](08-nonparametric-tests.md); it resamples with replacement nowhere, which is [Bootstrap Tests](10-bootstrap-tests.md); it corrects for no family of strategies, which is [Part XV](../part-15-multiple-testing/index.md); and it never claims that a shuffle is a null.

The trading stake is a methodological instruction the course gives in passing and then obeys. [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) builds its null with `np.roll`, reporting `actual Sharpe 0.30, null mean +0.23, null SD 0.20` and `permutation p = 0.296`, and explains the choice: "the permutation null must be constructed with care (shift, don't shuffle — shuffling would also destroy the position series' own structure and flatter the strategy): the test is only as sharp as the property you hold fixed." Section 5 measures what shuffling would have cost.

## Exchangeability Under the Null Is What Makes the Reference Distribution Computable Without Any Model

The construction needs one ingredient: a set $\mathcal{G}$ of transformations of the data such that, *if the null is true*, applying any of them leaves the joint law unchanged. For a two-sample problem with the null "both samples come from the same distribution", $\mathcal{G}$ is the set of relabellings — which observation belongs to which group is arbitrary under that null, so permuting the labels changes nothing. For a paired problem with a symmetric null, $\mathcal{G}$ is the set of sign flips. For a stationary series, as section 5 develops, it is the set of circular shifts.

The essential point is that $\mathcal{G}$ is *chosen*, not discovered. The analyst asserts that these transformations are harmless under the null, and everything downstream — exactness, the p-value, the verdict — is conditional on that assertion and on nothing else. No density is written down, no moment is assumed finite, no limit theorem is invoked. This is what [Probability Axioms](../part-02-probability-foundations/02-probability-axioms.md) means in observing that a permutation test *imposes* the uniform law on purpose rather than hoping the world supplies it: the equal likelihood of the $|\mathcal{G}|$ transformed datasets is a fact about a procedure the analyst wrote, which is why the resulting p-value is exact rather than asymptotic.

A hypothesis is called **exchangeable** under $\mathcal{G}$ when this invariance holds. Exchangeability is weaker than independence — it permits arbitrary dependence, provided the dependence is symmetric under the group — and it is stronger than identical marginals, which is the gap [Nonparametric Tests](08-nonparametric-tests.md) identified and section 5 makes expensive.

## The Rank of the Observed Statistic Among Its Own Transformations Is Uniform, So the Test Is Exact at Every Sample Size

Given $\mathcal{G}$ and a statistic $T$, compute $T$ on the observed data and on every transformed version. The observed value's position in that list is the p-value, and its uniformity is a one-line consequence of the invariance.

??? note "Proof that the observed statistic's rank among its own group-transformed values is uniform under the null, so the permutation p-value is exactly valid at every sample size"

    Let $\mathcal{G}=\{g_1,\dots,g_M\}$ be a finite group of transformations, including the identity, and suppose the null implies $g(X)\overset{d}{=}X$ for every $g\in\mathcal{G}$. Write $T_j=T(g_j X)$ for the statistic evaluated on each transformed dataset, and assume the $T_j$ are almost surely distinct, so ranks are well defined.

    The key step is that the *whole vector* $(T_1,\dots,T_M)$ is exchangeable. Applying a fixed $h\in\mathcal{G}$ to the data permutes the list, because $\{hg_1,\dots,hg_M\}$ is a relabelling of $\mathcal{G}$ itself — that is precisely the group property. Since $hX\overset{d}{=}X$, the permuted list has the same joint law as the original. So the rank of the identity element's value $T_1$ among all $M$ values is equally likely to be any of $1,\dots,M$:
    $$\mathbf{P}\big(\mathrm{rank}(T_1)=r\big)=\frac{1}{M},\qquad r=1,\dots,M .$$
    Rejecting when that rank falls in the top $\lfloor\alpha M\rfloor$ therefore has probability exactly $\lfloor\alpha M\rfloor/M\le\alpha$, at every sample size and for every underlying distribution.

    Two consequences follow. The attainable levels form a lattice of spacing $1/M$, so the test is conservative by at most $1/M$ — the same discreteness [p-values](03-p-values.md) analysed, arriving here from a group rather than from a lattice-valued statistic. And when $M$ is too large to enumerate, drawing $B$ elements of $\mathcal{G}$ at random and using $(1+\#\{T^{\ast}\ge T\})/(1+B)$ preserves validity, for reasons and at a Monte Carlo cost that [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) develops in full.

    The load-bearing step is the group property: $h\mathcal{G}=\mathcal{G}$ is what turns invariance of the data's law into exchangeability of the statistic list. A set of transformations that is not closed under composition does not give this, which is why "shuffle it a few ways and see" is not the same procedure. **Exactness comes from the algebra, not from the data, which is why no assumption about the distribution appears anywhere in the argument.**

Sixteen observations are few enough to enumerate every split, so the claim can be checked rather than trusted:

```python
import numpy as np
from math import comb
from itertools import combinations
from scipy import stats

rng = np.random.default_rng(12091)
n1 = n2 = 8
N, reps = n1 + n2, 4_000
splits = comb(N, n1)
sel = np.zeros((splits, N))
for i, c in enumerate(combinations(range(N), n1)):   # the first row is the observed labelling
    sel[i, list(c)] = 1.0

def draw(law, k):
    if law == "normal":
        return rng.standard_normal((k, N))
    if law == "Cauchy mix":
        return (rng.standard_cauchy((k, N)) * (rng.random((k, N)) < 0.25)
                + rng.standard_normal((k, N)))
    if law == "lognormal":
        return rng.lognormal(0, 1.5, (k, N))
    return rng.exponential(1.0, (k, N))

print(f"  all {splits:,} equal splits of 16 observations enumerated; the largest")
print(f"  attainable level at or below 5% is {np.floor(0.05 * splits) / splits:.4f}")
print("    law           permutation   pooled t")
for law in ("normal", "Cauchy mix", "lognormal", "exponential"):
    pm = tt = 0
    for _ in range(0, reps, 500):
        x = draw(law, 500)
        tot = x.sum(1, keepdims=True)
        diffs = np.abs(2 * (x @ sel.T) - tot)        # |sum1 - sum2| over every split
        pm += ((diffs >= diffs[:, [0]] - 1e-12).mean(1) <= 0.05).sum()
        a, b = x[:, :n1], x[:, n1:]
        sp = ((n1 - 1) * a.var(1, ddof=1) + (n2 - 1) * b.var(1, ddof=1)) / (N - 2)
        t = (a.mean(1) - b.mean(1)) / np.sqrt(sp * (1 / n1 + 1 / n2))
        tt += (np.abs(t) > stats.t.isf(0.025, N - 2)).sum()
    print(f"    {law:12s}  {pm / reps:11.4f}   {tt / reps:8.4f}")
# =>   all 12,870 equal splits of 16 observations enumerated; the largest
#      attainable level at or below 5% is 0.0500
#        law           permutation   pooled t
#        normal             0.0498     0.0493
#        Cauchy mix         0.0503     0.0328
#        lognormal          0.0503     0.0260
#        exponential        0.0483     0.0403
```

The permutation column is the theorem. Across four laws with nothing in common — a normal, a mixture with infinite-variance contamination, a heavily right-skewed lognormal, and an exponential — the size reads $0.0498$, $0.0503$, $0.0503$ and $0.0483$, all indistinguishable from the attainable $0.0500$ at four thousand replications. There is no trend, no degradation, and no distributional condition being met: the numbers are the same because the proof does not consult the distribution.

The pooled $t$ column is what the alternative costs at eight observations per group. It is correct on the normal, at $0.0493$, and wrong everywhere else — $0.0328$ under the Cauchy mixture, $0.0260$ under the lognormal, $0.0403$ under the exponential. The errors are conservative here rather than anti-conservative, which is worth stating plainly because it is the less-feared direction: the $t$-test is not manufacturing false positives on this data, it is quietly spending half its error budget and losing the power that went with it. Which direction the distortion takes depends on the law, and the analyst does not know the law — that is the whole reason a distribution-free procedure is attractive.

**Sixteen observations, no assumptions, and an exactly correct level — the permutation test's guarantee is the strongest in this part and the cheapest to obtain.**

## What Gets Permuted Is the Hypothesis, and Permuting the Wrong Thing Tests the Wrong Null

Because the group encodes the null, choosing the group *is* choosing the hypothesis, and the choice is made in code rather than in prose. Permuting group labels tests whether the labels carry information. Flipping the signs of paired differences tests symmetry about zero. Shifting a position series against a return series tests whether the *timing* carries information, holding both series' own structure fixed — which is a different and much sharper null than "the strategy has no edge", because it grants the strategy its exposure and asks only whether the timing added anything.

That distinction is the whole content of the course's null mean. Under the shift null, `null mean +0.23` against an actual Sharpe of $0.30$: randomly timed positions with the same persistence and the same long bias still earn a Sharpe of $0.23$, because the position series is long $73\%$ of the time and the market drifted up. Only $0.07$ of the headline number is attributable to timing, and the shift null is what isolates it. A different group would have asked a different question and got a different answer, with no error committed in either case.

This is also where [Hypergeometric Distribution](../part-05-common-distributions/05-hypergeometric-distribution.md) fits: Fisher's exact test is the permutation test for a $2\times2$ table, where enumerating the relabellings gives the hypergeometric law directly. It is the oldest instance of the argument above and a useful reminder that the exact null distribution is a counting problem, not an approximation.

## A Two-Sample Permutation Test's Null Is Equality of Distributions, and Studentizing Is What Narrows It to Means

Relabelling is only invariant if the two samples come from the *same distribution*. That is a stronger null than "the two means are equal", and when the analyst intends the weaker one — equal means, different variances allowed — the group is wrong and the exactness evaporates.

??? note "Proof that a permutation test of equal means is exact for equality of distributions and only asymptotically valid for means, and that studentizing restores the weaker guarantee"

    Under $H_0^{\text{dist}}\!:F_1=F_2$, relabelling leaves the joint law unchanged, the previous section's argument applies verbatim, and the test is exact for any statistic whatsoever. Now weaken the null to $H_0^{\text{mean}}\!:\mu_1=\mu_2$ with $\sigma_1\neq\sigma_2$ permitted. Relabelling is no longer invariant: moving an observation from the wide group to the narrow one changes the joint law, so the group argument fails and nothing guarantees the level.

    What the permutation distribution converges to can be computed. Pool the $N=n_1+n_2$ observations and permute; the permuted difference in means has, conditionally on the pooled sample, variance
    $$\mathrm{var}^{\ast}(\bar X_1^{\ast}-\bar X_2^{\ast})\approx\hat\sigma^{2}_{\text{pool}}\left(\frac{1}{n_1}+\frac{1}{n_2}\right),\qquad \hat\sigma^{2}_{\text{pool}}\to\lambda\sigma_1^{2}+(1-\lambda)\sigma_2^{2},$$
    with $\lambda=n_1/N$, because a permutation draws from the pooled empirical law and cannot tell the two variances apart. The true variance of the observed difference is $\sigma_1^{2}/n_1+\sigma_2^{2}/n_2$. These agree only when $\sigma_1=\sigma_2$ or $n_1=n_2$; otherwise the reference is too narrow or too wide by exactly the Behrens–Fisher ratio that [Parametric Tests](07-parametric-tests.md) derived, and for the same reason — a pooled quantity weights by counts while the truth weights by $1/n$.

    Studentizing repairs it asymptotically. If the statistic is $(\bar X_1-\bar X_2)/\sqrt{s_1^{2}/n_1+s_2^{2}/n_2}$, then both the observed value and every permuted value are divided by an estimate of their own dispersion, and the permutation distribution converges to the same standard normal the statistic does. The exactness under $H_0^{\text{dist}}$ is retained, because dividing by a function of the data does not disturb the group argument, and asymptotic validity under $H_0^{\text{mean}}$ is gained.

    The load-bearing observation is that the permutation reference is built from the *pooled* sample and therefore knows only pooled quantities. **A permutation test is exact about the hypothesis its group encodes, and asking it a weaker question does not weaken the group.**

The size of the gap is worth seeing, with the means equal by construction throughout:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12093)
n1, n2, reps, B = 20, 80, 2_000, 499
N = n1 + n2

def run(s1, s2):
    hits = np.zeros(3)
    for _ in range(reps):
        pool = np.concatenate([rng.normal(0, s1, n1), rng.normal(0, s2, n2)])
        perm = np.argsort(rng.random((B + 1, N)), axis=1)
        perm[0] = np.arange(N)                                     # the observed labelling
        g1, g2 = pool[perm[:, :n1]], pool[perm[:, n1:]]
        d = g1.mean(1) - g2.mean(1)
        v1, v2 = g1.var(1, ddof=1), g2.var(1, ddof=1)
        sp = ((n1 - 1) * v1 + (n2 - 1) * v2) / (N - 2)
        stat = [np.abs(d),
                np.abs(d) / np.sqrt(sp * (1 / n1 + 1 / n2)),
                np.abs(d) / np.sqrt(v1 / n1 + v2 / n2)]
        for j, s in enumerate(stat):
            hits[j] += ((1 + (s[1:] >= s[0]).sum()) / (B + 1)) <= 0.05
    return hits / reps

print("  permutation test of EQUAL means, equal by construction, n1=20 and n2=80")
print("  the two groups differ only in variance, so every rejection is a false one")
print("    sigma1   sigma2   raw difference   pooled t   studentized (Welch)")
for s1, s2 in ((1.0, 1.0), (1.0, 3.0), (3.0, 1.0)):
    r = run(s1, s2)
    print(f"    {s1:6.1f}   {s2:6.1f}   {r[0]:14.4f}   {r[1]:8.4f}   {r[2]:19.4f}")
# =>   permutation test of EQUAL means, equal by construction, n1=20 and n2=80
#      the two groups differ only in variance, so every rejection is a false one
#        sigma1   sigma2   raw difference   pooled t   studentized (Welch)
#           1.0      1.0           0.0525     0.0525                0.0535
#           1.0      3.0           0.0015     0.0015                0.0425
#           3.0      1.0           0.2535     0.2535                0.0580
```

The equal-variance row is the control and all three statistics sit at nominal, $0.0525$, $0.0525$ and $0.0535$ — here $H_0^{\text{dist}}$ genuinely holds and the exactness theorem applies to any statistic. The other rows violate it while leaving the means equal, and the raw difference reports $0.0015$ and $0.2535$: the same procedure the previous section proved exact is now wrong by a factor of five in one direction and thirty in the other, driven entirely by which group is the small one. The pattern is the Behrens–Fisher asymmetry of [Parametric Tests](07-parametric-tests.md) reappearing inside a method advertised as assumption-free.

The first two columns are identical to four decimals in every row, and that is not a coincidence. Within a fixed pooled sample the total sum of squares is constant, so the pooled variance is a strictly decreasing function of the difference in means; the pooled $t$ is therefore a monotone transform of $|d|$, the two statistics order the permutations identically, and they produce the same p-value by construction. Studentizing only changes anything when the divisor uses the two variances *separately*, which is what the Welch column does — and it holds $0.0425$ to $0.0580$ across all three rows.

**A permutation test is not a way of avoiding the equal-variance assumption; it is a way of moving that assumption from the reference table into the choice of group, where it is easier to overlook.**

!!! note "A permutation test, a randomization test, a bootstrap test and a Monte Carlo test all resample, and only the first two have a group behind them"
    A **randomization** test derives its null from randomization the experimenter physically performed, so the group is a fact about the design; a **permutation** test asserts the same invariance for observational data, so the group is a modelling assumption wearing the same algebra. Both inherit the exactness of section 2. A **bootstrap** test resamples *with replacement* from an estimated distribution, which is not a group action — the resampled datasets are not relabellings of the original — so it has no finite-sample guarantee and is [Bootstrap Tests](10-bootstrap-tests.md). A **Monte Carlo** test simulates from a fully specified parametric null, which is exact for a different reason: the null was known, not estimated. The four are routinely called "resampling", and only two of them are exact at any sample size.

## A Time Series Is Not Exchangeable, So Shuffling Manufactures a Null the Market Never Offered

The group has to match the structure actually present. An arbitrary permutation of a dependent series is not a symmetry of its law, and using it builds a reference distribution for a world with no memory.

??? note "Proof that a circular shift is a group action on a stationary series while an arbitrary permutation is not, so shifting preserves the autocovariance and shuffling destroys it"

    Let $(X_t)_{t=1}^{n}$ be stationary with autocovariance $\gamma(k)=\mathrm{cov}(X_t,X_{t+k})$, and consider the circular shift $\tau_j$ sending $X_t\mapsto X_{t+j \bmod n}$. The shifts $\{\tau_0,\dots,\tau_{n-1}\}$ form a group isomorphic to $\mathbb{Z}_n$ under composition, and for a circularly stationary series $\tau_j X\overset{d}{=}X$ for every $j$: the lag-$k$ covariance of the shifted series is $\gamma(k)$ again, because shifting both arguments by the same amount leaves the difference $k$ unchanged. The invariance required by section 2 therefore holds, and the exactness argument applies unchanged.

    An arbitrary permutation $\pi$ does not have this property. Under $\pi$ the lag-$k$ covariance of the permuted series becomes $\gamma(\pi(t+k)-\pi(t))$, which for a randomly chosen $\pi$ averages to $\bar\gamma\approx0$ whenever $n$ is large and $\gamma$ is summable. So the permuted series is approximately serially independent whatever the original was: shuffling does not transform the law, it *replaces* it with the law of an independent series having the same marginal.

    The consequence for a strategy statistic is quantitative. For $T=\tfrac1n\sum_t p_t r_t$ with independent position and return processes, both mean zero,
    $$\mathrm{var}(T)=\frac{\sigma_p^{2}\sigma_r^{2}}{n}\sum_{|k|<n}\left(1-\tfrac{|k|}{n}\right)\rho_p(k)\,\rho_r(k),$$
    and the shuffled null sets $\rho_p(k)=0$ for $k\neq0$, leaving only the $k=0$ term. The shuffled reference is therefore too narrow by the factor $\sum_k\rho_p(k)\rho_r(k)$, which exceeds one whenever the two processes are persistent with the same sign — the ordinary case for a trend-follower on a trending market.

    The load-bearing quantity is that sum: it is one exactly when either series is serially uncorrelated, which is why shuffling is harmless for many statistics and lethal for this one. **The group must preserve the structure the statistic is sensitive to, and the way to find out which structure that is, is to write down the variance.**

Measured on a strategy with no edge by construction, so that every rejection is false:

```python
import numpy as np

rng = np.random.default_rng(12097)
reps, n, B, blk = 600, 1_260, 299, 21
phi_r, phi_s = 0.25, 0.97                          # return persistence, signal persistence

def one_history():
    e = rng.standard_normal(n)
    r, v, prev = np.empty(n), 1.0, 0.0
    for t in range(n):                             # AR(1) mean, GARCH(1,1) variance
        shock = np.sqrt(v) * e[t]
        r[t] = phi_r * prev + shock
        prev = r[t]
        v = (1 - 0.126 - 0.856) + 0.126 * shock**2 + 0.856 * v
    s = np.empty(n)                                # a slow signal INDEPENDENT of the returns
    s[0] = rng.standard_normal()
    z = rng.standard_normal(n)
    for t in range(1, n):
        s[t] = phi_s * s[t - 1] + z[t]
    return np.sign(s), r

print("  a zero-edge strategy: positions from a signal independent of the returns,")
print(f"  on AR(1) returns with phi={phi_r}; every rejection below is a false one")
print("    null built by        size    mean |null| sd")
hits, sds = np.zeros(3), np.zeros(3)
for _ in range(reps):
    pos, r = one_history()
    obs = (pos * r).mean()
    nb = n // blk
    sh = np.array([(rng.permutation(pos) * r).mean() for _ in range(B)])
    bl = np.array([(pos[:nb * blk].reshape(nb, blk)[rng.permutation(nb)].ravel()
                    * r[:nb * blk]).mean() for _ in range(B)])
    cs = np.array([(np.roll(pos, k) * r).mean() for k in rng.integers(1, n, B)])
    for j, nul in enumerate((sh, bl, cs)):
        hits[j] += ((1 + (np.abs(nul) >= abs(obs)).sum()) / (B + 1)) <= 0.05
        sds[j] += nul.std()
for name, h, sd in zip(("iid shuffle", "block shuffle (21)", "circular shift"),
                       hits / reps, sds / reps):
    print(f"    {name:20s} {h:.4f}   {sd:14.5f}")
# =>   a zero-edge strategy: positions from a signal independent of the returns,
#      on AR(1) returns with phi=0.25; every rejection below is a false one
#        null built by        size    mean |null| sd
#        iid shuffle          0.0967          0.02802
#        block shuffle (21)   0.0433          0.03420
#        circular shift       0.0433          0.03435
```

The iid shuffle rejects $9.67\%$ of the time on strategies with no edge whatsoever — nearly twice its nominal level — while the block shuffle and the circular shift both hold at $0.0433$. The three procedures differ only in which group was used; the statistic, the data, the number of resamples and the p-value formula are identical.

The second column is the mechanism, and it is exactly what the proof predicts. The shuffled null distribution has a standard deviation of $0.02802$ against the correct $0.03435$ — it is $18\%$ too narrow, because destroying the position series' persistence removes the $\rho_p(k)\rho_r(k)$ terms that inflate the true variance. A reference distribution that is too narrow makes the observed statistic look further into the tail than it is, and the over-rejection follows arithmetically. Block shuffling recovers $0.03420$ by preserving persistence within blocks, and the circular shift recovers $0.03435$ by preserving it exactly.

This is precisely the course's instruction, priced. Shuffling "would destroy the position series' own structure and flatter the strategy" — flatter, because a too-narrow null makes any given result look more significant. Had the lesson shuffled rather than shifted, its `permutation p = 0.296` would have come out smaller, and the honest conclusion that momentum's timing adds nothing might not have survived.

**The shuffle and the shift are the same three lines of code and the same p-value formula, and one of them is a test of the strategy while the other is a test of a market with no memory.**

!!! warning "A shuffle preserves the marginal and destroys the ordering, so a permutation p-value on a time series is answering a question about a world with no memory"
    The failure has no signature in the output. The resampled null looks like a proper null distribution, the p-value is in $(0,1)$, the formula is right, and the procedure is described in the write-up as "nonparametric" and "exact" — both words being true of the algebra and false of the application. Nothing warns that the group was wrong, because the group is an assumption and assumptions do not raise exceptions. It is the same shape as every failure in this part: correct arithmetic, correct reference distribution, wrong problem. **The free diagnostic is to run your scheme on a strategy you know has no edge: generate positions from a signal built out of noise that cannot possibly predict your returns, keep everything else — the resampling code, the statistic, $B$, the p-value formula — identical, and repeat a few hundred times; if the rejection rate is not near $\alpha$, the group you chose is not a symmetry of your data, and the number to fix is the scheme rather than the strategy.**

## The Analyst Supplies the Randomization, Which Is the Only Assumption Here That Can Be Checked by Reading Code

This page established that a permutation test needs only a group of transformations under which the null leaves the law invariant; that the observed statistic's rank among its own transformed values is uniform by the group property, giving exactness at every sample size and for every distribution, confirmed at $0.0498$, $0.0503$, $0.0503$ and $0.0483$ across a normal, a Cauchy mixture, a lognormal and an exponential where the pooled $t$ wandered to $0.0260$; that the group encodes the hypothesis, so the shift null isolates the $0.07$ of the course's $0.30$ Sharpe that timing contributes against a null mean of $+0.23$; that relabelling is invariant only under equality of *distributions*, so with unequal variances the raw-difference test reads $0.0015$ and $0.2535$ while the studentized version holds near nominal; and that an iid shuffle of a persistent position series gives a null $18\%$ too narrow and a size of $0.0967$, where a block shuffle and a circular shift hold $0.0433$.

The unusual feature of this method is where its assumption lives. Every other test in this part hides its assumptions in a reference distribution — a $t$ table, a $\chi^{2}$ limit, a rank table — where they are invisible at the point of use and can only be recovered by knowing how the table was derived. A permutation test puts its single assumption in the resampling loop, in the analyst's own code, in the line that chooses between `np.random.permutation` and `np.roll`. That does not make the assumption weaker. It makes it *readable*, which is a genuine and rare advantage, and it is why the failure in section 5 is one of the few in this part that a reviewer can catch by looking rather than by simulating.

What the method cannot do is escape the requirement that the group be a symmetry. When no group is available — when the null is "the mean is zero" for a single dependent series, and there is no set of transformations preserving its law while imposing that mean — the exactness argument has nothing to work with, and the only remaining option is to resample from an estimated distribution and accept an asymptotic guarantee in place of a finite-sample one. That is [Bootstrap Tests](10-bootstrap-tests.md).

**A permutation test is exact about the hypothesis encoded in its group, and the group is chosen by the analyst, which makes it the one assumption in this part that is written down in the source rather than inherited from a table.**
