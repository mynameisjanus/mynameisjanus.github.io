# p-values

The p-value is the most reported number in empirical work and the most misdescribed, and almost every misdescription shares one root: it is read as a summary of the evidence when it is a draw from a distribution. Fix the null and fix the sample size, and the p-value is a random variable with a law of its own — one whose spread across replications of the *same true effect* runs across orders of magnitude, and which nobody reports because the convention is to quote a single realization to three decimals. Everything the p-value is accused of doing badly, it does exactly as specified. The gap is between what it was specified to do and what it is asked to carry.

This page covers the p-value as the smallest level at which the data rejects, the super-uniformity inequality that is its actual validity requirement, the strict conservatism a discrete null forces and the mid-p repair that trades validity for it, the sampling distribution of the p-value across replications, and the drift to zero that any false null produces as $n$ grows. It does not define the level, the size or the rejection region, which is [The Hypothesis Testing Framework](01-hypothesis-testing-framework.md); it does not choose the statistic whose tail it measures, which is [Test Statistics](02-test-statistics.md); it does not prove the probability integral transform or measure what dependence does to a p-value's uniformity, which is [Continuous Uniform Distribution](../part-05-common-distributions/09-continuous-uniform-distribution.md); it does not compute the posterior probability of a hypothesis, which is [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md); it does not compute power, which is [Statistical Power](05-statistical-power.md); it corrects nothing for the number of p-values produced, which is [Part XV](../part-15-multiple-testing/index.md); and it never measures the size of an effect.

The trading stake is a sentence the course writes after fitting an ARIMA model to daily SPY returns. [Time Series](../../part-03-statistics/03-time-series.md) reports coefficients at `p = 2.4e-04, 5.5e-08` and then refuses them: "The coefficients are statistically significant — with 6,400 observations, almost anything is — and economically feeble: they repackage a lag-one autocorrelation of −0.086, worth well under a basis point of daily predictability before costs." That autocorrelation is pinned at `daily lag-1 autocorr -0.086 (-6.9 se, n 6410)` in [Mean Reversion and Pairs Trading](../../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md). Section 5 shows that the collision between an overwhelming p-value and a negligible effect is not a coincidence of this dataset but the guaranteed behaviour of the quantity.

## A p-Value Is the Smallest Level at Which the Data Rejects, Which Makes It a Statistic and Not a Probability About a Hypothesis

Take the family of level-$\alpha$ tests of one null, indexed by $\alpha$, and nested so that a smaller $\alpha$ rejects less. The **p-value** is
$$p(x)=\inf\{\alpha:\text{the level-}\alpha\text{ test rejects on }x\},$$
the tightest standard the observed data can clear. Rejecting at level $\alpha$ and observing $p\le\alpha$ are then the same event by construction, which is the entire reason the number is useful: it reports the outcome of every test in the family at once, so a reader with a different $\alpha$ in mind does not need the analysis rerun.

Two consequences follow immediately and both are routinely denied. First, $p$ is a function of the data alone, so it is a statistic — it has a sampling distribution, it varies from sample to sample, and quoting it without an indication of that variability is the same error as quoting an estimate without a standard error. Second, $p$ is computed entirely under the null: no alternative and no prior enters the calculation. It therefore cannot be the probability that the null is true, not because that quantity is forbidden but because the ingredients for it — a prior over hypotheses and the alternative's likelihood — were never supplied. The arithmetic converting the one into the other, and the base rates it needs, is worked in [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md).

The usual construction, $p=\mathbf{P}_{H_0}(T\ge t_{\text{obs}})$ for an upper-tail statistic, is a special case rather than the definition. It coincides with the infimum whenever the tests in the family are the nested tails of one statistic, which is the common case and not the universal one.

## Validity Is the Inequality $\mathbf{P}(p\le\alpha)\le\alpha$ at Every Level Simultaneously, and Uniformity Is the Case Where It Binds

The property that makes a p-value a p-value is not uniformity. It is **super-uniformity**: $\mathbf{P}_{\theta}(p\le\alpha)\le\alpha$ for every $\alpha\in(0,1)$ and every $\theta$ in the null. This is exactly the statement that using $p\le\alpha$ as a rejection rule gives a test of level $\alpha$ — for all $\alpha$ at once, from one number. Uniformity is the boundary case in which the inequality is an equality everywhere, and it is a bonus rather than a requirement.

??? note "Proof that the smallest rejecting level is super-uniform under any null, and that it is exactly uniform if and only if the test family attains its level at every $\alpha$"

    Let $R_\alpha$ be the rejection region of the level-$\alpha$ test, nested so $\alpha'<\alpha$ implies $R_{\alpha'}\subseteq R_\alpha$, and let $p(x)=\inf\{\alpha:x\in R_\alpha\}$. If $p(x)\le\alpha$ then $x\in R_{\alpha'}$ for every $\alpha'>\alpha$, hence $x$ lies in the closure of $R_\alpha$; up to that boundary, $\{p\le\alpha\}\subseteq R_\alpha$. Taking probabilities under any null $\theta$ and using the level guarantee $\mathbf{P}_\theta(R_\alpha)\le\alpha$,
    $$\mathbf{P}_\theta(p\le\alpha)\ \le\ \mathbf{P}_\theta(R_\alpha)\ \le\ \alpha .$$
    Nothing about the statistic, the family, or continuity was used. Super-uniformity is therefore automatic for anything constructed this way, which is why a valid p-value is cheap and an *exactly uniform* one is not.

    Equality requires both inequalities to bind. The second binds exactly when the test attains its level, $\mathbf{P}_\theta(R_\alpha)=\alpha$, which fails whenever the attainable rejection probabilities do not include $\alpha$ — the discrete case of the next section. The first binds when $\{p\le\alpha\}$ and $R_\alpha$ differ by a null set. When both hold at every $\alpha$, $\mathbf{P}_\theta(p\le\alpha)=\alpha$ for all $\alpha$, which is the definition of the uniform law on $(0,1)$. For a continuous statistic this is the probability integral transform applied to $T$, proved in [Continuous Uniform Distribution](../part-05-common-distributions/09-continuous-uniform-distribution.md), where the consequences of dependence for that uniformity are also measured.

    The load-bearing step is the level guarantee $\mathbf{P}_\theta(R_\alpha)\le\alpha$ holding *simultaneously* for every $\alpha$: super-uniformity is a statement about a whole family of tests, and it is what licenses a reader to apply their own threshold to a number computed by someone else. **A p-value is valid when it is at least as large as a uniform, and useful in proportion to how little larger it is.**

Super-uniformity is a one-sided guarantee, and the direction matters. A p-value that is too large is conservative: the test rejects less than $\alpha$, the error budget goes unspent, and the cost is power. A p-value that is too *small* is not a p-value at all — the object has failed the only property it was required to have, and every downstream number inherits the failure. The distortions measured on the previous two pages were of the second kind and came from the null's law being wrong. The distortion in the next section is of the first kind and comes from arithmetic that is entirely correct.

## A Discrete Null Makes the Inequality Strict, So the Test Spends Less Than Its Budget and Buys Less Than Its Power

When the statistic lives on a lattice, the attainable rejection probabilities form a finite set, and a nominal $\alpha$ that falls between two of them cannot be attained. The test must fall back to the largest attainable value below $\alpha$, so the size is strictly less than the level, the p-value is stochastically larger than uniform, and the shortfall is power given away for nothing.

??? note "Proof that a lattice-valued statistic makes the p-value stochastically larger than uniform, and that the mid-p correction is uniform in mean while failing to be valid"

    Let $T$ take values $t_0<t_1<\dots$ with null probabilities $\pi_j=\mathbf{P}_{H_0}(T=t_j)$, and let the upper-tail p-value be $p_j=\mathbf{P}_{H_0}(T\ge t_j)=\sum_{i\ge j}\pi_i$. The p-value takes only the finitely many values $\{p_j\}$, so its distribution function is a step function. For $\alpha$ strictly between two consecutive attainable values $p_{j+1}<\alpha<p_j$,
    $$\mathbf{P}_{H_0}(p\le\alpha)=\sum_{i:\,p_i\le\alpha}\pi_i=p_{j+1}<\alpha,$$
    so the super-uniform inequality is strict on every such $\alpha$, and the gap $\alpha-p_{j+1}$ can approach the size of an atom. The p-value is stochastically larger than uniform, and the wasted budget is the shortfall of the attainable size below $\alpha$.

    The **mid-p** value replaces $p_j$ by $p_j-\tfrac12\pi_j$, splitting the atom at the observed value. Its null mean is exactly $\tfrac12$, matching the uniform, which is the sense in which it is calibrated "on average". But validity is a statement at every $\alpha$, not on average, and the correction moves mass *downward* across thresholds: at any $\alpha$ lying between $p_j-\tfrac12\pi_j$ and $p_j$, the mid-p rejects on the whole atom $\pi_j$ where the exact p-value rejected on none of it, and $\mathbf{P}(p_{\text{mid}}\le\alpha)$ can exceed $\alpha$. The mid-p is therefore not a valid p-value at every level, and the block below finds it above nominal at three of six sample sizes.

    The load-bearing quantity is the size of the atom at the observed value: it bounds both the conservatism of the exact p-value and the anti-conservatism of the mid-p repair, and it shrinks like $n^{-1/2}$, which is why the whole issue evaporates in large samples and dominates in small ones. **Discreteness does not make a p-value wrong, it makes it expensive, and the standard repair pays for the power with the guarantee.**

The cost is exactly computable by enumeration for the one-sided binomial test — the natural test of a hit rate, and the shape of every "did this signal beat a coin" question:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12033)
grid = np.linspace(0.001, 0.999, 999)

print("  exact one-sided binomial test of H0: p = 0.5, nominal level 0.05")
print("      n   cutoff   actual size   budget spent   mid-p size   sup(alpha - F_p)")
for n in (10, 20, 50, 100, 250, 1000):
    k = np.arange(n + 1)
    pmf = stats.binom.pmf(k, n, 0.5)
    tail = stats.binom.sf(k - 1, n, 0.5)           # the p-value when X = k
    cut = int(k[tail <= 0.05][0])
    size = tail[cut]
    midp = tail - 0.5 * pmf
    gap = max(a - pmf[tail <= a].sum() for a in grid)
    print(f"  {n:5d}   {cut:6d}   {size:11.4f}   {size / 0.05:12.1%}   "
          f"{pmf[midp <= 0.05].sum():10.4f}   {gap:16.4f}")

x = rng.binomial(20, 0.5, 400_000)
tail20 = stats.binom.sf(np.arange(21) - 1, 20, 0.5)
print(f"  n=20 simulated: the nominal-5% test rejects {(tail20[x] <= 0.05).mean():.4f} "
      f"of 400,000 true nulls")
# =>   exact one-sided binomial test of H0: p = 0.5, nominal level 0.05
#          n   cutoff   actual size   budget spent   mid-p size   sup(alpha - F_p)
#         10        9        0.0107          21.5%       0.0547             0.2460
#         20       15        0.0207          41.4%       0.0577             0.1761
#         50       32        0.0325          64.9%       0.0595             0.1121
#        100       59        0.0443          88.6%       0.0443             0.0788
#        250      139        0.0438          87.5%       0.0438             0.0502
#       1000      527        0.0468          93.7%       0.0468             0.0248
#      n=20 simulated: the nominal-5% test rejects 0.0210 of 400,000 true nulls
```

The size column is the exact conservatism and it is severe where samples are short. At ten observations a nominal $5\%$ test has size $0.0107$ and spends $21.5\%$ of its error budget; at twenty it is $0.0207$ and $41.4\%$; four hundred thousand simulated true nulls at $n=20$ reject $0.0210$ of the time, confirming the enumeration. Not until $n=1000$ does the test spend $93.7\%$ of what it was allowed. The unspent budget is not free caution — it is power the analyst was entitled to and did not take, and it is invisible in the output, because the reported p-value is correct and the test is valid.

The mid-p column shows what the standard repair costs. At $n=10$, $20$ and $50$ it reads $0.0547$, $0.0577$ and $0.0595$ — above the nominal $0.05$ in all three, which is to say the repaired test is no longer level $0.05$. By $n=100$ the mid-p and exact columns coincide, because the correction no longer moves the decision across the threshold. The final column tracks the sup-distance between the p-value's distribution function and the uniform, falling from $0.2460$ to $0.0248$: the whole phenomenon is an $n^{-1/2}$ effect that is invisible in the sample sizes textbooks use for examples and decisive in the sample sizes a desk has for a new signal.

**The exact test gives away power to keep its promise and the mid-p test breaks the promise to take the power back, and neither line appears on the report.**

## A p-Value Has a Sampling Distribution Wider Than Anything Its Third Decimal Suggests

The p-value's own variability is the fact most thoroughly hidden by convention. Below, an edge is *assumed real* — annualized Sharpe exactly $0.30$, the course's momentum number, taken as truth rather than estimated — and the identical study is run twenty thousand times at four track-record lengths. Every rejection here is a correct one and every non-rejection is a miss, since the null is false by construction:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12031)
reps = 20_000
sd = 0.012
mu = 0.30 / np.sqrt(252) * sd                      # the course's Sharpe 0.30, assumed TRUE

print("  a real edge of Sharpe 0.30, tested 20,000 times at four track-record lengths")
print("    years        n   median p   5th pct p   95th pct p   P(p<0.05)")
for years, n in ((5, 1260), (10, 2520), (24, 6158), (50, 12600)):
    p = np.empty(reps)
    for i in range(0, reps, 2_000):                # chunked to keep memory flat
        x = rng.normal(mu, sd, (2_000, n))
        t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n))
        p[i:i + 2_000] = 2 * stats.t.sf(np.abs(t), n - 1)
    print(f"    {years:5d}   {n:6d}   {np.median(p):8.4f}   {np.quantile(p, 0.05):9.5f}   "
          f"{np.quantile(p, 0.95):10.4f}   {(p < 0.05).mean():9.4f}")
# =>   a real edge of Sharpe 0.30, tested 20,000 times at four track-record lengths
#        years        n   median p   5th pct p   95th pct p   P(p<0.05)
#            5     1260     0.3974     0.02056       0.9372      0.1009
#           10     2520     0.3109     0.00924       0.9254      0.1585
#           24     6158     0.1375     0.00176       0.8555      0.3150
#           50    12600     0.0347     0.00016       0.6062      0.5591
```

Start with the twenty-four-year row, because it is the course's own experiment. The median p-value across twenty thousand replications is $0.1375$, and the lesson's single realization on real SPY data was $p = 0.135$. The observed p-value was not bad luck and not a sign the effect is absent: it is almost exactly the *typical* outcome of testing a genuine Sharpe-$0.30$ edge on twenty-four years of daily data. What the same row says next is harder to unsee. The central ninety percent of those p-values runs from $0.00176$ to $0.8555$ — a spread of nearly five hundred fold, on identical truth, identical sample size and identical code. A replication returning $0.002$ and a replication returning $0.86$ are both routine, and a literature containing both would read as a contradiction when it is a single distribution sampled twice.

The rightmost column converts this into the only currency that matters. With a real edge of $0.30$ and twenty-four years of data, the study returns $p<0.05$ just $31.50\%$ of the time. Five years returns it $10.09\%$ of the time, with a median p-value of $0.3974$ — a real strategy, correctly tested, that will look like nothing on nine attempts in ten. Even fifty years of daily history only reaches $55.91\%$, and its median p-value has just crossed the conventional bar at $0.0347$. Every one of these numbers is about a strategy that genuinely works.

**Quoting a p-value to three decimals describes the sample precisely and the phenomenon not at all, because the third decimal is the part that would move most if the study were run again.**

!!! note "The level, the p-value, the false-positive rate and the false-discovery rate are four numbers between zero and one routinely called 'the five percent', and only the first is chosen by the analyst"
    The **level** is a design constant fixed in advance and belongs to [The Hypothesis Testing Framework](01-hypothesis-testing-framework.md). The **p-value** is a statistic computed from the data and is this page. The **false-positive rate** is the long-run frequency of rejecting true nulls, which equals the size and is a property of the procedure, not of any one result — the accounting is [Type I and Type II Errors](04-type-i-and-type-ii-errors.md). The **false-discovery rate** is the share of *rejections* that are wrong, which depends on how many hypotheses were tested and how many were true, and is [Part XV](../part-15-multiple-testing/index.md). Only the first is chosen; the second is observed; the third follows from the first; and the fourth cannot be computed from a single test at all, no matter how small its p-value.

## Any False Null Drives the p-Value to Zero, So the Number Measures Surprise and Says Nothing About Size

For a fixed alternative, a consistent test rejects with probability tending to one, and the p-value tends to zero — typically exponentially in $n$. This is a desirable property and it has an unwelcome corollary: since $n$ appears in the limit and the effect size does not, the p-value can be driven arbitrarily small by an arbitrarily negligible effect, provided enough data. A small p-value is evidence that the null is false. It carries no information about *by how much*.

??? note "Proof that the p-value collapses at a rate governed by $n\delta^{2}$, so the effect size and the sample size are interchangeable in it and only their product is identified"

    Take a statistic that is asymptotically normal in the standard way — $T_n=\sqrt n\,(\hat\delta-0)/\sigma$ with $\hat\delta\to\delta$ — so that under the null $T_n\Rightarrow\mathcal{N}(0,1)$ and the two-sided p-value is $p_n=2\Phi(-|T_n|)$. Under a fixed alternative $\delta\neq0$, $T_n\approx\sqrt n\,\delta/\sigma\to\pm\infty$, so $p_n\to0$. The rate follows from the normal tail bound $\Phi(-z)\le\tfrac12e^{-z^{2}/2}$:
    $$\log p_n\ \lesssim\ -\frac{T_n^{2}}{2}\ \approx\ -\frac{n\,\delta^{2}}{2\sigma^{2}},$$
    so the p-value decays exponentially, at a rate set by $n\delta^{2}$ and by nothing else.

    That product is the whole content of the section. The p-value is a function of $n\delta^2$ to leading order, so halving the effect and quadrupling the sample leaves it unchanged: $n$ and $\delta^2$ are interchangeable inside it and only their product is identified. No rearrangement recovers $\delta$ from $p$ without knowing $n$, which is why a p-value cannot report an effect size even in principle.

    The load-bearing hypothesis is that $\delta$ is held *fixed* as $n$ grows. Under local alternatives $\delta_n=h/\sqrt n$ the product $n\delta_n^{2}=h^{2}$ is constant, $T_n$ has a non-degenerate limit, and the p-value converges to a genuine random variable rather than to zero — which is the regime in which power comparisons are meaningful and is how [Statistical Power](05-statistical-power.md) sets its questions up. **The p-value collapses on the product of what you found and how long you looked, and reports only the product.**

The two orderings can therefore be exactly reversed:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12037)
reps = 400

def median_t(rho, n):                              # AR(1), vectorised across replications
    e = rng.normal(0, 1, (reps, n + 1))
    x = np.empty((reps, n + 1))
    x[:, 0] = e[:, 0]
    for j in range(1, n + 1):
        x[:, j] = rho * x[:, j - 1] + e[:, j]
    a, b = x[:, 1:], x[:, :-1]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.median(r * np.sqrt(n - 1) / np.sqrt(1 - r**2))

print("  AR(1) reversal, H0: rho_1 = 0, at four effect sizes and four sample sizes")
print("      rho        n   median t     median p   R^2 = rho^2   rank by R^2   rank by p")
rows = [(rho, n, median_t(rho, n)) for rho, n in
        ((-0.40, 60), (-0.20, 250), (-0.086, 6_410), (-0.030, 60_000))]
rows = [(rho, n, t, 2 * stats.norm.sf(abs(t)), rho**2) for rho, n, t in rows]
r2rank = {v: i + 1 for i, v in enumerate(sorted((r[4] for r in rows), reverse=True))}
prank = {v: i + 1 for i, v in enumerate(sorted(r[3] for r in rows))}
for rho, n, t, p, r2 in rows:
    print(f"   {rho:6.3f}   {n:6d}   {t:9.2f}   {p:10.2e}   {r2:11.5f}   "
          f"{r2rank[r2]:11d}   {prank[p]:9d}")
print(f"  the strongest evidence sits on the weakest effect: R^2 "
      f"{rows[0][4] / rows[3][4]:.0f}x larger, p {rows[0][3] / rows[3][3]:.1e} times weaker")
# =>   AR(1) reversal, H0: rho_1 = 0, at four effect sizes and four sample sizes
#          rho        n   median t     median p   R^2 = rho^2   rank by R^2   rank by p
#       -0.400       60       -3.48     4.95e-04       0.16000             1           3
#       -0.200      250       -3.26     1.10e-03       0.04000             2           4
#       -0.086     6410       -6.91     4.72e-12       0.00740             3           2
#       -0.030    60000       -7.33     2.25e-13       0.00090             4           1
#      the strongest evidence sits on the weakest effect: R^2 178x larger, p 2.2e+09 times weaker
```

The third row is the course's series: a lag-one autocorrelation of $-0.086$ on $6{,}410$ observations gives a median $t$ of $-6.91$, matching the lesson's pinned `-6.9 se`, and a p-value of $4.72\times10^{-12}$. By any conventional reading that is overwhelming evidence, and it is — of the proposition that the true autocorrelation is not exactly zero. The $R^2$ column prices the same finding at $0.00740$: the predictable share of tomorrow's variance is three-quarters of one percent, which is the arithmetic behind the course's "worth well under a basis point of daily predictability before costs."

Now read the two rank columns against each other. Ordered by effect size the rows run $1,2,3,4$ top to bottom; ordered by p-value they run $3,4,2,1$. The proof above says the ordering should follow $n\rho^{2}$, and it does: the four rows carry $9.6$, $10.0$, $47.4$ and $54.0$, which reproduces the p-ranking exactly, with the top two separated by $0.4$ and therefore effectively tied. The largest effect in the table, an autocorrelation of $-0.400$ with an $R^2$ of $0.16$, carries only the third-strongest p-value; the smallest, $-0.030$ with an $R^2$ of $0.00090$, carries the strongest. The final line states the extreme pair: the top row's effect is $178$ times larger in $R^2$ and its p-value is weaker by more than nine orders of magnitude. Nothing has gone wrong in any of the four rows. Each p-value is a correct answer to the question "how surprising is this data under the null", and that question's answer is a function of the effect *and* the sample size, in which the second term dominates once the first is nonzero.

**A p-value is a statement about how much data was collected as much as about what is in it, which is why it can be made arbitrarily impressive by an effect nobody would trade.**

!!! warning "A p-value quoted to three decimals is being read as a measurement when it is a draw, and the spread is free to compute"
    The convention of reporting a bare p-value invites two readings that the number cannot support: that $0.03$ and $0.07$ are meaningfully different, and that a replication would land nearby. Section 4 measured both. Against a real Sharpe-$0.30$ edge on twenty-four years of data the central ninety percent of p-values spans $0.00176$ to $0.8555$, so the distance between $0.03$ and $0.07$ is far inside the noise of a single realization, and a replication landing anywhere in that range is unremarkable. The temptation is strongest exactly where the stakes are highest, because a p-value near the threshold is the case in which the decision hinges on the digit that moves most. **The free diagnostic is to simulate your own study: take your own point estimate as the truth, generate ten thousand replications at your own sample size, run the identical testing code, and print the 5th and 95th percentiles of the resulting p-values — if that interval spans an order of magnitude, and at realistic effect sizes it will, then the third decimal of your p-value is decoration and any conclusion that depends on it is a conclusion about this sample rather than about the market.**

## The p-Value Answers One Question Exactly and Is Read as the Answer to Three Others

This page established that the p-value is the smallest level at which the data rejects, hence a statistic computed entirely under the null; that its defining property is super-uniformity at every level simultaneously rather than uniformity, which is the boundary case where the test attains its level; that a lattice null makes the inequality strict, so a nominal $5\%$ binomial test has size $0.0107$ at $n=10$ and spends $21.5\%$ of its budget, while the mid-p repair reads $0.0547$, $0.0577$ and $0.0595$ at the three smallest sample sizes and is therefore not level $0.05$; that a genuine Sharpe-$0.30$ edge tested on twenty-four years of daily data has a median p-value of $0.1375$ against the lesson's realized $0.135$, with a $5$th-to-$95$th spread of $0.00176$ to $0.8555$ and a $31.50\%$ chance of clearing the conventional bar; and that ordering four findings by p-value reverses their ordering by effect size, the smallest effect carrying the strongest evidence by nine orders of magnitude.

The question the p-value answers exactly is: *if the null and every assumption behind the statistic's null law were true, how often would data at least this extreme arise?* The three questions it is read as answering are whether the null is true, whether the effect is large, and whether a replication would agree. It is silent on the first because no prior or alternative entered the computation; silent on the second because $n$ enters the limit and the effect size does not; and silent on the third in the strongest possible way, since section 4 measured the disagreement between replications directly and found it spanning three orders of magnitude on a real effect.

What remains is the accounting the p-value cannot do on its own. A threshold turns the number into a decision, that decision can be wrong in two ways, and the two are not symmetric, not equally repairable and not summarizable by any single figure — including the one this page has been about. That is [Type I and Type II Errors](04-type-i-and-type-ii-errors.md).

**The p-value is a precise answer to a question about a world assumed false, and it is quoted as though it described the one the trade will be placed in.**
