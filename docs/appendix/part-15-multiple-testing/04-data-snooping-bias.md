# Data Snooping Bias

The two preceding pages built corrections that are exact arithmetic on one input, and this page is about the input. Every formula so far takes the number of hypotheses as given; no dataset records it, and the ways it goes wrong are not small. An analyst who tries variants until one clears $5\%$, reports honestly how many were tried, and applies Bonferroni correctly at that number achieves a realized family-wise error rate of $0.1579$ — the correction was performed properly and the answer is wrong by a factor of $3.16$. A search over a thousand variants that admits to fifty runs at $0.6326$ against a nominal $0.05$. And the instrument built to detect this, the probability of backtest overfitting, is itself a single draw with a standard deviation of $0.1702$: on families where the true value is $0.50$ by construction, it returns anything from $0.0305$ to $0.9831$, and lands more than $0.10$ away from the truth on $55\%$ of them.

This page covers selection bias as the expected maximum under the null and the closed form that predicts it, what a search does to the identity of the winner rather than only to its statistic, the two ways a reported count of hypotheses understates the search that produced it, and the sampling distribution of the overfitting diagnostic itself. It does not derive the distribution of a maximum or its asymptotics, which is [Sampling Distributions](../part-10-statistics-foundations/03-sampling-distributions.md); it establishes no error rate on a $2\times2$ table, which is [Multiple Comparisons](01-multiple-comparisons.md); it divides no level and ranks no p-values, which are [Bonferroni Correction](02-bonferroni-correction.md) and [False Discovery Rate](03-false-discovery-rate.md); it resamples no family jointly and tests no composite null, which are [White's Reality Check](05-whites-reality-check.md) and [Hansen's SPA Test](06-hansens-spa-test.md); it constructs no cross-validation scheme and proves nothing about fold geometry, which is [Cross-Validation](../part-14-model-selection/02-cross-validation.md); and it never proposes that a search can be reconstructed from the data it was run on.

The trading stake is a number the course computes and this page grades. [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) runs combinatorially symmetric cross-validation over the fifty-variant momentum grid, splitting sixteen blocks $\binom{16}{8}=12{,}870$ ways, and reports a probability of backtest overfitting of $0.60$. Section 4 runs the identical construction on families built to have a true value of exactly $0.50$ and finds the statistic's own spread across families is $0.1702$, with an interquartile range of $0.3915$ to $0.6211$. The course's $0.60$ sits inside that range, which does not make it wrong — it makes it a number that a family with nothing whatever to overfit would produce about as often as not.

## Selection Bias Is the Expected Maximum Under the Null, So the Gap Between an In-Sample Winner and Its Future Is a Function of the Search Width and Nothing Else

The bias introduced by picking a winner is not a vague hazard. It is a specific quantity with a closed form, computable before any strategy is coded.

??? note "Proof that the expected in-sample performance of a selected best is the expected maximum of the family's estimates, that it converges to zero out of sample under the null, and that the closed form for it assumes independence"

    Let $\hat\theta_1,\dots,\hat\theta_m$ estimate true values $\theta_1,\dots,\theta_m$ from an in-sample period, with $\hat\theta_k=\theta_k+\varepsilon_k$ and $\mathbb{E}[\varepsilon_k]=0$. The reported figure after a search is $\hat\theta_{(m)}=\max_k\hat\theta_k$, and
    $$\mathbb{E}\big[\hat\theta_{(m)}\big]=\mathbb{E}\Big[\max_k(\theta_k+\varepsilon_k)\Big]\ge\max_k\theta_k,$$
    by Jensen applied to the convex maximum, with the gap widening in $m$. Under the global null every $\theta_k=0$ and the entire reported figure is $\mathbb{E}[\max_k\varepsilon_k]$ — noise, in full. On an independent out-of-sample period the errors are redrawn, so for the *selected* index $w$ the expectation is $\mathbb{E}[\theta_w+\varepsilon'_w]=\mathbb{E}[\theta_w]$, which under the null is zero regardless of $m$. The decay from in-sample to out-of-sample is therefore not a decay at all: it is the difference between a maximum of noise and a fresh draw of it.

    For Gaussian $\varepsilon_k$ with standard deviation $\sigma$ and independence, Bailey and López de Prado give
    $$\mathbb{E}\big[\max_k\varepsilon_k\big]\approx\sigma\Big[(1-\gamma)\,\Phi^{-1}\!\big(1-\tfrac{1}{m}\big)+\gamma\,\Phi^{-1}\!\big(1-\tfrac{1}{me}\big)\Big],\qquad\gamma\approx0.5772,$$
    which is the basis of the deflated Sharpe ratio: a reported Sharpe is compared not against zero but against this benchmark. Two conditions on its use are routinely dropped. The $\sigma$ is the standard error of the *estimate*, so for a Sharpe ratio it is the per-observation quantity of [Statistical Power](../part-12-hypothesis-testing/05-statistical-power.md) rather than an annualized figure, and mixing the two inflates the hurdle. And the display assumes the $m$ trials are independent, whereas [Multiple Comparisons](01-multiple-comparisons.md) established that correlated families explore less and therefore have a *smaller* expected maximum.

    The load-bearing fact is that the formula's only free parameter is $m$, which the data does not contain. **Selection bias is the one quantity in this part that could be computed exactly before a single strategy was coded and cannot be recovered from the results afterwards**, because the sample records what was tested and is silent, in exactly the same way, about what was not.

## The Closed Form Is Accurate to a Hundredth Under Independence, Sets Roughly Double the Honest Hurdle on a Correlated Grid, and Is Silent About Which Candidate a Search Returns

The prediction and its failure mode are both measurable, and so is what happens to the winner itself:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(15041)
T, reps, gam = 12.0, 40_000, 0.5772156649
sig = np.sqrt(1 / T)                       # sd of an annualized Sharpe over T years


def bldp(m):
    """Bailey-Lopez de Prado expected maximum Sharpe under the null, m independent trials."""
    return sig * ((1 - gam) * stats.norm.ppf(1 - 1 / m)
                  + gam * stats.norm.ppf(1 - 1 / (m * np.e)))


def sweep(m, rho):
    """In-sample best and its out-of-sample Sharpe, m null strategies correlated at rho."""
    a = (np.sqrt(rho) * rng.standard_normal((reps, 1))
         + np.sqrt(1 - rho) * rng.standard_normal((reps, m))) * sig
    b = (np.sqrt(rho) * rng.standard_normal((reps, 1))
         + np.sqrt(1 - rho) * rng.standard_normal((reps, m))) * sig
    w = a.argmax(axis=1)
    return a.max(axis=1).mean(), b[np.arange(reps), w].mean()


print(f"  {T:.0f} years in sample and {T:.0f} out, sd of an annualized Sharpe"
      f" {sig:.4f}, all strategies truly worthless, {reps:,} searches")
print("        m   IS best indep   formula   err   IS best rho=0.7   formula   err"
      "   OOS of winner")
for m in (1, 5, 25, 100, 500, 2500):
    i0, o0 = sweep(m, 0.0)
    i7, _ = sweep(m, 0.7)
    f = bldp(m) if m > 1 else 0.0
    print(f"    {m:5d}   {i0:13.4f}   {f:7.4f}   {f - i0:+5.2f}   {i7:15.4f}"
          f"   {f:7.4f}   {f - i7:+4.2f}   {o0:13.4f}")

print(f"  one strategy with a TRUE Sharpe of 0.50 hidden among m-1 worthless ones")
print("        m   P(the real one wins)   E[IS best]   E[OOS of winner]"
      "   OOS of the real one")
for m in (1, 5, 25, 100, 500, 2500):
    a = rng.standard_normal((reps, m)) * sig
    b = rng.standard_normal((reps, m)) * sig
    a[:, 0] += 0.50
    b[:, 0] += 0.50
    w = a.argmax(axis=1)
    print(f"    {m:5d}   {(w == 0).mean():20.4f}   {a.max(axis=1).mean():10.4f}"
          f"   {b[np.arange(reps), w].mean():16.4f}   {b[:, 0].mean():19.4f}")
# =>   12 years in sample and 12 out, sd of an annualized Sharpe 0.2887, all strategies truly worthless, 40,000 searches
#            m   IS best indep   formula   err   IS best rho=0.7   formula   err   OOS of winner
#            1         -0.0025    0.0000   +0.00            0.0028    0.0000   -0.00         -0.0022
#            5          0.3359    0.3443   +0.01            0.1832    0.3443   +0.16         -0.0016
#           25          0.5669    0.5765   +0.01            0.3066    0.5765   +0.27         -0.0019
#          100          0.7237    0.7305   +0.01            0.3972    0.7305   +0.33          0.0007
#          500          0.8767    0.8812   +0.00            0.4817    0.8812   +0.40         -0.0003
#         2500          1.0099    1.0124   +0.00            0.5553    1.0124   +0.46          0.0014
#      one strategy with a TRUE Sharpe of 0.50 hidden among m-1 worthless ones
#            m   P(the real one wins)   E[IS best]   E[OOS of winner]   OOS of the real one
#            1                 1.0000       0.4994             0.4993                0.4993
#            5                 0.7160       0.5636             0.3589                0.5016
#           25                 0.4283       0.6628             0.2166                0.5019
#          100                 0.2392       0.7679             0.1205                0.5001
#          500                 0.1106       0.8918             0.0551                0.4994
#         2500                 0.0448       1.0152             0.0222                0.5031
```

The independent columns are the closed form working, and it works well: predicted $0.3443$, $0.5765$, $0.7305$, $0.8812$, $1.0124$ against measured $0.3359$, $0.5669$, $0.7237$, $0.8767$, $1.0099$, an error of a hundredth of a Sharpe or less at every width. A researcher who searched two and a half thousand worthless variants over twelve years should expect a champion at $1.01$, and the formula says so before the search begins.

The `OOS of winner` column is what that champion is worth: $-0.0022$, $-0.0016$, $-0.0019$, $0.0007$, $-0.0003$, $0.0014$. Zero at every search width. There is no decay curve to draw, because the in-sample figure was never an estimate of anything — a maximum of noise in sample, a fresh draw of noise out of it, and the entire apparent edge is the selection.

The correlated columns are the deflated Sharpe ratio's own failure mode. At $\rho=0.7$ the honest expected maximum is $0.1832$, $0.3066$, $0.3972$, $0.4817$, $0.5553$, while the formula still returns the independent figures, overstating by $0.16$, $0.27$, $0.33$, $0.40$ and $0.46$ of a Sharpe. At $m=2500$ it demands a champion beat $1.0124$ when the family's honest bar is $0.5553$ — nearly double. Since a variant grid is exactly the correlated case, **the deflated Sharpe ratio applied to the family it is most often applied to is not conservative in the harmless direction, it is a hurdle roughly twice as high as the search actually warrants**, and correcting that requires resampling the family rather than evaluating a formula, which is [White's Reality Check](05-whites-reality-check.md).

The second panel is the part that survives even when something real is present, and it is the more uncomfortable half. One strategy with a true Sharpe of $0.50$ is placed among $m-1$ worthless siblings. Its out-of-sample performance is $0.4993$, $0.5016$, $0.5019$, $0.5001$, $0.4994$, $0.5031$ — the real edge is real at every width and is not damaged by the search in any way. What degrades is the search's ability to find it: the probability that the genuine strategy wins its own selection falls $1.0000$, $0.7160$, $0.4283$, $0.2392$, $0.1106$, $0.0448$. Among a hundred candidates the real one is chosen less than a quarter of the time; among twenty-five hundred, less than one time in twenty. The two remaining columns move in opposite directions as this happens — the reported in-sample figure climbs $0.4994$ to $1.0152$ while what the chosen strategy actually delivers falls $0.4993$ to $0.0222$. **A wide search does not merely inflate the number attached to the winner; it changes who the winner is, and past a few dozen candidates the procedure is selecting a lucky null in preference to a genuine edge that was sitting in the same family the whole time.**

## A Reported Count Is a Lower Bound on the Search, and Two Ordinary Research Habits Push It Arbitrarily Far Below

Section 1's formula and the corrections of the preceding pages need $m$. Researchers do report counts, often honestly. The gap between an honestly reported count and the search that generated the result is where the corrections fail:

```python
import numpy as np

rng = np.random.default_rng(15043)
reps, cap = 400_000, 200

print(f"  an analyst tries worthless variants one at a time, stops at the first that")
print(f"  clears alpha, reports m = the number tried, then Bonferroni-corrects at"
      f" alpha/m")
print("    alpha   P(reports)   mean m reported   FWER after the correction"
      "   nominal   inflation")
p = rng.random((reps, cap))
for alpha in (0.10, 0.05, 0.01):
    hit = p < alpha
    found = hit.any(axis=1)
    k = np.argmax(hit, axis=1) + 1                     # 1-based trial number of the stop
    pk = p[np.arange(reps), k - 1]
    claim = found & (pk < alpha / k)
    print(f"    {alpha:5.2f}   {found.mean():10.4f}   {k[found].mean():15.2f}"
          f"   {claim.mean():25.4f}   {alpha:7.2f}   {claim.mean() / alpha:8.2f}x")

print(f"  the file drawer: {reps:,} searches over m_true worthless variants, best one")
print(f"  taken forward, Bonferroni applied at the m the write-up admits to")
print("    m reported   m truly tried   per-test bar   FWER   nominal   inflation")
for rep_m, true_m in ((50, 50), (50, 100), (50, 200), (50, 500), (50, 1000)):
    q = rng.random((reps, true_m))
    f = (q.min(axis=1) < 0.05 / rep_m).mean()
    print(f"    {rep_m:10d}   {true_m:13d}   {0.05 / rep_m:13.5f}   {f:6.4f}"
          f"   {0.05:7.2f}   {f / 0.05:8.2f}x")
# =>   an analyst tries worthless variants one at a time, stops at the first that
#      clears alpha, reports m = the number tried, then Bonferroni-corrects at alpha/m
#        alpha   P(reports)   mean m reported   FWER after the correction   nominal   inflation
#         0.10       1.0000             10.02                      0.2555      0.10       2.56x
#         0.05       1.0000             20.02                      0.1579      0.05       3.16x
#         0.01       0.8665             69.09                      0.0462      0.01       4.62x
#      the file drawer: 400,000 searches over m_true worthless variants, best one
#      taken forward, Bonferroni applied at the m the write-up admits to
#        m reported   m truly tried   per-test bar   FWER   nominal   inflation
#                50              50         0.00100   0.0486      0.05       0.97x
#                50             100         0.00100   0.0950      0.05       1.90x
#                50             200         0.00100   0.1817      0.05       3.63x
#                50             500         0.00100   0.3943      0.05       7.89x
#                50            1000         0.00100   0.6326      0.05      12.65x
```

The first panel contains no dishonesty anywhere. The analyst tries variants one at a time, stops on the first that clears the threshold, reports exactly how many were tried, and applies the correction to exactly that number. Every step is defensible in isolation and the realized family-wise error rate is $0.2555$, $0.1579$ and $0.0462$ against nominal levels of $0.10$, $0.05$ and $0.01$ — inflations of $2.56$, $3.16$ and $4.62$ times.

The mechanism is that the stopping rule is itself part of the search. A count of trials is a sufficient description of a family only if the decision to stop was independent of the results, and here it is the opposite: the analyst stopped *because* of the result, so the reported $m$ is a random variable correlated with the very p-value being corrected. Conditional on stopping at trial $k$, the observed p-value is uniform on $[0,\alpha)$ rather than on $[0,1)$, and comparing it to $\alpha/k$ passes with probability $1/k$ regardless of $k$ — which is why the inflation *grows* as $\alpha$ tightens and the searches get longer. The correction has been applied to a count that the correction's derivation assumed was fixed in advance.

The second panel is the file drawer, and it needs no subtlety at all. Hold the write-up's admitted family at fifty and let the true search widen: the realized rate runs $0.0486$, $0.0950$, $0.1817$, $0.3943$, $0.6326$. At fifty truly tried the correction is exactly right, at $0.97$ times nominal. At a thousand truly tried — five stages of a project, each discarding a grid, with only the last written up — it is $12.65$ times nominal, and a "Bonferroni-corrected, $p<0.05$" result is one that a global null produces on nearly two occasions in three.

Neither panel involves a mistake in the arithmetic. **Both corrections were computed correctly from the number the researcher had, and the number the researcher had was a description of the write-up rather than of the search**, which is a distinction no amount of care with the formula can repair.

## The Diagnostic Built to Detect All of This Is One Draw From a Distribution With a Standard Deviation of Seventeen Points

If the search width cannot be recovered, the remaining hope is a statistic that detects overfitting from the results themselves. Combinatorially symmetric cross-validation is the standard one: split the record into $S$ blocks, form every way of calling half of them in-sample, find the in-sample winner in each, and record how often it lands below the median out of sample. The construction is elegant, needs no knowledge of $m$, and reports a probability. It also has a sampling distribution, and the reason that distribution is wider than it looks is worth establishing before measuring it.

??? note "Proof that the number of combinatorial splits contributes precision about one history and nothing about whether that history is typical, so the effective sample size behind the statistic is governed by the block count rather than the split count"

    Fix a return panel $r$ of $T$ observations on $N$ strategies, cut into $S$ blocks, and let $\widehat{\mathrm{PBO}}(r)=\binom{S}{S/2}^{-1}\sum_{c}\mathbb{1}\{\lambda_c\le0\}$, where $c$ indexes the symmetric splits and $\lambda_c$ records whether the in-sample winner fell below the out-of-sample median. The quantity of interest is $\mathrm{PBO}=\mathbb{E}_r[\widehat{\mathrm{PBO}}(r)]$, an expectation over *panels*.

    The variance of the estimator decomposes over the two sources by the law of total variance:
    $$\operatorname{var}\big(\widehat{\mathrm{PBO}}\big)=\underbrace{\operatorname{var}_r\Big(\mathbb{E}\big[\widehat{\mathrm{PBO}}\mid r\big]\Big)}_{\text{between panels}}+\underbrace{\mathbb{E}_r\Big(\operatorname{var}\big[\widehat{\mathrm{PBO}}\mid r\big]\Big)}_{\text{within a panel}},$$
    and the split count only touches the second term. Conditional on $r$ the average is over a *deterministic* enumeration — every split of a fixed panel is computed exactly, so once all $\binom{S}{S/2}$ of them are used the within-panel term is zero and no further splitting is possible or useful. The first term does not fall at all: it is a property of how much one draw of $T$ observations pins down the family's behaviour, and it is unchanged whether the enumeration used sixteen splits or twelve thousand.

    What controls the first term is the block structure. The $\binom{S}{S/2}$ splits are recombinations of $S$ blocks, any two of them sharing at least $S/2-1$ blocks in common, so the $\lambda_c$ are severely dependent and the enumeration carries roughly the information of $S$ independent pieces rather than $\binom{S}{S/2}$. At $S=16$ that is sixteen against $12{,}870$, a ratio of eight hundred.

    The load-bearing distinction is between precision about a sample and precision about a population. **Enumerating every rearrangement of one dataset measures that dataset exactly and leaves the sampling variability of the dataset itself entirely untouched, so a statistic reported from twelve thousand splits carries a split count where a reader will read a sample size.**

The measurement confirms it:

```python
import numpy as np
from itertools import combinations

rng = np.random.default_rng(15045)
T, N, S, fams = 2520, 50, 16, 200

idx = np.array_split(np.arange(T), S)
bn = np.array([len(b) for b in idx], dtype=float)
M = np.array([[1.0 if j in c else 0.0 for j in range(S)]
              for c in combinations(range(S), S // 2)])
print(f"  CSCV: {T:,} observations, {N} strategies, {S} blocks,"
      f" {M.shape[0]:,} symmetric splits per family")


def sharpes(mask, bs, bq):
    """Annualized Sharpe of every strategy on the blocks selected by each row of mask."""
    n = mask @ bn
    mu = (mask @ bs) / n[:, None]
    var = (mask @ bq) / n[:, None] - mu ** 2
    return mu / np.sqrt(var)


def pbo(r):
    """Probability of backtest overfitting from a (T, N) panel of returns."""
    bs = np.array([r[b].sum(0) for b in idx])
    bq = np.array([(r[b] ** 2).sum(0) for b in idx])
    a, b = sharpes(M, bs, bq), sharpes(1 - M, bs, bq)
    w = a.argmax(axis=1)
    rank = (b < b[np.arange(len(w)), w][:, None]).sum(axis=1)
    return (rank / N < 0.5).mean()


print("  every strategy truly worthless, so the honest PBO is 0.50 by construction")
print("    families   mean    sd     q05     q25     q50     q75     q95    min    max")
vals = np.array([pbo(rng.standard_normal((T, N))) for _ in range(fams)])
qs = np.quantile(vals, [0.05, 0.25, 0.50, 0.75, 0.95])
print(f"    {fams:8d}  {vals.mean():.4f}  {vals.std(ddof=1):.4f}  {qs[0]:.4f}"
      f"  {qs[1]:.4f}  {qs[2]:.4f}  {qs[3]:.4f}  {qs[4]:.4f}  {vals.min():.4f}"
      f"  {vals.max():.4f}")

print("  how often a single family's PBO lands outside a band around the truth")
for w in (0.05, 0.10, 0.20, 0.30):
    print(f"    |PBO - 0.50| > {w:.2f}:{(np.abs(vals - 0.5) > w).mean():9.4f}")
# =>   CSCV: 2,520 observations, 50 strategies, 16 blocks, 12,870 symmetric splits per family
#      every strategy truly worthless, so the honest PBO is 0.50 by construction
#        families   mean    sd     q05     q25     q50     q75     q95    min    max
#             200  0.5046  0.1702  0.2301  0.3915  0.4984  0.6211  0.7833  0.0305  0.9831
#      how often a single family's PBO lands outside a band around the truth
#        |PBO - 0.50| > 0.05:   0.7550
#        |PBO - 0.50| > 0.10:   0.5500
#        |PBO - 0.50| > 0.20:   0.2250
#        |PBO - 0.50| > 0.30:   0.0750
```

Averaged over families the statistic is unbiased and accurate: $0.5046$ against a construction in which every strategy is worthless, so that the in-sample winner is a coin flip out of sample and the true value is exactly $0.50$. As a diagnostic of a *procedure*, CSCV is sound.

The rest of the row is what a single application of it is worth. Across two hundred families identical in every respect except their random draw, the statistic has a standard deviation of $0.1702$, a $5$–$95$ range of $0.2301$ to $0.7833$, and observed extremes of $0.0305$ and $0.9831$. One family in five lands more than $0.20$ from the truth, and more than half land further than $0.10$. A researcher who runs CSCV once on a worthless grid and reads $0.23$ has been told, by a correctly implemented instrument, that their overfitting risk is low.

That the spread is this wide despite $12{,}870$ splits is the proof above arriving as a measurement, and it is the same error the statistic exists to warn about: twelve thousand of anything reads as a large sample, and these are twelve thousand rearrangements of *one* history of *one* family. The enumeration drove the within-panel term to zero, which is why the mean is accurate; the between-panel term it never touched is the entire $0.1702$.

This is where the course's own figure belongs. [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) reports `PBO = 0.60` on the fifty-variant momentum grid using the identical sixteen-block, $12{,}870$-split construction. Against the null distribution above, $0.60$ sits between the median $0.4984$ and the $75$th percentile $0.6211$: it is an unremarkable draw from a family with no signal in it whatsoever. That does not make the lesson's conclusion wrong — its verdict on that grid is supported by the Reality Check p-value and the out-of-sample decay it reports alongside — but the $0.60$ itself carries almost no information on its own, and it is quoted, here and generally, as though it carried a great deal. **A statistic computed from twelve thousand overlapping resamples of one dataset is a precise measurement of that dataset and a single noisy observation of the question anybody is asking.**

## The Searches That Cannot Be Counted at All

Sections 2 and 3 both assume a search exists that could in principle be enumerated: variants coded, families screened, blocks recombined. The residual problem is that most of the search leaves no trace of any kind.

Four kinds of uncounted test are worth separating, because they fail differently. The *abandoned variant* is section 2's file drawer — real code that ran, produced a discouraging number, and was deleted; it is countable in principle and simply is not recorded, and a research log fixes it entirely. The *unexecuted rejection* is worse: an idea considered and discarded after one glance at a chart, or a specification never coded because the analyst already knew from an adjacent project that it would not work. No log captures this, because no test was run, yet the information used to reject it came from the same data and the selection is identical in effect. The *inherited search* is worse again — a strategy chosen because a literature suggested it, where the literature is itself the surviving tail of thousands of unpublished tests run by other people over decades, so the family's true width is a property of a research community rather than of any researcher. And the *specification cascade* is the quiet one: a decision to use log returns, to winsorize at four standard deviations, to drop 2008, to rebalance monthly — each defensible, each made after seeing what the alternative did, and none of them ever described as a hypothesis test.

Two observations follow, and they are the practical content of this page. The first is that only the first category is fixable by discipline, and it is the smallest. The second is that all four have the same signature: a reported result whose search history cannot be reconstructed is not weak evidence, it is *unquantified* evidence — the correction's input is missing rather than uncertain, and there is no conservative default to substitute, because the sensible-looking choice of "use the count I have" is exactly what section 2 measured failing by a factor of twelve.

What remains available is a change of target. Every construction on this page tries to price a search whose width must be known. The last two pages of this part give up on knowing it and resample the family that actually exists — which does not solve the file drawer, and does solve the part of the problem that a correlated grid of *reported* candidates creates, without needing a count at all.

!!! note "Data snooping, backtest overfitting, selection bias, the file-drawer problem, $p$-hacking and the garden of forking paths are six names for one operation, and they differ in who performed it and whether anyone could have counted it"
    **Data snooping** is the general case — reusing one dataset for many questions — and is the term White's test is named against. **Backtest overfitting** is its strategy-research instance, where the many questions are parameter settings and the statistic is a performance metric; section 3's CSCV is built for this case specifically. **Selection bias** names the *consequence* rather than the practice, and section 1 shows it is exactly the expected maximum under the null, which makes it the only member of the list with a closed form. **The file-drawer problem** is the case where the tests were genuinely run and simply not reported, which is section 2's second panel and the one a research log actually fixes. **$p$-hacking** is the active version — continuing to vary the specification until the threshold is cleared — and section 2's first panel measures it at $3.16$ times the nominal rate with every individual step defensible. **The garden of forking paths** is Gelman and Loken's name for the case that needs no repeated testing at all: a single analysis whose choices would have been made differently under different data, so the effective multiplicity exists even though the analyst tried exactly one specification and can honestly report $m=1$. The distinction that matters operationally is that the first five have a countable $m$ that someone failed to record, and the last has no countable $m$ even in principle.

!!! warning "A search width is the one input to every correction in this part, and a reported one is a lower bound whose gap from the truth has no upper bound"
    Section 2 measured a correctly applied Bonferroni correction delivering $0.1579$ against a nominal $0.05$ when the only irregularity was that the analyst stopped searching once something worked, and $0.6326$ against $0.05$ when the write-up admitted to fifty variants out of a thousand. Section 1 measured the reported in-sample figure climbing to $1.0152$ while the selected strategy's true out-of-sample value fell to $0.0222$, and the probability that a genuinely good strategy won its own search falling to $0.0448$ among twenty-five hundred candidates. None of this is visible anywhere in a result: the corrected p-value is printed, the arithmetic is right, and the input it was computed from is a number no reader can check and no dataset can confirm. The direction is always the same — a reported count understates a true search, never the reverse — so every correction in this part is biased toward permissiveness by an unknown and unbounded factor. **The free diagnostic is to write the count down before the search rather than after it: pre-register the family — how many parameter settings, which universes, which metrics, and the stopping rule — and treat the pre-registered number as $m$ even if the search terminates early, since section 2 shows that stopping early is what breaks the correction rather than what saves it.** Where a result's search history genuinely cannot be reconstructed, the honest report is not a corrected p-value with a caveat but the observation that the correction has no input, and the fallback is out-of-sample evidence from data that did not exist when the search was run, which is the only currency this page's failures cannot counterfeit.

## An Exact Correction for a Number Nobody Has

This page established that selection bias is the expected maximum under the null, with the closed form predicting $0.3443$, $0.5765$, $0.7305$, $0.8812$ and $1.0124$ against measured maxima of $0.3359$, $0.5669$, $0.7237$, $0.8767$ and $1.0099$ under independence, while the out-of-sample value of every one of those champions was zero to three decimals; that the same formula overstates by $0.16$ to $0.46$ of a Sharpe on a family correlated at $0.7$, so the deflated Sharpe ratio sets roughly double the honest hurdle on exactly the correlated grids it is used for; that a wide search changes the winner's identity rather than only inflating its statistic, the probability that a genuine $0.50$ Sharpe wins its own selection falling $1.0000$, $0.7160$, $0.4283$, $0.2392$, $0.1106$, $0.0448$ while the reported figure climbed to $1.0152$ and the selection's actual out-of-sample value fell to $0.0222$; that an analyst who stops at the first success and honestly reports the trial count achieves a family-wise rate of $0.2555$, $0.1579$ and $0.0462$ against nominal $0.10$, $0.05$ and $0.01$, and that admitting to fifty variants out of a thousand yields $0.6326$ against $0.05$; and that the probability of backtest overfitting is unbiased at $0.5046$ across families and has a standard deviation of $0.1702$ within them, ranging $0.0305$ to $0.9831$ on data with nothing to overfit, landing more than $0.10$ from the truth on $55\%$ of families and putting the course's own pinned $0.60$ between the null's median and its third quartile.

The three exhibits fail in one shared way and it is not the way the subject is usually taught. Nothing above is a case of a researcher being fooled by a number that was computed wrongly. The maximum-of-null formula is accurate to a hundredth. The Bonferroni corrections were applied to the counts their users actually had. CSCV is unbiased. Each instrument does exactly what its derivation promises, and each is then handed either an input that describes the write-up instead of the search, or a single draw from a distribution whose width is never displayed. This part's earlier pages could say that a procedure was conservative or that a guarantee was an average; this page cannot say even that, because the object being corrected is not mismeasured, it is unobserved.

The response available is to stop requiring the count. A resampling scheme applied to the family of candidates that does exist can price the maximum over *those* candidates exactly, using their real joint distribution rather than an independence assumption, and it needs no $m$ beyond the ones in front of it. It does not recover the file drawer and nothing can. It does dispose of section 1's correlated-family error and section 2's stopping-rule error at once, and it is what [White's Reality Check](05-whites-reality-check.md) builds.

**Every correction in this part is exact arithmetic on the number of hypotheses tested, that number is the only quantity in statistics that is known with certainty at the moment it stops being recoverable, and the interval between those two moments is the entire research process.**
