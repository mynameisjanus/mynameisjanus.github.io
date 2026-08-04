# Multiple Comparisons

The standard warning about testing many hypotheses is that the error rate explodes, and it is imprecise in the one way that decides everything after it: *which* rate explodes is a choice, and one of the two obvious candidates does not move at all. Below, fifty null tests taken from independent to correlated at $0.90$ produce expected false-rejection counts of $2.4990$, $2.4979$ and $2.4702$ against a prediction of $2.50$ that assumed nothing whatever about their dependence, while over the identical draws the probability that at least one of them fires runs $0.9234$, $0.6209$ and $0.1917$. The two published shortcuts for how many independent tests a correlated family is worth overstate the honest answer by $1.99\times$ and $1.36\times$ at $\rho=0.5$ and, being functions of the correlation matrix alone, return those same numbers at $\alpha=0.01$ where the honest answer has moved from $19.10$ to $22.79$. And five defensible error rates computed from one set of draws at one threshold read $0.044999$, $1.0000$, $1.0000$, $0.3445$ and $1.0000$.

This page covers the two quantities a family of tests makes available and which of them dependence can touch, the effective number of independent tests hiding inside a correlated family and the two heuristics that get it wrong in the same direction, the $2\times2$ table of outcomes and the error rates definable on it, and the question of what counts as a family at all. It does not divide a level by anything, which is [Bonferroni Correction](02-bonferroni-correction.md); it controls no proportion of discoveries, which is [False Discovery Rate](03-false-discovery-rate.md); it charges nothing for a search whose width went unreported, which is [Data Snooping Bias](04-data-snooping-bias.md); it resamples no family jointly, which is [White's Reality Check](05-whites-reality-check.md); it derives no sampling distribution for a maximum and no expected best of $N$ nulls, which is [Sampling Distributions](../part-10-statistics-foundations/03-sampling-distributions.md); it defines neither error of a single test nor the frontier between them, which is [Type I and Type II Errors](../part-12-hypothesis-testing/04-type-i-and-type-ii-errors.md); it inverts no power function into a sample size, which is [Statistical Power](../part-12-hypothesis-testing/05-statistical-power.md); and it never counts a test that was run and not written down.

The trading stake is a screen the course runs and then convicts. [Mean Reversion and Pairs Trading](../../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md) forms every pair from thirteen investable series and reports `12 of 78 pairs reject at 5% (luck alone would find 3.9)`, then puts the twelve in front of five years they have never seen and gets `still cointegrated on 2020-2025: 1 of 12`. That $3.9$ is section 1's dependence-free count, $78\times0.05$, and it is correct whether or not overlapping sector funds make the tests dependent — which they emphatically do. The lesson's verdict that "roughly a third of the in-sample list was expected to be luck" is the only reading of that family the arithmetic supports, and section 5 is about why the family was seventy-eight rather than one.

## The Expected Count of False Rejections Is Fixed by the Level Alone While the Probability of at Least One Is Not, So the Error Rate You Choose Decides Whether Dependence Is Relevant

A family of $m$ tests, of which $m_0$ have true nulls, supports two immediate summaries of how much damage the nulls do: how many of them are rejected, and whether any of them is. They behave completely differently, and almost every confusion in this part comes from treating them as one quantity.

??? note "Proof that the expected number of false rejections is $m_0\alpha$ under arbitrary dependence, while the probability of at least one lies anywhere in $[\alpha,\min(1,m_0\alpha)]$ and equals $1-(1-\alpha)^{m_0}$ only under independence"

    Let $R_i=\mathbb{1}\{\text{test }i\text{ rejects}\}$ and let $V=\sum_{i\in\mathcal{N}}R_i$ count rejections among the $m_0$ true nulls. Each test has exact level $\alpha$, so $\mathbb{E}[R_i]=P(R_i=1)=\alpha$ for $i\in\mathcal{N}$. Expectation is linear whatever the joint law, so
    $$\mathbb{E}[V]=\sum_{i\in\mathcal{N}}\mathbb{E}[R_i]=m_0\alpha,$$
    and no term in that sum saw another. The joint distribution of $(R_1,\dots,R_{m_0})$ never enters, which is why the count is invariant to every correlation structure a family can have. Exactness matters here and conservativeness does not survive it: for a discrete test with $P(R_i=1)<\alpha$ the identity becomes the bound $\mathbb{E}[V]\le m_0\alpha$.

    The probability that at least one fires is $\mathrm{FWER}=P(V\ge1)=P\big(\bigcup_{i\in\mathcal{N}}\{R_i=1\}\big)$, and a union is not linear. Two bounds are immediate and both are attained. Boole's inequality gives $P(\bigcup R_i)\le\sum P(R_i)=m_0\alpha$, with equality exactly when the rejection events are disjoint. And the union contains any one of its members, so $P(\bigcup R_i)\ge\alpha$, with equality when the events coincide — the case of perfectly dependent tests, which is one test written $m_0$ times. Under mutual independence the complement factorizes, $P(V=0)=\prod(1-\alpha)=(1-\alpha)^{m_0}$, giving the familiar $1-(1-\alpha)^{m_0}$, and [Independence](../part-02-probability-foundations/05-independence.md) is where the factorization of the *complements* is licensed rather than assumed. That value sits strictly inside the bracket for $m_0\ge2$, so the independence formula is neither the worst nor the best case; it is one point in a range spanning $\alpha$ to $m_0\alpha$.

    The load-bearing asymmetry is that a sum of indicators has a dependence-free mean and a dependence-sensitive law. **A researcher who asks how many false positives to expect has a question the family's correlation structure cannot affect, and a researcher who asks whether any of them is false has a question that cannot be answered without it** — and on a family of fifty the two questions differ by a factor of twenty.

The bracket is worth holding onto because it says something the independence formula alone hides. As tests become more redundant the family-wise probability falls toward the level of a single test, which is exactly right: fifty perfectly correlated tests *are* one test, and there is nothing to correct. Redundancy is a discount rather than a penalty. What it discounts is the probability, never the count.

## Both Rates Are Measurable on the Same Draws and Only One of Them Moves, While Dependence Attacks the Count's Spread Instead of Its Mean

The identity and the bracket are both checkable on simulated families of null tests, with a common factor tuning the dependence:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(15011)
alpha, reps = 0.05, 100_000
crit = stats.norm.isf(alpha / 2)

print(f"  global null, two-sided tests at alpha = {alpha}, {reps:,} families each row")
print("      m    rho     FWER   1-(1-a)^m       E[V]   m*alpha   se(E[V])")
for m in (5, 20, 50, 200):
    for rho in (0.0, 0.5, 0.9):
        g = rng.standard_normal((reps, 1))
        z = np.sqrt(rho) * g + np.sqrt(1 - rho) * rng.standard_normal((reps, m))
        v = (np.abs(z) > crit).sum(axis=1)
        print(f"    {m:3d}   {rho:4.2f}   {(v > 0).mean():6.4f}   {1 - (1 - alpha) ** m:9.4f}"
              f"   {v.mean():8.4f}   {m * alpha:7.2f}   {v.std() / np.sqrt(reps):8.4f}")
# =>   global null, two-sided tests at alpha = 0.05, 100,000 families each row
#          m    rho     FWER   1-(1-a)^m       E[V]   m*alpha   se(E[V])
#          5   0.00   0.2252      0.2262     0.2488      0.25     0.0015
#          5   0.50   0.1822      0.2262     0.2490      0.25     0.0019
#          5   0.90   0.1022      0.2262     0.2534      0.25     0.0028
#         20   0.00   0.6431      0.6415     1.0048      1.00     0.0031
#         20   0.50   0.4217      0.6415     1.0094      1.00     0.0059
#         20   0.90   0.1537      0.6415     0.9755      1.00     0.0104
#         50   0.00   0.9234      0.9231     2.4990      2.50     0.0049
#         50   0.50   0.6209      0.9231     2.4979      2.50     0.0138
#         50   0.90   0.1917      0.9231     2.4702      2.50     0.0261
#        200   0.00   1.0000      1.0000    10.0004     10.00     0.0097
#        200   0.50   0.9039      1.0000    10.0489     10.00     0.0531
#        200   0.90   0.2547      1.0000     9.8108     10.00     0.1028
```

The $E[V]$ column is the identity and it does not move. At $m=50$ it reads $2.4990$, $2.4979$ and $2.4702$ against $m\alpha=2.50$ as $\rho$ goes $0$, $0.5$, $0.9$; at $m=200$ it reads $10.0004$, $10.0489$ and $9.8108$ against $10.00$. Every one of the twelve rows sits within about two of its own Monte Carlo standard errors of the prediction, and the prediction was made without looking at $\rho$.

The FWER column is the bracket being traversed. At $m=50$ it runs $0.9234$, $0.6209$, $0.1917$ — from near-certainty to less than a fifth — while the independence formula holds at $0.9231$ throughout and is right in exactly one of the three rows. At $m=200$ the independence formula reads $1.0000$ where the correlated truth is $0.2547$. Quoting a family-wise error rate from $1-(1-\alpha)^{m}$ on a correlated family is not a conservative approximation; it is a number about a different family, and on a strategy grid it is wrong by a factor of four.

The `se(E[V])` column is the part that does move, and it is a finding rather than a diagnostic. At $m=200$ the Monte Carlo standard error of the mean count inflates from $0.0097$ to $0.1028$ as $\rho$ goes $0$ to $0.9$ — a factor of $10.6$ on the identical number of families, which means the count itself is ten times more dispersed. **Dependence does not touch the mean of the false-rejection count and multiplies its spread tenfold, so a correlated family delivers the expected number of false positives on average and almost never on the occasion in front of you.** That is the same object [Monte Carlo Simulation](../part-09-monte-carlo-methods/03-monte-carlo-simulation.md) reaches from the other side when it finds a correlated grid has a lower expected maximum and a fatter upper tail — "a correlated grid is less likely to manufacture a good-looking winner and more likely to manufacture a spectacular one" — and it is the shape that returns in [False Discovery Rate](03-false-discovery-rate.md) as a controlled mean sitting on an uncontrolled tail.

## A Correlated Family Behaves Like a Smaller Independent One, and Both Published Shortcuts for How Much Smaller Overstate It and Cannot Depend on the Level

Section 2 says a family of fifty correlated tests is worth fewer than fifty independent ones. Turning "fewer" into a number is the natural next move, and it is where a widely-used class of shortcut breaks.

??? note "Proof that the effective test count implied by a family's error rate is a function of the level as well as the correlation, so no statistic of the correlation matrix alone can report it"

    Define the effective number of tests $M_{\rm eff}$ as the count of *independent* level-$\alpha$ tests that would produce the family's observed error rate: the solution of $\mathrm{FWER}=1-(1-\alpha)^{M}$, namely
    $$M_{\rm eff}(\alpha)=\frac{\log\big(1-\mathrm{FWER}(\alpha)\big)}{\log(1-\alpha)}.$$
    This is a definition rather than a theorem, and it is the only one that makes $M_{\rm eff}$ mean what it is used to mean, because it is the number you would substitute into a correction to get the level right.

    The argument $\alpha$ on both sides is not decoration. $\mathrm{FWER}(\alpha)=P(\max_i|Z_i|>c_\alpha)$ is a tail probability of the family's maximum, and how much a common factor suppresses that tail depends on how far out the tail is. Deep in the tail an exceedance requires the common factor itself to be extreme, so the exceedance events coincide more strongly and the family looks *more* redundant; nearer the centre the idiosyncratic components still separate the tests and the family looks *less* redundant. So $M_{\rm eff}$ is a function of two arguments, the correlation structure and the level, and it rises as $\alpha$ falls.

    The shortcuts in circulation are both functions of the eigenvalues $\lambda_1,\dots,\lambda_m$ of the correlation matrix alone. Cheverud and Nyholt propose $M_{\rm eff}=1+(m-1)\big(1-\operatorname{var}(\lambda)/m\big)$; Li and Ji propose $M_{\rm eff}=\sum_i\big[\mathbb{1}\{\lambda_i\ge1\}+(\lambda_i-\lfloor\lambda_i\rfloor)\big]$. Both are motivated by the observation that an independent family has every $\lambda_i=1$ while a perfectly redundant one has a single $\lambda_1=m$, and both interpolate between those poles. Neither takes $\alpha$ as an argument.

    The load-bearing observation is that a function of one variable cannot equal a function of two. **Any effective-test-count heuristic computed from the correlation matrix alone returns one number for every significance level, so if it is calibrated at one level it is wrong at all the others**, and the direction of that error is not free either, since the true count rises as the level falls.

Both claims are measurable on an equicorrelated family, whose eigenvalues are $1+(m-1)\rho$ once and $1-\rho$ with multiplicity $m-1$, so the heuristics have closed forms and the truth can be calibrated by simulation:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(15013)
m, reps = 50, 200_000


def nyholt(lam):
    """Cheverud-Nyholt effective test count from the correlation eigenvalues."""
    return 1 + (len(lam) - 1) * (1 - lam.var() / len(lam))


def li_ji(lam):
    """Li-Ji effective test count: an integer part plus a fractional part per eigenvalue."""
    return float(((lam >= 1) + (lam - np.floor(lam))).sum())


print(f"  m = {m} equicorrelated null tests, {reps:,} families per row")
print("    alpha    rho     FWER   M_eff calibrated   Nyholt   over   Li-Ji   over")
for alpha in (0.05, 0.01):
    crit = stats.norm.isf(alpha / 2)
    for rho in (0.0, 0.2, 0.5, 0.8, 0.95):
        g = rng.standard_normal((reps, 1))
        z = np.sqrt(rho) * g + np.sqrt(1 - rho) * rng.standard_normal((reps, m))
        fwer = (np.abs(z) > crit).any(axis=1).mean()
        cal = np.log(1 - fwer) / np.log(1 - alpha)
        lam = np.r_[1 + (m - 1) * rho, np.full(m - 1, 1 - rho)]
        ny, lj = nyholt(lam), li_ji(lam)
        print(f"    {alpha:5.2f}   {rho:4.2f}   {fwer:6.4f}   {cal:16.2f}"
              f"   {ny:6.2f}   {ny / cal:4.2f}x   {lj:5.2f}   {lj / cal:4.2f}x")
# =>   m = 50 equicorrelated null tests, 200,000 families per row
#        alpha    rho     FWER   M_eff calibrated   Nyholt   over   Li-Ji   over
#         0.05   0.00   0.9231              50.01    50.00   1.00x   50.00   1.00x
#         0.05   0.20   0.8667              39.28    48.08   1.22x   41.00   1.04x
#         0.05   0.50   0.6245              19.10    37.99   1.99x   26.00   1.36x
#         0.05   0.80   0.2997               6.95    19.27   2.77x   11.00   1.58x
#         0.05   0.95   0.1372               2.88     6.66   2.32x    4.00   1.39x
#         0.01   0.00   0.3960              50.16    50.00   1.00x   50.00   1.00x
#         0.01   0.20   0.3428              41.76    48.08   1.15x   41.00   0.98x
#         0.01   0.50   0.2047              22.79    37.99   1.67x   26.00   1.14x
#         0.01   0.80   0.0878               9.14    19.27   2.11x   11.00   1.20x
#         0.01   0.95   0.0346               3.51     6.66   1.90x    4.00   1.14x
```

The independent rows are the control and both heuristics pass it: at $\rho=0$ the calibrated count is $50.01$ and $50.16$ against a construction of exactly fifty, and both shortcuts return $50.00$. Everything after those rows is disagreement.

At $\alpha=0.05$ a family correlated at $0.5$ — a mild figure for fifty variants of one trading rule — is worth $19.10$ independent tests. Cheverud–Nyholt says $37.99$ and Li–Ji says $26.00$, overstating by $1.99\times$ and $1.36\times$. At $\rho=0.8$ the truth is $6.95$ against $19.27$ and $11.00$, which is $2.77\times$ and $1.58\times$. Both shortcuts err upward at every correlated row, and an overstated $M_{\rm eff}$ is a correction that is too harsh — so the effect of using either is to spend power buying a level you already had.

The two blocks together are the proof's second half made visible. Every entry in the `Nyholt` and `Li-Ji` columns repeats exactly between the $\alpha=0.05$ block and the $\alpha=0.01$ block, because those functions never saw $\alpha$. The calibrated column does not repeat: $19.10$ becomes $22.79$ at $\rho=0.5$, $6.95$ becomes $9.14$ at $\rho=0.8$, $2.88$ becomes $3.51$ at $\rho=0.95$. The overstatement factors move with it, Cheverud–Nyholt going from $1.99\times$ to $1.67\times$ and Li–Ji from $1.36\times$ to $1.14\times$. **A correlated family's effective size is a property of the family and the threshold jointly, and every shortcut in common use reports it as a property of the family alone**, which is why one published $M_{\rm eff}$ can be quoted in a study testing at $5\%$ and a study testing at $1\%$ and be wrong in both.

## One Family of Tests Supports Five Defensible Error Rates, and on Identical Draws at One Threshold They Span From Four Hundredths to Certainty

The count and the probability of section 1 are two entries on a longer list. Each is a summary of the same $2\times2$ table of outcomes — $V$ false rejections and $U$ correct retentions among the true nulls, $S$ correct rejections and $T$ misses among the real effects, $R=V+S$ rejections in total — and choosing between them is the decision the next two pages execute:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(15015)
m, m1, delta, reps, k, gam = 1000, 100, 3.0, 20_000, 5, 0.10
m0 = m - m1

mu = np.r_[np.zeros(m0), np.full(m1, delta)]
p = 2 * stats.norm.sf(np.abs(rng.standard_normal((reps, m)) + mu))

print(f"  {m:,} tests, {m0} true nulls, {m1} real effects at delta = {delta},"
      f" {reps:,} families")
rej = p < 0.05
V, S = rej[:, :m0].sum(1), rej[:, m0:].sum(1)
print("    mean counts at a per-test level of 0.05      kept   rejected      total")
print(f"      true null                              {(m0 - V).mean():9.1f}"
      f"  {V.mean():9.1f}  {float(m0):9.1f}")
print(f"      real effect                            {(m1 - S).mean():9.1f}"
      f"  {S.mean():9.1f}  {float(m1):9.1f}")
print(f"      total                                  {(m - V - S).mean():9.1f}"
      f"  {(V + S).mean():9.1f}  {float(m):9.1f}")

print("    per-test level      PCER     FWER   5-FWER      FDR      FDX    power"
      "    E[R]")
for a in (0.05, 0.01, 0.001, 1e-4, 1e-5):
    rej = p < a
    V, S = rej[:, :m0].sum(1), rej[:, m0:].sum(1)
    R = V + S
    fdp = np.where(R > 0, V / np.maximum(R, 1), 0.0)
    print(f"    {a:14.5f}   {V.mean() / m:7.6f}   {(V >= 1).mean():6.4f}"
          f"   {(V >= k).mean():6.4f}   {fdp.mean():6.4f}   {(fdp > gam).mean():6.4f}"
          f"   {S.mean() / m1:6.4f}   {R.mean():5.1f}")
# =>   1,000 tests, 900 true nulls, 100 real effects at delta = 3.0, 20,000 families
#        mean counts at a per-test level of 0.05      kept   rejected      total
#          true null                                  855.0       45.0      900.0
#          real effect                                 14.9       85.1      100.0
#          total                                      869.9      130.1     1000.0
#        per-test level      PCER     FWER   5-FWER      FDR      FDX    power    E[R]
#               0.05000   0.044999   1.0000   1.0000   0.3445   1.0000   0.8508   130.1
#               0.01000   0.008999   0.9999   0.9472   0.1185   0.6831   0.6641    75.4
#               0.00100   0.000897   0.5910   0.0020   0.0225   0.0046   0.3852    39.4
#               0.00010   0.000090   0.0867   0.0000   0.0048   0.0015   0.1860    18.7
#               0.00001   0.000008   0.0080   0.0000   0.0010   0.0044   0.0779     7.8
```

The count table is section 1's identity again, now with real effects present. Among $900$ true nulls, $45.0$ are rejected at a per-test level of $0.05$ — exactly $m_0\alpha$ — and among $100$ real effects, $85.1$ are found. The headline a researcher would write is that the screen produced $130.1$ discoveries; the arithmetic says $45.0$ of them, better than one in three, are nothing.

The first row of the second table is why the phrase "the error rate" has no referent. On one set of draws at one threshold, the per-comparison error rate is $0.044999$, the family-wise error rate is $1.0000$, the probability of at least five false rejections is $1.0000$, the false discovery rate is $0.3445$, and the probability that more than a tenth of the discoveries are false is $1.0000$. These are not five estimates of one quantity that disagree; they are five different quantities, all correctly computed, spanning a factor of twenty-two. Which of them a research programme controls is the entire content of the next two pages.

The columns also fail to order each other, which is why the choice cannot be made by taking the strictest. Between the second and third rows the family-wise rate falls from $0.9999$ to $0.5910$ while the false discovery rate falls from $0.1185$ to $0.0225$ — a factor of $1.7$ against a factor of $5.3$ — because they are sensitive to different things. And $\mathrm{FDX}$, the probability that the false-discovery *proportion* exceeds a tenth, does not decrease at all over the last three rows: $0.0046$, $0.0015$, $0.0044$. The `E[R]` column is the explanation. At a level of $10^{-5}$ the family makes $7.8$ discoveries on average, so a single false one is already more than a tenth of them, and an FDX of $0.0044$ against a family-wise rate of $0.0080$ says better than half of the families with any false rejection at all breach the proportion threshold. **Tightening a threshold reliably reduces the number of false discoveries and does not reliably reduce their share, because it shrinks the denominator at the same time** — which is the pathology [False Discovery Rate](03-false-discovery-rate.md) inherits and has to work around.

## A Family Is Whatever Was Searched Over, and Nothing in the Data Records Where Its Boundary Was Drawn

Every quantity above takes $m$ as given. Nothing so far says where $m$ comes from, and the answer is that it comes from the researcher, which is the weakest link in the apparatus.

The theory is indifferent to what the tests are *of*. Fifty lookback lengths for one momentum rule, seventy-eight candidate pairs from thirteen funds, nineteen calendar cells, hundreds of factors in a published cross-section — as far as sections 1 to 4 are concerned these are one object, and the arithmetic does not care whether the family was assembled by varying a parameter, by enumerating a universe or by reading a literature. [Counting Principles](../part-01-mathematical-foundations/02-counting-principles.md) is where the size of such a family stops being obvious: thirteen series make $\binom{13}{2}=78$ pairs, and a grid of four parameters at ten settings each makes ten thousand tests out of what a research note would describe as one strategy.

One boundary deserves drawing explicitly because it resembles this problem and is not it. This part concerns a family of *hypotheses* examined once. Examining *one* hypothesis repeatedly as data arrives — stopping the moment it clears a threshold — inflates the error rate by a different mechanism, the optional stopping that [Martingales](../part-08-stochastic-processes/10-martingales.md) analyses, and its repair is an alpha-spending schedule rather than a correction for $m$. A research programme can be guilty of both at once, and correcting for the number of strategies does nothing whatever about the number of looks.

The harder boundary is the one nobody can audit. The $m$ entering every formula on this page is the number of tests the researcher *counts*, and a dataset records the tests run against it exactly as faithfully as it records the ones abandoned, which is to say not at all. Variants coded and deleted, entry rules rejected after one glance at an equity curve, the nine ideas that died before this one was born — each was a draw from the null, and no correction can include a test it never saw. That is [Data Snooping Bias](04-data-snooping-bias.md), and it is why the last two pages of this part stop counting candidates and start resampling them.

!!! note "The per-comparison error rate, the family-wise error rate, the $k$-family-wise error rate, the false discovery rate and the false discovery exceedance are five error rates definable on one $2\times2$ table, and they differ in whether they average a count, bound a probability or average a proportion"
    **The per-comparison error rate** is $\mathbb{E}[V]/m$, the expected false rejections per test, and it equals $\alpha$ by construction under any dependence — controlling it is what running uncorrected tests already does, which is why it is the baseline rather than a procedure. **The family-wise error rate** is $P(V\ge1)$, the probability of any false rejection at all, and it is the rate a single irreversible decision needs; it is also the one dependence moves, across the whole bracket from $\alpha$ to $m_0\alpha$ that section 1 derives. **The $k$-family-wise error rate** is $P(V\ge k)$, which relaxes the family-wise rate by tolerating up to $k-1$ mistakes and is the natural criterion when a pipeline can absorb a few duds. **The false discovery rate** is $\mathbb{E}[V/R\cdot\mathbb{1}\{R>0\}]$, the average *share* of discoveries that are false, and it is the only entry on the list whose denominator is random. **The false discovery exceedance** is $P(V/R>\gamma)$, the probability that share breaches a stated bound, and it stands to the false discovery rate as the family-wise rate stands to the per-comparison rate — a tail where the other is a mean. The distinction that matters operationally is which random quantity is being pinned: the first three constrain $V$ directly and are unaffected by how many real effects the family contains, while the last two constrain a ratio and therefore get easier as the science gets better.

!!! warning "A family-wise error rate quoted from the independence formula is a statement about a family nobody tested, and the correlated truth can be four times smaller"
    Section 2 measured $1-(1-\alpha)^{m}$ at $1.0000$ for two hundred tests whose realized family-wise rate was $0.2547$, and section 3 measured two published effective-test-count shortcuts overstating the honest figure by $1.99\times$ and $1.36\times$ at a correlation of $0.5$, then returning identical numbers at a level where the truth had moved by a fifth. Both errors point the same way and neither is visible in any output, because a correction computed from $m$ and $\alpha$ prints no diagnostic and the researcher never sees the family-wise rate it actually delivered. The cost is not a broken guarantee — an overstated $M_{\rm eff}$ is conservative, and conservative corrections do control the level they promised. The cost is power, spent on a redundancy the family did not have, and [Statistical Power](../part-12-hypothesis-testing/05-statistical-power.md) has already established that a strategy research programme has none to spare. **The free diagnostic is to calibrate the number rather than look it up: simulate your own family under the global null once, at your own $\alpha$, count the fraction of runs with any rejection, and read $M_{\rm eff}=\log(1-\mathrm{FWER})/\log(1-\alpha)$ off it — thirty lines and a few seconds, returning a figure that is right at the level you are actually testing at, which no published heuristic can be at more than one.** Write it down before the family is run, because the same simulation performed afterwards is a different and much weaker object.

## Two Rates, One Table, and an $m$ That Comes From the Researcher Rather Than the Data

This page established that the expected number of false rejections is $m_0\alpha$ under arbitrary dependence while the probability of at least one lies anywhere in $[\alpha,\min(1,m_0\alpha)]$, measured at $2.4990$, $2.4979$ and $2.4702$ against $2.50$ while the family-wise rate over the identical draws ran $0.9234$, $0.6209$ and $0.1917$; that the independence formula holds at $0.9231$ across all three of those rows and at $1.0000$ where two hundred correlated tests deliver $0.2547$; that dependence leaves the count's mean untouched and inflates its spread by a factor of $10.6$; that a correlated family's effective size is a function of the level as well as the correlation, running $19.10$ at $\alpha=0.05$ and $22.79$ at $\alpha=0.01$ for one family at $\rho=0.5$, while Cheverud–Nyholt returns $37.99$ and Li–Ji $26.00$ at both, overstating by $1.99\times$ and $1.36\times$ and then by $1.67\times$ and $1.14\times$; and that five defensible error rates on one set of draws at one threshold read $0.044999$, $1.0000$, $1.0000$, $0.3445$ and $1.0000$, with the exceedance rate refusing to fall over the last three rows because tightening a threshold shrinks the denominator as fast as the numerator.

What the three exhibits have in common is that none of them is a correction and all of them are arithmetic that precedes one. Nothing above changed a threshold, dropped a hypothesis or rescued a result; the page is a description of what a family of tests does before anybody intervenes, and that description turns out to contain the choice determining what intervening can even mean. A researcher who has not said which of the five rates is the target has not stated a problem, and a correction chosen without a stated problem is a number applied to a question nobody asked.

Two of those rates have procedures attached, and they are the next two pages. The family-wise rate has the older and cruder repair, the one following directly from the union bound of section 1 and inheriting both its freedom from assumptions and its indifference to everything section 2 measured. That is [Bonferroni Correction](02-bonferroni-correction.md).

**The number of tests is the only input every correction in this part requires, it is the one quantity knowable exactly before any data is touched, and it is the one no dataset can be made to confess afterwards.**
