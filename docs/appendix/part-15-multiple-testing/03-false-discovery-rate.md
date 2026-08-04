# False Discovery Rate

The false discovery rate is usually introduced as a gentler family-wise error rate, and the description is wrong in a way that matters for how the output should be read. It is not a weaker bound on the same quantity; it is the expectation of a *ratio* whose denominator is random, and an expectation controlled at $q$ says nothing about the realization on the one dataset anybody has. Below, Benjamini–Hochberg controls it at $0.1029$ on a family where $84.18\%$ of realizations have a false discovery proportion of exactly zero and $5\%$ have three quarters or more of their discoveries false, so the guaranteed number describes no run of the procedure that ever occurs. Dependence makes this worse in a direction the mean cannot show: as correlation rises from $0$ to $0.5$ the controlled average *falls* from $0.0897$ to $0.0607$, looking safer, while the worst realization rises from $0.2330$ to $1.0000$. And what it buys against a family-wise procedure is real and large — $0.7049$ of true effects found against Bonferroni's $0.1963$ — bought with a family-wise error rate of $0.9978$.

This page covers the Benjamini–Hochberg procedure and the sharp form of the bound it attains, the distribution of the false discovery proportion behind the expectation that is controlled, what dependence does to the mean and to the tail in opposite directions, the adaptive refinement that estimates the fraction of true nulls, and the Bayesian object the whole construction is approximating. It does not establish the $2\times2$ table or the family of rates definable on it, which is [Multiple Comparisons](01-multiple-comparisons.md); it controls no family-wise rate and derives no step-down dominance, which is [Bonferroni Correction](02-bonferroni-correction.md); it charges nothing for a search of unrecorded width, which is [Data Snooping Bias](04-data-snooping-bias.md); it resamples no joint distribution, which is [White's Reality Check](05-whites-reality-check.md); it computes no posterior and assumes no prior, which is [Part XVI](../part-16-bayesian-statistics/index.md); it inverts no power function, which is [Statistical Power](../part-12-hypothesis-testing/05-statistical-power.md); and it never reports a controlled average as though it were a property of the run in front of it.

The trading stake is the course putting a family of calendar effects through both criteria and having neither one save it. [Seasonality and Calendar Effects](../../part-04-strategy-development/04-seasonality-and-calendar-effects.md) assembles five weekdays, twelve months, turn-of-month and Sell-in-May and reports `19 calendar effects tested: 1 pass raw 5%, 0 survive BH-FDR, 0 survive Bonferroni`, with September the lone raw survivor at $p=0.043$. The lesson's reading is that "even it finds nothing", and that is the correct verdict on a family where the honest estimate of the true-null fraction is essentially one — which is section 4's regime, where the whole adaptive apparatus of this page is measured to be worth $0.00$ extra discoveries.

## Benjamini–Hochberg Controls the Expectation of a Ratio at $m_0q/m$, Which Is the Family-Wise Rate Exactly When Every Null Is True and Progressively Looser as the Science Improves

The procedure is three lines and the theorem behind it is one of the few in this part that is genuinely sharp rather than a bound nobody attains.

??? note "Proof that the step-up procedure at level $q$ controls $\mathbb{E}[V/R\cdot\mathbb{1}\{R>0\}]$ at $m_0q/m$ under independence, and that this equals the family-wise error rate when $m_0=m$"

    Order the p-values $p_{(1)}\le\dots\le p_{(m)}$, let $k=\max\{i:p_{(i)}\le iq/m\}$, and reject $H_{(1)},\dots,H_{(k)}$, rejecting nothing if no such $i$ exists. The quantity controlled is
    $$\mathrm{FDR}=\mathbb{E}\!\left[\frac{V}{R}\,\mathbb{1}\{R>0\}\right],$$
    and the indicator is not decoration. On a realization with $R=0$ the ratio is $0/0$; the convention assigns it zero, so a procedure that rejects nothing has an FDP of zero rather than an undefined one. Any statement about the FDR is therefore a statement averaged over runs that made no discoveries at all, which section 2 shows is most of them when discoveries are scarce.

    For the bound, fix a true null $i$ and note that the procedure rejects it exactly when $p_i\le Rq/m$ with $R$ the final count. Decompose over the value of $R$:
    $$\mathbb{E}\!\left[\frac{V}{R}\mathbb{1}\{R>0\}\right]=\sum_{i\in\mathcal{N}}\sum_{r=1}^{m}\frac{1}{r}P\big(p_i\le rq/m,\ R=r\big).$$
    The step-up construction has the property that the event $\{R=r\}$ can be arranged, for each $i$, so that the inner sum telescopes: independence of $p_i$ from the others gives $P(p_i\le rq/m,\ R=r\mid\cdot)=\tfrac{rq}{m}P(R=r\mid\cdot)$ for a uniform null p-value, and summing $\tfrac{1}{r}\cdot\tfrac{rq}{m}=\tfrac{q}{m}$ over a partition of the probability space contributes $q/m$ per true null. Summing over $\mathcal{N}$,
    $$\mathrm{FDR}\le m_0\cdot\frac{q}{m}=\frac{m_0}{m}\,q,$$
    with equality for continuous, independent null p-values.

    Two consequences follow immediately and both are load-bearing later. When every null is true, $m_0=m$ and $V=R$, so the FDP is $1$ whenever anything is rejected and $0$ otherwise; its expectation is then $P(R\ge1)=P(V\ge1)$, which is the family-wise error rate exactly. **Under the global null the false discovery rate *is* the family-wise error rate, so a procedure controlling one controls the other, and the two criteria can only diverge to the extent that real effects exist.** And the factor $m_0/m$ means the procedure is conservative by exactly the fraction of hypotheses that are false — it delivers $q$ only when nothing is real, and $0.7q$ when three tenths are, which is the slack section 4's refinement is built to reclaim.

    The load-bearing distinction is between the two random quantities. Bonferroni pins $V$, whose behaviour does not depend on how much real signal is present; Benjamini–Hochberg pins $V/R$, and $R$ grows with the signal. A criterion whose denominator improves as the science improves is a genuinely different object from one whose numerator is bounded outright, and the gap between them is not a matter of strictness.

## The Controlled Average Is Attained Almost Exactly, and It Describes the Realization Only When Discoveries Are Plentiful

The bound and the procedure's cost against a family-wise alternative are both measurable, and so is the distribution behind the expectation:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(15031)
m, m1, delta, q, reps = 1000, 100, 3.2, 0.10, 20_000
m0 = m - m1
bh_t = q * np.arange(1, m + 1) / m
holm_t = 0.05 / (m - np.arange(m))

mu = np.r_[np.zeros(m0), np.full(m1, delta)]
p = 2 * stats.norm.sf(np.abs(rng.standard_normal((reps, m)) + mu))
order = np.argsort(p, axis=1)
ps = np.take_along_axis(p, order, axis=1)
rank = np.empty_like(order)
np.put_along_axis(rank, order, np.arange(m)[None, :], axis=1)

und = ps <= bh_t
nbh = np.where(und.any(1), m - np.argmax(und[:, ::-1], axis=1), 0)
over = ps > holm_t
nhm = np.where(over.any(1), np.argmax(over, axis=1), m)

print(f"  {m:,} tests, {m0} true nulls, {m1} real effects at delta = {delta},"
      f" {reps:,} families")
print("    procedure         E[R]   E[V]   power     FDR    FWER   P(FDP>2q)"
      "   q50 FDP   q95 FDP   max FDP")
for tag, rej in (("uncorrected", p < 0.05),
                 ("Bonferroni", p < 0.05 / m),
                 ("Holm", rank < nhm[:, None]),
                 ("Benjamini-Hochberg", rank < nbh[:, None])):
    V, S = rej[:, :m0].sum(1), rej[:, m0:].sum(1)
    R = V + S
    fdp = np.where(R > 0, V / np.maximum(R, 1), 0.0)
    print(f"    {tag:18s} {R.mean():6.1f} {V.mean():6.1f}  {S.mean() / m1:6.4f}"
          f"  {fdp.mean():6.4f}  {(V >= 1).mean():6.4f}   {(fdp > 2 * q).mean():9.4f}"
          f"  {np.quantile(fdp, 0.50):8.4f}  {np.quantile(fdp, 0.95):8.4f}"
          f"  {fdp.max():8.4f}")

print(f"  Benjamini-Hochberg at q = {q}, holding {m:,} tests and varying how many"
      f" are real")
print("      m1   m0*q/m     FDR     E[R]   P(R=0)   P(FDP=0)   q50 FDP   q95 FDP"
      "   max FDP")
for k in (2, 5, 20, 100, 300):
    mu = np.r_[np.zeros(m - k), np.full(k, delta)]
    pp = 2 * stats.norm.sf(np.abs(rng.standard_normal((reps, m)) + mu))
    pss = np.sort(pp, axis=1)
    u = pss <= bh_t
    n = np.where(u.any(1), m - np.argmax(u[:, ::-1], axis=1), 0)
    cut = np.where(n > 0, pss[np.arange(reps), np.maximum(n, 1) - 1], -1.0)
    rj = pp <= cut[:, None]
    V, S = rj[:, :m - k].sum(1), rj[:, m - k:].sum(1)
    R = V + S
    fdp = np.where(R > 0, V / np.maximum(R, 1), 0.0)
    print(f"    {k:4d}   {(m - k) * q / m:6.4f}  {fdp.mean():6.4f}  {R.mean():7.2f}"
          f"  {(R == 0).mean():7.4f}   {(fdp == 0).mean():8.4f}"
          f"  {np.quantile(fdp, 0.50):8.4f}  {np.quantile(fdp, 0.95):8.4f}"
          f"  {fdp.max():8.4f}")
# =>   1,000 tests, 900 true nulls, 100 real effects at delta = 3.2, 20,000 families
#        procedure         E[R]   E[V]   power     FDR    FWER   P(FDP>2q)   q50 FDP   q95 FDP   max FDP
#        uncorrected         134.3   45.1  0.8926  0.3340  1.0000      0.9999    0.3333    0.3878    0.4601
#        Bonferroni           19.7    0.0  0.1963  0.0024  0.0464      0.0000    0.0000    0.0000    0.1333
#        Holm                 19.8    0.0  0.1976  0.0024  0.0474      0.0000    0.0000    0.0000    0.1333
#        Benjamini-Hochberg   77.6    7.1  0.7049  0.0902  0.9978      0.0011    0.0889    0.1481    0.2333
#      Benjamini-Hochberg at q = 0.1, holding 1,000 tests and varying how many are real
#          m1   m0*q/m     FDR     E[R]   P(R=0)   P(FDP=0)   q50 FDP   q95 FDP   max FDP
#           2   0.0998  0.1029     0.76   0.4990     0.8418    0.0000    0.7500    1.0000
#           5   0.0995  0.0978     1.94   0.2017     0.7633    0.0000    0.5000    1.0000
#          20   0.0980  0.0987    10.68   0.0008     0.3631    0.0909    0.2727    1.0000
#         100   0.0900  0.0901    77.60   0.0000     0.0018    0.0886    0.1474    0.2346
#         300   0.0700  0.0700   270.58   0.0000     0.0000    0.0695    0.0968    0.1443
```

The first table is the trade in one line each. Benjamini–Hochberg finds $0.7049$ of the real effects against Bonferroni's $0.1963$ and Holm's $0.1976$ — three and a half times the power, which is why the procedure exists. Its realized FDR is $0.0902$ against a promise of $q=0.10$ and a theoretical $m_0q/m=0.09$, so the bound of section 1 is attained to the third decimal rather than merely respected. And the price is in the column beside it: BH's family-wise error rate is $0.9978$, meaning it is essentially certain to reject at least one true null, where Bonferroni's is $0.0464$. That is not a defect. It is the criterion doing precisely what it was designed to do, and anyone who wanted $P(V\ge1)$ small should not have changed criteria.

The `q95 FDP` and `max FDP` columns say the average is a fair summary *here*: the median realization has an FDP of $0.0889$, the ninety-fifth percentile $0.1481$, and the worst of twenty thousand families $0.2333$. Nothing is far from $0.09$.

The second table is what happens when that stops being true, and it is the honest limit of the criterion. Holding the family at a thousand tests and reducing how many are real, the FDR column tracks $m_0q/m$ perfectly — $0.1029$ against $0.0998$, $0.0978$ against $0.0995$, $0.0987$ against $0.0980$, $0.0901$ against $0.0900$, $0.0700$ against $0.0700$ — so the theorem is in no trouble at all. The distribution behind it disintegrates anyway. At two real effects among a thousand tests, half of all families make no discovery whatsoever ($P(R=0)=0.4990$), $84.18\%$ have a false discovery proportion of exactly zero, the median is $0.0000$, the ninety-fifth percentile is $0.7500$ and the maximum is $1.0000$. The controlled mean of $0.1029$ is an average over a distribution with almost all its mass at zero and a thick tail at one, and there is no realization anywhere near $0.10$.

**The false discovery rate is a faithful description of the dataset in front of you exactly when discoveries are plentiful, and becomes an average over a bimodal distribution describing no possible run as they grow scarce** — which is the regime a strategy family sits in, where the honest expectation is that almost nothing is real. Reading "FDR controlled at $10\%$" as "about a tenth of my discoveries are false" is correct at $m_1=300$ and is, at $m_1=2$, a statement about neither the $84\%$ of runs where none of them is nor the $5\%$ where three quarters are.

## Dependence Moves the Mean and the Tail in Opposite Directions, So the Controlled Quantity Looks Safer Exactly as the Realization Gets Worse

Section 2 varied how much signal was present and held the tests independent. Correlated tests are the normal case in strategy research, and the theorem's status under dependence is the most misreported thing about this procedure.

??? note "Proof that the step-up procedure remains valid under positive regression dependence, and that the distribution-free repair costs a factor of $c(m)=\sum_{i\le m}1/i$"

    Benjamini and Yekutieli showed that the $m_0q/m$ bound survives if the p-values are positively regression dependent on the subset of true nulls (PRDS): for any increasing set $D$ and any true null $i$, $P(\mathbf{p}\in D\mid p_i\le u)$ is non-decreasing in $u$. Equicorrelated one-sided normal statistics with non-negative correlation satisfy it, and so do most families of overlapping strategy variants, since they share a common factor with positive loading. Under PRDS, therefore, nothing needs fixing.

    Without any assumption, the same authors show the bound holds after replacing $q$ by $q/c(m)$ with $c(m)=\sum_{i=1}^{m}1/i\approx\log m+\gamma$. The proof reweights the telescoping argument of section 1 across the $m$ possible values of $R$; the harmonic sum is the price of not knowing which value the denominator will take.

    It is worth being precise about how that compares to a family-wise procedure, because the usual summary — that Benjamini–Yekutieli is "nearly Bonferroni" — is wrong at both ends. The step-up thresholds become $iq/(mc(m))$, so at rank $i=1$ the bar is $q/(mc(m))$, which is $c(m)$ times *stricter* than Bonferroni's $q/m$ at the same nominal level, while at rank $i=m$ it is $q/c(m)$, which is enormously more lenient. Benjamini–Yekutieli is not a shifted Bonferroni; it is a procedure that demands more of your single best result and much less of your hundredth.

    The load-bearing asymmetry is that dependence is a statement about the joint law while the FDR is an expectation, and expectations of ratios are not determined by marginals. **Dependence cannot break a bound that positive association already protects, and it can and does reshape the distribution the bound is an average of, so validity under dependence and reliability under dependence are different questions with different answers.**

Both halves are measurable on the same equicorrelated families the previous two pages used:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(15033)
m, m1, delta, q, reps = 1000, 100, 3.2, 0.10, 10_000
m0 = m - m1
cm = float((1 / np.arange(1, m + 1)).sum())
mu = np.r_[np.zeros(m0), np.full(m1, delta)]


def run(rho, thresh):
    """BH with the given step thresholds on an equicorrelated family; returns FDP and R."""
    g = rng.standard_normal((reps, 1))
    p = 2 * stats.norm.sf(np.abs(np.sqrt(rho) * g
                                 + np.sqrt(1 - rho) * rng.standard_normal((reps, m)) + mu))
    ps = np.sort(p, axis=1)
    u = ps <= thresh
    n = np.where(u.any(1), m - np.argmax(u[:, ::-1], axis=1), 0)
    cut = np.where(n > 0, ps[np.arange(reps), np.maximum(n, 1) - 1], -1.0)
    rj = p <= cut[:, None]
    V, S = rj[:, :m0].sum(1), rj[:, m0:].sum(1)
    R = V + S
    return np.where(R > 0, V / np.maximum(R, 1), 0.0), R, S / m1


bh = q * np.arange(1, m + 1) / m
print(f"  {m:,} tests, {m1} real effects at delta = {delta}, Benjamini-Hochberg at"
      f" q = {q}, {reps:,} families")
print(f"  the guarantee is on the mean; the columns to its right are the same draws")
print("     rho      FDR   q50 FDP   q95 FDP   P(FDP>0.2)   max FDP     E[R]   power")
for rho in (0.0, 0.2, 0.5, 0.8):
    fdp, R, pw = run(rho, bh)
    print(f"    {rho:4.2f}   {fdp.mean():6.4f}  {np.quantile(fdp, 0.50):8.4f}"
          f"  {np.quantile(fdp, 0.95):8.4f}   {(fdp > 0.2).mean():10.4f}"
          f"  {fdp.max():8.4f}  {R.mean():7.1f}  {pw.mean():6.4f}")

print(f"  Benjamini-Yekutieli divides q by c(m) = sum 1/i = {cm:.4f}, valid under any"
      f" dependence")
print(f"    its bar for the single most significant test is q/(m*c(m)) ="
      f" {q / (m * cm):.3e},")
print(f"    against Bonferroni's q/m = {q / m:.3e} at the same nominal level"
      f" -- stricter by {cm:.2f}x")
print("     rho   BY FDR   BY q95 FDP     BY E[R]   BY power   power kept vs BH")
for rho in (0.0, 0.5):
    fdp, R, pw = run(rho, bh / cm)
    _, _, pw_bh = run(rho, bh)
    print(f"    {rho:4.2f}   {fdp.mean():6.4f}   {np.quantile(fdp, 0.95):10.4f}"
          f"   {R.mean():9.1f}   {pw.mean():8.4f}   {pw.mean() / pw_bh.mean():17.1%}")
# =>   1,000 tests, 100 real effects at delta = 3.2, Benjamini-Hochberg at q = 0.1, 10,000 families
#      the guarantee is on the mean; the columns to its right are the same draws
#         rho      FDR   q50 FDP   q95 FDP   P(FDP>0.2)   max FDP     E[R]   power
#        0.00   0.0897    0.0889    0.1477       0.0009    0.2330     77.6  0.7048
#        0.20   0.0875    0.0581    0.2727       0.0898    0.9302     76.3  0.6898
#        0.50   0.0607    0.0000    0.3711       0.0785    1.0000     77.0  0.6570
#        0.80   0.0387    0.0000    0.1453       0.0465    1.0000     88.7  0.6473
#      Benjamini-Yekutieli divides q by c(m) = sum 1/i = 7.4855, valid under any dependence
#        its bar for the single most significant test is q/(m*c(m)) = 1.336e-05,
#        against Bonferroni's q/m = 1.000e-04 at the same nominal level -- stricter by 7.49x
#         rho   BY FDR   BY q95 FDP     BY E[R]   BY power   power kept vs BH
#        0.00   0.0121       0.0476        40.3     0.3981               56.5%
#        0.50   0.0074       0.0213        39.3     0.3850               59.2%
```

The `FDR` column is the theorem holding and then some. It reads $0.0897$, $0.0875$, $0.0607$, $0.0387$ as $\rho$ climbs to $0.8$ — never above the $m_0q/m=0.09$ bound, and falling steadily below it. Read alone, that column says dependence is harmless and even helpful, which is the conclusion most treatments stop at.

The columns beside it say the opposite about the same draws. The probability that more than a fifth of the discoveries are false goes $0.0009$, $0.0898$, $0.0785$, $0.0465$ — a hundredfold rise from independence to $\rho=0.2$. The ninety-fifth percentile of the FDP goes $0.1477$, $0.2727$, $0.3711$, $0.1453$. The worst realization goes $0.2330$, $0.9302$, $1.0000$, $1.0000$: under correlation it is entirely possible for *every* discovery in a family to be false, which under independence never once happened in twenty thousand tries. And the median goes $0.0889$, $0.0581$, $0.0000$, $0.0000$ — at $\rho\ge0.5$ the typical family has no false discoveries at all.

Those three facts are one fact. A common factor makes the family's p-values rise and fall together, so a run is either clean or contaminated wholesale, and the mean of a variable that is usually $0$ and occasionally $1$ can be small while no realization is ever near it. That is [Multiple Comparisons](01-multiple-comparisons.md)'s finding that dependence leaves the count's mean alone and multiplies its spread, arriving one level up. **A falling average false discovery rate under rising dependence is not the procedure getting safer; it is the same tail risk being reported by a statistic that cannot see tails.**

The Benjamini–Yekutieli panel prices the distribution-free repair, and it is expensive for something the family probably did not need. Its harmonic factor is $c(m)=7.4855$ at a thousand tests, so its bar for the single most significant p-value is $1.336\times10^{-5}$ against Bonferroni's $1.000\times10^{-4}$ at the same nominal level — $7.49$ times stricter, which is the opposite of the "nearly Bonferroni" summary. The realized FDR collapses to $0.0121$ and $0.0074$, an order of magnitude below the $0.10$ requested, and power falls to $0.3981$ and $0.3850$, keeping $56.5\%$ and $59.2\%$ of what plain BH achieved. Since equicorrelated positive dependence is PRDS and plain BH was valid throughout — as the first panel's FDR column confirms — that is roughly half the power spent insuring against a dependence structure the family did not have.

## The Adaptive Refinement Estimates the True-Null Fraction Correctly and Is Worth Nothing When That Fraction Is One

Section 1 showed the procedure is conservative by exactly $m_0/m$. Knowing that fraction would recover the gap, and it is estimable: under the null p-values are uniform, so the mass above a cut $\lambda$ is almost entirely null and can be extrapolated.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(15035)
m, delta, q, lam, reps = 1000, 3.2, 0.10, 0.5, 20_000
step = q * np.arange(1, m + 1) / m


def bh_count(ps, scale):
    """Number rejected by BH at level q/scale, from sorted p-values."""
    u = ps <= step[None, :] / scale[:, None]
    return np.where(u.any(1), m - np.argmax(u[:, ::-1], axis=1), 0)


print(f"  {m:,} tests, Benjamini-Hochberg at q = {q} against Storey's adaptive version,"
      f" lambda = {lam}, {reps:,} families")
print("     pi0    mean pi0hat   sd pi0hat     BH FDR   Storey FDR     BH R"
      "   Storey R   extra discoveries")
for pi0 in (1.00, 0.95, 0.90, 0.70, 0.50):
    k = int(round(m * (1 - pi0)))
    m0 = m - k
    mu = np.r_[np.zeros(m0), np.full(k, delta)]
    p = 2 * stats.norm.sf(np.abs(rng.standard_normal((reps, m)) + mu))
    ps = np.sort(p, axis=1)
    p0h = np.minimum((p > lam).sum(1) / ((1 - lam) * m), 1.0)
    out = []
    for scale in (np.ones(reps), p0h):
        n = bh_count(ps, scale)
        cut = np.where(n > 0, ps[np.arange(reps), np.maximum(n, 1) - 1], -1.0)
        rj = p <= cut[:, None]
        V, R = rj[:, :m0].sum(1), rj.sum(1)
        out.append((np.where(R > 0, V / np.maximum(R, 1), 0.0).mean(), R.mean()))
    print(f"    {pi0:4.2f}   {p0h.mean():12.4f}   {p0h.std():9.4f}   {out[0][0]:8.4f}"
          f"   {out[1][0]:10.4f}   {out[0][1]:6.1f}   {out[1][1]:8.1f}"
          f"   {out[1][1] - out[0][1]:17.2f}")
# =>   1,000 tests, Benjamini-Hochberg at q = 0.1 against Storey's adaptive version, lambda = 0.5, 20,000 families
#         pi0    mean pi0hat   sd pi0hat     BH FDR   Storey FDR     BH R   Storey R   extra discoveries
#        1.00         0.9873      0.0185     0.0988       0.1003      0.1        0.1                0.00
#        0.95         0.9502      0.0291     0.0950       0.1002     33.6       34.3                0.67
#        0.90         0.9009      0.0302     0.0903       0.1006     77.7       80.4                2.72
#        0.70         0.7034      0.0267     0.0700       0.0998    270.6      292.6               21.93
#        0.50         0.5054      0.0229     0.0500       0.0993    466.9      523.1               56.18
```

The estimator works, and it is worth saying so before saying what follows. Storey's $\hat\pi_0$ reads $0.9873$, $0.9502$, $0.9009$, $0.7034$ and $0.5054$ against truths of $1.00$, $0.95$, $0.90$, $0.70$ and $0.50$, with a standard deviation of two or three points. The `BH FDR` column confirms the conservativeness it is aimed at, tracking $\pi_0q$ exactly at $0.0988$, $0.0950$, $0.0903$, $0.0700$ and $0.0500$. And the adaptive version reclaims it in full: Storey's realized FDR is $0.1003$, $0.1002$, $0.1006$, $0.0998$ and $0.0993$, sitting on the nominal $0.10$ in every row. As a piece of statistics this is a complete success.

The last column is what it purchases. Extra discoveries run $0.00$, $0.67$, $2.72$, $21.93$ and $56.18$ as the fraction of real effects rises from nothing to a half. The gain is proportional to $1-\pi_0$ because that is precisely the slack being reclaimed — Storey multiplies the threshold by $1/\hat\pi_0$, and when $\hat\pi_0\approx1$ the multiplier is one.

The top row is strategy research. A family in which essentially nothing is real returns $\hat\pi_0=0.9873$ and $0.00$ extra discoveries, and the two procedures reject $0.1$ hypotheses apiece. The machinery is not misbehaving; it is reporting, accurately, that there is no conservativeness to reclaim because the family is very nearly all null. **The adaptive refinement's value is exactly the fraction of hypotheses that are true, so it pays best in domains where discoveries are common and returns nothing in the one domain that most needs the power** — and a research programme that finds Storey's correction making a visible difference to its results has learned something about its own $\pi_0$ that it should probably examine directly.

## The Whole Construction Is a Frequentist Route to a Posterior Probability, and the Tail It Leaves Uncontrolled Has Procedures of Its Own

The false discovery rate has an interpretation that makes its behaviour in sections 2 and 3 unsurprising. Under a two-groups model in which each hypothesis is null with probability $\pi_0$ and the p-value is uniform if null, the probability that a rejected hypothesis at threshold $t$ is null is $\pi_0t/F(t)$ with $F$ the marginal p-value distribution — which is the false discovery rate as a *posterior* quantity, and its pointwise version, the local fdr evaluated at the observed statistic rather than averaged over the rejection region, is the object a researcher usually means. [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md) already carries the arithmetic in the form that matters for trading: at a prior of one in a hundred, a significant result is overwhelmingly likely to be false regardless of the test's quality, and Storey's $\hat\pi_0$ of $0.9873$ in section 4 is that prior being estimated from the family rather than asserted.

This is also why the criterion is the right one for a research pipeline and the wrong one for a single decision. A pipeline advancing many candidates cares about the share of its shortlist that is junk, and can absorb a known dud rate; a committee approving one strategy for capital cares about that strategy, and the FDR says nothing about any individual rejection — a discovery in a set with FDR $0.10$ is not $90\%$ likely to be real, because the average is over the set and over repetitions of the whole procedure.

The finance literature's canonical application is Harvey, Liu and Zhu's audit of the published cross-section of expected returns, which applies these criteria to hundreds of factor tests and concludes that a new factor needs a $t$-statistic above about $3.0$ rather than $2.0$ to be credible. Their choice of Benjamini–Yekutieli over Benjamini–Hochberg is section 3's panel as an editorial decision: factor tests are dependent, the dependence structure is unknown rather than merely positive, and they preferred the distribution-free bound and paid the harmonic factor for it. The same audit supplies the input page 4 argues cannot be recovered — an estimate of how many factor tests were run and never published.

What sections 2 and 3 leave open is the tail, and it has its own literature. Controlling $P(\mathrm{FDP}>\gamma)\le\alpha$ rather than $\mathbb{E}[\mathrm{FDP}]\le q$ is false discovery exceedance, and Lehmann and Romano give step-down procedures for it; controlling $P(V\ge k)$ is the $k$-family-wise error rate of [Multiple Comparisons](01-multiple-comparisons.md), which answers the same worry by bounding the count instead of the share. Both cost power relative to Benjamini–Hochberg, and both deliver a statement about the realization rather than about its average, which is the thing section 2 showed the false discovery rate cannot supply when discoveries are scarce.

!!! note "The false discovery rate, the false discovery proportion, the positive false discovery rate, the local fdr and false discovery exceedance are five objects built on one ratio, and only one of them is a statement about your dataset"
    **The false discovery proportion** is $V/R$, the realized share of discoveries that are false — a random variable, observable only if the truth is known, and the quantity everyone actually cares about. **The false discovery rate** is its expectation with the $\{R>0\}$ convention, $\mathbb{E}[V/R\cdot\mathbb{1}\{R>0\}]$, which is what Benjamini–Hochberg controls and which section 2 measures at $0.1029$ on a family where $84.18\%$ of realizations sit at zero. **The positive false discovery rate** conditions instead of padding, $\mathbb{E}[V/R\mid R>0]$, which removes the runs that discovered nothing and is therefore always the larger of the two and not controllable at any fixed level without further assumptions. **The local fdr** is the pointwise posterior null probability at an observed statistic rather than an average over the whole rejection region, so it is the only member of the list that says something about a *particular* discovery, and it is larger than the FDR at the boundary of the rejection region and smaller deep inside it. **False discovery exceedance** is $P(V/R>\gamma)$, a tail probability rather than a mean, and it is the criterion sections 2 and 3 keep implicitly asking for. The distinction that matters operationally is that the middle three are properties of a procedure averaged over repetitions, while the local fdr is a property of a result and the exceedance is a property of a realization.

!!! warning "A controlled false discovery rate and a badly contaminated result are entirely compatible, and dependence makes the average look better as it makes the realization worse"
    Section 3 measured the realized FDR falling from $0.0897$ to $0.0607$ as correlation rose from $0$ to $0.5$, while over the same draws the worst realization rose from $0.2330$ to $1.0000$, the ninety-fifth percentile from $0.1477$ to $0.3711$, and the probability that more than a fifth of discoveries were false from $0.0009$ to $0.0785$. Every one of those families satisfied the guarantee. The mechanism is that a common factor makes p-values move together, so runs become clean or contaminated wholesale, and a mean is the wrong summary of a bimodal variable — section 2 shows the extreme version, a family with two real effects where the controlled mean is $0.1029$ and $84.18\%$ of realizations have an FDP of exactly zero while $5\%$ exceed $0.7500$. None of this is visible in a result. The procedure prints a threshold and a list of discoveries; it does not print the distribution its guarantee is an average over, and the guarantee gets *tighter*-looking precisely as the family gets more correlated. **The free diagnostic is to simulate your own family under its own null and report the ninety-fifth percentile of the false discovery proportion beside the nominal $q$ — the same null simulation [Bonferroni Correction](02-bonferroni-correction.md) prescribes for calibrating a threshold, read at a different quantile — and if the two numbers differ by a factor of three or more, say so when the discoveries are reported, because the average is then describing a run that does not happen.** Where the family is small enough that most runs discover nothing, quote $P(R=0)$ as well: a criterion averaged over a majority of empty runs is not summarizing the non-empty ones.

## An Average Over Runs That Did Not Happen

This page established that Benjamini–Hochberg controls $\mathbb{E}[V/R\cdot\mathbb{1}\{R>0\}]$ at $m_0q/m$ with the bound attained rather than merely respected — measured at $0.1029$, $0.0978$, $0.0987$, $0.0901$ and $0.0700$ against predictions of $0.0998$, $0.0995$, $0.0980$, $0.0900$ and $0.0700$ — and that under the global null it coincides exactly with the family-wise error rate; that it buys $0.7049$ of the real effects against Bonferroni's $0.1963$ and pays for them with a family-wise rate of $0.9978$ against $0.0464$; that the controlled average describes the realization when discoveries are plentiful and stops doing so as they thin, with a family of two real effects returning a mean of $0.1029$ while $84.18\%$ of runs have an FDP of exactly zero, half discover nothing at all, and the ninety-fifth percentile is $0.7500$; that dependence moves the mean and the tail in opposite directions, the mean falling $0.0897$ to $0.0387$ while the maximum rises $0.2330$ to $1.0000$ and the median collapses to $0.0000$; that the distribution-free repair costs a harmonic factor of $7.4855$, making its bar for the best single test $7.49$ times stricter than Bonferroni's while keeping only $56.5\%$ of the power, to insure against a dependence the family's positive correlation already made safe; and that Storey's estimator recovers the $m_0/m$ slack exactly, reading $0.9873$ to $0.5054$ against truth and restoring the realized rate to a nominal $0.10$ in every row, for a gain running $0.00$, $0.67$, $2.72$, $21.93$, $56.18$ discoveries — nothing at all in the regime where nothing is real.

The shape shared by all four exhibits is that every guarantee on this page is an expectation, and an expectation is a property of a procedure repeated rather than of the dataset it was applied to. That was also true of [Bonferroni Correction](02-bonferroni-correction.md), where the realized family-wise rate was measured an order of magnitude below the nominal one, but a probability of at least one error is at least a statement about a single run's outcome. A ratio of two random quantities is not, and this page's three failures — the bimodality when discoveries are scarce, the tail that dependence inflates while shrinking the mean, and the adaptive gain that vanishes exactly where power is most wanted — are three readings of that one fact.

Both criteria now stand fully described, and both take the same input on trust. The family-wise rate divides by $m$; the false discovery rate ranks against $iq/m$; each is exact arithmetic given the number of hypotheses, and neither has any way to learn that number from the data. What the count actually is, why no dataset records it, and how large the resulting error is, is [Data Snooping Bias](04-data-snooping-bias.md).

**Controlling the false discovery rate guarantees that the average share of false discoveries is small across many repetitions of a procedure nobody repeats, and the run you have is drawn from a distribution the guarantee never describes.**
