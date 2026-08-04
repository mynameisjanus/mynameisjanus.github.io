# Bonferroni Correction

The complaint against Bonferroni is that it is too conservative, and the usual evidence offered is that Šidák's exact version permits a slightly larger threshold. That evidence is the wrong two orders of magnitude. Šidák's threshold is larger than Bonferroni's by a factor of $1.0206$ at $m=5$ and $1.0259$ at $m=5000$ — essentially the constant $1+\alpha/2$, whatever $m$ is — and below it buys $0.0021$ of power. The conservatism that matters comes from somewhere else entirely: on fifty tests correlated at $0.95$, Bonferroni delivers a realized family-wise error rate of $0.0044$ against the $0.05$ it was asked for, and a threshold calibrated to that family would have used a per-test level of $0.015614$ rather than $0.001$ and detected real effects $0.9431$ of the time against Bonferroni's $0.7593$. The refinement everybody debates is worth two tenths of a percentage point. The one nobody computes is worth eighteen.

This page covers the union bound as the source of both Bonferroni's validity and its waste, the realized error rate a correlated family actually receives and the power that buys nothing, the step-down refinement that dominates Bonferroni on every draw at no cost in assumptions, and the effect size a corrected threshold demands of a strategy. It does not establish the two rates a family makes available or the bracket the family-wise rate lives in, which is [Multiple Comparisons](01-multiple-comparisons.md); it controls no proportion of discoveries, which is [False Discovery Rate](03-false-discovery-rate.md); it charges nothing for a search that went unreported, which is [Data Snooping Bias](04-data-snooping-bias.md); it resamples nothing and estimates no joint distribution, which is [White's Reality Check](05-whites-reality-check.md); it derives no sample size from a power function, which is [Statistical Power](../part-12-hypothesis-testing/05-statistical-power.md); it proves no property of the union bound itself, which is [Probability Axioms](../part-02-probability-foundations/02-probability-axioms.md); and it never treats a correction as evidence that the tests it corrected were worth running.

The trading stake is the course applying this correction and getting nothing back. [Hypothesis Testing and Multiple Testing](../../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) takes fifty momentum variants and reports that "for fifty variants, the per-test bar drops to 0.001 — a t-statistic near 3.3 rather than 2.0", then runs it and prints `bonferroni at 0.05: 0 survivors`. The lesson's own verdict on the price is that "real but modest edges, the only kind markets leave lying around, systematically fail a bar set that high", and section 4 is that sentence turned into a number: at fifty variants and twenty-four years, the bar is a true annualized Sharpe of $0.8031$, and a genuine $0.30$ edge clears it $0.0525$ of the time.

## The Union Bound Gives a Family-Wise Guarantee Under Any Dependence Whatever, and Šidák's Exact Version Is Larger by a Factor That Does Not Grow With the Number of Tests

Bonferroni is one line of probability, and the line is the same one [Multiple Comparisons](01-multiple-comparisons.md) used to bracket the family-wise rate from above. Reading it as a procedure rather than a bound is the whole construction.

??? note "Proof that testing each of $m$ hypotheses at level $\alpha/m$ controls the family-wise error rate at $\alpha$ for every joint distribution, and that the exact independent threshold exceeds it by the constant factor $1+\alpha/2$ rather than by anything increasing in $m$"

    Let $A_i$ be the event that true null $i$ is rejected, and suppose each test is run at level $\alpha'$. Boole's inequality gives
    $$\mathrm{FWER}=P\Big(\bigcup_{i\in\mathcal{N}}A_i\Big)\le\sum_{i\in\mathcal{N}}P(A_i)=m_0\alpha'\le m\alpha',$$
    so setting $\alpha'=\alpha/m$ yields $\mathrm{FWER}\le\alpha$. Nothing in that chain referenced the joint law of the $A_i$: subadditivity is an axiom of a probability measure rather than a consequence of independence, so the guarantee holds for perfectly correlated tests, negatively associated tests and anything between. That is the entire content of the procedure, and it is why Bonferroni cannot be wrong.

    The price of not looking at the joint law is that the inequality is tight only when the $A_i$ are disjoint — the case in which no two tests could ever reject together, which is the opposite of a strategy family. Under mutual independence the exact requirement is $1-(1-\alpha')^{m}=\alpha$, giving Šidák's threshold $\alpha'_{S}=1-(1-\alpha)^{1/m}$. Expanding for small $\alpha$,
    $$\alpha'_{S}=\frac{\alpha}{m}+\frac{\alpha^{2}}{2m}-\frac{\alpha^{2}}{2m^{2}}+O(\alpha^{3})=\frac{\alpha}{m}\Big(1+\frac{\alpha}{2}\Big)+O(\alpha^{3}),$$
    so the *ratio* $\alpha'_{S}/(\alpha/m)$ tends to $1+\alpha/2$ and the *difference* is $O(\alpha^{2}/m)$, vanishing as $m$ grows. This is worth stating precisely because the folklore runs the other way: Bonferroni is often described as increasingly conservative relative to Šidák in large families, and the arithmetic says the opposite — the two thresholds converge in ratio to a constant $2.5\%$ apart at $\alpha=0.05$ and converge absolutely to each other.

    Šidák's threshold is also more robust than its derivation suggests. Šidák's lemma states that for a centred multivariate normal, $P\big(\bigcap_i\{|Z_i|\le c_i\}\big)\ge\prod_i P(|Z_i|\le c_i)$ for *every* correlation matrix, so for two-sided normal tests the independent calculation is conservative rather than merely exact — it is a valid FWER procedure under arbitrary dependence too, not only under the independence it was derived from.

    The load-bearing fact is that both thresholds are computed from $m$ and $\alpha$ alone and neither ever looks at the data's dependence. **The choice between Bonferroni and Šidák is a choice between two numbers that differ by two and a half percent, and it is made in place of the choice that would have mattered, which is whether to look at the family's joint distribution at all.**

## The Realized Error Rate a Correlated Family Receives Is Far Below the One Requested, and the Power That Buys Is Not Spent on Anything

The proof says Bonferroni's guarantee is an inequality. How slack the inequality is on a realistic family, and what the slack costs, are measurable — against a threshold calibrated by simulation to deliver exactly $\alpha$ on the same family, which is [Multiple Comparisons](01-multiple-comparisons.md)'s effective test count cashed out as a procedure:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(15021)
alpha, m, m1, delta, reps = 0.05, 50, 5, 4.0, 200_000
m0 = m - m1

print("  Sidak against Bonferroni, per-test two-sided thresholds")
print("         m   Bonferroni        Sidak    ratio")
for mm in (5, 50, 5000):
    b, s = alpha / mm, 1 - (1 - alpha) ** (1 / mm)
    print(f"    {mm:6d}   {b:10.8f}   {s:10.8f}   {s / b:6.4f}")


def draw(rho, n, mean):
    """n families of m equicorrelated tests, shifted by mean."""
    g = rng.standard_normal((n, 1))
    return np.sqrt(rho) * g + np.sqrt(1 - rho) * rng.standard_normal((n, m)) + mean


bon, sid = alpha / m, 1 - (1 - alpha) ** (1 / m)
mu = np.r_[np.zeros(m0), np.full(m1, delta)]
print(f"  m = {m} tests ({m0} nulls + {m1} real effects at delta = {delta}),"
      f" {reps:,} families")
print("     rho   FWER uncorr   FWER Bonf   FWER oracle   oracle level"
      "   power Bonf   power Sidak   power oracle")
for rho in (0.0, 0.2, 0.5, 0.8, 0.95):
    cal = np.abs(draw(rho, reps, 0.0)[:, :m0]).max(axis=1)
    ora = 2 * stats.norm.sf(np.quantile(cal, 1 - alpha))
    p = 2 * stats.norm.sf(np.abs(draw(rho, reps, mu)))
    out = []
    for a in (alpha, bon, sid, ora):
        r = p < a
        out += [(r[:, :m0].any(axis=1)).mean(), r[:, m0:].mean()]
    print(f"    {rho:4.2f}   {out[0]:11.4f}   {out[2]:9.4f}   {out[6]:11.4f}"
          f"   {ora:12.6f}   {out[3]:10.4f}   {out[5]:11.4f}   {out[7]:12.4f}")
# =>   Sidak against Bonferroni, per-test two-sided thresholds
#             m   Bonferroni        Sidak    ratio
#             5   0.01000000   0.01020622   1.0206
#            50   0.00100000   0.00102534   1.0253
#          5000   0.00001000   0.00001026   1.0259
#      m = 50 tests (45 nulls + 5 real effects at delta = 4.0), 200,000 families
#         rho   FWER uncorr   FWER Bonf   FWER oracle   oracle level   power Bonf   power Sidak   power oracle
#        0.00        0.9005      0.0439        0.0500       0.001139       0.7613        0.7634         0.7725
#        0.20        0.8407      0.0423        0.0513       0.001228       0.7604        0.7626         0.7781
#        0.50        0.5995      0.0289        0.0492       0.001844       0.7607        0.7629         0.8119
#        0.80        0.2902      0.0128        0.0496       0.005138       0.7608        0.7630         0.8851
#        0.95        0.1343      0.0044        0.0491       0.015614       0.7593        0.7616         0.9431
```

The preamble settles section 1's arithmetic. Bonferroni's threshold at $m=5$ is $0.01000000$ against Šidák's $0.01020622$, and at $m=5000$ it is $0.00001000$ against $0.00001026$ — the ratio moves from $1.0206$ to $1.0259$ across a thousand-fold change in $m$ and never leaves the neighbourhood of $1+\alpha/2=1.025$. Whatever is wrong with Bonferroni, it is not this.

The `FWER Bonf` column is the guarantee being kept and progressively wasted. At $\rho=0$ it reads $0.0439$, already below the $0.05$ requested because the union bound double-counts overlaps that independence still permits. By $\rho=0.95$ it is $0.0044$, an eleventh of the nominal level. The `FWER oracle` column is the control that proves this is slack rather than an artefact: a threshold calibrated by simulation to that family holds at $0.0500$, $0.0513$, $0.0492$, $0.0496$ and $0.0491$ across the whole sweep. Both procedures are valid; only one of them is spending its budget.

The power columns price the two candidate repairs against each other, and they are not close. Šidák improves on Bonferroni by $0.7634$ against $0.7613$ at $\rho=0$ and by $0.7616$ against $0.7593$ at $\rho=0.95$ — about $0.0021$ throughout, two tenths of a percentage point, exactly as a $2.5\%$ threshold adjustment should. The oracle improves on Bonferroni by $0.7725$ against $0.7613$ when the family is independent, which is nearly nothing, and by $0.9431$ against $0.7593$ when it is correlated at $0.95$, which is $18.4$ percentage points. The `oracle level` column says why: it rises from $0.001139$ to $0.015614$, so on a strongly correlated family the honest threshold is nearly sixteen times looser than $\alpha/m$.

One column deliberately does not move. Bonferroni's power reads $0.7613$, $0.7604$, $0.7607$, $0.7608$, $0.7593$ across the entire correlation sweep, because a procedure with a fixed threshold detects each real effect with a probability determined by that effect's own marginal distribution, and the mean number detected is a sum of indicators — [Multiple Comparisons](01-multiple-comparisons.md)'s dependence-free expectation, arriving on the other side of the ledger. **Dependence changes what a fixed-threshold correction delivers and cannot change what it detects, so every bit of the conservatism in the level is a pure loss with nothing bought on the other side of the trade.**

## Holm Rejects Everything Bonferroni Rejects on Every Draw, Under the Same Assumptions and With the Same Guarantee, and Hochberg's Extra Assumption Buys About One Draw in Ten Thousand

The oracle of section 2 needs a simulation of the family's joint distribution, which is the last two pages of this part. There is also a repair that needs nothing at all, costs four lines, and is strictly better than Bonferroni on every dataset that will ever be tested.

??? note "Proof that Holm's step-down procedure controls the family-wise error rate under arbitrary dependence and that its rejection set contains Bonferroni's on every realization"

    Order the p-values $p_{(1)}\le\dots\le p_{(m)}$ and compare each to $\alpha/(m-i+1)$. Let $k$ be the smallest index with $p_{(k)}>\alpha/(m-k+1)$ and reject $H_{(1)},\dots,H_{(k-1)}$, rejecting everything if no such $k$ exists.

    For validity, let $\mathcal{N}$ be the true nulls, $m_0=|\mathcal{N}|$, and suppose at least one is rejected. Let $j$ be the smallest index at which a true null appears in the ordering. Every hypothesis ordered before $j$ is false, so at most $j-1\le m-m_0$ of them exist, giving $j\le m-m_0+1$ and hence $\alpha/(m-j+1)\le\alpha/m_0$. A false rejection requires $p_{(j)}\le\alpha/(m-j+1)\le\alpha/m_0$, so
    $$P(V\ge1)\le P\Big(\min_{i\in\mathcal{N}}p_i\le\alpha/m_0\Big)\le\sum_{i\in\mathcal{N}}P(p_i\le\alpha/m_0)=m_0\cdot\frac{\alpha}{m_0}=\alpha,$$
    the last step being Boole's inequality again. The joint law never appeared, so Holm inherits Bonferroni's indifference to dependence exactly.

    For dominance, suppose Bonferroni rejects $B$ hypotheses, so $p_{(i)}<\alpha/m$ for every $i\le B$. Since $\alpha/m\le\alpha/(m-i+1)$ for all $i\ge1$, each such $p_{(i)}$ also clears its Holm threshold, so no index $i\le B$ can be the first exceedance, and Holm rejects at least $B$. Because both procedures reject a prefix of the same ordering, Holm's rejected set *contains* Bonferroni's — not merely a larger count, the same hypotheses plus possibly more, on every realization rather than on average.

    The load-bearing observation is that Holm's first comparison is against $\alpha/m$, which is Bonferroni's threshold exactly, and every subsequent comparison is looser. **Holm is Bonferroni with the tests that already succeeded removed from the denominator, so it is free in the strict sense that there is no dataset, no dependence structure and no configuration of true nulls on which using it instead costs anything.**

Both properties are checkable, and the second one is checkable in the strong form — not as an average but as a count of counterexamples:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(15023)
alpha, m, m1, rho, reps = 0.05, 50, 5, 0.5, 200_000
m0 = m - m1
step = alpha / (m - np.arange(m))          # alpha/m, alpha/(m-1), ..., alpha


def counts(ps):
    """Rejection counts for Holm (step-down) and Hochberg (step-up) from sorted p."""
    over = ps > step
    holm = np.where(over.any(1), np.argmax(over, axis=1), m)
    und = ps <= step
    hoch = np.where(und.any(1), m - np.argmax(und[:, ::-1], axis=1), 0)
    return holm, hoch


print(f"  m = {m} tests ({m0} nulls + {m1} real effects), equicorrelated at rho = {rho},"
      f" alpha = {alpha}, {reps:,} families")
print("    delta   FWER Bonf   FWER Holm   FWER Hoch   det Bonf   det Holm"
      "   det Hoch   Holm<Bonf   Hoch<Holm   Hoch>Holm")
for delta in (0.0, 2.5, 3.0, 3.5, 4.0):
    mu = np.r_[np.zeros(m0), np.full(m1, delta)]
    g = rng.standard_normal((reps, 1))
    z = np.sqrt(rho) * g + np.sqrt(1 - rho) * rng.standard_normal((reps, m)) + mu
    p = 2 * stats.norm.sf(np.abs(z))
    order = np.argsort(p, axis=1)
    ps = np.take_along_axis(p, order, axis=1)
    nh, nc = counts(ps)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(m)[None, :], axis=1)
    bonf = p < alpha / m
    holm, hoch = rank < nh[:, None], rank < nc[:, None]
    row = []
    for r in (bonf, holm, hoch):
        row += [r[:, :m0].any(axis=1).mean(), r[:, m0:].mean()]
    print(f"    {delta:5.2f}   {row[0]:9.4f}   {row[2]:9.4f}   {row[4]:9.4f}"
          f"   {row[1]:8.4f}   {row[3]:8.4f}   {row[5]:8.4f}"
          f"   {int((bonf & ~holm).any(axis=1).sum()):9d}"
          f"   {int((holm & ~hoch).any(axis=1).sum()):9d}"
          f"   {(hoch.sum(1) > holm.sum(1)).mean():9.5f}")
# =>   m = 50 tests (45 nulls + 5 real effects), equicorrelated at rho = 0.5, alpha = 0.05, 200,000 families
#        delta   FWER Bonf   FWER Holm   FWER Hoch   det Bonf   det Holm   det Hoch   Holm<Bonf   Hoch<Holm   Hoch>Holm
#         0.00      0.0294      0.0294      0.0294     0.0010     0.0010     0.0010           0           0     0.00003
#         2.50      0.0291      0.0300      0.0300     0.2142     0.2163     0.2163           0           0     0.00009
#         3.00      0.0290      0.0303      0.0303     0.3857     0.3896     0.3896           0           0     0.00005
#         3.50      0.0290      0.0304      0.0304     0.5838     0.5888     0.5888           0           0     0.00012
#         4.00      0.0294      0.0310      0.0310     0.7605     0.7653     0.7653           0           0     0.00009
```

The dominance columns are the point and they are exact. Across five configurations at two hundred thousand families each — a million draws — the number of realizations on which Bonferroni rejected something Holm did not is $0$, and the number on which Holm rejected something Hochberg did not is $0$. This is a different kind of evidence from a power comparison: it is not that Holm wins on average, it is that Holm never loses, which is what the dominance half of the proof asserts and what makes the choice free rather than a trade.

What the dominance is worth is small and worth stating plainly. Detections run $0.7605$ for Bonferroni against $0.7653$ for Holm at $\delta=4.0$, and $0.3857$ against $0.3896$ at $\delta=3.0$ — about half a percentage point, in the same currency where section 2's oracle was worth eighteen. Holm's family-wise rate rises correspondingly from $0.0290$ to $0.0303$, still less than two thirds of the nominal $0.05$, because the step-down loosens only the thresholds *after* the first and the first is still $\alpha/m$. The step-down repair recovers a sliver of the waste and leaves the bulk of it untouched, for the reason section 2 identified: the waste is caused by dependence, and Holm looks at the dependence exactly as hard as Bonferroni does, which is not at all.

The last column prices the next refinement. Hochberg's step-up procedure uses identical thresholds and scans from the other end, rejecting everything up to the *largest* index that clears its bar rather than stopping at the first that fails, which requires the p-values to be positively regression dependent rather than merely arbitrary. That extra assumption yields a strictly larger rejection set on $0.00003$ to $0.00012$ of draws — between three and twelve families in a hundred thousand — and the detection columns are identical to four decimals in every row. **Hochberg, Hommel and the rest of the step-up literature are refinements of a refinement worth half a percentage point, and they are the part of this subject with the most written about it and the least at stake.**

## Correcting for the Number of Variants Raises the Detectable Sharpe Faster Than Any Available History Can Follow

Sections 2 and 3 measured what a correction costs in the abstract currency of detection probability. A strategy family lets it be priced in the only currency a desk has, which is the true edge required before the evidence could ever be conclusive. [Statistical Power](../part-12-hypothesis-testing/05-statistical-power.md) inverts the power function into a span of calendar time at a single fixed level; the multiplicity question adds one axis, holding the span at the twenty-four years the course actually has and moving the level instead:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(15025)
T, k, power, alpha = 24.0, 252, 0.80, 0.05
zb = stats.norm.ppf(power)


def min_sharpe(a):
    """Smallest true annualized Sharpe detectable in T years at level a with 80% power."""
    c = stats.norm.isf(a) + zb
    return c / np.sqrt(T - c * c / (2 * k))


def power_at(sr, a):
    """Power against a true annualized Sharpe sr at one-sided level a over T years."""
    return stats.norm.sf(stats.norm.isf(a) - sr * np.sqrt(T / (1 + sr * sr / (2 * k))))


print(f"  T = {T:.0f} years of daily data, one-sided Bonferroni level 0.05/m,"
      f" {power:.0%} power target")
print("         m   per-test level   z crit   min detectable SR   power at SR 0.30"
      "   power at SR 0.50")
for m in (1, 10, 50, 200, 1000, 10_000):
    a = alpha / m
    print(f"    {m:6d}   {a:14.7f}   {stats.norm.isf(a):6.3f}   {min_sharpe(a):17.4f}"
          f"   {power_at(0.30, a):16.4f}   {power_at(0.50, a):16.4f}")

n, m = int(round(T * k)), 50
sr = min_sharpe(alpha / m)
r = rng.standard_normal((20_000, n)) + sr / np.sqrt(k)
sh = np.sqrt(k) * r.mean(axis=1) / r.std(axis=1, ddof=1)
hit = (sh / np.sqrt((1 + sr * sr / (2 * k)) / T) > stats.norm.isf(alpha / m)).mean()
print(f"  simulated check at m = {m}: a true Sharpe of {sr:.4f} over {n:,} days"
      f" clears 0.05/{m} {hit:.4f} of the time")
# =>   T = 24 years of daily data, one-sided Bonferroni level 0.05/m, 80% power target
#             m   per-test level   z crit   min detectable SR   power at SR 0.30   power at SR 0.50
#             1        0.0500000    1.645              0.5077             0.4304             0.7893
#            10        0.0050000    2.576              0.6979             0.1343             0.4495
#            50        0.0010000    3.090              0.8031             0.0525             0.2606
#           200        0.0002500    3.481              0.8830             0.0222             0.1511
#          1000        0.0000500    3.891              0.9669             0.0077             0.0747
#         10000        0.0000050    4.417              1.0747             0.0016             0.0245
#      simulated check at m = 50: a true Sharpe of 0.8031 over 6,048 days clears 0.05/50 0.8005 of the time
```

The $m=1$ row is the anchor and it agrees with the page that owns it: power against a true Sharpe of $0.30$ over twenty-four years reads $0.4304$, which is [Statistical Power](../part-12-hypothesis-testing/05-statistical-power.md)'s figure for the same configuration, reached here from the other direction. Uncorrected, a genuine $0.30$ edge is detected on the longest history most desks will see about two times in five.

The multiplicity column is what a fifty-variant grid does to that. The critical $z$ climbs $1.645$, $2.576$, $3.090$, $3.481$, $3.891$, $4.417$ — the well-known slow growth, since the threshold moves like $\sqrt{2\log m}$ and a hundred-fold increase in the search costs barely a doubling. But power is a tail probability and does not care that the growth is slow: against a true Sharpe of $0.30$ it collapses $0.4304$, $0.1343$, $0.0525$, $0.0222$, $0.0077$, $0.0016$. At fifty variants the test detects a real edge $5.25\%$ of the time, which is to say barely more often than a test with no edge to find would fire at the uncorrected level.

The `min detectable SR` column is the same statement as a hurdle. To have an even-money-plus chance of demonstrating itself over twenty-four years, a strategy must have a true annualized Sharpe of $0.5077$ if it is the only thing tested, $0.8031$ if it is one of fifty, and $1.0747$ if it is one of ten thousand. The simulated check confirms the arithmetic at $0.8005$ against the $0.80$ target. Set those against the range a liquid market actually leaves, which the course puts at $0.2$ to $0.5$, and the conclusion is arithmetic rather than pessimism: **a correctly applied family-wise correction on a realistic strategy grid does not set a high bar for a real edge, it sets a bar that no real edge in a liquid market can clear on any history that exists**, and the strategies that *can* clear it are the ones whose Sharpes are large enough to be suspicious on their face.

Two escapes are visible in the table and neither is available here. The bar falls if $m$ falls, which is the argument for testing a small number of well-motivated hypotheses rather than a grid — a research-design decision made before any data is touched. And it falls if the family-wise rate is not the target, which is [False Discovery Rate](03-false-discovery-rate.md).

## The Correction Is Exact Arithmetic on a Number the Researcher Supplies

Everything above conditions on $m$, and $m$ is an input rather than a measurement. That makes Bonferroni exactly as trustworthy as the count it is handed, and the count is the one quantity in this part with no audit trail.

The failure is not that researchers lie about $m$. It is that $m$ is genuinely ambiguous, and every honest resolution of the ambiguity gives a different answer. A grid of fifty lookbacks tested and reported is $m=50$. The same grid, plus a signal construction abandoned in week one because its equity curve looked wrong, is a larger family with an unrecorded size. A pairs screen over seventy-eight combinations that was itself chosen after two other universes were considered is larger again. And a strategy that survived because the analyst's priors had already been shaped by a literature of published anomalies inherits that literature's accumulated $m$, which is on the order of hundreds and belongs to nobody. Applying $\alpha/50$ to the first of these while the actual search was the fourth produces a number that is precise, defensible, correctly derived and wrong, with no diagnostic anywhere in the output to say so.

This is why the union bound's assumption-freedom is easy to over-value. Bonferroni makes no assumption about the *dependence* between tests, which is genuinely remarkable and is section 1's whole content. It makes a total assumption about the *number* of them, and that assumption is neither checkable nor conservative in any direction one can reason about. A procedure that is exactly right given its input and cannot validate its input has moved the problem rather than solved it, which is [Data Snooping Bias](04-data-snooping-bias.md).

!!! note "Bonferroni, Šidák, Holm, Hochberg and Hommel are five procedures controlling one error rate on one family, and they differ in the direction they step and in what they assume about dependence"
    **Bonferroni** compares every p-value to $\alpha/m$ in a single pass, assumes nothing whatever about the joint distribution, and is the only one of the five that can be applied without sorting. **Šidák** replaces that threshold with $1-(1-\alpha)^{1/m}$, which is exact under independence and, by Šidák's lemma, still conservative for two-sided normal tests under arbitrary correlation; section 2 measures the difference at $0.0021$ of power. **Holm** steps *down* from the smallest p-value against $\alpha/(m-i+1)$ and stops at the first failure, matching Bonferroni's assumptions exactly while rejecting a superset of its hypotheses on every draw. **Hochberg** uses Holm's thresholds and steps *up* from the largest, rejecting everything below the last success, which is more powerful but requires positive regression dependence and can fail without it. **Hommel** is uniformly at least as powerful as Hochberg under the same condition and is defined through closed testing over all $2^{m}-1$ intersection hypotheses, which is why it is rarely implemented. The distinction that matters operationally is that the first three are valid for any dependence and the last two are not, and section 3 measures the entire gap between the most and least powerful of them at half a percentage point.

!!! warning "A correction reports the level it was asked for and never the level it delivered, and on a correlated family those differ by an order of magnitude"
    Section 2 measured Bonferroni delivering a realized family-wise error rate of $0.0044$ against a nominal $0.05$ on fifty tests correlated at $0.95$, while a calibrated threshold on the same family held at $0.0491$ and detected real effects $0.9431$ of the time against Bonferroni's $0.7593$. Nothing in a research output exposes this. The procedure prints a threshold, the threshold is applied correctly, the guarantee is honoured, and the fact that the guarantee was honoured eleven times over — at a cost of eighteen percentage points of power — appears nowhere, because computing it requires simulating the family under the null and no correction formula asks you to. The direction is the uncomfortable part: correlated families are the normal case in strategy research, since variants of one rule share a signal and members of a screen share a market, so the regime where Bonferroni wastes the most is the regime where it is most often used. **The free diagnostic is to stop choosing between Bonferroni and Šidák and start choosing between Bonferroni and Holm: it is four lines, it needs the identical assumption set, it rejects a superset of Bonferroni's hypotheses on every dataset that can be constructed — $0$ counterexamples in a million draws above — and there is no configuration of dependence, effect sizes or true-null count on which it costs anything.** Then, if the family is correlated and the power matters, simulate it under the null and read off the threshold that actually delivers $\alpha$, which is the same calibration [Multiple Comparisons](01-multiple-comparisons.md) prescribes and the same object [White's Reality Check](05-whites-reality-check.md) builds properly.

## One Line of Probability, an Unimprovable Guarantee, and a Bar No Real Edge Can Clear

This page established that testing at $\alpha/m$ controls the family-wise error rate under every joint distribution because subadditivity is an axiom rather than a consequence, and that Šidák's exact independent threshold exceeds Bonferroni's by a ratio tending to $1+\alpha/2$ — measured at $1.0206$, $1.0253$ and $1.0259$ as $m$ runs $5$ to $5000$, so the gap closes absolutely as the family grows; that the guarantee is kept with increasing slack as dependence rises, Bonferroni's realized rate falling $0.0439$, $0.0423$, $0.0289$, $0.0128$, $0.0044$ across $\rho=0$ to $0.95$ while a calibrated threshold held at $0.05$ throughout, and that the slack is a pure loss, since Šidák's refinement buys $0.0021$ of power while calibration buys $18.4$ percentage points at $\rho=0.95$ and Bonferroni's own detection rate sits unmoved at about $0.760$ across the entire sweep; that Holm controls the same rate under the same assumptions and rejects a superset of Bonferroni's hypotheses on every realization, with $0$ counterexamples in a million draws, for a gain of about half a percentage point, while Hochberg's additional dependence assumption yields a strictly larger set on between $0.00003$ and $0.00012$ of draws; and that correcting for $m$ over twenty-four years raises the Sharpe a strategy must truly possess from $0.5077$ at $m=1$ to $0.8031$ at $m=50$ and $1.0747$ at $m=10{,}000$, collapsing the power against a genuine $0.30$ edge from $0.4304$ to $0.0525$ to $0.0016$.

The through-line is that this page's subject divides very unevenly into the part that is discussed and the part that matters. The threshold arithmetic — Bonferroni against Šidák, step-down against step-up, the closed-testing constructions behind Hommel — is the part with a literature, and section 3 measures the whole span of it at half a percentage point of detection. The two quantities that move the answer by an order of magnitude are the family's dependence, which every procedure here refuses to look at, and the family's size, which none of them can verify. A correction is therefore an exact operation on the one input nobody checks, calibrated for a joint distribution nobody examined, and the precision of the arithmetic in between is not evidence about either.

Section 4's hurdle is the reason the next page exists rather than a counsel of despair. If controlling the probability of *any* false rejection sets a bar no real edge in a liquid market can clear, the response available to a research pipeline is not a better family-wise procedure but a different question — how many of the ideas advanced are junk, rather than whether any of them is. That is [False Discovery Rate](03-false-discovery-rate.md).

**Bonferroni is the rare procedure that is exactly as good as its reputation and wrong about which of its properties earned it: the assumption-freedom is real and nearly free, the conservatism everyone corrects for is worth two tenths of a percentage point, and the conservatism nobody measures is worth eighteen.**
