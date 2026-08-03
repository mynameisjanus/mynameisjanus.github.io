# Nonparametric Tests

Replacing observations by their ranks removes the distributional assumption from a test, and the removal is genuine: the null distribution of a rank statistic can be written down exactly, with no family assumed and no limit theorem invoked. What the transform does not do is remove the *other* assumptions, and it quietly changes the question. A rank test of "is there an edge" is a test about a median, and the median of a trading P&L is not the quantity anyone is paid on. Two substitutions happen at once — the family is dropped and the functional is changed — and only the first is advertised.

This page covers the exact null distribution ranks buy, the sign and signed-rank tests as statements about a median rather than a mean, the efficiency the rank transform costs under normality and repays under heavier tails, rank correlation as the robust alternative to Pearson's, and the assumption that "distribution-free" leaves untouched. It does not define the level, the size or the p-value, which are [The Hypothesis Testing Framework](01-hypothesis-testing-framework.md) and [p-values](03-p-values.md); it does not develop the parametric family or its failures, which is [Parametric Tests](07-parametric-tests.md); it does not derive Spearman's coefficient as a functional of a copula, which is [Correlation](../part-04-expectation-and-moments/05-correlation.md); it builds no permutation null and proves no exactness by group invariance, which is [Permutation Tests](09-permutation-tests.md); it resamples nothing, which is [Bootstrap Tests](10-bootstrap-tests.md); and it never claims a rank test is assumption-free.

The trading stake is the shape of the course's own trade log. [Performance Metrics and Reporting](../../part-05-backtesting-engine/04-performance-metrics-and-reporting.md) pins `trips 329, median hold 3 days` and `hit rate 34%, avg win +5.7%, avg loss -2.0%, payoff 2.9`, with `expectancy +66 bp per trip (= 0.34 x +569 + 0.66 x -196)`, and reads it: "half of all 329 trips are whipsaws that die within days, while the money is made by the tail of the distribution — multi-month rides the median never sees." Section 2 takes that sentence literally and asks what happens when a test looks at the median instead.

## Replacing Observations by Ranks Fixes the Null Distribution Without Fixing Anything Else

A **rank test** replaces each observation by its position in the sorted sample and computes a statistic from the positions alone. The payoff is that under a suitable null the joint law of the ranks is completely known — it does not depend on the underlying distribution at all — so critical values can be tabulated once, exactly, for each sample size.

??? note "Proof that the null distribution of any rank statistic depends on the sample size alone, provided the observations are exchangeable"

    Let $X_1,\dots,X_N$ be **exchangeable**: for every permutation $\pi$ of $\{1,\dots,N\}$, the vector $(X_{\pi(1)},\dots,X_{\pi(N)})$ has the same joint law as $(X_1,\dots,X_N)$. Assume the common law is continuous, so ties have probability zero, and let $R=(R_1,\dots,R_N)$ be the vector of ranks.

    Fix any permutation $\rho$. The event $\{R=\rho\}$ is the event that the observations fall in one specific order, and applying $\rho^{-1}$ to the indices maps it bijectively onto $\{R=\mathrm{id}\}$. Exchangeability says that relabelling does not change probabilities, so $\mathbf{P}(R=\rho)=\mathbf{P}(R=\mathrm{id})$ for every $\rho$. There are $N!$ permutations and they partition the sample space up to a null set, so each has probability $1/N!$: the rank vector is **uniform on the symmetric group**, whatever the underlying distribution was.

    Any statistic $T=g(R)$ is a function of that uniform vector, so its null law is determined by $g$ and $N$ alone. This is why a Wilcoxon table has rows indexed by sample size and no column for the distribution, and why the resulting p-value is exact in finite samples rather than asymptotic.

    The load-bearing hypothesis is exchangeability, and it is doing far more work than the phrase "distribution-free" suggests. It is implied by the observations being independent and identically distributed, but it is not implied by identical distribution alone: a stationary time series has identical marginals at every date and is emphatically not exchangeable, because reordering it destroys the autocovariance. Section 5 measures what that costs. **Ranks buy exactness against the distribution and pay for it with a requirement about the dependence, which is the assumption that fails in every series this course studies.**

The second thing ranks do is less often noticed. Discarding magnitudes changes what is being tested. The one-sample $t$-test asks about $\mathbb{E}[X]$; the sign test asks whether $\mathbf{P}(X>0)=\tfrac12$, which is a statement about the median; the Wilcoxon signed-rank test asks about a centre of symmetry, which coincides with the median under symmetry and is a third thing otherwise. On symmetric data all three questions have the same answer and the choice looks like a matter of robustness. On asymmetric data they are different questions, and a P&L distribution is about as asymmetric as data gets.

The family divides by what the ranks are computed over, and it is worth having the map before the sections that use it. **One-sample** tests rank a single series against a hypothesized centre: the sign test counts positives, the signed-rank test weights those counts by the ranks of the magnitudes. **Two-sample** tests pool both samples and rank across the pool, so the Wilcoxon rank-sum statistic — identical to Mann–Whitney's $U$ up to an affine map — asks whether one sample's observations tend to sit higher in the pooled order. **Distributional** tests compare whole empirical distribution functions rather than a location: Kolmogorov–Smirnov takes the largest vertical gap between two of them, Anderson–Darling a weighted average that puts more emphasis on the tails. **Association** tests rank each coordinate separately and correlate the ranks, which is Spearman's coefficient, or count concordant against discordant pairs, which is Kendall's. All four groups inherit the same exactness from the same theorem, and all four inherit the same hypothesis with it.

## The Sign Test and the Signed-Rank Test Ask About a Median, and a Trading P&L Has the Wrong One

A trend-following trade log is a long string of small losses punctuated by a few large gains. Its mean is positive and its median is negative, by construction and not by accident: that *is* the strategy. Feeding it to three tests that are routinely described as interchangeable produces three different verdicts:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12081)
trips, reps = 329, 2_000
hit, win, loss = 0.34, 0.0569, 0.0196             # the course's tearsheet, in decimals
sw, sl = 0.60, 0.50                                # log-scale spread of the magnitudes

def trade_log(k):
    w = rng.lognormal(np.log(win) - sw**2 / 2, sw, (k, trips))
    l = rng.lognormal(np.log(loss) - sl**2 / 2, sl, (k, trips))
    return np.where(rng.random((k, trips)) < hit, w, -l)

x = trade_log(reps)
print(f"  {trips} round trips at a {hit:.0%} hit rate, mean win {win:+.2%}, "
      f"mean loss {-loss:+.2%}")
print(f"  simulated expectancy {1e4 * x.mean():+.0f} bp per trip "
      f"against the course's +66 bp")
print("    test                 median p   rejects at 5%   of those, says LOSES money")
t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(trips))
pos = (x > 0).sum(1)
z = (pos - trips / 2) / np.sqrt(trips / 4)
r = stats.rankdata(np.abs(x), axis=1)
wsr = (r * (x > 0)).sum(1)
zw = (wsr - trips * (trips + 1) / 4) / np.sqrt(trips * (trips + 1) * (2 * trips + 1) / 24)
for name, s in (("one-sample t", t), ("sign test", z), ("Wilcoxon signed-rank", zw)):
    p = 2 * stats.norm.sf(np.abs(s))
    rej = p < 0.05
    print(f"    {name:20s} {np.median(p):10.2e}   {rej.mean():13.4f}   "
          f"{(s[rej] < 0).mean():24.4f}")
# =>   329 round trips at a 34% hit rate, mean win +5.69%, mean loss -1.96%
#      simulated expectancy +64 bp per trip against the course's +66 bp
#        test                 median p   rejects at 5%   of those, says LOSES money
#        one-sample t           7.62e-03          0.7815                     0.0000
#        sign test              7.09e-09          1.0000                     1.0000
#        Wilcoxon signed-rank   4.02e-01          0.1120                     0.0848
```

The simulated log reproduces the tearsheet: $+64$ basis points per trip against the course's $+66$, from the same hit rate and the same average win and loss. The strategy makes money, and the question every test is being asked is whether that is real.

The $t$-test answers correctly. Its median p-value is $7.62\times10^{-3}$, it rejects in $78.15\%$ of replications, and in *none* of those rejections does it point the wrong way. The sign test also rejects — in $100\%$ of replications, at a median p-value of $7.09\times10^{-9}$, evidence a million times stronger than the $t$-test's — and in $100\%$ of those rejections it reports that the strategy **loses money**. It is not malfunctioning. It is answering its own question flawlessly: the median trip really is a loss, $66\%$ of trips really are losses, and the probability of that arising from a symmetric coin is genuinely about one in a hundred million. The sign test is right, overwhelmingly, about a quantity nobody is paid on.

The Wilcoxon signed-rank test lands between, and is arguably the worst of the three here. Its median p-value is $0.402$ and it rejects only $11.20\%$ of the time, because it is a test about a centre of symmetry and this distribution has none — the few large positive ranks and the many small negative ones roughly cancel. It gives almost no answer to a question with a clear answer, and unlike the sign test it does not even do so with a coherent alternative interpretation.

**Three tests, one trade log, and the disagreement is not about strength of evidence but about which functional is being tested — a distinction the word "nonparametric" does not mention.**

## The Rank Transform Costs About Five Percent Under the Normal and Is Repaid by Every Heavier Tail

If ranks are the right functional for the question, what do they cost in efficiency? The classical answer is the asymptotic relative efficiency, and reading it as a *sample size* rather than as a power difference is what makes it legible.

??? note "Proof that the Wilcoxon signed-rank test needs $\pi/3$ times as many observations as the $t$-test at the normal, and fewer at every heavier-tailed law"

    Asymptotic relative efficiency compares two consistent tests by the ratio of sample sizes they need for the same power against the same shrinking sequence of alternatives, and for tests of location it equals the ratio of the squares of their **efficacies**. For a statistic with asymptotic mean slope $\mu'(0)$ in the shift and null standard deviation $\sigma_T$, the efficacy is $\mu'(0)/\sigma_T$.

    For the $t$-test the efficacy is $1/\sigma$, where $\sigma$ is the data's standard deviation. For the Wilcoxon signed-rank test on a symmetric density $f$ the standard calculation gives efficacy $\sqrt{12}\int f^{2}$. Their ratio squared is
    $$\mathrm{ARE}(W,t)=12\,\sigma^{2}\left(\int f^{2}\right)^{2}.$$
    At the normal, $\int f^{2}=1/(2\sigma\sqrt\pi)$, so the bracket is $12\sigma^{2}/(4\pi\sigma^{2})=3/\pi\approx0.9549$. Since efficiency is the reciprocal of relative sample size, Wilcoxon needs $\pi/3\approx1.047$ times as many observations — a loss of under five percent.

    The same formula run the other way is the interesting half. Heavier tails inflate $\sigma^{2}$ without inflating $\int f^{2}$, because the tail contributes to the variance and almost nothing to the squared density, so the product rises above one and Wilcoxon needs *fewer* observations than the $t$-test. The bound $\mathrm{ARE}(W,t)\ge0.864$ holds over all symmetric densities, so the transform can never cost more than about $16\%$ while its gain is unbounded.

    The load-bearing quantity is $\sigma^{2}\left(\int f^{2}\right)^{2}$, which measures how much of the variance lives in the tail. **The rank transform charges a five-percent insurance premium at the normal and pays out on every distribution a market has ever produced.**

Measured directly, by finding the sample size at which each test reaches $80\%$ power against a fixed shift:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12083)
reps, grid = 4_000, (20, 30, 45, 65, 95, 140, 200, 300, 450)

def power_at(law, n, shift):
    if law == "normal":
        x = rng.standard_normal((reps, n))
    elif law == "t(5)":
        x = rng.standard_t(5, (reps, n)) / np.sqrt(5 / 3)
    else:
        x = rng.standard_t(3, (reps, n)) / np.sqrt(3.0)
    x = x + shift
    t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n))
    zs = ((x > 0).sum(1) - n / 2) / np.sqrt(n / 4)
    r = stats.rankdata(np.abs(x), axis=1)
    w = (r * (x > 0)).sum(1)
    zw = (w - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    c = stats.norm.isf(0.025)
    return [(np.abs(v) > c).mean() for v in (t, zs, zw)]

def n_for_80(law, shift, j):
    pw = [power_at(law, n, shift)[j] for n in grid]
    return float(np.interp(0.80, pw, grid))

print("  sample size each test needs for 80% power against a fixed shift of 0.30 sd")
print("    law        n for t   n for sign   n for Wilcoxon   Wilcoxon/t   sign/t")
for law in ("normal", "t(5)", "t(3)"):
    nt, ns, nw = (n_for_80(law, 0.30, j) for j in (0, 1, 2))
    print(f"    {law:8s} {nt:9.1f}   {ns:10.1f}   {nw:14.1f}   {nw / nt:10.3f}   {ns / nt:6.3f}")

n, m = 252, 3_000
u, v = rng.standard_normal((m, n)), rng.standard_normal((m, n))   # INDEPENDENT by construction
big = rng.random((m, n)) < 0.02                                   # 2% joint outliers
u, v = np.where(big, u * 10, u), np.where(big, v * 10, v)
pe = np.array([stats.pearsonr(a, b).pvalue for a, b in zip(u, v)])
sp = np.array([stats.spearmanr(a, b).pvalue for a, b in zip(u, v)])
print(f"  independent series with 2% joint outliers, n=252: size of the Pearson test "
      f"{(pe < 0.05).mean():.4f}, Spearman {(sp < 0.05).mean():.4f}")
# =>   sample size each test needs for 80% power against a fixed shift of 0.30 sd
#        law        n for t   n for sign   n for Wilcoxon   Wilcoxon/t   sign/t
#        normal        89.0        137.4             94.1        1.058    1.544
#        t(5)          88.1         99.7             74.3        0.844    1.132
#        t(3)          82.5         60.5             52.6        0.638    0.733
#      independent series with 2% joint outliers, n=252: size of the Pearson test 0.6657, Spearman 0.0533
```

The normal row confirms the theory to two decimals. Wilcoxon needs $94.1$ observations where the $t$-test needs $89.0$, a ratio of $1.058$ against the predicted $\pi/3=1.047$; the sign test needs $137.4$, a ratio of $1.544$ against its own theoretical $\pi/2=1.571$. Under normality the rank transform is a five-percent tax and the sign test a fifty-percent one.

The heavier rows reverse it. At $t(5)$ Wilcoxon needs $74.3$ observations against the $t$-test's $88.1$, a ratio of $0.844$; at $t(3)$ it needs $52.6$ against $82.5$, a ratio of $0.638$. On data with the tails a market actually produces, the "inefficient" test reaches the same power on a third fewer observations, and even the crude sign test overtakes the $t$-test at $t(3)$ with a ratio of $0.733$. The premium is repaid several times over on anything heavier than a normal, which — per [Returns and Distributions](../../part-03-statistics/02-returns-and-distributions.md)'s excess kurtosis of $11.4$ — is everything.

The final line is a different and sharper kind of robustness, measured on *independent* series so that every rejection is false. Contaminating $2\%$ of the pairs with joint outliers leaves the Pearson correlation test with a size of $0.6657$: a nominal $5\%$ test of independence rejects two-thirds of the time when the two series are independent by construction. Spearman's holds at $0.0533$.

**Ranks cost five percent where the normal holds and buy back a third of the sample where it does not, which on financial data points the classical trade-off the other way.**

## A Rank Correlation Tests Monotone Association and Survives the Outliers That Break Pearson's

The $0.6657$ deserves unpacking, because it is a failure of a different kind from anything in the efficiency table. Nothing about the *dependence* changed — the series are independent — and nothing about the marginals is asymmetric. What changed is that a handful of pairs are large in both coordinates at once, and Pearson's coefficient is a ratio of sums of products, so a single such pair moves the numerator by an amount comparable to the entire rest of the sample. The estimate is still consistent and still centred at zero; what breaks is the *reference distribution*, whose tails the standard approximation badly understates once the summands are that heavy. It is the same mechanism as the $F$-test's kurtosis failure in [Parametric Tests](07-parametric-tests.md): a test about a second-moment quantity calibrated by a fourth moment nobody checked.

Ranks are immune because the transform is bounded. The largest observation contributes rank $n$ whether it is three standard deviations out or thirty, so no single pair can move the statistic by more than $O(1/n)$ of its range, and the null distribution of section 1 holds exactly regardless. That is what [Correlation](../part-04-expectation-and-moments/05-correlation.md) means in developing Spearman's coefficient as a functional of the copula alone: the transform discards the marginals, and with them every pathology the marginals could contribute. That page derives the coefficient and explicitly defers the testing of it to here; the size figures above are what the deferral was worth.

The cost is the usual one, and it is the same cost as in section 2. Spearman's tests *monotone association*, which is not the quantity a portfolio calculation needs — a covariance matrix is built from Pearson correlations because variance is a second moment, and rank correlations do not assemble into one. So the honest use is diagnostic rather than substitutive: run both, and treat a large gap between them as evidence that a few observations are carrying the linear estimate.

!!! note "Nonparametric, distribution-free, robust and rank-based are four different claims, and only one of them is about the null distribution"
    **Distribution-free** means the null distribution of the statistic does not depend on the underlying law — the property proved in section 1, and the narrowest of the four. **Nonparametric** describes a *model* with infinitely many parameters, which is a modelling choice rather than a property of a test, and belongs with [Statistical Models](../part-10-statistics-foundations/04-statistical-models.md). **Robust** describes an estimator whose value is insensitive to contamination, measured by a breakdown point, and is [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md). **Rank-based** is a construction, not a guarantee. The four are correlated and routinely conflated: the Wilcoxon test is distribution-free and rank-based, its associated estimator is robust, and none of that makes the underlying *model* nonparametric or the test valid under dependence.

## Distribution-Free Names the Assumption That Was Dropped, and Not the One That Was Kept

The exactness proved in section 1 rests entirely on exchangeability, and a time series is not exchangeable. The natural test of whether a strategy's behaviour has changed — split the history in half and compare the halves — puts that assumption under maximum strain, because the two samples are consecutive stretches of one dependent series with, under the null, identical marginals:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12087)
reps, n = 4_000, 1_260                             # two consecutive halves of 2n observations

def halves(phi):
    e = rng.standard_normal((reps, 2 * n))
    x = np.empty((reps, 2 * n))
    x[:, 0] = e[:, 0]
    for j in range(1, 2 * n):
        x[:, j] = phi * x[:, j - 1] + np.sqrt(1 - phi**2) * e[:, j]
    return x[:, :n], x[:, n:]

def three(a, b):
    v1, v2 = a.var(1, ddof=1), b.var(1, ddof=1)
    t = (a.mean(1) - b.mean(1)) / np.sqrt(v1 / n + v2 / n)
    both = np.concatenate([a, b], axis=1)
    r = stats.rankdata(both, axis=1)
    u = r[:, :n].sum(1) - n * (n + 1) / 2
    mw = (u - n * n / 2) / np.sqrt(n * n * (2 * n + 1) / 12)
    lab = np.concatenate([np.ones(n), -np.ones(n)])
    order = np.argsort(both, axis=1)
    walk = np.take_along_axis(np.broadcast_to(lab, both.shape), order, axis=1)
    ks = np.abs(np.cumsum(walk, axis=1)).max(1) / n * np.sqrt(n / 2)
    return t, mw, ks

print("  two samples that are consecutive halves of ONE stationary series, n=1260 each")
print("  identical marginals by construction, so every rejection is a false one")
print("    phi    t-test   Mann-Whitney   KS   (calibrated MW)")
for phi in (0.00, 0.20, 0.40):
    t, mw, ks = three(*halves(phi))
    c = stats.norm.isf(0.025)
    row = [(np.abs(t) > c).mean(), (np.abs(mw) > c).mean(), (ks > 1.3581).mean()]
    _, mw2, _ = three(*halves(phi))              # a second, independent set of histories
    cmw = np.quantile(np.abs(mw), 0.95)          # judged against a null simulated at this phi
    print(f"    {phi:.2f}   {row[0]:7.4f}   {row[1]:12.4f}   {row[2]:5.4f}   "
          f"{(np.abs(mw2) > cmw).mean():14.4f}")
# =>   two samples that are consecutive halves of ONE stationary series, n=1260 each
#      identical marginals by construction, so every rejection is a false one
#        phi    t-test   Mann-Whitney   KS   (calibrated MW)
#        0.00    0.0435         0.0452   0.0403           0.0500
#        0.20    0.1148         0.1135   0.0945           0.0530
#        0.40    0.2105         0.2023   0.1635           0.0508
```

The $\phi=0$ row is the control and every test is at nominal: $0.0435$, $0.0452$ and $0.0403$. Exchangeability holds, the theory applies, and the rank tests are exact as promised.

The other two rows are the page's point, and they are almost boring to read, which is itself the finding. At $\phi=0.20$ the $t$-test's size is $0.1148$ and Mann–Whitney's is $0.1135$. At $\phi=0.40$ they are $0.2105$ and $0.2023$. The rank test tracks the parametric test to within a percentage point at every level of dependence — neither better nor meaningfully different. Kolmogorov–Smirnov is marginally less inflated at $0.1635$ but travels in the same direction and by the same order. Whatever protection the rank transform provides, none of it is protection against this.

That is not a defect in the tests; it is the proof of section 1 read correctly. Ranks are uniform on the symmetric group *under exchangeability*, and dependence violates exchangeability rather than the distributional assumption ranks were built to drop. The final column shows what does work: taking the same Mann–Whitney statistic and judging it against a null simulated with the same $\phi$ restores $0.0500$, $0.0530$ and $0.0508$. The statistic was fine throughout; only its reference distribution was wrong, and the repair is the one every page in this part has ended up recommending.

**A rank test drops the assumption that was not binding and keeps the one that was, which is why swapping a $t$-test for a Wilcoxon on a time series changes the label on the output and nothing else.**

!!! warning "A rank test on a time series has traded a normality assumption it did not need for an exchangeability assumption the series violates more severely"
    The exchange looks free because the discarded assumption is the famous one. Nothing in the output records it: the Wilcoxon p-value is computed from an exact table, the table is correct, and the number is wrong only because the table's hypothesis is false for this data. The failure is invisible in the same way the pooled $t$-test's was in [Parametric Tests](07-parametric-tests.md) — correct arithmetic, correct reference, wrong problem — and here it is more insidious, because "distribution-free" reads as a guarantee about the data rather than about the statistic. **The free diagnostic is to run your rank test twice against two different nulls: once against the tabulated one, and once against a null you generate by shuffling your own series in blocks long enough to preserve its autocorrelation — if the two p-values differ materially, exchangeability was the binding assumption, the table was never describing your data, and the block-based number is the one to report.**

## The Robustness a Rank Test Buys Is Against the Failure That Was Not Happening

This page established that the rank vector is uniform on the symmetric group under exchangeability, so a rank statistic's null law depends on $n$ alone and is exact in finite samples; that the transform changes the functional as well as the reference, so on the course's own trade log the $t$-test rejects at a median p-value of $7.62\times10^{-3}$ and always in the profitable direction while the sign test rejects at $7.09\times10^{-9}$ and *always* reports the strategy losing money, with Wilcoxon at $0.402$ answering neither question; that the transform costs a sample-size ratio of $1.058$ at the normal against a theoretical $\pi/3$, and repays it at $0.844$ under $t(5)$ and $0.638$ under $t(3)$; that Pearson's correlation test has size $0.6657$ on independent series with $2\%$ joint outliers where Spearman's holds $0.0533$; and that against dependence the rank tests are not robust at all, Mann–Whitney reaching $0.2023$ where the $t$-test reaches $0.2105$, until the reference distribution is simulated rather than looked up.

The pattern is that a rank test is precisely as robust as advertised, and the advertisement is narrower than the phrase. It is exactly valid against any distributional shape, provably, with no asymptotics — and that is the whole of it. It is not robust to dependence, because dependence violates the hypothesis the exactness rests on. It is not a substitute for a mean, because it tests a different functional. It is robust to outliers in the strong sense the Pearson comparison shows, which is real and is the one benefit that generalizes. Reaching for a nonparametric test as general-purpose insurance buys one specific policy against one specific hazard, and on financial data that hazard is rarely the one that occurs.

Both remaining pages take the repair the last section arrived at — stop looking up the null and generate it — and build it properly rather than as an improvisation. The first constructs the null from a group of transformations the analyst chooses, which recovers exactness in finite samples without assuming a distribution *or* independence, provided the group is chosen to match the structure that is actually present in the series. That is [Permutation Tests](09-permutation-tests.md), and it is the same uniformity-over-a-group argument as section 1, with the group finally chosen deliberately instead of inherited.

**Dropping the distributional assumption is a real achievement and a small one, because on a time series the distribution was never the assumption that was going to fail.**
