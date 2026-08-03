# Parametric Tests

The standard defence of the $t$-test is that it is robust: returns are not normal, everyone knows it, and the central limit theorem repairs the damage. The defence is correct and it is also a distraction, because normality is not the assumption these tests actually depend on. The $t$-test survives fat tails comfortably. What it does not survive is a second group with a different variance, and what the $F$-test for equal variances does not survive is any fat tail at all — it reads a fourth moment, and financial returns barely have one. The family fails in an order almost exactly opposite to the one its reputation suggests.

This page covers the shape common to every test in the family, the pooled two-sample $t$-test whose size depends on which group was labelled first, Welch's correction and what it costs, the $F$-test for equal variances as a disguised statement about kurtosis, and the way a self-standardized tail diagnostic is defeated by the observations it is hunting. It does not measure the $t$-test's size under heavy tails or at small $n$, which is [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md) and [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md); it does not measure the size distortion dependence produces, which is also [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md); it does not compute the $F$-test's power or detection thresholds under normality, which is [F Distribution](../part-05-common-distributions/17-f-distribution.md); it builds no rank statistic, which is [Nonparametric Tests](08-nonparametric-tests.md); it constructs no resampled null, which are [Permutation Tests](09-permutation-tests.md) and [Bootstrap Tests](10-bootstrap-tests.md); and it never claims a distributional assumption is harmless.

The trading stake is the first thing the course establishes about returns. [Returns and Distributions](../../part-03-statistics/02-returns-and-distributions.md) reports `skew -0.20  excess kurt 11.4` and then counts the tail directly: `|z| > 3: observed  100   normal-expected   17.306` and `|z| > 4: observed   41   normal-expected    0.406`. The lesson's summary is "Fact one, **fat tails**: excess kurtosis over 11 against the normal's 0." Sections 4 and 5 price what that single number does to two tests that are run on financial data every day and are almost never checked against it.

## Every Test in This Family Is a Statistic Divided by an Estimate of Its Own Standard Error

The parametric tests share one shape. Take a quantity of interest, subtract its null value, divide by an estimate of its standard deviation, and refer the result to a distribution derived under an assumed family. The $t$-statistic divides a mean by an estimated standard error; the $F$-statistic is a ratio of two variance estimates, which is the same construction with the subtraction replaced by a quotient; Pearson's $X^2$ sums squared standardized cell discrepancies. In every case the denominator is estimated from the same data as the numerator, and in every case the reference distribution is exact only under a specific family.

Two distinct things can therefore go wrong, and conflating them is the source of the family's reputation. The **reference distribution** can be wrong because the family is wrong — this is the failure everyone anticipates, and [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md) shows it is mild for a mean at realistic sample sizes, with the size of a nominal $5\%$ $t$-test staying inside $[0.0416,0.0514]$ across normal, $t(5)$, $t(2.65)$ and lognormal data. The **denominator** can be wrong because it estimates the wrong quantity — because the two groups do not share the variance the formula pools, or because the fourth moment the variance-of-a-variance depends on is much larger than the normal's. That second failure has no central limit theorem to rescue it, and it is where this page lives.

The distinction matters because the two are usually discussed as one. "Returns are not normal, so be careful with $t$-tests" is advice aimed at the first failure, which is the one that turns out not to matter much; the second failure is invisible in that sentence and is the one that produces sizes of $0.34$ and $0.44$ below.

## The Pooled Two-Sample t-Test Assumes a Common Variance, and Its Size Depends on Which Group Was Labelled First

The pooled two-sample test estimates one variance from both groups, weighting by degrees of freedom, and uses it for both. When the variances genuinely differ, that single number is too large for one group and too small for the other, and which error dominates depends on how the sample sizes line up against the variances.

??? note "Proof that the pooled two-sample statistic has a $t$ law only under equal variances, and that otherwise its null variance is wrong by a factor with no bound"

    Let group one have $n_1$ observations with variance $\sigma_1^{2}$ and group two $n_2$ with $\sigma_2^{2}$, and consider the difference in means $D=\bar X_1-\bar X_2$, whose true variance is
    $$\mathrm{var}(D)=\frac{\sigma_1^{2}}{n_1}+\frac{\sigma_2^{2}}{n_2}.$$
    The pooled test instead estimates a single $\sigma^{2}$ by $s_p^{2}=\big[(n_1-1)s_1^{2}+(n_2-1)s_2^{2}\big]/(n_1+n_2-2)$, whose expectation is the degrees-of-freedom weighted average $\bar\sigma^{2}=\big[(n_1-1)\sigma_1^{2}+(n_2-1)\sigma_2^{2}\big]/(n_1+n_2-2)$, and uses $s_p^{2}(1/n_1+1/n_2)$ as the variance of $D$. The ratio of what it assumes to what is true is
    $$R=\frac{\bar\sigma^{2}\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}{\frac{\sigma_1^{2}}{n_1}+\frac{\sigma_2^{2}}{n_2}} .$$
    If $\sigma_1=\sigma_2$ then $R=1$ identically and the statistic is exactly $t_{n_1+n_2-2}$. Otherwise $R\neq1$, and it is unbounded in both directions: fix $n_1\ll n_2$ and let $\sigma_1/\sigma_2\to\infty$, and the numerator's weighted average is dominated by $\sigma_2^{2}$ while the true variance is dominated by $\sigma_1^{2}/n_1$, so $R\to0$ and the statistic is inflated without limit. Swapping the labels sends $R\to\infty$ and deflates it without limit.

    The asymmetry is the point and it is purely combinatorial: the pooled estimate weights by *degrees of freedom*, so the large group dominates the variance estimate, while the true variance of $D$ weights by $1/n$, so the small group dominates the truth. When the small group is the volatile one the test believes the difference is more precise than it is; when the small group is the calm one it believes the opposite.

    The load-bearing assumption is that one number can serve as both groups' variance, and it is stated nowhere in the output — the software returns a $t$ and a p-value whether or not it holds. **The pooled test does not fail gracefully as the variances diverge; it fails in whichever direction the labelling happens to point.**

The size can be read straight off a simulation in which the two means are equal by construction, so every rejection is false:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12071)
reps, n1, n2 = 100_000, 50, 250
zc = stats.t.isf(0.025, n1 + n2 - 2)

def sizes(s1, s2):
    a = rng.normal(0, s1, (reps, n1))
    b = rng.normal(0, s2, (reps, n2))
    v1, v2 = a.var(1, ddof=1), b.var(1, ddof=1)
    d = a.mean(1) - b.mean(1)
    sp = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    t_pool = d / np.sqrt(sp * (1 / n1 + 1 / n2))
    se_w = np.sqrt(v1 / n1 + v2 / n2)
    df_w = se_w**4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t_welch = d / se_w
    return ((np.abs(t_pool) > zc).mean(),
            (np.abs(t_welch) > stats.t.isf(0.025, df_w)).mean(),
            np.median(df_w))

print("  two-sample test of EQUAL means that are equal by construction, n1=50, n2=250")
print("  so every rejection below is a false one; nominal level 0.05")
print("    sigma1   sigma2   pooled-t size   Welch size   median Welch df")
for s1, s2 in ((1.0, 1.0), (3.0, 1.0), (1.0, 3.0), (5.0, 1.0), (1.0, 5.0)):
    p, w, df = sizes(s1, s2)
    print(f"    {s1:6.1f}   {s2:6.1f}   {p:13.4f}   {w:10.4f}   {df:15.1f}")
# =>   two-sample test of EQUAL means that are equal by construction, n1=50, n2=250
#      so every rejection below is a false one; nominal level 0.05
#        sigma1   sigma2   pooled-t size   Welch size   median Welch df
#           1.0      1.0          0.0506       0.0505              70.2
#           3.0      1.0          0.2819       0.0506              51.2
#           1.0      3.0          0.0006       0.0490             235.8
#           5.0      1.0          0.3415       0.0493              49.8
#           1.0      5.0          0.0001       0.0505             297.1
```

The first row is the control: with equal variances the pooled test is exact, at $0.0506$, and Welch agrees at $0.0505$. Every other row has equal means and unequal variances, so the correct answer is still $0.05$ and every departure is a defect.

The second and third rows are the same numbers with the labels swapped, and they are the finding. With the small group three times as volatile the pooled test's size is $0.2819$ — it rejects a true null more than five times as often as advertised. Reverse the assignment, so the small group is the calm one, and the size collapses to $0.0006$: the test has effectively stopped rejecting, and would fail to detect anything. The same data-generating process, the same code, the same nominal level, and a factor of nearly five hundred between the two false-positive rates, determined entirely by which sample got which variance. At a ratio of five the two extremes are $0.3415$ and $0.0001$, a factor of over three thousand.

**This is not a test that degrades under a violated assumption; it is two different tests wearing one name, and the output cannot tell you which one you ran.**

## Welch's Correction Costs Degrees of Freedom and Buys Back the Size the Pooled Test Loses

Welch's repair is to stop pooling. Estimate each group's variance separately, add $s_1^{2}/n_1+s_2^{2}/n_2$ to get the standard error the difference actually has, and — because that sum of two scaled chi-squares is not itself a scaled chi-square — approximate the resulting reference distribution by a $t$ whose degrees of freedom are estimated from the data by the Satterthwaite formula. The Welch column above holds between $0.0490$ and $0.0506$ across every row, including the two the pooled test gets catastrophically wrong.

Where the design permits it, there is a second and better structural fix. If the two groups are naturally matched — the same strategy before and after a cost change, the same universe under two execution schedules, the same days under two risk models — then differencing within each pair before testing removes the between-unit variance entirely, and what remains is a one-sample test on the differences. That is the only manoeuvre in this family that increases power without weakening an assumption, and it works precisely because it eliminates the nuisance quantity the pooled test was trying to estimate rather than estimating it better. It is unavailable when the groups are genuinely independent, which is why Welch rather than pairing is the general answer.

The final column is the price of Welch, and it is smaller than the reputation suggests. The Welch degrees of freedom are $70.2$ in the equal-variance case against the pooled test's $298$, and they move sharply with the configuration: $51.2$ when the small group is volatile, $235.8$ when it is calm, $49.8$ and $297.1$ at the more extreme ratio. Degrees of freedom that low do widen the critical value, which costs a little power when the variances really are equal — and that is the entire cost. The usual advice to test for equal variances first and then choose between the two procedures is worse than simply always using Welch, because the preliminary test is itself unreliable in exactly the conditions that make the choice matter, which is the subject of the next section.

## The F-Test for Equal Variances Reads a Fourth Moment, and Financial Returns Do Not Have One

The natural preliminary test — compare $s_1^{2}$ to $s_2^{2}$ and refer the ratio to an $F$ distribution — is the least robust procedure in common use. Its reference distribution is derived by assuming each sample variance is a scaled $\chi^{2}$, which is a statement about the fourth moment of the data, and the fourth moment is exactly what fat tails inflate.

??? note "Proof that the null variance of a log variance ratio is $(\kappa+2)/n$ rather than $4/n$, so the $F$-test's critical value is wrong by a factor of $\sqrt{(\kappa+2)/4}$ under excess kurtosis $\kappa$"

    For an independent sample of size $n$ from a law with variance $\sigma^{2}$ and excess kurtosis $\kappa$, the sample variance has asymptotic variance
    $$\mathrm{var}(s^{2})\approx\frac{\sigma^{4}(\kappa+2)}{n},$$
    which follows from the delta method applied to the second moment, as [The Delta Method](../part-07-asymptotic-theory/04-delta-method.md) develops. Under normality $\kappa=0$ and this is the familiar $2\sigma^{4}/n$. Applying the delta method once more to $\log s^{2}$, whose derivative at $\sigma^{2}$ is $1/\sigma^{2}$,
    $$\mathrm{var}(\log s^{2})\approx\frac{\kappa+2}{n},$$
    and for two independent samples the log ratio $\log(s_1^{2}/s_2^{2})$ has variance $(\kappa+2)(1/n_1+1/n_2)$.

    The $F$ test's critical values are those of the normal case, $\kappa=0$, so they are built for a statistic whose standard deviation is $\sqrt{2(1/n_1+1/n_2)}$ when the truth is $\sqrt{(\kappa+2)(1/n_1+1/n_2)}$. The statistic is therefore too dispersed by the factor
    $$\sqrt{\frac{\kappa+2}{2}},$$
    and the size is the probability that a normal of that inflated width exceeds a fixed cutoff. At the course's measured $\kappa=11.4$ the factor is $\sqrt{13.4/2}=2.59$, so a nominal $1.96$-sigma cutoff is really a $0.76$-sigma cutoff, which a two-sided test crosses about $45\%$ of the time.

    The load-bearing quantity is $\kappa$, and the reason this failure has no asymptotic rescue is that $\kappa$ does not shrink with $n$: it is a property of the law, so the distortion is the same at $250$ observations and at $250{,}000$. Contrast the mean, whose sampling distribution is repaired by the central limit theorem precisely because the offending moments are divided away. **A test about a variance depends on the fourth moment the way a test about a mean depends on the second, and financial data is fat-tailed in exactly the moment that matters.**

The prediction is $45\%$, and it can be measured directly against the two standard robust alternatives:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12073)
reps, n = 40_000, 252

def draw(law, k):
    if law == "normal":
        return rng.standard_normal((k, n))
    if law.startswith("t("):
        nu = int(law[2:-1])
        return rng.standard_t(nu, (k, n)) / np.sqrt(nu / (nu - 2))
    m = rng.standard_normal((k, n))                # 5% contamination at 4.15 sigma
    hit = rng.random((k, n)) < 0.05
    return np.where(hit, m * 4.15, m)

def three_tests(a, b):
    f = a.var(1, ddof=1) / b.var(1, ddof=1)
    lo, hi = stats.f.ppf(0.025, n - 1, n - 1), stats.f.isf(0.025, n - 1, n - 1)
    ftest = (f < lo) | (f > hi)
    out = []
    for centre in (np.mean, np.median):            # Levene, then Brown-Forsythe
        za = np.abs(a - centre(a, axis=1, keepdims=True))
        zb = np.abs(b - centre(b, axis=1, keepdims=True))
        d = za.mean(1) - zb.mean(1)
        sp = ((n - 1) * za.var(1, ddof=1) + (n - 1) * zb.var(1, ddof=1)) / (2 * n - 2)
        out.append((np.abs(d / np.sqrt(sp * 2 / n)) > stats.t.isf(0.025, 2 * n - 2)).mean())
    return ftest.mean(), out[0], out[1]

print("  EQUAL variances by construction: every rejection is a false one, nominal 0.05")
print("    law          excess kurt   F-test size   Levene   Brown-Forsythe")
for law in ("normal", "t(8)", "t(6)", "t(5)", "contaminated"):
    a, b = draw(law, reps), draw(law, reps)
    k = stats.kurtosis(a.ravel()[:200_000])
    f, lev, bf = three_tests(a, b)
    print(f"    {law:12s} {k:11.2f}   {f:11.4f}   {lev:6.4f}   {bf:14.4f}")
# =>   EQUAL variances by construction: every rejection is a false one, nominal 0.05
#        law          excess kurt   F-test size   Levene   Brown-Forsythe
#        normal             -0.00        0.0509   0.0509           0.0500
#        t(8)                1.48        0.1354   0.0519           0.0504
#        t(6)                2.68        0.1866   0.0524           0.0508
#        t(5)                7.67        0.2405   0.0513           0.0500
#        contaminated       11.61        0.4406   0.0518           0.0497
```

The normal row confirms all three tests are exact where they should be, at $0.0509$, $0.0509$ and $0.0500$. From there the $F$-test degrades monotonically in the kurtosis and nothing else: $0.1354$ at $t(8)$, $0.1866$ at $t(6)$, $0.2405$ at $t(5)$, and $0.4406$ at a contamination calibrated to excess kurtosis $11.61$ — essentially the $11.4$ the course measures on SPY. At that level a nominal $5\%$ test of whether two volatilities differ fires on $44\%$ of samples whose volatilities are *identical by construction*, which matches the proof's prediction of about $45\%$ closely enough to confirm the mechanism.

Levene's test and Brown–Forsythe are untouched, holding between $0.0497$ and $0.0524$ across every row. Both replace the squared deviations with absolute ones — Levene about the mean, Brown–Forsythe about the median — which turns a question about fourth moments into a question about second moments of a transformed variable, and second moments are what the central limit theorem repairs. The fix is one line of code and it is not the default anywhere.

The practical consequence closes the previous section. Testing for equal variances before choosing between pooled and Welch means running, on financial data, a preliminary test whose false-positive rate is $44\%$ — so the branch is taken essentially at random, and the pipeline inherits both tests' defects. Using Welch unconditionally costs a little power under equality and removes the branch entirely.

**Fat tails are nearly harmless for a test about a mean and nearly fatal for a test about a variance, and the difference is which moment sits in the denominator.**

!!! note "A t-test names four different procedures that share a symbol and disagree about which variance is being estimated"
    The **one-sample** test divides by $s/\sqrt n$; the **paired** test is the one-sample test applied to differences, which removes the between-unit variance entirely and is the only free increase in power in this family; the **pooled two-sample** test estimates one variance from both groups; and **Welch's** test estimates two and approximates the reference distribution. They answer different questions and rest on different assumptions, and software exposes all four behind names differing by a keyword argument. Two further collisions are worth flagging: the $t$ *distribution* of [Student's t Distribution](../part-05-common-distributions/16-students-t-distribution.md) is the reference law here and also, separately, a popular *model* for returns — the same symbol on both sides of the inference — and the $F$ *test* above is not the $F$ test of a regression's joint significance, which is [Part XIII](../part-13-regression/index.md).

## A Tail Diagnostic That Standardizes by the Sample Deviation Is Defeated by the Observations It Is Hunting

The course's tail count — `|z| > 3: observed 100 normal-expected 17.306` — is the natural direct diagnostic, and on SPY it works, because the excess is nearly sixfold. Its sensitivity limit is worth knowing, because the quantity it standardizes by is computed from the same data that contains the outliers.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12077)
reps, n = 2_000, 6_410
edges = stats.norm.ppf(np.linspace(0, 1, 11))      # ten equiprobable bins
edges5 = np.array([-np.inf, -3.0, -1.0, 1.0, 3.0, np.inf])
exp10, exp5 = n * np.diff(stats.norm.cdf(edges)), n * np.diff(stats.norm.cdf(edges5))

def measure(x):
    sd = x.std(1, ddof=1, keepdims=True)
    z = (x - x.mean(1, keepdims=True)) / sd
    obs = np.stack([((z > edges[i]) & (z <= edges[i + 1])).sum(1) for i in range(10)], 1)
    cell = (obs - exp10) ** 2 / exp10              # each bin's contribution to X2
    o5 = np.stack([((z > edges5[i]) & (z <= edges5[i + 1])).sum(1) for i in range(5)], 1)
    return {"X2 (10 equiprobable bins)": cell.sum(1),
            "X2 (5 bins, cut at |z|=3)": ((o5 - exp5) ** 2 / exp5).sum(1),
            "count of |z| > 3": (np.abs(z) > 3).sum(1).astype(float)}, sd[:, 0], cell

null, sd0, cell0 = measure(rng.standard_normal((reps, n)))
m = rng.standard_normal((reps, n))
alt, sd1, cell1 = measure(np.where(rng.random((reps, n)) < 0.002, m * 8.0, m))

print("  the alternative puts 0.2% of days at 8 sigma -- about 13 days in 6,410")
print(f"  mean |z|>3 count: null {null['count of |z| > 3'].mean():.1f}, "
      f"contaminated {alt['count of |z| > 3'].mean():.1f}")
print(f"  mean sample sd:   null {sd0.mean():.4f}, contaminated {sd1.mean():.4f}")
print("  every test calibrated by simulation to an exact 5% size")
print("    test                        critical value   power")
for k in null:
    c = np.quantile(null[k], 0.95)
    print(f"    {k:26s}   {c:14.2f}   {(alt[k] > c).mean():5.4f}")

outer = [0, 9]
a0, a1 = cell0[:, outer].sum() / cell0.sum(), cell1[:, outer].sum() / cell1.sum()
print(f"  share of the X2 statistic coming from the two outermost bins:")
print(f"    null {a0:.1%}, contaminated {a1:.1%} -- the rest is the body being reshaped")
# =>   the alternative puts 0.2% of days at 8 sigma -- about 13 days in 6,410
#      mean |z|>3 count: null 17.2, contaminated 18.6
#      mean sample sd:   null 0.9996, contaminated 1.0602
#      every test calibrated by simulation to an exact 5% size
#        test                        critical value   power
#        X2 (10 equiprobable bins)             14.91   0.8230
#        X2 (5 bins, cut at |z|=3)              6.77   0.8730
#        count of |z| > 3                      23.05   0.1005
#      share of the X2 statistic coming from the two outermost bins:
#        null 9.4%, contaminated 57.6% -- the rest is the body being reshaped
```

The contamination is thirteen days out of $6{,}410$ drawn at eight standard deviations — a severe and unambiguous violation of normality, of exactly the kind a risk system exists to notice. The direct tail count barely moves: a mean of $18.6$ exceedances against the null's $17.2$, and a detection rate of $0.1005$ at a correctly calibrated $5\%$ level. Nine times in ten, the diagnostic the course uses would report nothing.

The middle line explains it. Those thirteen enormous observations raise the mean sample standard deviation from $0.9996$ to $1.0602$, and the standardization divides by that inflated number, so the threshold $|z|>3$ moves outward by six percent for every observation in the sample. The outliers push the fence back far enough to stand behind it. The diagnostic is self-defeating in a precise sense: the statistic it uses to define "extreme" is contaminated by the extremes it is looking for, and the contamination moves in the direction that hides them.

The two binned tests are not fooled, at $0.8230$ and $0.8730$. The last line shows why: under the null the two outermost bins contribute $9.4\%$ of the $X^{2}$ statistic, and under contamination they contribute $57.6\%$. The binned tests suffer the identical scale inflation, but they compare *counts against expected counts in every bin at once*, so the same six-percent shift that neutralizes a single threshold shows up as a consistent deficit across the interior bins and a surplus in the outer ones. Aggregating many small distortions recovers what one threshold loses.

**A threshold measured in units estimated from the same sample is not a fixed threshold, and the observations that would trip it are the ones that move it.**

!!! warning "The assumption that broke a parametric test is almost never normality, and the substitute null is free to generate"
    Everything on this page failed for a reason not named in the usual warning. The pooled $t$-test failed on unequal variances while remaining exactly normal; the $F$-test failed on a fourth moment while every marginal stayed symmetric and mean-zero; the tail count failed on its own denominator. In each case a reader checking "are the data normal?" would have been looking in the wrong place, and in two of the three the data genuinely were normal. Meanwhile the failure everyone does anticipate — non-normality of a mean — is the one [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md) shows is nearly harmless at these sample sizes, and the largest distortion in this part comes from dependence, measured in [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md). **The free diagnostic is to stop reasoning about which assumption matters and generate the null you actually face: resample your own two groups *within* group, which destroys any difference in means while preserving each group's own shape, dependence and sample size, then run your exact testing code five thousand times and read the rejection rate — if it is not near $\alpha$ the test is broken on your data, and the printout will not tell you which assumption did it, which is why the check is worth more than the reasoning.**

## The Assumption That Breaks a Parametric Test Is Almost Never the One Named in the Textbook

This page established that every test in the family divides a quantity by an estimate of its own standard error and refers the result to a family-specific law, so two independent failures are possible and only one of them has a limit theorem to repair it; that the pooled two-sample $t$-test has size $0.2819$ when the small group is three times as volatile and $0.0006$ when the labels are swapped, a factor of nearly five hundred at unchanged nominal level, widening to $0.3415$ against $0.0001$ at a ratio of five; that Welch's correction holds between $0.0490$ and $0.0506$ throughout, at a cost of degrees of freedom falling from $298$ to as low as $49.8$; that the $F$-test's size climbs monotonically in kurtosis to $0.4406$ at the course's measured $11.4$, matching a delta-method prediction of $45\%$, while Levene and Brown–Forsythe hold near $0.05$; and that a self-standardized count of $|z|>3$ detects thirteen eight-sigma days only $10.05\%$ of the time, because those days lift the sample standard deviation from $0.9996$ to $1.0602$ and push the threshold out past themselves.

The unifying observation is that these tests are precise about the wrong thing. Each is exact under its family, and the exactness attaches to the reference distribution rather than to the quantity a user cares about. Where the family is wrong in the *first* moment structure, the central limit theorem repairs it and the reputation for robustness is earned. Where it is wrong in the second — an unequal variance, an inflated fourth moment, a denominator estimated from contaminated data — nothing repairs it, the distortion does not shrink with $n$, and the reported p-value is as clean as ever.

Every repair on this page was a substitution: pool nothing and estimate two variances, take absolute deviations instead of squares, count against expected counts in many bins rather than past one threshold. Each replaced an assumption about the family with a weaker one. The natural next question is how far that substitution can be pushed — whether the distributional assumption can be dropped altogether rather than weakened, and what the drop costs. That is [Nonparametric Tests](08-nonparametric-tests.md).

**A parametric test is exact about a world specified in advance, and it is most confident precisely where that specification and this world differ in a moment nobody printed.**
