# The Hypothesis Testing Framework

A hypothesis test is usually introduced as a recipe — compute a statistic, look up a critical value, compare — and the recipe is the least interesting thing about it. What a test actually is, is a function from the sample space to $\{0,1\}$, chosen so that one of its two error rates is held below a number the analyst picks and the other falls wherever the data and the alternative put it. That asymmetry is forced rather than conventional, because only one of the two hypotheses is specific enough to compute a probability under, and every argument about testing that has ever been had — about $0.05$, about "accepting the null", about whether a rejection means anything — descends from that one structural fact.

This page covers the hypothesis as a subset of a model rather than a sentence, the test as a function to $\{0,1\}$ and its rejection region, the gap between the level a test promises and the size it delivers, the asymmetry that lets a test reject and never accept, the duality that makes every confidence interval an inverted test, and the Neyman–Pearson programme that fixes one error and optimizes the other. It chooses no statistic and locates no null distribution, which is [Test Statistics](02-test-statistics.md); it does not characterize the p-value, which is [p-values](03-p-values.md); it does not trade the two errors against each other, which is [Type I and Type II Errors](04-type-i-and-type-ii-errors.md); it computes no power function and inverts none for a sample size, which is [Statistical Power](05-statistical-power.md); it proves no optimality theorem and states no limiting law for a likelihood ratio, which is [Likelihood Ratio Tests](06-likelihood-ratio-tests.md); it derives no coverage, which is [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md); it corrects nothing for the number of hypotheses examined, which is [Part XV](../part-15-multiple-testing/index.md); and it never says a hypothesis is true.

The trading stake is a sentence the course puts at the top of its own testing lesson and then spends the lesson paying for. [Hypothesis Testing and Multiple Testing](../../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) opens: "Every backtest ends with the same implicit claim: *this would have made money, and not by luck*. That is a hypothesis test, whether or not anyone writes it down." The defendant is twelve-month sign momentum on SPY at `n 6158, ann ret 5.9%, ann vol 19.3%, Sharpe 0.30`, and the verdict is `t = 1.50, p = 0.135`. Sections 2 and 5 price what that test promised and what it can certify when the promise is kept.

## A Hypothesis Is a Subset of the Model and a Test Is a Function From the Data to Zero or One

Everything here happens inside a model, in the sense [Statistical Models](../part-10-statistics-foundations/04-statistical-models.md) fixed: a set $\mathcal{P}=\{P_\theta:\theta\in\Theta\}$ of candidate laws, asserted before the calculation and not itself under test. A **hypothesis** is a subset of $\Theta$; the null $\Theta_0$ and the alternative $\Theta_1$ partition it, and the entire content of "the strategy has no edge" is the claim $\theta\in\Theta_0$ for a $\Theta_0$ somebody wrote down. A hypothesis is **simple** when the subset is a single point, so the data's law is completely specified, and **composite** when it is anything larger — a distinction that sounds like bookkeeping and is the source of most of the trouble in this part.

A **test** is a function $\varphi:\mathcal{X}\to\{0,1\}$, with $\varphi(x)=1$ meaning reject, and $R=\{x:\varphi(x)=1\}$ is its **rejection region**. Nothing in the definition mentions a statistic, a critical value or a distribution: those describe rejection regions compactly, they are not what a region is. The region is the primitive, and that matters because the two error probabilities this part argues about are properties of the *set* $R$ under the two families of laws, determined before any statistic is named.

The shift from [Part XI](../part-11-parameter-estimation/index.md) is a change of question, not of machinery. There the question was how precisely $\theta$ could be pinned down and the output was a number with a wobble attached; here it is whether $\theta$ lies in a set named in advance, and the output is one bit. The same sampling distribution does the work in both, which is why section 4 finds them to be one object seen from two sides — but the commonest error in applied testing is to read the bit as though it were the interval.

## The Level Is a Promise and the Size Is What the Test Delivers, and They Differ Whenever the Null Is a Set

The **size** of a test is
$$\alpha(\varphi)=\sup_{\theta\in\Theta_0}\mathbf{P}_\theta(\varphi=1),$$
the largest false-rejection probability anywhere in the null set, and a test has **level** $\alpha$ when $\alpha(\varphi)\le\alpha$. The supremum is not a flourish: promising level $\alpha$ means asserting $\beta_\varphi(\theta)=\mathbf{P}_\theta(R)\le\alpha$ for every $\theta\in\Theta_0$ at once, and a family of inequalities indexed by a set is equivalent to one inequality on its supremum. The level is therefore a promise the analyst makes and the size a property the test has, and they coincide only when the test was calibrated at the point attaining the supremum. When $\Theta_0$ is a single point the distinction is pedantry. When $\Theta_0$ is a set — which is to say, in almost every test anyone runs — the supremum is over a region the analyst never visits, and the number reported as "the significance level" is a claim about a corner of the null nobody checked.

Nothing forces that supremum to sit at an extreme point. If $\beta_\varphi$ is monotone in $\theta$ — as for a one-sided region in a family with monotone likelihood ratio, the case [Test Statistics](02-test-statistics.md) develops — it sits at the boundary and calibrating there calibrates everywhere. But when $\Theta_0$ is indexed by a **nuisance parameter** the null leaves free, $\beta_\varphi$ has no reason to be monotone on that space, and when the statistic is lattice-valued it is not even continuous. The pooled two-sample $z$-test of $H_0\!: p_1=p_2$ — the natural reach when comparing the breach rate of two risk models — has both defects: its null set is a curve indexed by the common rate, and because both samples are lattice-valued the rejection probability can be enumerated exactly rather than simulated, and the supremum taken over that curve:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12013)
zc = stats.norm.ppf(0.975)
pgrid = np.arange(0.002, 0.501, 0.001)             # the null set, symmetric under p -> 1-p

def size_surface(m1, m2):
    k1, k2 = np.arange(m1 + 1), np.arange(m2 + 1)
    pool = (k1[:, None] + k2[None, :]) / (m1 + m2)
    var = pool * (1 - pool) * (1 / m1 + 1 / m2)
    z = (k1[:, None] / m1 - k2[None, :] / m2) / np.sqrt(np.where(var > 0, var, np.inf))
    rej = np.abs(z) > zc
    return np.array([stats.binom.pmf(k1, m1, p) @ rej @ stats.binom.pmf(k2, m2, p)
                     for p in pgrid])

print("  pooled two-sample z-test of H0: p1 = p2, nominal level 0.05")
print("  exact size by enumeration, maximised over the nuisance the null leaves free")
print("     n1    n2   size at p=0.50   size over the null set   attained at p")
worst = None
for m1, m2 in [(25, 25), (50, 50), (250, 250), (20, 200), (30, 300)]:
    s = size_surface(m1, m2)
    i = int(s.argmax())
    print(f"    {m1:3d}   {m2:3d}          {s[np.argmin(abs(pgrid - 0.5))]:.4f}   "
          f"{s[i]:22.4f}   {pgrid[i]:13.3f}")
    if worst is None or s[i] > worst[2]:
        worst = (m1, m2, s[i], pgrid[i])

m1, m2, exact, p = worst
a = rng.binomial(m1, p, 400_000) / m1
b = rng.binomial(m2, p, 400_000) / m2
pool = (a * m1 + b * m2) / (m1 + m2)
var = pool * (1 - pool) * (1 / m1 + 1 / m2)
hit = (np.abs(a - b) / np.sqrt(np.where(var > 0, var, np.inf)) > zc).mean()
print(f"  worst cell n1={m1}, n2={m2} at p={p:.3f}: exact {exact:.4f}, "
      f"400,000 simulated {hit:.4f}")
print(f"  the level was promised at 0.0500 and the test delivers {exact / 0.05:.2f}x it")
# =>   pooled two-sample z-test of H0: p1 = p2, nominal level 0.05
#      exact size by enumeration, maximised over the nuisance the null leaves free
#         n1    n2   size at p=0.50   size over the null set   attained at p
#         25    25          0.0649                   0.0649           0.500
#         50    50          0.0569                   0.0569           0.500
#        250   250          0.0544                   0.0544           0.500
#         20   200          0.0517                   0.0834           0.009
#         30   300          0.0489                   0.0834           0.006
#      worst cell n1=30, n2=300 at p=0.006: exact 0.0834, 400,000 simulated 0.0831
#      the level was promised at 0.0500 and the test delivers 1.67x it
```

The three balanced rows behave as a textbook leads one to expect. With $n_1=n_2$ the worst point of the null is the middle, $p=0.500$, and the size falls from $0.0649$ at twenty-five observations per group to $0.0544$ at two hundred and fifty as the lattice fills in. A nominal $0.05$ delivering $0.0649$ is already thirty percent more false rejections than advertised, but the error has the direction and magnitude anyone who has thought about discreteness would guess, and it shrinks as it should.

The unbalanced rows are the point, and they are the shape a desk actually has: a short new series against a long established one. At $n_1=20$ against $n_2=200$ the size at the natural checkpoint $p=0.500$ is $0.0517$ — essentially nominal, and an analyst who verified the tool on balanced coin flips would report it fine. Over the null set it was actually promised on, the same test has size $0.0834$, attained at $p=0.009$. At $n_1=30$ against $n_2=300$ the checkpoint reads $0.0489$, *below* nominal, while the supremum is again $0.0834$, at $p=0.006$; four hundred thousand simulated replications there return $0.0831$. A test promised at $0.0500$ delivers $1.67$ times it, precisely in the low-rate regime where a $99\%$ VaR model's breaches live.

**The number that was checked and the number that was promised were computed at two different points of the same null, and only one of them is the size.**

!!! note "The words level, size and significance name three different things, and the literature uses the third for all of them and for a fourth thing that is not a probability at all"
    The **level** is a bound chosen before the data arrives; the **size** is the supremum above, a property the test has whether or not anyone computes it; and every test of level $\alpha$ is also of level $\alpha'$ for any $\alpha'>\alpha$, so "the level" is not even unique while the size is. "Significance level" is used for the first and, sloppily, for the second. "Statistically significant" names the verdict $\varphi=1$ — an event, not a number — and "significant" in a research memo frequently names a fourth thing, the claim that an effect is large enough to matter, which carries no probability at all. That the p-value cannot speak to the fourth is [p-values](03-p-values.md); the two error rates of which the level bounds one are [Type I and Type II Errors](04-type-i-and-type-ii-errors.md).

## The Two Hypotheses Are Not Symmetric, So a Test Can Reject and Can Never Accept

The asymmetry between $\Theta_0$ and $\Theta_1$ comes from the definition of size, not from any philosophy about burden of proof. Bounding the false-rejection rate needs $\mathbf{P}_\theta(R)$ for every $\theta\in\Theta_0$, which requires the null's laws to be specified. Bounding the false-*acceptance* rate would need $\mathbf{P}_\theta(R^c)$ for every $\theta\in\Theta_1$, and the alternative is typically the complement of a point — it contains values arbitrarily close to the null, at which no test of any size distinguishes anything. The supremum of the second error over the alternative is therefore $1-\alpha$ or nearly so, for every test, always: there is no calibration to be done because there is no finite bound to hit.

This is why "fail to reject" is not a euphemism. A test that does not reject has said one thing: the data was not extreme enough, at the chosen level, under the null's law. It has not said the null is true, nor that the effect is small, nor that the data is more consistent with the null than with some particular alternative. What a non-rejection *would* need in order to carry information is a power function, and that is a property of the alternative the test never bounded. [Statistical Power](05-statistical-power.md) computes it and finds that on trading data it is usually small enough that a non-rejection is nearly uninformative by construction.

It also explains the convention that the null is the boring hypothesis. Boredom does not deserve privilege; the boring hypothesis is simply the one specific enough to simulate. "The strategy's mean return is zero" pins down a law up to nuisance parameters and can be sampled from; "the strategy has an edge" names no number and cannot. The null is whichever hypothesis is computable, and the direction of the whole inference follows from that.

## Every Confidence Interval Is the Set of Nulls a Test Failed to Reject, and Every Test Is an Interval Read Backwards

[Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md) states the duality and declines to prove it, on the ground that it is spent there and proved here. Given a family of level-$\alpha$ tests, one per candidate $\theta_0$, define $C(x)$ as the set of $\theta_0$ the corresponding test does not reject. Then $C(x)$ is a $1-\alpha$ confidence set, and conversely any $1-\alpha$ confidence set defines level-$\alpha$ tests by rejecting $\theta_0$ exactly when $\theta_0\notin C(x)$.

??? note "Proof that the set of nulls a family of level-$\alpha$ tests fails to reject is a $1-\alpha$ confidence set, and that the converse construction returns the family it started from"

    For each $\theta_0\in\Theta$ let $\varphi_{\theta_0}$ be a level-$\alpha$ test of the simple null $\{\theta_0\}$, with acceptance region $A(\theta_0)=\{x:\varphi_{\theta_0}(x)=0\}$, so that by the level guarantee
    $$\mathbf{P}_{\theta_0}\big(X\in A(\theta_0)\big)\ \ge\ 1-\alpha \quad\text{for every }\theta_0 .$$
    Define $C(x)=\{\theta_0:x\in A(\theta_0)\}$. The statements $x\in A(\theta_0)$ and $\theta_0\in C(x)$ are, by that definition, the same statement about the pair $(x,\theta_0)$ — $C$ is the relation $A$ read along its other coordinate. Substituting one for the other inside the probability gives
    $$\mathbf{P}_{\theta_0}\big(\theta_0\in C(X)\big)=\mathbf{P}_{\theta_0}\big(X\in A(\theta_0)\big)\ \ge\ 1-\alpha,$$
    which is the definition of a $1-\alpha$ confidence set. Running it the other way, the tests $\tilde\varphi_{\theta_0}(x)=\mathbf{1}\{\theta_0\notin C(x)\}$ have acceptance region $\{x:\theta_0\in C(x)\}=A(\theta_0)$, so the two constructions are mutually inverse and the coverage guarantee becomes the level guarantee read right to left.

    The load-bearing step is that substitution, and what makes it legal is that the probability on both sides is taken under the *same* $\theta_0$ appearing inside the set. Drop that and the identity dissolves: there is no statement here about a fixed interval covering a random parameter, and coverage remains a property of the procedure in exactly the sense [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md) insists on. **A confidence interval is a test run at every null at once and asked to report which ones survived.**

The duality is worth confirming numerically, because it is easy to believe inverting a test is a different computation that happens to land nearby. It is not. The block below inverts the two-sided $t$-test by root-finding on the acceptance region's boundary and compares the endpoints to the textbook formula, then does the same for a variance — where the answer is more interesting, because the equal-tailed test is not the only level-$\alpha$ test available:

```python
import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng(12011)
n = 252
x = rng.normal(0.0004, 0.0120, n)                 # one sample, daily scale
xbar, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
crit = stats.t.ppf(0.975, n - 1)

def rejects(mu0):                                  # the level-0.05 test, run at mu0
    return abs((xbar - mu0) / se) - crit           # > 0 means reject

lo = optimize.brentq(rejects, xbar - 10 * se, xbar, xtol=1e-18, rtol=1e-15)
hi = optimize.brentq(rejects, xbar, xbar + 10 * se, xtol=1e-18, rtol=1e-15)
grid = np.linspace(xbar - 6 * se, xbar + 6 * se, 2001)
kept = np.abs((xbar - grid) / se) <= crit

gap = max(abs(lo - (xbar - crit * se)), abs(hi - (xbar + crit * se)))

print("  the mean: the set of nulls the test keeps, against the textbook interval")
print(f"    inverted test   [{lo:+.9f}, {hi:+.9f}]")
print(f"    closed form     [{xbar - crit * se:+.9f}, {xbar + crit * se:+.9f}]")
print(f"    largest endpoint disagreement  {gap:.1e}")
print(f"    2001 candidate nulls: {kept.sum()} kept, "
      f"{int(np.sum(np.diff(kept.astype(int)) != 0))} sign changes -- one interval")

print("  the variance: two level-0.05 tests of the same hypothesis, both inverted")
print("      n   equal-tailed   shortest   shortest is narrower by")
for m in (21, 63, 252):
    df = m - 1
    w_eq = 1 / stats.chi2.ppf(0.025, df) - 1 / stats.chi2.ppf(0.975, df)

    def width(c1):
        c2 = stats.chi2.ppf(stats.chi2.cdf(c1, df) + 0.95, df)
        return 1 / c1 - 1 / c2

    best = optimize.minimize_scalar(width, bounds=(1e-6, stats.chi2.ppf(0.05, df)),
                                    method="bounded", options={"xatol": 1e-12})
    print(f"    {m:3d}       {w_eq:.6f}   {best.fun:.6f}   {100 * (1 - best.fun / w_eq):20.2f}%")
# =>   the mean: the set of nulls the test keeps, against the textbook interval
#        inverted test   [-0.002179370, +0.000569408]
#        closed form     [-0.002179370, +0.000569408]
#        largest endpoint disagreement  0.0e+00
#        2001 candidate nulls: 657 kept, 2 sign changes -- one interval
#      the variance: two level-0.05 tests of the same hypothesis, both inverted
#          n   equal-tailed   shortest   shortest is narrower by
#         21       0.075001   0.068957                   8.06%
#         63       0.012063   0.011729                   2.77%
#        252       0.001415   0.001405                   0.70%
```

The mean panel is a control and behaves like one. The endpoints found by root-finding on the test's boundary agree with $\bar x\pm t_{0.975,251}\,\mathrm{se}$ to a largest disagreement of $0.0$ — not a small number but the floating-point zero, because the two are algebraically the same expression solved two ways. Scanning $2{,}001$ candidate nulls across six standard errors keeps $657$ in a single contiguous run with exactly two sign changes: the acceptance set is an interval, which is a fact about this test rather than about tests in general, and the duality is usually invisible because for a mean it returns the object everyone already had.

The variance panel is where it earns its keep, because it exposes what the closed form hides. The hypothesis $H_0\!:\sigma=\sigma_0$ can be tested at level $0.05$ by putting $0.025$ in each tail of the $\chi^2$ law, and equally by any other split totalling $5\%$. Both are level-$0.05$ tests of the same hypothesis; inverted, they give different intervals. At $n=21$ the equal-tailed inversion has width $0.075001$ in units of $(n-1)s^2$ against the shortest level-$0.05$ interval's $0.068957$ — the conventional interval is $8.06\%$ wider than necessary at identical coverage. The gap closes as the $\chi^2$ law becomes more symmetric, to $2.77\%$ at $n=63$ and $0.70\%$ at $n=252$, which is why the choice is invisible on long samples and matters exactly where volatility estimation happens, on a short recent window.

**"The" ninety-five percent interval is not unique, because the level-$\alpha$ test it inverts is not unique, and equal tails is a convention that was never optimized for anything.**

## Neyman–Pearson Fixes One Error and Optimizes the Other, Which Is Why a Valid Test Can Be Worse Than Useless

The programme organizing the rest of this part is a constrained optimization: among all tests of size at most $\alpha$, find the one maximizing $\mathbf{P}_\theta(\varphi=1)$ at the alternatives. Fixing one error rate and optimizing the other is the only way to make "best test" mean anything, for the reason [Point Estimation](../part-11-parameter-estimation/01-point-estimation.md) found that risk functions cross: unconstrained, the test that always rejects has perfect power and the test that never rejects has perfect size. When both hypotheses are simple the optimization has an exact solution and it is a likelihood ratio, which is [Likelihood Ratio Tests](06-likelihood-ratio-tests.md).

What the constraint does *not* do is make every test satisfying it worth running. A test can hold its size at exactly $\alpha$, be entirely valid, be the one the textbooks name — and reject *less* often at some alternatives than under the null. Such a test is **biased**, and the property ruling it out, $\beta_\varphi(\theta)\ge\alpha$ for every $\theta\in\Theta_1$, must be imposed as a separate restriction, exactly as unbiasedness of an estimator is imposed in [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md). It does not come free with validity.

??? note "Proof that a valid level-$\alpha$ test can have power strictly below $\alpha$ at some alternatives, so validity does not make a rejection into evidence"

    Take $X_1,\dots,X_n$ normal and test $H_0\!:\sigma=\sigma_0$ with the equal-tailed region: reject when $S=(n-1)s^2/\sigma_0^2$ falls below $c_1=\chi^2_{0.025,\,n-1}$ or above $c_2=\chi^2_{0.975,\,n-1}$. Under $\sigma=r\sigma_0$ we have $S/r^2\sim\chi^2_{n-1}$, so with $F_{n-1}$ the distribution function and $f_{n-1}$ the density the power is
    $$\beta(r)=F_{n-1}\!\big(c_1/r^{2}\big)+1-F_{n-1}\!\big(c_2/r^{2}\big),$$
    and $\beta(1)=0.05$ by construction, so the test is valid. Differentiating at $r=1$ with $\mathrm{d}(c/r^2)/\mathrm{d}r=-2c/r^3$,
    $$\beta'(1)=-2\big[c_1f_{n-1}(c_1)-c_2f_{n-1}(c_2)\big].$$
    The map $c\mapsto c\,f_{n-1}(c)$ is, up to the constant $n-1$, the $\chi^2_{n+1}$ density at $c$; since $c_1$ sits far below the mode of that density and $c_2$ far out in its right tail, $c_1f_{n-1}(c_1)>c_2f_{n-1}(c_2)$ and $\beta'(1)<0$. The power function is *decreasing* as it passes through the null, so on an interval of $r$ ending at $1$ — true volatilities slightly below the hypothesized one — it sits beneath $0.05$, which is the size.

    The load-bearing quantity is the sign of $c_1f_{n-1}(c_1)-c_2f_{n-1}(c_2)$, fixed by the asymmetry of the $\chi^2$ law and known before any data is collected; splitting the $5\%$ unequally so the two products balance is exactly the condition defining the unbiased test. **A valid test keeps a promise about the null, and keeping it says nothing about its behaviour under the alternatives it was built to detect.**

The defect is worth measuring rather than asserting. Both power functions follow in closed form from the $\chi^2$ law, with a simulation at the worst point as a check:

```python
import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng(12017)
r = np.linspace(0.30, 2.50, 4401)                  # true sigma / sigma0

def umpu(df, alpha=0.05):                          # unbiased cutoffs: unequal tails
    def balance(c1):
        c2 = stats.chi2.ppf(stats.chi2.cdf(c1, df) + 1 - alpha, df)
        return stats.chi2.pdf(c1, df + 2) - stats.chi2.pdf(c2, df + 2)
    c1 = optimize.brentq(balance, 1e-9, stats.chi2.ppf(alpha, df) - 1e-12)
    return c1, stats.chi2.ppf(stats.chi2.cdf(c1, df) + 1 - alpha, df)

def power(c1, c2, df):
    return stats.chi2.cdf(c1 / r**2, df) + stats.chi2.sf(c2 / r**2, df)

print("  equal-tailed chi-square test of H0: sigma = sigma0, exact power at level 0.05")
print("     n   min power   at r   power < 0.05 for r in   pw(r=0.5)   UMPU pw(r=0.5)")
for m in (3, 4, 5, 8, 20):
    df = m - 1
    eq = power(stats.chi2.ppf(0.025, df), stats.chi2.ppf(0.975, df), df)
    un = power(*umpu(df), df)
    band = r[eq < 0.05]
    half = int(np.argmin(abs(r - 0.5)))
    print(f"    {m:2d}      {eq.min():.4f}   {r[eq.argmin()]:.3f}   "
          f"[{band.min():.3f}, {band.max():.3f}]{'':10s}{eq[half]:.4f}   {un[half]:13.4f}")

df, m = 4, 5
eq_c = (stats.chi2.ppf(0.025, df), stats.chi2.ppf(0.975, df))
un_c = umpu(df)
eq, un = power(*eq_c, df), power(*un_c, df)
rd = r[eq.argmin()]
s = rng.normal(0, rd, (400_000, m)).var(axis=1, ddof=1) * df
sim = ((s < eq_c[0]) | (s > eq_c[1])).mean()
print(f"  n=5 at the dip r={rd:.3f}: exact {eq.min():.4f}, 400,000 simulated {sim:.4f}")
print(f"  unbiased version over the same grid: min power {un.min():.4f}, never below 0.05")
# =>   equal-tailed chi-square test of H0: sigma = sigma0, exact power at level 0.05
#         n   min power   at r   power < 0.05 for r in   pw(r=0.5)   UMPU pw(r=0.5)
#         3      0.0405   0.857   [0.708, 1.000]          0.0963          0.1559
#         4      0.0429   0.899   [0.795, 1.000]          0.1657          0.2434
#         5      0.0444   0.922   [0.843, 1.000]          0.2528          0.3424
#         8      0.0466   0.954   [0.907, 1.000]          0.5457          0.6318
#        20      0.0487   0.982   [0.966, 1.000]          0.9883          0.9922
#      n=5 at the dip r=0.922: exact 0.0444, 400,000 simulated 0.0446
#      unbiased version over the same grid: min power 0.0500, never below 0.05
```

Read the minimum-power column first. At $n=5$ the equal-tailed test bottoms out at $0.0444$ against a size of exactly $0.0500$: the standard, valid, level-$5\%$ test of a volatility is *less* likely to fire when the true volatility is $92.2\%$ of the hypothesized one than when the null is exactly true. The absolute dip is small — $0.0056$ — and the honest way to report the defect is not the dip but the band. Power stays below the level for every $r$ in $[0.843,\,1.000]$ at $n=5$ and every $r$ in $[0.708,\,1.000]$ at $n=3$: across true volatilities running down to seventy percent of the hypothesized value, a rejection is evidence *against* the alternative that produced it. Four hundred thousand simulated samples at the worst point return $0.0446$ against the exact $0.0444$.

The last two columns cost money. Halving the true volatility — from a hypothesized $20\%$ to a realized $10\%$, not a subtle regime change — is detected at $n=5$ only $25.28\%$ of the time by the equal-tailed test, against $34.24\%$ for the unbiased version at the same level on the same data. At $n=3$ the two read $9.63\%$ and $15.59\%$, an improvement of over sixty percent; by $n=20$ they have converged to $0.9883$ against $0.9922$ and the choice stops mattering. The unbiased test's minimum power over the whole grid is $0.0500$: it touches the level at the null and never goes beneath it. Every number here is available in closed form before any data is collected, and the correction is a different pair of cutoffs from the same distribution.

**The defect is not that the conventional test is wrong; it is that it is right about the only thing it was asked to be right about, and nobody asked about the rest.**

!!! warning "A critical value taken from a table has assumed the null distribution the table was built for, and the assumption is free to check and almost never checked"
    Every number in this part's tables — $1.96$, $3.841$, $t_{0.975,251}$ — is a quantile of a law holding under hypotheses about the data: a distributional family, independence, a sample size at which an approximation has arrived, a nuisance parameter at a particular value. The test verifies none of them and cannot; it consumes the critical value and emits a bit. When an assumption fails, the failure appears as a size that is not the level and nowhere else — no error message, no diagnostic, the reported "$p<0.05$" unchanged in form. The distortions measured here are the mild case; those from dependence are far larger, are measured in [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md), and are what [Permutation Tests](09-permutation-tests.md) and [Bootstrap Tests](10-bootstrap-tests.md) exist to repair. **The free diagnostic is to stop looking the null up and generate it: simulate twenty thousand datasets of your own $n$ from the null you actually believe, dependence included, run the identical code on each, and count rejections at your own $\alpha$ — if the count is not near $\alpha$, the report shows a label rather than a rate.**

## A Test Is a Decision Rule and the Only Thing It Certifies Is Its Own Error Rate

This page established that a hypothesis is a subset of a model and a test a function to $\{0,1\}$ whose rejection region is the primitive; that the size is a supremum over the whole null set, so a pooled proportion test verified at $p=0.500$ and reading $0.0517$ has a true size of $0.0834$ at $p=0.009$; that the asymmetry between the hypotheses is forced by which one is computable, which is why a test rejects and never accepts; that inverting a family of level-$\alpha$ tests returns a $1-\alpha$ confidence set exactly, to an endpoint disagreement of $0.0$, while the non-uniqueness of that family shows up as an equal-tailed variance interval $8.06\%$ wider than necessary at $n=21$; and that a valid test can be biased, the conventional $\chi^2$ volatility test rejecting less often than its own size throughout $[0.843,\,1.000]$ at $n=5$ and detecting a halving $25.28\%$ of the time where its unbiased twin manages $34.24\%$.

What survives is one narrow guarantee. A level-$\alpha$ test certifies that, if the null is true and every assumption behind the critical value holds, it will reject at most $\alpha$ of the time in repeated sampling. It does not certify that a rejection is evidence for the alternative, because power below the level is possible; nor that a non-rejection is evidence for the null, because the second error rate was never bounded; nor that the effect is large, or tradeable, or honestly arrived at. The symmetry with [Point Estimation](../part-11-parameter-estimation/01-point-estimation.md) is exact: there a theorem about a model was read as a fact about an estimate, here a theorem about a procedure is read as a fact about a decision. In both the arithmetic is correct, and the gap between what was proved and what was concluded never appears in the output.

The next question is the one this page left open. A rejection region was treated as primitive, but nobody builds one directly: a statistic is chosen, its null distribution is obtained from somewhere, and the region is a tail of it. Which statistic, and where that null distribution comes from, determine which departures the test can see at all — and a test blind to the departure actually present has power equal to its size, the failure just shown to be possible and shown next to be common. That is [Test Statistics](02-test-statistics.md).

**A test is a promise about how often it will be wrong when nothing is happening, and every other thing anyone wants from it has to be bought separately.**
