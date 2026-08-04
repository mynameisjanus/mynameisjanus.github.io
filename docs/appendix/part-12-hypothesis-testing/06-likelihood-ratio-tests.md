# Likelihood Ratio Tests

Everything so far has treated the choice of test statistic as open, and shown what the choice costs. For one problem the choice is closed: when both hypotheses fully specify the data's law, a single statistic is provably optimal and every other statistic is measurably worse. That statistic is the ratio of the two likelihoods, and its optimality is the one clean theorem in this part. What happens next is the interesting half. Real hypotheses are composite, the ratio is generalized by replacing each likelihood with its maximum, and the resulting statistic keeps almost all of its good behaviour — asymptotically, under regularity conditions, one of which fails at exactly the questions a risk system asks.

This page covers the likelihood ratio as the optimal statistic for a simple pair of hypotheses, its generalization to composite hypotheses by maximization, the chi-square calibration Wilks' theorem supplies, the finite-sample disagreement among the Wald, score and likelihood-ratio statistics that agree in the limit, and the failure of the whole expansion when the null sits on the boundary of the parameter space. It does not define the level, the size or the rejection region, which is [The Hypothesis Testing Framework](01-hypothesis-testing-framework.md); it does not develop the properties a test statistic needs, which is [Test Statistics](02-test-statistics.md); it does not construct the likelihood, derive the score equations, or interpret the observed information, which is [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md); it derives no information bound, which is [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md); it ranks no non-nested models and penalizes no complexity, which is [Part XIV](../part-14-model-selection/index.md); it computes no Bayes factor, which is [Part XVI](../part-16-bayesian-statistics/index.md); and it never tests the family itself.

The trading stake is a pair of tests the course writes by hand and then reads correctly. [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md) defines `kupiec()` and `christoffersen()` as generalized likelihood-ratio statistics referred to a `chi2.cdf(..., 1)`, and grades three VaR models with them: `historical VaR 1.121% breaches 48 (1.01x) Kupiec p 9.51e-01` against `parametric-normal breaches 119 (2.50x) Kupiec p 0.00e+00`. The verdict is that "the parametric number is not approximately wrong; it is wrong by a factor of two and a half, with a Kupiec p-value that underflows to zero." Section 4 reproduces both of those p-values exactly, and explains the underflow.

## The Likelihood Ratio Is the Optimal Statistic for a Simple Null Against a Simple Alternative, and Nothing Else Comes Close

When $H_0$ and $H_1$ each specify the law completely, with densities $p_0$ and $p_1$, the **likelihood ratio** $\Lambda(x)=p_1(x)/p_0(x)$ orders samples by how much better the alternative explains them than the null. The Neyman–Pearson lemma says that ordering is the right one: rejecting for large $\Lambda$ is most powerful at its size, and no other statistic does better at any level.

??? note "Proof that rejecting for a large likelihood ratio is most powerful at its own size, which is the Neyman–Pearson lemma"

    Let $\varphi^{\ast}$ reject when $p_1/p_0>k$, accept when $p_1/p_0<k$, and randomize on the boundary so that $\mathbb{E}_0[\varphi^{\ast}]=\alpha$ exactly. Let $\varphi$ be any other test with $\mathbb{E}_0[\varphi]\le\alpha$. Consider the quantity
    $$D=\int\big(\varphi^{\ast}(x)-\varphi(x)\big)\big(p_1(x)-k\,p_0(x)\big)\,\mathrm{d}x .$$
    The integrand is non-negative pointwise. Where $p_1-kp_0>0$ the optimal test has $\varphi^{\ast}=1$, which is the largest value $\varphi$ can take, so $\varphi^{\ast}-\varphi\ge0$ and the product is $\ge0$. Where $p_1-kp_0<0$ we have $\varphi^{\ast}=0\le\varphi$, so $\varphi^{\ast}-\varphi\le0$ and the product is again $\ge0$. Where the two are equal the second factor vanishes. Hence $D\ge0$.

    Expanding $D$ and rearranging,
    $$\mathbb{E}_1[\varphi^{\ast}]-\mathbb{E}_1[\varphi]\ \ge\ k\big(\mathbb{E}_0[\varphi^{\ast}]-\mathbb{E}_0[\varphi]\big)=k\big(\alpha-\mathbb{E}_0[\varphi]\big)\ \ge\ 0,$$
    the last step because $k>0$ and $\mathbb{E}_0[\varphi]\le\alpha$. So $\varphi^{\ast}$ has power at least that of $\varphi$, which is the claim.

    The load-bearing device is adding and subtracting the threshold to build a pointwise non-negative integrand: it converts a constrained optimization over functions into an inequality that holds at every $x$ separately, which is the same manoeuvre that priced the cost-optimal level in [Type I and Type II Errors](04-type-i-and-type-ii-errors.md). Note what the lemma does *not* say: it is silent unless both hypotheses are simple, and its threshold $k$ depends on the alternative, so nothing here yet delivers a test that is optimal against a whole set. **Optimality is available exactly once, for the one case in which there is nothing left to choose.**

The spread the lemma promises is worth measuring rather than asserting, because "most powerful" is a comparative claim and comparisons are only fair at equal size. Below, five statistics are each calibrated by simulation to an exact $5\%$ size against normal data, so the only thing that can differ between them is power:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12061)
n, reps = 252, 40_000
shift = 2.5 / np.sqrt(n)                           # 2.5 standard errors, in sd units

def stats_of(x):
    return {"LR (the mean)": x.mean(1),
            "20% trimmed": stats.trim_mean(x, 0.2, axis=1),
            "median": np.median(x, axis=1),
            "sign test": (x > 0).sum(1).astype(float),
            "first observation": x[:, 0]}

null = stats_of(rng.standard_normal((reps, n)))
alt = stats_of(rng.standard_normal((reps, n)) + shift)

print("  simple H0: mu=0 against H1: mu = 2.5 standard errors, n=252, normal data")
print("  every test calibrated by simulation to an exact 5% size, so only power differs")
print("    statistic              critical value   power")
for name in null:
    c = np.quantile(null[name], 0.95)
    print(f"    {name:20s}   {c:14.4f}   {(alt[name] > c).mean():5.4f}")
# =>   simple H0: mu=0 against H1: mu = 2.5 standard errors, n=252, normal data
#      every test calibrated by simulation to an exact 5% size, so only power differs
#        statistic              critical value   power
#        LR (the mean)                  0.1038   0.8037
#        20% trimmed                    0.1107   0.7588
#        median                         0.1286   0.6426
#        sign test                    139.0000   0.6162
#        first observation              1.6330   0.0677
```

For a normal location problem the likelihood ratio is a monotone function of the sample mean, so the first row *is* the optimal test, and its power of $0.8037$ is the ceiling. Nothing in the table exceeds it, which is the lemma being obeyed. What the lemma does not tell you is the size of the gaps, and they are large enough to matter: the $20\%$ trimmed mean gives up four points at $0.7588$, the median gives up sixteen at $0.6426$, and the sign test — which discards every magnitude and keeps only signs — gives up nearly nineteen at $0.6162$.

The last row is the floor and it is there to calibrate the reader's sense of what a bad statistic costs. Rejecting on the first observation alone is a perfectly valid level-$5\%$ test: its size is exactly $0.05$ by construction, its p-values are uniform under the null, and it will pass any calibration check. Its power against a real effect is $0.0677$, barely above its own size, because $251$ of the $252$ observations were thrown away. This is the failure [Test Statistics](02-test-statistics.md) proved possible, now priced against the optimum on the same data: same level, same hypotheses, and a factor of twelve in detection.

**The lemma guarantees a best statistic exists; the table is what the guarantee is worth, and it is worth about three-quarters of the available power.**

## Generalizing to Composite Hypotheses Replaces Two Likelihoods With Two Maxima and Keeps Almost Everything

Real hypotheses are sets, so neither likelihood is a single number. The **generalized likelihood ratio** replaces each with its maximum over the relevant set,
$$\Lambda=\frac{\sup_{\theta\in\Theta_0}L(\theta)}{\sup_{\theta\in\Theta}L(\theta)}\in(0,1],$$
and the test rejects when $\Lambda$ is small — when the best the null can do falls far short of the best available. In practice one fits the model twice, once with the restriction imposed and once without, and compares the maximized log-likelihoods; the statistic $-2\log\Lambda$ is twice their difference. The estimation machinery is entirely that of [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md), used twice.

What survives the generalization is substantial but weaker than the lemma. The generalized ratio is no longer provably most powerful in finite samples against a composite alternative — no statistic is, as [Test Statistics](02-test-statistics.md) showed when uniform optimality failed two-sided — but it remains most powerful *asymptotically* under regularity, it reduces to the optimal test whenever a uniformly most powerful test exists, and it is defined for any nested pair of models without requiring the analyst to invent a statistic. That last property is why it is the default. The same construction underlies the overidentification statistic of [Method of Moments](../part-11-parameter-estimation/04-method-of-moments.md), which that page produces as "a residual the estimator produced without being asked" and explicitly declines to turn into a decision; the decision is the referral to a null distribution, which is the next section.

## Wilks' Theorem Turns Twice the Log Ratio Into a Chi-Square With the Degrees of Freedom the Restriction Removed

The generalized ratio would be useless without a null distribution, and the remarkable fact is that one exists which does not depend on the model. Under the null and under regularity conditions, $-2\log\Lambda$ converges to a chi-square whose degrees of freedom count the restrictions imposed — nothing else about the family enters.

??? note "Proof that twice the log generalized likelihood ratio converges to a chi-square with degrees of freedom equal to the number of restrictions, by a quadratic expansion about the unrestricted maximum"

    Let $\hat\theta$ maximize the log-likelihood $\ell$ over $\Theta\subseteq\mathbb{R}^{d}$ and let $\hat\theta_0$ maximize it over the null set, a smooth $(d-q)$-dimensional surface, with the true $\theta_0$ in its interior. Expand $\ell$ to second order about $\hat\theta$. The first-order term vanishes because $\hat\theta$ is an interior maximum, so
    $$\ell(\theta)\approx\ell(\hat\theta)-\tfrac12(\theta-\hat\theta)^{\top}\,n\mathcal{I}\,(\theta-\hat\theta),$$
    with $\mathcal{I}$ the Fisher information per observation, which the observed information estimates. Evaluating at $\theta=\hat\theta_0$ and doubling the difference,
    $$-2\log\Lambda=2\big[\ell(\hat\theta)-\ell(\hat\theta_0)\big]\approx n(\hat\theta-\hat\theta_0)^{\top}\mathcal{I}(\hat\theta-\hat\theta_0).$$
    By the asymptotic normality of the maximum likelihood estimator, $\sqrt n(\hat\theta-\theta_0)\Rightarrow\mathcal{N}(0,\mathcal{I}^{-1})$, so $\sqrt n\,\mathcal{I}^{1/2}(\hat\theta-\theta_0)$ is asymptotically standard normal in $\mathbb{R}^{d}$. Restricting to the null surface is, in that coordinate system, an orthogonal projection onto a $(d-q)$-dimensional subspace, and $\hat\theta-\hat\theta_0$ corresponds to the residual in the remaining $q$ directions. The squared length of a standard normal vector's projection onto a $q$-dimensional subspace is $\chi^{2}_{q}$, which is the theorem.

    Three hypotheses did the work and each fails somewhere. The expansion needs $\ell$ twice differentiable near the maximum; the projection argument needs $\mathcal{I}$ non-singular; and — the one that matters here — the vanishing first-order term needs $\hat\theta$ to be an **interior** maximum. Section 5 removes that hypothesis and the conclusion changes.

    The load-bearing quantity is the information matrix, which supplies the metric in which the restriction becomes an orthogonal projection; that it cancels out of the final distribution is why the answer is free of the model. **The chi-square is not a fact about likelihoods, it is a fact about projecting a Gaussian onto a subspace, and the likelihood only had to be smooth enough to look Gaussian.**

## The Wald, Score and Likelihood Ratio Statistics Agree in the Limit and Disagree on Any Sample a Desk Has

The same quadratic picture admits three natural measurements of the same distance. The **likelihood ratio** compares heights of $\ell$ at the two maxima; the **Wald** statistic measures the horizontal distance $\hat\theta-\theta_0$ scaled by the curvature at $\hat\theta$; the **score** statistic measures the slope of $\ell$ at the restricted point $\theta_0$, scaled by the curvature there. On an exactly quadratic log-likelihood the three are identical, and since $\ell$ becomes quadratic as $n$ grows, they share the $\chi^2_q$ limit. On a finite sample they are three different numbers, and the VaR backtest is a place where the difference decides:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12063)
p0, crit = 0.01, stats.chi2.isf(0.05, 1)           # a 99% VaR promises a 1% breach rate

def three(n, x):
    ph = x / n
    inner = (x > 0) & (x < n)
    safe = np.where(inner, ph * (1 - ph) / n, 1.0)
    wald = np.where(inner, (ph - p0) ** 2 / safe, np.inf)    # infinite when x = 0
    score = (ph - p0) ** 2 / (p0 * (1 - p0) / n)
    lo = np.where(x > 0, x * np.log(np.where(x > 0, ph / p0, 1.0)), 0.0)
    hi = np.where(x < n, (n - x) * np.log(np.where(x < n, (1 - ph) / (1 - p0), 1.0)), 0.0)
    return wald, score, 2 * (lo + hi)

print(f"  H0: breach rate = 1%, chi-square(1) critical value {crit:.4f}")
print("        n      x     Wald    score       LR   W/S/LR      LR p (sf)   LR p (1-cdf)")
for n, x in ((250, 6), (500, 11), (1000, 18), (4758, 48), (4758, 119)):
    w, s, l = three(n, np.array(float(x)))
    v = "".join("R" if q > crit else "-" for q in (w, s, l))
    print(f"    {n:5d}  {x:5d}  {w:7.2f}  {s:7.2f}  {l:7.2f}   {v:>6s}   "
          f"{stats.chi2.sf(l, 1):12.3e}   {1 - stats.chi2.cdf(l, 1):13.2e}")

print("  actual size at a nominal 5%, enumerated over the whole support 0..n")
print("        n   P(x=0)     Wald    score       LR")
for n in (250, 500, 1000, 4758):
    k = np.arange(n + 1).astype(float)
    pmf = stats.binom.pmf(k, n, p0)
    w, s, l = three(n, k)
    print(f"    {n:5d}   {pmf[0]:6.4f}   {pmf[w > crit].sum():6.4f}   {pmf[s > crit].sum():6.4f}   "
          f"{pmf[l > crit].sum():6.4f}")

x0 = rng.binomial(250, p0, 400_000).astype(float)
w, s, l = three(250, x0)
print(f"  n=250 simulated: Wald {(w > crit).mean():.4f}   score {(s > crit).mean():.4f}   "
      f"LR {(l > crit).mean():.4f}")
# =>   H0: breach rate = 1%, chi-square(1) critical value 3.8415
#            n      x     Wald    score       LR   W/S/LR      LR p (sf)   LR p (1-cdf)
#          250      6     2.09     4.95     3.56      -R-      5.935e-02        5.94e-02
#          500     11     3.35     7.27     5.42      -RR      1.992e-02        1.99e-02
#         1000     18     3.62     6.46     5.23      -RR      2.226e-02        2.23e-02
#         4758     48     0.00     0.00     0.00      ---      9.513e-01        9.51e-01
#         4758    119    43.96   108.29    76.43      RRR      2.287e-18        0.00e+00
#      actual size at a nominal 5%, enumerated over the whole support 0..n
#            n   P(x=0)     Wald    score       LR
#          250   0.0811   0.0851   0.0412   0.0948
#          500   0.0066   0.1286   0.0377   0.0709
#         1000   0.0000   0.0730   0.0365   0.0551
#         4758   0.0000   0.0474   0.0487   0.0487
#      n=250 simulated: Wald 0.0851   score 0.0411   LR 0.0946
```

The last two rows of the first table are the course's own numbers and both reproduce. At $48$ breaches in $4{,}758$ days all three statistics are $0.00$ and the likelihood ratio's p-value is $9.513\times10^{-1}$, matching the lesson's `Kupiec p 9.51e-01`; at $119$ breaches all three reject overwhelmingly, with the likelihood ratio at $76.43$. The final column explains the lesson's `Kupiec p 0.00e+00`: the true p-value is $2.287\times10^{-18}$, but computing it as `1 - chi2.cdf(...)` subtracts a number indistinguishable from one in floating point and returns exactly zero. The `sf` column recovers it. The p-value did not underflow — the *subtraction* did, and the effect is cosmetic here only because the verdict is the same either way.

The small-sample rows are where the trinity stops being an academic distinction. At $250$ days with $6$ breaches the three read $2.09$, $4.95$ and $3.56$ against a critical value of $3.8415$: the score test rejects, the Wald test does not, and the likelihood ratio does not. One hypothesis, one dataset, three standard statistics, and the answer depends on which was typed. At $500$ and $1{,}000$ days two of three reject. Only by the course's own $4{,}758$ observations do the three agree everywhere.

The size table says which to trust, and the ranking is not the intuitive one. The score statistic is the best calibrated at every sample size, holding $0.0412$, $0.0377$, $0.0365$ and $0.0487$ — always at or below nominal. The likelihood ratio is anti-conservative when the sample is short, at $0.0948$ against a nominal $0.05$. The Wald statistic is both erratic and worst, at $0.0851$, $0.1286$, $0.0730$, $0.0474$ — non-monotone in $n$, and more than twice nominal at $n=500$. The $P(x=0)$ column diagnoses most of the Wald's problem at $n=250$: the outcome of zero breaches has probability $0.0811$, and there $\hat p=0$ makes the Wald statistic's estimated standard error zero and the statistic infinite, so a model that never breached is rejected with infinite confidence. Four hundred thousand simulated samples confirm the enumeration at $0.0851$, $0.0411$ and $0.0946$.

**The three statistics measure one distance in three coordinate systems, and on the sample sizes a risk report actually covers they disagree about the verdict.**

!!! note "A likelihood ratio, a Bayes factor, a likelihood-ratio test statistic and an information criterion are four things built from the same two numbers, and only the third has a null distribution"
    All four begin from two fitted likelihoods. The **likelihood ratio** is their quotient. The **test statistic** is $-2$ times its log, which acquires a null distribution from Wilks and therefore a level and a p-value. An **information criterion** adds a penalty in the number of parameters and produces a ranking with no error rate attached, which is [Information Criteria (AIC/BIC)](../part-14-model-selection/03-information-criteria.md) and which — unlike the test — can compare non-nested models. A **Bayes factor** integrates each likelihood against a prior rather than maximizing it, giving a quantity that is not a function of the maxima at all, and is [Bayesian Model Comparison](../part-16-bayesian-statistics/06-bayesian-model-comparison.md). A related trap is that likelihood *levels* are not invariant to reparameterization even though the argmax is, as [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md) shows with a shift of $n\log 100=29519.1$ from a change of units; the ratio is invariant because the Jacobians cancel, which is the only reason any of this works.

## A Parameter on the Boundary Breaks Wilks' Expansion, and the Repair Is a Mixture Nobody Looks Up

Wilks' proof needed the unrestricted maximum to be interior, so that the first-order term vanishes. When the null sits on the *edge* of the parameter space the estimate cannot move past it, the expansion loses its leading term in that direction, and the limit is no longer a chi-square. This is not a curiosity: "is the variance of this random effect zero", "is this GARCH coefficient zero", "are the returns normal rather than heavy-tailed" are all boundary questions, and all are asked routinely.

??? note "Proof that a null on the boundary replaces the chi-square limit with a half-and-half mixture, because the estimate is truncated in one direction"

    Take one parameter $\psi\ge0$ with the null $\psi=0$ on the boundary, and let $\hat\psi_{\text{u}}$ be the unconstrained maximizer, asymptotically $\mathcal{N}(0,\sigma^{2}/n)$ under the null by the usual expansion. The constrained maximizer over $\psi\ge0$ is the truncation $\hat\psi=\max(\hat\psi_{\text{u}},0)$, since the log-likelihood is locally concave with its peak at $\hat\psi_{\text{u}}$.

    Split on the sign, which is a fair coin asymptotically. If $\hat\psi_{\text{u}}<0$, which happens with probability $\tfrac12$, the constrained maximum is at $\psi=0$; the restricted and unrestricted fits coincide, and $-2\log\Lambda=0$ **exactly**. If $\hat\psi_{\text{u}}>0$, the maximum is interior, the ordinary argument applies in that one direction, and $-2\log\Lambda\Rightarrow\chi^{2}_{1}$. Combining,
    $$-2\log\Lambda\ \Rightarrow\ \tfrac12\,\chi^{2}_{0}+\tfrac12\,\chi^{2}_{1},$$
    where $\chi^{2}_{0}$ is the point mass at zero. The $95$th percentile of that mixture is the $90$th percentile of $\chi^{2}_{1}$, namely $2.7055$ rather than $3.8415$, because half the mass sits at zero and the upper $5\%$ must come out of the remaining half.

    Using the naive $\chi^{2}_{1}$ cutoff therefore produces a test of size $0.025$ rather than $0.05$ — conservative by a factor of two, permanently, with no amount of data repairing it, since the defect is in the limit rather than in the approach to it.

    The load-bearing hypothesis is the interiority of the maximum, and it is the only one of Wilks' three conditions that a modeller controls: it is a consequence of how the alternative was parameterized. **A restriction written as an equality inside the space and the same restriction written at the edge of it are different tests, and only one of them is in the table.**

The theory says the naive cutoff halves the size and the mixture repairs it. Measured at a realistic sample size, the first half is true and the second is not. The block below tests normality against a Student-$t$ alternative, profiling over $1/\nu\ge0$ so that the normal sits exactly at the boundary $1/\nu=0$:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12067)
n, reps = 252, 2_500
inv_nu = np.concatenate([[0.0], np.geomspace(1e-3, 0.5, 29)])   # 1/nu, normal at the boundary

def profile_2dl(x):
    """2 * (best scaled-t log-likelihood - normal log-likelihood), profiling over 1/nu >= 0."""
    m, v = x.mean(1, keepdims=True), x.var(1, keepdims=True)
    ll_norm = -0.5 * n * (np.log(2 * np.pi * v[:, 0]) + 1.0)
    best = ll_norm.copy()
    for q in inv_nu[1:]:
        nu, mu, s2 = 1.0 / q, m.copy(), v.copy()
        for _ in range(25):                        # EM for the scaled-t MLE
            w = (nu + 1) / (nu + (x - mu) ** 2 / s2)
            mu = (w * x).sum(1, keepdims=True) / w.sum(1, keepdims=True)
            s2 = (w * (x - mu) ** 2).sum(1, keepdims=True) / n
        ll = (stats.t.logpdf((x - mu) / np.sqrt(s2), nu) - 0.5 * np.log(s2)).sum(1)
        best = np.maximum(best, ll)
    return 2 * (best - ll_norm)

d0 = profile_2dl(rng.standard_normal((reps, n)))
naive, mix = stats.chi2.isf(0.05, 1), stats.chi2.isf(0.10, 1)
sim = np.quantile(d0, 0.95)

print("  H0: normal against Student-t, so the null sits on the boundary 1/nu = 0")
print(f"  replications whose t-fit cannot beat the normal (2*dl exactly 0): {(d0 <= 1e-9).mean():.4f}")
print("    cutoff                     value   size    power t(8)   power t(5)")
d8, d5 = profile_2dl(rng.standard_t(8, (reps, n))), profile_2dl(rng.standard_t(5, (reps, n)))
for lab, c in (("naive chi2(1)", naive), ("half-half mixture", mix), ("simulated null", sim)):
    print(f"    {lab:22s}  {c:7.4f}   {(d0 > c).mean():.4f}   {(d8 > c).mean():9.4f}   "
          f"{(d5 > c).mean():9.4f}")
# =>   H0: normal against Student-t, so the null sits on the boundary 1/nu = 0
#      replications whose t-fit cannot beat the normal (2*dl exactly 0): 0.6052
#        cutoff                     value   size    power t(8)   power t(5)
#        naive chi2(1)            3.8415   0.0156      0.6136      0.9016
#        half-half mixture        2.7055   0.0328      0.6916      0.9320
#        simulated null           2.0296   0.0500      0.7432      0.9508
```

The boundary is doing exactly what the proof says. In $60.52\%$ of replications the best-fitting Student-$t$ cannot beat the normal at all and the statistic is exactly zero — a spike of mass at the origin that no chi-square has, and that alone makes the standard calibration wrong. The theoretical figure is one half; the excess here is a finite-sample effect at $n=252$, where the $t$ likelihood is flat enough near $1/\nu=0$ that the fit fails to improve slightly more often than the asymptotic coin-flip predicts.

The consequence is the size column. Referring the statistic to the naive $\chi^2_1$ cutoff of $3.8415$ gives an actual size of $0.0156$ — not the nominal $0.05$, and not even the $0.025$ the asymptotic argument predicts. The textbook repair, the $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ cutoff of $2.7055$, improves matters to $0.0328$ and still does not reach nominal. Only the simulated null delivers $0.0500$, at a cutoff of $2.0296$ — well below both published values. The honest reading is that both table lookups are wrong here, one badly and one mildly, and the asymptotic repair is closer to right without being right.

The power columns price the conservatism. Against genuinely heavy-tailed $t(8)$ data, the naive cutoff detects the departure $61.36\%$ of the time, the mixture $69.16\%$, and the correctly calibrated test $74.32\%$ — so using the number from the chi-square table costs thirteen points of power on a question a risk system asks constantly. Against $t(5)$ the same ordering holds at $0.9016$, $0.9320$ and $0.9508$, compressed because the effect is easier. Every one of these tests is *valid*; they simply spend less of their error budget than they were entitled to, which is the same conservatism [p-values](03-p-values.md) measured for discrete nulls, arriving here by a different route.

**The distribution of a likelihood-ratio statistic is a fact about the geometry of the parameter space near the null, and moving the null to the edge changes it in a way no sample size undoes.**

!!! warning "A likelihood-ratio test compares two members of one family and is silent about the family, so a decisive rejection is compatible with both models being wrong"
    Every statistic on this page is a comparison *within* an assumed family: the restricted fit against the unrestricted fit, both computed from the same likelihood. Nothing in the construction can notice that the family itself is misspecified, and the consequences are not symmetric — under misspecification the maximum likelihood estimator converges to the Kullback–Leibler projection rather than to a truth, as [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md) shows, and the ratio of two such projections has no reason to follow Wilks' limit. A rejection then means the restricted projection fits worse than the unrestricted one, which is a statement about two approximations and not about the world. **The free diagnostic is a parametric bootstrap: simulate several thousand datasets from the *fitted unrestricted* model, run the identical two fits and the identical statistic on each, and compare your observed value to that simulated ensemble instead of to a chi-square table — it costs one loop, it is the same move that rescued the size in the block above, and if the simulated null looks nothing like the table you were using, the table was never describing your problem.**

## The Likelihood Ratio Inherits Every Assumption the Likelihood Made, Including the One About the Family

This page established that the likelihood ratio is most powerful for a simple pair of hypotheses, worth $0.8037$ power where a trimmed mean gets $0.7588$, a median $0.6426$, a sign test $0.6162$ and a valid one-observation test $0.0677$; that the generalized ratio replaces both likelihoods with maxima and keeps asymptotic optimality while losing the finite-sample guarantee; that Wilks' theorem calibrates it by projecting a Gaussian onto a subspace, so the degrees of freedom count restrictions and the family cancels; that the Wald, score and likelihood-ratio statistics reproduce the course's `Kupiec p 9.51e-01` and `0.00e+00` exactly while disagreeing about the verdict at $250$ observations, where they read $2.09$, $4.95$ and $3.56$ against a cutoff of $3.8415$, with enumerated sizes putting the score test at $0.0412$, the ratio at $0.0948$ and the Wald at $0.0851$; and that a null on the boundary puts $60.52\%$ of the mass at exactly zero, making the naive cutoff a size-$0.0156$ test that gives up thirteen points of power against $t(8)$ relative to a simulated calibration.

The pattern across the page is that optimality is local and calibration is fragile. The lemma's guarantee is real and is available only when nothing is left to choose; everything after it is an asymptotic argument whose conditions are stated once, rarely checked, and violated by ordinary modelling decisions such as writing a restriction at the edge of a parameter space rather than inside it. What makes this the hardest failure mode in the part is that all three defects — the small-sample disagreement, the boundary spike, the misspecified family — produce output that is completely well-formed. A statistic, a chi-square reference, a p-value to three decimals, and no indication anywhere that the reference was for a different problem.

The next two pages take the same question outward rather than deeper. Rather than assuming a family and comparing two members of it, they ask what can be tested when the family is only partly assumed, or not assumed at all — starting with the tests that do assume a distributional form and are usually defended on the grounds that the assumption does not matter much. That is [Parametric Tests](07-parametric-tests.md).

**A likelihood ratio is the sharpest instrument in this part and it is sharp only about the question of which member, never about which family, and the second question is the one that decides whether the first was worth asking.**
