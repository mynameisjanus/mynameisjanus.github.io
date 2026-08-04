# Simple Linear Regression

The usual complaint about a fitted line is that the relationship might not be linear, and the usual reassurance is that the line is still a reasonable summary. Both halves are wrong in the same way. Least squares does not attempt the conditional mean and fail; it succeeds exactly, at a different problem — it returns the best affine approximation to $\mathbb{E}[y\mid x]$ *over the distribution of $x$ that happened to be sampled*. That qualifier is the whole subject. Change the window of $x$ and the answer changes, not because the estimate is noisy but because the target moved. Below, one fixed relationship with no time variation and no regime change yields slopes of $+0.50$, $-0.38$ and $+1.38$ according to which values of $x$ were in the sample, and every one of those numbers is correct.

This page covers least squares as a projection onto the affine functions, the slope as a covariance over a variance and therefore a property of the sampled window, the Gauss–Markov theorem and the precise thing it guarantees, the attenuation that noise in the predictor forces on the slope, and $R^{2}$ as a statement about the spread of $x$. It does not define or characterize $\mathbb{E}[y\mid x]$ itself, which is [Conditional Expectation](../part-04-expectation-and-moments/06-conditional-expectation.md); it handles no second predictor and needs no matrix inverse, which is [Multiple Linear Regression](02-multiple-linear-regression.md); it derives the estimator from no likelihood, which is [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md); it computes no leverage and deletes no observation, which is [Model Diagnostics](06-model-diagnostics.md); it reads no residual plot, which is [Residual Analysis](07-residual-analysis.md); it penalizes nothing, which is [Regularization](05-regularization.md); it chooses between no two models, which is [Part XIV](../part-14-model-selection/index.md); and it never treats a slope as a property of the world rather than of the sample.

The trading stake is the falsification test the course commits to before it runs anything. [Momentum and Trend Following](../../part-04-strategy-development/01-momentum-and-trend-following.md) regresses next month's return on the sign of the trailing twelve-month return and reports `SPY: slope +0.50%/mo on sign(12m)  (t = +1.11, n = 294)`, with TLT at `+0.03` and `t = +0.11` and GLD at `+0.29` and `t = +0.89`, reading the result as "present, faint, and utterly incapable of impressing a hypothesis test asset by asset." Two features of that specification are the subject of sections 4 and 5: the predictor is a sign, which is a predictor measured with enormous error, and it is two-valued, which fixes what $R^{2}$ can be before any data arrives.

## Least Squares Chooses the Affine Function Closest in Squared Error, Which Equals the Conditional Mean Only When the Conditional Mean Is Affine

The population problem has nothing to do with samples. Fix a joint law for $(x,y)$ and ask for the pair $(\beta_0,\beta_1)$ minimizing $\mathbb{E}\big[(y-\beta_0-\beta_1x)^2\big]$. Differentiating and setting both derivatives to zero gives the two orthogonality conditions $\mathbb{E}[y-\beta_0-\beta_1x]=0$ and $\mathbb{E}[x(y-\beta_0-\beta_1x)]=0$, whose solution is

$$\beta_1=\frac{\operatorname{cov}(x,y)}{\operatorname{var}(x)},\qquad \beta_0=\mathbb{E}[y]-\beta_1\mathbb{E}[x].$$

This object has a name — the **best linear predictor** — and it exists whenever both variances are finite. It does not require linearity, normality, independence, or any statement about $\mathbb{E}[y\mid x]$ whatsoever. It is a projection: [Conditional Expectation](../part-04-expectation-and-moments/06-conditional-expectation.md) shows that $\mathbb{E}[y\mid x]$ is the projection of $y$ onto *all* functions of $x$, and the best linear predictor is the projection onto the two-dimensional subspace spanned by $1$ and $x$. Two nested projections, two different answers, and the second equals the first precisely when $\mathbb{E}[y\mid x]$ already lies in that subspace.

That is the sense in which linearity is, as [Conditional Expectation](../part-04-expectation-and-moments/06-conditional-expectation.md) puts it, "a real assumption and not a formality." The case where it provably holds is joint normality, worked out in [Conditional Gaussian Distributions](../part-06-multivariate-probability/06-conditional-gaussian.md), which is most of why the Gaussian assumption is so hard to abandon: it is the assumption under which the thing regression computes and the thing regression is wanted for coincide.

## The Slope Is a Covariance Over a Variance, So It Is a Property of the Window of $x$ You Sampled and Not of the Relationship

Since $\beta_1$ is a ratio of two moments of the joint law, and both moments depend on the marginal law of $x$, the best linear predictor is indexed by that marginal. Restrict $x$ to a sub-range and you have changed the law, hence the target, hence the correct answer. Nothing about the mechanism generating $y$ from $x$ has to change for the slope to change.

??? note "Proof that the least-squares slope estimates $\operatorname{cov}(x,y)/\operatorname{var}(x)$ under the sampled marginal of $x$, whatever the true mean function is"

    Write the conditional mean as an arbitrary measurable $g(x)=\mathbb{E}[y\mid x]$ and decompose $y=g(x)+\varepsilon$ with $\mathbb{E}[\varepsilon\mid x]=0$ by construction. Then
    $$\operatorname{cov}(x,y)=\operatorname{cov}\big(x,g(x)\big)+\operatorname{cov}(x,\varepsilon)=\operatorname{cov}\big(x,g(x)\big),$$
    the second term vanishing by the tower property. So the population slope is
    $$\beta_1=\frac{\operatorname{cov}\big(x,g(x)\big)}{\operatorname{var}(x)},$$
    a functional of $g$ *and* of the marginal law of $x$ jointly, with the noise playing no role at all. The sample slope $\hat\beta_1=\widehat{\operatorname{cov}}(x,y)/\widehat{\operatorname{var}}(x)$ is a ratio of two sample moments, each consistent for its population counterpart, so by the continuous mapping theorem $\hat\beta_1\to\beta_1$ in probability under whatever marginal generated the sample.

    Concretely, take $g(x)=\alpha x+\gamma x^{2}$. Then $\operatorname{cov}(x,g(x))=\alpha\operatorname{var}(x)+\gamma\operatorname{cov}(x,x^{2})$, and $\operatorname{cov}(x,x^{2})=\mathbb{E}[x^{3}]-\mathbb{E}[x]\mathbb{E}[x^{2}]$, which for a normal $x$ with mean $\mu$ and unit variance equals $2\mu$. The slope is therefore $\alpha+2\gamma\mu$: it moves with where the sample sits, at a rate set by the curvature.

    The load-bearing quantity is $\operatorname{cov}(x,g(x))$, and its dependence on the marginal is not an approximation error that shrinks with $n$ — it is the definition of the estimand. **A slope is a summary of a relationship over a region, and the region is chosen by whoever collected the data rather than by whoever fits the line.**

The prediction is that the same curved relationship, sampled four ways, gives four different slopes, each matching its own population best linear predictor:

```python
import numpy as np
from scipy import integrate, stats

rng = np.random.default_rng(13011)
n = 200_000


def blp_slope(lo, hi):                     # population best linear predictor
    d = stats.norm()
    m = d.cdf(hi) - d.cdf(lo)
    mom = lambda k: integrate.quad(lambda t: t**k * d.pdf(t), lo, hi)[0] / m
    m1, m2, m3 = mom(1), mom(2), mom(3)
    cov_xy = 0.5 * (m2 - m1**2) + 0.4 * (m3 - m1 * m2)
    return cov_xy / (m2 - m1**2)


x = rng.standard_normal(n)
y = 0.5 * x + 0.4 * x**2 + rng.standard_normal(n)

print("  E[y|x] = 0.5x + 0.4x^2 + N(0,1) noise: one relationship, four windows of x")
print("    window            n        slope   population   intercept      R^2")
for name, keep, lo, hi in (("all x", np.ones(n, bool), -np.inf, np.inf),
                           ("x < 0", x < 0, -np.inf, 0.0),
                           ("x > 0", x > 0, 0.0, np.inf),
                           ("0.5 < x < 1.5", (x > 0.5) & (x < 1.5), 0.5, 1.5)):
    xs, ys = x[keep], y[keep]
    b1 = np.cov(xs, ys, ddof=1)[0, 1] / xs.var(ddof=1)
    b0 = ys.mean() - b1 * xs.mean()
    r2 = np.corrcoef(xs, ys)[0, 1] ** 2
    print(f"    {name:13s} {len(xs):7d}  {b1:11.4f}  {blp_slope(lo, hi):11.4f}"
          f"  {b0:10.4f}  {r2:7.4f}")
# =>   E[y|x] = 0.5x + 0.4x^2 + N(0,1) noise: one relationship, four windows of x
#        window            n        slope   population   intercept      R^2
#        all x          200000       0.5003       0.5000      0.4023   0.1600
#        x < 0           99872      -0.3765      -0.3783     -0.3015   0.0476
#        x > 0          100128       1.3834       1.3783     -0.3035   0.4025
#        0.5 < x < 1.5   48223       1.2948       1.2731     -0.3592   0.1147
```

The estimates track their population targets to three decimals in every row — $0.5003$ against $0.5000$, $-0.3765$ against $-0.3783$, $1.3834$ against $1.3783$, $1.2948$ against $1.2731$ — so nothing here is sampling error, and $200{,}000$ observations would not help. The four numbers disagree because they are estimates of four different quantities.

The second and third rows are the finding. On the negative half of the $x$ axis the fitted slope is $-0.3765$; on the positive half it is $+1.3834$. The relationship is identical in both halves, is deterministic up to additive noise, and contains a positive linear term throughout. A researcher handed the left half reports that $x$ predicts $y$ *negatively* at overwhelming significance, and is not making an error. The fourth row shows the same effect without truncation at zero: a narrow interior window gives $1.2948$, two and a half times the full-sample answer.

Notice also that the intercepts are all negative while the full-sample intercept is $+0.4023$, and that $R^{2}$ ranges from $0.0476$ to $0.4025$ across windows of one relationship. Every summary the output offers is window-dependent, and the output does not carry the window.

**A regression coefficient is not a parameter of the world that the sample estimates with error; it is a parameter of the sample's own distribution of $x$, and two honest analysts with different samples are entitled to different answers.**

## Gauss–Markov Buys Efficiency Among Linear Unbiased Estimators and Buys Nothing at All Against a Wrong Mean Function

The theorem that gives least squares its authority is narrower than its reputation. Assume the model $y_i=\beta_0+\beta_1x_i+\varepsilon_i$ with $\mathbb{E}[\varepsilon_i]=0$, $\operatorname{var}(\varepsilon_i)=\sigma^{2}$ and $\operatorname{cov}(\varepsilon_i,\varepsilon_j)=0$; then among all estimators that are linear in $y$ and unbiased for $\beta$, least squares has the smallest variance. No normality is required — that assumption buys the exact $t$ distribution of [Student's t Distribution](../part-05-common-distributions/16-students-t-distribution.md) for the standardized coefficient, and nothing else.

Read the quantifiers carefully. The theorem is conditional on the first line, which asserts that the conditional mean *is* affine. If it is not, $\hat\beta_1$ is not unbiased for anything anyone wanted, and the theorem simply does not apply — it has not been violated, it has been made irrelevant. Section 2's second row is a minimum-variance unbiased estimator of $-0.3783$, efficiently.

The other two conditions are about the errors, and each has a standard failure mode. Unequal error variances leave $\hat\beta_1$ unbiased and cost efficiency, so the estimator is still honest and the standard error is wrong. Correlated errors do the same thing and more severely, which is the subject of this page's closing warning and the reason the course's momentum regression above is fitted with `cov_type="HAC"`. Both are denominator failures in the sense [Parametric Tests](../part-12-hypothesis-testing/07-parametric-tests.md) develops: the estimate survives and the error bar does not.

!!! note "The correlation, the slope, the beta and $R^{2}$ are four readings of one covariance"
    In a one-predictor regression these are the same number in four costumes, and confusing them is easy because three of them are dimensionless. The **covariance** $\operatorname{cov}(x,y)$ carries the units of both variables. The **slope** $\hat\beta_1=\operatorname{cov}(x,y)/\operatorname{var}(x)$ divides by one variance and carries units of $y$ per unit of $x$ — it is what a forecast needs and what changes when the window changes. The **correlation** $\rho=\operatorname{cov}(x,y)/(\sigma_x\sigma_y)$ divides by both and is unitless, so it is invariant to rescaling either variable but *not* to restricting the window. The **$R^{2}$** of the fit is exactly $\rho^{2}$ here and only here — with two or more predictors it stops being any single correlation, which is [Multiple Linear Regression](02-multiple-linear-regression.md). The finance word **beta** names the slope specifically when $x$ is a market return, and the same symbol is used for the whole coefficient vector on the next page. A reported "beta of $0.5$" and a reported "correlation of $0.5$" describe different worlds unless the two volatilities happen to be equal.

## Noise in the Predictor Attenuates the Slope Toward Zero, and the $t$-Statistic Does Not Notice

Every derivation above puts the noise on $y$. Move some of it to $x$ — because the predictor is an estimate, a proxy, a smoothed series, or a sign — and the slope acquires a bias that does not shrink with the sample size and does not appear in any diagnostic.

??? note "Proof that classical measurement error in $x$ multiplies the slope's probability limit by $\sigma^{2}_{x^{*}}/(\sigma^{2}_{x^{*}}+\sigma^{2}_u)$, a factor the regression cannot recover from its own output"

    Let the true relationship be $y=\beta x^{*}+\varepsilon$ with $\operatorname{cov}(x^{*},\varepsilon)=0$, and suppose the analyst observes $x=x^{*}+u$ where $u$ is independent of both $x^{*}$ and $\varepsilon$ — the *classical* error model. The regression of $y$ on the observed $x$ has population slope
    $$\frac{\operatorname{cov}(x,y)}{\operatorname{var}(x)}=\frac{\operatorname{cov}(x^{*}+u,\ \beta x^{*}+\varepsilon)}{\operatorname{var}(x^{*})+\operatorname{var}(u)}=\frac{\beta\,\sigma^{2}_{x^{*}}}{\sigma^{2}_{x^{*}}+\sigma^{2}_{u}}=\lambda\beta,$$
    since every cross term involving $u$ or $\varepsilon$ vanishes. The factor $\lambda\in(0,1]$ is the **reliability ratio**, and the bias is always toward zero: measurement error in a predictor cannot inflate a slope, only shrink it.

    The asymmetry with error in $y$ is the part worth holding onto. Noise added to $y$ enters the numerator's covariance with mean zero and leaves the slope alone, inflating only the residual variance and therefore the standard error. Noise added to $x$ enters the *denominator* as a variance, which is a positive quantity that cannot average out. One kind of noise costs precision; the other costs correctness.

    What makes this untreatable from the regression alone is that $\lambda$ involves $\sigma^{2}_{u}$, and the fit sees only $\operatorname{var}(x)=\sigma^{2}_{x^{*}}+\sigma^{2}_{u}$ — the sum, never the split. Two datasets with identical $\operatorname{var}(x)$, identical $\hat\beta$, identical residuals and identical standard errors can have $\lambda=1$ and $\lambda=0.2$. **The information needed to detect attenuation is not in the sample; it has to come from knowing how the predictor was measured.**

The prediction is a slope multiplied by $\lambda$ and a $t$-statistic that stays enormous:

```python
import numpy as np

rng = np.random.default_rng(13013)
n, reps, beta = 3000, 4000, 0.50

print("  y = 0.50 x* + noise, but x* is observed as x = x* + u; nominal slope 0.50")
print("    sd(u)   attenuation   mean slope   mean t-stat   reject 5%      R^2")
for su in (0.0, 0.25, 0.50, 1.00, 2.00):
    lam = 1.0 / (1.0 + su**2)
    b = np.empty(reps)
    t = np.empty(reps)
    r2 = np.empty(reps)
    for r in range(reps):
        xs = rng.standard_normal(n)
        y = beta * xs + rng.standard_normal(n)
        x = xs + su * rng.standard_normal(n)
        xc, yc = x - x.mean(), y - y.mean()
        b[r] = xc @ yc / (xc @ xc)
        e = yc - b[r] * xc
        s2 = e @ e / (n - 2)
        t[r] = b[r] / np.sqrt(s2 / (xc @ xc))
        r2[r] = 1.0 - (e @ e) / (yc @ yc)
    print(f"    {su:5.2f}   {lam:11.4f}   {b.mean():10.4f}   {t.mean():11.2f}"
          f"   {(np.abs(t) > 1.96).mean():9.4f}  {r2.mean():7.4f}")
# =>   y = 0.50 x* + noise, but x* is observed as x = x* + u; nominal slope 0.50
#        sd(u)   attenuation   mean slope   mean t-stat   reject 5%      R^2
#         0.00        1.0000       0.5000         27.37      1.0000   0.2000
#         0.25        0.9412       0.4706         26.36      1.0000   0.1883
#         0.50        0.8000       0.4000         23.91      1.0000   0.1603
#         1.00        0.5000       0.2501         18.27      1.0000   0.1004
#         2.00        0.2000       0.0997         11.16      1.0000   0.0401
```

The measured slopes are the theoretical ones to four decimals: $0.4706$ against a predicted $0.5\times0.9412$, $0.4000$ against $0.5\times0.8$, $0.2501$ against $0.5\times0.5$, and $0.0997$ against $0.5\times0.2$. The formula is not an approximation and the agreement is not a coincidence.

The last two columns are the reason this matters. In the bottom row the estimated slope is $0.0997$ when the truth is $0.50$ — an error of $80\%$ — and the mean $t$-statistic is $11.16$, with the nominal $5\%$ test rejecting in $1.0000$ of $4{,}000$ replications. Every inferential guarantee on this page is intact: the estimator is consistent for the estimand, the confidence interval of [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md) covers at its stated rate, the p-value is valid. They are all guarantees about $\lambda\beta$, and the analyst wanted $\beta$. The only column that moves is $R^{2}$, from $0.2000$ down to $0.0401$, and a low $R^{2}$ on financial data is so expected that it is never read as evidence of anything.

This is where the course's `sign(12m)` predictor sits. Replacing a continuous trailing return with its sign discards the magnitude entirely, which is measurement error of a severe and deliberate kind, and it guarantees that the reported `+0.50%/mo` is a shrunk version of whatever the underlying relationship is. The direction of the bias is known — toward zero — so the three positive signs the lesson treats as its falsification test are, if anything, understated evidence, while the magnitudes are not interpretable as effect sizes at all.

**Attenuation is the one bias in this part that gets worse as the predictor gets noisier and better as the sample gets smaller, because it lives in the estimand rather than in the sampling error, and no amount of data touches it.**

## $R^{2}$ Measures the Spread of $x$ You Happened to Sample, Which Is Why It Rises With No Change to the Relationship

$R^{2}$ is the fraction of the variance of $y$ that the fit accounts for, which is the sample form of the decomposition in [Law of Total Variance](../part-04-expectation-and-moments/08-law-of-total-variance.md). Written out for one predictor,

$$R^{2}=\frac{\hat\beta_1^{2}\operatorname{var}(x)}{\hat\beta_1^{2}\operatorname{var}(x)+\hat\sigma^{2}},$$

which contains three quantities and only one of them describes the relationship. Hold $\beta_1$ and $\sigma^{2}$ fixed and drive $\operatorname{var}(x)$ up, and $R^{2}$ goes to one; drive it down, and $R^{2}$ goes to zero. The design does all the work:

```python
import numpy as np

rng = np.random.default_rng(13015)
n, reps, beta = 3000, 4000, 0.50

print("  y = 0.50 x + N(0,1): same slope, same error, only the spread of x changes")
print("    design of x        sd(x)   mean slope    se(slope)      R^2")
designs = (("uniform, w=0.5", lambda: rng.uniform(-0.25, 0.25, n)),
           ("uniform, w=2",   lambda: rng.uniform(-1.0, 1.0, n)),
           ("uniform, w=8",   lambda: rng.uniform(-4.0, 4.0, n)),
           ("uniform, w=32",  lambda: rng.uniform(-16.0, 16.0, n)),
           ("sign(z), +-1",   lambda: np.sign(rng.standard_normal(n))))
for name, draw in designs:
    b = np.empty(reps)
    se = np.empty(reps)
    r2 = np.empty(reps)
    sdx = np.empty(reps)
    for r in range(reps):
        x = draw()
        y = beta * x + rng.standard_normal(n)
        xc, yc = x - x.mean(), y - y.mean()
        b[r] = xc @ yc / (xc @ xc)
        e = yc - b[r] * xc
        se[r] = np.sqrt((e @ e / (n - 2)) / (xc @ xc))
        r2[r] = 1.0 - (e @ e) / (yc @ yc)
        sdx[r] = x.std(ddof=1)
    print(f"    {name:15s}  {sdx.mean():6.3f}   {b.mean():10.4f}   {se.mean():10.4f}"
          f"  {r2.mean():7.4f}")
# =>   y = 0.50 x + N(0,1): same slope, same error, only the spread of x changes
#        design of x        sd(x)   mean slope    se(slope)      R^2
#        uniform, w=0.5    0.144       0.5010       0.1265   0.0055
#        uniform, w=2      0.577       0.4997       0.0316   0.0771
#        uniform, w=8      2.309       0.5000       0.0079   0.5716
#        uniform, w=32     9.236       0.5000       0.0020   0.9552
#        sign(z), +-1      1.000       0.5001       0.0183   0.2002
```

Read the slope column first: $0.5010$, $0.4997$, $0.5000$, $0.5000$, $0.5001$. The relationship is the same in all five rows and the estimator finds it in all five. Now read $R^{2}$: $0.0055$, $0.0771$, $0.5716$, $0.9552$. A fit that explains half a percent of the variance and a fit that explains ninety-six percent of it are describing the identical mechanism, and the only thing that changed is how widely $x$ was sampled.

The standard-error column is the correction to the intuition that a low $R^{2}$ means a badly determined slope. It falls monotonically — $0.1265$, $0.0316$, $0.0079$, $0.0020$ — a factor of $63$ across the table, in the *same* direction as $R^{2}$. Wide sampling of $x$ buys both a higher $R^{2}$ and a sharper slope, which is why the two are so easily conflated; but they are not the same claim, because $R^{2}$ also falls when $\sigma^{2}$ rises while the standard error rises with it, and the slope stays right. $R^{2}$ answers "how much of $y$ moves with $x$ here", the standard error answers "how well is the slope pinned down", and only the second is a statement about the estimate.

The final row is the course's predictor. A two-valued $x$ taking $\pm1$ has $\operatorname{var}(x)=1$ by construction, so with a unit-variance error the ceiling on $R^{2}$ is fixed at $\beta^{2}/(\beta^{2}+1)$ regardless of anything else — here $0.2002$, matching the theoretical $0.25/1.25$. When the predictor is a sign, the design's contribution to $R^{2}$ is a constant chosen by the specification rather than measured from the data.

**Reporting an $R^{2}$ without the spread of the predictor that produced it is reporting a ratio while withholding its denominator, and on a sample chosen by anyone other than the analyst the number is not comparable to any other number.**

!!! warning "Autocorrelated residuals leave the slope correct and the standard error too small, so the only wrong column is the one everyone reads"
    Nothing in the output flags it. The coefficient is unbiased, the residual plot looks unremarkable at daily frequency, $R^{2}$ is whatever it was, and the fit converges instantly because there is nothing to converge. What breaks is the step from $\hat\sigma^{2}(X^\top X)^{-1}$ to a standard error, which assumed $\operatorname{cov}(\varepsilon_i,\varepsilon_j)=0$ — and on overlapping windows, on monthly returns built from daily data, on any smoothed or rolling predictor, that covariance is not zero and is usually positive, so the true sampling variance of $\hat\beta_1$ exceeds the reported one and every $t$-statistic on the page is too large. The effect scales with the persistence, not with $n$: more data makes the wrong error bar tighter around the right point estimate. This is why the course's momentum regression is fitted with `cov_type="HAC"` and `maxlags=6` rather than defaults. **The free diagnostic is to compute the same slope three ways and compare only the standard errors: the classical one, a Newey–West HAC one at a lag length matched to your overlap, and a moving-block bootstrap of the slope itself as [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) constructs it; the point estimates will agree to the last digit, and if the three standard errors disagree by more than a few percent then the classical $t$ in your report is the one that assumed your observations were independent.**

## A Fitted Line Answers a Question About Linear Approximation, and Nothing in the Output Records Whether That Was the Question

This page established that least squares solves the projection problem exactly rather than the conditional-mean problem approximately, so its target is the best linear predictor $\operatorname{cov}(x,y)/\operatorname{var}(x)$ under the sampled marginal of $x$; that this target moves with the window, so one curved relationship gave slopes of $+0.5003$, $-0.3765$, $+1.3834$ and $+1.2948$ against population values of $0.5000$, $-0.3783$, $1.3783$ and $1.2731$, with the sign reversing between two halves of the same data; that Gauss–Markov guarantees minimum variance among linear unbiased estimators and guarantees nothing when the mean function is wrong, since section 2's negative slope is efficiently unbiased for $-0.3783$; that classical measurement error in the predictor multiplies the slope by $\sigma^{2}_{x^{*}}/(\sigma^{2}_{x^{*}}+\sigma^{2}_u)$, driving an estimate to $0.0997$ when the truth is $0.50$ while the mean $t$-statistic holds at $11.16$ and the nominal $5\%$ test rejects $1.0000$ of the time; and that $R^{2}$ ran from $0.0055$ to $0.9552$ across five designs whose slopes were all $0.50$, with a two-valued predictor pinned at $0.2002$ by construction.

What unites these is that every one of them is a property the output cannot report because the output does not contain it. The window of $x$, the reliability of the measurement, the spread of the design — all three are facts about how the data came to exist, and a regression summary begins after that. The fit is a function of the sample, and the sample is a function of choices no fitting procedure has access to. This is the concrete form of what [Statistical Models](../part-10-statistics-foundations/04-statistical-models.md) means by a model being an assertion made before the calculation: the assertion here is that the sampled window, the measured predictor and the observed spread are the ones the question was about.

None of this is an argument against fitting lines. It is an argument for reporting those three things alongside the coefficient, and for treating a slope from someone else's sample as a slope from someone else's sample. The natural next move is to stop asking one variable to carry the whole explanation, which changes the geometry more than it changes the algebra: with a second column the coefficient stops being a covariance ratio and becomes a partial effect defined by what the other columns did not already explain. That is [Multiple Linear Regression](02-multiple-linear-regression.md).

**A simple regression returns the best linear approximation to a conditional mean over a region, measured with whatever predictor was available, and the three qualifiers in that sentence are exactly the three things the coefficient table leaves out.**
