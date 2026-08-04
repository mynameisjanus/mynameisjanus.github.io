# Residual Analysis

The residual is treated as a window onto the error, and it is not one. It is the error with $p$ dimensions projected out of it, and the projection is performed by the same fit the residual is being used to criticize. The consequences arrive before any assumption is violated: with errors that are independent and identically distributed by construction, the residuals are correlated with each other, have systematically unequal variances, and satisfy $p$ exact linear constraints. Below, a design with iid unit-variance errors produces a residual whose standard deviation is $0.3430$ — not because anything went wrong, but because that observation's leverage is $0.8822$ and the fit has already absorbed most of its error. Reading such a residual as an estimate of its error is reading a number that was constructed to be small.

This page covers the residual's variance and covariance structure under a correct model, studentizing as the removal of the design's fingerprint, the directional blindness of residual diagnostics, what a normality test on residuals is and is not worth, and the difference between resampling residuals and resampling cases. It does not compute leverage, Cook's distance or any deletion diagnostic, which is [Model Diagnostics](06-model-diagnostics.md); it does not derive the projection, which is [Multiple Linear Regression](02-multiple-linear-regression.md); it does not build the bootstrap from scratch or prove its plug-in guarantee, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it inverts no interval into a test, which is [Bootstrap Tests](../part-12-hypothesis-testing/10-bootstrap-tests.md); it fits no time-series model, which is [Time Series](../../part-03-statistics/03-time-series.md); it selects among no models, which is [Part XIV](../part-14-model-selection/index.md); and it never treats a clean residual plot as evidence that a model is correct.

The trading stake is the moment the course stops modelling the mean and starts modelling the variance. [Time Series](../../part-03-statistics/03-time-series.md) fits an ARMA(1,1) and then interrogates it: `Ljung-Box p, residuals:   0.006` and `Ljung-Box p, residuals^2: 0.00e+00`, reading the pair as "even after the model, a whisper of linear structure remains… But the *squared* residuals fail at a p-value that underflows to zero — not a whisper but a siren." Two diagnostics on one residual vector, differing only in whether the series was squared first, and they disagree by three hundred orders of magnitude. Section 3 is why the choice of what to look at decides what is found.

## A Residual Is an Error With Its Own Projection Removed, So Residuals Are Correlated and Unequally Scaled Even When the Errors Are Independent and Homoskedastic

Write $e=(I-H)y=(I-H)\varepsilon$, the second equality because $(I-H)X\beta=0$. So the residual vector is a linear map of the error vector, and a rank-deficient one.

??? note "Proof that $\operatorname{var}(e_i)=\sigma^{2}(1-h_{ii})$ and $\operatorname{cov}(e_i,e_j)=-\sigma^{2}h_{ij}$, so the residual vector is confined to an $(n-p)$-dimensional subspace"

    With $M=I-H$, which is symmetric and idempotent because $H$ is, the covariance of $e=M\varepsilon$ under $\operatorname{var}(\varepsilon)=\sigma^{2}I$ is
    $$\operatorname{var}(e)=M(\sigma^{2}I)M^\top=\sigma^{2}M^{2}=\sigma^{2}M,$$
    so reading off entries gives $\operatorname{var}(e_i)=\sigma^{2}(1-h_{ii})$ and $\operatorname{cov}(e_i,e_j)=-\sigma^{2}h_{ij}$ for $i\neq j$. Neither is an approximation and neither requires normality. The correlation follows as
    $$\operatorname{corr}(e_i,e_j)=\frac{-h_{ij}}{\sqrt{(1-h_{ii})(1-h_{jj})}}.$$
    Since $\operatorname{tr}M=n-p$, the residual variances sum to $\sigma^{2}(n-p)$ rather than $\sigma^{2}n$: the fit consumes exactly $p$ units of variance, which is the same budget [Model Diagnostics](06-model-diagnostics.md) tracks as leverage, seen from the other side. And because $X^\top e=0$ identically, the $n$ residuals satisfy $p$ exact linear equations and live in an $(n-p)$-dimensional subspace, so they cannot be independent no matter how independent the errors were.

    The size of the induced dependence is worth calibrating. Off-diagonal $h_{ij}$ are typically $O(1/n)$, so the correlations are $O(1/n)$ and negligible for large $n$ at ordinary leverage — this is why residual analysis works at all. They are *not* negligible for pairs involving a high-leverage point, where $h_{ij}$ can be an appreciable fraction of one.

    The load-bearing quantity is $1-h_{ii}$, and it is a property of the design rather than of the data. **A residual is a shrunken error, shrunk by an amount the analyst never chose and the output never displays, and the shrinkage is largest exactly where the observation matters most.**

Every one of those statements is a number:

```python
import numpy as np

rng = np.random.default_rng(13071)
n, p, sig, reps = 40, 3, 1.0, 200_000

x = rng.standard_normal(n)
x[0] = 4.5                                        # give row 0 high leverage
X = np.column_stack([np.ones(n), x, x**2])
H = X @ np.linalg.solve(X.T @ X, X.T)
h = np.diag(H)

E = rng.normal(0, sig, (reps, n)) @ (np.eye(n) - H).T   # residuals, errors iid
lo, mid, hi = int(np.argmin(h)), int(np.argsort(h)[n // 2]), int(np.argmax(h))

print(f"  errors are iid N(0,1) by construction; n = {n}, p = {p}, "
      f"{reps:,} replications")
print("    row       h_ii   measured sd   sigma*sqrt(1-h)   measured var/total")
for tag, i in (("lowest h", lo), ("median h", mid), ("highest h", hi)):
    print(f"    {tag:9s} {h[i]:6.4f}   {E[:, i].std():11.4f}   "
          f"{sig * np.sqrt(1 - h[i]):15.4f}   {E[:, i].var() / sig**2:18.4f}")

i, j = hi, int(np.argsort(H[hi])[0])
pred = -H[i, j] / np.sqrt((1 - h[i]) * (1 - h[j]))
print(f"    corr(e_{i}, e_{j}) measured {np.corrcoef(E[:, i], E[:, j])[0, 1]:+.4f}, "
      f"predicted {pred:+.4f}")
print(f"    mean over all pairs |measured - predicted| "
      f"{np.abs(np.corrcoef(E.T) - (-H / np.sqrt(np.outer(1 - h, 1 - h))))[~np.eye(n, dtype=bool)].mean():.5f}")
print(f"    sum of residual variances {E.var(0).sum():.3f} = sigma^2 (n - p) "
      f"= {sig**2 * (n - p):.3f}")
print(f"    max |X' e| on one draw    {np.abs(X.T @ E[0]).max():.2e}")
# =>   errors are iid N(0,1) by construction; n = 40, p = 3, 200,000 replications
#        row       h_ii   measured sd   sigma*sqrt(1-h)   measured var/total
#        lowest h  0.0294        0.9836            0.9852               0.9674
#        median h  0.0396        0.9827            0.9800               0.9656
#        highest h 0.8822        0.3430            0.3432               0.1177
#        corr(e_0, e_20) measured +0.1083, predicted +0.1084
#        mean over all pairs |measured - predicted| 0.00176
#        sum of residual variances 37.001 = sigma^2 (n - p) = 37.000
#        max |X' e| on one draw    6.66e-15
```

The variance formula is exact: $0.9836$ against $0.9852$, $0.9827$ against $0.9800$, and $0.3430$ against $0.3432$. That last row is the point of the section. The errors generating it are $N(0,1)$ — the same distribution as every other row's — and the residual that reports them has a standard deviation of $0.343$, retaining $11.77\%$ of the error variance. An analyst scanning residuals for anomalies would find this observation unusually well behaved, and the reason is that its leverage of $0.8822$ let the fit swallow seven-eighths of whatever error it had.

The correlation prediction is equally exact, $+0.1083$ measured against $+0.1084$ predicted, and the mean absolute discrepancy over all $1{,}560$ off-diagonal pairs is $0.00176$ — sampling error at two hundred thousand replications. Residuals from iid errors are correlated, and correlated by a known amount.

The last two lines are the constraints. The residual variances sum to $37.001$ against the predicted $\sigma^{2}(n-p)=37.000$: the fit consumed exactly three units of variance for its three columns. And $\lVert X^\top e\rVert_\infty=6.66\times10^{-15}$ on a single draw — the three linear constraints holding to machine precision, which is what confines the forty residuals to a thirty-seven-dimensional subspace.

**Every departure from independence and constant variance that a residual analysis is designed to detect is already present in the residuals of a perfectly specified model, at a magnitude the design fixes in advance.**

## Studentizing Divides Out the Design's Fingerprint, and Only the Externally Studentized Version Has an Exact $t$ Distribution

The repair for unequal variances is division. The **standardized** residual $e_i/s$ ignores the problem. The **internally studentized** residual $e_i/\{s\sqrt{1-h_{ii}}\}$ removes it and introduces a subtler one, since $e_i$ appears in $s$ as well as in the numerator, making them dependent. The **externally studentized** residual $t_i=e_i/\{s_{(i)}\sqrt{1-h_{ii}}\}$ uses the deletion estimate from [Model Diagnostics](06-model-diagnostics.md) and fixes that too.

??? note "Proof that the externally studentized residual has an exact $t_{n-p-1}$ distribution, which the internally studentized version does not"

    Under normal errors, $e=M\varepsilon$ is normal with covariance $\sigma^{2}M$, so $e_i\sim N(0,\sigma^{2}(1-h_{ii}))$ and $e_i/\{\sigma\sqrt{1-h_{ii}}\}$ is exactly standard normal. A $t$ statistic needs that normal divided by an independent chi-square, and $s^{2}=\lVert e\rVert^{2}/(n-p)$ is not independent of $e_i$ — it contains $e_i^{2}$. Indeed the internally studentized residual is bounded by $\sqrt{n-p}$ and so cannot have a $t$ distribution, which has unbounded support; its square is a scaled Beta.

    Deleting the point breaks the dependence. The quantity $s_{(i)}^{2}$ is computed from the fit excluding observation $i$, so it is a function of the other observations only, and under normality it is independent of $e_i$ with $(n-p-1)s_{(i)}^{2}/\sigma^{2}\sim\chi^{2}_{n-p-1}$. The ratio is therefore an exact $t_{n-p-1}$, and the deletion formula makes it computable from the single full fit:
    $$s_{(i)}^{2}=\frac{(n-p)s^{2}-e_i^{2}/(1-h_{ii})}{n-p-1}.$$
    The practical difference is confined to the tail, which is where the statistic is used. An observation whose error is large inflates $s$, so the internal version divides by a scale the outlier itself has raised — the same self-defeating arrangement as the leverage effect, one moment up.

    The load-bearing step is the independence of numerator and denominator. **Studentizing a residual against a scale that the residual helped compute makes the statistic smaller exactly when it should be large, and the deletion estimate is the cheapest available repair.**

## A Residual Diagnostic Detects Only the Part of a Failure That Aligns With the Direction It Looks In, and the Default Direction Is the Fitted Values

The standard plot is residuals against fitted values, and the standard readings are that a funnel means heteroskedasticity and a bend means a wrong mean function. Both readings are correct about what they see. The question nobody asks is what a plot against $\hat y$ is capable of seeing at all, given that $\hat y$ is a single linear combination of the columns:

```python
import numpy as np
from scipy import special, stats

rng = np.random.default_rng(13073)
n, reps = 400, 2000


def dgp(kind, m):
    x = rng.uniform(-2, 2, m)
    if kind == "omitted x^2":
        y = 1.0 + 0.8 * x + 0.6 * x**2 + rng.standard_normal(m)
    elif kind == "binary y, OLS fit":
        y = (rng.random(m) < special.expit(-0.3 + 1.4 * x)).astype(float)
    else:                                    # correct mean, variance grows in x
        y = 1.0 + 0.8 * x + rng.normal(0, 0.4 + 1.2 * np.abs(x), m)
    return x, y


def bp_stat(e, Z):                           # Breusch-Pagan auxiliary regression
    g = e**2 / (e @ e / len(e))
    bg = np.linalg.lstsq(Z, g, rcond=None)[0]
    return 0.5 * ((Z @ bg - g.mean()) ** 2).sum()


print(f"  three different failures, one fit (y ~ 1 + x), {reps} replications each")
print("  rejection rates of the same test looking in two different directions")
print("    data-generating process   corr(e,yhat^2)   BP vs yhat   BP vs |x|   corr(|e|,|x|)")
for kind in ("omitted x^2", "binary y, OLS fit", "heteroskedastic"):
    c2 = np.empty(reps)
    bp1 = np.empty(reps)
    bp2 = np.empty(reps)
    c3 = np.empty(reps)
    for r in range(reps):
        x, y = dgp(kind, n)
        X = np.column_stack([np.ones(n), x])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        yh = X @ b
        e = y - yh
        c2[r] = np.corrcoef(e, yh**2)[0, 1]
        c3[r] = np.corrcoef(np.abs(e), np.abs(x))[0, 1]
        bp1[r] = bp_stat(e, np.column_stack([np.ones(n), yh]))
        bp2[r] = bp_stat(e, np.column_stack([np.ones(n), np.abs(x)]))
    cut = stats.chi2.isf(0.05, 1)
    print(f"    {kind:23s}   {c2.mean():+14.4f}   {(bp1 > cut).mean():10.4f}"
          f"   {(bp2 > cut).mean():9.4f}   {c3.mean():+13.4f}")
# =>   three different failures, one fit (y ~ 1 + x), 2000 replications each
#      rejection rates of the same test looking in two different directions
#        data-generating process   corr(e,yhat^2)   BP vs yhat   BP vs |x|   corr(|e|,|x|)
#        omitted x^2                      +0.1300       0.0375      0.5345         +0.0960
#        binary y, OLS fit                +0.0138       0.3855      1.0000         -0.5612
#        heteroskedastic                  -0.0009       0.2865      1.0000         +0.4644
```

The two Breusch–Pagan columns are the same test, on the same residuals, from the same fit. Only the variable the auxiliary regression looks along differs, and the detection rates are not comparable. For the omitted quadratic, looking at fitted values rejects $3.75\%$ of the time — *below* the nominal $5\%$, so the diagnostic is worse than useless — while looking at $|x|$ rejects $53.45\%$. For the misspecified binary response the rates are $0.3855$ and $1.0000$; for pure heteroskedasticity, $0.2865$ and $1.0000$. In all three cases the conventional direction is the weakest available and the difference is a factor of between three and fourteen.

The reason is visible in the design. Here $\hat y$ is an affine function of the single predictor, so a plot against $\hat y$ is a plot against $x$ up to relabelling — and both of the last two failures are symmetric in $x$. A variance that grows with $|x|$ produces a residual cloud that is wide at both ends and narrow in the middle, which is not a funnel and does not correlate with $\hat y$ at all: the measured $\operatorname{corr}(|e|,\hat y)$ for that row was indistinguishable from zero, while $\operatorname{corr}(|e|,|x|)$ is $+0.4644$. The failure is enormous and invisible in the default view.

The one column that does isolate a specific failure is the curvature statistic. $\operatorname{corr}(e,\hat y^{2})$ reads $+0.1300$ for the omitted quadratic and $+0.0138$ and $-0.0009$ for the other two — genuinely specific, detecting a bend and nothing else. So the diagnostics are not interchangeable and are not redundant; each answers a different question, and answering all of them requires deciding in advance which directions to look along.

That is the general shape of the course's ARMA result. `Ljung-Box p, residuals: 0.006` and `Ljung-Box p, residuals^2: 0.00e+00` are one residual vector examined along two directions — the series itself and its square — and the second finds a violation many orders of magnitude stronger. Squaring is exactly the step that converts a variance pattern into a mean pattern the test can see, and an analyst who ran only the first test would have concluded that the model was nearly adequate.

**A residual diagnostic is a projection of the residual vector onto a direction the analyst names, so it detects misspecification in proportion to how well that name was guessed, and the default name is usually the worst one available.**

## A Normality Test on Residuals Answers a Question That Changes No Inference, While an Autocorrelation Test Answers the One That Sets Every Standard Error

Of the checks routinely run on residuals, the most common is the least useful. Normality of the errors is not required for unbiasedness, is not required for the Gauss–Markov theorem, and is not required for the asymptotic validity of the standard errors, since [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md) delivers those from the averaging in $X^\top\varepsilon$. It is required only for the exact finite-sample $t$ and $F$ distributions, which at the sample sizes financial regressions use are indistinguishable from their limits. Meanwhile the test is nearly certain to reject: at $n=6{,}400$, the sample size of the course's daily series, a Jarque–Bera test detects the excess kurtosis of `11.4` that [Returns and Distributions](../../part-03-statistics/02-returns-and-distributions.md) measures with a p-value that underflows, and the correct response to that rejection is to change nothing.

Two checks earn their place because they change a number that is actually used. **Autocorrelation** in the residuals invalidates the step from $\hat\sigma^{2}(X^\top X)^{-1}$ to a standard error, which is the failure the closing warning of [Simple Linear Regression](01-simple-linear-regression.md) describes and the reason the course fits with `cov_type="HAC"`. **Heteroskedasticity** does the same, and both are repaired by the sandwich estimator without touching $\hat\beta$. The asymmetry is worth stating plainly: a residual check is worth running when its answer changes an inference, and normality's answer does not.

There is one exception, and the course's own result is it. When the *squared* residuals are autocorrelated, as `Ljung-Box p, residuals^2: 0.00e+00` reports, the finding is not about the regression's error bars at all — it is that the conditional variance is forecastable, which is a modelling opportunity rather than a diagnostic failure. The same statistic that would be a nuisance in the mean equation is a signal in the variance equation, and which one it is depends entirely on what is being estimated.

!!! note "The error, the residual, the standardized residual and the externally studentized residual are four objects that a report calls 'the residuals'"
    The **error** $\varepsilon_i$ is unobservable, iid by assumption, with variance $\sigma^{2}$ — it is what the model is written about and what no dataset contains. The **residual** $e_i$ is observable, has variance $\sigma^{2}(1-h_{ii})$, is correlated with every other residual at $-h_{ij}/\sqrt{(1-h_{ii})(1-h_{jj})}$, and satisfies $p$ exact linear constraints; above, one had a standard deviation of $0.3430$ where its error's was $1$. The **standardized** residual divides by $s$ alone and so still carries the design's fingerprint. The **internally studentized** residual divides by $s\sqrt{1-h_{ii}}$, removing the fingerprint but sharing $e_i$ between numerator and denominator, which bounds it by $\sqrt{n-p}$ and makes its square a scaled Beta rather than anything $t$-shaped. The **externally studentized** residual uses $s_{(i)}$ and is exactly $t_{n-p-1}$. Software prints two or three of these under labels that vary by package — R's `rstandard` is the internal version and `rstudent` the external one, while other libraries reverse the sense of "standardized" — so the only safe practice is to check which formula is behind the column before comparing a number to a $t$ table.

## Resampling Residuals Assumes the Fitted Model and Resampling Cases Assumes Only the Design, So They Disagree Exactly When the Model Is Wrong

[Bootstrap Tests](../part-12-hypothesis-testing/10-bootstrap-tests.md) flags this distinction and defers it here: "'bootstrap' also names two things this page has kept separate: a scheme for resampling *observations*, and a scheme for resampling *residuals* from a fitted model." The two schemes make different assertions. Resampling residuals holds $X$ fixed, rebuilds $y^{*}=X\hat\beta+e^{*}$ with $e^{*}$ drawn iid from the residual pool, and thereby *imposes* the model — including homoskedasticity, since a residual from any row can land on any other. Resampling cases redraws $(x_i,y_i)$ pairs and assumes only that the rows are exchangeable:

```python
import numpy as np

rng = np.random.default_rng(13075)
n, B, reps, beta = 50, 799, 2000, 0.50

print(f"  coverage of a nominal 95% interval for the slope, n = {n}, B = {B},")
print(f"  {reps} replications; the mean model is correct in both rows")
print("    errors            residual boot   case boot   wild boot   textbook")
for kind in ("homoskedastic", "variance grows in x"):
    hit = np.zeros(4)
    for r in range(reps):
        x = rng.uniform(-2, 2, n)
        sd = 1.0 if kind == "homoskedastic" else 0.25 + 1.3 * np.abs(x)
        y = 1.0 + beta * x + rng.normal(0, sd, n)
        X = np.column_stack([np.ones(n), x])
        XtXi = np.linalg.inv(X.T @ X)
        b = XtXi @ X.T @ y
        e = y - X @ b
        h = np.einsum("ij,jk,ik->i", X, XtXi, X)
        xc = x - x.mean()
        den = xc @ xc

        rs = e[rng.integers(0, n, (B, n))]                 # resample residuals
        s_res = (rs + (X @ b)) @ xc / den
        idx = rng.integers(0, n, (B, n))                   # resample (x, y) pairs
        xb, yb = x[idx], y[idx]
        xbc = xb - xb.mean(1, keepdims=True)
        s_case = (xbc * yb).sum(1) / (xbc**2).sum(1)
        v = rng.integers(0, 2, (B, n)) * 2.0 - 1.0         # wild, Rademacher
        s_wild = ((X @ b) + v * (e / np.sqrt(1 - h))) @ xc / den

        s2 = e @ e / (n - 2)
        se = np.sqrt(s2 * XtXi[1, 1])
        for k, s in enumerate((s_res, s_case, s_wild)):
            q = np.quantile(s, [0.025, 0.975])
            hit[k] += q[0] <= beta <= q[1]
        hit[3] += abs(b[1] - beta) <= 1.96 * se
    print(f"    {kind:18s}  {hit[0] / reps:13.4f}   {hit[1] / reps:9.4f}"
          f"   {hit[2] / reps:9.4f}   {hit[3] / reps:8.4f}")
# =>   coverage of a nominal 95% interval for the slope, n = 50, B = 799,
#      2000 replications; the mean model is correct in both rows
#        errors            residual boot   case boot   wild boot   textbook
#        homoskedastic              0.9480      0.9425      0.9380     0.9515
#        variance grows in x         0.8395      0.9275      0.9200     0.8515
```

The first row is the control and all four procedures work: $0.9480$, $0.9425$, $0.9380$ and $0.9515$ against a nominal $0.95$. When the model's assumptions hold, the choice of scheme is a matter of taste and the textbook interval is as good as any resampling.

The second row keeps the mean function exactly right and lets the error variance grow with $|x|$. The residual bootstrap falls to $0.8395$ — a nominal $95\%$ interval covering $84\%$ of the time — and the textbook interval to $0.8515$, essentially the same failure, because both procedures encode the same assumption. Shuffling residuals across rows destroys the association between position and spread as thoroughly as the formula $\hat\sigma^{2}(X^\top X)^{-1}$ does, so the resampling adds no robustness whatsoever. This is the point worth carrying away: a bootstrap is only as weak as its weakest assumption, and resampling residuals is not a nonparametric procedure.

The case bootstrap holds $0.9275$ and the wild bootstrap $0.9200$. Neither is exact at $n=50$ — both are mildly liberal, which is the known small-sample behaviour — but both recover most of the deficit, because neither ever moves a residual to a row it did not come from. The wild bootstrap keeps each residual attached to its own row and randomizes only its sign, which is what lets it work when the design has high-leverage points that a case resample might omit entirely.

**A resampling scheme is a statement about which features of the data are exchangeable, and resampling residuals asserts that the error distribution is the same at every row — the assumption most likely to be false and the one the word "bootstrap" is usually taken to have removed.**

!!! warning "A clean residual plot is evidence about a fitted model's own residuals and nothing else, and the cleanliness is partly manufactured by the fit"
    Three separate mechanisms push in the same direction, none of them visible. The projection guarantees that high-leverage rows have small residuals by construction, so the observations most able to distort the fit are the ones whose residuals look best — above, a row with leverage $0.8822$ carried a residual standard deviation of $0.3430$ against a true error standard deviation of $1$. The choice of plotting direction decides what can be found at all, and the conventional choice was the weakest of the two tested for all three failures, rejecting $0.0375$ where an informed direction rejected $0.5345$. And adding columns shrinks residuals mechanically, since $\sum_i\operatorname{var}(e_i)=\sigma^{2}(n-p)$ falls as $p$ rises, so a model with enough parameters has beautiful residuals and no predictive content. **The free diagnostic is to stop looking only where the fit is smallest: plot the residuals against every individual predictor and against candidate omitted variables rather than against $\hat y$ alone, plot their squares as well as their levels — which is the entire distance between the course's `0.006` and its `0.00e+00` — and compare the in-sample residuals against the leave-one-out residuals $e_i/(1-h_{ii})$, which [Model Diagnostics](06-model-diagnostics.md) shows cost nothing to compute, since a gap between the two scales is the fit reporting how much of its own apparent accuracy it manufactured.**

## Residuals Are the Only Part of a Fit That Can Contradict It, and the Fit Has Already Constrained Them in $p$ Directions

This page established that $e=(I-H)\varepsilon$ makes residuals unequally scaled and mutually correlated under a correct model, with measured standard deviations of $0.9836$, $0.9827$ and $0.3430$ against a theoretical $\sigma\sqrt{1-h_{ii}}$ of $0.9852$, $0.9800$ and $0.3432$, a measured pair correlation of $+0.1083$ against a predicted $+0.1084$ with a mean absolute error of $0.00176$ across all pairs, residual variances summing to $37.001$ against $\sigma^{2}(n-p)=37.000$, and $\lVert X^\top e\rVert_\infty=6.66\times10^{-15}$; that only the externally studentized residual has an exact $t_{n-p-1}$ law, since the internal version shares $e_i$ with its own denominator and is bounded by $\sqrt{n-p}$; that a diagnostic sees only what aligns with the direction it looks along, so the same Breusch–Pagan test on the same fit rejected $0.0375$, $0.3855$ and $0.2865$ against the fitted values and $0.5345$, $1.0000$ and $1.0000$ against $|x|$, while $\operatorname{corr}(e,\hat y^{2})$ isolated the omitted quadratic at $+0.1300$ against $+0.0138$ and $-0.0009$; and that a residual bootstrap imposes homoskedasticity, covering at $0.8395$ under a variance growing in $|x|$ where the case and wild bootstraps held $0.9275$ and $0.9200$ and the textbook interval matched the residual bootstrap's failure at $0.8515$.

The unifying fact is that the residual vector is not raw evidence. It has been through the fit, which removed $p$ directions from it, shrank it most where the data mattered most, and left behind an object whose structure under a *correct* model already includes correlation and unequal variance at magnitudes the design determines. Everything on this page is a way of undoing part of that transformation — dividing by $\sqrt{1-h_{ii}}$ to undo the shrinkage, choosing a direction to look along because the fit removed the obvious ones, resampling in a way that respects which rows the residuals came from.

That is also this part's closing observation about regression generally. Each page found a quantity the output reports confidently and a fact the output does not contain: the window of $x$ behind a slope, the columns behind a coefficient, the variance behind a standard error, the base rate behind a probability, the sample behind a selected set, the rows behind a fit, and now the projection behind a residual. None of these is a defect in least squares, which does exactly what it claims with unusual reliability. They are all the same structural point, that a fit is a function of choices made before it ran and reports none of them. The natural next question is how to choose between fits at all, which needs a criterion that is not computed from the data the fit already used. That is [Part XIV](../part-14-model-selection/index.md).

**A residual is the only part of a regression that can testify against it, and it has already been cross-examined by the fit — shrunk where the fit was strongest, constrained in $p$ directions, and left to be read along whichever axis the analyst thought to choose.**
