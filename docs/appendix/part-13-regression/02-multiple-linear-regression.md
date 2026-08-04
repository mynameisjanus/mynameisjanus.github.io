# Multiple Linear Regression

Adding a second column changes the algebra hardly at all and changes the meaning of the answer completely. With one predictor the slope is a covariance over a variance, a quantity that exists as soon as the joint law does. With two, the coefficient on the first is no longer about the first: it is about the part of the first that the second could not account for, which is a different variable, constructed by the regression itself and never shown to the reader. Every difficulty on this page follows from that substitution. Coefficients move when unrelated columns are added, correlated predictors produce enormous estimates with enormous standard errors that cancel exactly, and the fitted values stay accurate throughout — because the projection is well determined even when the basis it is expressed in is not.

This page covers the normal equations as the stationarity condition of a convex quadratic, the Frisch–Waugh–Lovell theorem that identifies what a coefficient actually measures, the variance decomposition and the joint $F$ test built on it, multicollinearity as ill-conditioning of $X^\top X$, and the fact that in-sample fit is bought partly by column count. It does not treat the one-predictor case or the attenuation and windowing that come with it, which is [Simple Linear Regression](01-simple-linear-regression.md); it fits no non-identity link and no non-Gaussian response, which is [Generalized Linear Models](03-generalized-linear-models.md); it adds no penalty to the objective, which is [Regularization](05-regularization.md); it computes no leverage and no influence measure, which is [Model Diagnostics](06-model-diagnostics.md); it inspects no residual, which is [Residual Analysis](07-residual-analysis.md); it chooses among no set of candidate specifications, which is [Part XIV](../part-14-model-selection/index.md); it corrects for no family of tests, which is [Part XV](../part-15-multiple-testing/index.md); and it never reports a coefficient without the columns that defined it.

The trading stake is the admission gate the course builds for new strategies. [Risk Parity, Diversification, and Factors](../../part-08-portfolio-management/03-risk-parity-diversification-factors.md) regresses each sleeve on all the others and finds `tsmom` carrying an alpha of `−1.02% (t = −0.69)` with an $R^{2}$ of `59.6%` and a loading of `+1.01 on tsmom_meta (t = +33.1)`, against `shortvol` keeping `+13.68% at t = +6.87` with an $R^{2}$ of only `6.7%`, and states the rule it extracts: "a candidate strategy is admitted on its spanning alpha, not its standalone Sharpe." That intercept is a Frisch–Waugh–Lovell residual and section 2 is why; the loading of $+1.01$ at $t=+33.1$ is the collinearity of section 4 seen from the inside.

## The Normal Equations Are the Stationarity Condition of a Convex Quadratic, and the Fitted Values Are an Orthogonal Projection Onto the Column Space

Stack the predictors as the columns of an $n\times p$ matrix $X$, including a column of ones, and minimize $S(\beta)=\lVert y-X\beta\rVert^{2}$. This is the quadratic whose gradient [Calculus Essentials](../part-01-mathematical-foundations/06-calculus-essentials.md) computes: $\nabla S=-2X^\top(y-X\beta)$, and setting it to zero gives the **normal equations**

$$X^\top X\hat\beta=X^\top y,$$

with Hessian $2X^\top X$, positive semidefinite always and positive definite exactly when $X$ has full column rank. So the objective is convex, any stationary point is a global minimum, and the minimizer is unique precisely when no column is a linear combination of the others.

??? note "Proof that the normal equations characterize the minimizer and that the residual is orthogonal to every column, so the fitted values are the orthogonal projection of $y$ onto the column space"

    Expand around any candidate $\beta$ and any perturbation $\delta$:
    $$S(\beta+\delta)=\lVert y-X\beta-X\delta\rVert^{2}=S(\beta)-2\delta^\top X^\top(y-X\beta)+\lVert X\delta\rVert^{2}.$$
    If $\hat\beta$ satisfies $X^\top(y-X\hat\beta)=0$ then the middle term vanishes identically for every $\delta$, leaving $S(\hat\beta+\delta)=S(\hat\beta)+\lVert X\delta\rVert^{2}\ge S(\hat\beta)$ — a global minimum, with equality exactly when $X\delta=0$. Conversely if the middle term is nonzero for some $\delta$, choosing $\delta$ small and in that direction lowers $S$, so no non-solution can be a minimum. The characterization is therefore exact and requires no differentiability argument beyond the expansion.

    The condition $X^\top e=0$ with $e=y-X\hat\beta$ says the residual is orthogonal to every column of $X$ — one equation per column, $p$ exact linear constraints on the $n$ residuals. Writing $\hat y=X\hat\beta=X(X^\top X)^{-1}X^\top y=Hy$ exhibits the fitted values as a linear map of $y$, and $H$ is the orthogonal projector onto the column space: symmetric, idempotent, with $\operatorname{tr}H=p$. Those properties are the subject of [Model Diagnostics](06-model-diagnostics.md), and the $p$ constraints are what [Residual Analysis](07-residual-analysis.md) is about.

    The load-bearing object is the column *space*, not the columns. Any two bases spanning the same space give the same $\hat y$, the same residuals and the same $R^{2}$, while giving completely different coefficient vectors. **The projection is what least squares determines; the coefficients are merely the coordinates it happens to be reported in, and coordinates depend on the basis.**

That last observation is the whole page in one line. Everything that is stable about a fit — predictions, residuals, explained variance — is a property of the span. Everything that is unstable — the coefficients and their standard errors — is a property of the particular columns chosen to express it.

## Frisch–Waugh–Lovell Says Every Coefficient Is a Simple Regression on What Is Left of Its Predictor After the Others Have Explained It

The theorem makes the substitution explicit. To get $\hat\beta_j$ from the full regression, one may instead regress $x_j$ on all the other columns, keep the residual, regress $y$ on all the other columns, keep that residual, and run a one-predictor regression of the second residual on the first. The answer is not approximately the same; it is the same number.

??? note "Proof of the Frisch–Waugh–Lovell theorem, that the multiple-regression coefficient on $x_j$ is the simple-regression coefficient of $y$ on the part of $x_j$ the other columns cannot explain"

    Partition $X=[X_1\ \ X_2]$ and write $\hat\beta$ conformably as $(\hat\beta_1,\hat\beta_2)$. Let $M_2=I-X_2(X_2^\top X_2)^{-1}X_2^\top$ be the residual-maker for $X_2$: it is symmetric, idempotent, and annihilates every column of $X_2$, so $M_2X_2=0$. The normal equations for the full fit read
    $$X_1^\top(y-X_1\hat\beta_1-X_2\hat\beta_2)=0,\qquad X_2^\top(y-X_1\hat\beta_1-X_2\hat\beta_2)=0.$$
    Solving the second for $\hat\beta_2$ gives $\hat\beta_2=(X_2^\top X_2)^{-1}X_2^\top(y-X_1\hat\beta_1)$. Substituting into the first and collecting terms,
    $$X_1^\top M_2\,(y-X_1\hat\beta_1)=0\quad\Longrightarrow\quad \hat\beta_1=\big(X_1^\top M_2X_1\big)^{-1}X_1^\top M_2\,y.$$
    Now use idempotence and symmetry, $M_2=M_2^\top M_2$, to rewrite this as $\hat\beta_1=\big((M_2X_1)^\top(M_2X_1)\big)^{-1}(M_2X_1)^\top(M_2y)$ — which is exactly the least-squares coefficient from regressing the residualized $y$ on the residualized $X_1$. Residualizing $y$ is optional when $X_1$ is a single column, because $M_2 X_1$ is already orthogonal to $X_2$ and the extra projection changes nothing in the numerator.

    The load-bearing step is $M_2X_2=0$: the other columns are removed from $x_j$ *before* $x_j$ is allowed to explain anything, so $\hat\beta_j$ never sees the part of $x_j$ that $X_2$ could have accounted for. **A multiple-regression coefficient is a simple-regression coefficient on a variable that does not appear in the dataset and is constructed differently for every specification.**

Two consequences follow immediately and are visible in one table. If $x_j$ is uncorrelated with the other columns, $M_2$ leaves it alone and the coefficient does not move when they are added. If it is highly correlated, $M_2x_j$ is a small residual and the coefficient is estimated from very little variation:

```python
import numpy as np

rng = np.random.default_rng(13021)
n, rho = 4000, 0.90

z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
x1 = z1
x2 = rho * z1 + np.sqrt(1 - rho**2) * z2
x3 = rng.standard_normal(n)                       # irrelevant, uncorrelated
y = 1.0 * x1 + 0.5 * x2 + rng.standard_normal(n)


def ols(cols):
    X = np.column_stack([np.ones(n)] + list(cols))
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    s2 = e @ e / (n - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1.0 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, r2


print("  y = 1.00 x1 + 0.50 x2 + N(0,1), corr(x1,x2) = 0.90, n = 4000")
print("    specification            b1     se(b1)      R^2")
for name, cols in (("y ~ x1", (x1,)),
                   ("y ~ x1 + x3", (x1, x3)),
                   ("y ~ x1 + x2", (x1, x2)),
                   ("y ~ x1 + x2 + x3", (x1, x2, x3))):
    b, se, r2 = ols(cols)
    print(f"    {name:16s}  {b[1]:8.4f}   {se[1]:8.4f}  {r2:7.4f}")

# Frisch-Waugh-Lovell: residualize x1 and y on the other columns, then regress
W = np.column_stack([np.ones(n), x2])
P = W @ np.linalg.solve(W.T @ W, W.T)
rx1, ry = x1 - P @ x1, y - P @ y
b_fwl = (rx1 @ ry) / (rx1 @ rx1)
b_mult = ols((x1, x2))[0][1]
print(f"    FWL residual regression  {b_fwl:8.4f}")
print(f"    |FWL - multiple| = {abs(b_fwl - b_mult):.2e}")
print(f"    corr(resid x1, x2) = {np.corrcoef(rx1, x2)[0, 1]:.2e}")
# =>   y = 1.00 x1 + 0.50 x2 + N(0,1), corr(x1,x2) = 0.90, n = 4000
#        specification            b1     se(b1)      R^2
#        y ~ x1              1.4716     0.0160   0.6778
#        y ~ x1 + x3         1.4719     0.0160   0.6780
#        y ~ x1 + x2         1.0475     0.0363   0.6908
#        y ~ x1 + x2 + x3    1.0488     0.0363   0.6909
#        FWL residual regression    1.0475
#        |FWL - multiple| = 4.44e-16
#        corr(resid x1, x2) = -3.84e-16
```

The identity is exact: the residual regression returns $1.0475$ and the multiple regression returns $1.0475$, differing by $4.44\times10^{-16}$, which is one unit in the last place of a double. The residualized predictor is orthogonal to $x_2$ to $-3.84\times10^{-16}$, as the theorem requires.

The coefficient column is the finding. Regressed alone, $x_1$ carries $1.4716$; with $x_2$ present it carries $1.0475$, near its generating value of $1.00$. Neither is a mistake. The first estimates $\beta_1+\beta_2\rho=1.45$, because with $x_2$ absent the part of $y$ that $x_2$ generated has nowhere to go but onto the correlated $x_1$; the second estimates $\beta_1$ because $x_2$ is there to claim its own. Adding the *irrelevant* $x_3$ moves $b_1$ from $1.4716$ to $1.4719$ and from $1.0475$ to $1.0488$ — nothing, because $M_3$ leaves an uncorrelated column essentially untouched. Relevance to $y$ is not what determines whether a coefficient moves; correlation with the column in question is.

The standard error more than doubles, $0.0160$ to $0.0363$, for the same reason in reverse. After removing $x_2$, only $\sqrt{1-0.9^{2}}=0.436$ of $x_1$'s standard deviation survives to identify $\beta_1$, and the standard error scales as its reciprocal.

**A coefficient does not describe a predictor; it describes the residual of that predictor against everything else in the model, which is why "controlling for" a variable and "changing the question" are the same operation.**

## The Variance Decomposition Is the Sample Law of Total Variance, and the Joint $F$ Test Asks Whether an Entire Block of Columns Buys Any of It

Because $e$ is orthogonal to $\hat y$, the Pythagorean identity $\lVert y-\bar y\rVert^{2}=\lVert\hat y-\bar y\rVert^{2}+\lVert e\rVert^{2}$ holds exactly, and dividing through defines $R^{2}$. This is the sample form of the decomposition in [Law of Total Variance](../part-04-expectation-and-moments/08-law-of-total-variance.md), and comparing two nested fits by the sum of squares one adds over the other gives the $F$ statistic

$$F=\frac{(\mathrm{RSS}_0-\mathrm{RSS}_1)/q}{\mathrm{RSS}_1/(n-p)},$$

referred to $F_{q,\,n-p}$. This is emphatically not the $F$ test for equal variances that [Parametric Tests](../part-12-hypothesis-testing/07-parametric-tests.md) shows collapsing under kurtosis at a size of $0.4406$; that one compares two variance estimates from different samples and reads a fourth moment, while this one compares two residual sums of squares from nested fits on the *same* sample, and the ratio's numerator and denominator are independent by orthogonality rather than by assumption.

The test the course reaches for is exactly this one — "a regression of returns on the full set of weekday and month dummies… with joint F-tests asking whether *any* partition matters." Whether that is the right instrument depends on where the effect is, which can be measured by holding the total signal fixed and moving it around:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(13023)
n, g, reps, sd = 6400, 5, 20_000, 0.011
nd = n // g
day = np.tile(np.arange(g), nd)
fc = stats.f.isf(0.05, g - 1, n - g)

# two alternatives with identical total signal sum(mu_d - mubar)^2
one = np.array([6.0, 0.0, 0.0, 0.0, 0.0]) * 1e-4
k = np.sqrt(((one - one.mean()) ** 2).sum() / 4.0)
spread = np.array([1.0, 1.0, -1.0, -1.0, 0.0]) * k

print("  weekday means from 6,400 daily returns, sd 1.1%/day; nominal level 0.05")
print("  the two alternatives carry identical total signal, distributed differently")
print("    alternative       joint F (4 df)   targeted t on Mon   Bonferroni over 5")
for name, mu in (("no effect", np.zeros(g)),
                 ("all in Monday", one),
                 ("spread over 4 days", spread)):
    hit_f = hit_t = hit_b = 0
    for lo in range(0, reps, 500):
        m = min(500, reps - lo)
        r = rng.normal(0.0, sd, (m, n)) + mu[day]
        means = r.reshape(m, nd, g).mean(1)
        gm = r.mean(1)
        ssb = nd * ((means - gm[:, None]) ** 2).sum(1)
        ssw = ((r - means[:, day]) ** 2).sum(1)
        f = (ssb / (g - 1)) / (ssw / (n - g))
        se = np.sqrt(ssw / (n - g) / nd)
        t = means / se[:, None]                  # each day against zero, as the lesson does
        hit_f += (f > fc).sum()
        hit_t += (np.abs(t[:, 0]) > stats.t.isf(0.025, n - g)).sum()
        hit_b += (np.abs(t).max(1) > stats.t.isf(0.025 / g, n - g)).sum()
    print(f"    {name:18s}  {hit_f / reps:14.4f}   {hit_t / reps:17.4f}"
          f"   {hit_b / reps:17.4f}")
# =>   weekday means from 6,400 daily returns, sd 1.1%/day; nominal level 0.05
#      the two alternatives carry identical total signal, distributed differently
#        alternative       joint F (4 df)   targeted t on Mon   Bonferroni over 5
#        no effect                   0.0474              0.0550              0.0472
#        all in Monday               0.2505              0.4999              0.2984
#        spread over 4 days          0.2459              0.1389              0.1720
```

The first row is the control and all three procedures are correctly sized — $0.0474$, $0.0550$, $0.0472$ against a nominal $0.05$. Whatever follows is power, not error rate.

The joint $F$ reads $0.2505$ under an effect concentrated entirely in Monday and $0.2459$ under an effect of identical total magnitude spread across four days. Those are the same number, and they are the same number for a reason: the $F$ statistic's non-null law is a noncentral $F$ whose noncentrality is $n_d\sum_d(\mu_d-\bar\mu)^{2}/\sigma^{2}$, a single scalar that the two alternatives were constructed to match. **The joint test cannot tell where the signal is, and in exchange its power does not depend on guessing where the signal is.**

The targeted $t$ is the opposite instrument. Pointed at Monday when Monday is the whole effect, it delivers $0.4999$ — twice the $F$ test's power — and pointed at Monday when the effect is spread, it delivers $0.1389$, roughly half. A factor of $3.6$ separates the same test's power under two alternatives the $F$ test cannot distinguish, and the thing that moved was not the data but the analyst's prior guess. Bonferroni across the five weekday tests sits between the two at $0.2984$ and $0.1720$, paying a fixed toll for scanning and recovering some of the targeting benefit.

This is the honest reading of the course's calendar analysis. Its `19 calendar effects produced one raw rejection (September, p = 0.043)` and zero survivors under any correction, and the joint $F$ specification is what makes that a defensible conclusion rather than a failure to look hard enough: had a real effect been sitting in any single month, the omnibus test would have had the same modest chance of catching it as if the effect had been smeared across four, which is precisely the guarantee a fishing expedition cannot offer.

!!! note "$R^{2}$, adjusted $R^{2}$, the joint $F$ statistic and the squared $t$ of a single coefficient are four readings of the same sum of squares"
    All four are functions of $\mathrm{RSS}_0-\mathrm{RSS}_1$, and their differences are entirely in what they divide by. **$R^{2}$** divides the explained sum of squares by the total and answers "what fraction of the variation is accounted for", which section 5 shows is bought partly by column count. **Adjusted $R^{2}$** divides both pieces by their degrees of freedom, $1-(1-R^{2})\frac{n-1}{n-p}$, so it can fall when a column is added and is a crude penalty rather than a fraction of anything — it is not the $R^{2}$ of any regression and is not bounded below by zero. The **joint $F$** divides the same numerator by $q$ and the residual by $n-p$, converting the comparison into a test with a reference distribution. The **$t$ statistic** of one coefficient, squared, is exactly the $F$ for the single-column restriction $q=1$, so $t^{2}=F$ identically and the two never disagree on one variable. A report quoting $R^{2}=0.69$, an $F$ of $4{,}000$ and a $t$ of $28$ is quoting one quantity three times with three different denominators.

## Multicollinearity Is Ill-Conditioning of $X^\top X$, So Correlated Predictors Give Large Coefficients With Large Standard Errors That Cancel

[Linear Algebra Review](../part-01-mathematical-foundations/05-linear-algebra-review.md) states the diagnosis: "Correlated predictors make $X^\top X$ near-singular, and by the condition-number argument above, the coefficients become wildly sensitive." The coefficient covariance matrix is $\sigma^{2}(X^\top X)^{-1}$, so inverting a nearly-singular matrix inflates variances along its poorly determined directions, and the **variance inflation factor** $1/(1-R_j^{2})$ measures exactly how much for column $j$.

??? note "Proof that with two correlated columns the variance of an individual coefficient scales as $1/(1-\rho^{2})$ while the variance of their sum scales as $1/(1+\rho)$"

    For two centered columns with unit variance and sample correlation $\rho$, $X^\top X\approx n\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}$, whose inverse is $\frac{1}{n(1-\rho^{2})}\begin{pmatrix}1&-\rho\\-\rho&1\end{pmatrix}$. Reading off the entries,
    $$\operatorname{var}(\hat\beta_1)=\frac{\sigma^{2}}{n(1-\rho^{2})},\qquad \operatorname{cov}(\hat\beta_1,\hat\beta_2)=\frac{-\rho\,\sigma^{2}}{n(1-\rho^{2})}.$$
    The first diverges as $\rho\to1$, which is the variance inflation factor. But the covariance diverges too, and negatively, so the variance of the sum is
    $$\operatorname{var}(\hat\beta_1+\hat\beta_2)=\frac{\sigma^{2}}{n(1-\rho^{2})}\big(1+1-2\rho\big)=\frac{2\sigma^{2}}{n(1+\rho)},$$
    which does not diverge — it *decreases* in $\rho$, reaching $\sigma^{2}/n$ at perfect correlation. The eigen-decomposition says the same thing: the eigenvalues of the correlation matrix are $1+\rho$ and $1-\rho$, and the second direction, the difference $\beta_1-\beta_2$, is the one whose information vanishes.

    The load-bearing fact is that ill-conditioning is directional. The data determines the sum precisely and the difference not at all, and a coefficient vector reports neither — it reports two coordinates each of which mixes both. **Nothing is lost by collinearity except the ability to attribute, and attribution is the only thing a coefficient table claims to deliver.**

The prediction is a coefficient standard deviation that explodes, a sum that does not, and predictions that never notice:

```python
import numpy as np

rng = np.random.default_rng(13025)
n, reps = 200, 20_000

print("  y = x1 + x2 + N(0,1), n = 200: only the correlation between the columns moves")
print("    corr    cond(X'X)     sd(b1)   sd(b1+b2)      VIF   out-of-sample MSE")
for rho in (0.0, 0.90, 0.99, 0.999):
    b1 = np.empty(reps)
    bs = np.empty(reps)
    mse = np.empty(reps)
    cond = np.empty(reps)
    for r in range(reps):
        z = rng.standard_normal((n, 2))
        x1 = z[:, 0]
        x2 = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
        X = np.column_stack([np.ones(n), x1, x2])
        y = x1 + x2 + rng.standard_normal(n)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        b1[r], bs[r] = b[1], b[1] + b[2]
        cond[r] = np.linalg.cond(X.T @ X)
        zo = rng.standard_normal((n, 2))
        o1 = zo[:, 0]
        o2 = rho * zo[:, 0] + np.sqrt(1 - rho**2) * zo[:, 1]
        Xo = np.column_stack([np.ones(n), o1, o2])
        yo = o1 + o2 + rng.standard_normal(n)
        mse[r] = np.mean((yo - Xo @ b) ** 2)
    print(f"    {rho:5.3f}  {cond.mean():10.1f}  {b1.std():9.4f}  {bs.std():10.4f}"
          f"  {1 / (1 - rho**2):7.1f}  {mse.mean():17.4f}")
# =>   y = x1 + x2 + N(0,1), n = 200: only the correlation between the columns moves
#        corr    cond(X'X)     sd(b1)   sd(b1+b2)      VIF   out-of-sample MSE
#        0.000         1.3     0.0713      0.1011      1.0             1.0157
#        0.900        19.5     0.1622      0.0733      5.3             1.0150
#        0.990       204.1     0.5113      0.0715     50.3             1.0159
#        0.999      2052.4     1.6019      0.0713    500.3             1.0162
```

The condition number climbs from $1.3$ to $2052.4$ and the sampling standard deviation of $\hat\beta_1$ climbs with it, $0.0713$ to $1.6019$ — a factor of $22$, matching $\sqrt{1/(1-\rho^{2})}=\sqrt{500.3}=22.4$. In the bottom row the true coefficient is $1$ and the estimate has a standard deviation of $1.6$, so its sign is not reliably determined. A single sample from that row produces a coefficient table reporting something like $+2.7$ and $-0.7$ with standard errors of $1.6$ apiece, and a reader would conclude that the second predictor works in reverse.

The third column is why that reader would be wrong. The standard deviation of $\hat\beta_1+\hat\beta_2$ is $0.1011$, $0.0733$, $0.0715$, $0.0713$ — it *falls* as the columns become collinear, exactly as $\sqrt{2/(1+\rho)}$ predicts, and at $\rho=0.999$ the sum is determined fourteen times more precisely than either part. The $+2.7$ and $-0.7$ of the imagined table sum to $2.0$, which is right.

The final column closes the argument. Out-of-sample mean squared error is $1.0157$, $1.0150$, $1.0159$, $1.0162$ across the whole table — flat to three decimals, and equal to the irreducible noise variance of $1$. Collinearity did not damage the fit at all. It damaged only the attribution, and the prediction was never a function of the attribution.

**Multicollinearity is not a problem with the data or the estimator; it is the data truthfully reporting that the question "how much does $x_1$ contribute" was not answerable from a sample in which $x_1$ and $x_2$ always moved together.**

## Adding Columns Never Lowers $R^{2}$, So an In-Sample Fit Is Partly a Count of the Columns That Produced It

The minimum over a larger set cannot exceed the minimum over a smaller one, so appending any column — informative, useless, or random — weakly decreases $\mathrm{RSS}$ and weakly increases $R^{2}$. The increase is not zero on average even when the column is pure noise: adding $k$ irrelevant columns raises the expected $R^{2}$ by about $k(1-R^{2})/n$, which is why section 2's table moves from $0.6778$ to $0.6780$ when the independent $x_3$ is appended, and from $0.6908$ to $0.6909$ when it is appended to the full model. Both increases are real, both are worthless, and no amount of care in reading the number distinguishes them from a real one of the same size.

At $n=4000$ and one column the effect is a rounding error. The regime where it stops being one is the regime finance operates in: with $p$ comparable to $n$, expected $R^{2}$ under a null of no relationship approaches $p/n$, so a specification with thirty columns and two hundred observations explains fifteen percent of the variance of noise. The course's calendar regression carries fifteen dummy columns against $6{,}400$ rows and is safe by three orders of magnitude; a factor model with sixty candidate signals on five years of monthly data is not.

Adjusted $R^{2}$ exists to charge for this and charges the wrong amount for the purpose — it is a fixed degrees-of-freedom correction, not an estimate of out-of-sample performance, and section 4's table is the reminder that in-sample and out-of-sample fit are different measurements even when nothing is overfitted. The honest instruments are cross-validation and the information criteria, both of which are [Part XIV](../part-14-model-selection/index.md).

!!! warning "A coefficient's magnitude and sign are determined by the other columns, and only the final specification reaches the reader"
    The failure leaves no trace in the output. The reported table is internally consistent, the standard errors are correctly computed for the model that was run, the $F$ statistic is valid, and every number would survive an audit — because none of them is wrong. What is missing is that $b_1$ was $1.4716$ before $x_2$ entered and $1.0475$ after, that at $\rho=0.999$ a sign is decided by sampling noise of standard deviation $1.6019$, and that a reader who takes "controlling for $x_2$" as a refinement of the same quantity is reading the coefficient on a variable the model constructed and never displayed. Specification search makes this worse without making it visible, since the specification that survives is the one whose coefficients looked best. **The free diagnostic is to publish the coefficient path rather than the endpoint: refit as each column enters, print $b_j$ and its standard error at every step alongside the condition number of $X^\top X$ and the variance inflation factors, and treat any coefficient that moves by more than its own standard error when a column is added as a quantity the data cannot attribute — then report the well-determined combination, as section 4's sum is determined to $0.0713$ while its parts are determined to $1.6019$.**

## A Coefficient Is Defined by the Other Columns in the Model, Which Is Why It Has No Meaning Outside the Specification That Produced It

This page established that the normal equations $X^\top X\hat\beta=X^\top y$ are the stationarity condition of a convex quadratic whose solution makes the residual orthogonal to every column, so the fit determines a projection and the coefficients are only the coordinates of that projection in a chosen basis; that Frisch–Waugh–Lovell identifies each coefficient as a simple regression on the residual of its own predictor against the others, verified to $4.44\times10^{-16}$, which is why $b_1$ read $1.4716$ alone and $1.0475$ beside a predictor correlated at $0.90$ while an irrelevant orthogonal column moved it only to $1.4719$; that the sum-of-squares decomposition supports a joint $F$ whose power depends on a single noncentrality, delivering $0.2505$ and $0.2459$ under two alternatives of matched total signal where a targeted $t$ swung from $0.4999$ to $0.1389$; that multicollinearity is ill-conditioning, taking $\operatorname{cond}(X^\top X)$ from $1.3$ to $2052.4$ and $\operatorname{sd}(\hat\beta_1)$ from $0.0713$ to $1.6019$ while $\operatorname{sd}(\hat\beta_1+\hat\beta_2)$ *fell* to $0.0713$ and out-of-sample error stayed flat at $1.0157$ to $1.0162$; and that $R^{2}$ rises with every column appended, by about $(1-R^{2})/n$ per useless one.

The unifying observation is that least squares is far more reliable than the object everyone reads out of it. Predictions, residuals and explained variance are properties of the column space and are stable under any reparameterization; coefficients and their standard errors are properties of the basis and can be made arbitrarily unstable by choosing correlated columns, without any deterioration in the fit. A regression that is excellent at its job and useless for interpretation is not a contradiction, and it is the normal condition of a factor model built from signals that all measure the same thing.

The course's spanning gate is the constructive use of this. Regressing `tsmom` on the rest of the book and reading the intercept is asking for the part of one sleeve that the others cannot reproduce — a Frisch–Waugh–Lovell residual by construction — and the loading of `+1.01 at t = +33.1` on its own filtered version is the collinearity of section 4 stated as a finding rather than suffered as a nuisance. What that procedure cannot do is handle a response that is not a continuous number on the whole real line, which is where the next page starts: keeping the linear predictor and changing what it is a linear predictor *of*. That is [Generalized Linear Models](03-generalized-linear-models.md).

**Least squares determines a projection precisely and a set of coordinates loosely, and every controversy about what a regression "shows" is a controversy about the coordinates.**
