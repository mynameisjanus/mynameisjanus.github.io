# Generalized Linear Models

Least squares assumes the response is a real number that can take any value, with noise of constant size added to a mean. Counts are not that, proportions are not that, and durations are not that. The generalized linear model keeps the one part worth keeping — a linear predictor $X\beta$ — and replaces the two parts that were doing the damage: the response gets a distribution chosen from the exponential family, and the linear predictor is connected to its mean through a link rather than being the mean. What emerges is not a family of related methods but a single method, and the code below fits Gaussian, Poisson and binomial responses with the same fourteen lines and a different two-line branch. The price is that naming a family names a variance, the variance is never estimated, and when it is wrong the fit converges cleanly to correct coefficients with standard errors too small by a factor the output does not contain.

This page covers the exponential-family form as the thing that makes the class coherent, the canonical link and the score equations $X^\top(y-\mu)=0$ it produces, iteratively reweighted least squares as Newton's method with the Fisher information, overdispersion as a failure of the variance function rather than of the mean model, and the deviance and the sense in which its $\chi^{2}$ calibration is asymptotic. It does not derive the exponential-family form, the log-partition function's properties or the Pitman–Koopman–Darmois theorem, which is [Exponential Families](../part-10-statistics-foundations/06-exponential-families.md); it does not develop the binary case, its separation failure or its calibration, which is [Logistic Regression](04-logistic-regression.md); it fits nothing by ordinary least squares, which is [Multiple Linear Regression](02-multiple-linear-regression.md); it establishes no property of the maximum likelihood estimator, which is [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md); it penalizes no likelihood, which is [Regularization](05-regularization.md); it analyses no optimizer's convergence on a non-concave surface, which is [Numerical Optimization](../part-17-statistical-computing/01-numerical-optimization.md); it compares no two non-nested families, which is [Part XIV](../part-14-model-selection/index.md); and it never treats a converged fit as evidence that the family was right.

The trading stake is a count the course reasons about without ever writing the model down. [Seasonality and Calendar Effects](../../part-04-strategy-development/04-seasonality-and-calendar-effects.md) scans a calendar and reports that `19 calendar effects produced one raw rejection (September, p = 0.043) — almost exactly the one false positive nineteen null tests owe`, which is an assertion that a count of rejections has a known mean *and a known variance* under the null. That is the smallest possible generalized linear model, and section 4 is what happens to the second half of the assertion when the tests are not independent — because the count's mean survives dependence and its variance does not.

## A Generalized Linear Model Is a Choice of Exponential Family, a Linear Predictor and a Link, and Only the Third Is Free

The specification has exactly three parts. First, a **random component**: the response $y_i$ is drawn independently from a one-parameter exponential family, whose density [Exponential Families](../part-10-statistics-foundations/06-exponential-families.md) writes as

$$f(y;\theta,\phi)=\exp\!\left\{\frac{y\theta-b(\theta)}{\phi}+c(y,\phi)\right\},$$

with $\theta$ the natural parameter, $b$ the log-partition function and $\phi$ a dispersion. Second, a **systematic component**: a linear predictor $\eta_i=x_i^\top\beta$. Third, a **link** $g$ joining them, $g(\mu_i)=\eta_i$, where $\mu_i=\mathbb{E}[y_i]$.

The point of the first component is that it makes the second derivative of $b$ do double duty.

??? note "Proof that $b'(\theta)=\mu$ and $b''(\theta)=\operatorname{var}(y)/\phi$, so naming the family fixes the variance as a deterministic function of the mean"

    Integrating the density to one gives $\int\exp\{(y\theta-b(\theta))/\phi+c(y,\phi)\}\,dy=1$ for every $\theta$. Differentiate both sides with respect to $\theta$ and exchange the derivative with the integral, which the exponential family's regularity permits:
    $$\int\frac{y-b'(\theta)}{\phi}f(y;\theta,\phi)\,dy=0\quad\Longrightarrow\quad \mathbb{E}[y]=b'(\theta)=\mu.$$
    Differentiating a second time,
    $$\int\left[\left(\frac{y-b'(\theta)}{\phi}\right)^{2}-\frac{b''(\theta)}{\phi}\right]f\,dy=0\quad\Longrightarrow\quad \operatorname{var}(y)=\phi\,b''(\theta).$$
    Since $\mu=b'(\theta)$ is invertible wherever $b$ is strictly convex, $\theta$ is a function of $\mu$, and therefore so is $b''(\theta)$. Writing $V(\mu)=b''(\theta(\mu))$ — the **variance function** — gives $\operatorname{var}(y)=\phi V(\mu)$. For the Poisson, $b(\theta)=e^{\theta}$, $\mu=e^{\theta}$ and $V(\mu)=\mu$ with $\phi=1$: the variance equals the mean and there is no free parameter left to absorb a disagreement. For the binomial with $m$ trials, $V(\mu)=\mu(1-\mu/m)$, also with $\phi=1$. For the Gaussian, $b(\theta)=\theta^{2}/2$ and $V(\mu)=1$ with $\phi=\sigma^{2}$ free, which is the only common case where the dispersion is estimated rather than asserted.

    The load-bearing consequence is the direction of the implication. In least squares the mean model and the variance are separate assumptions and $\sigma^{2}$ is estimated from the residuals; here the family determines $V$, and $\phi$ is fixed at $1$ for the two most-used members. **Choosing Poisson or binomial is not choosing a shape for the noise — it is asserting a specific number for the variance at every fitted mean, and asserting it without measurement.**

The link is the only genuinely free choice, and one choice is distinguished. Setting $g$ so that $\eta=\theta$ — the **canonical link**, $\log\mu$ for the Poisson and $\log\frac{\mu}{m-\mu}$ for the binomial — makes the natural parameter linear in $\beta$, which is what collapses the whole class onto one algorithm.

## The Canonical Link Makes the Score Equations Read $X^\top(y-\mu)=0$, Which Is Why Every Model in the Class Fits With the Same Routine

Under the canonical link the log-likelihood is $\ell(\beta)=\sum_i\{y_i x_i^\top\beta-b(x_i^\top\beta)\}/\phi$ plus terms free of $\beta$, and differentiating gives a score of startling simplicity.

??? note "Proof that under the canonical link the score is $X^\top(y-\mu)/\phi$ and the observed and expected Hessians coincide, so Newton's method and iteratively reweighted least squares are the same algorithm"

    With $\eta_i=x_i^\top\beta=\theta_i$, the chain rule gives $\partial\ell/\partial\beta=\sum_i x_i\{y_i-b'(\theta_i)\}/\phi$, and since $b'(\theta_i)=\mu_i$ this is
    $$\nabla\ell(\beta)=\frac{1}{\phi}X^\top(y-\mu).$$
    Setting it to zero says the residual $y-\mu$ is orthogonal to every column of $X$ — structurally identical to the normal equations of [Multiple Linear Regression](02-multiple-linear-regression.md), except that $\mu$ is now a nonlinear function of $\beta$, so the system is not solved in closed form.

    Differentiating once more, $\partial\mu_i/\partial\beta=b''(\theta_i)x_i=V(\mu_i)x_i$, so
    $$\nabla^{2}\ell(\beta)=-\frac{1}{\phi}X^\top WX,\qquad W=\operatorname{diag}\big(V(\mu_i)\big),$$
    which contains no $y$ at all. The observed Hessian is therefore already non-random given $X$, so it equals its own expectation and Newton's method coincides exactly with Fisher scoring. Because $W$ has strictly positive entries wherever $b$ is strictly convex, $-\nabla^{2}\ell$ is positive definite whenever $X$ has full column rank: the log-likelihood is strictly concave and the maximum is unique.

    Newton's update $\beta^{+}=\beta+(X^\top WX)^{-1}X^\top(y-\mu)$ rearranges to $\beta^{+}=(X^\top WX)^{-1}X^\top Wz$ with the **working response** $z=\eta+W^{-1}(y-\mu)$, which is precisely a weighted least-squares fit of $z$ on $X$ with weights $W$. **The entire class is one weighted regression iterated, and the only thing that changes between a Poisson and a logistic model is two lines computing $\mu$ and $W$.**

That claim is checkable directly, by writing the routine once and pointing it at three different responses:

```python
import numpy as np

rng = np.random.default_rng(13031)
n = 2000
x = rng.standard_normal(n)
X = np.column_stack([np.ones(n), x])
true = np.array([0.40, 0.75])


def irls(X, y, family, m=1.0, tol=1e-12, maxit=100):
    """One routine for every canonical-link GLM in the exponential family."""
    b = np.zeros(X.shape[1])
    for it in range(1, maxit + 1):
        eta = X @ b
        if family == "gaussian":
            mu, w = eta, np.ones_like(eta)
        elif family == "poisson":
            mu = np.exp(eta)
            w = mu
        else:                                    # binomial, m trials per row
            p = 1.0 / (1.0 + np.exp(-eta))
            mu, w = m * p, m * p * (1.0 - p)
        z = eta + (y - mu) / w                   # working response
        WX = X * w[:, None]
        nb = np.linalg.solve(X.T @ WX, WX.T @ z)
        step = np.max(np.abs(nb - b))
        b = nb
        if step < tol:
            break
    return b, it, mu


eta = X @ true
ys = {"gaussian": eta + rng.standard_normal(n),
      "poisson": rng.poisson(np.exp(eta)),
      "binomial": rng.binomial(20, 1.0 / (1.0 + np.exp(-eta)))}

print("  the same IRLS routine, three exponential families, canonical links only")
print("    family     iters   ||X'(y-mu)||        b0        b1   max|b - truth|")
for fam, y in ys.items():
    m = 20.0 if fam == "binomial" else 1.0
    b, it, mu = irls(X, y.astype(float), fam, m=m)
    print(f"    {fam:9s}  {it:5d}   {np.abs(X.T @ (y - mu)).max():12.2e}"
          f"  {b[0]:8.4f}  {b[1]:8.4f}   {np.abs(b - true).max():14.4f}")
print(f"    truth                            {true[0]:8.4f}  {true[1]:8.4f}")

b_ols = np.linalg.lstsq(X, ys["gaussian"], rcond=None)[0]
b_irls = irls(X, ys["gaussian"], "gaussian")[0]
print(f"    gaussian IRLS vs lstsq: max diff {np.abs(b_ols - b_irls).max():.2e}")
# =>   the same IRLS routine, three exponential families, canonical links only
#        family     iters   ||X'(y-mu)||        b0        b1   max|b - truth|
#        gaussian       2       1.84e-12    0.4332    0.7146           0.0354
#        poisson        9       1.75e-12    0.4085    0.7422           0.0085
#        binomial       6       3.44e-12    0.4008    0.7303           0.0197
#        truth                              0.4000    0.7500
#        gaussian IRLS vs lstsq: max diff 9.99e-16
```

The score residual $\lVert X^\top(y-\mu)\rVert_\infty$ lands at $1.84\times10^{-12}$, $1.75\times10^{-12}$ and $3.44\times10^{-12}$ — the orthogonality the proof demands, satisfied to a dozen digits by a routine that was never told which family it was fitting beyond a two-line branch. The coefficients recover $(0.40,0.75)$ to $0.0354$, $0.0085$ and $0.0197$, which is sampling error at $n=2000$ and not algorithmic error.

The iteration counts carry the section's other claim. The Gaussian case terminates at $2$ — one step, plus one more to observe that nothing changed — because with $W=I$ and $z=y$ the first update *is* the least-squares solution, agreeing with `np.linalg.lstsq` to $9.99\times10^{-16}$. Ordinary least squares is not analogous to a GLM; it is the member of the class whose weights happen to be constant, which is why its normal equations required no iteration in the first place. The Poisson takes $9$ and the binomial $6$, both to a tolerance of $10^{-12}$ on the coefficient step, which is Newton's quadratic convergence on a strictly concave surface: single-digit iteration counts are what concavity buys, and a GLM that needs fifty is reporting something about its design matrix rather than about its family.

**A generalized linear model is not a generalization of least squares in the sense of being harder; it is the observation that least squares was one weighted regression and the rest of the class is the same weighted regression run more than once.**

## Iteratively Reweighted Least Squares Converges Because the Surface Is Concave, and Concavity Is a Property of the Family Rather Than of the Data

Two features of the previous proof deserve to be stated separately, because they are what make the fitting step boring — and boring is the correct ambition for an optimizer.

The first is uniqueness. Strict concavity means there is one maximum, no local optima, and no dependence on the starting value; the routine above starts at $\beta=0$ every time and would reach the same answer from anywhere. This is the exception rather than the rule in model fitting, and it is inherited entirely from the convexity of $b$, which is a fact about the exponential family established in [Exponential Families](../part-10-statistics-foundations/06-exponential-families.md) rather than a fact about the dataset. Contrast the non-concave surfaces of [Numerical Optimization](../part-17-statistical-computing/01-numerical-optimization.md), where the starting value is a modelling decision.

The second is that the Hessian $-X^\top WX/\phi$ contains no $y$. The curvature of the log-likelihood at any $\beta$ is fixed once $X$ and the fitted means are known, so the information matrix is available without a second pass over the data, and the coefficient covariance is read off as $(X^\top WX)^{-1}\phi$ — the exact analogue of $\sigma^{2}(X^\top X)^{-1}$, with $W$ standing where the constant variance used to. This is the formula the next section attacks, and the attack is not on its derivation, which is correct, but on the $\phi$ sitting at the front of it.

!!! note "The dispersion parameter, overdispersion, the scale factor and the Pearson $\chi^{2}/\mathrm{df}$ ratio are four names for one number"
    The **dispersion** $\phi$ is the parameter in the density above; for the Gaussian it is $\sigma^{2}$ and is estimated, while for the Poisson and binomial it is fixed at $1$ by the algebra of the family, not by a modelling choice anyone made deliberately. **Overdispersion** names the situation where the data's variance exceeds $\phi V(\mu)$ with $\phi$ held at $1$ — it is a statement about a discrepancy, not a separate parameter. The **scale factor** is what software multiplies the covariance matrix by to repair that discrepancy, and the quasi-likelihood estimate of it is the **Pearson $\chi^{2}/\mathrm{df}$**, $\frac{1}{n-p}\sum_i(y_i-\hat\mu_i)^{2}/V(\hat\mu_i)$. All four are the same quantity approached from four directions, and the practical content is that three of them are printed by default while the one that matters — whether $\phi=1$ was true — is not tested by default. A Poisson fit reporting no dispersion estimate is not reporting that the dispersion is one; it is reporting that it was told to assume so.

## The Variance Function Is an Assumption Rather Than an Estimate, So Overdispersion Divides Every Standard Error by a Number the Fit Never Prints

Counts in finance are almost never Poisson. Trades cluster, rejections cluster, defaults cluster, and a scan of nineteen calendar effects on overlapping data produces a count of rejections whose mean is right and whose spread is larger than the binomial's. The mean model can be perfectly specified while the variance is several times what the family asserts, and the question is what the fit does about it:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(13033)
n, reps = 400, 4000
b0 = 2.0                                  # mean count about 7.4, slope truly zero


def fit_poisson(X, y, tol=1e-11):
    b = np.zeros(X.shape[1])
    for _ in range(60):
        mu = np.exp(X @ b)
        WX = X * mu[:, None]
        nb = np.linalg.solve(X.T @ WX, WX.T @ (X @ b + (y - mu) / mu))
        if np.max(np.abs(nb - b)) < tol:
            b = nb
            break
        b = nb
    return b, np.exp(X @ b)


print("  Poisson regression on counts whose variance is phi times their mean;")
print("  the slope is truly zero, so every rejection below is a false one")
print("    phi   nominal se   empirical sd    ratio   Pearson/df   Wald   quasi")
for phi in (1.0, 2.0, 4.0, 8.0):
    b1 = np.empty(reps)
    se = np.empty(reps)
    disp = np.empty(reps)
    for r in range(reps):
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        mu = np.exp(b0 + 0.0 * x)
        if phi == 1.0:
            y = rng.poisson(mu, n).astype(float)
        else:
            y = rng.negative_binomial(mu / (phi - 1.0), 1.0 / phi, n).astype(float)
        b, mh = fit_poisson(X, y)
        cov = np.linalg.inv(X.T @ (X * mh[:, None]))
        b1[r] = b[1]
        se[r] = np.sqrt(cov[1, 1])
        disp[r] = ((y - mh) ** 2 / mh).sum() / (n - 2)
    z = b1 / se
    zc = stats.norm.isf(0.025)
    quasi = np.abs(b1 / (se * np.sqrt(disp))) > stats.t.isf(0.025, n - 2)
    print(f"    {phi:3.0f}   {se.mean():10.4f}   {b1.std():12.4f}"
          f"   {b1.std() / se.mean():6.2f}   {disp.mean():10.2f}"
          f"   {(np.abs(z) > zc).mean():6.4f}  {quasi.mean():6.4f}")
# =>   Poisson regression on counts whose variance is phi times their mean;
#      the slope is truly zero, so every rejection below is a false one
#        phi   nominal se   empirical sd    ratio   Pearson/df   Wald   quasi
#          1       0.0185         0.0183     0.99         1.00   0.0467  0.0485
#          2       0.0185         0.0263     1.43         2.00   0.1660  0.0503
#          4       0.0185         0.0362     1.96         3.99   0.3140  0.0435
#          8       0.0185         0.0516     2.79         7.95   0.4750  0.0505
```

The first column is the finding and it is a column of one number. The nominal standard error is $0.0185$ in every row — identical to four decimals whether the counts are Poisson or eight times as variable — because $(X^\top WX)^{-1}$ depends on the fitted means and the fitted means are unaffected. The estimator does not react to overdispersion at all, and neither does anything it prints.

The true sampling spread meanwhile runs $0.0183$, $0.0263$, $0.0362$, $0.0516$, so the ratio of truth to claim reaches $2.79$ at $\phi=8$, close to the predicted $\sqrt{8}=2.83$. The consequence is the Wald column: a nominal $5\%$ test of a slope that is *zero by construction* rejects $4.67\%$ of the time when the family is right and $47.50\%$ of the time when the variance is eight times the mean. Nearly half of all findings at $\phi=8$ are false, and the p-values producing them are computed correctly from a standard error that is correctly derived from a likelihood that is wrong about one thing.

The repair is one line and it is measured in the last column. Estimating $\phi$ by Pearson $\chi^{2}/\mathrm{df}$ — which reads $1.00$, $2.00$, $3.99$, $7.95$, recovering the true dispersion in every row — and inflating the standard errors by its square root restores sizes of $0.0485$, $0.0503$, $0.0435$, $0.0505$. The information needed was in the residuals the whole time. It was simply never consulted, because the family had already declared the answer.

This is what the course's rejection count is exposed to. Nineteen calendar tests on overlapping daily data are not nineteen independent Bernoulli trials; the mean count of false positives under the null is still $19\times0.05=0.95$, which is why "almost exactly the one false positive nineteen null tests owe" is a sound reading, but the *variance* of that count exceeds the binomial's, so a scan returning three rejections is less surprising than a binomial calculation would make it. Getting the mean right and the variance wrong is the characteristic error of this whole class, and it is the same error in a count of rejections as in a Poisson regression.

**Overdispersion is not a violated assumption that degrades an estimate; it is a correct estimate reported with an error bar computed under a variance nobody measured.**

## Deviance Generalizes the Residual Sum of Squares, and Its $\chi^{2}$ Calibration Is Asymptotic in the Number of Trials Rather Than the Number of Rows

The **deviance** is twice the log-likelihood gap between the fitted model and the saturated model that assigns each observation its own mean, $D=2\{\ell(\text{saturated})-\ell(\hat\beta)\}$. For the Gaussian it reduces exactly to $\mathrm{RSS}/\sigma^{2}$, which is why it is the natural generalization of the quantity the $F$ test of the previous page was built from, and differences of deviances between nested GLMs are the likelihood-ratio statistics of [Likelihood Ratio Tests](../part-12-hypothesis-testing/06-likelihood-ratio-tests.md), asymptotically $\chi^{2}_q$ under the smaller model.

The dangerous part is the second use, in which the *residual* deviance is compared to $\chi^{2}_{n-p}$ as an absolute goodness-of-fit test. That comparison needs each observation's contribution to be well approximated by a $\chi^{2}_1$, and the approximation improves with information *per row*, not with the number of rows:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(13035)
n, reps, true = 200, 3000, np.array([0.20, 0.40])


def fit_logit(X, y, m, tol=1e-11):
    b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        mu, w = m * p, m * p * (1.0 - p)
        WX = X * w[:, None]
        nb = np.linalg.solve(X.T @ WX, WX.T @ (X @ b + (y - mu) / w))
        if np.max(np.abs(nb - b)) < tol:
            b = nb
            break
        b = nb
    return 1.0 / (1.0 + np.exp(-(X @ b)))


def deviance(y, p, m):
    a = np.where(y > 0, y * np.log(np.where(y > 0, y / (m * p), 1.0)), 0.0)
    r = m - y
    c = np.where(r > 0, r * np.log(np.where(r > 0, r / (m * (1 - p)), 1.0)), 0.0)
    return 2.0 * (a + c).sum()


print("  residual deviance against its chi-square(n-p) reference, n = 200 rows")
print("  the model is correct in both rows; only the trials per row differ")
print("    trials/row   mean dev   E = n-p   sd dev   sqrt(2(n-p))   P(dev > 95th)")
for m in (30, 1):
    d = np.empty(reps)
    for r in range(reps):
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        p = 1.0 / (1.0 + np.exp(-(X @ true)))
        y = rng.binomial(m, p).astype(float)
        d[r] = deviance(y, fit_logit(X, y, float(m)), float(m))
    cut = stats.chi2.isf(0.05, n - 2)
    print(f"    {m:10d}   {d.mean():8.1f}   {n - 2:7d}   {d.std():6.1f}"
          f"   {np.sqrt(2 * (n - 2)):12.1f}   {(d > cut).mean():13.4f}")
# =>   residual deviance against its chi-square(n-p) reference, n = 200 rows
#      the model is correct in both rows; only the trials per row differ
#        trials/row   mean dev   E = n-p   sd dev   sqrt(2(n-p))   P(dev > 95th)
#                30      201.2       198     19.8           19.9          0.0660
#                 1      265.7       198      6.4           19.9          1.0000
```

With thirty trials per row the reference is sound: mean deviance $201.2$ against an expected $198$, standard deviation $19.8$ against a predicted $19.9$, and a goodness-of-fit test rejecting a correct model $6.60\%$ of the time — near enough to nominal to be usable.

With one trial per row every one of those agreements fails, on a model that is correct by construction and fitted from the same number of rows. The mean deviance is $265.7$ where $\chi^{2}_{198}$ predicts $198$; the standard deviation is $6.4$ where it predicts $19.9$, three times too narrow; and the test rejects the true model in $1.0000$ of $3{,}000$ replications. It is not that the Bernoulli deviance is a poor approximation to a $\chi^{2}$ — it is not approximately anything, because with $y_i\in\{0,1\}$ the saturated model fits each observation exactly and the deviance becomes a deterministic function of the fitted probabilities alone, carrying almost no information about fit. Increasing $n$ makes this worse rather than better, since each new row adds another non-$\chi^{2}$ term.

The general rule is that the residual deviance is a valid absolute goodness-of-fit statistic only when the data are grouped into cells with substantial counts. *Differences* of deviances remain valid in both regimes, because the likelihood-ratio asymptotics run in $n$ rather than in the per-row information, which is why model comparison survives where model checking does not.

!!! warning "An overdispersed fit converges, returns correct coefficients, and prints a complete diagnostic table in which nothing is wrong"
    There is no signature. The IRLS loop terminates in the usual handful of iterations because concavity is untouched by the variance being misstated; the score residual is orthogonal to machine precision; the coefficients are consistent for the right values; the standard errors are correctly derived from the assumed likelihood; and the p-values are correctly derived from those standard errors. The only defective quantity is $\phi$, which was never estimated because the family fixed it at $1$, and $\phi$ appears in no column of the output. Above, the printed standard error is $0.0185$ whether the truth is $0.0183$ or $0.0516$, and a slope of exactly zero is declared significant $47.50\%$ of the time. Reaching for the residual deviance as a check does not help and can actively mislead, since on ungrouped binary data it rejects a correct model $1.0000$ of the time. **The free diagnostic is the Pearson statistic the fit already has the pieces for: compute $\frac{1}{n-p}\sum_i(y_i-\hat\mu_i)^{2}/V(\hat\mu_i)$, which read $1.00$, $2.00$, $3.99$ and $7.95$ against true dispersions of one, two, four and eight, and if it is not near one then multiply every standard error by its square root before reading a single p-value — one line, no extra data, and it restored sizes of $0.0485$ to $0.0505$ across the whole table above.**

## Choosing a Family Is Choosing a Variance, and That Choice Is Made Before the Data Arrive and Is Never Revisited

This page established that a generalized linear model is a family, a linear predictor and a link, with $b'(\theta)=\mu$ and $b''(\theta)=\operatorname{var}(y)/\phi$ making the variance a deterministic function $\phi V(\mu)$ of the mean once the family is named; that the canonical link reduces the score to $X^\top(y-\mu)=0$ and makes the observed and expected Hessians identical, so one fourteen-line routine fitted Gaussian, Poisson and binomial responses to score residuals of $1.84\times10^{-12}$, $1.75\times10^{-12}$ and $3.44\times10^{-12}$ in $2$, $9$ and $6$ iterations, with the Gaussian case reproducing `lstsq` to $9.99\times10^{-16}$; that strict concavity makes the maximum unique and the starting value irrelevant; that overdispersion leaves the nominal standard error frozen at $0.0185$ while the true spread reaches $0.0516$, driving a nominal $5\%$ test of a zero slope to a size of $0.4750$ that the Pearson scale repairs to $0.0505$; and that the residual deviance is calibrated at thirty trials per row, reading $201.2$ against $198$ and rejecting at $0.0660$, and destroyed at one trial per row, reading $265.7$ with a standard deviation of $6.4$ and rejecting a correct model $1.0000$ of the time.

The through-line is a division of labour that the output does not display. A GLM asks the analyst for two things and estimates only one of them. The mean structure — which columns, which link — is fitted, checked, and reported with error bars. The variance structure is asserted by naming the family, is used in every weight, every standard error and every p-value, and is then never mentioned again. When it is right, the machinery is superb: unique optimum, quadratic convergence, closed-form information matrix, exact orthogonality. When it is wrong, all of that survives intact and only the inference is worthless, which is the least detectable arrangement of the two possible ones.

There is one member of this class the course actually trades, and it is the one where the variance function cannot be wrong — a Bernoulli response has variance $\mu(1-\mu)$ by arithmetic rather than by assumption, since a two-valued random variable's mean determines its distribution. That removes the failure of section 4 and introduces two others in its place, one about a maximum that fails to exist and one about a probability that is off by a constant. That is [Logistic Regression](04-logistic-regression.md).

**A generalized linear model estimates a mean and assumes a variance, prints diagnostics for the first and none for the second, and the second is what every standard error on the page is divided by.**
