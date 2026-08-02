# Conditional Gaussian Distributions

Condition a joint normal on part of itself and the answer is normal, with a mean that moves linearly in the observation and a covariance that does not move at all. The first half of that sentence is why the assumption is everywhere: regression, hedging, filtering and factor attribution are all one formula, and it is this one. The second half is a claim about crises, smuggled in as an algebraic convenience, and it is false in the direction that makes every hedge look more reliable than it is.

This page covers the partitioned normal and its two conditioning formulas, the proof that the conditional mean is affine and therefore that the best predictor and the best *linear* predictor coincide, the Schur complement read as the variance a regression removes, the zeros in the precision matrix that are conditional independences, the equivalence of conditioning sequentially and conditioning at once, and what a covariance that ignores its own conditioning value costs. It does not construct the joint normal, which is [Multivariate Gaussian Distribution](05-multivariate-gaussian.md); it does not define $\mathbb{E}[X\mid Y]$ or prove the best-predictor theorem, which are [Conditional Expectation](../part-04-expectation-and-moments/06-conditional-expectation.md); it does not build conditional densities in general, which is [Conditional Distributions](../part-03-random-variables/07-conditional-distributions.md); and it estimates none of the coefficients it writes down, which is [Part XIII](../part-13-regression/index.md) and, one observation at a time, [Particle and Kalman Filters](../../advanced/02-particle-and-kalman-filters.md).

The trading stake is a correlation measured twice on the same nine assets. [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) sorts every day by the market's return and finds an average pairwise sector correlation of $0.327$ in the middle $80\%$ of days against $0.516$ in the worst decile — $2.49$ effective bets falling to $1.76$, with no crisis label required. Under a joint normal that number cannot rise. The sixth section proves it must fall, computes by how much, and finds the model does not merely understate the crisis; it has the sign backwards.

## Partition the Vector and the Answer Is Two Formulas

Split a normal vector into a part to be forecast and a part to be observed, $(X,Y)$ jointly normal with

$$\begin{pmatrix}X\\Y\end{pmatrix}\sim\mathcal{N}\!\left(\begin{pmatrix}\mu_X\\\mu_Y\end{pmatrix},\ \begin{pmatrix}\Sigma_{XX}&\Sigma_{XY}\\\Sigma_{YX}&\Sigma_{YY}\end{pmatrix}\right).$$

Then the conditional law of $X$ given $Y=y$ is normal, with

$$\mu_{X\mid Y=y}=\mu_X+\Sigma_{XY}\Sigma_{YY}^{-1}(y-\mu_Y)\qquad\text{and}\qquad\Sigma_{X\mid Y}=\Sigma_{XX}-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}.$$

Read the two formulas side by side and the asymmetry is immediate. The mean carries $y$ and the covariance does not — no $y$ appears anywhere on the right of the second formula, so the conditional covariance is one matrix for every possible observation.

??? note "Proof that a partitioned normal conditions to a normal, with an affine mean and a covariance free of y"
    Define the residual after regressing $X$ on $Y$,

    $$W=X-\Sigma_{XY}\Sigma_{YY}^{-1}Y,$$

    which is a linear map of $(X,Y)$ and therefore jointly normal with $Y$, by the linear-combination definition of [Multivariate Gaussian Distribution](05-multivariate-gaussian.md). Its covariance with $Y$ is

    $$\mathrm{cov}(W,Y)=\Sigma_{XY}-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YY}=\Sigma_{XY}-\Sigma_{XY}=0,$$

    so $W$ and $Y$ are uncorrelated. Because they are *jointly normal*, the proof that uncorrelated jointly normal coordinates are independent upgrades that to full independence, and everything else is bookkeeping.

    Write $X=W+\Sigma_{XY}\Sigma_{YY}^{-1}Y$. Conditioning on $Y=y$ leaves the law of $W$ untouched, by independence, and replaces the second term by the constant $\Sigma_{XY}\Sigma_{YY}^{-1}y$. So the conditional law of $X$ is the law of $W$ shifted by a constant: normal, with mean $\mathbb{E}[W]+\Sigma_{XY}\Sigma_{YY}^{-1}y=\mu_X+\Sigma_{XY}\Sigma_{YY}^{-1}(y-\mu_Y)$ and covariance $\mathrm{cov}(W)$, which the sandwich formula of [Linear Transformations](04-linear-transformations.md) evaluates as $\Sigma_{XX}-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}$.

    Exactly one step is not bookkeeping, and it is "uncorrelated, therefore independent". That implication holds under joint normality and essentially nowhere else. Without it, $W$ is merely uncorrelated with $Y$: its conditional law may depend on $y$, so $\Sigma_{X\mid Y}$ stops being constant and $\mathbb{E}[X\mid Y]$ need not be affine. Everything distinctive on this page — the linear update, the constant residual risk, the gain computable in advance — rests on that single implication, and [Conditional Expectation](../part-04-expectation-and-moments/06-conditional-expectation.md) exhibits the general case where it fails.

```python
import numpy as np

rng = np.random.default_rng(691)
n, T = 4, 4_000_000
A = np.array([[1.0, 0.0, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0],
              [-0.3, 0.4, 0.87, 0.0], [0.5, 0.5, 0.5, 0.5]])
Sigma = A @ A.T
mu = np.array([0.4, -0.2, 0.1, 0.3])
X = mu + rng.standard_normal((T, n)) @ np.linalg.cholesky(Sigma).T
K = Sigma[:3, 3:] @ np.linalg.inv(Sigma[3:, 3:])               # the regression coefficient
Scond = Sigma[:3, :3] - K @ Sigma[3:, :3]
sy = np.sqrt(Sigma[3, 3])
print(f"  conditioning X = (X1,X2,X3) on Y = X4, in slabs of half-width 0.02 sd(Y)")
print("      y/sd(Y)     conditional mean, formula        in the slab"
      "            conditional sd, formula        in the slab")
for k in (-2.0, 0.0, 2.0):
    y = mu[3] + k * sy
    S = X[np.abs(X[:, 3] - y) < 0.02 * sy, :3]
    f = mu[:3] + K @ np.array([y - mu[3]])
    print(f"  {k:+9.1f}   {f[0]:8.4f} {f[1]:8.4f} {f[2]:8.4f}   {S[:, 0].mean():8.4f}"
          f" {S[:, 1].mean():8.4f} {S[:, 2].mean():8.4f}   {np.sqrt(Scond[0, 0]):8.4f}"
          f" {np.sqrt(Scond[1, 1]):8.4f} {np.sqrt(Scond[2, 2]):8.4f}   {S[:, 0].std():8.4f}"
          f" {S[:, 1].std():8.4f} {S[:, 2].std():8.4f}")
u, c = np.sqrt(np.diag(Sigma[:3, :3])), np.diag(Scond) / np.diag(Sigma[:3, :3])
print(f"  unconditional sd  {u[0]:8.4f} {u[1]:8.4f} {u[2]:8.4f}"
      f"      R-squared  {1 - c[0]:8.4f} {1 - c[1]:8.4f} {1 - c[2]:8.4f}")
# =>   conditioning X = (X1,X2,X3) on Y = X4, in slabs of half-width 0.02 sd(Y)
#          y/sd(Y)     conditional mean, formula        in the slab            conditional sd, formula        in the slab
#           -2.0    -0.6000  -1.6000  -0.8700    -0.6129  -1.5880  -0.8632     0.8660   0.7141   0.8785     0.8713   0.7210   0.8641
#           +0.0     0.4000  -0.2000   0.1000     0.4028  -0.1998   0.0934     0.8660   0.7141   0.8785     0.8662   0.7123   0.8799
#           +2.0     1.4000   1.2000   1.0700     1.4099   1.2016   1.0755     0.8660   0.7141   0.8785     0.8649   0.7189   0.8844
#      unconditional sd    1.0000   1.0000   1.0034      R-squared    0.2500   0.4900   0.2336
```

The formulas are checked by brute force rather than assumed: four million draws, sliced into thin slabs of half-width $0.02$ standard deviations around three conditioning values, with the empirical mean and standard deviation inside each slab compared against the closed forms. Every conditional mean matches to about three decimals, and the second coordinate's conditional mean sweeps from $-1.588$ at $y=-2\sigma$ to $+1.202$ at $y=+2\sigma$ — a range of $2.79$, against an unconditional standard deviation of $1.0000$.

The right-hand block is the claim that matters. The conditional standard deviations from the formula are $0.8660$, $0.7141$ and $0.8785$, and they are printed identically on all three rows because the formula contains no $y$. The measured values inside the slabs are $0.8713/0.7210/0.8641$, then $0.8662/0.7123/0.8799$, then $0.8649/0.7189/0.8844$ — agreeing with the formula and with each other to within sampling error, at conditioning values four standard deviations apart. Whatever $Y$ turned out to be, the uncertainty left over is the same.

The last line prices the update. Observing $Y$ removes $25.0\%$, $49.0\%$ and $23.4\%$ of the three variances, so the forecast is worth something and the residual risk is substantial. Both facts are consistent with the conditional covariance being constant, because $R^{2}$ is a property of the joint law and not of the realised observation.

## The Conditional Mean Is Affine, So the Best Predictor Is Linear

[Conditional Expectation](../part-04-expectation-and-moments/06-conditional-expectation.md) proves that $\mathbb{E}[X\mid Y]$ is the best predictor of $X$ from $Y$ under squared error — not the best linear one, the best among all measurable functions. Separately, ordinary least squares finds the best predictor among *affine* functions of $Y$. In general these are two different objects, and regression estimates the affine approximation to something that need not be affine.

Under joint normality they are the same object. The first proof shows $\mathbb{E}[X\mid Y]=\mu_X+\Sigma_{XY}\Sigma_{YY}^{-1}(Y-\mu_Y)$, which is affine in $Y$, so the unrestricted optimum lies inside the restricted class and the two optimisations return the same answer. That is the promise this page was written to discharge, and it is worth stating in its consequential form: under joint normality, *linear regression is not an approximation to anything*. It is the conditional expectation, exactly, and there is no nonlinear relationship left over for a more flexible model to find.

The converse is what makes the result load-bearing rather than reassuring. If the joint law is not normal, the conditional mean is generally curved, and a regression is then estimating the best affine fit to a curve — a perfectly well-defined object that is simply not the best forecast. The residual from such a regression is uncorrelated with $Y$ by construction and is *not* independent of it, and every diagnostic that checks orthogonality rather than independence will pass anyway.

## The Schur Complement Is the Variance a Regression Removes

The matrix $\Sigma_{X\mid Y}=\Sigma_{XX}-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}$ is the Schur complement of $\Sigma_{YY}$ in $\Sigma$, and its financial reading is direct: it is what is left of the risk in $X$ once everything explainable by $Y$ has been hedged out. In the scalar case with correlation $\rho$ it collapses to

$$\mathrm{var}(X\mid Y)=\sigma_X^{2}\big(1-\rho^{2}\big),$$

which is where $R^{2}=\rho^{2}$ comes from, and where the familiar fact that a correlation of $0.5$ removes only a quarter of the variance becomes visible as arithmetic rather than folklore.

Because the conditional covariance is constant, the law of total variance of [Law of Total Variance](../part-04-expectation-and-moments/08-law-of-total-variance.md) simplifies unusually far. Its within-group term $\mathbb{E}[\mathrm{var}(X\mid Y)]$ is an average of a constant, so

$$\mathrm{var}(X)=\underbrace{\Sigma_{X\mid Y}}_{\text{constant}}+\underbrace{\mathrm{cov}\big(\mathbb{E}[X\mid Y]\big)}_{\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}},$$

a clean two-term split with no averaging left to do. The decomposition has no residual heteroskedasticity by construction — a Gaussian model cannot produce a forecast whose error is larger when the forecast is extreme, because it has no mechanism by which the error could depend on the forecast at all. That is a strong empirical claim about markets and it is made silently.

## Zeros in the Precision Matrix Are Conditional Independences

The inverse covariance matrix $K=\Sigma^{-1}$ — the *precision* matrix — answers a different question from $\Sigma$, and the difference is exactly the marginal-versus-conditional distinction this part keeps returning to.

??? note "Proof that a zero in the inverse covariance matrix is a conditional independence"
    Take $X\sim\mathcal{N}(0,\Sigma)$ with $K=\Sigma^{-1}$, so the density is proportional to $\exp(-\tfrac12x^\top Kx)$. Expand the quadratic form and collect the terms involving the pair $(x_i,x_j)$:

    $$x^\top Kx=K_{ii}x_i^{2}+K_{jj}x_j^{2}+2K_{ij}x_ix_j+2x_i\!\!\sum_{k\neq i,j}\!\!K_{ik}x_k+2x_j\!\!\sum_{k\neq i,j}\!\!K_{jk}x_k+(\text{terms free of }x_i,x_j).$$

    Conditioning on all the other coordinates fixes every $x_k$ with $k\neq i,j$, so the conditional density of $(X_i,X_j)$ is proportional to the exponential of the displayed expression with those $x_k$ treated as constants. The only term coupling $x_i$ to $x_j$ is $2K_{ij}x_ix_j$. When $K_{ij}=0$ the exponent separates into a function of $x_i$ plus a function of $x_j$, the conditional density factorizes, and $X_i$ and $X_j$ are conditionally independent given the rest.

    Joint normality is the hypothesis again, and the sharper point is an asymmetry that is easy to state and easy to misuse. $\Sigma_{ij}=0$ says the pair is *marginally* uncorrelated; $K_{ij}=0$ says it is *conditionally* independent given everything else. Neither implies the other: a three-variable chain has all three covariances non-zero and one precision entry zero, while a common-cause structure can produce the reverse. A risk report that quotes correlations is quoting $\Sigma$; a factor attribution that regresses each sleeve on all the others is quoting $K$. They can disagree about the same pair, at length, with neither being wrong, because they were asked different questions.

```python
import numpy as np

rng = np.random.default_rng(701)
T, a, b = 2_000_000, 0.8, 0.7


def chain(direct):                                             # X1 -> X2 -> X3, plus a shortcut
    e = rng.standard_normal((T, 3))
    x1 = e[:, 0]
    x2 = a * x1 + e[:, 1]
    x3 = b * x2 + direct * x1 + e[:, 2]
    return np.column_stack([x1, x2, x3])


for name, direct in (("pure chain", 0.0), ("with a shortcut", 0.6)):
    X = chain(direct)
    C = np.corrcoef(X, rowvar=False)
    K = np.linalg.inv(np.cov(X, rowvar=False))
    partial = -K[0, 2] / np.sqrt(K[0, 0] * K[2, 2])
    print(f"  {name}:  X3 loads {direct} directly on X1")
    print(f"    Sigma has no zero entry: corr(X1,X2) {C[0, 1]:+.4f}"
          f"  corr(X2,X3) {C[1, 2]:+.4f}  corr(X1,X3) {C[0, 2]:+.4f}")
    print(f"    K = Sigma inverse:  K13 {K[0, 2]:+.6f}"
          f"     partial corr(X1,X3 | X2) = -K13 / sqrt(K11 K33) {partial:+.6f}")
# =>   pure chain:  X3 loads 0.0 directly on X1
#        Sigma has no zero entry: corr(X1,X2) +0.6239  corr(X2,X3) +0.6669  corr(X1,X3) +0.4165
#        K = Sigma inverse:  K13 -0.000943     partial corr(X1,X3 | X2) = -K13 / sqrt(K11 K33) +0.000737
#      with a shortcut:  X3 loads 0.6 directly on X1
#        Sigma has no zero entry: corr(X1,X2) +0.6237  corr(X2,X3) +0.7543  corr(X1,X3) +0.6882
#        K = Sigma inverse:  K13 -0.599489     partial corr(X1,X3 | X2) = -K13 / sqrt(K11 K33) +0.424335
```

The pure chain is the case where the two matrices disagree completely. $X_1$ drives $X_2$ drives $X_3$, with no direct link at all, and the correlation matrix has no zero anywhere: $X_1$ and $X_3$ correlate at $+0.4165$, which is a real, reproducible, economically meaningful association. The precision matrix reports $K_{13}=-0.000943$, and the partial correlation of $X_1$ and $X_3$ given $X_2$ is $+0.000737$ — zero to the precision two million draws supply.

Both numbers are correct. Asked "do $X_1$ and $X_3$ move together", the answer is yes, at $0.42$. Asked "does $X_1$ tell you anything about $X_3$ that $X_2$ has not already told you", the answer is no, exactly. The control is the second panel, where a direct loading of $0.6$ is added: the marginal correlation rises to $+0.6882$ and the partial correlation becomes $+0.4243$, with $K_{13}=-0.599$ recovering the injected loading almost exactly. The diagnostic is sharp in both directions — it does not merely fail to detect a link that is absent, it measures one that is present.

## Conditioning Twice Is Conditioning Once

The final structural property is the one that makes filtering possible: observations can be absorbed one at a time, in any order, and the answer does not depend on the schedule.

??? note "Proof that conditioning on two observations in sequence gives the same law as conditioning on both at once"
    Work with precisions. Let the prior be $X\sim\mathcal{N}(\mu_0,\Sigma_0)$ with $K_0=\Sigma_0^{-1}$, and let two observations be $y_i=H_iX+\varepsilon_i$ with $\varepsilon_i\sim\mathcal{N}(0,R_i)$, independent of $X$ and of each other. The joint density of $(X,y_1,y_2)$ factorizes as the prior times the two observation densities, so the log posterior is

    $$-\tfrac12(x-\mu_0)^\top K_0(x-\mu_0)-\tfrac12\sum_{i=1}^{2}(y_i-H_ix)^\top R_i^{-1}(y_i-H_ix)+\text{const},$$

    a quadratic in $x$. Collecting the second-order terms gives the posterior precision

    $$K_{\text{post}}=K_0+H_1^\top R_1^{-1}H_1+H_2^\top R_2^{-1}H_2,$$

    which is a sum, hence symmetric in the two observations and identical whether they are added together or one after the other. Absorbing $y_1$ first produces the intermediate precision $K_0+H_1^\top R_1^{-1}H_1$, and treating that as the prior for $y_2$ adds the second term — the same total. The mean follows the same pattern, $K_{\text{post}}\mu_{\text{post}}=K_0\mu_0+\sum_iH_i^\top R_i^{-1}y_i$.

    The load-bearing hypothesis is that the two observations are conditionally independent given the state, which is what makes their joint noise covariance block diagonal and lets the precisions add. Without it the cross-terms do not vanish, sequential updating counts the shared component of the two observations twice, and the posterior comes out overconfident — narrower than the evidence supports. That is the same defect a backtest has when it treats overlapping windows as independent observations, and it has the same signature: the point estimate is fine and the uncertainty around it is wrong, in the direction nobody checks.

That result is what [Conditional Distributions](../part-03-random-variables/07-conditional-distributions.md) is pointing at when it says that with Gaussian conditionals "the algebra closes in one line and the answer stays Gaussian". Closure is the whole of it. Every other family in this book leaves the conditional in a different family from the prior, so a second observation requires a fresh derivation; here the posterior is again normal with the same two parameters, so the same three lines of code run forever.

!!! note "The Kalman gain can be computed before any data arrives, because the conditional covariance does not depend on what is observed"
    Everything on the covariance side of a filter — the posterior covariance, the gain $\Sigma_{XY}\Sigma_{YY}^{-1}$, the steady state the recursion converges to — is a function of $\Sigma_0$, $H$ and $R$ alone. None of it involves $y$. So the entire error analysis of a linear filter can be run offline, before a single observation exists: how fast uncertainty falls, what it converges to, how many observations are needed to reach a target, whether the filter is even identifiable. [Particle and Kalman Filters](../../advanced/02-particle-and-kalman-filters.md) uses this constantly and it is genuinely remarkable — a forecasting system whose accuracy is known in advance of the data. It is also, read from the other side, the subject of the next section.

## The Conditional Covariance Does Not Know What Was Observed

The property just admired is an indictment when the conditioning variable is a market that has fallen. A hedge ratio $\Sigma_{XY}\Sigma_{YY}^{-1}$, a residual risk $\Sigma_{X\mid Y}$ and a filter gain are all computable before any data arrives — which is to say they are the *same numbers* on a calm day and on the worst day in the sample. The model contains no mechanism by which they could differ.

??? note "Proof that conditioning on a market bucket strictly lowers every pairwise correlation"
    Let $X$ be $N$ equicorrelated standard normals with common correlation $\rho$, and let $m=\tfrac1N\mathbf{1}^\top X$ be the equal-weight market. From the equicorrelation spectrum of [Correlation Matrices](03-correlation-matrices.md), $C\mathbf{1}=\lambda_1\mathbf{1}$ with $\lambda_1=1+(N-1)\rho$, so $\mathrm{cov}(X,m)=C\mathbf{1}/N=\lambda_1\mathbf{1}/N$ and $\mathrm{var}(m)=\lambda_1/N$. The regression coefficient of $X$ on $m$ is therefore exactly $\mathbf{1}$.

    Write $X=\mathbf{1}m+W$ with $W=X-\mathbf{1}m$. Then $\mathrm{cov}(W,m)=\mathrm{cov}(X,m)-\mathbf{1}\mathrm{var}(m)=0$, and by joint normality $W$ is *independent* of $m$ — page 05's implication again. Now condition on any event $\{m\in A\}$. Independence leaves the law of $W$ alone, so

    $$\mathrm{cov}\big(X\mid m\in A\big)=\mathrm{cov}(W)+\mathrm{var}(m\mid m\in A)\,\mathbf{1}\mathbf{1}^\top=C-s\,\mathbf{1}\mathbf{1}^\top,\qquad s=\mathrm{var}(m)-\mathrm{var}(m\mid m\in A).$$

    The diagonal is $1-s$ and every off-diagonal is $\rho-s$, so the conditional correlation is $(\rho-s)/(1-s)$. Since $(\rho-s)/(1-s)<\rho$ exactly when $\rho<1$ and $s>0$, and since selecting a tail bucket always reduces the variance of $m$ within it, conditioning on a bad market *lowers* the average pairwise correlation — for every $\rho<1$, at every $N$, in every bucket.

    The mechanism is worth naming because it is not a quirk of the equicorrelated case. Selecting on the market truncates the common factor and leaves the idiosyncratic part untouched, so the bucket contains less common variation than the full sample and the assets look *more* independent inside it, not less. Any elliptical model does something similar, because in all of them the conditioning value enters the mean and not the covariance. Producing the opposite sign requires dependence that is asymmetric between tails and centre, which no covariance matrix can express and which is what [Copulas](../part-18-quant-finance-applications/15-copulas.md) is for.

```python
import numpy as np

rng = np.random.default_rng(709)
N, rho, T, nu = 9, 0.619, 6_300, 4                             # published full-sample rho
C = (1 - rho) * np.eye(N) + rho
vm = np.ones(N) @ C @ np.ones(N) / N ** 2                      # variance of the equal-weight market
pub = {"worst 10%": (0.516, 1.76), "middle 80%": (0.327, 2.49), "all days": (0.619, 1.51)}


def stats(X):
    r = np.corrcoef(X, rowvar=False)[~np.eye(N, dtype=bool)].mean()
    return r, N / (1 + (N - 1) * r)


L = np.linalg.cholesky(C)
print(f"  nine equicorrelated sectors at rho = {rho}, {T} days, market = equal-weight average")
print("             law        bucket    var(m|b)/var(m)   closed form   simulated   N_eff   published")
for name, gauss, X in (("gaussian", True, rng.standard_normal((T, N)) @ L.T),
                       (f"multivariate t{nu}", False, rng.standard_normal((T, N))
                        / np.sqrt(rng.chisquare(nu, (T, 1)) / nu) @ L.T)):
    m = X.mean(axis=1)
    lo, hi = np.quantile(m, 0.10), np.quantile(m, 0.90)
    for b, sel in (("worst 10%", m <= lo), ("middle 80%", (m > lo) & (m < hi)),
                   ("all days", np.ones(T, bool))):
        v = m[sel].var()
        s = vm - v
        r, ne = stats(X[sel])
        cf = f"{(rho - s) / (1 - s):13.4f}" if gauss else f"{'--':>13s}"
        print(f"  {name:>16s} {b:>12s} {v / vm:17.4f} {cf}"
              f" {r:11.4f} {ne:7.3f} {pub[b][0]:11.3f}")
# =>   nine equicorrelated sectors at rho = 0.619, 6300 days, market = equal-weight average
#                 law        bucket    var(m|b)/var(m)   closed form   simulated   N_eff   published
#              gaussian    worst 10%            0.1623        0.1457      0.1544   4.027       0.516
#              gaussian   middle 80%            0.4313        0.3893      0.3876   2.195       0.327
#              gaussian     all days            0.9884        0.6160      0.6144   1.521       0.619
#       multivariate t4    worst 10%            1.1959            --      0.3415   2.412       0.516
#       multivariate t4   middle 80%            0.5663            --      0.3542   2.348       0.327
#       multivariate t4     all days            1.8743            --      0.6116   1.527       0.619
```

Start with the Gaussian panel, calibrated so its full-sample correlation is the published $0.619$ — which the "all days" row confirms at $0.6144$, with the closed form giving $0.6160$ and the market-variance ratio at $0.9884$ rather than exactly one because these are sample quantities. The closed form and the simulation agree in all three buckets, to within $0.009$, so the proof is doing what it claims.

Now read the Gaussian column downward against the published column beside it. The model says the average pairwise correlation is $0.614$ over all days, $0.388$ in the middle $80\%$, and $0.154$ in the worst decile: monotonically *falling* as conditions worsen, exactly as the proof requires, driven by the market variance inside the bucket collapsing to $16.2\%$ of its unconditional value. The measurement says $0.619$, then $0.327$, then $0.516$ — falling into the middle bucket and then *rising* in the tail. The model and the data do not disagree about a magnitude. They disagree about which way the effect goes, and the model's disagreement is a theorem rather than a calibration failure.

The $t_4$ panel is the smallest available change of assumption, and it is instructive in how little it buys. Replacing the Gaussian with an elliptical heavy-tailed law flattens the profile — $0.3415$ in the worst decile against $0.3542$ in the middle, essentially no conditional effect at all — so the decline is removed but the increase is not produced. That is what the proof's closing paragraph predicts: every elliptical family puts the conditioning value in the mean and not the covariance, so none of them can manufacture dependence that is stronger in one tail than in the centre. The published $0.516$ is out of reach of the whole family, not just of its Gaussian member.

!!! warning "A hedge ratio derived from a joint normal is the same number on a calm day and on the worst day of the sample, because the model contains no mechanism by which it could differ"
    [Drawdowns, Tail Risk, and Stress Testing](../../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md) measures the size of what is being assumed away: re-estimating betas inside stressed periods moves $\beta(\text{TLT},\text{SPY})$ from $-0.24$ to $+0.38$ and $\beta(\text{GLD},\text{SPY})$ from $+0.05$ to $+0.46$, and running a $-20\%$ equity shock through the stressed betas takes a $60/40$ book from $-10.1\%$ to $-15.1\%$ — half again the loss, from the hedge ratio alone. A Gaussian model reports one beta and one residual risk for both regimes, and reports them with a formula that has no argument in which the regime could be passed. The failure is not that the number is imprecise; it is that the model has no slot for the question. Where a hedge is load-bearing, the ratio has to be estimated conditionally and reported as a range, and the range belongs beside the point estimate in every document that quotes it.

## The Update Is Linear Because the Model Said So

Regression, hedging, filtering and factor attribution are linear operations, and it is worth being clear about why. It is not that the world is linear. It is that joint normality makes the conditional expectation affine, and every one of those four techniques is the conditional expectation wearing different clothes. Assume something else and all four become nonlinear, and three of them stop having closed forms.

The uncomfortable part is that the formulas do not know this. A regression run on non-normal data returns coefficients. A Kalman filter fed a heavy-tailed observation returns a posterior. A hedge ratio computed from a covariance matrix estimated in a calm market returns a number, and the number is the same one it would have returned had the market been anything at all. None of these constructions contains a diagnostic for its own applicability, and all of them fail quietly, in the direction of confidence.

The practical rule is to check the residual rather than the fit. Under joint normality the conditional covariance is constant, which is an unusually testable claim: bucket the residuals of a hedge by the conditioning variable — market decile, volatility regime, whatever the exposure is — and plot their variance against it. A flat line licenses everything on this page, including the closed-form gain and the offline error analysis. A sloped line says the ratio you are using is an average over regimes, and that the regime in which you will need it is not the one it was averaged toward. Do not accept a linear hedge without first asking what would have to be true for the ratio to change, and then checking whether it is.
