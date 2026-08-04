# Regularization

A penalty is usually introduced as a defence against overfitting, which makes it sound like a concession — accept some bias, buy back some variance, hope the trade nets out. The linear-algebra reading is sharper and less apologetic. [Linear Algebra Review](../part-01-mathematical-foundations/05-linear-algebra-review.md) states it as a fact about eigenvalues: "Replacing $X^\top X$ with $X^\top X+\lambda I$ adds $\lambda$ to every eigenvalue, bounding the condition number by construction." A ridge penalty is not a statistical compromise; it is a conditioning fix that acts almost entirely on the directions the data failed to determine and leaves the well-determined directions essentially untouched. Below, a design with a condition number of $10{,}000$ is repaired to $100$, and the coefficient standard deviation in its worst direction falls from $10.004$ to $0.099$ while the best direction moves from $0.100$ to $0.099$. The $\ell_1$ penalty is a different animal with a different geometry, and its cost is paid in a currency the $\ell_2$ penalty never touches: the identity of the variables it keeps.

This page covers a penalty as a constraint set and the geometry that makes one solution small and the other sparse, ridge in the singular-value basis, the existence theorem guaranteeing some $\lambda>0$ beats least squares, the instability of the lasso's selected set, and the fact that scaling is part of the estimator. It does not derive the correspondence between a Gaussian prior and ridge or a Laplace prior and lasso, which [Maximum a Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md) already establishes with the penalty weight fixed by a ratio of variances; it does not choose $\lambda$ by resampling, which is [Cross-Validation](../part-14-model-selection/02-cross-validation.md); it does not decompose prediction error into bias and variance, which is [The Bias-Variance Tradeoff](../part-14-model-selection/01-bias-variance-tradeoff.md); it fits nothing unpenalized, which is [Multiple Linear Regression](02-multiple-linear-regression.md); it runs no general-purpose optimizer, which is [Numerical Optimization](../part-17-statistical-computing/01-numerical-optimization.md); it corrects no family of tests, which is [Part XV](../part-15-multiple-testing/index.md); and it never reports a selected set as though it were a finding.

The trading stake is the course's most comprehensively negative result. [Risk Parity, Diversification, and Factors](../../part-08-portfolio-management/03-risk-parity-diversification-factors.md) forecasts the next lesson's outcome in one sentence: "mean-variance optimization loses to naive equal weighting in every configuration tested, **the standard repair of shrinking the covariance matrix makes it worse**, and the thing that actually rescues it turns out to be the crudest tool available." Shrinking a covariance matrix is exactly the operation of section 2, applied to a different matrix, and section 3 is the reason a shrinkage that is guaranteed to help in mean squared error can still lose on the objective anyone cares about.

## A Penalty Is a Constraint Set, and the Geometry of the Constraint Decides Whether the Solution Is Small or Sparse

Both estimators solve a constrained problem in disguise. Minimizing $\lVert y-X\beta\rVert^{2}+\lambda\lVert\beta\rVert_2^{2}$ is, by Lagrangian duality, identical to minimizing the residual sum of squares subject to $\lVert\beta\rVert_2\le t$ for some $t$ depending on $\lambda$, and the same holds for the $\ell_1$ penalty with an $\ell_1$ ball. So the picture is a family of ellipsoidal contours of the least-squares objective expanding from $\hat\beta_{\mathrm{OLS}}$ until they first touch the constraint set, and the shape of that set decides where they touch.

An $\ell_2$ ball is smooth, so the contact point is almost never on an axis and every coordinate is shrunk toward zero without reaching it. An $\ell_1$ ball is a cross-polytope with vertices on the axes, and a vertex is exactly a point where all but one coordinate vanish. Contact at a corner is not a coincidence to be marvelled at; it is the generic outcome of touching a set that is mostly corners.

??? note "Proof that the $\ell_1$ subdifferential at the origin is an interval, which is why the lasso zeroes a coefficient on a set of positive measure and ridge never does"

    Consider one coordinate with the others held fixed, and write the objective as a function of $\beta_j$ alone. With columns scaled so $x_j^\top x_j=1$ and $c_j=x_j^\top r_{(j)}$ the correlation of $x_j$ with the partial residual, the two problems are
    $$\tfrac{1}{2}(\beta_j-c_j)^{2}+\lambda|\beta_j|\qquad\text{and}\qquad \tfrac{1}{2}(\beta_j-c_j)^{2}+\tfrac{\lambda}{2}\beta_j^{2}.$$
    The second is differentiable everywhere with a unique stationary point $\beta_j=c_j/(1+\lambda)$, which is zero only when $c_j$ is exactly zero — a measure-zero event. The first is not differentiable at the origin, where its subdifferential is $\{-c_j+\lambda s: s\in[-1,1]\}=[-c_j-\lambda,\,-c_j+\lambda]$. Zero belongs to that interval, and hence $\beta_j=0$ is optimal, whenever $|c_j|\le\lambda$ — an event of positive probability for any $\lambda>0$. Away from it the solution is the soft threshold $\operatorname{sign}(c_j)\max(|c_j|-\lambda,0)$, which is the update the coordinate-descent loop in section 4 implements in one line.

    The load-bearing difference is a kink, not a magnitude. The $\ell_2$ penalty's derivative at zero is zero, so it exerts no force on a coefficient that is already small and can never quite push one out; the $\ell_1$ penalty's derivative at zero is a nonzero constant $\lambda$ from both sides, so it exerts a fixed force regardless of how small the coefficient is. **Sparsity is a consequence of non-differentiability at the origin, which is why every sparse estimator has a kink and every smooth penalty produces small coefficients rather than absent ones.**

That is the algebraic content of the claim in [Linear Algebra Review](../part-01-mathematical-foundations/05-linear-algebra-review.md) that "the $\ell_1$ penalty in Regularization produces sparse solutions where the $\ell_2$ penalty produces small ones."

## Ridge Adds $\lambda$ to Every Eigenvalue, Shrinking Each Principal Direction in Inverse Proportion to How Well the Data Determines It

The ridge solution $\hat\beta_\lambda=(X^\top X+\lambda I)^{-1}X^\top y$ becomes transparent in the singular-value basis, where it stops being a compromise and becomes a per-direction decision.

??? note "Proof that ridge shrinks the $i$th principal direction by $d_i^{2}/(d_i^{2}+\lambda)$, so the penalty acts where the data is least informative and almost nowhere else"

    Write the singular value decomposition $X=UDV^\top$ with $D=\operatorname{diag}(d_1,\dots,d_p)$. Then $X^\top X=VD^{2}V^\top$ and
    $$\hat\beta_\lambda=V(D^{2}+\lambda I)^{-1}DU^\top y,$$
    so in the rotated coordinates $\alpha=V^\top\beta$ each component is handled independently:
    $$\hat\alpha_{\lambda,i}=\frac{d_i}{d_i^{2}+\lambda}\,u_i^\top y=\frac{d_i^{2}}{d_i^{2}+\lambda}\cdot\hat\alpha_{\mathrm{OLS},i}.$$
    The multiplier $d_i^{2}/(d_i^{2}+\lambda)$ is near one when $d_i^{2}\gg\lambda$ and near zero when $d_i^{2}\ll\lambda$. Since $\operatorname{var}(\hat\alpha_{\mathrm{OLS},i})=\sigma^{2}/d_i^{2}$, the directions being shrunk hardest are exactly the directions the least-squares estimate knows least about. Ridge variance is $\sigma^{2}d_i^{2}/(d_i^{2}+\lambda)^{2}$, which is bounded by $\sigma^{2}/(4\lambda)$ *uniformly in $d_i$* — no direction, however badly determined, can contribute more than that.

    The eigenvalues of $X^\top X+\lambda I$ are $d_i^{2}+\lambda$, so the condition number becomes $(d_{\max}^{2}+\lambda)/(d_{\min}^{2}+\lambda)$, which is strictly smaller than $d_{\max}^{2}/d_{\min}^{2}$ and tends to $1$ as $\lambda$ grows. This is the sense in which the fix is algebraic rather than statistical: the bound holds for every dataset, before any assumption about where $\beta$ lies.

    The load-bearing quantity is the ratio $\lambda/d_i^{2}$, not $\lambda$ itself. **A single penalty parameter buys a different amount of shrinkage in every direction, and the amount is decided by the design matrix rather than by the analyst.**

Every claim in that proof is a number that can be measured:

```python
import numpy as np

rng = np.random.default_rng(13051)
n, p, sig, lam, reps = 400, 6, 1.0, 1.0, 20_000

d = np.geomspace(10.0, 0.1, p)                    # design spectrum, cond(X) = 100
U = np.linalg.qr(rng.standard_normal((n, p)))[0]
V = np.linalg.qr(rng.standard_normal((p, p)))[0]
X = (U * d) @ V.T
beta = V @ np.ones(p)

G = X.T @ X
ridge = np.linalg.inv(G + lam * np.eye(p)) @ X.T
ols = np.linalg.inv(G) @ X.T

a_ols = np.empty((reps, p))
a_rdg = np.empty((reps, p))
for r in range(reps):
    y = X @ beta + rng.normal(0, sig, n)
    a_ols[r] = V.T @ (ols @ y)                    # coefficients in principal coords
    a_rdg[r] = V.T @ (ridge @ y)

print(f"  ridge with lambda = {lam:.1f} on a design whose singular values span 10 to 0.1")
print("    d_i    d_i^2   shrink d^2/(d^2+L)   sd OLS   sd ridge   bias ridge")
for i in range(p):
    print(f"    {d[i]:6.3f}  {d[i]**2:7.3f}   {d[i]**2 / (d[i]**2 + lam):17.4f}"
          f"   {a_ols[:, i].std():6.3f}   {a_rdg[:, i].std():8.3f}"
          f"   {a_rdg[:, i].mean() - 1.0:10.4f}")
print(f"    cond(X'X) = {np.linalg.cond(G):.1f}, "
      f"cond(X'X + {lam:.0f}I) = {np.linalg.cond(G + lam * np.eye(p)):.1f}")
print(f"    total MSE   OLS {np.mean((a_ols - 1.0) ** 2).sum() * p:.4f}   "
      f"ridge {np.mean((a_rdg - 1.0) ** 2).sum() * p:.4f}")
# =>   ridge with lambda = 1.0 on a design whose singular values span 10 to 0.1
#        d_i    d_i^2   shrink d^2/(d^2+L)   sd OLS   sd ridge   bias ridge
#        10.000  100.000              0.9901    0.100      0.099      -0.0101
#         3.981   15.849              0.9406    0.251      0.236      -0.0615
#         1.585    2.512              0.7153    0.630      0.451      -0.2792
#         0.631    0.398              0.2847    1.587      0.452      -0.7214
#         0.251    0.063              0.0594    3.968      0.236      -0.9414
#         0.100    0.010              0.0099   10.004      0.099      -0.9898
#        cond(X'X) = 10000.0, cond(X'X + 1I) = 100.0
#        total MSE   OLS 118.8089   ridge 3.0061
```

The shrinkage column runs $0.9901$ to $0.0099$: the best-determined direction is passed through essentially unchanged and the worst is annihilated. The bias column is the same statement as a cost — $-0.0101$ at the top, meaning ridge gives up one percent of the truth in that direction, and $-0.9898$ at the bottom, meaning it gives up almost all of it. That looks catastrophic until the variance column is read alongside it. Least squares estimates the bottom direction with a standard deviation of $10.004$ around a true value of $1$; the estimate is noise. Discarding $99\%$ of a quantity that was never measured costs nothing worth having.

The ridge standard deviations are the proof's uniform bound, visible: $0.099$, $0.236$, $0.451$, $0.452$, $0.236$, $0.099$ — they rise and then fall, never exceeding $\sigma/(2\sqrt\lambda)=0.5$, while the OLS column spans two orders of magnitude. The condition number reads $10000.0$ before and $100.0$ after, exactly $(100+1)/(0.01+1)$, which is the eigenvalue claim discharged as arithmetic rather than as heuristic. Total mean squared error falls from $118.8089$ to $3.0061$, a factor of $40$.

**Ridge is not a trade of accuracy for stability across the board; it is a decision, taken separately in every principal direction, to keep what the data determined and discard what it did not — and the discarded part was mostly noise in the first place.**

## There Is Always a $\lambda>0$ That Beats Least Squares in Mean Squared Error, and the Value That Achieves It Depends on the $\beta$ You Do Not Know

The previous section's improvement was not luck. There is a theorem.

??? note "Proof that for every $\beta$ and every design there exists $\lambda>0$ with strictly smaller total mean squared error than least squares"

    In principal coordinates the total mean squared error of the ridge estimator is a sum over directions,
    $$M(\lambda)=\sum_i\left[\frac{\sigma^{2}d_i^{2}}{(d_i^{2}+\lambda)^{2}}+\frac{\lambda^{2}\alpha_i^{2}}{(d_i^{2}+\lambda)^{2}}\right],$$
    the first term the variance and the second the squared bias, with $\alpha=V^\top\beta$. Differentiate at $\lambda=0$:
    $$M'(0)=\sum_i\left[\frac{-2\sigma^{2}d_i^{2}}{d_i^{6}}+0\right]=-2\sigma^{2}\sum_i\frac{1}{d_i^{4}}<0,$$
    the bias term contributing nothing because it is $O(\lambda^{2})$ and its derivative vanishes at the origin. So $M$ is strictly decreasing at $\lambda=0$ and some positive $\lambda$ gives a strictly smaller error, for every $\beta$, every design and every $\sigma^{2}>0$. Setting each direction's derivative to zero separately gives the per-direction optimum $\lambda_i=\sigma^{2}/\alpha_i^{2}$, which depends on the unknown $\alpha_i$.

    The asymmetry driving the result is that variance is first order in $\lambda$ near zero while bias is second order. A small penalty buys a large variance reduction at negligible bias cost — always, unconditionally. That is why the theorem needs no assumption and also why it promises so little: it guarantees an improvement exists without identifying it, and the maximizing $\lambda$ is a function of exactly the quantity being estimated.

    The load-bearing fact is the mismatch between what is proved and what is available. **The existence of a beneficial penalty is a theorem; the value of the beneficial penalty is an estimation problem no easier than the original one.**

How much the theorem is worth in practice depends on where $\beta$ points relative to the design:

```python
import numpy as np

rng = np.random.default_rng(13053)
n, p, sig, reps = 60, 20, 1.0, 3000

d = np.geomspace(6.0, 0.3, p)
V = np.linalg.qr(rng.standard_normal((p, p)))[0]
U = np.linalg.qr(rng.standard_normal((n, p)))[0]
X = (U * d) @ V.T
G = X.T @ X
grid = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 61)])

print("  ridge MSE against lambda, n = 60, p = 20, cond(X'X) = "
      f"{np.linalg.cond(G):.0f}")
print("  the same design and noise; only where beta points changes")
print("    beta aligned with   MSE at L=0   best L   MSE at best   improvement")
for name, w in (("large sing. values", d**2),
                ("isotropic", np.ones(p)),
                ("small sing. values", 1.0 / d**2)):
    b = V @ (np.sqrt(w / w.sum()) * np.sqrt(p))
    Y = X @ b + rng.normal(0, sig, (reps, n))
    mse = np.empty(len(grid))
    for k, lam in enumerate(grid):
        A = np.linalg.inv(G + lam * np.eye(p)) @ X.T
        mse[k] = np.mean(np.sum((Y @ A.T - b) ** 2, axis=1))
    j = int(np.argmin(mse))
    print(f"    {name:17s}   {mse[0]:10.3f}   {grid[j]:6.2f}   {mse[j]:11.3f}"
          f"   {1 - mse[j] / mse[0]:11.1%}")
# =>   ridge MSE against lambda, n = 60, p = 20, cond(X'X) = 400
#      the same design and noise; only where beta points changes
#        beta aligned with   MSE at L=0   best L   MSE at best   improvement
#        large sing. values       40.461     2.51         2.250         94.4%
#        isotropic               40.146     1.00         8.190         79.6%
#        small sing. values       41.218     0.50        15.713         61.9%
```

The theorem holds in all three rows, as it must: every configuration has a positive $\lambda$ that improves on least squares, and the improvements are large — $94.4\%$, $79.6\%$, $61.9\%$. The unpenalized error is essentially identical across the three at $40.461$, $40.146$ and $41.218$, so the differences are entirely in what the penalty could recover.

The spread is the point. When $\beta$ lies along the well-determined directions, ridge shrinks only what was noise and removes $94.4\%$ of the error. When $\beta$ lies along the badly-determined directions — the case where the signal is genuinely in the part of the design the data barely resolves — the same machinery still helps, but recovers $61.9\%$, and the optimal $\lambda$ is five times smaller at $0.50$ against $2.51$. An analyst who tuned $\lambda$ on one problem and reused it on the other would be off by a factor of five in a quantity the theory says must be matched to an unknown.

This is where the course's covariance-shrinkage failure becomes legible. Shrinking a covariance matrix toward a target is the same operation with $\lambda$ set by a formula, and the guarantee it carries is of exactly the kind proved above: an improvement in the mean squared error *of the matrix*. Mean-variance optimization does not consume the matrix; it consumes $\Sigma^{-1}\mu$, and an estimator that is better in Frobenius error can be worse after inversion and multiplication by a noisy mean vector. **The theorem is about the loss it is stated for, and no loss function is entitled to assume another one agrees with it.**

!!! note "Ridge, Tikhonov regularization, weight decay and a Gaussian prior are four names for one estimator"
    **Ridge regression** is the statistician's name, introduced to fix ill-conditioned normal equations. **Tikhonov regularization** is the numerical-analysis name for the same operator applied to any ill-posed linear inverse problem, and the general form allows $\lambda\Gamma^\top\Gamma$ in place of $\lambda I$. **Weight decay** is the neural-network name, and in plain gradient descent it is identical, though with adaptive optimizers it is not — decoupling the two is a real algorithmic distinction rather than a naming one. A **Gaussian prior** on $\beta$ makes the posterior mode equal the ridge solution with $\lambda=\sigma^{2}/\tau^{2}$, the ratio of the noise variance to the prior variance, which [Maximum a Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md) derives along with the Laplace-prior-and-lasso pair. The useful content of the four-way identity is that it tells you what tuning $\lambda$ means: choosing a penalty is choosing a ratio of variances, so a $\lambda$ transplanted between problems with different noise levels is a prior transplanted between problems with different beliefs.

## The $\ell_1$ Penalty Selects a Set, and That Set Is a Random Variable Whose Ranking of Predictors Need Not Match the Truth

Ridge shrinks and keeps everything, so there is no selected set to be unstable. The lasso returns a subset, and because the subset is a function of the sample it is a random variable — one that gets reported as though it were a finding. What it does under correlated predictors is measurable with the soft-threshold update of section 1 written as a coordinate-descent loop:

```python
import numpy as np

rng = np.random.default_rng(13055)
n, p, reps, rho = 120, 8, 500, 0.95
beta = np.array([1.5, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])


def draw(m):
    Z = rng.standard_normal((m, p))
    Z[:, 1] = rho * Z[:, 0] + np.sqrt(1 - rho**2) * Z[:, 1]
    return Z, Z @ beta + rng.standard_normal(m)


def lasso(X, y, lam, tol=1e-9, maxit=2000):
    b = np.zeros(X.shape[1])
    r = y - X @ b
    nrm = (X**2).sum(0)
    for _ in range(maxit):
        step = 0.0
        for j in range(X.shape[1]):
            rj = r + X[:, j] * b[j]
            c = X[:, j] @ rj
            nb = np.sign(c) * max(abs(c) - lam, 0.0) / nrm[j]
            if nb != b[j]:
                r = rj - X[:, j] * nb
                step = max(step, abs(nb - b[j]))
                b[j] = nb
        if step < tol:
            break
    return b


Xv, yv = draw(20_000)                                 # held-out set to score lambda
Xt, yt = draw(n)
grid = np.geomspace(0.5, 300.0, 50)
star = grid[int(np.argmin([np.mean((yv - Xv @ lasso(Xt, yt, L)) ** 2) for L in grid]))]

print(f"  lasso selection over {reps} resamples, n = {n}; x1 truly 1.5, x2 a decoy truly 0,")
print(f"  correlated with x1 at {rho}; x3 truly 0.8; x4..x8 truly zero")
print("    lambda      x1      x2      x3   noise   both x1,x2   one only   OOS MSE")
for lam in (star, 30.0, 90.0, 150.0):
    sel = np.zeros(p)
    both = one = 0
    mse = np.empty(reps)
    for r in range(reps):
        X, y = draw(n)
        b = lasso(X, y, lam)
        on = b != 0.0
        sel += on
        both += on[0] and on[1]
        one += on[0] != on[1]
        mse[r] = np.mean((yv - Xv @ b) ** 2)
    tag = f"{lam:6.1f}*" if lam == star else f"{lam:6.1f} "
    print(f"    {tag}  {sel[0] / reps:6.3f}  {sel[1] / reps:6.3f}  {sel[2] / reps:6.3f}"
          f"  {sel[3:].mean() / reps:6.3f}   {both / reps:10.3f}   {one / reps:8.3f}"
          f"   {mse.mean():7.4f}")
print("    * lambda minimising held-out prediction error")
# =>   lasso selection over 500 resamples, n = 120; x1 truly 1.5, x2 a decoy truly 0,
#      correlated with x1 at 0.95; x3 truly 0.8; x4..x8 truly zero
#        lambda      x1      x2      x3   noise   both x1,x2   one only   OOS MSE
#          14.9*   1.000   0.426   1.000   0.176        0.426      0.574    1.0721
#          30.0    1.000   0.344   1.000   0.008        0.344      0.656    1.1735
#          90.0    0.998   0.178   0.636   0.000        0.176      0.824    2.1659
#         150.0    0.812   0.108   0.012   0.000        0.046      0.828    3.3087
#        * lambda minimising held-out prediction error
```

The first row is the $\lambda$ an honest tuning procedure would pick, chosen by held-out error on twenty thousand fresh observations — no cross-validation noise, no optimism. At that value the lasso selects the genuine $x_1$ every time and the genuine $x_3$ every time, which is the good news, and it also selects a decoy with a true coefficient of exactly zero $42.6\%$ of the time and each pure-noise column $17.6\%$ of the time. The $\lambda$ that predicts best is not the $\lambda$ that selects correctly, and the gap is not small: at the prediction optimum roughly one in five irrelevant variables is in the model.

Reading down the table shows what buying sparsity costs. Raising $\lambda$ to $30$ almost eliminates the noise columns, at $0.008$, while leaving both true predictors at $1.000$ — a genuinely better *selection* than the prediction optimum, obtained at a $9\%$ worse prediction error. Push further and the trade turns bad. At $\lambda=90$ the decoy $x_2$ is still selected $17.8\%$ of the time while the **true** predictor $x_3$ has fallen to $63.6\%$, and at $\lambda=150$ the decoy is selected $10.8\%$ of the time against $x_3$'s $1.2\%$ — a variable with no relationship to the outcome is retained nine times as often as one with a coefficient of $0.8$. The lasso is not ranking predictors by whether they matter; it is ranking them by correlation with what is already in the model, and a near-duplicate of a strong predictor outranks an independent weak one.

The last column is the honest accounting for the whole table. Out-of-sample error is $1.0721$, $1.1735$, $2.1659$, $3.3087$: monotonically worse as sparsity is bought. Every improvement in the selected set was paid for in prediction, and no row does well at both.

**A lasso solution answers "which small set of columns predicts nearly as well" and is routinely read as answering "which variables matter", and the second question has an answer that the first one's answer is not a consistent estimate of.**

## Scaling Is Part of the Estimator, Because a Penalty on Coefficients Is a Penalty in the Units of the Columns

Neither penalty is invariant to rescaling a column. Multiply $x_j$ by $100$ and its coefficient must fall by $100$ to leave the fit unchanged, so its contribution to $\lVert\beta\rVert^{2}$ falls by $10{,}000$ and the penalty essentially stops applying to it. A predictor measured in basis points and one measured in percent are the same variable and receive penalties differing by a factor of $10^{4}$.

The consequence is that standardization is not preprocessing, it is a component of the estimator, and the choice of what to standardize against — full-sample standard deviation, trailing standard deviation, or a fixed scale — changes the estimate. On financial data the first choice leaks future information into the training columns, which is the failure [Feature Engineering for ML](../../part-07-machine-learning/01-feature-engineering-for-ml.md) organizes itself around, so the correct construction standardizes within each training fold using only that fold's statistics and applies the same transformation to the validation rows.

The intercept is the standard exception and should be left unpenalized, since penalizing it makes the fit depend on where the origin of $y$ happens to sit — shifting every response by a constant would change the slopes. Software does this by default and the convention is worth knowing rather than rediscovering.

!!! warning "The selected set is a random variable, and it is the one part of the output that gets reported as a finding"
    Nothing in a lasso fit is marked provisional. The coefficient vector is a valid solution to a convex problem with a unique minimum in the fitted values; the out-of-sample error is honestly estimated; the zeros look like conclusions because a zero is a definite-seeming number. What the output cannot show is that the same procedure on a second sample from the same population would return a different set. Above, at $\lambda=90$, a variable with a true coefficient of zero appeared in $17.8\%$ of fits while a variable with a true coefficient of $0.8$ appeared in $63.6\%$, and at $\lambda=150$ the ordering inverted outright, $10.8\%$ against $1.2\%$. Two analysts running identical code on adjacent samples publish different feature stories at indistinguishable prediction error, and each one's write-up is internally consistent. **The free diagnostic is to refit on a few hundred bootstrap resamples and print each predictor's selection frequency next to its coefficient, exactly as the table above does: a variable selected in half the resamples is not a finding, a variable whose frequency is below that of a variable you believe to be noise is evidence about your $\lambda$ rather than about your data, and the stable quantity to report — as with the collinear pair of [Multiple Linear Regression](02-multiple-linear-regression.md) — is the prediction, which barely moved while the selected set was changing underneath it.**

## Shrinkage Buys Prediction With Bias, and the One Thing It Cannot Buy Is the Story About Which Variables Matter

This page established that a penalty is a constraint set whose geometry decides the outcome, with the $\ell_1$ ball's kink at the origin giving a subdifferential interval $[-c_j-\lambda,-c_j+\lambda]$ that makes exact zeros an event of positive probability while the smooth $\ell_2$ penalty produces them only on a measure-zero set; that ridge acts per principal direction with multiplier $d_i^{2}/(d_i^{2}+\lambda)$, measured from $0.9901$ down to $0.0099$, taking the coefficient standard deviation in the worst direction from $10.004$ to $0.099$ and $\operatorname{cond}(X^\top X)$ from $10000.0$ to exactly $100.0$, for a total mean squared error of $3.0061$ against least squares' $118.8089$; that some $\lambda>0$ always beats least squares because variance falls at first order while bias rises at second, delivering improvements of $94.4\%$, $79.6\%$ and $61.9\%$ at optima of $2.51$, $1.00$ and $0.50$ according to where $\beta$ pointed in a fixed design; and that the lasso's selected set is unstable, admitting a true-zero decoy $42.6\%$ of the time at the prediction-optimal $\lambda=14.9$ and, at $\lambda=150$, retaining that decoy $10.8\%$ of the time against a genuine predictor's $1.2\%$, while out-of-sample error rose monotonically from $1.0721$ to $3.3087$ as sparsity was purchased.

The two penalties fail in ways that mirror each other. Ridge is honest about keeping everything and dishonest about nothing; its cost is a bias whose size is knowable from the spectrum and whose benefit is guaranteed by a theorem, and the only thing it cannot tell you is which columns did the work. The lasso answers that question and answers it unreliably, because a subset chosen by a convex program under correlated columns is chosen partly by sampling noise, and the answer is delivered in the most confident-looking format available — a coefficient of exactly zero.

The course's shrinkage failure sits precisely at this seam. Shrinking a covariance matrix is guaranteed to improve the matrix, and the guarantee is discharged in the loss it was written for; the optimizer downstream consumes $\Sigma^{-1}\mu$ and is entitled to no guarantee at all, which is how a theorem and a negative backtest coexist without either being wrong. What none of this page has done is look at the fit it produced. A penalty changes the estimate and leaves untouched the question of whether the model's *form* is right — whether some rows are running the regression, whether the mean function bends, whether the errors are what the standard errors assumed. Those are questions for the fitted object rather than the objective, and they start with the projection matrix. That is [Model Diagnostics](06-model-diagnostics.md).

**Regularization is a statement about what to do when the data does not determine a direction, and every version of it answers that question well while quietly answering a second question — which variables are real — that it was never given the information to answer.**
