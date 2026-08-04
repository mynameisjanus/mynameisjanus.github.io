# Logistic Regression

The binomial member of the previous page's class is the one where the variance function cannot be wrong. A two-valued random variable's mean determines its whole distribution, so $\operatorname{var}(y)=\mu(1-\mu)$ is arithmetic rather than assumption, and the overdispersion that made a nominal $5\%$ test reject $47.50\%$ of the time has no way to occur. What replaces it are two failures of an entirely different character. The first is that the maximum likelihood estimate sometimes does not exist, and the diagnostic everyone reaches for reports the opposite of what happened: as a predictor becomes perfect, its Wald p-value climbs toward one. The second is that the model's probabilities and the model's ranking are separate properties, so a rebalanced training set leaves the ranking untouched to four decimals while moving the fraction of days a strategy would hold a position from $50\%$ to $91\%$.

This page covers the logit link as an additive log-odds update, the strict concavity of the log-likelihood and the separation case where the maximizer escapes to infinity, the $O(p/n)$ inflation of the fitted coefficients, the exact intercept shift a changed base rate produces, and the difference between ranking and calibration. It does not build the exponential-family or IRLS machinery this model is an instance of, which is [Generalized Linear Models](03-generalized-linear-models.md); it does not derive the odds form of the update, which is [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md); it fits no continuous response, which is [Multiple Linear Regression](02-multiple-linear-regression.md); it tunes no penalty and derives no prior correspondence, which are [Regularization](05-regularization.md) and [Maximum a Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md); it selects among no candidate feature sets and runs no cross-validation, which is [Part XIV](../part-14-model-selection/index.md); it computes no influence measure, which is [Model Diagnostics](06-model-diagnostics.md); and it never treats a discrimination metric as evidence that a probability is a probability.

The trading stake is the model that wins the course's own bake-off. [Tree Ensembles](../../part-07-machine-learning/02-tree-ensembles.md) pits four models against each other and reports `logistic: AUC 0.478   net Sharpe 0.38   days long 16%`, with the forest at `0.17`, LightGBM at `0.20` and XGBoost at `−0.01`, observing that "the logistic regression — one second of training, a model from the appendix — posts the best net Sharpe of the four models." It also notes that every model is "long only 13–38% of days" because "their probabilities hug the middle and rarely clear 0.5 with conviction." That last sentence is a calibration statement rather than an accuracy statement, and sections 4 and 5 are why the two cannot be read off each other — and why an AUC of $0.478$, below a coin, is compatible with the best P&L on the board.

## The Logit Link Turns the Linear Predictor Into an Additive Log-Odds Update, So Each Coefficient Is One Bayes Factor per Unit of Its Feature

Take the binomial family from [Generalized Linear Models](03-generalized-linear-models.md) with one trial. Its natural parameter is $\theta=\log\frac{\mu}{1-\mu}$, so the canonical link is the logit and the model is

$$\log\frac{p(x)}{1-p(x)}=x^\top\beta,\qquad p(x)=\frac{1}{1+e^{-x^\top\beta}}.$$

The left side is the quantity [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md) shows evidence accumulates in. Posterior odds are prior odds times a likelihood ratio, so on the log scale updating is *addition*, and a model that sums weighted features is performing one Bayesian update per feature. That page states the consequence directly: a linear predictor is "precisely what the linear predictor of a Logistic Regression is, and why its coefficients are read as log-odds contributions rather than probabilities."

Three readings of $\beta_j$ follow and only the first is unconditional. A unit increase in $x_j$ adds $\beta_j$ to the log-odds — always, everywhere, regardless of the other features. It multiplies the odds by $e^{\beta_j}$ — also always. It changes the *probability* by an amount depending on where the probability already was, maximal at $p=1/2$ where the derivative $p(1-p)$ is $1/4$ and vanishing at either extreme. This is why coefficients are quoted as log-odds or odds ratios and why "this feature raises the probability by three percent" is not a statement the model makes.

The intercept carries the base rate. With all features at zero the model returns $\log\frac{p}{1-p}=\beta_0$, so $\beta_0$ is the prior log-odds and the features are the evidence — a decomposition that section 4 turns into an exact and exploitable identity.

## The Log-Likelihood Is Strictly Concave, So the Fit Has One Optimum — Except When the Classes Are Separable, When It Has None

The concavity argument of the previous page applies unchanged: $-\nabla^{2}\ell=X^\top WX$ with $W=\operatorname{diag}(p_i(1-p_i))$ strictly positive, so the surface has a unique maximum whenever $X$ has full column rank. The gap in that sentence is that a unique maximum of a strictly concave function need not be *attained*.

??? note "Proof that the log-likelihood is strictly concave under full column rank, and that its supremum is attained if and only if no hyperplane separates the classes"

    Write $\ell(\beta)=\sum_i\{y_i x_i^\top\beta-\log(1+e^{x_i^\top\beta})\}$. Differentiating twice gives $\nabla^{2}\ell=-X^\top WX$ with $w_i=p_i(1-p_i)>0$ for every finite $\beta$, so for any $v\neq0$, $v^\top X^\top WXv=\sum_i w_i(x_i^\top v)^{2}>0$ unless $Xv=0$, which full column rank forbids. The surface is strictly concave and can have at most one stationary point.

    For existence, suppose a vector $v$ separates the classes in the weak sense that $x_i^\top v>0$ whenever $y_i=1$ and $x_i^\top v<0$ whenever $y_i=0$. Evaluate the likelihood along the ray $\beta=tv$ and let $t\to\infty$: every term $y_i x_i^\top\beta-\log(1+e^{x_i^\top\beta})$ increases monotonically toward zero, since for $y_i=1$ the expression is $-\log(1+e^{-t\,x_i^\top v})\to0^{-}$ and for $y_i=0$ it is $-\log(1+e^{t\,x_i^\top v})\to0^{-}$. So $\ell(tv)\to0$, which is the supremum of a log-likelihood built from probabilities, and it is approached but never reached at finite $t$. The maximizer is the ray's direction at infinite length: it does not exist.

    Conversely if no such $v$ exists, every direction has at least one observation contradicting it, that observation's contribution is bounded away from zero along the ray, and $\ell$ is coercive — it tends to $-\infty$ in every direction — so a maximum is attained. Separation is therefore not a numerical accident but an exact geometric condition, and it becomes *more* likely as $p$ grows relative to $n$, since more columns give more directions to separate along.

    The load-bearing observation is what the Wald statistic does about it. The estimated standard error is $\sqrt{[(X^\top WX)^{-1}]_{jj}}$, and along the escaping ray every $p_i$ goes to $0$ or $1$, so every $w_i=p_i(1-p_i)$ goes to zero and $X^\top WX$ goes to the zero matrix. The standard error diverges *faster* than the coefficient. **The Wald ratio $\hat\beta_j/\widehat{\operatorname{se}}_j$ therefore tends to zero as the evidence becomes overwhelming, so the p-value of a perfect predictor tends to one.**

That last claim inverts every intuition about what a p-value reports, and it is measurable by walking a dataset from noisy toward perfectly separated:

```python
import numpy as np
from scipy import special, stats

rng = np.random.default_rng(13041)
n = 200
x = rng.standard_normal(n)
X = np.column_stack([np.ones(n), x])


def fit(X, y, lam=0.0, tol=1e-10, maxit=200):
    b = np.zeros(X.shape[1])
    for it in range(1, maxit + 1):
        p = special.expit(X @ b)
        w = np.clip(p * (1 - p), 1e-12, None)
        H = X.T @ (X * w[:, None]) + lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - lam * b)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    p = special.expit(X @ b)
    w = np.clip(p * (1 - p), 1e-12, None)
    cov = np.linalg.inv(X.T @ (X * w[:, None]) + lam * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov)), it


print("  y = 1{x > 0} with a fraction of labels flipped; as the flips vanish the")
print("  classes become linearly separable and the maximum stops existing")
print("    flips   iters       b1     se(b1)   Wald z   Wald p   ridge b1 (lam=1e-3)")
for f in (0.20, 0.10, 0.05, 0.02, 0.00):
    y = (x > 0).astype(float)
    k = int(round(f * n))
    if k:
        idx = rng.choice(n, k, replace=False)
        y[idx] = 1.0 - y[idx]
    b, se, it = fit(X, y)
    z = b[1] / se[1]
    br = fit(X, y, lam=1e-3)[0]
    print(f"    {f:5.2f}   {it:5d}  {b[1]:8.2f}  {se[1]:9.2f}  {z:7.3f}"
          f"  {2 * stats.norm.sf(abs(z)):7.4f}  {br[1]:20.2f}")
# =>   y = 1{x > 0} with a fraction of labels flipped; as the flips vanish the
#      classes become linearly separable and the maximum stops existing
#        flips   iters       b1     se(b1)   Wald z   Wald p   ridge b1 (lam=1e-3)
#         0.20       6      1.29       0.21    6.229   0.0000                  1.29
#         0.10       7      2.31       0.33    7.080   0.0000                  2.30
#         0.05       8      3.63       0.53    6.907   0.0000                  3.63
#         0.02      11     11.82       2.73    4.334   0.0000                 11.73
#         0.00     200   1194.08   68532.56    0.017   0.9861                 47.27
```

The coefficient column behaves as the geometry demands: $1.29$, $2.31$, $3.63$, $11.82$ and then $1194.08$, which is not an estimate but a report of where the optimizer was when it ran out of iterations — the loop hit its cap of $200$ where every other row converged in $6$ to $11$. Run it longer and the number grows without bound.

The Wald column is the finding. The $z$ statistic peaks at $7.080$ with a tenth of the labels flipped and then *falls*: $6.907$, $4.334$, and finally $0.017$ at perfect separation, where the two-sided p-value is $0.9861$. A predictor that classifies every single observation correctly is reported as the least significant predictor in the table. This is the **Hauck–Donner effect**, and it is not a bug in any implementation — it is the exact consequence proved above, that $\widehat{\operatorname{se}}$ diverges faster than $\hat\beta$ when the fitted probabilities saturate. Software that prints this table is computing every entry correctly.

The last column is the repair and it is one line. Adding a ridge penalty of $\lambda=10^{-3}$ — a penalty small enough to change the four converged rows by at most $0.09$, from $2.31$ to $2.30$ and $11.82$ to $11.73$ — makes the objective coercive and pulls the separated fit back to a finite $47.27$. The penalized problem always has a maximizer because the penalty grows without bound in every direction while the log-likelihood is bounded above, which is the geometric content of [Regularization](05-regularization.md).

**Separation is the one failure in this part that announces itself in the coefficient and then conceals itself in the p-value, and the direction of the concealment is toward accepting that a perfect predictor does nothing.**

## Maximum Likelihood Is Biased Away From Zero at Order $p/n$, Which Is the Regime Every Financial Classifier Lives In

Separation is the extreme case of a bias that operates continuously. The maximum likelihood estimator is consistent as $n\to\infty$ with $p$ fixed, and financial classifiers are not in that regime — they have dozens of features and a few hundred effectively independent observations, and there the fitted coefficients are systematically too large:

```python
import numpy as np
from scipy import special, stats

rng = np.random.default_rng(13043)
reps, beta1 = 3000, 0.80


def fit(X, y, tol=1e-10, maxit=60):
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        p = special.expit(X @ b)
        w = np.clip(p * (1 - p), 1e-10, None)
        try:
            step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - p))
        except np.linalg.LinAlgError:
            return None, None
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    p = special.expit(X @ b)
    w = np.clip(p * (1 - p), 1e-10, None)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * w[:, None]))))
    return b, se

print("  logistic MLE bias: x1 truly carries 0.80, x2 truly carries 0, rest noise")
print("  nominal 5% Wald test on the null coefficient x2")
print("      n    p   mean b1/0.80   Wald size on x2   diverged")
for n in (50, 100, 250, 1000):
    for p in (2, 5, 10):
        ratio, hit, bad = [], 0, 0
        for r in range(reps):
            Z = rng.standard_normal((n, p))
            X = np.column_stack([np.ones(n), Z])
            eta = beta1 * Z[:, 0]
            y = (rng.random(n) < special.expit(eta)).astype(float)
            b, se = fit(X, y)
            if b is None or np.max(np.abs(b)) > 25:
                bad += 1
                continue
            ratio.append(b[1] / beta1)
            hit += abs(b[2] / se[2]) > stats.norm.isf(0.025)
        k = len(ratio)
        print(f"    {n:5d}  {p:3d}   {np.mean(ratio):12.3f}   {hit / k:15.4f}"
              f"   {bad / reps:8.4f}")
# =>   logistic MLE bias: x1 truly carries 0.80, x2 truly carries 0, rest noise
#      nominal 5% Wald test on the null coefficient x2
#          n    p   mean b1/0.80   Wald size on x2   diverged
#           50    2          1.123            0.0477     0.0000
#           50    5          1.219            0.0473     0.0000
#           50   10          1.485            0.0584     0.0013
#          100    2          1.053            0.0460     0.0000
#          100    5          1.084            0.0513     0.0000
#          100   10          1.156            0.0567     0.0000
#          250    2          1.016            0.0497     0.0000
#          250    5          1.033            0.0453     0.0000
#          250   10          1.053            0.0483     0.0000
#         1000    2          1.006            0.0493     0.0000
#         1000    5          1.008            0.0463     0.0000
#         1000   10          1.016            0.0540     0.0000
```

The inflation is governed by $p/n$ and not by either alone. At $n=1000$ with two predictors the fitted coefficient averages $1.006$ times the truth — consistency, visible. At $n=50$ with ten predictors it averages $1.485$: a model asked what the odds ratio is answers $e^{0.8\times1.485}=3.3$ when the truth is $e^{0.8}=2.2$. Reading down the $p=10$ column gives $1.485$, $1.156$, $1.053$, $1.016$ as $n$ goes $50$, $100$, $250$, $1000$, and reading across each $n$ shows the same quantity growing with the column count at fixed sample size. The rate is the ratio: $p/n$ of $0.2$ buys nearly $50\%$ inflation, $p/n$ of $0.01$ buys under $2\%$.

The second column is the part worth being careful about, because it does *not* show a failure. The nominal $5\%$ Wald test on a coefficient that is exactly zero holds between $0.0453$ and $0.0584$ across the entire table. The bias inflates estimated effect sizes; it does not inflate the false-positive rate on null coefficients. These are different claims and only the first is damaged, which means a small-sample logistic fit can be trusted about *which* features matter considerably more than about *how much* they matter — the reverse of the usual worry.

The divergence column records section 2 operating in the background: at $n=50$ and $p=10$, $0.13\%$ of samples separated outright and were discarded. That rate climbs steeply with $p/n$, and every discarded sample is one where an analyst would have seen an enormous coefficient with an enormous standard error and a p-value near one.

!!! note "The logit, the log-odds, the linear predictor and the score are four names for one quantity, and the sigmoid, the expit, the inverse logit and the mean function are four names for its inverse"
    The **logit** is the function $u\mapsto\log\frac{u}{1-u}$; the **log-odds** is its value at a particular probability; the **linear predictor** $\eta=x^\top\beta$ is the model's assertion about that value; and the **natural parameter** $\theta$ of the binomial family equals it under the canonical link, which is the whole reason the link is canonical. Going the other way, **sigmoid**, **expit**, **inverse logit** and the GLM's **mean function** all name $\eta\mapsto1/(1+e^{-\eta})$; `scipy.special.expit` is the numerically safe spelling and hand-rolling `1/(1+np.exp(-eta))` overflows for $\eta<-745$. Two further collisions matter in practice. The **score** in the GLM sense is $\nabla\ell=X^\top(y-p)$, a vector that is zero at the optimum, while the **score** a machine-learning pipeline reports is the model's output $p$ or $\eta$ for one row — the same word for a gradient and for a prediction. And **logistic regression** names both this model and, in the neural-network vocabulary, a single-layer network with a sigmoid activation and cross-entropy loss, which is the identical object fitted by a worse algorithm.

## A Shifted Base Rate Moves the Intercept and Nothing Else, So a Model Trained on Rebalanced Classes Is Wrong by a Known Constant

Class rebalancing — discarding majority-class rows, or resampling until the classes are equal — is standard practice and it has an exact, closed-form consequence.

??? note "Proof that changing the base rate between training and deployment shifts only the intercept, leaving every slope and therefore the entire ranking unchanged"

    Suppose the class-conditional feature distributions $f(x\mid y=1)$ and $f(x\mid y=0)$ are the same in training and deployment, and only the class prior differs — training at $\pi_{\mathrm{tr}}$, deployment at $\pi$. This is **label shift**, and it is exactly what rebalancing induces by construction. By Bayes' rule the deployment log-odds are
    $$\log\frac{P(y=1\mid x)}{P(y=0\mid x)}=\log\frac{\pi}{1-\pi}+\log\frac{f(x\mid 1)}{f(x\mid 0)},$$
    and the training log-odds are the same expression with $\pi_{\mathrm{tr}}$ in place of $\pi$. The second term — the log-likelihood ratio, which carries every dependence on $x$ — is identical in both. Subtracting,
    $$\eta_{\text{deploy}}(x)-\eta_{\text{train}}(x)=\log\frac{\pi}{1-\pi}-\log\frac{\pi_{\mathrm{tr}}}{1-\pi_{\mathrm{tr}}},$$
    a constant free of $x$. A model fitted on the rebalanced sample therefore estimates the correct slopes and an intercept displaced by exactly that amount, and adding the offset back recovers the deployment probabilities without refitting anything.

    Two corollaries do the damage. Because the shift is a constant added to $\eta$ and the sigmoid is strictly increasing, the *ordering* of the predicted probabilities across rows is untouched — so AUC, Spearman correlation with the outcome, and any metric depending only on ranks are exactly invariant, and cannot detect the problem even in principle. Because a decision rule thresholds $p$ at a fixed cutoff, typically $0.5$, the *set of rows selected* moves by exactly the amount the intercept moved.

    The load-bearing asymmetry is that the shift is invisible to every metric computed from ranks and fully determines every metric computed from levels. **Rebalancing is not a bias-variance trade or a heuristic with unclear cost; it is a known constant added to the output, and failing to subtract it is an arithmetic omission rather than a modelling choice.**

The whole argument fits in one measurement:

```python
import numpy as np
from scipy import special, stats

rng = np.random.default_rng(13045)
n_tr, n_te, base = 40_000, 40_000, 0.58


def fit(X, y, tol=1e-11, maxit=80):
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        p = special.expit(X @ b)
        w = np.clip(p * (1 - p), 1e-12, None)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    return b


def draw(m):                    # weak signal at a 0.58 base rate, as on daily bars
    z = rng.standard_normal((m, 2))
    eta = np.log(base / (1 - base)) + 0.22 * z[:, 0] + 0.14 * z[:, 1]
    y = (rng.random(m) < special.expit(eta)).astype(float)
    return np.column_stack([np.ones(m), z]), y


def auc(y, s):
    r = stats.rankdata(s)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def calib_intercept(y, p):      # logistic fit of y on 1, with logit(p) as offset
    lo, c = np.log(p / (1 - p)), 0.0
    for _ in range(80):
        q = special.expit(lo + c)
        step = (y - q).sum() / max((q * (1 - q)).sum(), 1e-12)
        c += step
        if abs(step) < 1e-12:
            break
    return c


Xtr, ytr = draw(n_tr)
Xte, yte = draw(n_te)

pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
keep = np.concatenate([rng.choice(pos, len(neg), replace=False), neg])
b_bal, b_raw = fit(Xtr[keep], ytr[keep]), fit(Xtr, ytr)
p_tr = ytr[keep].mean()
b_fix = b_bal + np.array([np.log(base / (1 - base)) - np.log(p_tr / (1 - p_tr)), 0, 0])

print(f"  test base rate {yte.mean():.4f}; training set rebalanced to {p_tr:.4f}")
print("    model                       AUC   mean p-hat    Brier   calib b0   % p>0.5")
for name, b in (("trained on rebalanced", b_bal),
                ("+ exact intercept fix", b_fix),
                ("trained on true base rate", b_raw)):
    p = special.expit(Xte @ b)
    print(f"    {name:25s}  {auc(yte, p):7.4f}   {p.mean():10.4f}   "
          f"{np.mean((p - yte) ** 2):6.4f}   {calib_intercept(yte, p):8.4f}"
          f"   {(p > 0.5).mean():7.4f}")
# =>   test base rate 0.5792; training set rebalanced to 0.5000
#        model                       AUC   mean p-hat    Brier   calib b0   % p>0.5
#        trained on rebalanced       0.5744       0.5004   0.2458     0.3227    0.5027
#        + exact intercept fix       0.5744       0.5792   0.2396    -0.0000    0.9050
#        trained on true base rate   0.5743       0.5777   0.2396     0.0062    0.9018
```

The AUC column reads $0.5744$, $0.5744$, $0.5743$. The rebalanced model and its corrected version have *identical* discrimination to four decimals, because they differ by a constant on the log-odds scale and AUC depends only on the ordering. Every rank-based metric would say the same. The theorem's first corollary, measured.

Everything computed from levels disagrees. The mean predicted probability is $0.5004$ from the rebalanced fit against a true test base rate of $0.5792$ — the model believes the market goes up half the time when it goes up $58\%$ of the time. The calibration intercept is $0.3227$, which is $\log\frac{0.58}{0.42}-\log\frac{0.50}{0.50}=0.3228$ to the third decimal: the offset is not approximately the theoretical shift, it *is* the theoretical shift. Applying it drives the calibration intercept to $-0.0000$, the mean prediction to $0.5792$ exactly matching the base rate, and the Brier score from $0.2458$ to $0.2396$ — the same value the model trained honestly on the raw base rate achieves.

The last column is where the money is. Thresholding at $0.5$, the rebalanced model takes a position on $50.27\%$ of days and the corrected model on $90.50\%$. Same model, same ranking, same features, same AUC — and a strategy that is invested nearly twice as often. Nothing in the AUC, the accuracy, the confusion matrix at rank-based thresholds, or the coefficient table distinguishes the two.

## Ranking and Calibration Are Different Properties, Which Is Why an AUC Below One Half and the Best Net Sharpe of Four Models Are Not a Contradiction

The course's scoreboard is now readable. Its logistic baseline posts `AUC 0.478` — below a coin — alongside `net Sharpe 0.38`, the best of four models, and the lesson notes that AUC rank and Sharpe rank disagree across the board, with the forest holding the best AUC and a middling Sharpe.

These measure different things and neither dominates. AUC is the probability that a randomly chosen up-day is scored above a randomly chosen down-day: pure ordering, invariant to any monotone transform of the score, blind to whether $p=0.6$ means anything happens $60\%$ of the time. A P&L is the opposite. It is computed from a thresholded decision and a position size, so it depends on the level of $p$ relative to the cutoff, on how often the cutoff is cleared, and — after costs — on the ordering only through those. A model can rank slightly worse than chance and still make money if the rows on which it clears its threshold happen to be favourable, and the exposure column of the course's table, `days long 16%`, is the quantity doing that work.

The practical consequence is that these two properties must be repaired by different means. Bad ranking is a modelling problem and needs better features, which is what [Feature Engineering for ML](../../part-07-machine-learning/01-feature-engineering-for-ml.md) measures when it reports an AUC floor of `0.515` on this data. Bad calibration is usually an arithmetic problem and needs an intercept, as section 4 shows, or a monotone recalibration fitted on held-out data when the distortion is not constant — which is what the course reaches for when it applies isotonic regression to the boosted trees. Applying the second repair to a ranking problem changes nothing at all, since a monotone map cannot alter AUC, and that invariance is precisely why calibrating a model never improves its discrimination and never should be expected to.

**A probability that ranks well and a probability that is true are different achievements, and the metric that is quoted most often is exactly the one that cannot tell them apart.**

!!! warning "Probabilities from a rebalanced training set are wrong by a constant that no standard classification metric can see"
    The output is uniformly reassuring. The fit converges in a handful of iterations, the coefficients on every feature are correct and correctly standard-errored, the AUC is right to four decimals, accuracy and the confusion matrix at rank-based thresholds are unaffected, and the residual deviance is what it should be. The single defective number is $\beta_0$, and above it was wrong by $0.3227$ — which moved the mean predicted probability from the true $0.5792$ to $0.5004$ and the fraction of days a $0.5$ threshold would hold a position from $90.50\%$ to $50.27\%$. Every position size computed from $p$ is wrong by the corresponding factor, and the error is largest exactly where sizing matters most, near the threshold. The same failure arrives without any rebalancing whenever the deployment base rate drifts from the training one, which on financial data it always does. **The free diagnostic is one comparison the fit already contains: check the mean predicted probability against the realized base rate of the evaluation sample, and if they differ then the intercept is wrong by exactly $\log\frac{\bar y}{1-\bar y}-\log\frac{\bar p}{1-\bar p}$ — add that constant to $\beta_0$ and nothing else changes, since a reliability curve binned on $p$ will confirm the shift is flat across bins rather than a distortion of shape.**

## A Logistic Fit Is a Statement About Log-Odds, and Every Common Way of Reporting It Discards Either the Calibration or the Ranking

This page established that the logit link makes each coefficient an additive log-odds contribution, so a linear predictor is one Bayesian update per feature and $e^{\beta_j}$ is an odds ratio rather than a probability change; that the log-likelihood is strictly concave under full column rank but its supremum is unattained exactly when the classes are separable, in which case the standard error diverges faster than the coefficient and the Wald p-value of a perfect predictor climbed to $0.9861$ at a fitted $b_1=1194.08$ with $\widehat{\operatorname{se}}=68532.56$, where a ridge of $10^{-3}$ restored a finite $47.27$ while moving the converged rows by at most $0.09$; that the estimator is inflated at order $p/n$, averaging $1.485$ times the truth at $n=50$ and $p=10$ against $1.006$ at $n=1000$ and $p=2$, while the Wald size on a null coefficient held between $0.0453$ and $0.0584$ throughout; and that a rebalanced training set displaces the intercept by exactly $\log\frac{\pi}{1-\pi}-\log\frac{\pi_{\mathrm{tr}}}{1-\pi_{\mathrm{tr}}}$, measured at $0.3227$ against a theoretical $0.3228$, leaving AUC identical at $0.5744$ while moving the invested fraction from $50.27\%$ to $90.50\%$.

The pattern across all three failures is that this model's output is a *number on the log-odds scale*, and every convenient summary discards part of it. Report an odds ratio and the base rate is gone. Report an AUC and the levels are gone, so a constant error of any size survives unnoticed. Report accuracy at a fixed threshold and both the levels and most of the ordering are gone. Report a Wald p-value and, in the one regime where the evidence is overwhelming, the answer inverts. The fitted $\beta$ contains everything; the reporting conventions each keep a projection of it, and the projections are chosen by convention rather than by what the decision needs.

For a trading application the decision needs levels, because a position size is a function of $p$ and not of $p$'s rank among other days. That is the sense in which the course's logistic baseline beating three ensembles is not a fluke of an unlucky AUC: the metric that ranked it fourth was measuring the property the strategy does not use. What none of this addresses is the coefficient inflation of section 3, whose repair is the same object that rescued the separated fit — a penalty. That is [Regularization](05-regularization.md).

**Logistic regression estimates a log-odds function exactly and honestly, and then hands it to an evaluation culture that quotes ranks when the decision needs levels and levels when the model was fitted on a different prior.**
