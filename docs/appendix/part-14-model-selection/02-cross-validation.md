# Cross-Validation

[Bias–Variance Tradeoff](01-bias-variance-tradeoff.md) ended by needing a curve it could not draw: the U-shape was measured with the truth in hand, and a desk has no truth. Cross-validation is the standard answer, and it is usually described as estimating out-of-sample error. It does estimate an out-of-sample error — the difficulty is that it is the error of a model nobody is going to ship, reported with an error bar roughly half the width of the real one, on a partition whose validity is an assumption about the data rather than a property of the method. Below, five-fold cross-validation is unbiased to within simulation noise for a model trained on eighty observations while the model actually shipped was trained on a hundred, and two-fold overstates that model's error by $34.4\%$; the standard error every library prints alongside the score is $0.51$ of the estimate's true spread at $K=2$ and never exceeds $0.88$ of it; and a five-fold split of the wrong kind manufactures a correlation of $+0.5023$ between features and labels that were constructed to be independent.

This page covers what $K$-fold is and is not unbiased for, the leave-one-out shortcut that costs one fit instead of $n$ together with the class of estimators it is exact for, the correlation among fold errors that makes the reported standard error too small, leakage as a function of how much boundary a split creates, and the optimism of a cross-validation score reported for a model that cross-validation chose. It does not decompose prediction error into bias, variance and a floor, which is [Bias–Variance Tradeoff](01-bias-variance-tradeoff.md); it estimates no optimism analytically from a likelihood, which is [Information Criteria (AIC/BIC)](03-information-criteria.md); it searches over no candidate predictors, which is [Feature Selection](04-feature-selection.md); it combines no models, which is [Model Averaging](05-model-averaging.md); it constructs no resampling scheme from first principles, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it derives no shrinkage geometry, which is [Regularization](../part-13-regression/05-regularization.md); it charges nothing for the number of candidates a search examined, which is [Part XV](../part-15-multiple-testing/index.md); and it never reports the score of a model that the same scores selected.

The trading stake is the closing lesson of the strategy part, which prices hindsight in Sharpe ratios. [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) reports "picking the best lookback in hindsight: 480 days, Sharpe 0.56" against "walk-forward (best lookback known only up to each year): Sharpe 0.34", and separately reports a cross-validated information coefficient of "shuffled 5-fold (the library default habit): +0.061" against "contiguous 5-fold: −0.087" on features that contain nothing by construction. Sections 4 and 5 are the machinery under both numbers: the first gap is what section 5 charges for, the second is what section 4 measures, and the lesson's own summary of the second — that "leakage is proportional to boundary-days over total days" — is a claim this page can put on a grid.

## $K$-Fold Is Unbiased for the Error of a Model Trained on $n(1-1/K)$ Observations, Which Is Not the Model Anyone Ships

Every fold trains on a strict subset. That is not a defect to be minimised but the definition of the procedure, and it fixes what the procedure estimates.

??? note "Proof that the $K$-fold estimate targets the expected error at training size $n-n/K$, so its bias against the shipped model is the learning curve's rise over that gap"

    Let $\mathrm{Err}(m)=\mathbb{E}_{\mathcal{D}_m}\mathbb{E}_{(x,Y)}[(Y-\hat f_{\mathcal{D}_m}(x))^{2}]$ be the expected prediction error of the fitting procedure trained on $m$ independent observations, the outer expectation running over training sets of that size. In $K$-fold cross-validation, fold $k$ fits on $n-n/K$ observations and scores on the held-out $n/K$, which are independent of them, so by the argument of [Bias–Variance Tradeoff](01-bias-variance-tradeoff.md) each fold's mean squared error has expectation exactly $\mathrm{Err}(n-n/K)$. Averaging over $k$ preserves the expectation, giving
    $$\mathbb{E}[\widehat{\mathrm{CV}}_K]=\mathrm{Err}\!\left(n-\tfrac{n}{K}\right),$$
    with no approximation anywhere. The quantity a practitioner wants is $\mathrm{Err}(n)$, the error of the model refitted on everything, so the bias is $\mathrm{Err}(n-n/K)-\mathrm{Err}(n)$ — the height the learning curve falls across the last $n/K$ observations.

    This has two consequences that pull in opposite directions and are usually collapsed into a single sentence about a bias–variance tradeoff in $K$. Raising $K$ shrinks the training-size gap and so shrinks the bias, monotonically, to essentially nothing at $K=n$. It also makes the $K$ training sets more nearly identical to one another, which is section 3's problem. **The bias in $K$ is not a mystery to be traded off by feel: it is the learning curve evaluated at two points, so it is large exactly when the sample is small relative to the model, and negligible when it is not.**

    The load-bearing detail is that the learning curve is steep precisely in the regime where cross-validation is reached for. **A procedure whose bias is the slope of the learning curve is least trustworthy on small samples and richly parameterized models, which is the situation that motivated resampling in the first place.**

The prediction is a table in which the cross-validation column matches the true error at its own training size and misses the shipped model's:

```python
import numpy as np

rng = np.random.default_rng(14021)
n, p, sig, reps = 100, 20, 1.0, 4000
beta = np.ones(p) / np.sqrt(p)


def draw(m):
    X = rng.standard_normal((m, p))
    return X, X @ beta + rng.normal(0.0, sig, m)


def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


Xv, yv = draw(50_000)                              # one held-out world for every row


def true_err(m):                                   # error of a model trained on m
    out = np.empty(reps)
    for r in range(reps):
        X, y = draw(m)
        out[r] = np.mean((yv - Xv @ fit(X, y)) ** 2)
    return out.mean()


print(f"  K-fold on n = {n}, p = {p}, sigma = {sig:.1f}, {reps:,} datasets")
print("    K    trains on   CV estimate   true err at that size   true err at n=100"
      "   overstatement")
shipped = true_err(n)
spread = {}
for K in (2, 5, 10, n):
    m = n - n // K
    cv, naive = np.empty(reps), np.empty(reps)
    for r in range(reps):
        X, y = draw(n)
        idx = rng.permutation(n)
        se = []
        for k in range(K):
            te = idx[k * (n // K):(k + 1) * (n // K)]
            tr = np.setdiff1d(idx, te)
            se.append((y[te] - X[te] @ fit(X[tr], y[tr])) ** 2)
        cv[r] = np.mean(np.concatenate(se))
        fold = np.array([s.mean() for s in se])
        naive[r] = fold.std(ddof=1) / np.sqrt(K)      # the SE every library reports
    spread[K] = (naive.mean(), cv.std())
    t = true_err(m)
    tag = "LOO" if K == n else f"{K}"
    print(f"    {tag:4s} {m:9d}   {cv.mean():11.4f}   {t:21.4f}   {shipped:17.4f}"
          f"   {cv.mean() / shipped - 1:+13.1%}")

print("    the error bar the folds report against the one the estimate actually has")
print("      K            2       5      10     LOO")
print("      naive SE" + "".join(f"{spread[K][0]:8.4f}" for K in (2, 5, 10, n)))
print("      true sd " + "".join(f"{spread[K][1]:8.4f}" for K in (2, 5, 10, n)))
print("      ratio   " + "".join(f"{spread[K][0] / spread[K][1]:8.2f}"
                                 for K in (2, 5, 10, n)))
# =>   K-fold on n = 100, p = 20, sigma = 1.0, 4,000 datasets
#        K    trains on   CV estimate   true err at that size   true err at n=100   overstatement
#        2           50        1.6883                  1.6936              1.2559          +34.4%
#        5           80        1.3371                  1.3407              1.2559           +6.5%
#        10          90        1.2928                  1.2878              1.2559           +2.9%
#        LOO         99        1.2508                  1.2577              1.2559           -0.4%
#        the error bar the folds report against the one the estimate actually has
#          K            2       5      10     LOO
#          naive SE  0.1885  0.1728  0.1746  0.1745
#          true sd   0.3731  0.2366  0.2156  0.1985
#          ratio       0.51    0.73    0.81    0.88
```

Read the third and fourth columns together, because their agreement is the proof discharged as arithmetic. Two-fold cross-validation returns $1.6883$ and the true error of a model trained on fifty observations is $1.6936$; five-fold returns $1.3371$ against $1.3407$; ten-fold $1.2928$ against $1.2878$; leave-one-out $1.2508$ against $1.2577$. The procedure is unbiased, to within simulation noise, for exactly the quantity the proof names.

The fifth column is what anyone actually wanted, and it does not move: $1.2559$ in every row, because the shipped model is the same model however it was scored. The last column is the gap — $34.4\%$, $6.5\%$, $2.9\%$ and $-0.4\%$, the last being zero within noise. A pessimistic estimate sounds like the safe kind of error, and for a single model it is; it stops being safe the moment two models are compared, because a richly parameterized candidate has a steeper learning curve and is therefore charged more for the same withheld data. **A comparison run at small $K$ is tilted toward the simpler candidate by an amount that has nothing to do with which one predicts better at full sample size.**

The second block prices the error bar. The naive standard error — spread across folds over $\sqrt{K}$, which is what a library reports and what a `±` in a paper almost always means — reads $0.1885$, $0.1728$, $0.1746$ and $0.1745$, essentially flat in $K$, against a true standard deviation across independent datasets of $0.3731$, $0.2366$, $0.2156$ and $0.1985$. The ratios are $0.51$, $0.73$, $0.81$ and $0.88$: at $K=2$ the published uncertainty is half the real uncertainty, and it is too small at every $K$. Section 3 is why the understatement is structural rather than a small-sample artifact.

## Leave-One-Out Costs One Fit Rather Than $n$, and the Shortcut Is Exact for a Narrower Class Than It Is Applied To

Leave-one-out sits at the good end of section 1's bias column and looks like the expensive option, since the definition calls for $n$ refits. For a large family of estimators it needs one, and the reason is a quantity [Model Diagnostics](../part-13-regression/06-model-diagnostics.md) already derived for a different purpose.

??? note "Proof that the deleted residual is $e_i/(1-h_{ii})$, so the entire leave-one-out score is available from a single fit"

    [Model Diagnostics](../part-13-regression/06-model-diagnostics.md) establishes by Sherman–Morrison that deleting observation $i$ from a least-squares fit moves the coefficient vector by a rank-one update, and that the fitted value at the deleted point satisfies
    $$\hat y_i^{(-i)}=\frac{\hat y_i-h_{ii}y_i}{1-h_{ii}},$$
    where $h_{ii}$ is the $i$th diagonal of the hat matrix. Subtracting from $y_i$ and simplifying gives the deleted residual as the ordinary residual inflated by the point's own leverage,
    $$y_i-\hat y_i^{(-i)}=\frac{y_i-\hat y_i}{1-h_{ii}}=\frac{e_i}{1-h_{ii}},$$
    so the leave-one-out score $\mathrm{PRESS}/n=\frac{1}{n}\sum_i e_i^{2}/(1-h_{ii})^{2}$ is a function of one fit's residuals and one fit's hat diagonal. Replacing each $h_{ii}$ by their common average $\operatorname{tr}(H)/n$ gives generalized cross-validation, $\mathrm{GCV}=\frac{1}{n}\sum_i e_i^{2}/(1-\operatorname{tr}(H)/n)^{2}$, which is rotation-invariant and cheaper still.

    The identity holds for any estimator whose fitted values are $Sy$ with $S$ *fixed independently of $y$* and whose deletion behaviour is governed by the same rank-one algebra — least squares and ridge at fixed $\lambda$ qualify, and so do smoothing splines. A $k$-nearest-neighbour rule does not, and the reason is instructive: its $S$ depends on which points are present, so deleting observation $i$ does not merely remove a row, it promotes the $(k+1)$th neighbour into the average. The shortcut is not an approximation there; it is an identity for a different estimator.

    The load-bearing condition is that $S$ must not react to the deletion. **The leave-one-out shortcut is exact where the smoother is fixed and silently wrong where the smoother is adaptive, and adaptivity is the property most modern estimators are chosen for.**

Both halves of that claim are measurable on one dataset, and the shortcut then makes penalty selection nearly free:

```python
import numpy as np

rng = np.random.default_rng(14023)
n, p, sig = 80, 6, 1.0

Z = rng.standard_normal((n, p - 1))
Z[:, 1:] = 0.92 * Z[:, [0]] + np.sqrt(1 - 0.92**2) * Z[:, 1:]   # collinear design
X = np.column_stack([np.ones(n), Z])
y = X @ np.arange(1.0, p + 1.0) / p + rng.normal(0.0, sig, n)
P = np.diag([0.0] + [1.0] * (p - 1))                        # intercept unpenalised


def hat(kind, par):
    if kind == "ridge":
        return X @ np.linalg.solve(X.T @ X + par * P, X.T)
    S = np.zeros((n, n))                                    # k-nearest-neighbour
    d = np.abs(X[:, 1][:, None] - X[:, 1][None, :])
    for i in range(n):
        S[i, np.argsort(d[i])[: int(par)]] = 1.0 / int(par)
    return S


print(f"  leave-one-out by refitting versus by algebra, n = {n}, p = {p}")
print("    smoother       brute-force LOO   shortcut PRESS/n        GCV"
      "   max |diff|   fits")
for kind, par, tag in (("ridge", 0.0, "OLS"), ("ridge", 0.5, "ridge L=0.5"),
                       ("ridge", 20.0, "ridge L=20"), ("knn", 5, "5-nn")):
    H = hat(kind, par)
    h = np.diag(H)
    e = y - H @ y
    short = e / (1.0 - h)                                   # deleted residual
    brute = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        if kind == "ridge":
            b = np.linalg.solve(X[keep].T @ X[keep] + par * P, X[keep].T @ y[keep])
            brute[i] = y[i] - X[i] @ b
        else:
            d = np.abs(X[keep, 1] - X[i, 1])
            brute[i] = y[i] - y[keep][np.argsort(d)[: int(par)]].mean()
    gcv = np.mean(e**2) / (1.0 - np.trace(H) / n) ** 2
    print(f"    {tag:13s}  {np.mean(brute**2):15.6f}   {np.mean(short**2):16.6f}"
          f"   {gcv:8.6f}   {np.abs(brute - short).max():10.2e}"
          f"   {'n' if kind == 'knn' else '1'}")

print("    choosing the ridge penalty by leave-one-out, one fit per candidate")
grid = np.geomspace(1e-2, 1e3, 9)
press = [np.mean(((y - hat("ridge", L) @ y)
                  / (1.0 - np.diag(hat("ridge", L)))) ** 2) for L in grid]
j = int(np.argmin(press))
print("      lambda  " + "".join(f"{v:8.3g}" for v in grid))
print("      LOO MSE " + "".join(f"{v:8.4f}" for v in press))
print(f"      minimising lambda {grid[j]:.3g}, LOO MSE {press[j]:.4f} against "
      f"OLS {press[0]:.4f}, tr(H) {np.trace(hat('ridge', grid[j])):.3f} of p = {p}")
# =>   leave-one-out by refitting versus by algebra, n = 80, p = 6
#        smoother       brute-force LOO   shortcut PRESS/n        GCV   max |diff|   fits
#        OLS                   1.233856           1.233856   1.240756     1.25e-14   1
#        ridge L=0.5           1.225136           1.225136   1.230724     1.07e-14   1
#        ridge L=20            1.265921           1.265921   1.272038     3.11e-15   1
#        5-nn                  1.571273           1.562266   1.562266     6.34e-01   n
#        choosing the ridge penalty by leave-one-out, one fit per candidate
#          lambda      0.01  0.0422   0.178    0.75    3.16    13.3    56.2     237   1e+03
#          LOO MSE   1.2336  1.2329  1.2301  1.2224  1.2150  1.2410  1.4590  2.8361  5.8079
#          minimising lambda 3.16, LOO MSE 1.2150 against OLS 1.2336, tr(H) 4.633 of p = 6
```

The first three rows are the identity, not an approximation to it. Brute-force refitting and the algebraic shortcut agree to $1.25\times10^{-14}$, $1.07\times10^{-14}$ and $3.11\times10^{-15}$ — machine precision — at a cost of one fit against eighty. Generalized cross-validation, which replaces every $h_{ii}$ by their average, lands at $1.240756$, $1.230724$ and $1.272038$ against exact values of $1.233856$, $1.225136$ and $1.265921$: within $0.6\%$, and cheaper again since it needs a trace rather than a diagonal.

The fourth row is the boundary of the class. For a five-nearest-neighbour rule the shortcut and the truth differ by $6.34\times10^{-1}$ at the worst point, and the *score* differs in the direction that matters: the shortcut reports $1.562266$ where the honest answer is $1.571273$. It is optimistic, because it charges the deleted point only for its own $1/k$ share of the average and never for the inferior neighbour that replaces it. The error is small here and it is not small in general, and nothing in the arithmetic announces that the estimator has left the class.

The second block spends the shortcut on the job [Regularization](../part-13-regression/05-regularization.md) explicitly deferred to this page. Nine penalties, one fit each, and the leave-one-out score traces a genuine minimum: $1.2336$ at $\lambda=0.01$, falling through $1.2301$ and $1.2224$ to $1.2150$ at $\lambda=3.16$, then rising through $1.2410$ and $1.4590$ to $5.8079$. The chosen penalty beats unpenalized least squares by $1.5\%$ and does so by spending $\operatorname{tr}(H)=4.633$ of a possible $6$ degrees of freedom — the fractional complexity of [Bias–Variance Tradeoff](01-bias-variance-tradeoff.md), now selected by data rather than assumed.

**The one quantity that made every deletion diagnostic free in Part XIII makes the entire leave-one-out curve free here, and both facts are the same line of algebra pointed at different questions.**

## The Folds Share Their Training Data, So the Standard Error They Report Is Too Small and No Correction Fixes It

Section 1 measured a reported standard error at $0.51$ of the true one. That is not a small-sample accident and it is not repaired by more folds; it is a consequence of what the folds are.

??? note "Proof that the variance of the $K$-fold estimate involves three distinct covariance parameters, only one of which the folds identify, so no unbiased estimator of it exists"

    Write the estimate as $\widehat{\mathrm{CV}}=\frac{1}{n}\sum_i \ell_i$ where $\ell_i$ is the loss at observation $i$ scored by the model fitted without $i$'s fold. Then
    $$\operatorname{var}(\widehat{\mathrm{CV}})=\frac{1}{n}\theta+\frac{m-1}{n}\omega+\frac{n-m}{n}\gamma,$$
    with $\theta=\operatorname{var}(\ell_i)$, $\omega$ the covariance of two losses scored *within* the same fold, $m=n/K$ the fold size, and $\gamma$ the covariance of two losses scored in *different* folds. All three are nonzero in general: $\omega$ because two test points in one fold are scored by the identical fitted model, and $\gamma$ because any two folds' training sets overlap in $n(1-2/K)$ observations, so even models from different folds are correlated through the data they share.

    The naive estimator uses the spread across the $K$ fold means and so estimates a combination in which $\gamma$ appears with the wrong sign — it treats the folds as if they were independent replicates, which would require $\gamma=0$. Since $\gamma>0$ whenever the training sets overlap, the naive quantity is biased downward, and the bias grows with $K$ because the overlap does. There is no repair by rescaling: the fold-level statistics supply $K$ numbers, and $\theta$, $\omega$ and $\gamma$ are three free parameters whose separate identification requires replications of the *entire* cross-validation on independent datasets — which is precisely what a practitioner does not have.

    The load-bearing structure is the overlap, and note that it is worst exactly where the bias of section 1 is best. Leave-one-out has $n$ training sets differing in one observation apiece, so $\gamma$ is nearly $\theta$. **Raising $K$ trades a bias you can bound for a variance you cannot estimate, which is why the honest reason for choosing five or ten folds is that the failure mode is milder there and not that some optimum was computed.**

Two practical consequences follow, and both are visible in section 1's ratio row of $0.51$, $0.73$, $0.81$ and $0.88$. The first is that a `±` printed next to a cross-validation score should be read as a lower bound on the uncertainty, not an estimate of it, and at small $K$ a badly wrong one. The second is that the popular *one-standard-error rule* — choose the simplest model within one standard error of the best — is calibrated against a standard error that is too small, so it selects a model less simple than its own logic intends. The rule remains useful because it is a heuristic in the right direction, and a practitioner who believes it is delivering a defined confidence level is mistaken about the object it is built from.

## Random Folds on Dependent Data Estimate the Error of a Model That Read the Answers, and the Leak Scales With Boundary

Sections 1 to 3 all assumed the held-out observations were independent of the training ones. Financial data violates that in two ways at once: labels built from forward windows overlap each other, and features built from trailing windows are autocorrelated. The following construction has *no relationship whatsoever* between features and labels, so every non-zero number below is manufactured:

```python
import numpy as np

rng = np.random.default_rng(14025)
T, h, k, reps, phi = 1000, 21, 25, 300, 0.97


def path():
    """Slow features and an overlapping forward label, independent by construction."""
    X = np.empty((T, 3))
    X[0] = rng.standard_normal(3)
    for t in range(1, T):
        X[t] = phi * X[t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(3)
    e = rng.standard_normal(T + h)
    y = np.array([e[t:t + h].sum() for t in range(T)])      # labels overlap h-1 days
    return X, y


def knn(Xtr, ytr, Xte):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    return ytr[np.argsort(d, axis=1)[:, :k]].mean(1)


def score(X, y, K, shuffle, purge, emb):
    pred, act = [], []
    edges = np.linspace(0, T, K + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if shuffle:                                     # blocks of a permutation
            perm = rng.permutation(T)
            te, tr = perm[a:b], np.setdiff1d(perm, perm[a:b])
        else:
            drop = np.zeros(T, bool)
            drop[max(0, a - h + 1) if purge else a:min(T, b + h) if purge else b] = True
            drop[b:min(T, b + emb)] = True
            te, tr = np.arange(a, b), np.flatnonzero(~drop)
        pred.append(knn(X[tr], y[tr], X[te]))
        act.append(y[te])
    return np.corrcoef(np.concatenate(pred), np.concatenate(act))[0, 1]


print(f"  {reps} independent paths, T = {T}, {h}-day overlapping labels, {k}-nn;")
print("  the features are independent of the labels by construction, so any"
      " signal is leakage")
print("    scheme                          5 folds, 8 boundaries   50 folds,"
      " 98 boundaries")
for tag, sh, pu, em in (("shuffled (every day a boundary)", True, False, 0),
                        ("contiguous", False, False, 0),
                        ("contiguous + 21-day purge", False, True, 0),
                        ("contiguous + purge + 21-day embargo", False, True, 21)):
    out = []
    for K in (5, 50):
        v = np.array([score(*path(), K, sh, pu, em) for _ in range(reps)])
        out.append(f"{v.mean():+10.4f} ({v.std():.3f})")
    print(f"    {tag:34s}  {out[0]:>19s}  {out[1]:>19s}")

print("    leakage against the number of contiguous folds, no purge, no embargo")
print("      folds       5      10      25      50     100")
row = [np.mean([score(*path(), K, False, False, 0) for _ in range(reps)])
       for K in (5, 10, 25, 50, 100)]
print("      mean IC" + "".join(f"{v:+8.4f}" for v in row))
print("      boundaries" + "".join(f"{2 * K - 2:8d}" for K in (5, 10, 25, 50, 100)))
# =>   300 independent paths, T = 1000, 21-day overlapping labels, 25-nn;
#      the features are independent of the labels by construction, so any signal is leakage
#        scheme                          5 folds, 8 boundaries   50 folds, 98 boundaries
#        shuffled (every day a boundary)         +0.5023 (0.070)      +0.5173 (0.066)
#        contiguous                              -0.0469 (0.133)      +0.1040 (0.122)
#        contiguous + 21-day purge               -0.0580 (0.137)      -0.0439 (0.121)
#        contiguous + purge + 21-day embargo      -0.0405 (0.136)      -0.0463 (0.137)
#        leakage against the number of contiguous folds, no purge, no embargo
#          folds       5      10      25      50     100
#          mean IC -0.0592 -0.0291 +0.0455 +0.1129 +0.2336
#          boundaries       8      18      48      98     198
```

The honest answer is a correlation of zero, and the honest *schemes* return a small negative number instead — around $-0.04$ to $-0.06$ — because an out-of-fold prediction borrowed from a distant block targets the wrong local level of a strongly autocorrelated label. That is the baseline the rest is read against, exactly as the course lesson read its own $-0.087$.

The first row is the standard library default and it is a catastrophe: $+0.5023$ at five folds and $+0.5173$ at fifty — a correlation of one half between a feature matrix and a label vector generated from independent random streams. Shuffling makes every observation adjacent to test data, so the twenty-one-day label overlap and the slow features together let the model retrieve neighbours whose answers it has already seen; the fold count barely moves it, because once every day is a boundary day there is nothing left to make worse.

The second row is the finding the usual advice hides. Contiguous folds are *safe at five* — $-0.0469$, indistinguishable from the honest baseline — and *not safe at fifty*, where they read $+0.1040$. Nothing changed but the number of blocks and therefore of boundaries, $8$ to $98$. "Use contiguous folds" is sufficient only in the regime where blocks are long relative to the label horizon. Rows three and four repair exactly that: purging drives the fifty-fold case from $+0.1040$ back to $-0.0439$, onto the baseline, while at five folds it moves $-0.0469$ to $-0.0580$, a third-decimal adjustment to a number that was already right. The embargo's $-0.0405$ and $-0.0463$ are within noise of the purge alone, because at this horizon the overlap the purge removes is the whole of the dependence.

The final block states the geometry as a measurement. Across $5$, $10$, $25$, $50$ and $100$ contiguous folds the leak runs $-0.0592$, $-0.0291$, $+0.0455$, $+0.1129$ and $+0.2336$ against boundary counts of $8$, $18$, $48$, $98$ and $198$ — monotone in the boundary count, crossing from honest to contaminated between ten and twenty-five folds, and reaching $+0.2336$ before shuffling's $+0.5023$ completes the curve at the limit where every day is a boundary.

**The dangerous parameter of a cross-validation scheme is not the estimator, the fold count or the metric but the total length of the seam between training and test data, and a split's validity is therefore a property of the data's dependence structure rather than of the splitting method's name.**

## A Cross-Validation Score Reported for the Model That Cross-Validation Chose Has Been Spent Twice

Everything above scores a *fixed* procedure. The moment cross-validation is used to pick among candidates — a penalty, a fold count, a feature set, a lookback — the winning score stops being an estimate of that model's error and becomes the maximum of a collection of noisy estimates, which is biased upward as an estimate of the best candidate's quality by an amount that grows with the number of candidates.

The repair is a nested loop: an outer split that is untouched by tuning, and an inner cross-validation run separately inside each outer training set, so that the outer score measures the whole procedure — search included — rather than its winner. The cost is real, since the outer folds train on less data and therefore inherit section 1's pessimism on top of their own, and the benefit is that the number reported describes something that could actually be run in production.

The size of the effect is not a footnote. [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) measures a hindsight Sharpe of $0.56$ against a walk-forward $0.34$ for the same family of rules, and separately reports "PBO = 0.60: the research process 'pick the grid's best Sharpe' delivers a strategy in the out-of-sample *bottom half* of its own family sixty percent of the time". That is the selection step, priced. Quantifying it in general requires counting the candidates and knowing their dependence, which is [White's Reality Check](../part-15-multiple-testing/05-whites-reality-check.md); what belongs here is the structural point that the inner scores were consumed by the search and cannot also serve as the report.

!!! note "$K$-fold, leave-one-out, the jackknife, a holdout and walk-forward are five names for deleting data and refitting, and only one of them respects the arrow of time"
    The mechanics are nearly identical and the questions are not. **$K$-fold** partitions once and rotates the held-out block, targeting $\mathrm{Err}(n-n/K)$ as section 1 proves. **Leave-one-out** is its $K=n$ limit, nearly unbiased for $\mathrm{Err}(n)$ and, for a fixed linear smoother, free by section 2's algebra. **The jackknife** of [Jackknife Methods](../part-09-monte-carlo-methods/08-jackknife-methods.md) performs the same deletion but feeds the refits to a different formula — it estimates the *bias and variance of an estimator*, not a prediction error, so the two procedures share a loop and share no target. **A holdout** deletes once and never rotates, which makes it unbiased for $\mathrm{Err}(n-m)$ at a variance nothing averages down. **Walk-forward** deletes only the future, so every training set is a prefix and no fitted model has seen an observation later than the one it scores; it is the only member of the list whose splits could have been executed by a process running in real time, and it pays for that with the least data per fit and no reuse of the early sample. The five are interchangeable on independent rows and are not interchangeable on a time series, where the first two are the ones section 4 shows manufacturing correlations of $+0.5023$.

!!! warning "Nothing in a cross-validation score records which of the four assumptions behind it were met, and three of them are properties of the data rather than of the code"
    The number arrives with no field for any of it. Section 1's target is a model trained on $n-n/K$ observations and the report will be read as describing the shipped model, a gap measured at $34.4\%$ for $K=2$. Section 3's error bar is understated by a factor the fold statistics cannot recover, measured at $0.51$ at $K=2$ and never better than $0.88$. Section 4's independence assumption is the one that fails silently and largest: the same code, the same estimator and the same metric returned $-0.0469$ and $+0.5023$ on identical data depending only on how the indices were partitioned, and the contaminated run is the one that looks like a discovery. Section 5's selection bias then attaches to whichever configuration won. A practitioner inheriting a notebook sees one float. **The free diagnostic is to run the whole pipeline once on labels that have been circularly shifted by more than the label horizon — a shift destroys any real relationship between features and labels while preserving every autocorrelation, overlap and fold boundary in the data — and to read the resulting score as the pipeline's own zero point: if a shifted run returns something near the honest baseline then the split is sound and the unshifted score can be taken at face value, and if it returns $+0.50$ where it should return $-0.05$ then the number your model produced was manufactured by the partition and no amount of tuning will separate it from the part that is real.**

## Cross-Validation Answers the Question It Was Asked, and the Question Contains the Split

This page established that $K$-fold is exactly unbiased for the prediction error of a model trained on $n-n/K$ observations, matching the true error at that size at $1.6883$ against $1.6936$, $1.3371$ against $1.3407$, $1.2928$ against $1.2878$ and $1.2508$ against $1.2577$, while overstating the shipped model's $1.2559$ by $34.4\%$, $6.5\%$, $2.9\%$ and $-0.4\%$ — a pessimism that is not neutral between candidates because it charges the richer model more; that the deleted residual $e_i/(1-h_{ii})$ reproduces brute-force leave-one-out to $1.25\times10^{-14}$ for one fit against eighty, with generalized cross-validation within $0.6\%$ at $1.240756$ against $1.233856$, and that the shortcut leaves the class silently for an adaptive smoother, reporting $1.562266$ where a five-nearest-neighbour rule's honest score is $1.571273$; that the shortcut makes penalty selection nearly free, tracing a minimum at $\lambda=3.16$ with a score of $1.2150$ against least squares' $1.2336$ at $\operatorname{tr}(H)=4.633$ of a possible $6$; that the folds' shared training data puts three covariance parameters into the estimate's variance where the folds identify one, so the reported standard error ran $0.51$, $0.73$, $0.81$ and $0.88$ of the truth and no rescaling repairs it; and that leakage is a function of seam length, with shuffled folds manufacturing $+0.5023$ from independent series, contiguous folds safe at $-0.0469$ with eight boundaries and contaminated at $+0.1040$ with ninety-eight, purging restoring $-0.0439$, and the leak running $-0.0592$, $-0.0291$, $+0.0455$, $+0.1129$, $+0.2336$ monotonically in the boundary count.

What the page has not produced is a defence. Cross-validation is the correct tool and every number above is a property of the correct tool used correctly; the failures are all failures of reading, in which a quantity conditional on a training size, a partition and a search is reported as though it were conditional on none of them. That pattern is the one [Part XIII](../part-13-regression/index.md) closed on, arriving here through a different door.

There is also a cost that no amount of care removes. Every fold refits, so a nine-point penalty grid on a ten-fold scheme is ninety fits, and a nested loop multiplies that again — and section 2's shortcut, which eliminated the cost outright, applied only to smoothers whose $S$ ignores the data. The alternative is to stop resampling and estimate the optimism analytically: the gap between training and test error has an expectation, section 3 of [Bias–Variance Tradeoff](01-bias-variance-tradeoff.md) already wrote it as $2\sigma^{2}\operatorname{tr}(S)/n$ for a linear smoother, and a general version of that quantity would price complexity from a single fit with no splitting at all. That is [Information Criteria (AIC/BIC)](03-information-criteria.md), and the price of its convenience is a set of assumptions that a resampling scheme never had to make.

**Cross-validation converts an unanswerable question about a population into an answerable one about a partition, and every property of the partition — how much data each fold withholds, how much the folds share, and how long the seam between them runs — is inherited by the answer without appearing anywhere in it.**
