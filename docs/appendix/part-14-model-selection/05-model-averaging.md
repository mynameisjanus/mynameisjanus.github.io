# Model Averaging

[Feature Selection](04-feature-selection.md) ended on a procedure that could not be trusted because it took a maximum, and the obvious escape is to stop maximising: keep every candidate and average them. The escape works, and it comes with a guarantee stronger than anything else in this part — an exact identity, holding on every dataset with no assumptions at all, saying that an average is never worse than its members' average error. What the identity does not say is that the average beats the *best* member, and the gap between those two statements is where the subject lives. Below, an ensemble of fifty stepwise fits improves on a single one by $18.8\%$ while the identical bagging operation applied to least squares makes it $27.4\%$ *worse*. Weights fitted on the data the members were fitted on put $1.0000$ of the weight on the one member that memorised the training labels, scoring $1.7728$ where equal weights score $0.7471$. And as the number of members grows, the value of optimal weights rises — the oracle improves from $0.7490$ to $0.5024$ — while the value of *estimated* optimal weights turns around and loses to $1/N$ by $18.1\%$.

This page covers the ambiguity identity and what it does and does not promise, bagging as variance reduction that pays only for unstable estimators, stacking weights and why they must be learned on data the members did not see, the estimation of combination weights as the same problem that defeated portfolio optimization, and the M-closed assumption that Bayesian model averaging needs and finance never supplies. It does not decompose prediction error, which is [Bias–Variance Tradeoff](01-bias-variance-tradeoff.md); it establishes no property of a resampling scheme, which is [Cross-Validation](02-cross-validation.md); it derives no penalized-likelihood score, which is [Information Criteria (AIC/BIC)](03-information-criteria.md); it runs no search over predictors, which is [Feature Selection](04-feature-selection.md); it constructs no bootstrap from first principles, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it computes no Bayes factor or posterior model probability, which is [Part XVI](../part-16-bayesian-statistics/index.md); it charges nothing for the size of the search that produced the candidates, which is [Part XV](../part-15-multiple-testing/index.md); and it never presents an ensemble as evidence that its members were any good.

The trading stake is the course's most comprehensively negative result, and it is this page's fourth section with different labels. [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) reports mean-variance optimization scoring a Sharpe of $0.377$ against naive equal weighting's $0.450$, the Ledoit–Wolf shrinkage repair making it *worse* at $0.115$, and concludes that "what does rescue the optimizer is the crudest tool available, a set of constraints with no statistical theory behind them at all." Optimal combination weights are $\Sigma^{-1}\mathbf{1}$ normalized, whether $\Sigma$ holds asset returns or member forecast errors, so sections 3 and 4 reproduce all three findings — the loss to equal weighting, the crossover point, and the constraint that rescues it — on data with no market in it.

## An Average's Error Is Its Members' Mean Error Minus Their Mean Disagreement, Exactly and Always

The case for combining is usually made statistically, as a claim about variance reduction under independence. There is a stronger and simpler statement available, and it is an algebraic identity rather than a probabilistic argument.

??? note "Proof that for any ensemble, any dataset and any target, the squared error of the average equals the mean squared error of the members minus their mean squared spread about the average"

    Let $f_1,\dots,f_M$ be arbitrary predictions at a point, $\bar f=\frac{1}{M}\sum_m f_m$, and $y$ any target. Expand the member error about the ensemble:
    $$\frac{1}{M}\sum_m (f_m-y)^{2}=\frac{1}{M}\sum_m\big[(f_m-\bar f)+(\bar f-y)\big]^{2}=\frac{1}{M}\sum_m (f_m-\bar f)^{2}+(\bar f-y)^{2},$$
    the cross term $\frac{2}{M}(\bar f-y)\sum_m(f_m-\bar f)$ vanishing because $\sum_m(f_m-\bar f)=0$ by the definition of the mean. Rearranging,
    $$(\bar f-y)^{2}=\underbrace{\frac{1}{M}\sum_m (f_m-y)^{2}}_{\text{mean member error}}-\underbrace{\frac{1}{M}\sum_m (f_m-\bar f)^{2}}_{\text{ambiguity}}.$$
    No expectation was taken, no independence assumed, no distribution named. It holds pointwise on any data whatever, and since the ambiguity term is a sum of squares it is non-negative, so the ensemble's error never exceeds the mean member error and is strictly below it whenever the members disagree at all.

    Read carefully, the identity is weaker than it first sounds and it is important to say how. The benchmark it beats is the *average* member, not the best one. If one member is excellent and $M-1$ are poor, the mean member error is dominated by the poor ones and the ensemble comfortably beats it while losing badly to the good member — and nothing in the identity indicates which situation you are in. Disagreement is also not free: the members are fitted to one dataset, so raising ambiguity by making members more different generally raises their individual errors too, and the identity says only that the two effects net out in the ensemble's favour relative to the mean, not that the net is positive relative to anything you would otherwise have done.

    The load-bearing observation is that the ambiguity term contains no $y$. **The one component of ensemble performance that is measurable without labels is the members' disagreement, which is why an ensemble's spread is a usable live diagnostic and its error is not.**

The course reaches the same identity from the signal side. [Feature and Signal Engineering](../../part-04-strategy-development/05-feature-and-signal-engineering.md) reports that "the equal-weight composite's IC of 0.016 beats both parents — combining uncorrelated signals adds information content", which is the identity with the ambiguity term supplied by the parents' lack of correlation.

## Bagging Buys Variance Reduction Only From an Unstable Estimator, and Charges Everyone Else for the Bootstrap

The standard way to manufacture ambiguity from a single procedure is to refit it on bootstrap resamples and average. Whether that helps depends entirely on how much the procedure moves when its data does:

```python
import numpy as np

rng = np.random.default_rng(14051)
n, p, q, B, reps = 40, 20, 3, 50, 300
beta = np.array([0.6, 0.5, 0.4] + [0.0] * (p - 3))

Xv = rng.standard_normal((4000, p))
fv = Xv @ beta


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def step(X, y):
    """Forward stepwise: keep q columns, return a full-length coefficient vector."""
    chosen, r, b = [], y.copy(), np.zeros(p)
    for _ in range(q):
        left = [j for j in range(p) if j not in chosen]
        chosen.append(left[int(np.argmax([abs(X[:, j] @ r) for j in left]))])
        c = ols(X[:, chosen], y)
        r = y - X[:, chosen] @ c
    b[chosen] = c
    return b


print(f"  bagging {B} bootstrap refits, n = {n}, p = {p}, {reps} datasets")
print("    estimator             single: bias^2   variance    total"
      "     bagged: bias^2   variance    total     gain")
for fn, tag in ((ols, f"OLS, all {p} columns"), (step, f"stepwise, {q} of {p}")):
    single, bagged = np.empty((reps, 4000)), np.empty((reps, 4000))
    for r in range(reps):
        X = rng.standard_normal((n, p))
        y = X @ beta + rng.standard_normal(n)
        single[r] = Xv @ fn(X, y)
        acc = np.zeros(p)
        for _ in range(B):
            i = rng.integers(0, n, n)
            acc += fn(X[i], y[i])
        bagged[r] = Xv @ (acc / B)
    out = []
    for M in (single, bagged):
        bias2 = np.mean((M.mean(0) - fv) ** 2)
        out += [bias2, np.mean(M.var(axis=0)), bias2 + np.mean(M.var(axis=0))]
    print(f"    {tag:20s}  {out[0]:14.4f}   {out[1]:8.4f}   {out[2]:6.4f}"
          f"   {out[3]:16.4f}   {out[4]:8.4f}   {out[5]:6.4f}"
          f"   {1 - out[5] / out[2]:+6.1%}")

X = rng.standard_normal((n, p))
y = X @ beta + rng.standard_normal(n)
F, cols = [], set()
for _ in range(B):
    i = rng.integers(0, n, n)
    b = step(X[i], y[i])
    cols |= set(np.flatnonzero(b))
    F.append(Xv @ b)
F = np.array(F)
mean_err = np.mean((F - fv) ** 2)
disagree = np.mean((F - F.mean(0)) ** 2)
ens_err = np.mean((F.mean(0) - fv) ** 2)
print(f"  one ensemble of {B} stepwise members, each keeping {q} of {p} columns:")
print(f"    distinct columns selected across the {B} members: {len(cols)}")
print(f"    mean member error {mean_err:.6f} - mean disagreement {disagree:.6f}"
      f" = {mean_err - disagree:.6f}")
print(f"    ensemble error    {ens_err:.6f}, identity gap "
      f"{abs(ens_err - mean_err + disagree):.2e}")
# =>   bagging 50 bootstrap refits, n = 40, p = 20, 300 datasets
#        estimator             single: bias^2   variance    total     bagged: bias^2   variance    total     gain
#        OLS, all 20 columns           0.0025     1.0582   1.0607             0.0048     1.3467   1.3516   -27.4%
#        stepwise, 3 of 20             0.0292     0.3205   0.3498             0.0892     0.1948   0.2840   +18.8%
#      one ensemble of 50 stepwise members, each keeping 3 of 20 columns:
#        distinct columns selected across the 50 members: 17
#        mean member error 0.583035 - mean disagreement 0.327325 = 0.255709
#        ensemble error    0.255709, identity gap 5.55e-17
```

The last block discharges section 1 as arithmetic. Mean member error $0.583035$ minus mean disagreement $0.327325$ is $0.255709$, and the ensemble's error is $0.255709$, with a gap of $5.55\times10^{-17}$. The identity is exact, on one ordinary dataset, with no assumption about the members having been drawn in any particular way.

The same block measures the instability that made bagging worth doing. Fifty members each keep three of twenty columns, and across the fifty they touch **seventeen distinct columns** — a procedure nominally reporting a three-variable model is, under resampling, willing to name almost every candidate. That is [Feature Selection](04-feature-selection.md)'s unstable selected set seen from the other side, and it is precisely the raw material an average consumes.

The top table shows the two outcomes. For forward stepwise the trade works: variance falls from $0.3205$ to $0.1948$, squared bias rises from $0.0292$ to $0.0892$ — bagging is not bias-free, because averaging over resampled fits smooths the fitted function — and the total improves $18.8\%$, from $0.3498$ to $0.2840$.

For least squares the identical operation is a loss of $27.4\%$. Its variance *rises*, $1.0582$ to $1.3467$. The reason is the one the theory predicts: least squares is very nearly linear in $y$, and the average of a linear estimator over resamples is approximately the estimator itself, so there is no variance to cancel — while each bootstrap resample contains only about $63\%$ distinct rows, so every member is fitted on effectively less data than the original. The ensemble pays the cost of resampling and collects none of the benefit.

**Bagging is not a general improvement but a trade of the estimator's own instability for bias, so it helps exactly the procedures whose reported output was least trustworthy and harms the ones that were already stable.**

## Weights Fitted on the Data the Members Were Fitted on Give Everything to Whichever Member Memorised It

Equal weights are one choice, and the natural improvement is to learn the weights. The question is which data they may be learned from, and the answer is not the training data — for a reason that has nothing to do with subtlety of degree:

```python
import numpy as np
from scipy.optimize import nnls

rng = np.random.default_rng(14053)
n, p, s, M, K, reps = 100, 15, 5, 8, 5, 400
beta = 0.8 / np.arange(1.0, p + 1.0)
subs = [rng.choice(p, s, replace=False) for _ in range(M - 1)]

Xv = rng.standard_normal((20_000, p))
yv = Xv @ beta


def members(Xtr, ytr, Xte):
    """Seven least-squares fits on random column subsets, plus one 1-nn rule."""
    P = [Xte[:, c] @ np.linalg.lstsq(Xtr[:, c], ytr, rcond=None)[0] for c in subs]
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    P.append(ytr[np.argmin(d, axis=1)])
    return np.column_stack(P)


print(f"  {M} members combined on n = {n}; member {M} is a 1-nearest-neighbour rule,")
print(f"  so its in-sample predictions are the training labels exactly."
      f" {reps} datasets")
print("    weights fitted on                          OOS error"
      "   weight on the 1-nn member   sum w")
acc = np.zeros((4, 3))
best = 0.0
for _ in range(reps):
    X = rng.standard_normal((n, p))
    y = X @ beta + rng.standard_normal(n)
    Pin, Pv = members(X, y, X), members(X, y, Xv)
    edges = np.linspace(0, n, K + 1).astype(int)
    Poof = np.empty((n, M))
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.setdiff1d(np.arange(n), np.arange(a, b))
        Poof[a:b] = members(X[tr], y[tr], X[a:b])
    for i, w in enumerate((np.linalg.lstsq(Pin, y, rcond=None)[0],
                           np.linalg.lstsq(Poof, y, rcond=None)[0],
                           nnls(Poof, y)[0],
                           np.ones(M) / M)):
        acc[i] += (np.mean((yv - Pv @ w) ** 2), w[-1], w.sum())
    best += np.min(np.mean((yv[:, None] - Pv) ** 2, axis=0))
for i, tag in enumerate(("the same rows the members saw", "held-out folds (stacking)",
                         "held-out folds, weights forced non-negative",
                         "nothing (equal weights)")):
    v = acc[i] / reps
    print(f"    {tag:43s}  {v[0]:9.4f}   {v[1]:+24.4f}   {v[2]:5.3f}")
print(f"    for reference, the best single member scores {best / reps:.4f}")
# =>   8 members combined on n = 100; member 8 is a 1-nearest-neighbour rule,
#      so its in-sample predictions are the training labels exactly. 400 datasets
#        weights fitted on                          OOS error   weight on the 1-nn member   sum w
#        the same rows the members saw                   1.7728                    +1.0000   1.000
#        held-out folds (stacking)                       0.8336                    +0.2056   0.380
#        held-out folds, weights forced non-negative     0.7654                    +0.1991   1.105
#        nothing (equal weights)                         0.7471                    +0.1250   1.000
#        for reference, the best single member scores 0.8028
```

The first row is the failure in its pure form. One of the eight members is a nearest-neighbour rule, whose in-sample prediction at each training point is that point's own label — so on the training rows it has zero error, and a least-squares combination fitted there hands it a weight of exactly $1.0000$ and the other seven exactly nothing. Out of sample it scores $1.7728$, more than twice the equal-weight book's $0.7471$ and more than twice the best single member's $0.8028$. The combination step did not fail to improve; it selected the worst available model with complete confidence.

Moving the weight-fitting onto held-out folds repairs most of that: the nearest-neighbour member's weight falls from $1.0000$ to $0.2056$ and the error from $1.7728$ to $0.8336$. The leak is closed. What remains is ordinary estimation error, and the `sum w` column shows its shape — the fitted weights sum to $0.380$ rather than $1$, meaning the unconstrained combination is shrinking hard toward predicting nothing, which is what least squares does when its regressors are noisy and collinear.

The last two rows are the uncomfortable part. Forcing the weights non-negative — a constraint with no statistical justification, imposed because negative weights on forecasts are usually nonsense — improves the honest stacker from $0.8336$ to $0.7654$. And fitting no weights at all beats every fitted alternative at $0.7471$. The ordering is monotone in how much freedom the weights are given: unconstrained-and-leaked $1.7728$, unconstrained-and-honest $0.8336$, non-negative $0.7654$, fixed $0.7471$.

Two things are true at once and both matter. The ensemble is worth building — equal weights at $0.7471$ beat the best single member's $0.8028$, which is section 1's identity delivering a real gain. And every attempt to improve the ensemble by estimating its weights made it worse.

**A stacking layer is a model fitted to model outputs, so it inherits every pathology of the layer below it and adds its own, and the member most likely to capture the weights is the one that fits the training data best rather than the one that predicts best.**

## Optimal Weights Need a Covariance of Member Errors, and Estimating It Is the Operation That Defeated the Optimizer in Part VIII

Section 3's result invites the objection that unconstrained least squares is a crude way to choose weights. The principled way is to minimise the combination's error variance directly, and doing so produces a formula this course has already met under another name.

??? note "Proof that the error-minimising weights summing to one are $\Sigma^{-1}\mathbf{1}$ normalised, and that the loss from estimating $\Sigma$ scales with the number of members over the number of observations"

    Let $e\in\mathbb{R}^{M}$ be the vector of member errors at a point, with $\mathbb{E}[e]=0$ and $\operatorname{var}(e)=\Sigma$. A combination with weights $w$ has error $w^\top e$ and mean squared error $w^\top\Sigma w$. Minimising subject to $\mathbf{1}^\top w=1$ gives the Lagrangian $w^\top\Sigma w-2\lambda(\mathbf{1}^\top w-1)$, whose stationarity condition $\Sigma w=\lambda\mathbf{1}$ yields
    $$w^{\star}=\frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}},\qquad w^{\star\top}\Sigma w^{\star}=\frac{1}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}.$$
    This is the minimum-variance portfolio of [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) with forecasters in place of assets and forecast errors in place of returns — the identical optimisation, reached from a different problem.

    Replacing $\Sigma$ by a sample estimate $\hat\Sigma$ from $T$ observations imports the identical failure. The sample covariance of $M$ series from $T$ rows has eigenvalues spread far wider than the truth's — the smallest are biased toward zero and the largest away from it, by a factor governed by $M/T$ — and inversion multiplies by the reciprocal, so the directions the data determined worst are exactly the ones $\hat\Sigma^{-1}$ amplifies most. The resulting weights load on whichever members happened to look least correlated in sample, which is a statement about the sample. At $M/T=1$ the sample covariance is singular outright and the optimisation has no solution at all.

    The load-bearing quantity is the ratio $M/T$ rather than either alone, and note what it implies for the trade: the *value* of optimal weighting rises with $M$, since more members offer more to exploit, while the *estimability* of those weights falls with $M$ at fixed $T$. **Two curves moving in opposite directions in the same parameter guarantee a crossover, so there is always a number of members beyond which knowing that better weights exist is of no use in obtaining them.**

The crossover is a number, and it can be located:

```python
import numpy as np

rng = np.random.default_rng(14055)
T, rho, reps = 120, 0.7, 4000

print(f"  combining M forecasters whose errors correlate at {rho}, estimated from"
      f" T = {T} rows")
print("    M    T/M   oracle weights   estimated optimal   equal weights"
      "   1/N margin")
for M in (3, 6, 12, 24, 48):
    d = np.linspace(0.9, 1.1, M)                       # members of near-equal skill
    S = np.outer(d, d) * (rho + (1 - rho) * np.eye(M))  # true error covariance
    L = np.linalg.cholesky(S)
    one = np.ones(M)
    w_or = np.linalg.solve(S, one)
    w_or /= w_or.sum()
    eq = one / M
    est = np.empty(reps)
    for r in range(reps):
        E = rng.standard_normal((T, M)) @ L.T
        Sh = np.cov(E, rowvar=False)
        w = np.linalg.solve(Sh, one)
        w = w / w.sum()
        est[r] = w @ S @ w                             # scored on the true covariance
    print(f"    {M:2d}  {T / M:5.1f}   {w_or @ S @ w_or:14.4f}   {est.mean():17.4f}"
          f"   {eq @ S @ eq:13.4f}   {est.mean() / (eq @ S @ eq) - 1:+10.1%}")
# =>   combining M forecasters whose errors correlate at 0.7, estimated from T = 120 rows
#        M    T/M   oracle weights   estimated optimal   equal weights   1/N margin
#         3   40.0           0.7490              0.7619          0.8007        -4.8%
#         6   20.0           0.6939              0.7237          0.7502        -3.5%
#        12   10.0           0.6449              0.7107          0.7251        -2.0%
#        24    5.0           0.5854              0.7283          0.7125        +2.2%
#        48    2.5           0.5024              0.8343          0.7063       +18.1%
```

Read the oracle column first, because it establishes that the opportunity is real and growing. A weighting scheme with perfect knowledge of $\Sigma$ achieves $0.7490$, $0.6939$, $0.6449$, $0.5854$ and $0.5024$ as the number of members runs $3$ to $48$. More members are unambiguously better, and the value of weighting them well *increases* with $M$: at $M=48$ the oracle beats equal weighting by $29\%$.

The estimated column is what a practitioner can actually have, and it does the opposite. It improves from $0.7619$ to $0.7107$ as $M$ goes $3$ to $12$, then reverses — $0.7283$ at $M=24$, $0.8343$ at $M=48$. The gap between the oracle and the estimate at $M=48$ is $0.5024$ against $0.8343$: two thirds of the achievable improvement is consumed by the error in $\hat\Sigma$, and then some.

The final column crosses zero between $M=12$ and $M=24$, which is between ten and five observations per member. Below that ratio, estimating the weights beats equal weighting by a few percent; above it, equal weighting wins, by $2.2\%$ and then $18.1\%$. Nothing about the members changed — their true covariance is the same structure at every row, their true skill differences are the same $\pm10\%$ — only the number of parameters being estimated from a fixed $T$.

This is the Part VIII result with different labels. [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) measured mean-variance optimization at Sharpe $0.377$ against $1/N$'s $0.450$ and found that shrinking the covariance made it worse still at $0.115$; the crossover above is the same phenomenon with the estimation ratio made explicit, and section 3's non-negativity row is the same lesson's finding that constraints with no theory behind them are what rescue the optimizer. **Equal weighting is not a failure to optimize; it is the estimator with zero parameters, and it wins whenever the parameters the alternative must estimate outnumber the observations available to estimate them.**

## Bayesian Model Averaging Concentrates on One Model, Which Is Correct When One of Them Is True

The Bayesian construction weights each candidate by its posterior probability, $p(M_m\mid y)\propto m_m(y)\pi(M_m)$, and predicts by mixing. It is the coherent answer to the averaging problem and it behaves quite differently from the schemes above.

Because the marginal likelihoods are exponential in $n$, the posterior concentrates: as data accumulates, one model's weight tends to one and every other tends to zero at a rate governed by the same $\log n$ that gives BIC its consistency in [Information Criteria (AIC/BIC)](03-information-criteria.md). Bayesian model averaging is therefore, asymptotically, model *selection* — which is exactly right when one candidate is true, the setting called M-closed, and exactly wrong otherwise. In the M-open setting, where every candidate is an approximation and none is correct, the posterior still concentrates, but on the model closest in Kullback–Leibler divergence rather than on anything true, and the mixture loses the diversification that section 1's identity was paying for. Section 4's exercise is instructive here: the weights that concentrate are the ones estimated from the data, and concentration is the failure mode, not the goal.

Financial modelling is M-open in a way that is not arguable. No candidate specification is the data-generating process for asset returns; the question is only which approximation is least bad over which period. This is why frequentist stacking, which optimises predictive performance of the mixture without asking any candidate to be true, is the better-motivated tool in this domain, and why the honest version of it is section 3's constrained one rather than section 3's unconstrained one.

!!! note "Bagging, boosting, stacking, Bayesian model averaging and a random forest are five ways to use more than one model, and only three of them are averages"
    **Bagging** averages one procedure refitted on resamples, with equal weights and no fitting, and section 2 shows it buying variance reduction only where the procedure is unstable. **Stacking** averages *different* procedures with weights learned on held-out predictions, which is section 3, and it is the only member of the list whose weights are estimated from data the members did not see. **Bayesian model averaging** weights by posterior probability, which is a genuine average that degenerates to a selection as $n$ grows. **Boosting** is not an average at all — its members are fitted sequentially to the previous ensemble's residuals, so they are not exchangeable and section 1's identity does not describe it; its members are deliberately biased and individually terrible, and the sum is a fitted object rather than a mean. **A random forest** is bagging plus per-split feature randomisation, the second ingredient existing only to raise the ambiguity term of section 1 by decorrelating trees that bootstrapping alone leaves too similar. The distinction that matters operationally is whether the members were fitted independently of one another: for the three that were, adding a member is safe and the identity applies, and for boosting adding a member is a fitting decision that can overfit.

!!! warning "An ensemble's error can be low because its members are good or because they are numerous and diverse, and the output looks the same either way"
    Section 1's identity guarantees an ensemble beats its members' average, which means a mediocre ensemble of terrible models is the expected outcome rather than a warning sign — and the number reported is the good one. [Tree Ensembles](../../part-07-machine-learning/02-tree-ensembles.md) is the course's instance: five hundred trees whose per-day spread of opinion was a substantial $0.198$, averaged down to $0.075$, producing an AUC of $0.492$ against the members' own mean of $0.497$ — the machinery worked perfectly and the result was nothing, because the ambiguity term was large and the members shared no signal to average toward. Above, equal weights scored $0.7471$ against a best single member's $0.8028$, a real gain; the identical arithmetic would have produced a similar-looking gain had every member been worthless. **The free diagnostic is to print the ensemble's score beside its best single member's and its median member's, all three on the same held-out data: if the ensemble beats the median and loses to the best, the averaging is masking a selection problem rather than solving one; if it beats the best by roughly the ambiguity term you measured, the diversification is real; and if all three sit at the same level as a constant prediction, the ensemble has faithfully averaged a set of models that do not know anything, which is section 2's $18.8\%$ improvement on a quantity that may not have been worth improving.**

## Averaging Removes the Maximum From the Procedure and Leaves the Question of Whether Anything Was There

This page established that the ambiguity identity holds exactly and unconditionally, verified at $0.583035-0.327325=0.255709$ against an ensemble error of $0.255709$ with a gap of $5.55\times10^{-17}$, and that it promises only a win against the *mean* member; that bagging trades the estimator's own instability for bias, improving forward stepwise by $18.8\%$ as variance fell $0.3205$ to $0.1948$ and squared bias rose $0.0292$ to $0.0892$, while degrading least squares by $27.4\%$ because a linear estimator has no instability to cancel and pays the bootstrap's cost anyway; that the fifty bagged stepwise members touched $17$ of $20$ columns while each nominally kept three; that weights fitted on the members' own training rows placed $1.0000$ on the one member that memorised the labels and scored $1.7728$, that moving them to held-out folds cut this to $0.8336$, that forcing non-negativity cut it to $0.7654$, and that fitting no weights at all won at $0.7471$ — beating the best single member's $0.8028$; and that optimal weights are worth progressively more as members accumulate, the oracle improving $0.7490$ to $0.5024$ from $M=3$ to $M=48$, while *estimated* optimal weights reverse at around ten observations per member and lose to $1/N$ by $2.2\%$ and then $18.1\%$.

The through-line of the part ends where it began. [Bias–Variance Tradeoff](01-bias-variance-tradeoff.md) showed that only two of three error terms are available to any procedure; [Cross-Validation](02-cross-validation.md) and [Information Criteria (AIC/BIC)](03-information-criteria.md) built two estimates of where those two terms are jointly smallest, each conditional on assumptions the output does not print; [Feature Selection](04-feature-selection.md) showed that the act of choosing corrupts every statistic reported afterwards. This page removes the choosing, and the corruption goes with it — an average has no maximum in it, so there is no distribution-of-a-maximum problem to correct. What replaces it is an estimation problem one level up, and section 4 measures its price: the weights are parameters, and parameters must be estimated from the same finite sample everything else came from.

That is the honest end of the technique. Every procedure in this part has now been shown to work, to be measurable, and to leave one quantity uncharged — the number of candidates that were examined before the winner was written down. Bagging over fifty resamples, stacking over eight members, and comparing four criteria are all searches, and none of the arithmetic here paid for them. Attaching a price to the size of a search, so that a result can be reported net of how hard it was looked for, is [Part XV](../part-15-multiple-testing/index.md).

**Averaging is the one operation in model selection that improves an estimate without asking anything of the data, which is why it works and why it cannot tell you whether the models it combined knew anything at all.**
