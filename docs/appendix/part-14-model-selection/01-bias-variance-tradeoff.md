# Bias–Variance Tradeoff

The tradeoff is usually drawn as a dial. Turn it left and the model is too simple; turn it right and it is too complicated; somewhere in the middle sits a minimum, and the work of model selection is finding it. The picture is not wrong so much as missing the term that dominates every number in it. Prediction error has three parts, not two, and the first belongs to nobody — no model, no amount of data and no tuning touches it. Below, that untouchable part is $1.0000$ out of a best-achievable $1.0678$, so $93.6\%$ of the error at the bottom of the U-curve was never available to be removed. Two further facts fall out of the same arithmetic and neither is in the usual picture: the complexity that minimises prediction error is *smaller* than the complexity of the truth — degree $3$ against a truth of degree $5$ — and stays smaller until $n$ reaches $20{,}000$; and the split itself, the one quantity that says whether to buy more data or a different model, is invisible in a single dataset, two smoothers measured at $1.2167$ and $1.2161$ having exactly opposite compositions while the attempt to recover the split from one sample returns a squared quantity of $-0.2463$.

This page covers the three-term decomposition of prediction error and the independence assumption that licenses it, the gap between the error-minimising model and the true one, effective degrees of freedom as a trace with the two traces that compete for the name, the unidentifiability of the bias–variance split from a single sample, and the failure of the whole apparatus to survive translation out of squared error. It does not decompose the mean squared error of an estimator about a fixed parameter, which is [Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md) and which stops at two terms on purpose; it estimates no error curve without knowing the truth, which is [Cross-Validation](02-cross-validation.md); it penalizes no complexity by a criterion, which is [Information Criteria (AIC/BIC)](03-information-criteria.md); it chooses among no candidate predictors, which is [Feature Selection](04-feature-selection.md); it combines no models, which is [Model Averaging](05-model-averaging.md); it derives no shrinkage geometry, which is [Regularization](../part-13-regression/05-regularization.md); it charges nothing for the size of a search, which is [Part XV](../part-15-multiple-testing/index.md); and it never reports a training error as evidence about a model.

The trading stake is the most expensive null result in the course. [Tree Ensembles](../../part-07-machine-learning/02-tree-ensembles.md) builds a five-hundred-tree random forest and reports that "the forest's variance reduction worked (per-day tree spread 0.198 → 0.075) and its AUC of 0.492 matched the 0.497 mean of its own trees: averaging removes variance, not noise, and noise was the constraint." Every word of that is the decomposition below, used as a diagnosis. The machinery attacked the variance term flawlessly, cutting the per-day spread of tree opinions by nearly two thirds, and moved the score by nothing at all, because the binding term was the one the forest has no instrument for. Sections 1 and 4 are why that outcome was legible only in hindsight, and why the desk had no way to see it coming from the data it had.

## Prediction Error Splits Into Three Terms, and the First One Is Not a Property of Any Model

[Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md) splits an estimator's error about a fixed parameter into a squared bias and a variance, with no third term, and states plainly that "the two-term decomposition is the whole content of this page and the three-term one is the whole content of Bias–Variance Tradeoff". The difference is the target. There, the target is $\theta$, a constant. Here, the target is $Y$, a random variable that will be drawn *after* the model is fitted, and its own randomness enters the error without passing through the estimator at all.

??? note "Proof that expected prediction error is the sum of a noise variance, a squared bias and a variance, and that the split requires the test observation to be independent of the training sample"

    Write the data-generating process as $Y=f(x)+\varepsilon$ with $\mathbb{E}[\varepsilon]=0$, $\operatorname{var}(\varepsilon)=\sigma^{2}_{\varepsilon}$, and let $\hat f$ be fitted on a training sample $\mathcal{D}$ that does not contain the pair $(x,Y)$ at which the error is evaluated. Expanding about $\mathbb{E}[\hat f(x)]$, where the expectation is over training samples,
    $$\mathbb{E}\big[(Y-\hat f(x))^{2}\big]=\mathbb{E}\big[(\varepsilon + f(x)-\hat f(x))^{2}\big]=\sigma^{2}_{\varepsilon}+\mathbb{E}\big[(f(x)-\hat f(x))^{2}\big],$$
    the cross term $2\,\mathbb{E}[\varepsilon\,(f(x)-\hat f(x))]$ vanishing because $\varepsilon$ is the noise on the *test* draw and $\hat f$ is a function of $\mathcal{D}$ alone, so the two factors are independent and one of them has mean zero. Adding and subtracting $\mathbb{E}[\hat f(x)]$ inside the remaining term splits it the same way, with its own cross term killed because $f(x)-\mathbb{E}[\hat f(x)]$ is deterministic:
    $$\mathbb{E}\big[(Y-\hat f(x))^{2}\big]=\sigma^{2}_{\varepsilon}+\big(f(x)-\mathbb{E}[\hat f(x)]\big)^{2}+\operatorname{var}\big(\hat f(x)\big).$$

    Both cross terms died for the same reason and it is worth naming which reason, because they are not the same independence. The second used the tower property on a constant, exactly as in the two-term case. The first used the assumption that the test noise was not seen during fitting — and that is an assumption about how the data was split, not about the model. It fails whenever the training and test observations share a shock, which on overlapping labels or serially dependent returns is the normal case rather than the pathological one, and when it fails the identity does not merely lose accuracy: the term that was supposed to be a floor becomes negotiable, and a model can appear to predict below it.

    The load-bearing distinction is that $\sigma^{2}_{\varepsilon}$ carries no $\hat f$ anywhere in its definition. **Two of the three terms describe the modelling choice and the third describes the world, and a research programme that reports only totals cannot tell a practitioner which of the three it has been moving.**

The consequence is a ceiling that no amount of effort passes, and its height is not knowable from the fit. A model reporting a mean squared error of $1.07$ against a floor of $1.00$ has captured essentially everything available and looks, on the raw number, almost exactly like a model reporting $1.07$ against a floor of $0.20$ that has missed most of it. The two situations call for opposite decisions and produce identical printouts.

## The Complexity That Minimises Prediction Error Is Smaller Than the Complexity of the Truth, and It Stays Smaller Until the Sample Is Enormous

Because the three terms are separately computable when the truth is known by construction, the whole curve can be measured rather than sketched. The prediction is that a polynomial truth of degree five is best fitted by something less than degree five:

```python
import numpy as np

rng = np.random.default_rng(14011)
n, sig, reps, dmax = 60, 1.0, 20_000, 12
coef = np.array([1.0, 2.0, -1.5, 0.8, -0.4, 0.25])      # the truth is degree 5

x = np.linspace(-1.0, 1.0, n)
f = np.polynomial.polynomial.polyval(x, coef)
Y = f + rng.normal(0.0, sig, (reps, n))                 # training draws
Ynew = f + rng.normal(0.0, sig, (reps, n))              # fresh test draws, same x

print(f"  polynomial fits to a degree-5 truth at n = {n}, sigma = {sig:.1f}, "
      f"{reps:,} resamples")
print("    degree   bias^2   variance    floor     total   measured   train MSE")
tot = np.empty(dmax + 1)
for d in range(dmax + 1):
    X = np.polynomial.polynomial.polyvander(x, d)
    S = X @ np.linalg.pinv(X)
    bias2 = np.mean((S @ f - f) ** 2)
    var = sig**2 * np.mean(np.einsum("ij,ij->i", S, S))
    fit = Y @ S.T
    meas = np.mean((Ynew - fit) ** 2)
    train = np.mean((Y - fit) ** 2)
    tot[d] = sig**2 + bias2 + var
    print(f"    {d:6d}   {bias2:6.4f}   {var:8.4f}   {sig**2:6.4f}   {tot[d]:7.4f}"
          f"   {meas:8.4f}   {train:9.4f}")
print(f"    minimising degree {int(np.argmin(tot))}, true degree 5")

print("    minimising degree against sample size, same truth and same noise")
print("        n      30       60      250     1000     4000    20000")
row = []
for m in (30, 60, 250, 1000, 4000, 20_000):
    xm = np.linspace(-1.0, 1.0, m)
    fm = np.polynomial.polynomial.polyval(xm, coef)
    t = []
    for d in range(dmax + 1):
        Xm = np.polynomial.polynomial.polyvander(xm, d)
        Sm = Xm @ np.linalg.pinv(Xm)
        t.append(np.mean((Sm @ fm - fm) ** 2)
                 + sig**2 * np.mean(np.einsum("ij,ij->i", Sm, Sm)))
    row.append(int(np.argmin(t)))
print("     argmin" + "".join(f"{v:9d}" for v in row))
# =>   polynomial fits to a degree-5 truth at n = 60, sigma = 1.0, 20,000 resamples
#        degree   bias^2   variance    floor     total   measured   train MSE
#             0   2.7058     0.0167   1.0000    3.7225     3.7178      3.6925
#             1   0.3573     0.0333   1.0000    1.3906     1.3897      1.3253
#             2   0.0309     0.0500   1.0000    1.0809     1.0813      0.9816
#             3   0.0012     0.0667   1.0000    1.0678     1.0681      0.9350
#             4   0.0001     0.0833   1.0000    1.0834     1.0837      0.9174
#             5   0.0000     0.1000   1.0000    1.1000     1.1002      0.9008
#             6   0.0000     0.1167   1.0000    1.1167     1.1168      0.8842
#             7   0.0000     0.1333   1.0000    1.1333     1.1331      0.8678
#             8   0.0000     0.1500   1.0000    1.1500     1.1495      0.8512
#             9   0.0000     0.1667   1.0000    1.1667     1.1663      0.8345
#            10   0.0000     0.1833   1.0000    1.1833     1.1829      0.8178
#            11   0.0000     0.2000   1.0000    1.2000     1.1995      0.8011
#            12   0.0000     0.2167   1.0000    1.2167     1.2166      0.7844
#        minimising degree 3, true degree 5
#        minimising degree against sample size, same truth and same noise
#            n      30       60      250     1000     4000    20000
#         argmin        2        3        3        3        4        5
```

Read the `total` and `measured` columns against each other first, because everything downstream depends on them agreeing. The exact decomposition and the brute-force simulation match to the fourth decimal at every degree — $1.0678$ against $1.0681$, $1.1000$ against $1.1002$, $1.2167$ against $1.2166$ — so the identity of section 1 is arithmetic here rather than assertion, and the columns can be trusted separately.

The variance column is a straight line: $0.0167$, $0.0333$, $0.0500$, rising by exactly $\sigma^{2}/n=1/60$ per parameter added. The bias column collapses: $2.7058$ at degree zero, $0.3573$ at degree one, $0.0309$ at two, $0.0012$ at three, and numerically nothing from degree five onward, as it must once the model contains the truth. The two meet at degree $3$ for a total of $1.0678$, and **the true degree of $5$ scores $1.1000$ — worse.** Fitting the correct model costs $0.0322$ more than fitting a knowingly wrong one, because the last two coefficients of the truth are small enough that estimating them costs more variance than the bias they remove. Nothing is broken; the correct model is simply not the best predictor at this sample size.

The `floor` column reframes the rest. The entire span of the U-curve, from the best model to the worst sensible one, is $1.0678$ to $1.2167$ — a range of $0.1489$ sitting on an untouchable $1.0000$. A practitioner who moved a model from degree $12$ to degree $3$ would report a $12\%$ reduction in error and would have captured $100\%$ of what was available. Meanwhile the training column falls monotonically — $3.6925$, $1.3253$, $0.9816$, $0.9350$, on down to $0.7844$ at degree $12$ — and never turns up: it reads $0.9350$ where prediction is best and $0.7844$, its lowest value in the table, where prediction is worst. **Training error ranks the candidates in the reverse of the order that matters, and does so smoothly, with no kink at the point where the ranking inverts.**

The last row prices the gap between the best model and the true one in units of data. The minimiser is degree $2$ at $n=30$, degree $3$ from $n=60$ through $n=1000$, degree $4$ at $n=4000$, and reaches the true degree $5$ only at $n=20{,}000$ — more observations of a stationary process with known functional form and independent noise than the course's entire daily price history, for a truth with five terms.

## Complexity Is a Trace, and Two Different Traces Compete for the Name

Counting parameters worked in the last section because a least-squares fit is a projection, and projections have integer complexity. Most estimators in use are not projections. A ridge fit, a moving average, a kernel smoother and a $k$-nearest-neighbour rule are all *linear smoothers* — the fitted values are $\hat f = Sy$ for a matrix $S$ that does not depend on $y$ — and for these the parameter count is undefined while the variance is not.

??? note "Proof that a linear smoother's average variance is $\sigma^{2}\operatorname{tr}(SS^\top)/n$, which equals $\operatorname{tr}(S)$ only when $S$ is a projection"

    With $\hat f=Sy$ and $\operatorname{var}(y)=\sigma^{2}I$, the fitted-value covariance is $\operatorname{var}(\hat f)=\sigma^{2}SS^\top$, so the variance at the $i$th point is $\sigma^{2}(SS^\top)_{ii}=\sigma^{2}\lVert s_i\rVert^{2}$ where $s_i$ is the $i$th row of $S$. Averaging over the design,
    $$\frac{1}{n}\sum_i\operatorname{var}\big(\hat f(x_i)\big)=\frac{\sigma^{2}}{n}\operatorname{tr}(SS^\top).$$
    If $S$ is the orthogonal projection onto a $p$-dimensional column space then $S$ is symmetric and idempotent, so $SS^\top=S^{2}=S$ and the two traces coincide at $p$, recovering $\sigma^{2}p/n$ — the straight line measured in section 2. If $S$ is not idempotent the two separate, and for a ridge smoother with singular values $d_i$ they are $\operatorname{tr}(S)=\sum_i d_i^{2}/(d_i^{2}+\lambda)$ against $\operatorname{tr}(SS^\top)=\sum_i d_i^{4}/(d_i^{2}+\lambda)^{2}$, the second strictly smaller because each summand is the first's summand squared and each lies in $(0,1)$.

    The quantity conventionally reported as "effective degrees of freedom" is $\operatorname{tr}(S)$, and it is the right quantity for a different purpose: it is the optimism of the training error, since $\mathbb{E}[\text{train}]=\mathbb{E}[\text{test}]-2\sigma^{2}\operatorname{tr}(S)/n$ for a linear smoother, which is what makes Mallows' $C_p$ and the criteria of [Information Criteria (AIC/BIC)](03-information-criteria.md) work at all. So the two traces are both correct and answer different questions, and the one printed by software is usually the one that does not set the variance.

    The load-bearing observation is that neither trace mentions the truth. **Degrees of freedom measure what a smoother costs, never what it buys, and two smoothers with identical complexity can sit at opposite ends of the bias axis.**

That last sentence is a claim about a table, so here is the table:

```python
import numpy as np

rng = np.random.default_rng(14013)
n, sig, reps = 60, 1.0, 20_000
coef = np.array([1.0, 2.0, -1.5, 0.8, -0.4, 0.25])

x = np.linspace(-1.0, 1.0, n)
f = np.polynomial.polynomial.polyval(x, coef)
Y = f + rng.normal(0.0, sig, (reps, n))

B = np.polynomial.polynomial.polyvander(x, 12)          # 13-column design
Z = np.column_stack([np.ones(n), (B[:, 1:] - B[:, 1:].mean(0)) / B[:, 1:].std(0)])
P = np.diag([0.0] + [1.0] * 12)                         # intercept unpenalised
G = Z.T @ Z


def smoother(kind, par):
    if kind == "ridge":
        return Z @ np.linalg.solve(G + par * P, Z.T)
    if kind == "poly":
        A = np.polynomial.polynomial.polyvander(x, int(par))
        return A @ np.linalg.pinv(A)
    S = np.zeros((n, n))                                # k-nearest-neighbour
    for i in range(n):
        nb = np.argsort(np.abs(x - x[i]))[: int(par)]
        S[i, nb] = 1.0 / int(par)
    return S


print(f"  linear smoothers on the same n = {n} design, sigma = {sig:.1f}, "
      f"{reps:,} resamples")
print("    smoother          tr(S)   tr(SS')   var: measured   sig^2 tr(SS')/n"
      "   bias^2")
for kind, par, tag in (("poly", 3, "OLS degree 3"), ("poly", 12, "OLS degree 12"),
                       ("ridge", 100.0, "ridge L=100"), ("ridge", 3.0, "ridge L=3"),
                       ("ridge", 0.03, "ridge L=0.03"), ("knn", 15, "15-nn"),
                       ("knn", 5, "5-nn"), ("knn", 2, "2-nn")):
    S = smoother(kind, par)
    trS = np.trace(S)
    trSS = np.einsum("ij,ij->", S, S)
    fit = Y @ S.T
    meas = np.mean(fit.var(axis=0))
    print(f"    {tag:15s}  {trS:6.3f}   {trSS:7.3f}   {meas:13.4f}"
          f"   {sig**2 * trSS / n:15.4f}   {np.mean((S @ f - f) ** 2):6.4f}")
# =>   linear smoothers on the same n = 60 design, sigma = 1.0, 20,000 resamples
#        smoother          tr(S)   tr(SS')   var: measured   sig^2 tr(SS')/n   bias^2
#        OLS degree 3      4.000     4.000          0.0665            0.0667   0.0012
#        OLS degree 12    13.000    13.000          0.2172            0.2167   0.0000
#        ridge L=100       2.981     2.267          0.0376            0.0378   0.4477
#        ridge L=3         5.531     4.833          0.0805            0.0806   0.0073
#        ridge L=0.03      8.373     7.823          0.1305            0.1304   0.0000
#        15-nn             4.000     4.000          0.0665            0.0667   0.1494
#        5-nn             12.000    12.000          0.2000            0.2000   0.0084
#        2-nn             30.000    30.000          0.5005            0.5000   0.0048
```

The two variance columns agree in every row to the third or fourth decimal — $0.0665$ against $0.0667$, $0.1305$ against $0.1304$, $0.5005$ against $0.5000$ — so $\operatorname{tr}(SS^\top)/n$ is the variance, measured, and not an approximation to it.

The two trace columns agree only where the proof says they must. The projections are exact integers with the columns identical, $4.000$ at degree three and $13.000$ at degree twelve, and the $k$-nearest-neighbour rows are identical at $n/k$ — $4.000$, $12.000$, $30.000$ for $k=15,5,2$ — since each row of that $S$ carries $k$ entries of $1/k$. The ridge rows are where they part: $2.981$ against $2.267$, $5.531$ against $4.833$, $8.373$ against $7.823$. At $\lambda=100$ the conventionally reported complexity exceeds the variance-setting one by $31\%$, so a ridge fit described as having "about three degrees of freedom" carries the variance of a projection onto $2.267$ dimensions. Ridge also shows the point of the whole construction, which is that complexity became continuous: those same three penalties moved the model between four and nine parameters without it ever having a fourth or a ninth, so section 2's U-curve can be searched on a real line rather than a lattice.

Now compare the first row against the sixth. `OLS degree 3` and `15-nn` have $\operatorname{tr}(S)=4.000$, $\operatorname{tr}(SS^\top)=4.000$, and a measured variance of $0.0665$ — the same to every printed digit, three times over. Their squared biases are $0.0012$ and $0.1494$, a factor of $124$ apart. Two smoothers that any complexity-based accounting must treat as interchangeable differ in total prediction error by $0.1482$, which is the entire useful range of section 2's U-curve.

**Degrees of freedom is a budget, and the table shows two estimators spending an identical budget to buy things that differ by two orders of magnitude.**

## Neither Component Is Visible in One Dataset, Because Separating Them Needs a Noise Floor That Only an Already-Unbiased Model Can Supply

Everything measured so far required two things a desk does not have: the truth $f$, and twenty thousand independent redraws of history. The practical question is what survives their loss. The answer is the total and nothing else, which would be tolerable if the total were what one acted on — but the remedy for a bias problem is a different model and the remedy for a variance problem is more data, so the split is the actionable quantity and the total is not.

```python
import numpy as np

rng = np.random.default_rng(14015)
n, sig, reps = 60, 1.0, 20_000
coef = np.array([1.0, 2.0, -1.5, 0.8, -0.4, 0.25])

x = np.linspace(-1.0, 1.0, n)
f = np.polynomial.polynomial.polyval(x, coef)
Y = f + rng.normal(0.0, sig, (reps, n))
Ynew = f + rng.normal(0.0, sig, (reps, n))


def poly(d):
    A = np.polynomial.polynomial.polyvander(x, d)
    return A @ np.linalg.pinv(A)


def knn(k):
    S = np.zeros((n, n))
    for i in range(n):
        S[i, np.argsort(np.abs(x - x[i]))[:k]] = 1.0 / k
    return S


print(f"  two smoothers with the same expected error, n = {n}, {reps:,} resamples")
print("    smoother       bias^2   variance   total   per-dataset held-out mean"
      "     sd")
for S, tag in ((poly(12), "OLS degree 12"), (knn(15), "15-nn")):
    fit = Y @ S.T
    per = np.mean((Ynew - fit) ** 2, axis=1)
    print(f"    {tag:13s}  {np.mean((S @ f - f) ** 2):6.4f}   "
          f"{sig**2 * np.einsum('ij,ij->', S, S) / n:8.4f}"
          f"  {sig**2 + np.mean((S @ f - f) ** 2) + sig**2 * np.einsum('ij,ij->', S, S) / n:6.4f}"
          f"   {per.mean():23.4f}  {per.std():5.4f}")

print("    the floor has to be estimated too, and every estimate assumes a model")
print("      working model   sigma^2 hat   implied floor error   15-nn bias^2 hat"
      "   truth")
C, trC = knn(15), np.einsum("ij,ij->", knn(15), knn(15))
tot15 = np.mean((Ynew - Y @ C.T) ** 2)
for d in (1, 2, 3, 5, 8, 12):
    S = poly(d)
    s2 = np.mean(np.sum((Y - Y @ S.T) ** 2, axis=1)) / (n - d - 1)
    b2 = tot15 - s2 - s2 * trC / n
    print(f"      degree {d:2d}       {s2:11.4f}   {s2 - sig**2:+19.4f}"
          f"   {b2:16.4f}   {0.1494:5.4f}")
# =>   two smoothers with the same expected error, n = 60, 20,000 resamples
#        smoother       bias^2   variance   total   per-dataset held-out mean     sd
#        OLS degree 12  0.0000     0.2167  1.2167                    1.2173  0.2329
#        15-nn          0.1494     0.0667  1.2161                    1.2150  0.2265
#        the floor has to be estimated too, and every estimate assumes a model
#          working model   sigma^2 hat   implied floor error   15-nn bias^2 hat   truth
#          degree  1            1.3699               +0.3699            -0.2463   0.1494
#          degree  2            1.0324               +0.0324             0.1138   0.1494
#          degree  3            1.0011               +0.0011             0.1472   0.1494
#          degree  5            0.9995               -0.0005             0.1489   0.1494
#          degree  8            0.9991               -0.0009             0.1493   0.1494
#          degree 12            0.9997               -0.0003             0.1487   0.1494
```

The first block is a matched pair constructed to be indistinguishable where it counts. `OLS degree 12` totals $1.2167$ and `15-nn` totals $1.2161$, a difference of $0.0006$. Their compositions are opposite: the first is all variance at $(0.0000,\,0.2167)$ and the second is mostly bias at $(0.1494,\,0.0667)$. One of them is starved of data and will improve on its own; the other is structurally wrong and will not. Nothing in the totals says which.

Nor can a held-out set say it. The per-dataset held-out error averages $1.2173$ and $1.2150$ — correct to within simulation noise — with standard deviations of $0.2329$ and $0.2265$ across datasets. The quantity to be resolved is $0.0006$ and the noise on a single measurement of it is roughly $0.23$, about four hundred times larger. A held-out set of this size does not distinguish these two models at all, let alone their compositions.

The second block asks whether the split can be reconstructed indirectly. It can, in principle: section 3 gives the variance as $\hat\sigma^{2}\operatorname{tr}(SS^\top)/n$ from the fit alone, so if the floor $\hat\sigma^{2}$ were known then bias$^{2}$ would follow by subtraction from the total. The floor is estimated the standard way, as a residual sum of squares over its degrees of freedom — and that estimate requires a working model, which is a choice, and the choice is the thing under investigation. A degree-one working model returns $\hat\sigma^{2}=1.3699$, overstating the floor by $+0.3699$, because its own squared bias of $0.3573$ has nowhere else to go and is booked as noise; degree two overstates by $+0.0324$; from degree three onward the estimate is honest at $1.0011$, $0.9995$, $0.9991$ and $0.9997$.

The last column is the consequence, and it is worse than an inaccuracy. Under the degree-one working model the implied squared bias of the `15-nn` candidate is $-0.2463$: a negative estimate of a squared quantity, not merely off by $0.3957$ but outside the range of the thing being estimated, and arriving with no error message attached. Degree two gives $0.1138$ against a truth of $0.1494$, understating by a quarter. Only once the working model is degree three or richer — that is, once it is *already* unbiased — do the estimates land at $0.1472$, $0.1489$, $0.1493$ and $0.1487$. The direction of the error is the part to keep: an underfit working model inflates the apparent floor and therefore *understates* every candidate's apparent bias, which reads on the page as "close to the information limit" precisely when the model is nowhere near it.

**The three-term split is recoverable from one sample if and only if you already possess an unbiased model, which is the thing the split was supposed to help you find, and when you do not possess one the procedure returns an answer that is both wrong and reassuring.**

## The Decomposition Is a Statement About Squared Error, and a Book Is Not Graded in Squared Error

Every identity on this page is a consequence of squaring. The cross terms vanished because squares expand into cross terms that are linear in the noise; the additivity holds because variance is additive; the U-curve is a sum of two functions of complexity that happen to move in opposite directions under that particular loss. None of it is a general fact about prediction.

Under absolute error the decomposition does not hold — the analogue involves the median rather than the mean and there is no clean additive split. Under a classifier's zero-one loss it fails more interestingly: only which side of the decision boundary the estimate falls on matters, not how far from the truth it lands, so variance can be *beneficial* when the bias points the wrong way. And a Sharpe ratio is not a pointwise loss at all; a change that lowers squared forecast error can move it in either direction, because the path from forecast to position to P&L runs through a threshold, a lag and a cost model.

This is the arithmetic behind an outcome the course meets repeatedly. [Tree Ensembles](../../part-07-machine-learning/02-tree-ensembles.md) reports a forest whose per-day spread of tree opinions fell from $0.198$ to $0.075$ and whose AUC was $0.492$ against its own members' mean of $0.497$ — a large, real, correctly-executed variance reduction that bought nothing, because the binding term was the floor. Section 2's table is the same story with the floor visible: from degree $12$ to degree $3$ the total moves $1.2167$ to $1.0678$, and $1.0000$ of what remains was never in play.

!!! note "Overfitting, high variance, excess degrees of freedom and a small training error are four descriptions of one gap, and only one of them is measurable from the fit"
    They are routinely used interchangeably and they are not the same object. **Overfitting** is a statement about the *difference* between training and test error, so it is a property of a pair of numbers, one of which is unobservable without held-out data. **High variance** is a property of the estimator across hypothetical redraws of the training sample — section 2 needed twenty thousand of them to measure it, and a desk has one. **Excess degrees of freedom** is $\operatorname{tr}(S)$ or $\operatorname{tr}(SS^\top)$, a property of the smoother matrix alone, computable from the design before any $y$ is observed and the only one of the four available from a single fit at zero cost. **A small training error** is a property of the fit that, as section 2's last column shows, moves *monotonically against* the quantity of interest and therefore carries no information about it in isolation. The four coincide in the textbook picture because that picture holds the truth fixed and varies only complexity; outside it they come apart, which is how a model can overfit while having few parameters — the search that chose it had many, which is [Feature Selection](04-feature-selection.md) — and how a model can carry enormous degrees of freedom and not overfit at all, the penalty having absorbed them, which is [Regularization](../part-13-regression/05-regularization.md).

!!! warning "A model beating a benchmark by a wide margin and a model beating it by a hair look identical when the floor is unknown, and the floor is always unknown"
    Section 2 is the uncomfortable case in miniature: the best model available scored $1.0678$ where the floor was $1.0000$, capturing everything there was, and the worst sensible model scored $1.2167$ — so the entire distance between competence and incompetence was $12\%$ of the reported number, and a reader shown only the totals would reasonably conclude that both models were mediocre and that a better one must exist. Section 4 shows that the obvious repair makes it worse: estimating the floor from an underfit working model returned $\hat\sigma^{2}=1.3699$ against a truth of $1.0000$ and drove a candidate's implied squared bias to $-0.2463$, an impossible value that arrives with no error message attached and whose sign flatters the model under study. The practical consequence is that "our error fell by $12\%$" is uninterpretable without a scale, and quarters get spent tuning models that were already at the limit. **The free diagnostic is to bracket the floor from both sides before tuning anything: refit the entire pipeline on randomly permuted labels, which destroys any real relationship and returns the error of a model that cannot possibly know anything, and separately fit a deliberately over-rich model whose bias is negligible and read its error as a loose upper bound on the floor; if your candidate sits close to the second number then the remaining headroom is small however large the raw error looks, and if it sits close to the first then the pipeline has found nothing at all and the next quarter belongs to features rather than to models.**

## A Model's Error Has Three Parts, Two Owners and One Number Anybody Can See

This page established that prediction error splits into a noise variance, a squared bias and a variance, with the first cross term vanishing only because the test observation was assumed independent of the training sample — an assumption about the split rather than the model, and the one that dependent data breaks; that the decomposition is exact rather than approximate, its predicted totals of $1.0678$, $1.1000$ and $1.2167$ matching brute-force simulation at $1.0681$, $1.1002$ and $1.2166$; that the error-minimising complexity is *below* the truth's, degree $3$ scoring $1.0678$ against the true degree $5$'s $1.1000$, with the minimiser climbing $2$, $3$, $3$, $3$, $4$, $5$ as $n$ runs $30$ to $20{,}000$ so that the correct model becomes worth fitting only in the last column; that training error fell monotonically from $3.6925$ to $0.7844$ and was therefore lowest exactly where prediction was worst; that complexity for a linear smoother is $\sigma^{2}\operatorname{tr}(SS^\top)/n$, matching measured variance at $0.0665$ against $0.0667$ and $0.5005$ against $0.5000$, while the conventionally reported $\operatorname{tr}(S)$ overstates it by $31\%$ for ridge at $2.981$ against $2.267$; that identical complexity buys wildly different things, `OLS degree 3` and `15-nn` sharing $\operatorname{tr}(S)=4.000$ and a variance of $0.0665$ while their squared biases differ by a factor of $124$ at $0.0012$ and $0.1494$; and that the split is not identifiable from one sample, two models totalling $1.2167$ and $1.2161$ with opposite compositions being separated by $0.0006$ against a per-dataset standard deviation of $0.2329$, while reconstructing the split through an estimated floor returned $-0.2463$ for a squared quantity when the working model was underfit and recovered the truth's $0.1494$ only once the working model was already unbiased.

The three terms have two owners. Bias and variance belong to the analyst, respond to different treatments, and are the subject of the remaining pages of this part. The noise floor belongs to the world, is usually the largest of the three, and is the reason the other two are so often argued about at length to no effect. What makes the arrangement hard is not the algebra, which is a page of expansions, but the observability: the term that dominates cannot be measured without a model, the split that determines the remedy cannot be measured without replications, and the one number available from a single fit — the training error — moves in the wrong direction.

That is a description of a problem rather than a method, and it leaves the practical question where it started. The curve of section 2 was drawn with the truth in hand; a desk needs the same curve drawn without it, and needs the minimum located well enough to pick a model. The obvious idea is to buy the missing test set out of the training data by holding some of it back, which converts a question about an unknown population into a question about a data split — and imports, as the price, every property of the split. That is [Cross-Validation](02-cross-validation.md).

**The bias–variance decomposition is the reason model selection is possible and the reason it is hard: it proves that a best model exists at a complexity below the truth's, and it proves in the same breath that the quantity distinguishing that model from its rivals is not a function of the data you have.**
