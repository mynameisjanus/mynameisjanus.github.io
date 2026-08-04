# Feature Selection

Selecting predictors feels like a step that happens before the analysis, in the way that loading a file happens before the analysis. It is not a step before; it is part of the estimator, and every statistic computed afterwards was derived under the assumption that it never occurred. The consequences are not subtle corrections. Below, forward selection of five columns from a hundred pure-noise candidates produces an $R^{2}$ of $0.2617$ and an $F$-test that rejects at the five percent level $1.0000$ of the time, with a median $p$-value of $0.000029$, on data in which nothing whatsoever is real. Screening twenty predictors from five thousand and then cross-validating returns an error of $0.4524$ where the honest answer is $1.00$ — the leak survives the resampling scheme that was supposed to catch it, because the screening happened first. And a nominal $95\%$ confidence interval for a coefficient chosen as the largest of a hundred covers its true value of zero $0.58\%$ of the time, while being *narrower* than the interval that would have been honest.

This page covers the search space that best subset, stepwise and screening all traverse, the inflation of every in-sample statistic by the act of searching, the failure of cross-validation to detect a leak that preceded it, the conditional distribution of a selected coefficient and the coverage collapse that follows, and the two repairs — sample splitting and reporting selection frequency — with their prices. It does not decompose prediction error, which is [Bias–Variance Tradeoff](01-bias-variance-tradeoff.md); it establishes no property of a resampling scheme, which is [Cross-Validation](02-cross-validation.md); it derives no penalized-likelihood score, which is [Information Criteria (AIC/BIC)](03-information-criteria.md); it combines no models, which is [Model Averaging](05-model-averaging.md); it derives no $\ell_1$ geometry and re-measures no lasso selection frequency, which is [Regularization](../part-13-regression/05-regularization.md); it constructs no family-wise or false-discovery correction for the number of candidates, which is [Part XV](../part-15-multiple-testing/index.md); it computes no posterior over models, which is [Part XVI](../part-16-bayesian-statistics/index.md); and it never reports a selected set as though it were a finding.

The trading stake is a warning the course issues in the imperative. [Feature and Signal Engineering](../../part-04-strategy-development/05-feature-and-signal-engineering.md) closes with the instruction that a signal report must record "the size of the search that produced it", and states the reason without hedging: "The IC you report is the maximum of every IC you computed, whether you admit it or not — every feature variant, every risk adjustment, every horizon you glanced at and moved past." That lesson's own family of twenty-four trials needs a $t$ of $3.1$ to clear a Bonferroni bar and its best signal brings $0.46$. Sections 2 and 4 are the mechanism behind the sentence, measured on data where the correct answer is known to be nothing.

## Best Subset, Stepwise and Screening Are Three Traversals of One Space, and They Differ in What They Can Afford Rather Than in What They Want

With $p$ candidate predictors there are $2^{p}$ subsets, and at $p=50$ that is $1.1\times10^{15}$. Nobody enumerates it, so every procedure in use is a heuristic traversal, and the traversals differ in a way that matters less than the fact that all of them search.

??? note "Proof that forward selection can miss the best pair of predictors entirely, so greedy and optimal agree only under conditions on the design that are not checked"

    Construct three centred predictors and a response with $x_1$ marginally most correlated with $y$ but $\{x_2,x_3\}$ jointly perfect. Let $u,v$ be orthonormal, set $x_2=u$, $x_3=v$, and $y=(u+v)/\sqrt2$, so the pair $\{x_2,x_3\}$ reproduces $y$ exactly and each alone achieves correlation $1/\sqrt2\approx0.707$. Now set $x_1=(u+v)/\sqrt2+\epsilon w$ for a third orthonormal $w$ and small $\epsilon$: its correlation with $y$ is $1/\sqrt{1+\epsilon^{2}}$, strictly greater than $0.707$. Forward selection takes $x_1$ first because it maximises the marginal criterion, and having taken it, the partial residual is $-\epsilon w/\sqrt{1+\epsilon^{2}}$, which is orthogonal to both $x_2$ and $x_3$; neither can reduce it, so the greedy path never recovers the exact pair. Best subset of size two returns $\{x_2,x_3\}$ with zero residual.

    The gap is not pathological — it is the generic situation whenever predictors are useful in combination rather than alone, which is the situation that motivates having more than one. Conditions under which greedy provably matches optimal exist and are strong: mutual incoherence of the design, or a restricted-isometry property, both of which bound how nearly collinear any small group of columns may be. Financial predictors built from overlapping windows of the same price series violate them routinely.

    The load-bearing fact is that the criterion being maximised is marginal while the object being built is joint. **A greedy search returns a defensible set rather than the best one, and the difference is invisible from the output because the sets it did not visit leave no trace.**

None of what follows depends on which traversal is used. The results below use forward selection and marginal screening because they are the two most common, and the mechanism — that a maximum over candidates is not distributed like a fixed candidate — applies to best subset most severely of all, since it searches hardest.

## A Search Over Nothing Returns a Model Whose $F$-Test Rejects, Because the Test Was Derived for Columns Named in Advance

The overall $F$-statistic of a regression has an exact $F$ distribution under the null, and that derivation assumes the columns were fixed before the data was seen. The prediction is that violating the assumption breaks the test in proportion to the size of the search:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(14041)
n, reps, q = 100, 3000, 5


def forward(X, y, q):
    """Greedily add the column that most reduces the residual sum of squares."""
    chosen, r = [], y.copy()
    for _ in range(q):
        left = [j for j in range(X.shape[1]) if j not in chosen]
        g = [abs(X[:, j] @ r) / np.linalg.norm(X[:, j]) for j in left]
        chosen.append(left[int(np.argmax(g))])
        A = X[:, chosen]
        r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return chosen, r


print(f"  forward selection of {q} predictors from p candidates, n = {n},"
      f" {reps:,} datasets")
print("  every column and the response are independent standard normals:"
      " nothing is real")
print("      p    mean R^2   nominal F p-value: median   P(p < 0.05)   honest rate")
for p in (5, 10, 25, 50, 100):
    r2, pv = np.empty(reps), np.empty(reps)
    for i in range(reps):
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        X -= X.mean(0)                                  # centring absorbs the intercept
        y -= y.mean()
        _, r = forward(X, y, q)
        tss, rss = np.sum(y**2), np.sum(r**2)
        r2[i] = 1 - rss / tss
        f = (tss - rss) / q / (rss / (n - 1 - q))       # as if the q were pre-named
        pv[i] = stats.f.sf(f, q, n - 1 - q)
    print(f"    {p:4d}    {r2.mean():8.4f}   {np.median(pv):25.6f}"
          f"   {np.mean(pv < 0.05):11.4f}   {0.05:11.2f}")
# =>   forward selection of 5 predictors from p candidates, n = 100, 3,000 datasets
#      every column and the response are independent standard normals: nothing is real
#          p    mean R^2   nominal F p-value: median   P(p < 0.05)   honest rate
#           5      0.0500                    0.502921        0.0447          0.05
#          10      0.0924                    0.124008        0.2937          0.05
#          25      0.1578                    0.007277        0.8767          0.05
#          50      0.2070                    0.000567        0.9963          0.05
#         100      0.2617                    0.000029        1.0000          0.05
```

The first row is the control and it validates the machinery. With $p=5$ candidates and $q=5$ selected there is no search — every column is taken — and the test behaves exactly as derived: $R^{2}$ of $0.0500$, which is $q/(n-1)=5/99$ to three decimals, a median $p$-value of $0.502921$ against a uniform distribution's $0.5$, and a rejection rate of $0.0447$ against a nominal $0.05$. Nothing is wrong with the $F$-test.

Every subsequent row changes only the number of candidates the same five slots were chosen from. At $p=10$ the rejection rate is already $0.2937$, six times nominal. At $p=25$ it is $0.8767$ and the median $p$-value has fallen to $0.007277$, so the *typical* dataset now looks significant at the one percent level. At $p=50$ it is $0.9963$. At $p=100$ the test rejects on every one of three thousand datasets, and the median $p$-value is $0.000029$ — a number that in a research note would be reported as overwhelming evidence, produced by a response that is independent of every column it was regressed on.

The $R^{2}$ column shows the same thing in the currency people quote. It runs $0.0500$, $0.0924$, $0.1578$, $0.2070$, $0.2617$: a fivefold increase driven entirely by widening the pool of candidates, with the fitted model's size held at five throughout. A researcher who tried harder — collected more candidate features, which is universally regarded as diligence — would report a better model on identical information.

**The distribution of a maximum is not the distribution of a draw, and every in-sample statistic printed after a search is computed from the wrong one of those two.**

## Screening Before the Resampling Loop Puts the Test Rows Into Their Own Training Set, and the Loop Cannot See It

The natural defence against section 2 is cross-validation, which scores on data the fit never saw. It works, provided the screening is inside it. Placing the screen first — rank all candidates on the full sample, keep the promising ones, then cross-validate the reduced set — is the single most common error in applied predictive modelling, and it defeats the resampling completely:

```python
import numpy as np

rng = np.random.default_rng(14043)
n, q, K, reps = 100, 20, 10, 400


def cv(X, y, inside):
    """Ten-fold CV; `inside` decides whether screening sees the test rows."""
    if not inside:
        keep = np.argsort(-np.abs(X.T @ y))[:q]           # screened on everything
    edges = np.linspace(0, n, K + 1).astype(int)
    err = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        te = np.arange(a, b)
        tr = np.setdiff1d(np.arange(n), te)
        sel = np.argsort(-np.abs(X[tr].T @ y[tr]))[:q] if inside else keep
        A = X[np.ix_(tr, sel)]
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        err += np.sum((y[te] - X[np.ix_(te, sel)] @ beta) ** 2)
    return err / n


print(f"  screening {q} of p columns, then {K}-fold cross-validation, n = {n},"
      f" {reps} datasets")
print("  the columns and the response are independent: the honest CV error is 1.00")
print("        p   screened once on all rows   screened inside each fold"
      "   predicting the mean")
for p in (50, 200, 1000, 5000):
    out = np.zeros(2)
    for _ in range(reps):
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        X -= X.mean(0)
        y -= y.mean()
        out += (cv(X, y, False), cv(X, y, True))
    print(f"    {p:5d}   {out[0] / reps:25.4f}   {out[1] / reps:24.4f}"
          f"   {1.0:19.2f}")
# =>   screening 20 of p columns, then 10-fold cross-validation, n = 100, 400 datasets
#      the columns and the response are independent: the honest CV error is 1.00
#            p   screened once on all rows   screened inside each fold   predicting the mean
#           50                      1.0413                     1.3941                  1.00
#          200                      0.7547                     1.3823                  1.00
#         1000                      0.5698                     1.3342                  1.00
#         5000                      0.4524                     1.3228                  1.00
```

The middle column is what the correct procedure reports, and reading it first fixes the scale. It runs $1.3941$, $1.3823$, $1.3342$ and $1.3228$ — *worse* than the $1.00$ of predicting the mean, and rightly so, because fitting twenty useless parameters to ninety training rows inflates error by roughly $1+20/70$, which is $1.29$. The honest procedure says: these features are worthless and using twenty of them costs you a third of your variance. That statement is correct and is available at every candidate-pool size.

The left column is the same data, the same folds, the same estimator, with the screen moved outside the loop. It reads $1.0413$, $0.7547$, $0.5698$ and $0.4524$. At $p=5000$ the reported cross-validated error is less than half the variance of the response, which converts to an apparent out-of-sample $R^{2}$ of about $55\%$ on data containing no signal at all. The two columns differ by a factor of nearly three at that pool size, and they differ in sign of conclusion at every pool size beyond the first.

The mechanism is that the screen consumed the whole sample. Once the twenty survivors are chosen using all one hundred rows, every subsequent fold's held-out rows have already contributed to deciding which columns the model may use — so they are not held out, whatever the loop's structure implies. The folds are honest about the *coefficients* and silent about the *columns*, and the columns are where the fitting happened.

Note also that the leak *grows with the candidate pool* while the honest column barely moves. That asymmetry is the tell: a procedure whose reported performance improves as you add candidate features that are known to be noise is measuring the search, not the features.

**Cross-validation protects only the operations performed inside it, so any step that touched the response before the split has already been paid for out of the test set, and no amount of folding recovers it.**

## A Selected Coefficient's Distribution Is Conditioned on the Selection Event, So Nominal Intervals Do Not Cover and Do Not Widen Either

Sections 2 and 3 concern predictive claims. The inferential claim is worse, because the standard error attached to a selected coefficient is not merely too small — it is computed from a distribution the coefficient no longer has.

??? note "Proof that conditioning on being the largest of $p$ truncates the sampling distribution, so the usual interval is centred on a biased point and has the wrong length"

    Let $\hat\beta_1,\dots,\hat\beta_p$ be independent estimates, each $N(0,\tau^{2})$ under a global null, and let $J=\arg\max_j\lvert\hat\beta_j\rvert$. The reported quantity is $\hat\beta_J$, whose distribution is that of the maximum absolute value of $p$ normals — not $N(0,\tau^{2})$. Its expected magnitude grows like $\tau\sqrt{2\log p}$, so the point estimate is biased away from zero by an amount that increases without bound in the size of the search, while the reported standard error $\hat\tau$ estimates the spread of a *single* $\hat\beta_j$ and does not change with $p$ at all.

    The exact statement is that the valid reference distribution is the law of $\hat\beta_J$ *conditional on the event $\{J=j\}$*, which is the unconditional law truncated to the region where coordinate $j$ dominates. That truncation removes precisely the small values, so the conditional density has no mass near zero, and an interval built by adding $\pm t\hat\tau$ to a point drawn from it will miss zero almost surely once $p$ is large. Selective-inference methods repair this by inverting the truncated law rather than the untruncated one, producing intervals that are correctly centred and much wider.

    The load-bearing asymmetry is that the bias grows in $p$ and the reported width does not. **The interval fails in both of the ways an interval can fail at once — it is displaced and it is too short — and the second failure guarantees that collecting more candidates makes the first one less visible rather than more.**

Both halves of that, and the standard repair, on data where every true coefficient is exactly zero:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(14045)
n, reps = 80, 5000


def slope_ci(x, y):
    """Simple-regression slope, its standard error and a nominal 95% interval."""
    m = len(x)
    b = (x @ y) / (x @ x)
    s2 = np.sum((y - b * x) ** 2) / (m - 1)
    se = np.sqrt(s2 / (x @ x))
    t = stats.t.ppf(0.975, m - 1)
    return b, se, b - t * se, b + t * se


print(f"  every true coefficient is zero; the interval is a nominal 95% one, n = {n}")
print("      p   select on all rows: coverage   width   select on half, infer on"
      " half: coverage   width")
for p in (1, 5, 20, 100):
    cov = np.zeros(2)
    wid = np.zeros(2)
    for _ in range(reps):
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        X -= X.mean(0)
        y -= y.mean()
        j = int(np.argmax(np.abs(X.T @ y) / np.linalg.norm(X, axis=0)))
        _, _, lo, hi = slope_ci(X[:, j], y)              # same rows chose and fitted
        cov[0] += lo <= 0.0 <= hi
        wid[0] += hi - lo
        h = n // 2
        jj = int(np.argmax(np.abs(X[:h].T @ y[:h]) / np.linalg.norm(X[:h], axis=0)))
        _, _, lo, hi = slope_ci(X[h:, jj], y[h:])        # disjoint rows
        cov[1] += lo <= 0.0 <= hi
        wid[1] += hi - lo
    print(f"    {p:4d}   {cov[0] / reps:28.4f}   {wid[0] / reps:5.4f}"
          f"   {cov[1] / reps:34.4f}   {wid[1] / reps:5.4f}")
# =>   every true coefficient is zero; the interval is a nominal 95% one, n = 80
#          p   select on all rows: coverage   width   select on half, infer on half: coverage   width
#           1                         0.9508   0.4481                               0.9494   0.6481
#           5                         0.7734   0.4439                               0.9484   0.6484
#          20                         0.3402   0.4365                               0.9478   0.6473
#         100                         0.0058   0.4300                               0.9436   0.6530
```

The $p=1$ row is again the control, and again the classical machinery is exactly right: coverage $0.9508$ against a nominal $0.95$. There is no selection when there is one candidate.

Coverage then collapses: $0.7734$ at five candidates, $0.3402$ at twenty, $0.0058$ at a hundred. At $p=100$ a procedure advertising ninety-five percent confidence contains the truth in six of every thousand attempts. It is not approximately valid, not conservative in the other direction, not usable with a mental adjustment — it is wrong by a factor of a hundred and sixty.

The width column is the proof's second half and the part that makes the failure undetectable from the output. The intervals read $0.4481$, $0.4439$, $0.4365$ and $0.4300$ — they *narrow slightly* as the search widens. Nothing in the reported uncertainty registers that a search occurred. A reader comparing the $p=100$ result against the $p=1$ result sees a marginally tighter interval around a larger estimate and concludes, reasonably and wrongly, that the larger study found a stronger effect more precisely.

The right-hand pair is the repair, and it is arithmetic rather than theory: choose the predictor on one half of the rows, estimate and interval it on the other half. Coverage holds at $0.9494$, $0.9484$, $0.9478$ and $0.9436$ across the whole range of search sizes, because the second half never participated in the choice. The price is in the width column — $0.6481$ against $0.4481$, about $45\%$ wider — which is what estimating on forty rows instead of eighty costs, and it is the honest width rather than an inflated one.

**Sample splitting converts an invalid interval into a valid one by spending data, and the resulting interval is wider not because splitting damaged the estimate but because the narrow one was never describing anything.**

## What Survives a Search Is the Prediction and the Frequency, Not the Set

Three quantities came through the preceding sections intact, and they are worth naming because everything else on a selection output did not.

The first is out-of-sample prediction, provided the entire pipeline including the search sits inside the resampling loop. Section 3's middle column is such a number: $1.3228$, an unflattering result honestly obtained, and it answers the question of whether the procedure is worth running. The second is selection frequency. Refitting on a few hundred bootstrap resamples and printing how often each predictor is retained costs nothing and converts a binary claim into a measured one — the diagnostic [Regularization](../part-13-regression/05-regularization.md) applies to the lasso, where a true-zero decoy is selected $42.6\%$ of the time at the prediction-optimal penalty. The third is the count of candidates examined, which is the only input to the corrections of [Part XV](../part-15-multiple-testing/index.md) and, as [Data Snooping Bias](../part-15-multiple-testing/04-data-snooping-bias.md) measures, the only one that cannot be recovered after the fact.

What does not survive is the identity of the selected set treated as a conclusion. Section 1 shows the greedy path can miss the best pair outright; section 2 shows the winning set's in-sample statistics are drawn from the distribution of a maximum; section 4 shows its coefficients carry intervals that do not cover. A selected set is a summary of one sample's noise realization as much as of its structure, and the honest report of one is a frequency next to a prediction, not a list.

!!! note "Best subset, forward stepwise, backward elimination, marginal screening and the lasso are five searches over one space of $2^{p}$ subsets, and they differ in what they can afford"
    **Best subset** evaluates every subset and returns the optimum of the criterion, which is why it is the gold standard and why it is unavailable beyond about forty predictors. **Forward stepwise** adds the column that most improves the fit and never reconsiders, so it is $O(pq)$ and provably misses jointly-useful pairs by section 1's construction. **Backward elimination** starts full and removes, which sees joint effects the forward path cannot but requires $n>p$ to start at all. **Marginal screening** ranks by univariate association once and is the cheapest of all, at $O(p)$, which is why it is what gets applied to five thousand columns and why section 3 is written about it. **The lasso** searches implicitly, its penalty path visiting a sequence of subsets as $\lambda$ falls, and it is the only one of the five that is a convex program rather than a combinatorial search — which buys a unique solution and a fast algorithm, and buys nothing at all in the way of inferential validity, since a set chosen by data is a set chosen by data however elegantly. The five differ in cost and in which optima they can reach; they are identical in that each returns a subset determined by the response, which is the property that breaks every statistic downstream.

!!! warning "The size of the search is the one input to every correction and the one quantity that no dataset records"
    Everything needed to repair sections 2 and 4 is a function of how many candidates were examined, and that number lives only in the analyst's memory. Worse, it is systematically undercounted: the features that were built and discarded, the horizons glanced at, the transformations tried in a notebook cell that was later cleared, and the variants a colleague explored before handing over the promising one all belong in the count and none of them appear in the file. Section 2 measures the cost of undercounting — a search over $100$ candidates that is reported as though it were $5$ produces a median $p$-value of $0.000029$ and rejects $1.0000$ of the time on pure noise — and section 4 measures it as coverage of $0.0058$ against a claimed $0.95$. The course states the same thing as an operational rule in [Feature and Signal Engineering](../../part-04-strategy-development/05-feature-and-signal-engineering.md): "The IC you report is the maximum of every IC you computed, whether you admit it or not." **The free diagnostic is to run the entire pipeline — screening, selection, fitting, scoring, every step in the order you actually performed it — on a response that has been randomly permuted, and to record the best statistic it returns; that number is what your procedure produces from nothing, so a real result must beat it rather than beat zero, and if your permuted run returns an $R^{2}$ of $0.26$ and a $p$-value of $10^{-5}$, as section 2's does, then those are your zero points and any unpermuted result below them is not a finding but a restatement of how hard you looked.**

## A Selected Model Is a Random Variable, and Every Number Printed Beside It Assumes Otherwise

This page established that $2^{p}$ subsets make every practical procedure a heuristic, with forward selection provably unable to recover a jointly-perfect pair once a marginally-superior decoy is present; that searching inflates every in-sample statistic computed as though it had not happened, forward selection of five columns from $5$, $10$, $25$, $50$ and $100$ pure-noise candidates giving mean $R^{2}$ of $0.0500$, $0.0924$, $0.1578$, $0.2070$ and $0.2617$ and $F$-test rejection rates of $0.0447$, $0.2937$, $0.8767$, $0.9963$ and $1.0000$ against a nominal $0.05$, with the median $p$-value falling to $0.000029$; that cross-validation does not detect a leak that preceded it, screening twenty of $50$, $200$, $1000$ and $5000$ columns on the full sample returning $1.0413$, $0.7547$, $0.5698$ and $0.4524$ where the same folds with the screen moved inside report $1.3941$, $1.3823$, $1.3342$ and $1.3228$ against a mean-prediction baseline of $1.00$; and that a selected coefficient's nominal $95\%$ interval covered its true value of zero $0.9508$, $0.7734$, $0.3402$ and $0.0058$ of the time as the candidate count ran $1$ to $100$, while *narrowing* from $0.4481$ to $0.4300$, with sample splitting restoring coverage to $0.9436$ at a width of $0.6530$.

The pattern across the three exhibits is one pattern. A statistic derived for a fixed object is computed for an object that was chosen, and the choosing used the same data. Nothing in the arithmetic detects the substitution because the arithmetic was never given the search as an input — the $F$-distribution does not know how many columns were considered, the folds do not know when the screen ran, and the $t$-interval does not know it is describing a maximum. Each of the three failures is invisible in exactly the same way, and each is repaired by the same move: perform the search inside whatever structure is supposed to be validating it, or spend data to buy a portion of the sample that the search never touched.

That leaves the question this page has deliberately not answered. Sections 2 and 4 both scale with the number of candidates, and both repairs — resampling around the search, splitting the sample — control the damage without ever *quantifying* it. Turning the count of candidates into an explicit price, so that a single reported statistic can be corrected rather than merely quarantined, is [Part XV](../part-15-multiple-testing/index.md). The alternative is to stop choosing altogether: if the selected set is unstable and the winner's statistics are the distribution of a maximum, then averaging over the candidates rather than maximising over them removes the maximum from the procedure. That is [Model Averaging](05-model-averaging.md).

**Feature selection produces exactly one honest output, which is a prediction scored by a procedure that included the selection, and exactly one honest description of the chosen set, which is how often it would be chosen again.**
