# Model Diagnostics

A fitted regression is a weighted average, and the weights are not equal. [Linear Algebra Review](../part-01-mathematical-foundations/05-linear-algebra-review.md) names the object that holds them: the hat matrix $H=X(X^\top X)^{-1}X^\top$, "symmetric, idempotent, and has trace equal to $p$", whose diagonal entries "are the leverage values used in Model Diagnostics." What that diagonal measures is how much of its own fitted value each observation supplies, and the uncomfortable consequence is immediate — a point with high leverage drags the line toward itself and is therefore left with a small residual. The observation doing the most damage is systematically among the ones that look most innocent. Below, a point generated eight units off the true line ends up with a residual of $2.4293$, because the fit absorbed $73\%$ of its own error, and the slope it produces is $0.8629$ where the truth is $0.5$.

This page covers the hat matrix and its diagonal, the closed form for leave-one-out deletion, Cook's distance as the product of leverage and residual that neither factor supplies alone, the limit of single-deletion reasoning when more than one observation is bad, and what robust standard errors do and do not repair. It does not derive the projection or the normal equations, which is [Multiple Linear Regression](02-multiple-linear-regression.md); it does not study the residual vector's own distribution or resample it, which is [Residual Analysis](07-residual-analysis.md); it corrects no ill-conditioning by penalty, which is [Regularization](05-regularization.md); it tests no hypothesis about a coefficient, which is [Part XII](../part-12-hypothesis-testing/index.md); it compares no two models, which is [Part XIV](../part-14-model-selection/index.md); it builds no resampling scheme from scratch, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); and it never treats a clean diagnostic table as evidence that a fit is sound.

The trading stake is a strategy the course kills three separate ways. [Seasonality and Calendar Effects](../../part-04-strategy-development/04-seasonality-and-calendar-effects.md) reports that turn-of-month "earns 33% of SPY's entire twenty-five-year return in 19% of the days" at `+5.2 bp/day vs other days +2.4 bp  (t = +0.73)`, and names the framing itself as the tell: "Concentration-of-return framings inherit all the noise of the sample paths they sum." A handful of observations carrying a result is what this page calls leverage, and section 4 measures what happens to the standard remedy — find the worst point and drop it — when the concentration is spread across several observations rather than one.

## The Hat Matrix Is the Projection Onto the Column Space, and Its Diagonal Prices Each Observation's Vote in Its Own Fitted Value

Since $\hat y=Hy$, the partial derivative $\partial\hat y_i/\partial y_i$ is exactly $h_{ii}$: move one response by a unit and its own fitted value moves by $h_{ii}$ of that unit. The quantity is a property of $X$ alone — it does not involve $y$ — so leverage is decided by the design before a single response is observed.

??? note "Proof that $H$ is symmetric and idempotent with $\operatorname{tr}H=p$, so the leverages lie in $[0,1]$ and average to $p/n$"

    Symmetry is immediate from $(X^\top X)^{-1}$ being symmetric. Idempotence is a cancellation:
    $$H^{2}=X(X^\top X)^{-1}X^\top X(X^\top X)^{-1}X^\top=X(X^\top X)^{-1}X^\top=H.$$
    For the trace, use $\operatorname{tr}(AB)=\operatorname{tr}(BA)$ with $A=X$ and $B=(X^\top X)^{-1}X^\top$:
    $$\operatorname{tr}H=\operatorname{tr}\big((X^\top X)^{-1}X^\top X\big)=\operatorname{tr}(I_p)=p,$$
    so $\sum_i h_{ii}=p$ exactly and the average leverage is $p/n$ for every design. The bound follows from symmetry plus idempotence: $h_{ii}=(H^{2})_{ii}=\sum_j h_{ij}^{2}=h_{ii}^{2}+\sum_{j\neq i}h_{ij}^{2}$, so $h_{ii}-h_{ii}^{2}=\sum_{j\neq i}h_{ij}^{2}\ge0$, giving $0\le h_{ii}\le1$. Equality at $1$ forces every off-diagonal $h_{ij}$ to vanish, meaning the point is fitted exactly and contributes nothing to any other fitted value.

    The consequence for residuals is the part that matters here. Since $e=(I-H)y$ and $I-H$ is also a projection, $\operatorname{var}(e_i)=\sigma^{2}(1-h_{ii})$ — a point with leverage near one has a residual with variance near zero, *whatever its true error was*. The fit is not detecting that such a point fits well; it is guaranteeing it.

    The load-bearing identity is $\operatorname{tr}H=p$. Leverage is a fixed budget of $p$ units distributed across $n$ observations by the design, so concentrating it somewhere requires taking it from somewhere else. **A regression has exactly $p$ units of self-determination to hand out, and any observation that collects an outsized share of it is by construction an observation whose residual cannot report its own error.**

Both halves are directly measurable:

```python
import numpy as np

rng = np.random.default_rng(13061)
n, p = 60, 2

x = rng.standard_normal(n)
x[-1] = 15.0                                  # one predictor value far from the rest
X = np.column_stack([np.ones(n), x])
H = X @ np.linalg.solve(X.T @ X, X.T)
h = np.diag(H)

y = 1.0 + 0.5 * x + rng.standard_normal(n)
y[-1] += 8.0                                  # and it is put 8 units off the line
b = np.linalg.lstsq(X, y, rcond=None)[0]
e = y - X @ b
s = np.sqrt(e @ e / (n - p))

print("  n = 60, intercept plus one predictor, with x[59] placed far from the rest")
print(f"    max |H - H'|            {np.abs(H - H.T).max():.2e}")
print(f"    max |HH - H|            {np.abs(H @ H - H).max():.2e}")
print(f"    trace(H)                {h.sum():.4f}   (p = {p})")
print(f"    mean h_ii               {h.mean():.4f}   (p/n = {p / n:.4f})")
print(f"    min, max h_ii           {h.min():.4f}, {h.max():.4f}")
print("  the far point was generated 8.0 above the line; residuals see almost none of it:")
print("    row          h_ii   residual   resid/s   fitted pull dy_hat/dy")
order = [n - 1, int(np.argsort(h)[len(h) // 2]), int(np.argmin(h))]
for tag, i in zip(("far point", "median h", "lowest h"), order):
    print(f"    {tag:11s}  {h[i]:6.4f}   {e[i]:8.4f}   {e[i] / s:7.3f}   {H[i, i]:20.4f}")
b_del = np.linalg.lstsq(X[:-1], y[:-1], rcond=None)[0]
print(f"    slope with far point {b[1]:.4f}, without it {b_del[1]:.4f}")
# =>   n = 60, intercept plus one predictor, with x[59] placed far from the rest
#        max |H - H'|            2.78e-17
#        max |HH - H|            2.78e-17
#        trace(H)                2.0000   (p = 2)
#        mean h_ii               0.0333   (p/n = 0.0333)
#        min, max h_ii           0.0167, 0.7312
#      the far point was generated 8.0 above the line; residuals see almost none of it:
#        row          h_ii   residual   resid/s   fitted pull dy_hat/dy
#        far point    0.7312     2.4293     2.334                 0.7312
#        median h     0.0195     0.0622     0.060                 0.0195
#        lowest h     0.0167    -1.3832    -1.329                 0.0167
#        slope with far point 0.8629, without it 0.4239
```

The algebraic claims hold to machine precision: $H$ is symmetric and idempotent to $2.78\times10^{-17}$, its trace is $2.0000$ against $p=2$, and the mean leverage is $0.0333$ against $p/n=0.0333$. None of this is approximate.

The distribution of that fixed budget is the finding. One observation holds $0.7312$ of the total $2$ units while the median holds $0.0195$ — a factor of thirty-seven — purely because its predictor value sits far from the rest. And the consequence for detection is exactly the one the proof warned about: the point was *constructed* to be $8.0$ units off the line, and its residual is $2.4293$. The fit moved to meet it, absorbing $73.12\%$ of the discrepancy, and what remains is a residual of $2.33$ standard deviations — noticeable, but comfortably inside the range a sixty-observation sample produces by chance, and smaller than one would get by planting the same error at a typical design point.

The damage is in the last line. The slope is $0.8629$ with that point and $0.4239$ without it, against a truth of $0.5$: a single observation out of sixty doubles the estimated effect, and the residual that was supposed to announce it has been suppressed by the very influence that makes it dangerous.

**Leverage and residual size run in opposite directions by construction, so the diagnostic instinct of looking for large residuals searches hardest exactly where the fit has already hidden the evidence.**

## Deleting One Observation Has a Closed Form, So Every Leave-One-Out Diagnostic Costs One Fit Rather Than $n$

The natural question — how would the answer change without observation $i$ — sounds like it costs $n$ regressions. It costs none.

??? note "Proof that $\hat\beta-\hat\beta_{(i)}=(X^\top X)^{-1}x_ie_i/(1-h_{ii})$, the leave-one-out identity, so every deletion diagnostic is a function of the single full fit"

    Removing row $i$ changes the cross-product matrix by a rank-one update, $X_{(i)}^\top X_{(i)}=X^\top X-x_ix_i^\top$, and the Sherman–Morrison formula inverts it in closed form:
    $$\big(X^\top X-x_ix_i^\top\big)^{-1}=(X^\top X)^{-1}+\frac{(X^\top X)^{-1}x_ix_i^\top(X^\top X)^{-1}}{1-h_{ii}},$$
    the denominator being nonzero precisely when $h_{ii}<1$. Applying this to $X_{(i)}^\top y_{(i)}=X^\top y-x_iy_i$ and simplifying gives
    $$\hat\beta_{(i)}=\hat\beta-\frac{(X^\top X)^{-1}x_i\,e_i}{1-h_{ii}}.$$
    Every ingredient — $\hat\beta$, $e_i$, $h_{ii}$, $(X^\top X)^{-1}$ — comes from the one fit already computed. The deleted-point prediction error follows as $y_i-x_i^\top\hat\beta_{(i)}=e_i/(1-h_{ii})$, so the leave-one-out prediction residual is the ordinary residual inflated by $1/(1-h_{ii})$, and the sum of their squares is the PRESS statistic. At $h_{ii}=0.7312$ that inflation is a factor of $3.72$: the far point above, which the fit predicts to within $2.4293$, would be missed by $9.04$ if it had not been used to fit.

    The load-bearing quantity is $1-h_{ii}$, appearing in the denominator of every deletion formula on this page. **Leave-one-out cross-validation for a linear model is not an approximation or a resampling scheme; it is an algebraic identity available for free, and any code that refits $n$ times is computing something it already had.**

Two standard quantities are built from this. The **externally studentized residual** divides $e_i$ by an estimate of its standard deviation computed *without* observation $i$, $t_i=e_i/\{s_{(i)}\sqrt{1-h_{ii}}\}$, which removes the point's own contribution from the scale it is judged against. **Cook's distance** measures the whole coefficient vector's movement,

$$D_i=\frac{(\hat\beta-\hat\beta_{(i)})^\top X^\top X(\hat\beta-\hat\beta_{(i)})}{p\,s^{2}}=\frac{e_i^{2}}{p\,s^{2}}\cdot\frac{h_{ii}}{(1-h_{ii})^{2}},$$

whose factored form is the whole of the next section: a residual term multiplied by a leverage term.

## Cook's Distance Multiplies Leverage by Residual Because Neither Factor Alone Identifies a Point That Moves the Answer

Both ingredients are necessary and neither is sufficient, which is checkable by planting one extra observation in four positions against a fixed clean sample:

```python
import numpy as np

rng = np.random.default_rng(13063)
n, p = 60, 2

x0 = rng.standard_normal(n - 1)
y0 = 1.0 + 0.5 * x0 + rng.standard_normal(n - 1)

print("  one extra observation added to the same 59 clean points, placed four ways")
print("    extra point          h_ii   stud resid   Cook D   4/n rule   slope   d slope")
for tag, xa, ya in (("typical", 0.3, 1.0 + 0.5 * 0.3),
                    ("far x, on the line", 10.0, 1.0 + 0.5 * 10.0),
                    ("central, y off by 8", 0.0, 1.0 + 8.0),
                    ("far x, y off by 8", 10.0, 1.0 + 0.5 * 10.0 + 8.0)):
    x = np.append(x0, xa)
    y = np.append(y0, ya)
    X = np.column_stack([np.ones(n), x])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    s2 = e @ e / (n - p)
    i = n - 1
    s2_i = ((n - p) * s2 - e[i] ** 2 / (1 - h[i])) / (n - p - 1)
    t = e[i] / np.sqrt(s2_i * (1 - h[i]))                  # externally studentized
    d = e[i] ** 2 * h[i] / (p * s2 * (1 - h[i]) ** 2)      # Cook's distance
    b_del = np.linalg.lstsq(X[:-1], y[:-1], rcond=None)[0]
    print(f"    {tag:19s}  {h[i]:6.4f}   {t:10.3f}   {d:6.3f}   "
          f"{'FLAG' if d > 4 / n else '  - ':8s}   {b[1]:5.3f}   {b[1] - b_del[1]:+6.3f}")
# =>   one extra observation added to the same 59 clean points, placed four ways
#        extra point          h_ii   stud resid   Cook D   4/n rule   slope   d slope
#        typical              0.0171        0.081    0.000     -        0.569   +0.000
#        far x, on the line   0.5739       -0.464    0.147   FLAG       0.534   -0.034
#        central, y off by 8  0.0168        9.546    0.306   FLAG       0.556   -0.013
#        far x, y off by 8    0.5739        5.751   14.340   FLAG       0.993   +0.424
```

Read the last column first, because it is the ground truth: the four points move the slope by $+0.000$, $-0.034$, $-0.013$ and $+0.424$. Only the fourth matters, and by a wide margin.

Now read the two single-factor diagnostics against it. **Leverage alone** is $0.5739$ for both the second and fourth rows, identically, and those two points differ in effect by a factor of twelve — a high-leverage point sitting on the line does nothing at all, because it has nothing to pull toward. **Residual alone** is worse than uninformative: the largest studentized residual in the table, $9.546$, belongs to the central point that moves the slope by $-0.013$, while the point that moves it by $+0.424$ scores only $5.751$. Ranking observations by residual puts the harmless one first and the destructive one second.

Cook's distance, being the product, gets it right: $0.000$, $0.147$, $0.306$, $14.340$. The dangerous point scores forty-seven times the next-highest, and the ordering matches the $\Delta$-slope column exactly. That is the whole argument for the product form.

The `4/n` column is a caution about the conventional cutoff. At $n=60$ the threshold is $0.0667$, and it flags three of the four rows, including the two that change the slope by $0.034$ and $0.013$. The rule of thumb is a screen with a high false-alarm rate, not a test, and reading it as a test produces a stream of investigations into points that do not matter — which is the standard route to ignoring it entirely.

!!! note "Leverage, the hat value, the self-influence and the $i$th diagonal of the projection matrix are four names for $h_{ii}$"
    **Leverage** is the statistical name and carries the mechanical metaphor: distance from the centre of the design gives an observation a longer lever on the fitted line. The **hat value** is the same number named after $H$, the matrix that "puts the hat on $y$". The **self-influence** $\partial\hat y_i/\partial y_i$ is its operational definition and the one to reason with, since it says directly what fraction of an observation's own fitted value the observation supplies. The **diagonal of the projection matrix** is the geometric reading, and it is what makes $\sum_i h_{ii}=p$ obvious rather than surprising. Two adjacent terms are worth keeping separate from all four. **Influence** is not leverage: it is leverage combined with residual, which is Cook's distance, and the second row of the table above is a high-leverage point with no influence. And in the machine-learning vocabulary the diagonal of a smoother matrix is called the **effective degrees of freedom** when summed, which is the same $\operatorname{tr}H=p$ generalized to fits that are not projections — the quantity a ridge fit has less than $p$ of, since its shrinkage factors $d_i^{2}/(d_i^{2}+\lambda)$ sum to less than the column count.

## A Single Deletion Repairs a Single Bad Observation and Nothing Else, So the Diagnostic's Reach Is One Observation Wide

Every quantity above is defined by removing one point. The standard workflow inherits that: compute Cook's distance, find the maximum, drop it, refit. What that workflow does as the number of bad observations grows is measurable:

```python
import numpy as np

n, p = 60, 2


def build(k, rng):
    x = rng.standard_normal(n)
    y = 1.0 + 0.5 * x + rng.standard_normal(n)
    if k:
        x[-k:] = 9.0 + 0.05 * np.arange(k)     # a matched cluster, far out and low
        y[-k:] = 1.0 + 0.5 * x[-k:] - 7.0
    return np.column_stack([np.ones(n), x]), y


def cooks(X, y):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    s2 = e @ e / (len(y) - X.shape[1])
    return b, e**2 * h / (X.shape[1] * s2 * (1 - h) ** 2)


def lms(X, y):
    """Least median of squares, exhaustive over all n(n-1)/2 elemental pairs."""
    best, bres = None, np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if X[i, 1] == X[j, 1]:
                continue
            sl = (y[j] - y[i]) / (X[j, 1] - X[i, 1])
            m = np.median((y - (y[i] - sl * X[i, 1]) - sl * X[:, 1]) ** 2)
            if m < bres:
                best, bres = sl, m
    return best


print(f"  a growing cluster of matched outliers, n = {n}, true slope 0.5000,")
print(f"  Cook's D threshold 4/n = {4 / n:.4f}")
print("    k   OLS slope   max Cook D   flagged   drop worst   drop all k   LMS")
for k in (0, 1, 2, 4, 6, 8):
    rng = np.random.default_rng(13065)
    X, y = build(k, rng)
    b, d = cooks(X, y)
    keep = np.ones(n, bool)
    keep[int(np.argmax(d))] = False
    b_one = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    kk = np.ones(n, bool)
    if k:
        kk[-k:] = False
    b_all = np.linalg.lstsq(X[kk], y[kk], rcond=None)[0]
    print(f"    {k:1d}   {b[1]:+9.4f}   {d.max():10.4f}   "
          f"{'yes' if d.max() > 4 / n else 'no ':7s}   {b_one[1]:+10.4f}"
          f"   {b_all[1]:+10.4f}   {lms(X, y):+.4f}")
# =>   a growing cluster of matched outliers, n = 60, true slope 0.5000,
#      Cook's D threshold 4/n = 0.0667
#        k   OLS slope   max Cook D   flagged   drop worst   drop all k   LMS
#        0     +0.4126       0.1133   yes          +0.4679      +0.4126   +0.4434
#        1     -0.0122       8.9974   yes          +0.3920      +0.3920   +0.4434
#        2     -0.1117       0.8036   yes          -0.0149      +0.3854   +0.4434
#        4     -0.1776       0.0906   yes          -0.1538      +0.3546   +0.4434
#        6     -0.1954       0.0760   yes          -0.2049      +0.3980   -0.2438
#        8     -0.2108       0.0932   yes          -0.2199      +0.3368   -0.2130
```

With a single bad observation the machinery is superb. Cook's distance reads $8.9974$ — a hundred and thirty times the threshold, impossible to miss — and dropping that one point restores the slope from $-0.0122$ to $+0.3920$. This is the case the diagnostic was designed for, and it works.

Adding a second matched observation breaks the workflow while leaving the alarm ringing. The maximum Cook's distance *falls* from $8.9974$ to $0.8036$ — more contamination, an eleven-fold weaker signal, because each of the pair now hides behind the other's identical pull. It is still above the threshold, so the point is still flagged, and that is precisely what makes the situation dangerous: an analyst who follows the flag, deletes the worst point and refits gets a slope of $-0.0149$ against a truth of $0.5$. The diagnostic fired, the prescribed remedy was applied, and the answer is still wrong. Deleting both recovers $+0.3854$.

By $k=4$ and $k=6$ the maximum Cook's distance has collapsed to $0.0906$ and $0.0760$, hovering just above a threshold that the *uncontaminated* $k=0$ sample already exceeds at $0.1133$ — so the signal from four genuinely planted outliers is weaker than the routine false alarm from clean data. Meanwhile the slope has degraded to $-0.1776$ and $-0.1954$, and deleting the single worst point now makes things *worse* rather than better: $-0.1538$ and $-0.2049$. The two columns move in opposite directions, which is the breakdown-point statement made numerical.

The high-breakdown alternative holds where the deletion workflow fails and then fails in its own way. Least median of squares returns $+0.4434$ for every $k$ up to $4$ — untouched by contamination that has driven least squares to $-0.1776$ — and then breaks at $k=6$, returning $-0.2438$. The reason is worth stating because it is not a defect of the estimator: the planted cluster sits at $(9,-1.5)$ and the clean bulk is centred near $(0,1)$, so a line of slope about $-0.28$ passes near both. Once six observations sit on that line, they are no longer outliers relative to it, and a criterion that asks which line the majority of the data is close to has no basis for preferring the true one. Robustness buys resistance to points that disagree with the bulk, not to points arranged so the bulk agrees with them.

**Every diagnostic on this page is computed from a fit that the suspect observations helped produce, so the quantity being used to indict them has already been corrupted by them, and the corruption grows in exactly the cases where the indictment matters most.**

## Robust Standard Errors Repair the Denominator and Leave the Estimator Alone, Which Is Why They Never Repair a Wrong Mean Function

The other standard response to a failed assumption is to keep $\hat\beta$ and recompute its standard error under weaker conditions. The sandwich estimator replaces $\sigma^{2}(X^\top X)^{-1}$ with $(X^\top X)^{-1}\big(\sum_i e_i^{2}x_ix_i^\top\big)(X^\top X)^{-1}$, which is consistent under arbitrary heteroskedasticity, and the HAC version of [Simple Linear Regression](01-simple-linear-regression.md) extends it to autocorrelation — the correction the course applies with `cov_type="HAC"` throughout.

What this fixes is precise and narrow. It fixes the *second* moment of the sampling distribution of an estimator that is already estimating the right thing. It does not change $\hat\beta$ by a single digit, so it cannot repair the leverage problem of section 1, where the point estimate itself was $0.8629$ instead of $0.5$; it cannot repair the contamination of section 4, where the slope was negative; and it cannot repair a wrong mean function, since a misspecified model's coefficients converge to the best linear approximation and the sandwich estimator obligingly supplies a correct standard error *for that*. The result is a confidence interval that covers the wrong quantity at exactly its nominal rate.

The division of labour is worth stating as a rule. Robust standard errors are the right tool when the estimator is fine and its error bar is not — heteroskedastic or dependent errors around a correct mean. Robust *estimators*, like the least median of squares above, are the right tool when the estimator itself is being moved by a minority of the data. Reaching for the first when the problem is the second produces a well-calibrated statement about a corrupted number, and nothing in the output distinguishes that from success.

!!! warning "Every influence measure is computed from the fit it is meant to criticize, so the diagnostics look best exactly when the contamination is worst"
    The failure is silent in both directions. A high-leverage point is guaranteed a small residual by $\operatorname{var}(e_i)=\sigma^{2}(1-h_{ii})$, so the observation with the most influence is systematically among those the residual column ranks as least suspicious — above, a point planted $8.0$ off the line produced a residual of $2.4293$ while doubling the slope. And when several observations agree with each other, each one's deletion statistic is computed against a fit the others are still holding in place, so the alarm weakens as the damage grows: Cook's distance fell from $8.9974$ at one bad point to $0.0906$ at four, below the $0.1133$ that clean data produced by chance, while the slope went from $-0.0122$ to $-0.1776$. Following the standard remedy at that point makes the estimate worse, from $-0.1776$ to $-0.1538$ and from $-0.1954$ to $-0.2049$. **The free diagnostic is to stop asking the fit about itself: refit on a few hundred random half-samples and compare the spread of the coefficient across them with the standard error the full fit reported, since a regression whose subsample spread is several times its own reported standard error is a regression a small set of rows is running — and follow it with a high-breakdown fit such as the least median of squares above, whose disagreement with least squares is a measurement of contamination rather than an opinion about it, remembering that it too fails once the bad points are numerous enough to constitute a majority story of their own.**

## Every Diagnostic Here Is Computed From the Fit It Is Meant to Criticize, and That Is the Limit of All of Them

This page established that $H$ is symmetric and idempotent to $2.78\times10^{-17}$ with $\operatorname{tr}H=2.0000$ and mean leverage $0.0333=p/n$, so leverage is a fixed budget of $p$ units distributed by the design alone; that $\operatorname{var}(e_i)=\sigma^{2}(1-h_{ii})$ makes an influential point's residual small by construction, so an observation planted $8.0$ units off the line returned a residual of $2.4293$ while moving the slope from $0.4239$ to $0.8629$; that Sherman–Morrison gives $\hat\beta-\hat\beta_{(i)}=(X^\top X)^{-1}x_ie_i/(1-h_{ii})$, making every leave-one-out quantity free and inflating the deleted prediction error by $1/(1-h_{ii})$, a factor of $3.72$ at the leverage above; that Cook's distance needs both factors, since leverage alone scored $0.5739$ identically for points differing twelvefold in effect and the largest studentized residual $9.546$ belonged to a point moving the slope by $-0.013$ while $5.751$ belonged to one moving it by $+0.424$, where the product ranked them $0.306$ against $14.340$; and that single-deletion reasoning reaches exactly one observation, with Cook's distance falling from $8.9974$ at $k=1$ to $0.0906$ at $k=4$ — below the $0.1133$ a clean sample produced — while the slope degraded from $-0.0122$ to $-0.1776$ and the prescribed remedy of dropping the worst point moved it to $-0.1538$.

The structural problem is circularity, and it is not fixable within the framework. Each diagnostic asks what the fit would look like without some observation, and computes the answer from a fit that includes it. With one bad point the contamination in the reference is small and the answer is excellent. With several the reference is itself corrupted, and — because the corruption is in the direction that makes the bad points look ordinary — the diagnostics degrade fastest precisely as the problem worsens. This is not a threshold that could be tuned or a statistic that could be improved; it is what "leave one out" means when more than one thing is wrong.

The escape is to compute something that does not condition on the full fit: a high-breakdown estimator that never trusted all the data, or resampling that measures the coefficient's spread directly. Both were reached for above and both are outside this page's subject. What remains inside it is the object all of these diagnostics were reading — the residual vector — which has a structure of its own that has been used repeatedly here without being derived. Residuals are not errors, they are errors with $p$ dimensions of the fit projected out of them, and that projection makes them correlated and unequally scaled before any assumption is violated. That is [Residual Analysis](07-residual-analysis.md).

**A diagnostic asks the fit to report on its own reliability, and the answer is trustworthy in exactly the cases where it was least needed.**
