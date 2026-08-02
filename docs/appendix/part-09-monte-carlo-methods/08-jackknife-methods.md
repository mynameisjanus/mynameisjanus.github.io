# Jackknife Methods

The jackknife is the bootstrap's older, weaker, deterministic sibling, and it is the only resampling scheme in this part that has no Monte Carlo error at all. It considers exactly $n$ resamples — the original data with one observation removed, once for each observation — so its answer is a function of the sample rather than of a seed, and running it twice gives the same number to the last bit. That is a real advantage and it is not why the method survives. It survives because it does two things the bootstrap does not do at all: it removes the leading term of a bias exactly, and it measures how much of a statistic rests on each single observation. Both are answers to questions practitioners ask constantly and rarely name.

This page covers the leave-one-out construction and the inflation factor that looks backwards until it is derived, bias correction as the jackknife's strongest result, the non-smooth statistics on which its variance estimate does not converge at all, the acceleration constant that puts a jackknife inside every BCa interval, and leave-one-out as a sensitivity analysis rather than a standard error. It does not resample with replacement, which is [Bootstrap Methods](07-bootstrap-methods.md); it builds no confidence-interval taxonomy, which is [Part XI](../part-11-parameter-estimation/index.md); it does not use leave-one-out for model selection, which is [Part XIV](../part-14-model-selection/index.md); it derives no influence function in general, which is [Part XIII](../part-13-regression/index.md); it constructs no test, which is [Part XII](../part-12-hypothesis-testing/index.md); and it does not certify a strategy.

The trading stake is a procedure the course performs without naming. [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) warns that a heavy tail "strains the machinery beneath the bootstrap" and prescribes a remedy in five words — "check stability by deleting the worst day" — which is a leave-one-out calculation, done once, on the observation most likely to matter. That same lesson calls `method="BCa"` and quotes the resulting interval, and the acceleration constant inside that call is computed from all $n$ leave-one-out values. The fourth section measures what it buys, and the answer is the last eight-tenths of a percentage point of coverage.

## Delete One, Recompute, and Read the Spread

Let $\hat\theta=\hat\theta(x_1,\dots,x_n)$ be a statistic and let $\hat\theta_{(i)}$ be the same statistic computed with observation $i$ removed. Write $\hat\theta_{(\cdot)}$ for the average of the $n$ leave-one-out values. The **jackknife estimate of variance** is

$$\hat v_{\text{jack}}=\frac{n-1}{n}\sum_{i=1}^{n}\big(\hat\theta_{(i)}-\hat\theta_{(\cdot)}\big)^{2},$$

and the factor $(n-1)/n$ is the part that looks wrong. A sample variance carries $1/(n-1)$; this carries its reciprocal, inflating the sum by nearly $n$. The reason is that the leave-one-out values are far *less* dispersed than the original observations — each differs from the full-sample statistic by roughly $1/n$ of an observation's influence — so the raw spread understates the sampling variance by exactly that factor and the inflation puts it back.

??? note "Proof that the inflation factor is exactly what makes the jackknife exact for a mean"
    Take $\hat\theta=\bar x$. Then $\hat\theta_{(i)}=(S-x_i)/(n-1)$ with $S=\sum_j x_j$, and averaging over $i$ gives $\hat\theta_{(\cdot)}=\bar x$. The deviations are

    $$\hat\theta_{(i)}-\hat\theta_{(\cdot)}=\frac{S-x_i}{n-1}-\frac{S}{n}=\frac{n(S-x_i)-(n-1)S}{n(n-1)}=\frac{S-nx_i}{n(n-1)}=-\frac{x_i-\bar x}{n-1}.$$

    Each leave-one-out deviation is the corresponding residual shrunk by $n-1$, which is the precise sense in which the leave-one-out values are too tightly packed. Substituting,

    $$\hat v_{\text{jack}}=\frac{n-1}{n}\sum_i\frac{(x_i-\bar x)^{2}}{(n-1)^{2}}=\frac{1}{n(n-1)}\sum_i(x_i-\bar x)^{2}=\frac{s^{2}}{n},$$

    which is exactly the usual unbiased estimate of the variance of a sample mean. The factor was reverse-engineered from this case and then applied everywhere else.

    The load-bearing structure is the linearity. For a general $\hat\theta$ the deviations $\hat\theta_{(\cdot)}-\hat\theta_{(i)}$ are a finite-difference approximation to the **empirical influence function** — the derivative of the statistic with respect to the weight on observation $i$ — and the jackknife variance is the sum of squared influences divided by $n$, which is exactly the sandwich formula a delta-method calculation would produce. **The jackknife is not an alternative to the delta method; it is the delta method with the derivative computed numerically**, which is why it inherits the delta method's requirement that the derivative exist, and why the third section is about what happens when it does not.

## Bias Correction Is Where the Jackknife Earns Its Keep

Most estimators used in practice are biased at order $1/n$: the maximum-likelihood variance, any ratio of estimates, any nonlinear function of a mean. The jackknife removes that term exactly, without knowing what it is.

??? note "Proof that the leading bias term is removed exactly, and what is left behind"
    Suppose the bias admits an expansion in powers of $1/n$,

    $$\mathbb{E}[\hat\theta_n]=\theta+\frac{a_1}{n}+\frac{a_2}{n^{2}}+O(n^{-3}),$$

    with coefficients not depending on $n$. Each leave-one-out statistic is the same estimator on $n-1$ observations, so

    $$\mathbb{E}\big[\hat\theta_{(\cdot)}\big]=\theta+\frac{a_1}{n-1}+\frac{a_2}{(n-1)^{2}}+O(n^{-3}).$$

    Form the **bias-corrected** estimator $\tilde\theta=n\hat\theta-(n-1)\hat\theta_{(\cdot)}$. Its expectation is

    $$n\theta+a_1+\frac{a_2}{n}-(n-1)\theta-a_1-\frac{a_2}{n-1}=\theta+a_2\left(\frac1n-\frac1{n-1}\right)=\theta-\frac{a_2}{n(n-1)},$$

    so the $1/n$ term cancels identically and what remains is $O(n^{-2})$. The quantities $\tilde\theta_i=n\hat\theta-(n-1)\hat\theta_{(i)}$ are the **pseudovalues**, and $\tilde\theta$ is their average; treating them as approximately independent observations and taking their sample variance reproduces $\hat v_{\text{jack}}$, which is where the name "pseudovalue" earns its keep.

    The load-bearing hypothesis is that the bias expands in powers of $1/n$ with the *same* coefficients at $n$ and $n-1$. That holds for smooth functionals of means and fails for anything whose bias is dominated by a boundary or an extreme — a sample maximum has bias of order $1/\log n$ or worse depending on the tail, and the cancellation above does not occur. The correction is also not free: it adds variance, since $\tilde\theta$ is a difference of two nearly-equal large numbers, and for an estimator whose bias is already small relative to its standard error the trade is a loss.

```python
import numpy as np

rng = np.random.default_rng(9081)
reps, sr, nu = 200_000, 0.30, 6.0                              # annualized Sharpe, and a mild tail


def loo_stats(x):                                              # leave-one-out mean and variance
    n = x.shape[1]
    s, q = x.sum(axis=1, keepdims=True), (x ** 2).sum(axis=1, keepdims=True)
    m = (s - x) / (n - 1)
    return m, (q - x ** 2 - (n - 1) * m ** 2) / (n - 2)


print(f"  jackknife bias correction on {reps} samples, plain estimate against n*hat - (n-1)*bar")
print("   statistic              n     truth    plain    jackknifed    bias removed")
for n in (12, 36):
    x = rng.standard_normal((reps, n))
    plain = x.var(axis=1)                                      # the maximum-likelihood variance
    m, _ = loo_stats(x)
    loo = ((x ** 2).sum(axis=1, keepdims=True) - x ** 2) / (n - 1) - m ** 2
    jack = n * plain - (n - 1) * loo.mean(axis=1)
    print(f"  {'MLE variance':<20} {n:5d} {1.0:9.4f} {plain.mean():8.4f} {jack.mean():13.4f}"
          f" {1 - abs(jack.mean() - 1) / abs(plain.mean() - 1):15.3f}")

for n, per in ((36, 12), (120, 12)):
    z = rng.standard_normal((reps, n)) / np.sqrt(nu / (nu - 2))
    x = z * np.sqrt(nu / rng.chisquare(nu, (reps, n))) + sr / np.sqrt(per)
    plain = np.sqrt(per) * x.mean(axis=1) / x.std(axis=1, ddof=1)
    m, v = loo_stats(x)
    loo = np.sqrt(per) * m / np.sqrt(v)
    jack = n * plain - (n - 1) * loo.mean(axis=1)
    print(f"  {'annualized Sharpe':<20} {n:5d} {sr:9.4f} {plain.mean():8.4f} {jack.mean():13.4f}"
          f" {1 - abs(jack.mean() - sr) / abs(plain.mean() - sr):15.3f}")
# =>   jackknife bias correction on 200000 samples, plain estimate against n*hat - (n-1)*bar
#       statistic              n     truth    plain    jackknifed    bias removed
#      MLE variance            12    1.0000   0.9168        1.0002           0.998
#      MLE variance            36    1.0000   0.9720        0.9997           0.991
#      annualized Sharpe       36    0.3000   0.3128        0.3005           0.962
#      annualized Sharpe      120    0.3000   0.3038        0.2998           0.940
```

The first two rows are the case where the answer is known in advance and they are a check on the machinery rather than a discovery. The maximum-likelihood variance has expectation $\sigma^{2}(n-1)/n$, so at $n=12$ it should read $0.9167$ and does, and at $n=36$ it should read $0.9722$ and does. The jackknifed versions read $1.0002$ and $0.9997$, removing $99.8\%$ and $99.1\%$ of the bias — and the correction here is *exactly* the familiar $n/(n-1)$ factor, rediscovered by a procedure that was never told what the estimator was.

The Sharpe rows are the useful case, because nobody corrects them. A Sharpe ratio is a ratio of two estimates and is biased upward at order $1/n$: on thirty-six monthly observations of a mildly fat-tailed return series with a true annualized Sharpe of $0.30$, the plain estimator averages $0.3128$, an overstatement of $4.3\%$. The jackknifed version averages $0.3005$, removing $96.2\%$ of it. At a hundred and twenty months the plain bias has fallen to $0.0038$ and the jackknife removes $94\%$ of that.

**A four-percent overstatement is not the largest problem with a thirty-six-month Sharpe** — [Bootstrap Methods](07-bootstrap-methods.md) measures the interval around one at $2.31$ units wide, which dwarfs it — but it is systematic rather than random, so it does not average out across a book of managers, and it costs one line to remove. The asymmetry is worth stating plainly: the noise in a short track record is visible to everyone and gets discussed; the bias is invisible to everyone and never does.

## The Jackknife Fails Where the Statistic Is Not Smooth

The first proof identified the jackknife as a numerical derivative, and a numerical derivative of a function with a jump does not estimate anything. The canonical case is the median, and its failure is worth seeing in full because the mechanism is completely explicit.

??? note "Proof that the median's leave-one-out values take only three distinct numbers, and why that makes the variance estimate inconsistent"
    Let $n=2k+1$ be odd and write $x_{(0)}\leq\cdots\leq x_{(n-1)}$ for the order statistics, so the median is $x_{(k)}$. Removing one observation leaves an even count $2k$, whose median is the average of the two central order statistics of what remains. Three cases exhaust it:

    - removing any of the $k$ observations *below* the median shifts the window up, giving $\tfrac12\big(x_{(k)}+x_{(k+1)}\big)$;
    - removing the median itself gives $\tfrac12\big(x_{(k-1)}+x_{(k+1)}\big)$;
    - removing any of the $k$ observations *above* it gives $\tfrac12\big(x_{(k-1)}+x_{(k)}\big)$.

    So the $n$ leave-one-out values take exactly **three** distinct values regardless of $n$, and the jackknife variance is a function of the two spacings $x_{(k)}-x_{(k-1)}$ and $x_{(k+1)}-x_{(k)}$ alone. Those spacings are of order $1/n$ and, crucially, are *asymptotically exponential and independent* — they do not concentrate. Their squares therefore do not concentrate either, and $\hat v_{\text{jack}}/v_{\text{true}}$ converges in distribution to a non-degenerate random variable rather than to $1$.

    The load-bearing distinction is between *unbiased* and *consistent*. The jackknife variance for the median is asymptotically unbiased — the limiting ratio has mean one — so averaging it over many samples gives the right answer and any check based on averages will pass. It is not consistent: the estimate computed from *your* sample has a relative error that does not shrink as $n$ grows, because it depends on two order-statistic gaps rather than on the whole sample. **The bootstrap escapes this by resampling with replacement, which lets the median land on many order statistics rather than three.** The general repair is the delete-$d$ jackknife, which removes $d$ observations at a time with $d\to\infty$ and $d/n\to0$, restoring enough variety in the recomputed statistic to make the estimate concentrate.

```python
import numpy as np

rng = np.random.default_rng(9083)
reps, b, subsets = 1_000, 400, 400
print(f"  standard error of the sample median, {reps} samples, {b} resamples, truth from spread")
print("        n     truth    jackknife    scatter    bootstrap    scatter    delete-d    scatter")
for n in (51, 201, 801):
    d = int(round(n ** 0.75))                                  # delete-d with d/n -> 0 slowly
    k = (n - 1) // 2
    x = rng.standard_normal((reps, n))
    xs = np.sort(x, axis=1)
    hi = (xs[:, k] + xs[:, k + 1]) / 2                         # drop any point below the median
    mid = (xs[:, k - 1] + xs[:, k + 1]) / 2                    # drop the median itself
    lo = (xs[:, k - 1] + xs[:, k]) / 2                         # drop any point above it
    bar = (k * hi + mid + k * lo) / n                          # only three distinct loo values
    jack = np.sqrt((n - 1) / n * (k * (hi - bar) ** 2 + (mid - bar) ** 2 + k * (lo - bar) ** 2))
    boot = np.array([np.median(row[rng.integers(n, size=(b, n))], axis=1).std(ddof=1)
                     for row in x])
    keep = np.argsort(rng.random((subsets, n)), axis=1)[:, :n - d]
    deld = np.array([np.sqrt((n - d) / d * np.median(row[keep], axis=1).var()) for row in x])
    truth = xs[:, k].std(ddof=1)
    print(f"  {n:9d} {truth:9.4f} {jack.mean():12.4f} {jack.std(ddof=1) / jack.mean():10.3f}"
          f" {boot.mean():12.4f} {boot.std(ddof=1) / boot.mean():10.3f}"
          f" {deld.mean():11.4f} {deld.std(ddof=1) / deld.mean():10.3f}")
# =>   standard error of the sample median, 1000 samples, 400 resamples, truth from spread
#            n     truth    jackknife    scatter    bootstrap    scatter    delete-d    scatter
#             51    0.1739       0.1707      0.671       0.1811      0.257      0.1753      0.293
#            201    0.0867       0.0907      0.703       0.0899      0.192      0.0900      0.257
#            801    0.0441       0.0447      0.719       0.0445      0.141      0.0447      0.208
```

Read the level columns first and the method looks fine. The true standard error of the median is $0.1739$, $0.0867$ and $0.0441$ at the three sample sizes, and the jackknife averages $0.1707$, $0.0907$ and $0.0447$ — within a few percent everywhere, and no worse than the bootstrap's $0.1811$, $0.0899$ and $0.0445$. A study that computed the jackknife standard error across many datasets and compared the average to the truth would find nothing wrong.

The scatter columns are the failure and they are unambiguous. The scatter is the coefficient of variation of the estimate across the thousand samples — how far a *single* run's answer typically sits from the average of all of them. For the jackknife it reads $0.671$, $0.703$, $0.719$: **it does not shrink as the sample grows sixteenfold, and if anything it grows.** For the bootstrap it reads $0.257$, $0.192$, $0.141$, falling as it should. For the delete-$d$ jackknife with $d=n^{3/4}$ it reads $0.293$, $0.257$, $0.208$, also falling, which is the repair working.

The practical translation is blunt. A jackknife standard error for the median computed on eight hundred observations is, typically, wrong by seventy percent — not seventy percent of the time, but by seventy percent — and gathering more data does not improve it. **An estimator can be right on average and useless in every individual instance, and averaging over datasets is exactly the check that cannot see the difference.**

!!! warning "A statistic whose leave-one-out values are dominated by a few observations has a standard error, and it is not the one you computed"
    The median's failure is visible because it is extreme — three distinct values out of $n$ — but the same mechanism operates in degree wherever a statistic is driven by a handful of points. A maximum drawdown is determined by two dates. A tail quantile is determined by the few observations beyond it. A regression slope on a sample with one high-leverage point is largely determined by that point. In each case the leave-one-out values cluster into a small number of groups and the jackknife variance becomes a statement about those few observations rather than about the sample, so it inherits their noise without inheriting the averaging that would damp it. The diagnostic is free and follows from the construction: **print the number of distinct leave-one-out values, or the share of $\sum_i(\hat\theta_{(\cdot)}-\hat\theta_{(i)})^{2}$ contributed by the largest single term.** If one observation contributes more than a few percent of that sum, the statistic is not smooth at this sample and the jackknife should be replaced by a bootstrap or a delete-$d$ scheme. This is the quantitative version of the lesson's advice to "check stability by deleting the worst day", and the answer it returns is a number rather than an impression.

## Every BCa Interval Has a Jackknife Inside It

[Bootstrap Methods](07-bootstrap-methods.md) found BCa to be the only interval construction with correct coverage on short samples, and deferred what its two corrections are. Both are computable. The bias constant $z_0=\Phi^{-1}\big(\#\{\hat\theta^{\ast}_b<\hat\theta\}/B\big)$ measures how far the bootstrap distribution's median sits from the estimate, and comes from the resamples. The **acceleration** $a$ measures how fast the standard error changes with the parameter, and comes from the jackknife:

$$\hat a=\frac{\sum_i\big(\hat\theta_{(\cdot)}-\hat\theta_{(i)}\big)^{3}}{6\Big[\sum_i\big(\hat\theta_{(\cdot)}-\hat\theta_{(i)}\big)^{2}\Big]^{3/2}},$$

which is one sixth of the standardized skewness of the empirical influence values. The two constants then shift the percentile levels the interval is read at, and the shift is asymmetric — which is the point, because the errors they correct are.

```python
import numpy as np
from scipy.special import ndtri
from scipy.stats import bootstrap, norm
from scipy.stats import t as tdist

rng = np.random.default_rng(9087)
per, nu, sr, boots = 12, 2.6, 0.30, 4_000


def sharpe(y, axis=-1):
    return np.sqrt(per) * y.mean(axis=axis) / y.std(axis=axis, ddof=1)


def sample(n):
    return tdist.rvs(nu, size=n, random_state=rng) / np.sqrt(nu / (nu - 2)) + sr / np.sqrt(per)


def endpoints(x, b, accel=True):
    n, hat = x.size, sharpe(x)
    star = sharpe(x[rng.integers(n, size=(b, n))], axis=1)
    z0 = ndtri(np.clip((star < hat).mean(), 1e-6, 1 - 1e-6))
    s, q = x.sum(), (x ** 2).sum()
    m = (s - x) / (n - 1)
    loo = np.sqrt(per) * m / np.sqrt((q - x ** 2 - (n - 1) * m ** 2) / (n - 2))
    dev = loo.mean() - loo                                     # the jackknife's whole contribution
    a = (dev ** 3).sum() / (6 * (dev ** 2).sum() ** 1.5) if accel else 0.0
    lv = [norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))) for z in (-1.96, 1.96)]
    return np.quantile(star, lv), a, z0, np.quantile(star, [0.025, 0.975])


x = sample(120)
ci, a, z0, pct = endpoints(x, 20_000)
ref = bootstrap((x,), sharpe, n_resamples=20_000, method="BCa",
                random_state=np.random.default_rng(1)).confidence_interval
print(f"  one sample of {x.size} months, Sharpe {sharpe(x):.4f}: z0 = {z0:.4f},"
      f" jackknife acceleration a = {a:.4f}")
print(f"    percentile [{pct[0]:.4f}, {pct[1]:.4f}]   BCa by hand [{ci[0]:.4f}, {ci[1]:.4f}]"
      f"   scipy BCa [{ref.low:.4f}, {ref.high:.4f}]")

n = 36
print(f"  coverage of a nominal 95% interval, {boots} samples of {n} months, true Sharpe {sr}")
print("   interval                     coverage    misses low    misses high    median width")
hits = {"percentile": [0, 0, 0, []], "z0 only, no jackknife": [0, 0, 0, []], "BCa": [0, 0, 0, []]}
for _ in range(boots):
    x = sample(n)
    full, _, _, pct = endpoints(x, 2_000)
    bc, _, _, _ = endpoints(x, 2_000, accel=False)
    for key, (lo, hi) in (("percentile", pct), ("z0 only, no jackknife", bc), ("BCa", full)):
        hits[key][0] += lo <= sr <= hi
        hits[key][1] += hi < sr
        hits[key][2] += lo > sr
        hits[key][3].append(hi - lo)
for key, (cov, below, above, w) in hits.items():
    print(f"  {key:<28} {cov / boots:8.3f} {below / boots:13.3f} {above / boots:14.3f}"
          f" {np.median(w):15.3f}")
# =>   one sample of 120 months, Sharpe 1.0079: z0 = -0.0484, jackknife acceleration a = -0.0265
#        percentile [0.3788, 1.6715]   BCa by hand [0.3095, 1.6109]   scipy BCa [0.3374, 1.6201]
#      coverage of a nominal 95% interval, 4000 samples of 36 months, true Sharpe 0.3
#       interval                     coverage    misses low    misses high    median width
#      percentile                      0.922         0.019          0.059           2.313
#      z0 only, no jackknife           0.942         0.015          0.043           2.309
#      BCa                             0.950         0.017          0.034           2.304
```

The first panel is the construction checked against a library. On one sample of a hundred and twenty months the two constants are small — $z_0=-0.0484$ and $a=-0.0265$ — and they move the interval from the percentile $[0.3788,1.6715]$ to $[0.3095,1.6109]$, dragging both ends down. SciPy's own BCa, running its own twenty thousand resamples from a different seed, returns $[0.3374,1.6201]$. The two agree to within the Monte Carlo scatter of two independent runs of that size, which is what a cross-check of this kind can establish and is enough to say the hand computation is the same procedure the library performs.

The second panel prices each correction separately, and this is the number the section exists for. The plain percentile interval covers $0.922$ against a nominal $0.95$. Adding the bias constant alone — the part that needs no jackknife — lifts it to $0.942$. Adding the acceleration, which is the jackknife's entire contribution, lifts it to $0.950$, exactly nominal. **The jackknife inside a BCa interval is worth the last eight-tenths of a percentage point of coverage, and it is the difference between an interval that is nearly right and one that is right.**

The miss columns show how. The percentile interval's failures are lopsided — it lands entirely below the truth $1.9\%$ of the time and entirely above it $5.9\%$ of the time, a three-to-one asymmetry that a symmetric interval has no way to notice. The two corrections attack that asymmetry rather than the width: $z_0$ takes the high-side misses from $5.9\%$ to $4.3\%$, the acceleration takes them to $3.4\%$, and the median width barely moves across all three rows, from $2.313$ to $2.304$. **The interval is not being widened; it is being moved**, which is why the correction is invisible in any summary that reports interval width alone.

## Leave-One-Out Is a Sensitivity Analysis Before It Is a Standard Error

The jackknife's most valuable output is often not a variance at all. The quantities $\hat\theta_{(\cdot)}-\hat\theta_{(i)}$ are the empirical influence of each observation, and reading them individually answers a question that no aggregate does: *which observations is this result made of?*

The uses compound quickly. A Sharpe ratio whose largest leave-one-out deviation is a tenth of its value is a Sharpe ratio that one day of the sample can move by ten percent, which is worth knowing before the number is presented and is invisible in any interval. A backtest whose profit is concentrated in three days is a common finding and the leave-one-out values state it as a number rather than an anecdote. A regression coefficient with one high-leverage observation shows the same signature, and in that setting the leave-one-out deviations have a closed form and are the standard regression diagnostic under a different name. A cross-sectional strategy can be jackknifed by *asset* rather than by date, which answers the different and equally important question of whether the result survives the removal of any single name.

The generalization to blocks is the same idea with the same caveat as [Bootstrap Methods](07-bootstrap-methods.md): with dependent data, deleting one observation removes almost none of the information, so the block jackknife deletes contiguous runs. And the leave-one-out idea recurs one part later under a different name — deleting an observation, refitting a model, and scoring the prediction on the held-out point is cross-validation, which is [Part XIV](../part-14-model-selection/index.md) and is a bias-variance question rather than a resampling one.

!!! note "The jackknife's determinism is worth more than its accuracy, because it makes a diagnostic auditable"
    Every other method in this part returns a different number on a different seed, and the difference is real: [Bootstrap Methods](07-bootstrap-methods.md) measures a standard deviation of $0.0107$ on a $p$-value near $0.06$ at five hundred resamples, which is large enough to change a conclusion. The jackknife has $n$ resamples, all of them, always, so two people running it on the same data get the same number and a disagreement is a disagreement about the data or the code. That property is why the jackknife survives inside BCa rather than being replaced by a bootstrap estimate of the same acceleration — an interval whose endpoints wobbled under reseeding would be harder to defend than one whose endpoints are slightly less accurate — and it is why leave-one-out remains the right instrument for a sensitivity check even where it is the wrong instrument for a standard error. A diagnostic that must be reproducible is a different requirement from an estimator that must be efficient, and the two are best served by different methods.

## Deterministic, Cheap, and Honest About Less

The jackknife does three things and the ordering of their value is not the ordering in the textbooks. It removes an $O(1/n)$ bias exactly, which is its strongest result and the one nobody uses — a thirty-six-month Sharpe overstates by $4.3\%$ and the correction is one line. It supplies the acceleration constant that makes a BCa interval reach its nominal coverage, which is the use everyone relies on and almost nobody knows they are relying on. And it estimates a variance, which is the use it is named for and the one where it is beaten: correct on average for the median and wrong by seventy percent in any individual sample, at any $n$.

Its limitation and the bootstrap's are the same limitation reached from different sides. The bootstrap resamples with replacement and can approximate any smooth functional's sampling distribution, at the cost of a Monte Carlo error and a licence that fails for non-smooth statistics. The jackknife perturbs the sample by exactly one observation at a time, which is deterministic and cheap and gives it a strictly narrower view — it sees the first derivative of the statistic and nothing else, so a statistic without a first derivative is a statistic it cannot see.

That closes the part, and the shape it closes on is the one every page arrived at independently. A generator whose defect was invisible in one dimension and total in three. A tilted estimator that was wrong by a factor of two while reporting a small standard error and a healthy effective sample size. An antithetic pair that doubled the variance of a butterfly. A Monte Carlo estimate whose reported precision improved fourfold while its actual precision stopped improving at all. A bootstrap $p$-value of $0.06$ that was really $0.06\pm0.02$. A jackknife standard error that is right on average and wrong by seventy percent in every instance. In each case the estimate and its stated precision were computed from the same draws, so the defect that corrupted one corrupted the other, and the diagnostic that would have caught it — running the whole procedure again, or checking a condition analytically before starting — was outside what a single run can see.

That is the price of computing an answer that no formula supplies. It is worth paying, because the alternative is not a better estimate but a missing one, and every result in this part exists because the derivation does not. But the price is real, and it is paid in the one currency simulation cannot mint for itself: **a number produced by a procedure that grades its own homework needs a second opinion from outside the procedure, and the second opinion is always cheaper than the consequence of not having it.**
