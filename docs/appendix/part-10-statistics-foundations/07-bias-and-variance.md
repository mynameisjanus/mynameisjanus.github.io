# Bias and Variance

A model narrows the population to a set and an estimator picks one point in it from one sample, and misses. This page is the anatomy of the miss. It splits into exactly two pieces — a constant offset that is the same in every sample, and a wobble that averages to nothing — and the split earns its keep because the two pieces respond to opposite treatments. More data shrinks one and leaves the other precisely where it was. Averaging across assets shrinks one and leaves the other precisely where it was. And shrinking an estimate toward a constant trades the second for the first at a rate that is always favourable at the margin and never favourable all the way, which is why the best estimator in routine use is almost never the unbiased one.

This page covers the mean-squared-error decomposition of a single estimator's error about a fixed parameter, the Bessel correction derived from the geometry of the residual vector together with the reason no analogous correction exists for a standard deviation, the case for a biased estimator when the bias buys more variance than it costs, the asymmetry that makes bias survive averaging while variance does not, and shrinkage as the deliberate purchase of bias with a factor nobody can compute. It builds no taxonomy of estimator properties, so consistency, efficiency and the information bound are [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md); it decomposes no *prediction* error and says nothing about underfitting or overfitting, which is [Bias–Variance Tradeoff](../part-14-model-selection/01-bias-variance-tradeoff.md); it removes no bias by resampling, which is [Jackknife Methods](../part-09-monte-carlo-methods/08-jackknife-methods.md); it derives no estimator attaining a bound, which is [Part XI](../part-11-parameter-estimation/index.md); it charges nothing for the size of a search, which is [Part XV](../part-15-multiple-testing/index.md); it optimizes no portfolio, which is [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md); and it never trades the estimator it corrects.

The trading stake is a ranking that reverses depending on which column is read. [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md) races three volatility estimators against realized volatility and finds that "Parkinson wins on correlation (0.672) and on mean absolute error (4.98%) while being by far the *most biased* — it runs 2.66 percentage points low", against a close-to-close estimator that is nearly unbiased at $+0.04\%$ and carries a mean absolute error of $5.598\%$. The third section prices that trade exactly, and the fourth reverses it: the biased estimator wins on one asset and loses on a book of three, because averaging divides one of the two error components and leaves the other alone.

## Estimation Error Splits Into a Constant and a Wobble, With No Cross Term

Let $\hat\theta$ estimate a fixed number $\theta$, with the expectation taken over samples.

??? note "Proof that mean squared error is squared bias plus variance, and that a third term appears the moment the target stops being a parameter"
    Add and subtract the estimator's own mean:

    $$\mathbb{E}\big[(\hat\theta-\theta)^{2}\big]=\mathbb{E}\Big[\big((\hat\theta-\mathbb{E}\hat\theta)+(\mathbb{E}\hat\theta-\theta)\big)^{2}\Big].$$

    Expanding gives three terms. The first is $\mathrm{var}(\hat\theta)$; the third is $(\mathbb{E}\hat\theta-\theta)^{2}=\mathrm{bias}(\hat\theta)^{2}$, a constant; and the cross term is

    $$2\,\mathbb{E}\big[\hat\theta-\mathbb{E}\hat\theta\big]\cdot\big(\mathbb{E}\hat\theta-\theta\big)=2\cdot0\cdot\mathrm{bias}=0,$$

    because the first factor has mean zero by construction and the second is not random. Hence $\mathrm{MSE}=\mathrm{bias}^{2}+\mathrm{var}$ exactly, with no interaction and no approximation.

    Now let the target be a future observation $Y=f(x)+\varepsilon$ rather than a parameter, with $\varepsilon$ independent of the sample and of variance $\sigma^{2}_\varepsilon$. The same expansion yields

    $$\mathbb{E}\big[(Y-\hat f(x))^{2}\big]=\sigma^{2}_{\varepsilon}+\mathrm{bias}\big(\hat f(x)\big)^{2}+\mathrm{var}\big(\hat f(x)\big),$$

    and the leading term is untouchable by any estimator whatever.

    The load-bearing step is that $\theta$ is a constant, which is what makes the second factor of the cross term deterministic and kills it. **The two-term decomposition is the whole content of this page and the three-term one is the whole content of [Bias–Variance Tradeoff](../part-14-model-selection/01-bias-variance-tradeoff.md)**, and confusing them is how an irreducible noise floor gets attributed to a model that someone then spends a quarter trying to tune.

The split is not a decomposition of an estimator into two estimators, and neither piece is separately observable from a single sample: one dataset yields one number, and whether its error came from a bias or a wobble is exactly what a single dataset cannot say. That is why every measurement on this page is taken across replications.

## Dividing by $n-1$ Is a Bias Correction, and the Square Root Undoes It

[Mathematical Notation](../part-01-mathematical-foundations/03-mathematical-notation.md) states that dividing by $n-1$ "compensates for having estimated the two means from the same data — the bias correction derived in Bias and Variance". This is that derivation, and it comes with a consequence the correction's fame tends to hide.

??? note "Proof that the sum of squared deviations has expectation $(n-1)\sigma^{2}$, and that no rescaling makes $s$ unbiased for $\sigma$"
    Write the deviations about the sample mean in terms of deviations about the true mean:

    $$\sum_{i}(X_i-\bar X)^{2}=\sum_i(X_i-\mu)^{2}-n(\bar X-\mu)^{2}.$$

    Taking expectations, the first term is $n\sigma^{2}$ and the second is $n\cdot\mathrm{var}(\bar X)=n\cdot\sigma^{2}/n=\sigma^{2}$, so

    $$\mathbb{E}\Big[\sum_i(X_i-\bar X)^{2}\Big]=(n-1)\sigma^{2},$$

    and dividing by $n-1$ rather than $n$ is what makes $s^{2}$ unbiased. Geometrically the residual vector $X-\bar X\mathbf 1$ is the projection of $X-\mu\mathbf 1$ onto the orthogonal complement of $\mathbf 1$, a subspace of dimension $n-1$, and each of those dimensions carries $\sigma^{2}$ — the same rotation that produced the $\chi^{2}_{n-1}$ law in [Sampling Distributions](03-sampling-distributions.md) and the same orthogonality that made the analysis-of-variance identity exact in [Descriptive Statistics](02-descriptive-statistics.md).

    The consequence is that $s^{2}$ is unbiased for $\sigma^{2}$ and $s$ is therefore **biased** for $\sigma$. Since $\sqrt{\cdot}$ is strictly concave, Jensen's inequality gives $\mathbb{E}[s]<\sqrt{\mathbb{E}[s^{2}]}=\sigma$ strictly, for every $n$. Under normality the exact factor is

    $$c_4(n)=\mathbb{E}[s]/\sigma=\sqrt{\frac{2}{n-1}}\cdot\frac{\Gamma(n/2)}{\Gamma\big((n-1)/2\big)}<1.$$

    The load-bearing quantity is the one degree of freedom spent estimating $\bar X$, and it is worth noting what fails without the estimation: if $\mu$ is known, dividing by $n$ is correct and dividing by $n-1$ is the error. **The familiar correction makes the variance unbiased and thereby makes the volatility biased, and volatility is the quantity every risk report actually prints.**

```python
import numpy as np
from scipy.special import gammaln

rng = np.random.default_rng(10071)
reps = 200_000                                                 # unit variance, so 1.0 is the truth

print("        n    E[s2] div n    E[s2] div n-1    c4 = E[s]/sigma    vol bias at 19.5%"
      "    MSE n-1    MSE n+1")
for n in (5, 10, 21, 63, 252):
    x = rng.standard_normal((reps, n))
    ss = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    v0, v1, v2 = ss / n, ss / (n - 1), ss / (n + 1)
    c4 = np.exp(0.5 * np.log(2 / (n - 1)) + gammaln(n / 2) - gammaln((n - 1) / 2))
    print(f"  {n:9d} {v0.mean():14.5f} {v1.mean():16.5f} {c4:18.5f}"
          f" {100 * 0.195 * (c4 - 1):20.3f} {((v1 - 1) ** 2).mean():10.5f}"
          f" {((v2 - 1) ** 2).mean():10.5f}")
# =>         n    E[s2] div n    E[s2] div n-1    c4 = E[s]/sigma    vol bias at 19.5%    MSE n-1    MSE n+1
#              5        0.79966          0.99958            0.93999               -1.170    0.49917    0.33315
#             10        0.89911          0.99902            0.97266               -0.533    0.22176    0.18180
#             21        0.95249          1.00012            0.98758               -0.242    0.09987    0.09078
#             63        0.98380          0.99966            0.99598               -0.078    0.03213    0.03115
#            252        0.99582          0.99979            0.99900               -0.019    0.00798    0.00792
```

The first two columns are the correction doing its job. With divisor $n$ the estimate reads $0.79966$, $0.89911$, $0.95249$, $0.98380$ and $0.99582$ against a truth of $1$ — biased low by exactly $(n-1)/n$, which at $n=5$ is a twenty-percent understatement of variance. With divisor $n-1$ it reads $0.99958$, $0.99902$, $1.00012$, $0.99966$, $0.99979$: unbiased at every sample size, as derived.

The next two columns are what survives. The factor $c_4(n)$ is $0.93999$, $0.97266$, $0.98758$, $0.99598$, $0.99900$, so the standard deviation is still biased low even though the variance is not. Translated into the units a risk report uses, on a $19.5\%$ asset a five-day volatility runs $1.170$ percentage points low and a twenty-one-day volatility runs $0.242$ points low — every window, in the same direction, forever. **The correction everybody knows is unbiased for the quantity almost nobody reports.**

The last two columns are the first hint of this page's real subject. Dividing by $n+1$ produces an estimator that is more biased than either of the others and has lower mean squared error than the unbiased one at every row: $0.33315$ against $0.49917$ at $n=5$, and $0.09078$ against $0.09987$ at $n=21$. Unbiasedness is not a synonym for accuracy, and here it costs a third of the achievable error at small $n$.

## An Unbiased Estimator Is Almost Never the Best One

The volatility table the course prints is the cleanest demonstration of that principle in the whole corpus, because the ranking reverses between columns and both rankings are correct.

```python
import numpy as np

rng = np.random.default_rng(10073)
days, steps, w, ann = 6_000, 78, 21, np.sqrt(252)
phi, vv = 0.985, 0.38 * np.sqrt(1 - 0.985 ** 2)                # a realistic vol-of-vol

lv = np.zeros(days)
for t in range(1, days):
    lv[t] = phi * lv[t - 1] + vv * rng.standard_normal()
sd = 0.165 / ann * np.exp(lv - lv.var() / 2)
gap = 0.0055 * (sd / sd.mean()) * rng.standard_normal(days)    # the range never sees this
inc = sd[:, None] / np.sqrt(steps) * rng.standard_normal((days, steps))
path = gap[:, None] + np.cumsum(inc, axis=1)
r = path[:, -1]
hi, lo = np.maximum(path.max(axis=1), 0.0), np.minimum(path.min(axis=1), 0.0)

lam, e = 0.94, np.zeros(days)
e[w] = r[:w].var()
for t in range(w + 1, days):
    e[t] = lam * e[t - 1] + (1 - lam) * r[t - 1] ** 2
idx = range(w - 1, days - w - 1)
fwd = np.array([r[t + 1:t + 1 + w].std(ddof=1) * ann for t in idx])
cc = np.array([r[t - w + 1:t + 1].std(ddof=1) * ann for t in idx])
pk = np.array([np.sqrt((((hi - lo)[t - w + 1:t + 1]) ** 2).mean() / (4 * np.log(2))) * ann
               for t in idx])
ew = np.array([np.sqrt(e[t]) * ann for t in idx])

print(f"  target = realized vol over the next {w} days, mean {100 * fwd.mean():.2f}%")
print("   estimator             corr      mean      bias       MAE     error var    bias^2 share")
for name, s in (("close-to-close 21d", cc), ("EWMA lambda 0.94", ew), ("Parkinson 21d", pk)):
    b, v = (s - fwd).mean(), (s - fwd).var()
    print(f"  {name:<20} {np.corrcoef(s, fwd)[0, 1]:8.4f} {100 * s.mean():9.2f}%"
          f" {100 * b:+8.2f}% {100 * np.abs(s - fwd).mean():8.3f}% {1e4 * v:11.1f}"
          f" {b ** 2 / (b ** 2 + v):15.3f}")
bc, vc = (cc - fwd).mean(), (cc - fwd).var()
bp, vp = (pk - fwd).mean(), (pk - fwd).var()
print(f"  averaging over K independent assets: close-to-close {1e4 * vc:.1f}/K,"
      f" Parkinson {1e4 * bp ** 2:.1f} + {1e4 * vp:.1f}/K, crossover at K = {(vc - vp) / bp ** 2:.2f}")
# =>   target = realized vol over the next 21 days, mean 19.68%
#       estimator             corr      mean      bias       MAE     error var    bias^2 share
#      close-to-close 21d     0.6356     19.71%    +0.03%    4.736%        39.5           0.000
#      EWMA lambda 0.94       0.6535     19.87%    +0.19%    4.494%        34.8           0.001
#      Parkinson 21d          0.6978     17.78%    -1.90%    4.427%        29.0           0.111
#      averaging over K independent assets: close-to-close 39.5/K, Parkinson 3.6 + 29.0/K, crossover at K = 2.93
```

The three rows reproduce the course's structure line for line. The close-to-close estimator is essentially unbiased at $+0.03\%$ against a published $+0.04\%$. The Parkinson estimator runs $1.90$ percentage points low, because a session's high–low range cannot see the overnight gap, and the course measures the same defect at $2.66$ points. And the biased one wins: correlation $0.6978$ against $0.6356$, mean absolute error $4.427\%$ against $4.736\%$, published as $0.6715$ against $0.6437$ and $4.982\%$ against $5.598\%$.

The decomposition column says why the win is possible. Parkinson's squared bias accounts for $0.111$ of its total error and its error variance is $29.0$ against close-to-close's $39.5$, so it is buying a twenty-seven percent reduction in variance for a bias that costs it eleven percent of its mean squared error. That is a good trade, and it is a trade that an unbiasedness criterion is structurally unable to see, because unbiasedness scores one of the two terms and ignores the other entirely.

**The estimator that wins is the one that is wrong in a stable way rather than the one that is right on average**, and the lesson's own prescription follows immediately from the decomposition — take the range estimator's shape and correct its level, because a bias whose size and sign are stable costs one subtraction to remove while a variance costs data.

## Bias Survives Averaging and Variance Does Not, Which Decides Which One to Fear

The two components are not merely different; they respond differently to the single most common operation in portfolio construction, which is averaging over assets.

Average $K$ independent estimates of $K$ different quantities. The variance component of the mean squared error divides by $K$; the squared bias, being the same constant in every one of them, does not move at all. So the MSE-optimal choice of estimator is not a property of the estimator — it depends on how many of them you intend to average, and the crossover is computable from numbers already in the table above. Close-to-close carries $39.5/K$; Parkinson carries $3.6+29.0/K$; these cross at $K=2.93$.

That is the reversal, and it is not a marginal one. On a single asset Parkinson wins on every accuracy column. On three assets the two are level. On a book of fifty the comparison is $0.79$ against $4.18$, and the estimator that lost every column in the published table wins by a factor of five. **Nothing about this is visible in a study that races estimators one asset at a time, which is how estimator comparisons are almost always conducted.**

The same asymmetry organizes several results elsewhere in the appendix. The four-percent Sharpe overstatement that [Jackknife Methods](../part-09-monte-carlo-methods/08-jackknife-methods.md) measures on a thirty-six-month record does not average away across a book of fifty managers while its noise does, which is why a small systematic bias matters more to an allocator than to a single manager. The survivorship shift of [Population vs Sample](01-population-vs-sample.md) is a bias, so a larger universe makes it more statistically significant rather than less. And in the other direction, the Monte Carlo error of a bootstrap is pure variance, which is why the fix is always more resamples and never a different estimator.

!!! note "The split on this page is one estimator against one parameter, and the identically named split in model selection is a prediction against an outcome, with a third term no estimator can touch"
    Three ownerships are worth stating explicitly, because the names collide and the published [Part IX](../part-09-monte-carlo-methods/index.md) index already routes one of them elsewhere. The mean-squared-error decomposition of a single estimator about a fixed parameter is here. The catalogue of estimator properties — unbiasedness, consistency, efficiency, the information bound and the estimators that attain it — is [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md). The decomposition of expected *prediction* error, and with it everything about underfitting, overfitting and model complexity, is [Bias–Variance Tradeoff](../part-14-model-selection/01-bias-variance-tradeoff.md). The two decompositions are not the same theorem in different notation: the prediction version carries an irreducible $\sigma^{2}_\varepsilon$, so it has an achievable error floor strictly above zero, while the estimation version's floor is zero and is approached as $n$ grows. Practically, that is the difference between "this model cannot be improved further" and "this estimator needs more data", and they call for opposite responses.

## Shrinkage Buys Variance With Bias, and Only If It Is Applied to the Right Input

If unbiasedness is not optimal, the natural move is to introduce bias deliberately. Shrinkage does exactly that, and the theory is unusually clean until it meets the question of what to shrink.

??? note "Proof that shrinking any unbiased estimator toward any constant strictly reduces mean squared error for some shrinkage, so unbiasedness is never optimal"
    Let $\hat\theta$ be unbiased for $\theta$ with variance $v>0$, let $c$ be any constant, and set $\hat\theta_\lambda=(1-\lambda)\hat\theta+\lambda c$. Then $\mathbb{E}[\hat\theta_\lambda]=(1-\lambda)\theta+\lambda c$, so the bias is $\lambda(c-\theta)$ and the variance is $(1-\lambda)^{2}v$, giving

    $$\mathrm{MSE}(\lambda)=(1-\lambda)^{2}v+\lambda^{2}(\theta-c)^{2}.$$

    Differentiating at $\lambda=0$ gives $\mathrm{MSE}'(0)=-2v<0$ whenever the estimator has any variance at all, so a strictly better estimator exists for every unbiased $\hat\theta$ and every target $c$. Setting the derivative to zero gives the optimum

    $$\lambda^{\ast}=\frac{v}{v+(\theta-c)^{2}}\in(0,1),\qquad \mathrm{MSE}(\lambda^{\ast})=\lambda^{\ast}(\theta-c)^{2}<v.$$

    The load-bearing quantity is $\lambda^{\ast}$, and it depends on $\theta$ — the thing being estimated. **The theorem guarantees a better estimator exists and refuses to say which one**, so every shrinkage method in practice is an estimator of $\lambda^{\ast}$ carrying a variance of its own, and it can be wrong in the direction that matters. That is why Ledoit–Wolf reports an intensity — $1.85\%$ at five hundred and four observations on nine assets in the course's measurement, rising to $9.69\%$ at sixty-three — rather than a constant chosen in advance.

```python
import numpy as np

rng = np.random.default_rng(10077)
reps, K, n_a, obs, rho, vol = 4_000, 50, 9, 504, 0.58, 0.18    # nine sectors, two years, corr 0.58

print("   lambda    out-of-sample MSE of the shrunk estimates (x 1e4)")
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    acc = []
    for _ in range(reps):
        a = 0.20 * rng.standard_normal(K)                      # every variant's truth is zero
        b = 0.20 * rng.standard_normal(K)
        acc.append((((1 - lam) * a + lam * a.mean() - b) ** 2).mean())
    print(f"  {lam:8.2f} {1e4 * np.mean(acc):40.1f}")

C = (rho * np.ones((n_a, n_a)) + (1 - rho) * np.eye(n_a)) * vol ** 2
sr = {k: [] for k in ("mvo", "LW-mvo", "minvar", "1/N")}
for _ in range(400):
    tru = 0.06 + 0.010 * rng.standard_normal(n_a)              # sectors barely differ, as they do
    ins = rng.multivariate_normal(tru / 252, C / 252, obs)
    oos = rng.multivariate_normal(tru / 252, C / 252, 252)
    mh, Sh = ins.mean(axis=0) * 252, np.cov(ins.T) * 252
    Slw = 0.9815 * Sh + 0.0185 * np.trace(Sh) / n_a * np.eye(n_a)
    for k, w in (("mvo", np.linalg.solve(Sh, mh)), ("1/N", np.ones(n_a)),
                 ("minvar", np.linalg.solve(Sh, np.ones(n_a))),
                 ("LW-mvo", np.linalg.solve(Slw, mh))):
        w = w / np.abs(w).sum()
        p = oos @ w
        sr[k].append(np.sqrt(252) * p.mean() / p.std(ddof=1))
print(f"  {n_a} sectors, {obs} in-sample observations, 400 out-of-sample years")
print("   book          out-of-sample Sharpe")
for k in ("mvo", "LW-mvo", "minvar", "1/N"):
    print(f"  {k:<14} {np.mean(sr[k]):20.3f}")
# =>    lambda    out-of-sample MSE of the shrunk estimates (x 1e4)
#          0.00                                    800.6
#          0.25                                    625.4
#          0.50                                    505.4
#          0.75                                    432.6
#          1.00                                    409.9
#      9 sectors, 504 in-sample observations, 400 out-of-sample years
#       book          out-of-sample Sharpe
#      mvo                           0.078
#      LW-mvo                        0.080
#      minvar                        0.350
#      1/N                           0.351
```

The first panel is shrinkage working, on a family whose members all have a true value of zero. Out-of-sample mean squared error falls monotonically as $\lambda$ rises — and it keeps falling all the way to $\lambda=1$, which is complete shrinkage to the family average. The course's own sweep has the identical shape, `lambda 0.0: OOS MSE 6.51`, `0.5: 4.40`, `1.0: 3.42`, and its verdict transfers directly: the individual histories contained no usable information beyond the family average, so the optimal amount of shrinkage was all of it.

The second panel is the honest failure, and it is a failure of aim rather than of method. Mean-variance optimization on nine sectors with two years of data returns an out-of-sample Sharpe of $0.078$ against $1/N$'s $0.351$, and applying Ledoit–Wolf shrinkage to the covariance matrix moves it to $0.080$ — a correct repair, competently executed, worth two thousandths of a Sharpe point. Minimum variance, the same optimizer with the expected-return vector deleted, returns $0.350$ and is level with $1/N$, which is the tell. The course reaches the same conclusion on real sectors, with mean-variance at $0.377$ against $1/N$'s $0.450$ and Ledoit–Wolf making it worse at $0.115$, and states the diagnosis in one line: "**Shrinking $\hat\Sigma$ cannot fix mean-variance, because the error mean-variance maximizes lives in $\hat\mu$.**"

That sentence is this page's thesis applied to a portfolio. Shrinkage is a correct instrument for reducing variance, Ledoit–Wolf estimates its intensity correctly, and the intensity it correctly estimates is for the wrong input — the covariance matrix, whose estimation error is modest, rather than the mean vector, whose estimation error [Sampling Distributions](03-sampling-distributions.md) showed cannot be reduced by any amount of sampling. **A correct repair applied to the wrong term is indistinguishable in the output from no repair at all**, and the only way to tell them apart is to ask which input the objective is most sensitive to before choosing where to intervene.

!!! warning "An estimator validated one asset at a time and deployed across a book has been graded on the component that averages away and excused on the component that does not"
    The pattern recurs wherever a method is benchmarked in a setting narrower than the one it ships into: a volatility model tested on one index and run on five hundred names, a factor exposure checked per stock and consumed by a portfolio risk model, a fill model calibrated on one liquid symbol. In each case the validation reports a single accuracy number, usually a mean absolute error or a correlation, and a single number cannot distinguish the term that will shrink from the term that will not. The free diagnostic is to stop reporting one number: **estimate the bias and the error standard deviation separately, then plot $\mathrm{bias}^{2}+\mathrm{var}/K$ against the $K$ you will actually average over, and read the crossover off the chart.** For the volatility table above that crossover is $K=2.93$, so the published ranking is correct for a single asset and inverted for any real book. A second habit costs nothing and prevents most of the damage: whenever a bias is stable in sign and size, subtract it — a known constant error is the cheapest defect in this appendix to fix, and leaving it in place to preserve a claim of unbiasedness is choosing the worse estimator on purpose.

## Two Components, Opposite Cures, and a Part That Kept Answering the Wrong Question

This page established that estimation error is a constant plus a wobble with no interaction; that the $n-1$ everybody types removes the constant for a variance and reinstates it for a volatility, which is the quantity actually reported; that an unbiased estimator is almost never optimal, so the most biased of three volatility estimators wins every accuracy column that races them one asset at a time; and that the optimal amount of shrinkage depends on the unknown, so shrinkage is always an estimate and can be aimed at the wrong input.

Bias and variance are cured by opposite instruments, and each cure is inert against the other. More data and more averaging destroy variance and do nothing whatever to bias. Shrinkage, regularization and any structural restriction destroy variance by installing bias deliberately. What neither can do is tell you, from inside a single sample, which of the two you are looking at — a bias and a wobble leave the same residual on the one dataset you have, which is why every honest measurement in this part was taken across replications that a practitioner does not get.

That closes the part, and the shape it closes on is the one every page arrived at from a different direction. A path whose sample mean is unbiased at every horizon covers the ensemble mean $30.4\%$ of the time after two hundred and fifty years, with coverage falling as the sample grows. A sample excess kurtosis climbs from $3.93$ to $68.38$ instead of converging on the $11.41$ the course reported, while an explained-variance share of $63\%$ sits against a floor of $11.4\%$ that nine independent series produce with nothing in common. A nominal $95\%$ interval for a variance covers $0.4566$ under a realistic tail, beside a $t$ interval built in the same proof that holds $0.9507$ everywhere. Two ARMA coefficients significant at $p=2.4\times10^{-4}$ scatter over four times their own standard errors while the one combination the data determines stays pinned at $-0.083$. A one-percent loss threshold computed from a running sum and sum of squares is exact under normality and breached $1.454\%$ of the time once the tail thickens. A two-component mixture's likelihood is unbounded, so the estimate every implementation returns is a property of its variance floor. And a volatility estimator that is the most biased of three wins on correlation and on mean absolute error and loses the moment three of them are averaged.

In every one of those cases the arithmetic is correct, the estimator does exactly what its derivation promised, and the promise was made about a population the sample was not a fair draw from — a process the path could not explore, a moment the law does not possess, a normality the returns do not obey, a parameter the model does not pin down, a family the summary was not sufficient for, a guarantee that stopped at the class boundary, or a single asset the book does not consist of. The number is a true statement about $\hat F_n$ and it is read as a statement about $F$. Closing that gap is what [Part XI](../part-11-parameter-estimation/index.md) begins, by asking not what a sample says but what an estimator built from it can be guaranteed to do.

**Every number in statistics is an answer about the sample, and the whole discipline is the argument that it is also an answer about the world — an argument made once, before the calculation, and almost never checked again.**
