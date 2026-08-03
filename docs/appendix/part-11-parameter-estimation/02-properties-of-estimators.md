# Properties of Estimators

[Point Estimation](01-point-estimation.md) ended on a partial order and no winner, and the standard repair is to stop asking which estimator is best and start asking which promises it keeps. The promises form a ladder. At the bottom sit unbiasedness and consistency, the two every practitioner can name and the two that constrain finite-sample behaviour least — one is a statement about samples nobody drew and the other about a limit nobody reaches, and an estimator can hold both while being arbitrarily bad at the sample size you have. Above them sit asymptotic normality, which is where standard errors actually come from, and efficiency, the only rung that compares an estimator to what was achievable rather than to zero. Every rung is priced in assumptions, and the price is a model that has to be right.

This page covers unbiasedness as a restriction on the class of competitors rather than a certificate on a member of it, Rao–Blackwell and the uniqueness completeness buys, consistency together with a demonstration that almost nothing follows from it, Fisher information and the Cramér–Rao bound with the regularity condition that makes it a theorem, asymptotic normality as the machinery that manufactures a standard error, and efficiency and breakdown as two numbers computed under opposite assumptions about the model. It splits no error into a squared bias and a variance, which is [Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md); it proves no limit theorem and takes convergence in probability and in distribution as given, which are [The Weak Law of Large Numbers](../part-07-asymptotic-theory/01-weak-law-of-large-numbers.md) and [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md); it defines sufficiency nowhere and proves no factorization theorem, which is [Statistics and Sufficiency](../part-10-statistics-foundations/05-statistics-and-sufficiency.md); it constructs no estimator, which are [Maximum Likelihood Estimation](03-maximum-likelihood-estimation.md) and [Method of Moments](04-method-of-moments.md); it evaluates nothing against a prior, which is [Bayesian Estimation](05-bayesian-estimation.md); it turns no standard error into an interval, which is [Confidence Intervals](07-confidence-intervals.md); it inverts no likelihood ratio, which is [Part XII](../part-12-hypothesis-testing/index.md); it selects no model, which is [Part XIV](../part-14-model-selection/index.md); and it never claims that an estimator carrying every property on the list is the one to run.

The trading stake is the most quoted asymmetry in the course. [Probability and Random Variables](../../part-03-statistics/01-probability-and-random-variables.md) reports the equity premium as `ann mean 0.075 +/- 0.039` on twenty-five years of daily data and draws the conclusion in four words — "**Vol is estimable; the mean barely is**" — explaining that the mean's "standard error shrinks with calendar time, not observation count — sampling the same decade more finely adds almost no information about it." The third section prices that exactly and upgrades it from a measurement to a bound: the Fisher information for a drift is $T/\sigma^{2}$, so $0.039$ is not this estimator's standard error but a floor no unbiased estimator can go under, at any sampling frequency, ever.

## Unbiasedness Is a Constraint on the Class, and Rao–Blackwell Says Where to Look Inside It

An estimator is **unbiased** if $\mathbb{E}_\theta[\hat\theta]=\theta$ for every $\theta$. Read as a certificate this says almost nothing, since [Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md) already showed that the biased estimator is usually the better one and that an $n+1$ divisor beats the unbiased variance on mean squared error at every sample size. Read as a *constraint* it earns its keep: fixing the class of unbiased rules removes the constant estimator that made the uniformly-best question unanswerable on the previous page, and inside that class a best element frequently exists. Unbiasedness is not so much a property worth having as a fence worth drawing, and everything in the next two sections is a statement about what lives inside the fence.

The first thing that lives inside it is a construction. Given any unbiased estimator, however clumsy, conditioning it on a sufficient statistic produces another unbiased estimator that is never worse.

??? note "Proof that conditioning an unbiased estimator on a sufficient statistic never increases its variance, and that completeness is what makes the improved estimator unique"
    Let $\hat\theta$ be unbiased for $\theta$ and let $T$ be sufficient. Define $\tilde\theta=\mathbb{E}[\hat\theta\mid T]$. Sufficiency is what makes this an estimator at all: the conditional law of the data given $T$ does not involve $\theta$, so the conditional expectation is computable without knowing the parameter, which is the property [Statistics and Sufficiency](../part-10-statistics-foundations/05-statistics-and-sufficiency.md) defines sufficiency by.

    Unbiasedness survives by the tower property,

    $$\mathbb{E}[\tilde\theta]=\mathbb{E}\big[\mathbb{E}[\hat\theta\mid T]\big]=\mathbb{E}[\hat\theta]=\theta,$$

    and the variance falls by the [Law of Total Variance](../part-04-expectation-and-moments/08-law-of-total-variance.md),

    $$\mathrm{var}(\hat\theta)=\mathbb{E}\big[\mathrm{var}(\hat\theta\mid T)\big]+\mathrm{var}\big(\mathbb{E}[\hat\theta\mid T]\big)\ \ge\ \mathrm{var}(\tilde\theta),$$

    with equality exactly when $\mathbb{E}[\mathrm{var}(\hat\theta\mid T)]=0$, that is, when $\hat\theta$ was already a function of $T$.

    Uniqueness needs one more ingredient. A sufficient statistic is **complete** if $\mathbb{E}_\theta[g(T)]=0$ for all $\theta$ forces $g\equiv0$. If $T$ is complete and $\tilde\theta_1,\tilde\theta_2$ are two unbiased functions of it, their difference has mean zero at every $\theta$ and is therefore identically zero — so the improved estimator does not depend on which clumsy estimator you started from, and it is the unique minimum-variance unbiased estimator. That is the Lehmann–Scheffé theorem, and completeness is the property [Exponential Families](../part-10-statistics-foundations/06-exponential-families.md) supplies for essentially the only class of models where a fixed-dimension sufficient statistic exists at all.

    The load-bearing quantity is $\mathbb{E}[\mathrm{var}(\hat\theta\mid T)]$, the variation the original estimator carried in directions the sufficient statistic ignores, and the theorem says all of it is waste. **Rao–Blackwell turns any unbiased estimator into the best one and charges a sufficient statistic for it, so the whole minimum-variance theory is a corollary of a reduction whose availability [Exponential Families](../part-10-statistics-foundations/06-exponential-families.md) showed is essentially exclusive to one class of models** — which is why the theory is beautiful for the normal and the Poisson and silent for the models a desk actually fits.

## Consistency Is the Weakest Property Worth Naming and Almost Nothing Follows From It

An estimator is **consistent** if $\hat\theta_n\to\theta$ in probability as $n$ grows. [The Weak Law of Large Numbers](../part-07-asymptotic-theory/01-weak-law-of-large-numbers.md) puts it plainly: consistency "is the weakest property an estimator can have that is worth naming. It says the estimator is not wrong in the limit. It says nothing about bias at any finite $n$, nothing about the spread at the $n$ you have, and nothing about how the error is distributed." This section makes the second clause quantitative, because the gap between "converges eventually" and "behaves at $n=252$" is larger than the phrasing suggests.

Consistency is genuinely cheap. It is preserved by continuous transformation, so any smooth function of consistent estimators is consistent; it follows from the law of large numbers for essentially every plug-in estimator; and it is unaffected by adding any term that vanishes in probability, which means an estimator can be perturbed arbitrarily at every finite $n$ and stay consistent provided the perturbation eventually switches off. That last clause is not a technicality. It is a recipe.

??? note "Proof that consistency constrains nothing at any finite sample size, by exhibiting a consistent and asymptotically normal estimator whose scaled risk diverges"
    Let $X_1,\dots,X_n$ be independent $\mathcal{N}(\theta,1)$ and define **Hodges' estimator**

    $$\hat\theta_H=\bar X\cdot\mathbf 1\big\{|\bar X|>n^{-1/4}\big\},$$

    which reports the sample mean unless the sample mean is small, in which case it reports zero.

    It is consistent. For $\theta\neq0$ the mean $\bar X\to\theta$ while the threshold $n^{-1/4}\to0$, so the indicator is eventually one with probability tending to one and $\hat\theta_H$ agrees with $\bar X$. For $\theta=0$ we have $\bar X=O_p(n^{-1/2})$ against a threshold of $n^{-1/4}$, larger by a factor of $n^{1/4}\to\infty$, so the indicator is eventually zero with probability tending to one and $\hat\theta_H\to0=\theta$. The same argument makes it asymptotically normal at every fixed $\theta\ne0$.

    It is also **superefficient**. At $\theta=0$ it equals the truth with probability tending to one, so $n\,\mathbb{E}[(\hat\theta_H)^{2}]\to0$ while the sample mean has $n\,\mathbb{E}[(\bar X)^{2}]=1$ — it beats the Cramér–Rao bound of the next section at that one point, which is possible only because it is biased at every finite $n$ and the bound constrains unbiased estimators.

    And it is a disaster. Take $\theta_n=n^{-1/4}$, sitting exactly on the threshold. The indicator is then close to a coin flip, so roughly half the samples return $\bar X$ and half return $0$, an error of $\theta_n$. The scaled risk is therefore of order $n\cdot\theta_n^{2}=n\cdot n^{-1/2}=\sqrt n$, so

    $$\sup_\theta\ n\,\mathbb{E}_\theta\big[(\hat\theta_H-\theta)^{2}\big]\longrightarrow\infty,$$

    while the sample mean's is exactly $1$ at every $\theta$ and every $n$.

    The load-bearing word is *pointwise*. Consistency and asymptotic normality are statements at each fixed $\theta$ separately, and an estimator can satisfy both while its risk, viewed as a function on the parameter space, develops a spike that grows without bound and slides toward zero as $n$ increases. **Consistency is a promise about a limit nobody reaches, and it is compatible with any behaviour whatever at the sample size you have** — which is the precise sense in which the weak law is an ancestor of this property rather than a substitute for measuring it.

## Fisher Information Is the Curvature of the Log-Likelihood and It Caps Every Unbiased Estimator at Once

The fence drawn in the first section has a floor. The **score** is the derivative of the log-likelihood in the parameter, $S(\theta)=\partial_\theta\log f(X;\theta)$, and the **Fisher information** is its variance, $I(\theta)=\mathbb{E}[S^{2}]$. It measures how sharply the likelihood distinguishes nearby parameter values — a flat log-likelihood carries no information and a sharply peaked one carries a great deal — which is why the information also equals $-\mathbb{E}[\partial_\theta^{2}\log f]$, the expected curvature.

??? note "Proof that the Cramér–Rao bound is one Cauchy–Schwarz applied to the score, and that its regularity condition fails by a whole order for models with parameter-dependent support"
    The score has mean zero, since differentiating $\int f(x;\theta)\,dx=1$ under the integral gives $\mathbb{E}[S]=0$. Differentiating the unbiasedness constraint $\int\hat\theta(x)f(x;\theta)\,dx=\theta$ the same way gives

    $$\int\hat\theta(x)\,\partial_\theta f(x;\theta)\,dx=\mathbb{E}\big[\hat\theta\,S\big]=1,$$

    and because $\mathbb{E}[S]=0$ this says $\mathrm{cov}(\hat\theta,S)=1$. Cauchy–Schwarz then gives $1=\mathrm{cov}(\hat\theta,S)^{2}\le\mathrm{var}(\hat\theta)\,\mathrm{var}(S)$, hence

    $$\mathrm{var}(\hat\theta)\ \ge\ \frac{1}{I(\theta)},$$

    with equality exactly when $\hat\theta-\theta$ is proportional to the score — which is why the bound is attained precisely by the exponential families of [Exponential Families](../part-10-statistics-foundations/06-exponential-families.md) and by nothing else.

    For the drift of a diffusion the computation is worth doing because the answer surprises people. Observe $N=mT$ increments over $T$ years at $m$ observations per year; each is $\mathcal{N}(\mu/m,\sigma^{2}/m)$, so its score is $(\Delta X-\mu/m)/\sigma^{2}$ with variance $1/(m\sigma^{2})$, and the total is

    $$I(\mu)=N\cdot\frac{1}{m\sigma^{2}}=\frac{mT}{m\sigma^{2}}=\frac{T}{\sigma^{2}}.$$

    The sampling frequency cancels. Information about a drift accrues in calendar time and not in observation count, so $\mathrm{var}(\hat\mu)\ge\sigma^{2}/T$ for every unbiased estimator at every frequency.

    The regularity condition doing the work is the interchange of $\partial_\theta$ and $\int$, and it fails whenever the support depends on the parameter. For $X_i\sim\mathrm{Uniform}(0,\theta)$ the estimator $\frac{n+1}{n}X_{(n)}$ is unbiased with variance $\theta^{2}/\big(n(n+2)\big)$, which is $O(n^{-2})$, against a naive "bound" of order $n^{-1}$ — beaten not by a constant but by a whole order in $n$.

    The load-bearing step is the differentiation under the integral sign. **The bound is a theorem about a derivative passing through an integral, and every estimator in finance that appears to beat it is either biased, like Hodges', or a boundary estimator in disguise, like a maximum or a running high-water mark.**

The frequency cancellation is not a curiosity. It says the entire high-frequency data industry, which multiplies observation counts by factors of thousands, buys exactly nothing for the one parameter every allocation decision depends on.

```python
import numpy as np
from scipy.stats import trim_mean

rng = np.random.default_rng(11021)
years, sig, reps = 25, 0.195, 10_000
crlb = sig / np.sqrt(years)                                    # information T/sigma^2, over calendar

print(f"  drift over {years} years at {100 * sig:.1f}% vol: CRLB = {crlb:.5f} annualized")
print("  obs/yr        N    sd(sample mean)    sd(median)    sd(trim 10%)    eff(mean)"
      "    max |mean - endpoints|")
for m in (12, 52, 252, 1560):
    n = years * m
    x = rng.standard_normal((reps, n)) * (sig / np.sqrt(m))
    a, b = m * x.mean(axis=1), m * np.median(x, axis=1)
    c, d = m * trim_mean(x, 0.1, axis=1), (x.sum(axis=1)) / years
    print(f"  {m:6d} {n:8d} {a.std():18.5f} {b.std():13.5f} {c.std():15.5f}"
          f" {crlb ** 2 / a.var():12.5f} {np.abs(a - d).max():25.2e}")
# =>   drift over 25 years at 19.5% vol: CRLB = 0.03900 annualized
#      obs/yr        N    sd(sample mean)    sd(median)    sd(trim 10%)    eff(mean)    max |mean - endpoints|
#          12      300            0.03922       0.04908         0.04042      0.98905                  2.78e-17
#          52     1300            0.03859       0.04830         0.03973      1.02131                  2.78e-17
#         252     6300            0.03898       0.04899         0.04019      1.00083                  2.78e-17
#        1560    39000            0.03912       0.04870         0.04027      0.99390                  2.78e-17
```

The header line is the course's number derived rather than measured. With $\sigma=19.5\%$ and $T=25$ years the bound is $\sigma/\sqrt T=0.03900$ annualized, and the lesson's published $\pm0.039$ is not a coincidence of its sample but the floor of the problem. The third column confirms the floor is reached: the sample mean's standard deviation reads $0.03922$, $0.03859$, $0.03898$, $0.03912$ across sampling frequencies spanning a factor of $130$, and the efficiency column sits at $1$ throughout. **Going from monthly to ten-minute data multiplies the observation count by one hundred and thirty and moves the precision of the drift estimate by nothing at all.**

Columns four and five are the cost of not knowing that. The median's standard deviation is $0.04908$ against the mean's $0.03922$, a ratio of $1.2513$ against the $\sqrt{\pi/2}=1.2533$ asymptotic theory predicts, so under a normal model the median throws away thirty-six percent of the available information. The ten-percent trimmed mean loses about four percent. Both are consistent, both are asymptotically normal, and neither attains the bound.

The last column is the section's real finding and it reads $2.78\times10^{-17}$ at every row, which is floating-point zero. The estimator $(X_T-X_0)/T$, which reads the first and last prices and discards everything in between, is *identically equal* to the sample mean of all $39{,}000$ increments — not approximately, not asymptotically, but as an algebraic identity, because the increments telescope. **The efficient estimator of a drift consumes two numbers and the other $38{,}998$ contribute nothing**, which is what [Statistics and Sufficiency](../part-10-statistics-foundations/05-statistics-and-sufficiency.md) means by a reduction being free — and it is also why every attempt to sharpen a drift estimate by sampling more finely is a category error rather than an engineering problem.

## Asymptotic Normality Is What Manufactures a Standard Error, and Superefficiency Is What It Costs to Beat the Bound

Consistency says the estimator lands on the truth and the bound says how tightly it can be packed around it, and neither says what the packing looks like. **Asymptotic normality** supplies the shape: $\sqrt n(\hat\theta_n-\theta)\Rightarrow\mathcal{N}(0,v)$ for some asymptotic variance $v$. That is the property turning a point estimate into a reported number with an error bar, because it licenses $\hat\theta\pm z\sqrt{\hat v/n}$ and nothing else does. An estimator is **asymptotically efficient** when $v=1/I(\theta)$, meeting the bound in the limit.

The natural question is whether an estimator can do better, and the second section already answered it: Hodges' beats the bound at a point. The reason this is not a route to free precision is that the improvement has to be paid for elsewhere on the parameter space, and the payment is not proportional.

```python
import numpy as np

rng = np.random.default_rng(11023)
reps = 400_000

print("        n    n^(-1/4)    theta    n*MSE(sample mean)    n*MSE(Hodges)    ratio")
for n in (100, 1_000, 10_000, 100_000):
    t = n ** -0.25
    for th in (0.0, 0.5 * t, t, 2 * t):
        m = rng.normal(th, 1 / np.sqrt(n), reps)               # exact law of the sample mean
        h = np.where(np.abs(m) > t, m, 0.0)
        a, b = n * ((m - th) ** 2).mean(), n * ((h - th) ** 2).mean()
        print(f"  {n:9d} {t:11.5f} {th:8.5f} {a:21.4f} {b:16.4f} {b / a:8.2f}")
# =>         n    n^(-1/4)    theta    n*MSE(sample mean)    n*MSE(Hodges)    ratio
#            100     0.31623  0.00000                1.0003           0.0189     0.02
#            100     0.31623  0.15811                1.0001           2.5953     2.60
#            100     0.31623  0.31623                0.9965           5.5005     5.52
#            100     0.31623  0.63246                1.0009           1.0244     1.02
#           1000     0.17783  0.00000                1.0008           0.0000     0.00
#           1000     0.17783  0.08891                0.9989           7.9103     7.92
#           1000     0.17783  0.17783                0.9965          16.3061    16.36
#           1000     0.17783  0.35566                0.9974           0.9974     1.00
#          10000     0.10000  0.00000                1.0014           0.0000     0.00
#          10000     0.10000  0.05000                1.0009          25.0000    24.98
#          10000     0.10000  0.10000                1.0035          50.5842    50.41
#          10000     0.10000  0.20000                0.9965           0.9965     1.00
#         100000     0.05623  0.00000                1.0063           0.0000     0.00
#         100000     0.05623  0.02812                1.0012          79.0569    78.97
#         100000     0.05623  0.05623                1.0017         158.9119   158.65
#         100000     0.05623  0.11247                1.0013           1.0013     1.00
```

The sample mean's column is the control and it is flat at $1.0003$, $1.0008$, $1.0014$, $1.0063$ and everywhere else — the scaled risk of an efficient estimator is the constant the bound predicts, at every parameter value and every sample size. Read every other column against that constant.

At $\theta=0$ Hodges' estimator scores $0.0189$ at $n=100$ and $0.0000$ thereafter. It is exactly right, with probability tending to one, at the single point where the sample mean pays full price. That is superefficiency, and a practitioner testing a rule only at the null would see a strict improvement over an estimator that provably cannot be improved.

At $\theta$ equal to the threshold the same rule scores $5.5005$, $16.3061$, $50.5842$ and $158.9119$ — ratios of $5.52$, $16.36$, $50.41$ and $158.65$ against the sample mean. **The penalty grows without bound in $n$, so the estimator gets worse as data accumulates**, which is behaviour neither consistency nor asymptotic normality forbids. And at twice the threshold the ratio is $1.00$ at every $n$: the bad region is a shrinking window no fixed test point will ever sit inside, so a simulation study run at any handful of parameter values reports that the estimator is fine.

!!! note "An unbiased estimator averages to the truth over samples that were never drawn, and an unbiased forecast averages to the outcome over days that actually happened, and only the second can be checked from a single history"
    The two uses of the word differ in what the expectation runs over, and the difference decides whether the property is verifiable at all. Estimator unbiasedness is $\mathbb{E}_\theta[\hat\theta]=\theta$ with the average taken across replications of the sampling — hypothetical repeats of the entire history — at a fixed unknown $\theta$; a single dataset gives one draw from that distribution and can never test it, which is why every measurement on this page is taken across replications a practitioner does not get. Forecast unbiasedness is $\mathbb{E}[y_t-\hat y_t]=0$ with the average taken across time within one realized history, and a single history supplies thousands of draws, so it is testable by regressing the outcome on the forecast and checking for an intercept of zero and a slope of one. The collision matters because the same word licenses two different repairs: a biased estimator is fixed by changing the formula, and a biased forecast is fixed by recalibrating on data you already have. **A backtest can prove a forecast biased and can never prove an estimator biased**, and confusing the two is how a stable calibration error gets attributed to bad luck.

## Efficiency Is Computed Under a Model and Breakdown Is What Is Left When the Model Goes

Every number in the previous two sections was computed under an assumed density. Efficiency is a ratio of variances at a specified law, the information bound is an integral against a specified likelihood, and both move when the law does. **Relative efficiency** of $\hat\theta_1$ to $\hat\theta_2$ is the variance ratio $\mathrm{var}(\hat\theta_2)/\mathrm{var}(\hat\theta_1)$, and it is a function of the model in exactly the way the word "efficiency" hides.

Robustness measures the other side. The **breakdown point** is the smallest fraction of observations that, replaced by arbitrary values, can move the estimate arbitrarily far. For the sample mean it is $1/n$ — a single row can take the estimate anywhere — while the median's is $1/2$ and a ten-percent trimmed mean's is $0.1$. Efficiency and breakdown are computed under opposite assumptions, the first that the model is exactly right and the second that an adversary controls part of the sample, and an estimator choice is a position on where between those two the data actually sits.

```python
import numpy as np
from scipy.stats import t as tdist

rng = np.random.default_rng(11027)
n, reps, k = 252, 40_000, 1.345

def huber(x):
    m = np.median(x, axis=1, keepdims=True)
    s = 1.4826 * np.median(np.abs(x - m), axis=1, keepdims=True)
    for _ in range(25):
        w = np.minimum(1.0, k * s / np.maximum(np.abs(x - m), 1e-12))
        m = (w * x).sum(axis=1, keepdims=True) / w.sum(axis=1, keepdims=True)
    return m[:, 0]

def draw(law):
    if law == "normal":
        return rng.standard_normal((reps, n))
    if law.startswith("t("):
        v = float(law[2:-1])
        return tdist.rvs(v, size=(reps, n), random_state=rng) / np.sqrt(v / (v - 2))
    z = rng.standard_normal((reps, n))
    return np.where(rng.random((reps, n)) < 0.01, 10 * z, z) / np.sqrt(1.99)

print("  law                   sd(mean)    sd(median)    sd(Huber)    eff(mean)"
      "    one-outlier shift: mean    Huber")
for law in ("normal", "t(5)", "t(3.4)", "1% at 10 sigma"):
    x = draw(law)
    e = np.array([x.mean(axis=1), np.median(x, axis=1), huber(x)])
    v = np.sqrt(n) * e.std(axis=1)
    y = x.copy()
    y[:, 0] = 10.0
    sm = np.sqrt(n) * np.abs(y.mean(axis=1) - e[0]).mean()
    sh = np.sqrt(n) * np.abs(huber(y) - e[2]).mean()
    print(f"  {law:<18} {v[0]:11.5f} {v[1]:13.5f} {v[2]:12.5f} {(v.min() / v[0]) ** 2:12.5f}"
          f" {sm:27.5f} {sh:8.5f}")
# =>   law                   sd(mean)    sd(median)    sd(Huber)    eff(mean)    one-outlier shift: mean    Huber
#      normal                 1.00112       1.24997      1.02546      1.00000                     0.62960  0.10319
#      t(5)                   1.00436       1.02568      0.90357      0.80937                     0.63006  0.08966
#      t(3.4)                 0.99424       0.86532      0.78716      0.62682                     0.63040  0.07847
#      1% at 10 sigma         1.00459       0.89708      0.74171      0.54511                     0.63045  0.07457
```

All four laws are standardized to the same variance, so the sample mean's dispersion is identical in every row by construction — $1.00112$, $1.00436$, $0.99424$, $1.00459$ in units of $\sigma/\sqrt n$ — and every difference between rows is a difference in what a *better* estimator could have achieved. Under normality it could not: the mean's efficiency is $1.00000$ and the median's $1.24997$ reproduces the textbook $\sqrt{\pi/2}=1.2533$, so the median discards a third of the precision for a robustness nobody needed.

The efficiency column then falls to $0.80937$, $0.62682$ and $0.54511$. At the tail thickness the course fits for daily equity returns the sample mean throws away thirty-seven percent of the achievable precision, and under one percent contamination at ten sigma it throws away forty-five. **The mean is optimal under exactly one model and is losing a third to a half of the information under every model that describes the data**, which is the sense in which efficiency is a property of the assumption rather than of the estimator.

The last two columns are the breakdown point made concrete. Replacing a single observation with a value of $10$ moves the sample mean by $0.62960$, $0.63006$, $0.63040$, $0.63045$ — *the same number in every law*, because the displacement is $(10-x_1)/n$ and has nothing whatever to do with the tails. Huber's estimator moves by $0.10319$ down to $0.07457$, six to eight times less, and its sensitivity *falls* as the law gets heavier because the scale it calibrates against grows. **One row moves the sample mean by an amount independent of everything the modelling argument was about**, which is why a robustness claim can never be read off an efficiency table.

!!! warning "A property proved under a model is a property of the model, and the estimator that is efficient under the normal is the one a single fat-tailed week can move further than the entire remaining sample"
    The failure is not that anyone believes returns are Gaussian; it is that the estimator was selected under an assumption made years earlier by somebody else and never re-examined when the data changed. Every mean, every ordinary least squares fit, every correlation and every covariance matrix in a production stack is the efficient choice under normality and is between $0.55$ and $0.63$ efficient under the tails those same stacks measure. The free diagnostic costs one line and is not the same as deleting outliers: **recompute the estimate with the five largest absolute observations removed, divide the change by the estimator's own standard error, and treat anything above one as a statement that the number is a function of five rows rather than of the sample.** A ratio of three on a covariance entry means the correlation being fed to an optimizer is a description of one week. The leave-one-out machinery that turns this into a systematic scan rather than a spot check is [Jackknife Methods](../part-09-monte-carlo-methods/08-jackknife-methods.md), and the point of expressing it as a ratio is that the threshold then does not depend on the units, the asset or the sample size.

## The Catalogue Is a Ladder and Every Rung Is Priced in Assumptions

This page established that unbiasedness is worth having as a fence rather than as a certificate, and that inside the fence Rao–Blackwell and Lehmann–Scheffé produce a unique best element whenever a complete sufficient statistic exists; that consistency permits any finite-sample behaviour whatever, with Hodges' estimator beating the bound at a point while its scaled risk climbs to $158.65$ times the sample mean's and keeps climbing; that the Fisher information for a drift is $T/\sigma^{2}$, so the course's $\pm0.039$ is a floor rather than a measurement and a hundred-and-thirty-fold increase in sampling frequency buys nothing; that the efficient estimator of that drift is algebraically identical to one reading two prices; and that an efficiency of $1.000$ under the normal becomes $0.627$ under the tails the course fits, while a single row moves the sample mean by the same amount under every law.

The rungs are ordered by price and the price is measured in assumptions. Consistency needs almost nothing and delivers almost nothing. Asymptotic normality needs a moment condition and a limit theorem and delivers the standard error every report prints. Efficiency needs the density in full and delivers a comparison against what was achievable — the only rung that can say an estimator is *finished* rather than merely convergent. Unbiasedness sits outside the ordering entirely: it costs a restriction on the class and is worth paying only because it makes the other three answerable.

What the page does not deliver is an estimator. Every result here is conditional on a likelihood that was assumed rather than constructed, and the bound in particular is an integral against a density nobody has yet written down for a real return series. Turning that density into a rule is the next move and it is a single one — write the likelihood, take logs, differentiate, set to zero — which is [Maximum Likelihood Estimation](03-maximum-likelihood-estimation.md), the estimator that attains this page's bound in the limit and inherits every one of its conditions in the process.

**A property is a promise about behaviour under a description of the world, and the estimator that keeps the most promises is not the best one; it is the one whose description was checked the least.**
