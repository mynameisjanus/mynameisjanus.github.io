# Method of Moments

The oldest estimator in statistics is also the least demanding. Write down what the model says the first few moments should be, write down what the sample says they are, set the two lists equal, and solve. There is no density to specify, usually no optimizer to run, and frequently a closed form where [Maximum Likelihood Estimation](03-maximum-likelihood-estimation.md) needs a numerical search. The price is paid in a single place and it is easy to miss: the method assumes the moments it matches exist. In finance that assumption fails precisely for the models that were introduced because the data has heavy tails — so the estimator's cheapness and its failure have the same cause, and the failure does not announce itself, because a sample moment is always a finite number no matter what the population is doing.

This page covers moment matching as a system of equations rather than an optimization, consistency and asymptotic normality obtained from the delta method with no density anywhere, the existence cost of matching the $k$th moment, an inconsistent estimator whose standard error shrinks while its bias does not, and overidentification as the one place in this part where an estimator diagnoses its own model. It maximizes no likelihood and computes no score, which is [Maximum Likelihood Estimation](03-maximum-likelihood-estimation.md); it derives the moments of no named law, which are [Part IV](../part-04-expectation-and-moments/index.md) and [Part V](../part-05-common-distributions/index.md); it proves no law of large numbers and no delta method, which are [The Weak Law of Large Numbers](../part-07-asymptotic-theory/01-weak-law-of-large-numbers.md) and [The Delta Method](../part-07-asymptotic-theory/04-delta-method.md); it establishes no efficiency bound, which is [Properties of Estimators](02-properties-of-estimators.md); it turns its overidentification statistic into no formal test, which is [Part XII](../part-12-hypothesis-testing/index.md); it fits no regression by instruments, which is [Part XIII](../part-13-regression/index.md); it estimates no tail index by an extreme-value method, which is [Extreme Value Theory](../part-18-quant-finance-applications/14-extreme-value-theory.md); and it never argues that a moment which does not exist can be estimated by averaging.

The trading stake is a mean-reversion horizon that a position sizer reads directly. [Time Series Analysis](../../part-03-statistics/03-time-series.md) fits the SPY–IVV spread and prints `spread std 23 bp, AR(1) rho 0.82, half-life 3.4 days`, calling it "as tradeable a spread as exists anywhere" and a "violently mean-reverting component with a days-scale half-life — the ETF creation-redemption arbitrage at work". That $0.82$ is a moment estimator, a sample autocorrelation matched to the model's. The second section prices it on the rolling windows a live system actually uses, and the fourth section shows what the same style of estimator does to the course's other headline number, the fitted tail index of $2.65$.

## Matching $k$ Sample Moments to $k$ Model Moments Is a System of Equations, Not an Optimization

Let the model have $p$ parameters $\theta$ and write $m_k(\theta)=\mathbb{E}_\theta[X^{k}]$ for its theoretical moments. The **method of moments** estimator solves

$$m_k(\hat\theta)=\frac1n\sum_{i=1}^{n}X_i^{k},\qquad k=1,\dots,p,$$

$p$ equations in $p$ unknowns. There is no objective function, no maximum, and no first-order condition — nothing is being optimized, which is why the estimator often has a closed form and why it needs no starting value, no convergence criterion and no restarts. For the normal it returns the sample mean and the sample variance with divisor $n$, agreeing with maximum likelihood exactly. For the gamma, the beta and the two-parameter Weibull it returns simple algebraic expressions where the likelihood requires a numerical solve.

Two structural facts follow from the definition and both matter. First, the estimator depends on the data only through $p$ sample averages, so it inherits their behaviour and nothing else — it cannot see the shape of the sample beyond the moments it consumed, which is exactly the reduction that [Statistics and Sufficiency](../part-10-statistics-foundations/05-statistics-and-sufficiency.md) warns is free only when the summary is sufficient, and moment vectors almost never are. Second, the map $\theta\mapsto m(\theta)$ has to be invertible on the parameter space for the system to have a unique solution, and the inverse can leave the parameter space entirely — a moment estimator of a variance can come out negative, and a moment estimator of a degrees-of-freedom parameter can come out below the value at which the moment being matched exists at all. The estimator does not check.

## Consistency and Asymptotic Normality Follow From the Delta Method Alone, With No Density Anywhere

Everything the estimator promises comes from two theorems about averages and one about smooth functions, and none of them mentions a likelihood.

??? note "Proof that a moment estimator is consistent and asymptotically normal whenever the moment map is invertible and differentiable, with no density anywhere in the argument"
    Write $\hat m$ for the vector of the first $p$ sample moments and $m(\theta)$ for the model's. The law of large numbers gives $\hat m\to m(\theta_0)$ almost surely, and if $m^{-1}$ is continuous at $m(\theta_0)$ the continuous mapping theorem gives

    $$\hat\theta=m^{-1}(\hat m)\ \longrightarrow\ m^{-1}\big(m(\theta_0)\big)=\theta_0,$$

    so the estimator is consistent. The multivariate central limit theorem gives $\sqrt n(\hat m-m(\theta_0))\Rightarrow\mathcal{N}(0,V)$ with $V$ the covariance matrix of the vector $(X,X^{2},\dots,X^{p})$, and [The Delta Method](../part-07-asymptotic-theory/04-delta-method.md) applied to $m^{-1}$ gives

    $$\sqrt n(\hat\theta-\theta_0)\ \Rightarrow\ \mathcal{N}\big(0,\ G^{-1}VG^{-\top}\big),\qquad G=\frac{\partial m}{\partial\theta}\Big|_{\theta_0}.$$

    Read the two ingredients. $G$ is a derivative of the model's moment map, and it is near-singular exactly when two parameters move the matched moments in nearly the same way, which is where a moment estimator becomes unstable while remaining formally consistent. And $V$ is the covariance of the matched moments themselves: its $(k,k)$ entry is $\mathbb{E}[X^{2k}]-\mathbb{E}[X^{k}]^{2}$.

    The load-bearing quantity is $V$. Matching the $k$th moment puts the $2k$th moment inside the asymptotic variance, so a variance estimator needs four moments to have a standard error at all, a skewness estimator needs six, and a kurtosis estimator needs eight. **Matching the $k$th moment charges the existence of the $2k$th, and the model that motivated the fit is usually one where the higher of the two does not exist** — at which point the central limit theorem in this proof does not apply, the estimator has no limiting normal law, and every standard error computed from this formula is a number with no referent.

The consistency is real and the finite-sample bias is not covered by it. A moment estimator can be biased at order $1/n$, and when the quantity a desk reads is a nonlinear function of the estimate — a half-life is a logarithm of a correlation — the bias survives and grows.

```python
import numpy as np

rng = np.random.default_rng(11043)
rho, reps = 0.82, 20_000                                       # the course's pair-spread AR(1)
true_hl = np.log(0.5) / np.log(rho)

print(f"  true rho = {rho}, true half-life = {true_hl:.2f} days")
print("       n    Yule-Walker rho    OLS rho    Kendall -(1+3rho)/n    med HL(YW)    med HL(OLS)"
      "    P(HL under true)")
for n in (30, 60, 250, 1000, 5000):
    e = np.sqrt(1 - rho ** 2) * rng.standard_normal((reps, n))
    x = np.empty((reps, n))
    x[:, 0] = rng.standard_normal(reps)
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    c = x - x.mean(axis=1, keepdims=True)
    yw = (c[:, 1:] * c[:, :-1]).sum(axis=1) / (c ** 2).sum(axis=1)
    ols = (c[:, 1:] * c[:, :-1]).sum(axis=1) / (c[:, :-1] ** 2).sum(axis=1)
    hl = np.log(0.5) / np.log(np.clip(yw, 1e-3, 0.999))
    print(f"  {n:6d} {yw.mean():18.4f} {ols.mean():10.4f} {-(1 + 3 * rho) / n:22.4f}"
          f" {np.median(hl):13.2f} {np.median(np.log(0.5) / np.log(np.clip(ols, 1e-3, 0.999))):14.2f}"
          f" {(hl < true_hl).mean():19.3f}")
# =>   true rho = 0.82, true half-life = 3.49 days
#           n    Yule-Walker rho    OLS rho    Kendall -(1+3rho)/n    med HL(YW)    med HL(OLS)    P(HL under true)
#          30             0.6617     0.6927                -0.1153          1.82           2.07               0.891
#          60             0.7436     0.7585                -0.0577          2.49           2.67               0.793
#         250             0.8022     0.8056                -0.0138          3.21           3.27               0.651
#        1000             0.8157     0.8165                -0.0035          3.42           3.44               0.577
#        5000             0.8192     0.8194                -0.0007          3.48           3.48               0.525
```

The autocorrelation column is biased downward at every sample size and the bias is a known function of $n$ rather than a property of the data: $0.6617$, $0.7436$, $0.8022$, $0.8157$, $0.8192$ against a truth of $0.82$. Kendall's approximation $-(1+3\rho)/n$ predicts $-0.0577$ at $n=60$ against an observed $-0.0764$ and $-0.0138$ at $n=250$ against $-0.0178$, tracking the least-squares column almost exactly and understating the Yule–Walker version, which carries extra downward bias because its denominator sums $n$ terms while its numerator sums $n-1$. **The course's full-sample $0.82$ is essentially unbiased because its $n$ is in the thousands; the same estimator on a rolling window is not.**

The half-life columns are where the bias is paid, and they are paid at a leverage the correlation column does not show. A logarithm near $1$ is steep, so a $9\%$ understatement of $\rho$ at $n=60$ becomes a $29\%$ understatement of the horizon: $2.49$ days against a truth of $3.49$. At a sixty-day estimation window — a completely ordinary choice for a live pairs system — the trade is sized and timed against a mean-reversion horizon a full day shorter than the real one, in the direction that makes the strategy look faster and cheaper than it is.

The last column says this is not a tail event but the normal case. The estimated half-life is below the truth $89.1\%$ of the time at $n=30$, $79.3\%$ at $n=60$, and still $52.5\%$ at $n=5000$, so a desk running many pairs sees a systematically compressed distribution of horizons rather than a noisy one. **The correction costs one subtraction and the diagnosis costs one simulation, and neither is run because the estimator has no step at which anything is checked.**

## Matching the $k$th Moment Charges the Existence of the $2k$th

The proof made the accounting explicit and the accounting has an uncomfortable consequence for the models finance actually fits. The Student-$t$ with $\nu$ degrees of freedom has moments only up to order $\nu$: the variance exists for $\nu>2$, the kurtosis for $\nu>4$, and the eighth moment — the one a kurtosis-matched estimator's standard error requires — only for $\nu>8$. [Returns and Their Distributions](../../part-03-statistics/02-returns-and-distributions.md) fits $\nu=2.65$ to daily SPY.

At that value the population kurtosis does not exist. Neither does the sixth moment, the eighth, or anything above the second. A method that estimates $\nu$ by matching the kurtosis is therefore matching a sample quantity to a population quantity that is not a number, and the sample quantity is nevertheless perfectly well defined on every dataset ever collected — finite, computable, and increasing with $n$. [Descriptive Statistics](../part-10-statistics-foundations/02-descriptive-statistics.md) measured the increase directly, watching a sample excess kurtosis climb from $3.93$ to $68.38$ rather than converge on the $11.41$ the course reported.

!!! note "Method of moments matches unconditional moments of the data, generalized method of moments matches conditional moment restrictions implied by a model, and moment matching in a Monte Carlo is a variance-reduction device that estimates nothing at all"
    The three share a name and a piece of arithmetic and differ in what the moment condition is a statement about. Classical method of moments sets $\mathbb{E}_\theta[X^{k}]$ equal to a sample average of the same power, so the conditions are chosen by convenience and the count equals the parameter count. Generalized method of moments sets $\mathbb{E}[g(X_t,\theta)\mid\mathcal{F}_{t-1}]=0$ for conditions the model implies — an Euler equation, an orthogonality between a residual and an instrument — so the conditions carry economic content, may be conditional rather than unconditional, and are usually more numerous than the parameters, which is the subject of the fifth section. Moment matching in a simulation is a third thing entirely: rescaling a set of simulated draws so their sample mean and variance equal the target's exactly, which removes Monte Carlo error in those two summaries and estimates no parameter whatever. That last one belongs to [Variance Reduction](../part-09-monte-carlo-methods/06-variance-reduction.md), and calling it a moment estimator in a code comment is how a variance-reduction trick ends up in a model-fitting pipeline where it silently forces a fit through summaries it should have been testing.

## An Inconsistent Estimator With a Shrinking Standard Error Passes Every Diagnostic a Practitioner Runs

For the Student-$t$ the moment recipe is a single inversion. The excess kurtosis of a $t(\nu)$ is $\kappa=6/(\nu-4)$ for $\nu>4$, so matching it gives

$$\hat\nu=4+\frac{6}{\hat\kappa}.$$

The formula is derived under $\nu>4$ and evaluated wherever the sample kurtosis happens to land, which is the entire problem.

??? note "Proof that the moment estimator of a Student $t$'s degrees of freedom is inconsistent for every $\nu\le4$, and that it converges anyway"
    Fix $\nu\le4$, so the population fourth moment is infinite. The sample excess kurtosis $\hat\kappa$ is a ratio of sample moments and it diverges: $\hat\kappa\to\infty$ in probability as $n$ grows, because the numerator is an average of terms with infinite mean and the denominator converges to a positive constant whenever $\nu>2$.

    Substituting into the inversion,

    $$\hat\nu=4+\frac{6}{\hat\kappa}\ \longrightarrow\ 4+0=4,$$

    in probability, whatever the true $\nu$ is. The estimator has a limit, the limit exists, and the limit is $4$ — the boundary of the region where the formula it came from was valid — regardless of whether the truth is $3.9$ or $2.0$.

    Worse, the convergence looks healthy from inside. Because $\hat\kappa$ diverges slowly, the dispersion of $6/\hat\kappa$ shrinks as $n$ grows, so the estimator's standard error falls at something close to the usual rate. Every finite-sample diagnostic a practitioner runs — repeat the fit on subsamples, bootstrap it, watch it stabilize as data accumulates — reports convergence, and the thing it is converging to is wrong.

    The load-bearing step is $\hat\kappa\to\infty$, which is not a failure to converge but a convergence to a point where the inverse map is degenerate. **An inconsistent estimator with a shrinking standard error is far more dangerous than a noisy one, because every check a practitioner knows how to run reports that it is working**, and the only diagnostic that would catch it is a comparison against an estimator built on different information.

```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kurtosis, t as tdist

rng = np.random.default_rng(11041)

def fit_t(x):
    def neg(lv):
        v, m, s2 = np.exp(lv), x.mean(), x.var()
        for _ in range(60):
            w = (v + 1) / (v + (x - m) ** 2 / s2)
            m = (w * x).sum() / w.sum()
            s2 = (w * (x - m) ** 2).mean()
        return -tdist.logpdf(x, v, m, np.sqrt(s2)).sum()
    return np.exp(minimize_scalar(neg, bounds=(np.log(1.05), np.log(60.0)),
                                  method="bounded").x)

print("  true nu       n    MoM median    MoM sd    MLE median    MLE sd"
      "    (MoM med - nu)/MoM sd    P(MoM > 10)")
for nu0 in (2.65, 6.0):
    for n, reps in ((252, 400), (1260, 400), (6410, 200)):
        a, b = np.empty(reps), np.empty(reps)
        for i in range(reps):
            x = tdist.rvs(nu0, size=n, random_state=rng)
            k = kurtosis(x, fisher=True)
            a[i] = 4 + 6 / k if k > 0 else np.inf                # invert kappa = 6/(nu-4)
            b[i] = fit_t(x)
        am = np.median(a)
        print(f"  {nu0:7.2f} {n:7d} {am:13.3f} {np.std(a[np.isfinite(a)]):9.3f}"
              f" {np.median(b):13.3f} {b.std():9.3f} {(am - nu0) / np.std(a[np.isfinite(a)]):23.2f}"
              f" {(a > 10).mean():14.3f}")
# =>   true nu       n    MoM median    MoM sd    MLE median    MLE sd    (MoM med - nu)/MoM sd    P(MoM > 10)
#         2.65     252         4.735     0.866         2.719     0.688                    2.41          0.005
#         2.65    1260         4.286     0.257         2.656     0.221                    6.37          0.000
#         2.65    6410         4.127     0.112         2.654     0.107                   13.19          0.000
#         6.00     252         8.102    32.862         6.277     9.487                    0.06          0.338
#         6.00    1260         6.757     1.396         5.996     1.079                    0.54          0.033
#         6.00    6410         6.423     0.750         6.010     0.431                    0.56          0.000
```

The first three rows are the theorem happening. At the course's fitted tail the moment estimator returns $4.735$, $4.286$, $4.127$ as the sample grows — descending toward $4$ exactly as the proof says, and never approaching $2.65$. The maximum likelihood estimator on the identical samples returns $2.719$, $2.656$, $2.654$. Both estimators are consuming the same data; one is reading a moment that does not exist and the other is reading a density that does.

The standard-error columns are the trap. The moment estimator's dispersion falls from $0.866$ to $0.257$ to $0.112$, a clean $1/\sqrt n$ improvement that would satisfy any convergence check, while its distance from the truth does not move at all. The ratio of error to its own standard deviation is therefore $2.41$, $6.37$, $13.19$ — **the estimate is thirteen of its own standard deviations away from the truth and getting further in those units with every year of data added**. Its reported precision at the course's sample size, $\pm0.112$, would exclude the true value at any confidence level anyone quotes.

The last three rows are the control and they say the method is not broken, only conditional. At $\nu=6$ the population kurtosis exists and the moment estimator is consistent: $8.102$, $6.757$, $6.423$, converging on $6$ from above with the usual finite-sample bias. It is simply worse, carrying a standard deviation of $0.750$ against maximum likelihood's $0.431$ at $n=6410$ — a variance ratio near three, so the cheap estimator throws away two-thirds of the information. And at one year the moment estimator's sampling distribution is barely usable at all, with a standard deviation of $32.862$ and $33.8\%$ of estimates landing above $10$. **The method works exactly where the moment exists and fails silently where it does not, and nothing in the output distinguishes the two cases.**

!!! warning "A moment estimator that returns a plausible number on data with an infinite moment has not detected the problem and cannot, because the sample moment it consumed is always finite"
    Every sample moment ever computed is a finite number. That is a property of averaging finitely many observations and carries no information whatever about whether the population moment exists, which means the failure above is invisible to any check run on the estimate itself: the point estimate is plausible, the standard error is small, the bootstrap agrees with it, and the subsample fits converge. The free diagnostic runs on the input instead of the output: **plot the running sample kurtosis against $n$ before matching anything, because a moment that exists settles down and a moment that does not climbs.** [Descriptive Statistics](../part-10-statistics-foundations/02-descriptive-statistics.md) already ran that plot on SPY and watched the excess kurtosis rise from $3.93$ to $68.38$ instead of converging on the $11.41$ the course reported, which is the signature of a fourth moment that is not there — and is a complete refutation of any $\hat\nu$ obtained by matching it. The second habit is to fit the same parameter two ways whenever a cheap estimator is available: agreement is weak evidence and disagreement of the size in the table above, $4.127$ against $2.654$, is a specification failure that no amount of additional data will resolve.

## Overidentification Turns the System Into Weighted Least Squares and Buys a Specification Test

Nothing forces the number of moment conditions to equal the number of parameters. Supply more, and the system is generally inconsistent — no $\theta$ sets all of them to zero at once — so the equalities become a minimization,

$$\hat\theta=\arg\min_\theta\ \bar g(\theta)^{\top}W\,\bar g(\theta),$$

with $\bar g$ the vector of averaged moment conditions and $W$ a positive-definite weight. This is generalized method of moments, and the residual left at the minimum is information the just-identified version threw away.

??? note "Proof that the efficient weight is the inverse covariance of the moment conditions, and that the minimized criterion is a specification test the estimator gets for free"
    Under correct specification $\sqrt n\,\bar g(\theta_0)\Rightarrow\mathcal{N}(0,\Omega)$ with $\Omega=\mathrm{var}(g(X,\theta_0))$. Standard generalized-least-squares reasoning makes $W=\Omega^{-1}$ the choice minimizing the asymptotic variance of $\hat\theta$, because it downweights the conditions that are noisiest and the combinations that are most correlated.

    With that weight the minimized criterion is

    $$J=n\,\bar g(\hat\theta)^{\top}\hat\Omega^{-1}\bar g(\hat\theta)\ \Rightarrow\ \chi^{2}_{q-p},$$

    where $q$ is the number of conditions and $p$ the number of parameters. The degrees of freedom are the conditions the parameters could not be used up on: fitting $p$ parameters sets $p$ directions of $\bar g$ to zero exactly, and the remaining $q-p$ directions are free to be non-zero, so their size is a measurable statement about whether one $\theta$ can satisfy all the conditions at once.

    For the four conditions $\mathbb{E}[d]=0$, $\mathbb{E}[d^{2}]=\sigma^{2}$, $\mathbb{E}[d^{3}]=0$, $\mathbb{E}[d^{4}]=3\sigma^{4}$ fitted with two parameters, the first two are consumed by $\hat\mu$ and $\hat\sigma^{2}$ and the efficient weight on the survivors is diagonal with entries $n/6$ and $n/24$, so the criterion becomes $J=n\big(\hat S^{2}/6+\hat\kappa^{2}/24\big)$ — the statistic usually met under a different name.

    The load-bearing quantity is $q-p$, and it is zero for every estimator in the rest of this part. **Overidentification is the only place in this part where an estimator can announce that its own model is wrong, and the price is supplying more conditions than you need** — which is why the cheapest estimator in statistics is also the only self-diagnosing one, and why the version of it that a desk actually runs is usually the just-identified one that cannot complain.

```python
import numpy as np
from scipy.stats import chi2, t as tdist

rng = np.random.default_rng(11047)
reps, crit = 20_000, chi2.ppf(0.95, 2)                         # two conditions left over

def draw(law, n):
    if law == "normal":
        return rng.standard_normal((reps, n))
    if law == "t(6)":
        return tdist.rvs(6, size=(reps, n), random_state=rng) / np.sqrt(6 / 4)
    z = rng.standard_normal((reps, n))
    return np.where(rng.random((reps, n)) < 0.05, 3 * z, 0.5 * z) / np.sqrt(0.6875)

print(f"  four moment conditions, two parameters, so J is chi2(2) with 95% point {crit:.3f}")
print("  law                   n    mean skew    mean excess kurt    median J    mean J"
      "    reject rate")
for law in ("normal", "t(6)", "0.95/0.05 mixture"):
    for n in (252, 1260, 6410):
        d = draw(law, n)
        d -= d.mean(axis=1, keepdims=True)
        m2 = (d ** 2).mean(axis=1)
        s = (d ** 3).mean(axis=1) / m2 ** 1.5
        k = (d ** 4).mean(axis=1) / m2 ** 2 - 3
        j = n * (s ** 2 / 6 + k ** 2 / 24)                     # efficient weight under the null
        print(f"  {law:<18} {n:5d} {s.mean():12.4f} {k.mean():19.4f} {np.median(j):11.3f}"
              f" {j.mean():9.1f} {(j > crit).mean():14.4f}")
# =>   four moment conditions, two parameters, so J is chi2(2) with 95% point 5.991
#      law                   n    mean skew    mean excess kurt    median J    mean J    reject rate
#      normal               252      -0.0003             -0.0193       1.266       1.9         0.0463
#      normal              1260       0.0000             -0.0040       1.373       2.0         0.0514
#      normal              6410       0.0002             -0.0014       1.382       2.0         0.0462
#      t(6)                 252      -0.0018              2.1572      25.796     159.7         0.8322
#      t(6)                1260      -0.0015              2.6741     229.587     868.3         1.0000
#      t(6)                6410       0.0002              2.9060    1627.376    4319.8         1.0000
#      0.95/0.05 mixture    252       0.0100             18.0813    2979.378    4511.3         0.9990
#      0.95/0.05 mixture   1260       0.0071             22.0332   23814.090   27421.3         1.0000
#      0.95/0.05 mixture   6410      -0.0013             22.8749  136878.287  141956.3         1.0000
```

The normal rows are the calibration and they confirm the distributional claim rather than assume it. The mean $J$ is $1.9$, $2.0$, $2.0$ against the $\chi^{2}_{2}$ mean of exactly $2$, and the rejection rate at the nominal five-percent point is $0.0463$, $0.0514$, $0.0462$ — the statistic has the size the proof predicts, at sample sizes a desk actually has.

The other six rows are power. Under a $t(6)$ the excess kurtosis climbs to its true value of $3$ — $2.1572$, $2.6741$, $2.9060$ — and the criterion rises with it, rejecting $83.22\%$ of the time on a single year of data and every time thereafter. Under a five-percent mixture the kurtosis sits near $22.9$ and the median criterion reaches $136{,}878$ against a critical value of $5.991$. **The two conditions the parameters could not absorb are shouting, and they cost nothing but the decision to write them down.**

Read against the previous section, this is the page's real argument. The just-identified moment estimator of $\nu$ was silently wrong by a factor of $1.56$ and reported a shrinking standard error while it happened. The overidentified version of the same idea, applied to the same kind of data, announces a specification failure with a statistic four orders of magnitude past its critical value. Same method, same arithmetic, and the difference is entirely whether $q-p$ was allowed to be positive. Turning that statistic into a formal decision with a stated error rate is [Part XII](../part-12-hypothesis-testing/index.md); here it is a residual the estimator produced without being asked.

## The Cheapest Estimator in Statistics and the Bill Arrives in the Fourth Moment

This page established that moment matching is a solve rather than an optimization and depends on the data only through the moments it consumed; that consistency and asymptotic normality follow from the law of large numbers, the continuous mapping theorem and the delta method with no density anywhere, at the cost of putting the $2k$th moment inside the standard error for the $k$th; that a sample autocorrelation is biased downward by a known function of $n$ and that a logarithm turns a $9\%$ error in $\rho$ into a $29\%$ error in a tradeable half-life, $2.49$ days against $3.49$, understating the horizon in four samples out of five; that matching a kurtosis that does not exist produces an estimate converging on $4.127$ when the truth is $2.65$, with a standard deviation of $0.112$ and an error of thirteen of them; and that supplying two conditions more than the parameter count produces a statistic with the right size under the null and near-certain power against the tails the data actually has.

The method's reputation as a crude first pass is half right and misleadingly framed. It is not less accurate than maximum likelihood in the way a rougher approximation is less accurate; it is accurate under a different and weaker assumption, requiring only that certain moments exist and that a map be invertible, where the likelihood requires the entire density. When both assumptions hold, the likelihood wins on variance, by a factor of about three in the table above. When the density assumption fails, the moment estimator can be the more honest of the two. And when the *moment* assumption fails, the moment estimator does not degrade gracefully, it converges on the boundary of its own derivation and reports the convergence as precision.

That is the pattern worth carrying forward, because it is not special to moments. Both estimators in this part so far are silent about the assumption they rest on, and both convert that silence into a shrinking standard error. Maximum likelihood converges on the Kullback–Leibler projection and reports its tightness; the method of moments converges on the boundary and reports the same thing. Neither has any mechanism for expressing doubt about the model, because neither was given one — and the natural place to put a mechanism for doubt is a distribution over the parameter, which is what [Bayesian Estimation](05-bayesian-estimation.md) supplies and what it charges for.

**The method of moments will always return an answer, and the only question worth asking about that answer is whether the moment it came from exists.**
