# Bayesian Prediction

The predictive distribution is the only object in this part that a desk can act on, and it is routinely confused with the one thing it is not: a statement about a parameter. Integrating the posterior against the likelihood produces a distribution over the *next observation*, and the operation is exact, closed-form for every conjugate family, and buys a measurable improvement — a nominal ninety-five per cent interval for tomorrow's return covers $0.8520$ of the time when built by plugging in an estimate and $0.9502$ when built by integrating, at five days of history. The improvement is also small and shrinks fast, because the quantity it prices is small: parameter uncertainty is $1/(n+1)$ of the total predictive variance, which is $0.0099$ at a hundred observations and $0.0010$ at a thousand. Below that threshold sits the term the predictive has no way to price at all. A Gaussian predictive fitted to Student-$t$ returns needs $13.89\%$ more width to be honest about a one-per-cent daily loss, and parameter uncertainty supplies $0.85\%$ of it — a factor of sixteen short — so the interval breaches on $0.0165$ of days against a promised $0.0100$ while every calculation inside it is correct.

This page covers the posterior predictive distribution and its exact variance decomposition, the plug-in predictive as the same integral with the parameter uncertainty deleted, the closed forms conjugacy delivers, the calibration each one achieves, posterior predictive checking, and the limit of the whole construction. It constructs no prior, which is [Prior Distributions](02-prior-distributions.md); it derives no posterior and proves no asymptotics, which is [Posterior Distributions](03-posterior-distributions.md); it proves no closure and derives no conjugate update, which is [Conjugate Priors](04-conjugate-priors.md); it updates nothing sequentially, which is [Bayesian Updating](05-bayesian-updating.md); it computes no Bayes factor and averages over no models, which is [Bayesian Model Comparison](06-bayesian-model-comparison.md); it defines no prediction or tolerance interval and does not distinguish them from a confidence interval, which is [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md); it derives no conditional expectation as an $L^{2}$ projection, which is [Conditional Expectation](../part-04-expectation-and-moments/06-conditional-expectation.md); it estimates no out-of-sample error by resampling, which are [Cross-Validation](../part-14-model-selection/02-cross-validation.md) and [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it prices no option and simulates no portfolio, which are [Monte Carlo Option Pricing](../part-18-quant-finance-applications/08-monte-carlo-option-pricing.md) and [Portfolio Risk Simulation](../part-18-quant-finance-applications/09-portfolio-risk-simulation.md); and it never presents an interval around a parameter as an interval around an outcome.

The trading stake is a course lesson printing a number that is read as a forecast about a hundred times more often than it is read correctly. [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) updates a momentum strategy's edge across five vintages and reports `through 2025: ann mean +5.8% +/- 7.7%`, observing that "the interval narrows like $1/\sqrt{n}$, exactly as theory promises — and after twenty-four years it is still ±7.7% around a +5.8% mean." Every word of that is right, and the interval is around the *mean*: it says where the long-run edge sits, not what next year will do. The predictive interval for a single year is wider by the ratio section 2 measures, because it must also contain a year of volatility that no amount of history removes. A risk committee shown $\pm7.7\%$ and told it describes next year has been shown the wrong distribution.

## The Predictive Integrates the Posterior, and the Plug-In Is the Same Integral With One Term Deleted

The construction is one line and the line is worth writing out, because what it fixes and what it cannot fix are both visible in it.

??? note "Proof that $\mathrm{Var}(\tilde y\mid x)=\mathbb{E}[\mathrm{Var}(\tilde y\mid\theta)\mid x]+\mathrm{Var}(\mathbb{E}[\tilde y\mid\theta]\mid x)$, so the plug-in predictive drops exactly the second term, and that for a normal model the missing share is $1/(n+1)$"

    The posterior predictive density for a future observation $\tilde y$, conditionally independent of the past given $\theta$, is
    $$p(\tilde y\mid x)=\int f(\tilde y\mid\theta)\,\pi(\theta\mid x)\,\mathrm{d}\theta,$$
    a mixture of the model's own sampling distributions weighted by the posterior. Applying the law of total variance under the posterior,
    $$\mathrm{Var}(\tilde y\mid x)=\underbrace{\mathbb{E}\big[\mathrm{Var}(\tilde y\mid\theta)\,\big|\,x\big]}_{\text{sampling noise}}+\underbrace{\mathrm{Var}\big(\mathbb{E}[\tilde y\mid\theta]\,\big|\,x\big)}_{\text{parameter uncertainty}} .$$
    The plug-in predictive $f(\tilde y\mid\hat\theta)$ conditions on a single $\theta$, so its variance is $\mathrm{Var}(\tilde y\mid\hat\theta)$ and the second term is absent — not approximated, deleted. The plug-in is the posterior predictive under a posterior collapsed to a point mass.

    For $y_i\sim N(\mu,\sigma^{2})$ with the Jeffreys prior of [Prior Distributions](02-prior-distributions.md), the joint posterior is normal–inverse-gamma and the integral is available: $\mu\mid\sigma^{2},x\sim N(\bar y,\sigma^{2}/n)$ gives $\tilde y\mid\sigma^{2},x\sim N(\bar y,\sigma^{2}(1+1/n))$, and integrating $\sigma^{2}$ against its inverse-gamma posterior yields
    $$\tilde y\mid x\ \sim\ t_{n-1}\Big(\bar y,\ s^{2}\big(1+\tfrac1n\big)\Big).$$
    Two corrections appear where the plug-in $N(\bar y,s^{2})$ has none: the factor $1+1/n$ prices uncertainty about the mean, and the Student-$t$ tails price uncertainty about the variance. The share of predictive variance attributable to the parameter is
    $$\frac{\sigma^{2}/n}{\sigma^{2}+\sigma^{2}/n}=\frac{1}{n+1},$$
    so it is a sixth of the total at $n=5$ and a thousandth at $n=1000$.

    Conjugacy delivers the same integral in closed form throughout the catalogue of [Conjugate Priors](04-conjugate-priors.md): a Beta posterior against a Bernoulli likelihood gives a beta-binomial predictive, a Gamma posterior against a Poisson likelihood gives a negative binomial, and a normal–inverse-gamma posterior gives the Student-$t$ above. In each case the predictive is over-dispersed relative to the plug-in by a factor that depends only on the pseudo-count.

    **The load-bearing observation is that the second term measures uncertainty about $\theta$ within the model and nothing else, so it vanishes as $n$ grows whatever the model is — including a model that is wrong, where the first term is being computed from the wrong family and no amount of data corrects it.** That is section 4.

## Integrating Rather Than Plugging In Repairs the Interval Exactly, and the Repair Is Worth Least Where Most Data Exists

The correction is exact and its size is entirely predictable:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(16071)
sig, reps = 0.010, 200_000

print(f"  a nominal 95% interval for tomorrow's return, from n days of history at"
      f" {sig * 1e4:.0f}bp; the plug-in uses N(xbar, s^2) and the predictive integrates"
      f" the posterior, giving Student-t with n-1 degrees of freedom")
print("        n   plug-in width, bp   predictive width, bp   ratio   plug-in coverage"
      "   predictive coverage   parameter share of variance")
for n in (5, 10, 30, 100, 1000):
    x = rng.standard_normal((reps, n)) * sig
    xb, s = x.mean(1), x.std(1, ddof=1)
    y = rng.standard_normal(reps) * sig                # tomorrow
    zc, tc = stats.norm.isf(0.025), stats.t.isf(0.025, n - 1)
    hp = zc * s
    hq = tc * s * np.sqrt(1 + 1 / n)
    print(f"    {n:5d}   {2 * hp.mean() * 1e4:17.4f}   {2 * hq.mean() * 1e4:20.4f}"
          f"   {(hq / hp).mean():5.4f}   {(np.abs(y - xb) <= hp).mean():16.4f}"
          f"   {(np.abs(y - xb) <= hq).mean():21.4f}   {1 / (n + 1):27.4f}")
# =>   a nominal 95% interval for tomorrow's return, from n days of history at 100bp; the plug-in uses N(xbar, s^2) and the predictive integrates the posterior, giving Student-t with n-1 degrees of freedom
#            n   plug-in width, bp   predictive width, bp   ratio   plug-in coverage   predictive coverage   parameter share of variance
#            5            368.3654               571.6239   1.5518             0.8520                  0.9502                        0.1667
#           10            381.2821               461.5486   1.2105             0.9049                  0.9492                        0.0909
#           30            388.7991               412.4198   1.0608             0.9370                  0.9507                        0.0323
#          100            390.9232               397.7344   1.0174             0.9465                  0.9505                        0.0099
#         1000            391.8552               392.5267   1.0017             0.9491                  0.9495                        0.0010
```

The predictive column is correct at every sample size — coverage of $0.9502$, $0.9492$, $0.9507$, $0.9505$ and $0.9495$ against a nominal $0.95$, from five days of history to a thousand. That is the whole claim for the construction, and it is a clean one: integrating the posterior produces an interval whose stated coverage is its actual coverage even when almost nothing is known about the parameters.

The plug-in column shows what deleting the term costs: $0.8520$, $0.9049$, $0.9370$, $0.9465$ and $0.9491$. At five observations an interval sold as ninety-five per cent delivers eighty-five, and the shortfall is entirely explained by a width ratio of $1.5518$ — the predictive interval is half as wide again, $571.6239$ basis points against $368.3654$. By a hundred observations the ratio is $1.0174$ and the coverage gap is four tenths of a point; by a thousand it is $1.0017$ and the two are indistinguishable.

The last column is why. Parameter uncertainty's share of the total predictive variance is exactly $1/(n+1)$ — $0.1667$, $0.0909$, $0.0323$, $0.0099$ and $0.0010$ — so the thing the predictive prices correctly is a sixth of the problem at five observations and a tenth of a per cent at a thousand. **The posterior predictive is unambiguously the right object and its advantage over the plug-in is confined to small samples, which means a desk with years of history gains almost nothing from the correction and a desk evaluating a new strategy gains a great deal.** The mistake worth avoiding is the opposite one, of assuming that because the correction is small at large $n$ the predictive and the posterior are interchangeable — they are different distributions with different widths at every $n$, as the trading stake's $\pm7.7\%$ illustrates.

## The Same Integral in Counts Is a Beta-Binomial, and It Is the Difference Between a Bad Month and a Broken Model

Returns are continuous and the questions risk committees ask are frequently not. How many of the next twenty trades will win, how many days of the next month will lose, how long a losing run should be tolerated before a strategy is switched off — all are count questions, and the conjugate machinery answers them in closed form:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(16073)
p_true, m, reps = 0.54, 20, 200_000

print(f"  winning trades in the next {m}, forecast from n past trades at a true hit rate"
      f" of {p_true}; the plug-in is Binomial(m, khat/n) and the predictive is"
      f" beta-binomial, {reps:,} runs")
print("        n   plug-in sd   predictive sd   ratio   below the 5% point, plug-in"
      "   predictive   90% interval hits, plug-in   predictive")
for n in (20, 50, 200, 1000):
    k = rng.binomial(n, p_true, reps)
    a, b = 1 + k, 1 + n - k
    ph = k / n
    sd_p = np.sqrt(m * ph * (1 - ph))
    sd_q = np.sqrt(m * (a / (a + b)) * (b / (a + b)) * (a + b + m) / (a + b + 1))
    fut = rng.binomial(m, p_true, reps)
    lo_p, hi_p = stats.binom.ppf(0.05, m, ph), stats.binom.isf(0.05, m, ph)
    lo_q, hi_q = stats.betabinom.ppf(0.05, m, a, b), stats.betabinom.isf(0.05, m, a, b)
    print(f"    {n:5d}   {sd_p.mean():10.4f}   {sd_q.mean():13.4f}"
          f"   {sd_q.mean() / sd_p.mean():5.4f}   {(fut < lo_p).mean():26.4f}"
          f"   {(fut < lo_q).mean():10.4f}"
          f"   {((fut >= lo_p) & (fut <= hi_p)).mean():27.4f}"
          f"   {((fut >= lo_q) & (fut <= hi_q)).mean():10.4f}")
# =>   winning trades in the next 20, forecast from n past trades at a true hit rate of 0.54; the plug-in is Binomial(m, khat/n) and the predictive is beta-binomial, 200,000 runs
#            n   plug-in sd   predictive sd   ratio   below the 5% point, plug-in   predictive   90% interval hits, plug-in   predictive
#           20       2.1705          2.9491   1.3587                       0.0935       0.0393                        0.8249       0.9205
#           50       2.2062          2.5741   1.1668                       0.0614       0.0320                        0.8783       0.9353
#          200       2.2232          2.3252   1.0459                       0.0393       0.0324                        0.9237       0.9350
#         1000       2.2278          2.2488   1.0094                       0.0316       0.0309                        0.9339       0.9361
```

The beta-binomial's standard deviation exceeds the plug-in binomial's by a factor of $1.3587$ at twenty trades of history, $1.1668$ at fifty, $1.0459$ at two hundred and $1.0094$ at a thousand — the same $1/(n+1)$ story in different units, since the extra variance is $m(m+a+b)/(a+b+1)$ against the plug-in's $m$.

The calibration columns are where it matters operationally. A desk that sets its downside trigger at the fifth percentile of a plug-in binomial will see that trigger breached on $0.0935$ of months when it has twenty trades of history — nearly double the intended rate, so a rule designed to fire once in twenty months fires once in eleven. The beta-binomial's fifth percentile is breached on $0.0393$. The interval columns say the same thing symmetrically: a nominal ninety-per-cent range covers $0.8249$ under the plug-in and $0.9205$ under the predictive at twenty trades, converging to $0.9339$ and $0.9361$ by a thousand. Neither hits $0.90$ exactly at any $n$, because a discrete distribution cannot produce an exact interval and both are conservative in the limit — a fact about counts rather than about either method.

**The practical consequence is that every strategy-shutdown rule calibrated on a short track record fires roughly twice as often as intended, and the repair is to replace one distribution with another that is already in closed form.** The premature shutdown is a real cost: [Data Snooping Bias](../part-15-multiple-testing/04-data-snooping-bias.md) prices the damage of selecting on short records, and this is the same short record producing a different error at the other end of a strategy's life.

The same construction answers the waiting-time version of the question, which is the one risk committees actually phrase. Asking how many trades will pass before the next loss puts a geometric likelihood against the Beta posterior and yields a beta-geometric predictive; asking how many losses will occur in a fixed number of trades puts a Poisson likelihood against a Gamma posterior and yields a negative binomial, whose over-dispersion relative to the Poisson is exactly the ratio of the two variances $\lambda(1+\lambda/\alpha)$ to $\lambda$. Both are in the catalogue of [Conjugate Priors](04-conjugate-priors.md) and neither requires a simulation. The pattern across all of them is the one section 1's proof predicts: the predictive is the plug-in distribution with its dispersion inflated by a factor determined entirely by the pseudo-count, so the correction is large when the track record is short and negligible when it is long, and the direction is always toward admitting more extreme outcomes than the fitted point estimate suggests.

What none of these closed forms changes is the shape. A beta-binomial is still a distribution over a fixed number of exchangeable trials, and a negative binomial is still a mixture of Poissons; if losing trades cluster — because they share a market regime, which is the assumption [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md) exists to relax — then the true count distribution has a heavier tail than either, and the predictive's extra width is being computed in the wrong family. That is the same defect the next section measures in continuous returns, arriving here in counts.

## The Predictive Prices Uncertainty About the Parameter and Has No Term for Uncertainty About the Model

Everything above is correct arithmetic inside a model. The decomposition in section 1 has two terms and both are computed under $f(\cdot\mid\theta)$, so neither can register that $f$ itself is the wrong family.

??? note "Proof that the predictive's Kullback–Leibler divergence from the truth converges to a strictly positive constant under misspecification, so the error parameter uncertainty prices vanishes while the error it cannot price does not"

    Let the truth be $p_0$ and the model be $\{f(\cdot\mid\theta)\}$ with pseudo-true value $\theta^{*}$ minimizing $\mathrm{KL}(p_0\|f(\cdot\mid\theta))$, the object [Posterior Distributions](03-posterior-distributions.md) establishes. The posterior concentrates on $\theta^{*}$ at rate $n^{-1/2}$, so the predictive satisfies
    $$p(\tilde y\mid x)=\int f(\tilde y\mid\theta)\pi(\theta\mid x)\,\mathrm{d}\theta\ \longrightarrow\ f(\tilde y\mid\theta^{*}),$$
    and therefore
    $$\mathrm{KL}\big(p_0\,\|\,p(\cdot\mid x)\big)\ \longrightarrow\ \mathrm{KL}\big(p_0\,\|\,f(\cdot\mid\theta^{*})\big)=:D^{*}>0$$
    whenever $p_0$ is outside the model class. Expanding, the total predictive error decomposes into a term of order $D^{*}$ that does not depend on $n$ and a term of order $1/n$ from posterior spread. **Integrating the posterior removes the second term exactly and leaves the first untouched, so as data accumulates the predictive converges — to the wrong distribution, at which point the correction this page is about has become worthless while the error it does not address is the whole error.**

    The size of $D^{*}$ is a property of the model class, and for a normal predictive against heavy-tailed truth it concentrates in the tails, which is where a risk system reads. The load-bearing asymmetry is one of diagnosability: the parameter-uncertainty term is computable in closed form from the posterior, and $D^{*}$ can only be detected by comparing predictions with outcomes — which is what a posterior predictive check does, and why it is the only diagnostic in this part that could ever find the failure.

The two errors are directly comparable in the same units:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(16075)
sig, n, reps = 0.010, 250, 100_000

print(f"  a Gaussian predictive fitted to Student-t returns, {n} days of history,"
      f" {reps:,} runs; parameter uncertainty is priced exactly and model error is not")
print("       df   parameter inflation   model inflation needed   1% VaR, model"
      "   1% VaR, truth   predictive P(breach)   realized")
for df in (3, 4, 6, 12, 1000):
    sc = sig / np.sqrt(df / (df - 2))
    x = stats.t.rvs(df, scale=sc, size=(reps, n), random_state=rng)
    xb, s = x.mean(1), x.std(1, ddof=1)
    par = np.sqrt(1 + 1 / n) * stats.t.isf(0.01, n - 1) / stats.norm.isf(0.01)
    truth = stats.t.ppf(0.01, df, scale=sc)
    q = xb + s * np.sqrt(1 + 1 / n) * stats.t.ppf(0.01, n - 1)
    y = stats.t.rvs(df, scale=sc, size=reps, random_state=rng)
    print(f"    {df:5d}   {par:19.4f}   {truth / (sig * stats.norm.ppf(0.01)):22.4f}"
          f"   {q.mean() * 1e2:14.4f}%   {truth * 1e2:13.4f}%   {0.01:20.4f}"
          f"   {(y < q).mean():10.4f}")
# =>   a Gaussian predictive fitted to Student-t returns, 250 days of history, 100,000 runs; parameter uncertainty is priced exactly and model error is not
#           df   parameter inflation   model inflation needed   1% VaR, model   1% VaR, truth   predictive P(breach)   realized
#            3                1.0085                   1.1269          -2.2843%         -2.6216%                 0.0100       0.0158
#            4                1.0085                   1.1389          -2.3310%         -2.6495%                 0.0100       0.0165
#            6                1.0085                   1.1030          -2.3408%         -2.5660%                 0.0100       0.0152
#           12                1.0085                   1.0520          -2.3432%         -2.4474%                 0.0100       0.0129
#         1000                1.0085                   1.0006          -2.3445%         -2.3278%                 0.0100       0.0102
```

The `parameter inflation` column is what a correctly executed Bayesian predictive adds at two hundred and fifty days of history: a factor of $1.0085$, held constant across every row because it depends only on $n$. The `model inflation needed` column is the factor by which the Gaussian one-per-cent quantile would have to be widened to match the truth: $1.1269$, $1.1389$, $1.1030$, $1.0520$ and $1.0006$. At four degrees of freedom the model is short by $13.89\%$ and parameter uncertainty supplies $0.85\%$ of it, a factor of sixteen.

The consequence is in the last two columns. The predictive promises a breach rate of $0.0100$ and delivers $0.0158$, $0.0165$, $0.0152$, $0.0129$ and $0.0102$. At four degrees of freedom a limit designed to be exceeded two or three times a year is exceeded four, and the last row — where the data really is Gaussian — confirms that the machinery is not at fault: it returns $0.0102$ against a nominal $0.0100$. The Gaussian predictive puts the one-per-cent daily loss at $-2.3310\%$ where the truth is $-2.6495\%$.

**Integrating the posterior is the right thing to do and it addresses the smaller of the two errors by a factor of sixteen, which is the honest summary of this page and of a great deal of applied Bayesian work.** A team that has moved from plug-in to fully integrated predictive intervals has made a real improvement worth $0.85\%$ of width, and if their model class is Gaussian and their returns are not, they have left $13.89\%$ on the table and have no term anywhere in the calculation that would tell them.

## The One Check That Could Find It Uses the Data Twice, and Is Worth Running Anyway

The failure of section 4 is invisible to every internal quantity and visible to one procedure. A posterior predictive check draws replicate datasets $y^{\text{rep}}$ from $p(\cdot\mid x)$, computes a discrepancy statistic on each, and locates the observed value among the replicates; the tail area is a posterior predictive p-value. Choosing a statistic the model was *not* fitted to is the whole art — kurtosis or a count of four-sigma days for a Gaussian assumption, lag-one autocorrelation for an independence assumption, the maximum drawdown for a model fitted on daily moments — because a statistic the fitting procedure already matched will match by construction.

The known defect is that the data appears on both sides. Because $y$ was used to form $\pi(\theta\mid x)$ and is then compared against replicates from that same posterior, the posterior predictive p-value is conservative: its distribution under the null is not uniform but concentrated toward the middle, so a value of $0.2$ is weak evidence of nothing and only extreme values carry information. Cross-validated and prior-predictive variants repair the double use at the cost of computation, and the probability integral transform — checking that the predictive CDF evaluated at each realized outcome is uniform — gives a sequential version that is genuinely out of sample when applied to a live forecast record. That last is what a calibration score measures, and it is the reason [Production ML](../../part-07-machine-learning/05-production-ml.md) gates promotions on a Brier score rather than on a discrimination metric alone.

The scoring-rule literature completes the picture. A proper scoring rule — logarithmic, Brier, or continuous ranked probability — is minimized in expectation by the true predictive distribution, so scoring a predictive against realized outcomes measures exactly the quantity section 4's proof calls $D^{*}$ plus a constant. Comparing two predictives by their average score on data neither has seen is therefore a direct measurement of model error, and it requires no assumption that either model is correct — which is the property that makes it the right final check on everything in this part.

That property is worth contrasting with the marginal likelihood of [Bayesian Model Comparison](06-bayesian-model-comparison.md), because the two are closer than they look and differ where it counts. The log marginal likelihood decomposes as a sum of one-step-ahead predictive log scores, $\log m(y_{1:n})=\sum_t\log p(y_t\mid y_{1:t-1})$, so a Bayes factor *is* a cumulative predictive scoring comparison — conducted entirely inside the prior, over the whole sequence, and therefore inheriting the prior sensitivity that page measured. Scoring a predictive on genuinely held-out data is the same comparison conducted after the priors have been washed out by the training portion, which is why it is far less sensitive and also why it cannot be computed before the data exists. Leave-one-out cross-validated predictive scores, and the information criteria built on them, occupy the ground between: they approximate held-out scoring using the fitted posterior, at the cost of the double use this section began with.

None of these routes rescues a model class that excludes the truth; they only reveal it. The value of a predictive score is that revelation is possible at all — every other quantity in this part is computed inside the model and can only report on the model's internal consistency, whereas a score compares a distribution with the outcomes it claimed to describe. **The practical rule that falls out is that a Bayesian analysis should publish exactly one number computed outside its own assumptions, and the cheapest such number is the score its predictive earned on data it had not seen.**

!!! note "The prior predictive, the posterior predictive, the plug-in predictive, a prediction interval and a tolerance interval are five distributions over future data, and the first three differ only in which distribution the parameter is integrated against"
    All five answer questions about observations rather than parameters, and [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md) fixes the interval taxonomy that the last two belong to. The **prior predictive** $\int f(\tilde y\mid\theta)\pi(\theta)\mathrm{d}\theta$ integrates against the prior, uses no data, and is the object [Prior Distributions](02-prior-distributions.md) recommends generating from as a check and [Bayesian Model Comparison](06-bayesian-model-comparison.md) evaluates at the observed data and calls the marginal likelihood — the same function read as a check, as a forecast and as a score. The **posterior predictive** integrates against the posterior and is this page. The **plug-in predictive** integrates against a point mass at $\hat\theta$ and is the posterior predictive with the parameter-uncertainty term deleted, costing $0.8520$ coverage against $0.9502$ at five observations. A **prediction interval** is a frequentist interval for a single future observation, which the posterior predictive interval coincides with numerically here and interprets differently; a **tolerance interval** covers a stated proportion of the future population with stated confidence and is a different object again. Quoting a posterior interval for a mean as though it described a single future outcome — the $\pm7.7\%$ of the trading stake — is the error this list exists to prevent.

!!! warning "A predictive interval is exactly as good as the model class it integrates over, and every diagnostic available inside the calculation reports on the term that is already small"
    The two errors on this page are not comparable in size and are comparable in visibility. Parameter uncertainty is computable in closed form, is quoted in every well-run analysis, and accounts for $0.1667$ of predictive variance at five observations, $0.0099$ at a hundred and $0.0010$ at a thousand. Model error is invisible to the posterior, invisible to the predictive variance, invisible to the credible interval, and at two hundred and fifty days of Student-$t$ returns requires $13.89\%$ of extra width where parameter uncertainty supplies $0.85\%$ — so the predictive breaches its one-per-cent limit on $0.0165$ of days while reporting that it will breach on $0.0100$. **The free diagnostic is to keep a live record of the probability integral transform of every predictive you publish — the value your predictive CDF assigns to the outcome that actually occurred — and to test that running series for uniformity once a quarter, because parameter uncertainty is the term you can compute and calibration against realized outcomes is the only term that can tell you the model class is wrong.** It costs one number stored per forecast, it needs no theory, and it is the only check in this part that is genuinely out of sample.

## An Honest Width Around a Model Nobody Checked

This page established that the posterior predictive integrates the likelihood against the posterior, that the law of total variance splits its variance into sampling noise and parameter uncertainty, and that the plug-in predictive deletes the second term exactly rather than approximating it; that for a normal model the deleted share is $1/(n+1)$, measured at $0.1667$, $0.0909$, $0.0323$, $0.0099$ and $0.0010$, and that restoring it produces coverage of $0.9502$, $0.9492$, $0.9507$, $0.9505$ and $0.9495$ against a nominal $0.95$ where the plug-in delivers $0.8520$, $0.9049$, $0.9370$, $0.9465$ and $0.9491$, at a width ratio falling from $1.5518$ to $1.0017$; that the same integral in counts is a beta-binomial, over-dispersed by $1.3587$ at twenty trades of history, so a fifth-percentile shutdown trigger built on a plug-in binomial is breached on $0.0935$ of months against the predictive's $0.0393$ and a nominal ninety-per-cent range covers $0.8249$ against $0.9205$; and that none of this addresses model error, a Gaussian predictive on Student-$t$ returns needing $1.1389$ of extra width where parameter uncertainty supplies $1.0085$, putting the one-per-cent daily loss at $-2.3310\%$ against a truth of $-2.6495\%$ and breaching on $0.0165$ of days against a promised $0.0100$.

The shape shared by all three exhibits is the part's closing statement of its own thesis. Every correction on this page is exact, and every one of them is a correction *within* an assumed model: the $1+1/n$, the Student-$t$ tails, the beta-binomial over-dispersion are all computed from the posterior, and the posterior was computed from a prior and a likelihood neither of which the data was ever asked to confirm. The framework is scrupulous about the uncertainty it can see and silent about the uncertainty it cannot, and the second is larger by a factor of sixteen in the one case measured here.

That silence is where this part ends and the next begins. Every page here has either assumed the required integral was available in closed form or computed it on a grid small enough to enumerate, and both routes fail at the dimension a real hierarchical or state-space model occupies. The machinery that makes those models computable — chains whose stationary distribution is the posterior, the expectation-maximization recursion, and the optimizers underneath both — is [Part XVII](../part-17-statistical-computing/index.md).

**A posterior predictive gives an honest account of everything the model does not know about its own parameters, and no account whatsoever of the one thing that matters most, which is whether the model should have been that model.**
