# Geometric Brownian Motion

Prices are positive and move multiplicatively, so the natural model is not Brownian motion but its exponential. That one change produces the two facts that organize the whole of quantitative investing: the growth an investor actually experiences is lower than the average return by exactly $\sigma^{2}/2$, and the drift parameter is not estimable from any quantity of data collected over any horizon a business can wait for — while the volatility parameter is estimable to whatever precision you are willing to pay for.

This page covers the stochastic differential equation and the closed-form solution with its drag term, the lognormal marginals and the three different central values they supply, the estimation asymmetry between drift and volatility and the exact reason sampling frequency helps one and not the other, what that asymmetry does to the shape of a quantitative business, and the leverage arithmetic in which the mean rises monotonically while the median turns over. It does not derive Itô's lemma, which is [Stochastic Calculus](../../advanced/03-stochastic-calculus.md); it does not develop the distribution itself, which is [Lognormal Distribution](../part-05-common-distributions/12-lognormal-distribution.md); it does not build the driving process, which is [Brownian Motion](08-brownian-motion.md); it does not price a contingent claim, which is [Options Pricing](../../advanced/11-options-pricing.md); and it constructs no sizing rule, which is [Position Sizing and Risk Budgeting](../../part-04-strategy-development/06-position-sizing-and-risk-budgeting.md).

The trading stake is the asymmetry the appendix opened with. [Expected Value](../part-04-expectation-and-moments/01-expected-value.md) reports that on twenty-five years of index data "the annualized volatility is pinned to within $0.9\%$ of itself and the annualized mean to within $52\%$ of itself — a fifty-eight-fold gap between two numbers computed from one file by one procedure", and treats it as a fact about sample sizes. The third section shows it is not. It is a structural property of this process, the gap widens without limit as data arrives faster, and no amount of intraday data closes any of it.

## The Solution Carries a Drag Term, and It Is Not a Cost

**Geometric Brownian motion** is the process satisfying

$$dS_t=\mu S_t\,dt+\sigma S_t\,dW_t,$$

read as "the *proportional* change has a deterministic part $\mu\,dt$ and a random part $\sigma\,dW_t$". Dividing by $S_t$ is what makes it multiplicative and what keeps it positive: the increments scale with the level, so the process approaches zero without reaching it. Its solution is not what a first look at the equation suggests.

$$S_t=S_0\exp\Bigl(\bigl(\mu-\tfrac{\sigma^{2}}{2}\bigr)t+\sigma W_t\Bigr).$$

The $-\sigma^{2}/2$ in the exponent is the **volatility drag**, and the single most common error in this subject is reading it as a cost — a fee that volatility charges. It is not. It is the difference between two questions.

??? note "Proof that the log of the process drifts at mu minus half sigma squared, and that nothing was lost"
    The direct route is Itô's lemma applied to $f(S)=\log S$, which [Stochastic Calculus](../../advanced/03-stochastic-calculus.md) derives: the second-order term contributes $\tfrac12 f''(S)\sigma^{2}S^{2}=-\tfrac12\sigma^{2}$ because $(dW)^{2}=dt$, giving $d\log S_t=(\mu-\tfrac{\sigma^{2}}{2})dt+\sigma\,dW_t$, which integrates to the stated solution.

    The route that needs no stochastic calculus is a discrete multiplicative limit, and it makes the drag's origin visible. Compound $n$ independent gross returns $1+R_i$ with $\mathbb{E}[R_i]=\mu\,dt$ and $\mathrm{var}(R_i)=\sigma^{2}dt$ over $dt=t/n$. Then

    $$\log\prod_{i}(1+R_i)=\sum_i\log(1+R_i)=\sum_i\Bigl(R_i-\tfrac{R_i^{2}}{2}+O(R_i^{3})\Bigr),$$

    and taking expectations, $\mathbb{E}[\log(1+R_i)]\approx\mu\,dt-\tfrac12(\sigma^{2}dt+\mu^{2}dt^{2})\to\mu\,dt-\tfrac12\sigma^{2}dt$ as $dt\to0$. Summing over the $n$ periods gives a log-drift of $(\mu-\tfrac{\sigma^{2}}{2})t$, and the [Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md) applied to the sum of logs delivers the Gaussian exponent — which is why the marginal is [lognormal](../part-05-common-distributions/12-lognormal-distribution.md) and why that page calls the construction a multiplicative central limit theorem.

    The load-bearing step is the second-order term in the expansion of $\log(1+R)$, and it is spent on the concavity of the logarithm. **Nothing has been subtracted from anyone.** The arithmetic mean return is still $\mu$ and $\mathbb{E}[S_t]=S_0e^{\mu t}$ exactly; the drag is the gap between the log of an average and the average of a log, which [Change of Variables](../part-03-random-variables/09-change-of-variables.md) states precisely when it notes the wedge "is a property of the exponential map's curvature, not a cost charged by the market."

## Three Central Values, and the One an Investor Receives

Because $\log S_t$ is Gaussian, $S_t$ is lognormal, and a lognormal has three different central values that diverge as the horizon lengthens: mean $S_0e^{\mu t}$, median $S_0e^{(\mu-\sigma^{2}/2)t}$, and mode $S_0e^{(\mu-3\sigma^{2}/2)t}$. Their ordering is fixed and their separation grows linearly in $t$ in the exponent.

```python
import numpy as np

rng = np.random.default_rng(8091)
reps, mu, sig = 1_000_000, 0.08, 0.20
print(f"  terminal wealth of GBM with mu {mu} and sigma {sig}, {reps} paths, S0 = 1")
print("        T    mean    exp(mu T)    median    exp((mu-s2/2)T)    frac below the mean")
for T in (1, 5, 10, 30):
    s = np.exp((mu - 0.5 * sig ** 2) * T + sig * np.sqrt(T) * rng.standard_normal(reps))
    print(f"  {T:9d} {s.mean():7.3f} {np.exp(mu * T):12.3f} {np.median(s):9.3f}"
          f" {np.exp((mu - 0.5 * sig ** 2) * T):18.3f} {(s < np.exp(mu * T)).mean():22.4f}")
# =>   terminal wealth of GBM with mu 0.08 and sigma 0.2, 1000000 paths, S0 = 1
#            T    mean    exp(mu T)    median    exp((mu-s2/2)T)    frac below the mean
#              1   1.083        1.083     1.062              1.062                 0.5401
#              5   1.493        1.492     1.350              1.350                 0.5883
#             10   2.225        2.226     1.822              1.822                 0.6241
#             30  11.014       11.023     6.044              6.050                 0.7087
```

Both closed forms are reproduced to three decimals at every horizon — the simulated mean sits on $e^{\mu T}$ and the simulated median on $e^{(\mu-\sigma^{2}/2)T}$ — so nothing here is an approximation. At one year the two are $1.083$ and $1.062$ and the distinction looks pedantic. At thirty years they are $11.014$ and $6.044$: **the average outcome is $82\%$ larger than the typical one, on a process with an $8\%$ drift and an entirely ordinary $20\%$ volatility.**

The last column is who receives which. The fraction of paths finishing below the mean rises $0.5401$, $0.5883$, $0.6241$, $0.7087$, so after thirty years **seven investors in ten do worse than the average, and the average is being carried by the tail.** This is the terminal-wealth arithmetic that [Lognormal Distribution](../part-05-common-distributions/12-lognormal-distribution.md) calls the question of "who gets the average", and the answer is that almost nobody does.

The practical consequence is a translation rule between two vocabularies that are constantly mixed. An expected-return input to an optimizer, a risk premium in a capital-market assumption, and the $\mu$ in this equation are all arithmetic means and describe the first column. A compound annual growth rate, a realized track record, and the experience of a single investor who cannot diversify across parallel universes are all in the second. Feeding a thirty-year arithmetic mean into a plan that will be judged on realized compounding overstates by $\sigma^{2}/2$ per year, which at $20\%$ volatility is two full percentage points annually.

## The Drift Is Not Estimable at Any Sampling Frequency

Now the estimation problem, which is where the model stops being a convenience and starts dictating what a business can do. Over a fixed horizon $T$, the maximum-likelihood estimate of the log-drift $\nu=\mu-\sigma^{2}/2$ is

$$\hat\nu=\frac{\log S_T-\log S_0}{T},$$

which depends on the first and last observation and on nothing in between. Sample the path a million times inside $[0,T]$ and $\hat\nu$ does not move.

??? note "Proof that the drift's standard error depends only on the horizon and the volatility's only on the count"
    Under the model, $\log S_T-\log S_0=\nu T+\sigma W_T$ with $W_T\sim\mathcal{N}(0,T)$, so

    $$\hat\nu=\nu+\frac{\sigma W_T}{T},\qquad \operatorname{sd}(\hat\nu)=\frac{\sigma\sqrt T}{T}=\frac{\sigma}{\sqrt T}.$$

    The intermediate observations are absent from the expression, so the estimator is exactly as precise on monthly data as on tick data. Its error shrinks only in *calendar time*, at rate $1/\sqrt T$, and no amount of resolution substitutes.

    The volatility runs the other way. With $n$ observations at spacing $dt=T/n$, the realized variance $\hat\sigma^{2}=\frac1T\sum_i(\Delta\log S_i)^{2}$ is a scaled chi-square with $n$ degrees of freedom, so $\operatorname{sd}(\hat\sigma)\approx\sigma/\sqrt{2n}$: the error shrinks in the *number of observations*, without any reference to the calendar. Letting $n\to\infty$ at fixed $T$ sends it to zero, which is the [quadratic variation](08-brownian-motion.md) result — the volatility is a path property, measurable in principle from a single instant of a continuously observed trajectory.

    The load-bearing distinction is that the drift multiplies $t$ and the diffusion multiplies $\sqrt t$, so refining the grid shrinks the noise faster than the signal for the second parameter and at exactly the same rate for the first. **One parameter lives in the increments and the other lives only in the endpoints**, and no estimator, no filter and no amount of computation moves information across that boundary.

```python
import numpy as np

rng = np.random.default_rng(8093)
reps, mu, sig, T = 40_000, 0.08, 0.20, 1.0
print(f"  one year of data, {reps} histories, sampled at different frequencies")
print("      bars per year    se(drift-hat)    sigma/sqrt(T)    se(vol-hat)    sigma/sqrt(2n)")
for n in (12, 252, 6_300, 98_280):
    dt, s1, s2, left = T / n, np.zeros(reps), np.zeros(reps), n
    while left:
        k = min(left, 1_000)
        d = (mu - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * rng.standard_normal((reps, k))
        s1 += d.sum(1)
        s2 += (d * d).sum(1)
        left -= k
    nu = s1 / T                                                # the log-drift estimate
    vol = np.sqrt(s2 / T)                                      # realized volatility
    print(f"  {n:18d} {nu.std(ddof=1):16.5f} {sig / np.sqrt(T):16.5f}"
          f" {vol.std(ddof=1):14.5f} {sig / np.sqrt(2 * n):17.5f}")
# =>   one year of data, 40000 histories, sampled at different frequencies
#          bars per year    se(drift-hat)    sigma/sqrt(T)    se(vol-hat)    sigma/sqrt(2n)
#                      12          0.19970          0.20000        0.04056           0.04082
#                     252          0.20006          0.20000        0.00889           0.00891
#                    6300          0.19999          0.20000        0.00178           0.00178
#                   98280          0.20003          0.20000        0.00045           0.00045
```

The two middle columns are the result and they do not move. Going from monthly bars to minute bars is an $8{,}190$-fold increase in data, and the standard error of the drift estimate reads $0.19970$, $0.20006$, $0.19999$, $0.20003$ — the same number, equal to $\sigma/\sqrt T=0.20000$, at every frequency. **Eight thousand times more data buys exactly nothing about the expected return.**

The last two columns are the same experiment for the volatility and they collapse: $0.04056$, $0.00889$, $0.00178$, $0.00045$, tracking $\sigma/\sqrt{2n}$ to three significant figures throughout. On minute bars the volatility of a $20\%$ process is known to within $0.045$ percentage points after a single year.

Put the two together and the ratio is the point. On monthly data the drift is known to $\pm0.20$ against a true value of $0.06$ and the volatility to $\pm0.041$ against $0.20$ — bad and adequate respectively. On minute data the volatility is known four hundred times better and the drift is known exactly as badly as before. The gap does not narrow with effort, technology, or data spend; it widens.

## That Asymmetry Is the Shape of the Whole Business

The fifty-eight-fold gap [Expected Value](../part-04-expectation-and-moments/01-expected-value.md) measures is therefore not a sample-size accident to be fixed with a longer history. It is the ratio $\sqrt{2n}$ against $\sqrt T$, and the two denominators count different things. Twenty-five years of daily data is $T=25$ and $n=6{,}300$: the drift's error is $\sigma/5$ and the volatility's is $\sigma/112$, a factor of twenty-two before any of the finite-sample details, and the published fifty-eight includes the estimate being a mean rather than a log-drift.

Almost every structural feature of quantitative investing follows from that one inequality. Risk models are elaborate and return models are crude, because the data supports one and not the other. Volatility targeting is implementable and return timing is not. The Sharpe ratio's standard error, which [The Delta Method](../part-07-asymptotic-theory/04-delta-method.md) derives and finds to be dominated by the mean's contribution, is really this asymmetry in a ratio. Portfolio optimizers are notoriously unstable in their expected-return inputs and merely sensitive in their covariance inputs. And the honest reading of any "expected return" number in a model — a capital-market assumption, an alpha forecast, a drift in a simulation — is that it is a prior wearing an estimate's clothes, since the data available to inform it has a standard error the size of the quantity itself.

!!! warning "High-frequency data is an answer to the volatility question and to no part of the return question, and the two are routinely bundled in the same procurement decision"
    The reflex that more granular data must improve every estimate is wrong in a specific and expensive way. Tick data genuinely transforms what can be known about variance, covariance, liquidity, microstructure and execution quality — every one of which lives in the increments. It contributes nothing whatever to the estimate of an expected return over a fixed horizon, because that quantity depends on the endpoints and the endpoints do not change when the sampling does. The corollary for research is that a strategy whose edge is a drift claim cannot be validated faster by sampling faster; only more calendar time helps, and [The Weak Law of Large Numbers](../part-07-asymptotic-theory/01-weak-law-of-large-numbers.md) prices what that costs when it reports an equity-premium estimate that after twenty-five years still carries a meaningful chance of the wrong sign.

## Leverage Raises the Mean Monotonically and the Median Only to a Point

Scaling exposure by a constant $L$ turns the parameters into $(L\mu,L\sigma)$, so the log-drift becomes $L\mu-\tfrac12L^{2}\sigma^{2}$ — linear in the numerator and quadratic in the penalty. That expression is maximized at $L^{*}=\mu/\sigma^{2}$ and returns to zero at $2L^{*}$, while the arithmetic mean $e^{L\mu t}$ increases in $L$ without limit.

```python
import numpy as np

rng = np.random.default_rng(8097)
reps, T, mu, sig = 500_000, 30.0, 0.08, 0.20
print(f"  {T:.0f} years of leveraged GBM, mu {mu}, sigma {sig}, optimal leverage "
      f"{mu / sig ** 2:.1f}")
print("      leverage    growth rate    mean wealth    median wealth    P(end below 1)")
for L in (1.0, 2.0, 3.0, 4.0):
    g = L * mu - 0.5 * L ** 2 * sig ** 2
    w = np.exp(g * T + L * sig * np.sqrt(T) * rng.standard_normal(reps))
    print(f"  {L:14.1f} {g:14.4f} {w.mean():14.1f} {np.median(w):16.3f}"
          f" {(w < 1.0).mean():16.4f}")
# =>   30 years of leveraged GBM, mu 0.08, sigma 0.2, optimal leverage 2.0
#          leverage    growth rate    mean wealth    median wealth    P(end below 1)
#                 1.0         0.0600           11.1            6.057           0.0497
#                 2.0         0.0800          120.8           11.027           0.1370
#                 3.0         0.0600         1331.6            6.037           0.2922
#                 4.0        -0.0000        21814.8            1.005           0.4995
```

The mean column rises without hesitation: $11.1$, $120.8$, $1{,}331.6$, $21{,}814.8$. By the criterion of expected terminal wealth, more leverage is always better and there is no interior optimum — the quadratic penalty never appears, because the arithmetic mean of a lognormal is blind to it.

The median column turns over. It reads $6.057$ at unit leverage, peaks at $11.027$ at $L^{*}=2$, falls back to $6.037$ at $L=3$, and lands at $1.005$ at $L=4$. **Three times leverage delivers exactly the growth of one times leverage** — $0.0600$ in both rows — with three times the volatility and a probability of ending below where you started that has risen from $0.0497$ to $0.2922$. And at four times leverage, after thirty years, the median investor has $1.005$: they have broken even, while the mean of the same distribution is $21{,}815$.

!!! note "The growth curve is a downward parabola in leverage, so the penalty for overshooting is not symmetric with the reward for approaching"
    Writing the growth rate as $g(L)=L\mu-\tfrac12L^{2}\sigma^{2}=\tfrac12\sigma^{2}\bigl(L^{*2}-(L-L^{*})^{2}\bigr)$ makes the shape explicit: a parabola peaking at $L^{*}$, hitting zero at $0$ and at $2L^{*}$, and negative beyond. Undershooting by a factor of two costs a quarter of the maximum growth — the $L=1$ row gives $0.0600$ against the peak's $0.0800$, which is a bad outcome nobody notices. Overshooting by a factor of two costs *all* of it, and beyond $2L^{*}$ the growth rate is negative, so the position compounds toward zero with probability one while its expected value continues to rise. Since $L^{*}=\mu/\sigma^{2}$ is estimated with the third section's standard error, the estimate's own uncertainty routinely spans the interval from half-Kelly to twice-Kelly, and the asymmetry of the parabola is the entire argument for deliberately sizing below the computed optimum. [Position Sizing and Risk Budgeting](../../part-04-strategy-development/06-position-sizing-and-risk-budgeting.md) is where that argument becomes a rule.

That last row is the sharpest available statement of the ergodicity problem [Random Processes](01-random-processes.md) opens the part with. The ensemble average grows spectacularly and the time average does not grow at all, because the two are averages over different things — across parallel universes in the first case and along one path in the second — and a multiplicative process makes them diverge. Nobody receives the ensemble average. The quantity to maximize is the growth-rate column, its optimum is at $\mu/\sigma^{2}$, and the reason no practitioner runs there is that $\mu$ is the parameter the third section just showed to be unknowable: $L^{*}$ inherits the full standard error of the drift, so a leverage target computed from an estimated $\mu$ is a number with a standard error the size of itself, and the penalty for overshooting is the fourth row.

## One Equation, and the Two Parameters Are Not Comparable Objects

Geometric Brownian motion is the smallest model that respects the two things everyone knows about prices — they are positive and they compound — and both of its surprises follow from the compounding rather than from any subtlety in the randomness. The log-drift sits $\sigma^{2}/2$ below the arithmetic drift, so the average outcome and the typical outcome separate exponentially and seven investors in ten fall below the average after thirty years. That is not a fee and nothing was lost; it is the difference between two questions that use the same word.

The deeper asymmetry is between the parameters. The volatility lives in the increments and is estimable to arbitrary precision by sampling faster; the drift lives in the endpoints and is estimable only by waiting longer, at $\sigma/\sqrt T$, forever. Every attempt to be clever about this fails for the same structural reason, and the fifty-eight-fold gap the appendix has been citing since Part IV is that inequality rather than an artifact of one data set.

What the two facts do together is decide where effort belongs. Risk is measurable, so measure it; return is not, so treat every expected-return input as an assumption to be stress-tested rather than an estimate to be refined. The leverage arithmetic shows the cost of confusing the two: the optimal exposure is $\mu/\sigma^{2}$, its numerator is the unknowable parameter and its denominator the knowable one, and the penalty function around the optimum is asymmetric — at $2L^{*}$ thirty years of a good process returns exactly nothing. The formal machinery for the "fair game" structure underlying all of this, and for what a stopping rule does to it, is [Martingales](10-martingales.md).
