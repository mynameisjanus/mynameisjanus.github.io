# Variance Reduction

The previous two pages bought accuracy by changing the distribution being sampled. This one buys it without changing anything, by noticing that the estimator throws away structure it already has: a symmetry that lets one draw cancel another's noise, a related quantity whose exact answer is known, a comparison in which two runs could have shared their randomness. Each technique is a few lines, each composes with the others, and each has a variance factor computable in advance rather than discovered afterwards. What makes them worth a page of their own is that all three are widely believed to be free improvements, and two of the three can make a simulation strictly worse — quietly, with no error and no warning, in cases that are not exotic.

This page covers the efficiency criterion that decides whether a technique is worth using at all, antithetic pairing and the monotonicity it silently requires, control variates as a regression whose coefficient has to be estimated, common random numbers and the difference between pricing a level and pricing a comparison, and stratification as the device that removes between-stratum variance outright. It does not change the law being sampled, which are [Importance Sampling](04-importance-sampling.md) and [Rejection Sampling](05-rejection-sampling.md); it does not build the estimator or derive its rate, which is [Monte Carlo Simulation](03-monte-carlo-simulation.md); it constructs no low-discrepancy sequence and proves no quasi-Monte Carlo bound; it resamples no data, which is [Bootstrap Methods](07-bootstrap-methods.md); it decomposes no estimator into bias and variance, which is [Part XIV](../part-14-model-selection/index.md); and it prices no instrument for its own sake, which is [Options Pricing](../../advanced/11-options-pricing.md).

The trading stake is a section heading the course gives its own paragraph. [Reinforcement Learning for Execution](../../advanced/06-rl-for-execution.md) reports that evaluating every policy "on the same 20,000 random seeds" cut the standard error of the comparison "from 1.24 to 0.25 basis points — a **4.9-fold reduction**, equivalent to running twenty-four times as many episodes", and concludes that "any execution study that reports unpaired means is discarding most of its statistical power". The fourth section reproduces that arithmetic from scratch and shows exactly which term in the variance the pairing removes, along with the thing pairing costs that the lesson does not mention.

## The Only Currency Is Variance Multiplied by Work

A technique that halves the variance at triple the cost per draw is a loss, and the comparison that settles it is standard. If two estimators of the same quantity have variances $\sigma_1^{2},\sigma_2^{2}$ per replication and costs $c_1,c_2$ per replication, then at a fixed budget the first is preferable exactly when $\sigma_1^{2}c_1<\sigma_2^{2}c_2$. The product $\sigma^{2}c$ is the only figure of merit, and everything on this page is an attempt to move it.

Two accounting rules follow and both are routinely violated. The first is that comparisons must be made at equal work rather than at equal draw counts — an antithetic scheme using $N$ uniforms evaluates the payoff $2N$ times, so it must be compared against a crude scheme with $2N$ payoff evaluations and not $N$. The second is that the exchange rate between error and work is quadratic, inherited from the $N^{-1/2}$ rate: a technique that cuts the standard error by a factor $\kappa$ has done the work of $\kappa^{2}$ times as many paths. That is why the course's $4.9$-fold reduction in standard error is reported as twenty-four times the episodes, and why a $20\%$ improvement in standard error — which sounds negligible — is worth $44\%$ more compute.

## Antithetic Pairing Helps a Monotone Payoff and Hurts a Symmetric One

Every draw in a simulation is ultimately a uniform, and a uniform has a free symmetry: if $U\sim\mathrm{Unif}(0,1)$ then so does $1-U$. **Antithetic variates** exploit it by evaluating the integrand at both and averaging the pair,

$$\hat\theta_{\text{anti}}=\frac1N\sum_{i=1}^{N}\frac{f(U_i)+f(1-U_i)}{2},$$

which is unbiased because each term is. The variance of a pair average is $\tfrac12\mathrm{var}(f)\,(1+\rho)$ with $\rho=\mathrm{corr}\big(f(U),f(1-U)\big)$, so at equal payoff evaluations the whole technique reduces to the sign and size of one correlation.

??? note "Proof that the variance ratio is exactly $1+\rho$, that $\rho\leq0$ when $f$ is monotone, and that nothing prevents $\rho$ from being positive otherwise"
    Write $A=f(U)$, $B=f(1-U)$, both with variance $\sigma^{2}$ since $U$ and $1-U$ have the same law. Then

    $$\mathrm{var}\!\left(\frac{A+B}{2}\right)=\frac{\sigma^{2}+\sigma^{2}+2\rho\sigma^{2}}{4}=\frac{\sigma^{2}(1+\rho)}{2}.$$

    A crude estimator using the same $2N$ evaluations has variance $\sigma^{2}/(2N)$; the antithetic estimator averages $N$ independent pairs and has variance $\sigma^{2}(1+\rho)/(2N)$. The ratio is $1+\rho$ exactly, with no approximation.

    If $f$ is non-decreasing then $g(u)=f(u)$ is non-decreasing and $h(u)=f(1-u)$ is non-increasing, and Chebyshev's association inequality — that oppositely monotone functions of the same random variable have non-positive covariance — gives $\rho\leq0$. Hence the ratio is at most one and the technique cannot hurt. The proof of the inequality is one line: for independent copies $U,U'$, the product $\big(g(U)-g(U')\big)\big(h(U)-h(U')\big)$ is pointwise non-positive, and taking expectations gives $-2\,\mathrm{cov}(g,h)\leq0$.

    The load-bearing hypothesis is monotonicity, and it is a hypothesis about the composed map from the *uniform* to the payoff, not about the payoff as a function of price. A butterfly is a non-monotone function of the terminal price, so $f(U)$ and $f(1-U)$ are both large when $U$ is near $\tfrac12$ and both zero when $U$ is near either end, giving $\rho>0$ and a variance ratio above one. **The standard trick is not a free improvement; it is a bet on the shape of the integrand, and the bet can be lost.**

```python
import numpy as np
from scipy.special import ndtri

rng = np.random.default_rng(9061)
s0, r, sigma, T, m = 100.0, 0.03, 0.20, 1.0, 4_000_000
u = rng.random(m)


def terminal(uu):
    return s0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * ndtri(uu))


def call(s, k):
    return np.exp(-r * T) * np.maximum(s - k, 0.0)


payoffs = {
    "call, K = 105": lambda s: call(s, 105),
    "put, K = 95": lambda s: np.exp(-r * T) * np.maximum(95 - s, 0.0),
    "straddle, K = 100": lambda s: np.exp(-r * T) * np.abs(s - 100),
    "butterfly, 90/100/110": lambda s: call(s, 90) - 2 * call(s, 100) + call(s, 110),
}
lo, hi = terminal(u), terminal(1 - u)
print(f"  antithetic pairing on {m} uniforms, four payoffs on the same underlying")
print("   payoff                    price    corr(f(U), f(1-U))    variance ratio    paths saved")
for name, f in payoffs.items():
    a, b = f(lo), f(hi)
    rho = np.corrcoef(a, b)[0, 1]
    ratio = ((a + b) / 2).var() / (a.var() / 2)                # equal payoff evaluations
    print(f"  {name:<24} {a.mean():7.4f} {rho:21.4f} {ratio:17.4f} {1 / ratio:14.2f}")
# =>   antithetic pairing on 4000000 uniforms, four payoffs on the same underlying
#       payoff                    price    corr(f(U), f(1-U))    variance ratio    paths saved
#      call, K = 105             7.1288               -0.3236            0.6774           1.48
#      put, K = 95               4.3772               -0.3306            0.6687           1.50
#      straddle, K = 100        15.8779                0.7973            1.7992           0.56
#      butterfly, 90/100/110     1.8947                0.9007            1.9005           0.53
```

The first two rows are the case the technique was designed for and the gain is smaller than its reputation. A call is a monotone function of the uniform, the correlation is $-0.3236$, the variance ratio is $0.6774$ — matching $1+\rho$ to four decimals — and the effective saving is $1.48$ times the paths. Useful, free, and not the factor of two that the phrase "cancel the noise" suggests.

The reason the correlation is not closer to $-1$ is worth naming because it applies to every option. A call struck at $105$ pays nothing until the uniform exceeds about $0.6$, so $f(1-U)$ is nothing until $U$ falls below about $0.4$, and in the band between them **both members of every pair are exactly zero**. Antithetic pairing cancels nothing there, and the truncation at zero that defines an option payoff is precisely what caps the achievable correlation.

The last two rows are the failure. A straddle is a V in the terminal price and a butterfly is a tent, so in both cases $f(U)$ and $f(1-U)$ move together: the correlations are $+0.7973$ and $+0.9007$, and the variance ratios are $1.7992$ and $1.9005$. **Applying the standard variance-reduction technique to a butterfly nearly doubles the variance, so a run that used it and reported its own standard error would have paid the same compute for an error bar $38\%$ wider, and nothing in the output would say so.** The check costs one line — compute the correlation of the pair and confirm it is negative before believing the technique helped.

## A Control Variate Is a Regression, and the Coefficient Must Be Estimated

The second technique needs a second quantity $g$, correlated with $f$, whose expectation $\mu_g$ is known exactly. Then for any constant $\beta$,

$$\hat\theta_{\text{cv}}=\bar f-\beta\big(\bar g-\mu_g\big)$$

is unbiased, because the subtracted term has mean zero. The known answer is used as a ruler: whenever this particular sample of draws happened to overshoot on $g$, it probably overshot on $f$ too, and the correction removes the shared part.

??? note "Proof that the optimal coefficient is a regression slope and the variance factor is exactly $1-\rho^{2}$"
    Expanding,

    $$\mathrm{var}(\hat\theta_{\text{cv}})=\frac1N\Big(\mathrm{var}(f)-2\beta\,\mathrm{cov}(f,g)+\beta^{2}\mathrm{var}(g)\Big),$$

    a quadratic in $\beta$ minimized at

    $$\beta^{\ast}=\frac{\mathrm{cov}(f,g)}{\mathrm{var}(g)},$$

    which is exactly the ordinary least-squares slope of $f$ on $g$. Substituting back,

    $$\mathrm{var}(\hat\theta_{\text{cv}})=\frac{\mathrm{var}(f)}{N}\left(1-\rho^{2}\right),\qquad \rho=\mathrm{corr}(f,g),$$

    so the variance factor is $1-\rho^{2}$ and the method is a regression of the estimator on a variable whose mean is known. The naive choice $\beta=1$ gives factor $1-2\rho\sigma_g/\sigma_f+\sigma_g^{2}/\sigma_f^{2}$, which is optimal only when $\sigma_f=\rho\,\sigma_g$ and is strictly worse otherwise.

    The load-bearing quantity is $\rho^{2}$ and it is brutally nonlinear. A correlation of $0.9$ removes $81\%$ of the variance, which is a factor of five in paths; a correlation of $0.99$ removes $98\%$, a factor of fifty; a correlation of $0.999$ removes $99.9\%$, a factor of a thousand. **The whole method lives in the last two decimal places of the correlation**, which is why the search for a control variate should be a search for something almost identical to the target rather than something merely related to it. Estimating $\beta$ from the same draws introduces a bias of order $1/N$, negligible at any usable sample size and removable entirely by estimating $\beta$ on a pilot run.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(9063)
s0, k, r, sigma, T, steps, n = 100.0, 100.0, 0.03, 0.20, 1.0, 12, 2_000_000
t = np.arange(1, steps + 1) * T / steps
mu_g = np.log(s0) + (r - 0.5 * sigma ** 2) * t.mean()          # log of the geometric average
var_g = sigma ** 2 * np.minimum.outer(t, t).sum() / steps ** 2
d1 = (mu_g - np.log(k) + var_g) / np.sqrt(var_g)
geo_exact = np.exp(-r * T) * (np.exp(mu_g + 0.5 * var_g) * norm.cdf(d1)
                              - k * norm.cdf(d1 - np.sqrt(var_g)))

dw = rng.standard_normal((n, steps)) * np.sqrt(T / steps)
logs = np.log(s0) + np.cumsum((r - 0.5 * sigma ** 2) * T / steps + sigma * dw, axis=1)
arith = np.exp(-r * T) * np.maximum(np.exp(logs).mean(axis=1) - k, 0.0)
geo = np.exp(-r * T) * np.maximum(np.exp(logs.mean(axis=1)) - k, 0.0)

rho = np.corrcoef(arith, geo)[0, 1]
beta = np.cov(arith, geo)[0, 1] / geo.var(ddof=1)
print(f"  arithmetic Asian controlled by the geometric Asian, {n} paths, {steps} fixings")
print(f"  geometric Asian in closed form {geo_exact:.6f}, simulated {geo.mean():.6f}"
      f", correlation {rho:.6f}")
print("   estimator                  price    standard error    variance ratio    paths saved")
for name, est in (("crude", arith),
                  ("control, beta = 1", arith - (geo - geo_exact)),
                  (f"control, beta = {beta:.4f}", arith - beta * (geo - geo_exact))):
    ratio = est.var(ddof=1) / arith.var(ddof=1)
    print(f"  {name:<24} {est.mean():9.6f} {est.std(ddof=1) / np.sqrt(n):17.6f}"
          f" {ratio:17.6f} {1 / ratio:14.1f}")
# =>   arithmetic Asian controlled by the geometric Asian, 2000000 paths, 12 fixings
#      geometric Asian in closed form 5.435333, simulated 5.428803, correlation 0.999628
#       estimator                  price    standard error    variance ratio    paths saved
#      crude                     5.624835          0.005820          1.000000            1.0
#      control, beta = 1         5.631365          0.000235          0.001629          614.1
#      control, beta = 1.0307    5.631565          0.000159          0.000743         1345.3
```

The setup is the canonical one because it is the case where a perfect control exists. An arithmetic-average Asian option has no closed form; the *geometric*-average version does, because the geometric mean of lognormals is lognormal. The two payoffs are computed on the same paths and differ only by the gap between an arithmetic and a geometric mean of twelve numbers, which is small — the measured correlation is $0.999628$.

The consequence is the largest number in this part. The crude estimator prices the option at $5.624835\pm0.005820$. The optimally-controlled estimator prices it at $5.631565\pm0.000159$, a variance ratio of $0.000743$ and a saving of **$1{,}345$ times the paths** — the two million paths here do the work of two and a half billion. And $1-\rho^{2}=1-0.999628^{2}=7.44\times10^{-4}$ predicts the measured $7.43\times10^{-4}$ to three significant figures, so the proof is not an approximation being checked but a formula being read off.

The middle row prices the shortcut. Taking $\beta=1$ — the choice everyone makes first, on the reasoning that the two payoffs are nearly the same thing — recovers $614$ of the available $1{,}345$, so **less than half the benefit survives skipping a one-line regression**. The optimal $\beta$ is $1.0307$, barely different from one, and that three-percent difference is worth a factor of two in compute.

!!! note "The control variate is the only technique here that also detects a bug, because it computes a quantity whose answer is already known"
    Running the geometric payoff alongside the arithmetic one produces a number that can be checked: the simulated $5.428803$ against the closed-form $5.435333$, agreeing to within $1.2$ standard errors of the geometric estimator's own noise. That agreement is a test of the path generator, the discounting, the fixing dates and the payoff code all at once, and it is free because the control has to be computed anyway. Nothing else in this part offers it — an antithetic pair, a shared seed and a stratified allocation all produce numbers whose correctness is exactly as unverifiable as the crude estimate's. The practice worth adopting is to keep the control's own comparison in the output even after the variance reduction is working, since it is the one line that will notice when someone changes the discretization.

## Common Random Numbers Price a Difference, Not Two Levels

Most simulation questions are comparative: is this schedule cheaper than that one, does this parameter beat that one, is the new model better than the incumbent. The quantity of interest is a difference, and the variance of a difference is

$$\mathrm{var}(A-B)=\mathrm{var}(A)+\mathrm{var}(B)-2\,\mathrm{cov}(A,B),$$

so anything that raises the covariance shrinks the answer's error bar. **Common random numbers** raise it for free by evaluating both alternatives on the identical stream of draws, so that whatever the market happened to do is a shared input rather than an independent nuisance.

```python
import numpy as np

rng = np.random.default_rng(9067)
eps, slices, vol, lam = 20_000, 50, 12.0, 900.0                # daily vol and impact, in bp
front = np.exp(-0.06 * np.arange(slices))
front /= front.sum()                                           # a front-loaded schedule
flat = np.full(slices, 1.0 / slices)                           # and the uniform one


def cost(sched, m):                                            # shortfall against the arrival price
    price = np.cumsum(rng.standard_normal((m, slices)) * vol / np.sqrt(slices), axis=1)
    return price @ sched + lam * (sched ** 2).sum()


a_i, b_i = cost(front, eps), cost(flat, eps)                   # independent episodes
state = rng.bit_generator.state
a_p = cost(front, eps)
rng.bit_generator.state = state                                # replay the identical paths
b_p = cost(flat, eps)

print(f"  two execution schedules over {slices} slices, {eps} episodes, costs in basis points")
print("   scheme         mean front    mean flat    difference    se of difference"
      "    corr    episodes for parity")
se_i = (a_i - b_i).std(ddof=1) / np.sqrt(eps)
for name, a, b in (("independent", a_i, b_i), ("common paths", a_p, b_p)):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(eps)
    print(f"  {name:<14} {a.mean():11.3f} {b.mean():12.3f} {d.mean():13.4f} {se:19.4f}"
          f" {np.corrcoef(a, b)[0, 1]:7.4f} {eps * (se_i / se) ** 2:22.0f}")
va, vb, cab = a_p.var(ddof=1), b_p.var(ddof=1), np.cov(a_p, b_p)[0, 1]
print(f"  variance of the paired difference: {va:.2f} + {vb:.2f} - 2 x {cab:.2f}"
      f" = {va + vb - 2 * cab:.2f}")
# =>   two execution schedules over 50 slices, 20000 episodes, costs in basis points
#       scheme         mean front    mean flat    difference    se of difference    corr    episodes for parity
#      independent         29.803       18.107       11.6956              0.0604 -0.0081                  20000
#      common paths        29.803       17.968       11.8345              0.0206  0.9475                 172017
#      variance of the paired difference: 23.29 + 49.62 - 2 x 32.20 = 8.49
```

Both schemes answer the same question and agree on it: the front-loaded schedule costs about $11.7$ to $11.8$ basis points more than the uniform one, which is the impact penalty for trading faster. The independent scheme measures that with a standard error of $0.0604$, the paired scheme with $0.0206$ — a factor of $2.93$ in error and, squared, a factor of $8.6$ in work. Matching the paired precision without pairing would take $172{,}017$ episodes instead of $20{,}000$.

The last line shows where the saving comes from and is the whole mechanism in one arithmetic statement. The two paired cost series have variances $23.29$ and $49.62$, and a covariance of $32.20$; the difference therefore has variance $23.29+49.62-2\times32.20=8.49$ instead of the $72.91$ the sum of the variances alone would give. The correlation between the two schedules' costs is $0.9475$ under pairing and $-0.0081$ without it — **the same market noise passes through both schedules, and pairing lets it cancel rather than accumulate.** This is the arithmetic behind the course's report of a $4.9$-fold reduction over twenty thousand shared seeds, which by the quadratic exchange rate is the twenty-four-fold saving in episodes it quotes.

!!! warning "Pairing makes the difference precise and the two levels dependent, so any test applied to the levels afterwards is invalid"
    Common random numbers deliver a sharper comparison by manufacturing a correlation of $0.9475$ between two quantities that would otherwise be independent, and that correlation does not disappear when the analysis moves on. Three consequences follow and all three are seen in practice. A two-sample $t$-test on the paired level estimates is wrong, because it assumes the independence pairing destroyed; the paired test on the differences is the correct instrument and it is a different formula. A confidence interval for $A$ and a confidence interval for $B$ computed from paired runs may not be combined into an interval for $A-B$, since the covariance term is missing. And a sweep over many alternatives evaluated on one shared set of paths has correlated errors across the whole sweep, so the maximum over alternatives is *not* the maximum of independent draws — which is the correlated-family effect [Monte Carlo Simulation](03-monte-carlo-simulation.md) measures, where a common factor lowers the expected best and raises its dispersion. Pairing is close to free and it is not free: it buys precision in the differences and spends independence everywhere else.

## Stratification Removes the Variance Between Strata Outright

The last technique attacks the variance decomposition directly. Partition the sample space into strata $S_1,\dots,S_K$ with known probabilities $p_k$, and allocate $n_k$ draws to stratum $k$ rather than letting the sample sizes fall where chance puts them. The law of total variance splits $\mathrm{var}(f)$ into a within-stratum part and a between-stratum part, and proportional allocation $n_k=p_kN$ eliminates the second entirely: the stratified estimator has variance $\sum_k p_k\sigma_k^{2}/N$ where $\sigma_k^{2}$ is the variance inside stratum $k$, against $\big(\sum_k p_k\sigma_k^{2}+\sum_k p_k(\mu_k-\theta)^{2}\big)/N$ for the crude one. The gain is exactly the dispersion of the stratum means, so stratification pays when the strata differ and does nothing when they do not — and unlike antithetic pairing, it can never hurt under proportional allocation.

Neyman allocation goes further by putting draws where the variance is rather than where the probability is, setting $n_k\propto p_k\sigma_k$, which is optimal and requires knowing the $\sigma_k$ — usually estimated from a pilot run, which reintroduces a small bias for a large gain. The relationship to the previous page is worth naming: allocating disproportionately many draws to a low-probability, high-variance stratum and reweighting is importance sampling with a piecewise-constant proposal, so the two techniques are the same idea at different resolutions.

In one dimension the practical version is trivial and underused: instead of $N$ uniforms, use $\big((k-1)+U_k\big)/N$ for $k=1,\dots,N$, which places exactly one draw in each of $N$ equal bins. For a smooth integrand this converts the $N^{-1/2}$ rate into something closer to $N^{-1}$, at no cost. In high dimensions the same idea stratifies each coordinate independently — the Latin hypercube — and the gain survives only for the part of the integrand that is additive across coordinates, which is a real limitation and is also, empirically, most of the integrand in a great many pricing problems.

## Every Technique Here Spends a Quantity You Already Knew

The four devices are one idea in four costumes. Antithetic pairing spends a symmetry of the uniform. A control variate spends a closed form for a neighbouring problem. Common random numbers spend the fact that a comparison does not need two independent noise realizations. Stratification spends known stratum probabilities. In each case the resource was already in hand and unused, which is why the savings are large and the code is short — and why none of them helps when the integrand is genuinely unstructured.

The numbers separate them by two orders of magnitude, and the ordering is not the one folklore suggests. Antithetic pairing, the technique everyone learns first, bought a factor of $1.48$ on a call and *cost* a factor of $1.9$ on a butterfly. Common random numbers bought a factor of $8.6$. A control variate with a correlation of $0.9996$ bought a factor of $1{,}345$. The lesson is in the $1-\rho^{2}$: effort spent finding a closely related problem with a known answer dominates effort spent on any generic trick, and the payoff is quadratic in how close the relation is.

The failure mode is also different from anything else in this part, and it is the reason two of these sections carry warnings rather than proofs of correctness. Nothing here can make an estimate wrong — every estimator on this page is unbiased, and the antithetic butterfly and the unpaired comparison both converge to the right answer. What they produce instead is an advertised saving that was never realized, or a precision that is real in one quantity and destroyed in another. The diagnostic is uniform across all four: compute the correlation the technique depends on, from the same draws, and print it beside the result. It is one line, it is the quantity every proof on this page turns on, and it is almost never in the output.

That leaves one question the whole part has deferred. Every technique so far has assumed a law to sample from — specified, parameterized, and known. Real evaluation problems arrive with a few thousand rows of history and no law at all, and the resampling schemes that handle them have a different licence, a different failure mode, and a promise that the course has already made on their behalf. That is [Bootstrap Methods](07-bootstrap-methods.md).
