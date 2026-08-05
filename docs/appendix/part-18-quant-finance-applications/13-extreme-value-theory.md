# Extreme Value Theory

[Functions of Random Variables](../part-03-random-variables/08-functions-of-random-variables.md) promises this page by name: "sums of many variables converge to a Gaussian; maxima of many variables converge to one of three quite different shapes." The trichotomy is the subject, and its practical content is that a single parameter $\xi$ decides everything about a tail — positive means a power law and a finite number of moments, zero means exponential-like decay, negative means a hard upper bound — and that this parameter is estimable from exceedances without ever committing to a distribution for the body. The limit arrives at very different speeds, and the slowest is the one finance uses most: normalized maxima of Gaussians sit at a Kolmogorov–Smirnov distance of $0.0168$ from their limit at $n=1{,}000$ against $0.0011$ for a Pareto, and a shape fitted to them reads $-0.0688$ where the truth is $0$. Choosing where the tail starts is a bias–variance dial whose optimum moves with the sample: the best threshold on $t(4)$ data is the $90.0$th percentile at two thousand observations, the $95.0$th at twenty thousand and the $99.0$th at two hundred thousand. And the parameter that decides everything is often not determined at all — on Gaussian data at a thousand observations and a $99\%$ threshold the fitted shape is $-0.8192$ with a standard deviation of $0.9032$, an interval running from a hard bound to infinite variance.

This page covers the Fisher–Tippett–Gnedenko trichotomy and the max-stability that forces it, the three domains of attraction and the rates at which they are reached, the Pickands–Balkema–de Haan theorem that turns the block-maximum result into a threshold model, threshold selection as a bias–variance problem, and what a fitted shape parameter is worth when its standard error spans zero. It does not define regular variation, prove max-sum equivalence, or develop the Hill estimator, all of which are [Heavy-Tailed Returns](12-heavy-tailed-returns.md); it does not catalogue which moments exist, which is [Higher-Order Moments](../part-04-expectation-and-moments/03-higher-order-moments.md); it does not derive the Weibull's min-stability or the weakest-link reading, which is [Weibull Distribution](../part-05-common-distributions/18-weibull-distribution.md); it does not prove the Central Limit Theorem or discuss where it licenses inference, which is [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md); it does not compute a coherent risk measure, which is [Expected Shortfall](11-expected-shortfall.md); it does not fit a tail to a real book or compare a fitted return level against a Gaussian one, which is [Drawdowns, Tail Risk, and Stress Testing](../../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md); and it never reports a shape parameter without the exceedance count and the standard error that go with it.

The trading stake is a course lesson that runs this machinery on the course's own book and reports both a headline and a warning. [Drawdowns, Tail Risk, and Stress Testing](../../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md) fits a generalized Pareto by peaks over threshold and prints `surviving   0.945%     119  +0.327      3.67%       1.38%   2.65x      8.36%        1.67%   5.01x` — a shape of $\xi=+0.327$, so moments exist only to order $3.1$, and a $1$-in-$40$-year loss of $8.36\%$ against a Gaussian $1.67\%$. It then refits at three thresholds and reports that `tsmom` flips from $\xi=+0.266$ to $\xi=-0.223$ on $47$ exceedances, concluding that "that fit is running on 47 exceedances and is not to be believed." Sections 3 and 4 are that judgement made quantitative: section 4 measures a standard error of $0.1695$ at $50$ exceedances, against which a swing of $0.489$ is large but not extraordinary.

## Maxima Have Their Own Limit Theorem, and It Allows Exactly Three Shapes

The Central Limit Theorem is a statement about sums and says nothing about maxima. There is a separate theorem for maxima, its conclusion is more restrictive, and the restriction is what makes tail extrapolation possible at all.

??? note "Proof that max-stability forces exactly three limit types, which the generalized extreme value family unifies in a single shape parameter"

    Suppose the normalized maximum of $n$ i.i.d. draws has a non-degenerate limit: $\mathbf{P}\!\left((M_n-b_n)/a_n\le x\right)\to G(x)$. Since $M_{nk}$ is the maximum of $k$ blocks each of size $n$, the limit must satisfy
    $$G^{k}(a_kx+b_k)=G(x)\qquad\text{for every }k,$$
    the **max-stability** equation: a maximum of maxima is a maximum, so the limit law must reproduce itself under raising to a power and rescaling. This is the analogue of the stability property that forces the Gaussian for sums, and it is far more restrictive.

    Taking logarithms twice converts it into a functional equation whose only solutions are the three types. Writing $H=-\log G$, the requirement becomes $kH(a_kx+b_k)=H(x)$, and the solutions are $H(x)=x^{-\alpha}$ on $x>0$ (**Fréchet**), $H(x)=e^{-x}$ on the whole line (**Gumbel**) and $H(x)=(-x)^{\alpha}$ on $x<0$ (**Weibull**). Von Mises' unification writes all three as the **generalized extreme value** law
    $$G_\xi(x)=\exp\!\left\{-(1+\xi x)^{-1/\xi}\right\},\qquad 1+\xi x>0,$$
    with $\xi>0$ giving Fréchet, $\xi<0$ giving Weibull and $\xi\to0$ giving Gumbel by continuity.

    The shape is the only thing that matters and it is a statement about the parent's tail. $\xi>0$ occurs exactly when the parent is regularly varying with index $\alpha=1/\xi$, which is the join with [Heavy-Tailed Returns](12-heavy-tailed-returns.md); $\xi=0$ covers everything with a tail decaying faster than any power and no upper bound, including the Gaussian, the exponential and the lognormal; $\xi<0$ occurs exactly when the parent has a finite right endpoint. The three cases correspond to a tail with unbounded moments, all moments, and a hard ceiling — three qualitatively different worlds separated by the sign of one number.

    **The load-bearing feature is that the trichotomy is exhaustive: if any limit exists it is one of these, so fitting a GEV is not choosing a convenient family but naming which of the three cases holds. That is what licenses extrapolation past the data, and it is why the sign of $\xi$ carries more information than its magnitude.**

## The Three Domains, and the Gaussian Reaches Its Limit Slowest of All

The theorem says the limit exists and does not say when it arrives. Measuring the arrival rate across the three domains puts a number on how much data an extrapolation needs.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18131)
REPS = 200_000

print(f"  normalized maxima of n draws converge to one of exactly three shapes, indexed by a"
      f" single parameter xi: Frechet (xi > 0, power-law tail), Gumbel (xi = 0, exponential-like)"
      f" and Weibull (xi < 0, bounded). Below, the fitted shape of the block maximum against the"
      f" theoretical value, and the Kolmogorov-Smirnov distance to the limit law. {REPS:,} blocks")
print("     law                domain     xi: theory   fitted   " + "".join(
    f"n={n}: KS to limit   " for n in (10, 100, 1_000)))
for name, dom, xi, draw in (
        ("Pareto(3)", "Frechet", 1 / 3, lambda s: (1 - rng.random(s)) ** (-1 / 3)),
        ("Student t(4)", "Frechet", 1 / 4, lambda s: rng.standard_t(4.0, s)),
        ("exponential", "Gumbel", 0.0, lambda s: rng.exponential(1.0, s)),
        ("normal", "Gumbel", 0.0, lambda s: rng.standard_normal(s)),
        ("uniform", "Weibull", -1.0, lambda s: rng.random(s))):
    cells, fitted = "", None
    for n in (10, 100, 1_000):
        m = draw((REPS, n)).max(axis=1)
        if n == 1_000:
            fitted = -stats.genextreme.fit(m)[0]          # scipy's c is the negated shape
        # location and scale only: the shape is held at the theoretical value, so the
        # distance measures convergence to the limit law rather than goodness of fit
        _, loc, sc = stats.genextreme.fit(m, f0=-xi)
        z = (m - loc) / sc
        cells += f"{stats.kstest(z, lambda q: stats.genextreme.cdf(q, -xi)).statistic:16.4f}   "
    print(f"    {name:18s} {dom:9s}  {xi:11.4f}   {fitted:6.4f}   " + cells)
# =>   normalized maxima of n draws converge to one of exactly three shapes, indexed by a single parameter xi: Frechet (xi > 0, power-law tail), Gumbel (xi = 0, exponential-like) and Weibull (xi < 0, bounded). Below, the fitted shape of the block maximum against the theoretical value, and the Kolmogorov-Smirnov distance to the limit law. 200,000 blocks
#         law                domain     xi: theory   fitted   n=10: KS to limit   n=100: KS to limit   n=1000: KS to limit   
#        Pareto(3)          Frechet         0.3333   0.3349             0.0070             0.0012             0.0011   
#        Student t(4)       Frechet         0.2500   0.2386             0.0327             0.0091             0.0027   
#        exponential        Gumbel          0.0000   0.0005             0.0098             0.0012             0.0013   
#        normal             Gumbel          0.0000   -0.0688             0.0370             0.0237             0.0168   
#        uniform            Weibull        -1.0000   -1.0014             0.0215             0.0033             0.0017   
```

The theory is confirmed on the shape column: $0.3349$ against a predicted $1/3$ for Pareto$(3)$, $0.2386$ against $1/4$ for $t(4)$, $0.0005$ against $0$ for the exponential and $-1.0014$ against $-1$ for the uniform, whose right endpoint at one is exactly the finite-ceiling case. Four laws with nothing in common produce maxima described by one two-parameter family plus a shape.

The normal row is the exception on both counts and it is the important one. Its fitted shape at $n=1{,}000$ is $-0.0688$ rather than $0$, and its Kolmogorov–Smirnov distance to the limit runs $0.0370$, $0.0237$, $0.0168$ — falling, but an order of magnitude slower than the Pareto's $0.0070$, $0.0012$, $0.0011$. The reason is a known pathology: the Gaussian's normalizing constants involve $\sqrt{2\log n}$, so the approach to the Gumbel limit is governed by $1/\log n$ rather than by a power of $n$, and going from a thousand observations to a million buys a factor of two.

The practical consequence is a warning about direction. A Gaussian sample of realistic size produces a *negative* fitted shape, which reads as a bounded tail — the safest of the three worlds — while the truth is unbounded. **The one distribution whose extremes finance most often assumes is also the one whose extreme-value approximation is worst at every sample size a desk possesses, and the error points toward complacency.**

## Threshold Choice Is a Bias–Variance Dial Whose Optimum Moves With the Sample

Block maxima waste data: one observation per block, and the rest discarded. The threshold formulation recovers them, at the cost of introducing the only free parameter in the method.

??? note "Proof that exceedances over a high threshold converge to a generalized Pareto with the same shape, and that the threshold trades a bias that does not shrink against a variance that does"

    Fix a level $u$ below the right endpoint and consider the conditional excess distribution $F_u(y)=\mathbf{P}(X-u\le y\mid X>u)$. The **Pickands–Balkema–de Haan** theorem states that $F$ lies in the domain of attraction of $G_\xi$ if and only if there is a scaling $\beta(u)$ with
    $$\sup_y\left|F_u(y)-H_{\xi,\beta(u)}(y)\right|\to0\qquad\text{as }u\to x_{\mathrm{end}},$$
    where $H$ is the **generalized Pareto** law $H_{\xi,\beta}(y)=1-(1+\xi y/\beta)^{-1/\xi}$. The shape is the *same* $\xi$ as in the block-maximum limit, so the two formulations estimate one parameter, and the threshold version uses every exceedance rather than one per block.

    The convergence is in $u$, which is where the trade-off comes from. At a finite threshold the excess law is only approximately generalized Pareto, and the discrepancy is a bias in $\hat\xi$ that depends on $u$ and not on the sample size. Raising $u$ shrinks that bias and reduces the exceedance count $k$, and the estimator's variance behaves like $(1+\xi)^{2}/k$. Writing $k\approx n(1-F(u))$, the mean squared error is
    $$\mathrm{MSE}(u)\;\approx\;b(u)^{2}+\frac{(1+\xi)^{2}}{n\,(1-F(u))},$$
    in which the first term is a function of $u$ alone and the second falls with $n$ at fixed $u$. So the minimizing threshold *rises* with the sample size: more data buys the right to be fussier about where the tail starts.

    **The load-bearing consequence is that there is no universal threshold. A rule of thumb expressed as a percentile — the top five percent, the top one percent — is a statement about $n$ disguised as a statement about the distribution, and it is correct only at the sample size where it was calibrated.**

```python
import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng(18133)
REPS, XI = 400, 0.25                                    # t(4) has tail index 1/4
QUANTS = (0.90, 0.95, 0.975, 0.99, 0.995)


def gpd_shape(y):
    """Profile maximum likelihood for a generalized Pareto with zero location: with
    theta = xi/beta the shape solves in closed form, leaving a one-dimensional search."""
    neg = lambda th: np.log(np.mean(np.log1p(th * y)) / th) + np.mean(np.log1p(th * y))
    r = optimize.minimize_scalar(neg, bounds=(-1 / y.max() + 1e-9, 10.0 / y.mean()),
                                 method="bounded")
    return np.mean(np.log1p(r.x * y))


def pot_shape(x, q):
    u = np.quantile(x, q)
    return gpd_shape(x[x > u] - u)


print(f"  peaks over threshold on Student t(4) data, whose exceedances converge to a generalized"
      f" Pareto with shape {XI} but are not one at any finite threshold. Raising the threshold"
      f" removes bias and destroys the data needed to fit, so the best threshold is wherever the"
      f" two errors balance -- and that place moves with the sample size. {REPS} replications")
print("     n         " + "".join(f"u = {q:.1%}: bias    sd   RMSE   " for q in QUANTS)
      + "  best threshold")
for n in (2_000, 20_000, 200_000):
    cells, rmses = "", []
    for q in QUANTS:
        est = np.array([pot_shape(np.abs(stats.t.rvs(4.0, size=n, random_state=rng)), q)
                        for _ in range(REPS)])
        rmse = np.sqrt(np.mean((est - XI) ** 2))
        rmses.append(rmse)
        cells += f"{est.mean() - XI:+13.4f}  {est.std():5.4f}  {rmse:5.4f}   "
    print(f"    {n:7,d}   " + cells + f"  {QUANTS[int(np.argmin(rmses))]:.1%}")
# =>   peaks over threshold on Student t(4) data, whose exceedances converge to a generalized Pareto with shape 0.25 but are not one at any finite threshold. Raising the threshold removes bias and destroys the data needed to fit, so the best threshold is wherever the two errors balance -- and that place moves with the sample size. 400 replications
#         n         u = 90.0%: bias    sd   RMSE   u = 95.0%: bias    sd   RMSE   u = 97.5%: bias    sd   RMSE   u = 99.0%: bias    sd   RMSE   u = 99.5%: bias    sd   RMSE     best threshold
#          2,000         -0.0720  0.0861  0.1123         -0.0522  0.1316  0.1416         -0.0797  0.2004  0.2157         -0.1612  0.3747  0.4079         -0.4891  0.8309  0.9641     90.0%
#         20,000         -0.0584  0.0258  0.0638         -0.0449  0.0384  0.0591         -0.0327  0.0567  0.0654         -0.0349  0.0872  0.0940         -0.0497  0.1359  0.1447     95.0%
#        200,000         -0.0581  0.0083  0.0587         -0.0395  0.0117  0.0412         -0.0290  0.0169  0.0336         -0.0184  0.0278  0.0333         -0.0143  0.0415  0.0439     99.0%
```

The two terms behave exactly as the proof requires. Reading down any threshold column, the bias barely moves with the sample size — $-0.0720$, $-0.0584$, $-0.0581$ at the $90$th percentile — because it is a property of where the tail was cut rather than of how much data sits above the cut. Reading the standard deviation down the same column gives $0.0861$, $0.0258$, $0.0083$, falling like $1/\sqrt n$ as the exceedance count grows.

The optimum therefore climbs: the $90.0$th percentile at two thousand observations, the $95.0$th at twenty thousand, the $99.0$th at two hundred thousand. A desk with eight years of daily data and one with eight hundred should not use the same rule, and the widespread convention of taking the top five percent is right for exactly one of them. **The threshold is not a modelling preference but a function of the sample size, and it is the only parameter in this method that no theorem determines.**

!!! note "The shape parameter, the tail index, the threshold and the exceedance count are four numbers behind a return level, and only the last is ever printed"
    **The shape parameter** $\xi$ decides which of three worlds the tail lives in and is the only thing that survives from the limit theorem. **The tail index** $\alpha=1/\xi$ is the same information in the reciprocal, defined only when $\xi>0$, and is what [Heavy-Tailed Returns](12-heavy-tailed-returns.md) estimates by a different route — the two must agree and comparing them is a free check. **The threshold** is the free parameter of section 3, has an optimum that moves with the sample, and is usually set by a round percentile. **The exceedance count** $k$ is what actually determines the precision, entering the standard error as $(1+\xi)/\sqrt k$, and it is the one quantity a published fit reliably reports — which is fortunate, because it is enough to reconstruct the error bar that the fit usually omits.

## On Gaussian Data the Shape's Sign Is Undetermined, and the Sign Is the Whole Model

Section 1 established that $\xi$'s sign separates three qualitatively different tails. Section 3 established that its precision is governed by the exceedance count. Putting the two together on data whose answer is known is the test that matters.

```python
import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng(18135)
REPS, P_FAR = 3_000, 1 - 1e-4
QUANTS = (0.95, 0.975, 0.99)
TRUTH = stats.norm.ppf(P_FAR)


def gpd_fit(y):
    """Profile maximum likelihood for a generalized Pareto with zero location."""
    neg = lambda th: np.log(np.mean(np.log1p(th * y)) / th) + np.mean(np.log1p(th * y))
    r = optimize.minimize_scalar(neg, bounds=(-1 / y.max() + 1e-9, 10.0 / y.mean()),
                                 method="bounded")
    xi = np.mean(np.log1p(r.x * y))
    return xi, xi / r.x


def pot(x, q, p):
    u = np.quantile(x, q)
    exc = x[x > u] - u
    xi, beta = gpd_fit(exc)
    return xi, u + beta / xi * ((len(x) * (1 - p) / len(exc)) ** -xi - 1)


print(f"  the same procedure applied to Gaussian data, whose extreme-value shape is exactly zero"
      f" and whose 1-in-{1 / (1 - P_FAR):,.0f} loss is {TRUTH:.3f}. A positive fitted shape claims"
      f" a power-law tail that is not there, and the fit has no way to report that the premise"
      f" failed. {REPS:,} replications")
print("     n        threshold   exceedances   xi: mean      sd   P(xi > 0)   P(xi > 0.15)"
      "   return level: median   relative error   90th pct")
for n in (1_000, 5_000, 20_000):
    for q in QUANTS:
        out = np.array([pot(np.abs(rng.standard_normal(n)), q, P_FAR) for _ in range(REPS)])
        xi_hat, lvl = out[:, 0], out[:, 1]
        print(f"    {n:6,d}   {q:9.1%}   {int(n * (1 - q)):11d}   {xi_hat.mean():8.4f}"
              f"   {xi_hat.std():5.4f}   {np.mean(xi_hat > 0):9.4f}   {np.mean(xi_hat > 0.15):12.4f}"
              f"   {np.median(lvl):20.3f}   {np.median(lvl) / TRUTH - 1:+14.2%}"
              f"   {np.percentile(lvl, 90):8.3f}")
# =>   the same procedure applied to Gaussian data, whose extreme-value shape is exactly zero and whose 1-in-10,000 loss is 3.719. A positive fitted shape claims a power-law tail that is not there, and the fit has no way to report that the premise failed. 3,000 replications
#         n        threshold   exceedances   xi: mean      sd   P(xi > 0)   P(xi > 0.15)   return level: median   relative error   90th pct
#         1,000       95.0%            50    -0.1831   0.1672      0.1250         0.0150                  3.642           -2.08%      4.440
#         1,000       97.5%            25    -0.2431   0.3001      0.1870         0.0593                  3.591           -3.45%      4.542
#         1,000       99.0%            10    -0.8192   0.9032      0.1827         0.1053                  3.481           -6.41%      4.449
#         5,000       95.0%           250    -0.1205   0.0610      0.0217         0.0000                  3.814           +2.57%      4.116
#         5,000       97.5%           125    -0.1187   0.0920      0.0933         0.0013                  3.818           +2.67%      4.156
#         5,000       99.0%            50    -0.1474   0.1695      0.1850         0.0303                  3.800           +2.19%      4.181
#        20,000       95.0%          1000    -0.1085   0.0300      0.0000         0.0000                  3.854           +3.64%      4.005
#        20,000       97.5%           500    -0.0991   0.0428      0.0060         0.0000                  3.871           +4.10%      4.029
#        20,000       99.0%           200    -0.0970   0.0706      0.0750         0.0007                  3.866           +3.95%      4.042
```

The estimator is not biased toward finding heavy tails, which is worth saying because it is the opposite of the usual suspicion: on Gaussian data the mean fitted shape is negative at every setting, from $-0.0970$ to $-0.8192$, so the typical fit reports a *bounded* tail. What it lacks is precision. At a thousand observations and a $99\%$ threshold — ten exceedances — the shape reads $-0.8192$ with a standard deviation of $0.9032$, an interval that comfortably contains a hard ceiling, an exponential tail and an infinite-variance power law. The sign is the model, and the sign is undetermined.

The consequence is measured in the two probability columns. A material minority of Gaussian samples report a positive shape and therefore a power-law tail that does not exist: $0.1250$, $0.1870$ and $0.1827$ at a thousand observations, still $0.0750$ at twenty thousand with a $99\%$ threshold, and $0.0303$ report a shape above $0.15$, which would be read as a serious heavy tail. The extrapolated $1$-in-$10{,}000$ loss holds up at the median — within $4.10\%$ of the truth everywhere — while its upper decile reaches $4.542$ against a true $3.719$, so one fit in ten overstates by $22\%$.

This is what the published warning was detecting. The lesson's `tsmom` fit flips from $\xi=+0.266$ to $\xi=-0.223$ across one threshold on $47$ exceedances, a swing of $0.489$; the table above puts the standard deviation at $50$ exceedances at $0.1695$, so that swing is about $2.9$ standard errors — large, but nothing like impossible for a statistic this noisy. **The published judgement that a fit on forty-seven exceedances "is not to be believed" is not conservatism; it is the correct reading of an error bar that the fit does not print.**

## Every Repair Is More Exceedances, Two Thresholds, or a Sign That Is Reported as Unknown

The three findings admit three responses of decreasing ambition. Section 2's slow Gaussian convergence cannot be repaired by more data in any practical sense, since the error decays like $1/\log n$; the response is to distrust extreme-value approximations most where the parent is closest to Gaussian, which is the opposite of the usual intuition that light tails are the easy case.

Section 3's threshold has no free lunch either, but it has a diagnostic that costs nothing: fit at several thresholds and read the shape where it is flat, which is precisely the discipline the course lesson applies when it refits at three levels and reports the range. That is the same plateau logic [Heavy-Tailed Returns](12-heavy-tailed-returns.md) establishes for the Hill estimator, and the two estimators must agree where both apply, so running both is a free cross-check on a single sample.

Section 4's is the cheapest and least used. The standard error of $\hat\xi$ is approximately $(1+\hat\xi)/\sqrt k$ with $k$ the exceedance count, so it can be written down from numbers every published fit already reports, and it decides whether the sign — and therefore the entire qualitative model — is determined.

!!! warning "A fitted shape parameter is reported to three decimals and its sign is often not established at all"
    A generalized Pareto fit returns $\xi$ as a number, and every downstream extrapolation depends on which of three worlds that number places the tail in. Section 4 measures a case where the point estimate is $-0.8192$ and the standard deviation is $0.9032$: the fit is compatible with a bounded loss and with infinite variance simultaneously. **The free diagnostic is $(1+\hat\xi)/\sqrt k$, the shape's approximate standard error from the exceedance count alone: at $k=50$ it is about $0.17$, at $k=119$ about $0.12$, at $k=500$ about $0.06$ — so a fit is entitled to claim a power-law tail only when $\hat\xi$ exceeds roughly twice that.** The published $\xi=+0.327$ on $119$ exceedances clears it by a comfortable margin, at about $2.7$ standard errors; the $\xi=-0.223$ on $47$ does not, at about $1.3$. Both numbers appear in the same table with the same number of decimal places, and the count in the neighbouring column is what separates them.

## Three Worlds Separated by One Sign

This page established that max-stability forces exactly three limit types for normalized maxima, unified by the generalized extreme value family in a single shape parameter whose sign distinguishes a power-law tail, an exponential-like one and a hard ceiling — confirmed at fitted shapes of $0.3349$, $0.2386$, $0.0005$ and $-1.0014$ against theoretical $1/3$, $1/4$, $0$ and $-1$; that the limit arrives at wildly different speeds, the Gaussian's Kolmogorov–Smirnov distance falling only from $0.0370$ to $0.0168$ across a hundredfold increase in block size against a Pareto's $0.0070$ to $0.0011$, with a shape fitted to Gaussian maxima reading $-0.0688$ where the truth is zero; that exceedances converge to a generalized Pareto with the same shape, so threshold choice trades a bias fixed by the cut against a variance falling like $1/\sqrt n$, putting the optimum at the $90.0$th, $95.0$th and $99.0$th percentile at two thousand, twenty thousand and two hundred thousand observations; and that on Gaussian data the fitted shape averages between $-0.0970$ and $-0.8192$ with standard deviations up to $0.9032$, so $0.1250$ to $0.1870$ of samples report a positive shape at a thousand observations and one fit in ten overstates the $1$-in-$10{,}000$ loss by $22\%$.

The relationship to the previous page is a division of the same problem. [Heavy-Tailed Returns](12-heavy-tailed-returns.md) asks whether a power-law tail exists and estimates its index conditional on the answer being yes; this page asks which of three regimes holds and estimates a parameter whose sign supplies the answer. Both are threshold methods, both have an estimator whose precision is set by the exceedance count rather than the sample size, and both fail in the same silent way — by returning a confident number where the premise does not hold. What separates them is that the extreme-value formulation makes the premise itself a parameter, so the failure at least has a place to show up: a shape whose standard error spans zero is the model saying it does not know, which is more than the Hill estimator can say. Everything in the last four pages has concerned one series at a time. The next asks what happens when two of them go wrong together.

**Extreme value theory reduces every tail to one number, and the number that matters is its sign, which a realistic sample often fails to establish.**
