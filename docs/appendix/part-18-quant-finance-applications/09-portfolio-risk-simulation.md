# Portfolio Risk Simulation

A simulated risk number is produced in two stages — estimate a covariance from history, then draw paths from the estimate — and it therefore carries two errors, of which the simulation reports one. The reported one is the smaller by a wide and computable margin. For a ten-asset book at equal weights the parameter component is a fixed *relative* $1/\sqrt{2T}$ at every confidence level, measured at $0.0624$, $0.0442$, $0.0257$ and $0.0139$ against predictions of $0.0632$, $0.0446$, $0.0257$ and $0.0141$; at a hundred thousand paths it accounts for $0.9935$ of the total variance on a year of history and still $0.8881$ on ten years. The two errors cross at a path count that is small and rises with the confidence level — $832$, $1{,}298$, $4{,}651$ and $23{,}253$ paths at $95\%$, $99\%$, $99.9\%$ and $99.99\%$ on a year of data — so any simulation worth running is on the parameter-dominated side of it. Past that point more paths buy reported precision and nothing else: the actual error settles at a floor of $0.078090\%$ while the Monte Carlo standard error the run computes for itself keeps falling, and the ratio between what is claimed and what is delivered reaches $27.3$.

This page covers the two-stage error budget of a simulated risk number, the law of total variance that separates it, the closed form for the parameter component under fixed weights and what it does not depend on, the sampling component's dependence on the density at the quantile, the path count at which the two cross, and what a Monte Carlo standard error is actually measuring. It does not develop the sample covariance matrix, its eigenvalue spreading, shrinkage, or the repairs that return a matrix to the positive semi-definite cone, all of which are [Covariance Matrices](../part-06-multivariate-probability/02-covariance-matrices.md); it optimizes no portfolio and compares no estimator against equal weighting, which is [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md); it derives no quantile-estimator asymptotics and proves nothing about subadditivity, which is [Value at Risk](10-value-at-risk.md); it builds no Monte Carlo estimator and no confidence interval for a mean, which is [Monte Carlo Simulation](../part-09-monte-carlo-methods/03-monte-carlo-simulation.md); it reduces no variance, which is [Variance Reduction](../part-09-monte-carlo-methods/06-variance-reduction.md); it backtests no risk model against real breaches and shocks no correlation matrix, which are [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md) and [Drawdowns, Tail Risk, and Stress Testing](../../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md); and it never reports a Monte Carlo error bar as though it were the error.

The trading stake is a covariance matrix a course lesson reads diversification off, and the question of what its estimation error is worth. [Portfolio Construction and Transaction Costs](../../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md) prints "ten pairwise correlations, all between $-0.29$ and $+0.34$, most near zero — five return streams that genuinely do not share a factor," names this page as the place that "formalizes why that is worth money," and books the result at Sharpe $1.51$. Every number downstream of that matrix inherits its sampling error, and section 2 gives the exchange rate: at the $4{,}758$-day window [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md) uses, the relative error a simulated risk number inherits from the matrix alone is $1/\sqrt{2\times4758}=1.03\%$, and at one year of data it is $4.5\%$ — before a single path is drawn.

## A Simulated Risk Number Has Two Errors and Reports One

The two stages are usually described as one procedure, which is what makes the error budget invisible. Separating them is a single application of a standard identity, and the separation says immediately that one term cannot be reduced by computation.

??? note "Proof that the variance of a simulated risk number splits into a sampling term and a parameter term, and that the parameter term for fixed weights is a relative $1/\sqrt{2(T-1)}$ independent of the number of assets and of the confidence level"

    Let $\hat V$ be the risk number produced by drawing $N$ paths from an estimate $\hat\Sigma$ built on $T$ observations. Conditioning on $\hat\Sigma$ and applying the law of total variance,
    $$\mathrm{Var}(\hat V)=\underbrace{\mathbb{E}\!\left[\mathrm{Var}(\hat V\mid\hat\Sigma)\right]}_{\text{sampling}}+\underbrace{\mathrm{Var}\!\left(\mathbb{E}[\hat V\mid\hat\Sigma]\right)}_{\text{parameter}}.$$
    The first term is what the simulation's own standard error estimates and it falls like $1/N$. The second is the variance of the answer the simulation would give with infinitely many paths; it does not contain $N$ at all, so no amount of computation touches it. The total therefore has a floor, and the floor is set entirely by the history.

    For a book with *fixed* weights $w$ and Gaussian returns, the parameter term is exactly computable. The projected series $y_i=w^\top x_i$ is i.i.d. scalar Gaussian with variance $w^\top\Sigma w$, and its sample variance satisfies
    $$w^\top\hat\Sigma w=\frac{w^\top\Sigma w}{T-1}\chi^{2}_{T-1},$$
    so the simulated volatility $\sqrt{w^\top\hat\Sigma w}$ is a scaled chi variable with relative standard deviation $\approx1/\sqrt{2(T-1)}$. A Gaussian risk number is that volatility times a quantile multiplier, and the multiplier scales the estimate and its error together, so the *relative* parameter error is $1/\sqrt{2(T-1)}$ at every confidence level and does not depend on the number of assets.

    Both independences are worth stating because both are commonly assumed the other way. Dimension does not enter because the weights were fixed in advance: the estimator only ever sees the one-dimensional projection $w^\top x$, whatever $K$ is. Dimension *does* enter the moment the weights are chosen using $\hat\Sigma$, which is the in-sample optimism of [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md), and the eigenvalue spreading of [Covariance Matrices](../part-06-multivariate-probability/02-covariance-matrices.md) is what drives it.

    **The load-bearing consequence is an asymmetry in what is buyable. The sampling term is bought with computation, which is cheap and getting cheaper; the parameter term is bought with calendar time, which is neither. A risk system that improves year over year because its hardware improved has been improving the term that was already negligible.**

## The Parameter Component Is a Fixed Relative Error and Dominates Everything Past About a Thousand Paths

The decomposition is measurable directly by running the same procedure three ways: both stages, the parameter stage alone with infinite paths, and the sampling stage alone against the true covariance.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18091)
K, RHO, REPS, N_PATHS, D, P = 10, 0.30, 3_000, 100_000, 252, 0.01
VOLS = np.linspace(0.15, 0.25, K)
SIGMA = RHO * np.outer(VOLS, VOLS) + (1 - RHO) * np.diag(VOLS ** 2)
W = np.full(K, 1 / K)
CHOL = np.linalg.cholesky(SIGMA)
true_sd = np.sqrt(W @ SIGMA @ W / D)
true_var = -stats.norm.ppf(P) * true_sd

print(f"  a {K}-asset book at equal weights, {RHO:.0%} equicorrelated, true 1-day {1 - P:.0%} VaR"
      f" = {true_var:.4%}. A simulated risk number is produced in two stages -- estimate the"
      f" covariance from T days, then draw {N_PATHS:,} paths from the estimate -- so its error has"
      f" a parameter component and a sampling component. {REPS:,} replications")
print("     T days   total sd of VaR   parameter only   sampling only   quadrature check"
      "   parameter share   predicted 1/sqrt(2T)   measured relative sd")
for T in (126, 252, 756, 2520):
    both, par, samp = np.empty(REPS), np.empty(REPS), np.empty(REPS)
    for r in range(REPS):
        x = rng.standard_normal((T, K)) @ CHOL.T / np.sqrt(D)
        s_hat = np.cov(x, rowvar=False)
        sd_hat = np.sqrt(W @ s_hat @ W)
        par[r] = -stats.norm.ppf(P) * sd_hat                      # N = infinity
        z = rng.standard_normal(N_PATHS)
        both[r] = -np.quantile(z * sd_hat, P)                     # estimated Sigma, N paths
        samp[r] = -np.quantile(z * true_sd, P)                    # true Sigma, N paths
    print(f"    {T:6d}   {both.std():15.6%}   {par.std():14.6%}   {samp.std():13.6%}"
          f"   {np.sqrt(par.var() + samp.var()):16.6%}"
          f"   {par.var() / (par.var() + samp.var()):16.4f}"
          f"   {1 / np.sqrt(2 * (T - 1)):21.4f}   {par.std() / true_var:20.4f}")
# =>   a 10-asset book at equal weights, 30% equicorrelated, true 1-day 99% VaR = 1.7871%. A simulated risk number is produced in two stages -- estimate the covariance from T days, then draw 100,000 paths from the estimate -- so its error has a parameter component and a sampling component. 3,000 replications
#         T days   total sd of VaR   parameter only   sampling only   quadrature check   parameter share   predicted 1/sqrt(2T)   measured relative sd
#           126         0.111752%        0.111492%       0.009046%          0.111859%             0.9935                  0.0632                 0.0624
#           252         0.079390%        0.079039%       0.009094%          0.079560%             0.9869                  0.0446                 0.0442
#           756         0.046984%        0.045993%       0.009130%          0.046890%             0.9621                  0.0257                 0.0257
#          2520         0.026197%        0.024909%       0.008841%          0.026431%             0.8881                  0.0141                 0.0139
```

The decomposition is exact to the fourth decimal: the total standard deviations of $0.111752\%$, $0.079390\%$, $0.046984\%$ and $0.026197\%$ match the quadrature sums $0.111859\%$, $0.079560\%$, $0.046890\%$ and $0.026431\%$, confirming that the two components are orthogonal and that nothing else is present. The closed form holds too — measured relative parameter errors of $0.0624$, $0.0442$, $0.0257$ and $0.0139$ against $1/\sqrt{2(T-1)}$ values of $0.0632$, $0.0446$, $0.0257$ and $0.0141$.

The parameter-share column is the finding. At a hundred thousand paths, which is a modest simulation by any standard, the covariance estimate accounts for $99.35\%$ of the variance of the answer on a year of history and $88.81\%$ on ten years. The sampling column is nearly constant across rows at about $0.009\%$, as it must be — it depends on $N$ and not on $T$ — while the parameter column falls by a factor of four as the history lengthens sixteen-fold. **Every row is a simulation whose output is a measurement of its own input, and the paths are there to interpolate rather than to inform.**

## The Crossing Point Rises With the Confidence Level, Because Only One of the Two Errors Notices the Tail

The two components respond differently to being asked for a rarer loss, and the difference decides how many paths a given confidence level actually requires.

??? note "Proof that the relative sampling error grows as the confidence level rises while the relative parameter error does not, so the crossover path count is $2T\left[\sqrt{p(1-p)}/(\varphi(z_p)z_p)\right]^{2}$"

    A simulated quantile is a sample quantile of $N$ draws, and [Value at Risk](10-value-at-risk.md) establishes its asymptotic standard deviation as $\sqrt{p(1-p)/N}\big/f(q_p)$. Dividing by the quantile itself to make it relative, and specializing to the Gaussian where $f(q_p)=\varphi(z_p)/\sigma$ and $q_p=z_p\sigma$,
    $$\frac{\mathrm{sd}(\hat q_p)}{q_p}=\frac{1}{\sqrt N}\cdot\frac{\sqrt{p(1-p)}}{\varphi(z_p)\,z_p}.$$
    The bracket grows without bound as $p\to0$, because the standard normal density decays like $e^{-z^{2}/2}$ while $z_p$ grows only like $\sqrt{2\log(1/p)}$. Numerically it is $1.285$ at $95\%$, $1.605$ at $99\%$, $3.038$ at $99.9\%$ and $6.792$ at $99.99\%$.

    The parameter side has no such term. Section 1 showed its relative error to be $1/\sqrt{2(T-1)}$ with the quantile multiplier cancelling, so it is the same $4.5\%$ on a year of data whether the question is a $95\%$ loss or a $99.99\%$ one. Setting the two relative errors equal,
    $$N^{\ast}=2T\left[\frac{\sqrt{p(1-p)}}{\varphi(z_p)z_p}\right]^{2},$$
    which is the path count beyond which the simulation is measuring its covariance estimate rather than its own noise.

    **The load-bearing asymmetry is which error sees the tail. A rarer loss is harder to *simulate* because fewer paths land there, and no harder to *estimate* from a fixed history, because a Gaussian tail is a deterministic function of a scale parameter. The moment that last assumption fails — a fitted tail index rather than a fitted volatility — the parameter error acquires a confidence-level dependence of its own, which is [Extreme Value Theory](13-extreme-value-theory.md).**

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18093)
REPS, D = 4_000, 252
LEVELS = (0.05, 0.01, 0.001, 0.0001)

print(f"  where the two errors cross. The parameter error of a Gaussian VaR is a fixed *relative*"
      f" 1/sqrt(2T) at every confidence level, because the quantile multiplies both the estimate"
      f" and its error; the sampling error is sqrt(p(1-p)/N)/phi(z_p) relative to z_p, which grows"
      f" as the tail thins. Their crossing point is the path count beyond which a simulation is"
      f" measuring its covariance rather than its own noise. {REPS:,} replications")
print("     confidence   z       relative sampling error x sqrt(N)   " + "".join(
    f"T={t}: crossover N   " for t in (126, 252, 756, 2520)))
for p in LEVELS:
    z = -stats.norm.ppf(p)
    k = np.sqrt(p * (1 - p)) / stats.norm.pdf(z) / z          # relative sampling sd times sqrt(N)
    cells = "".join(f"{k ** 2 * 2 * t:16,.0f}   " for t in (126, 252, 756, 2520))
    print(f"    {1 - p:10.2%}   {z:5.3f}   {k:33.3f}   " + cells)

print("\n     the check at T = 252: at the predicted crossing the two errors should be equal")
print("     confidence   crossover N   parameter relative sd: predicted   measured"
      "   sampling relative sd at that N")
T = 252
for p_ in LEVELS:
    z = -stats.norm.ppf(p_)
    sds = np.array([np.std(rng.standard_normal(T), ddof=1) for _ in range(REPS)])
    k = np.sqrt(p_ * (1 - p_)) / stats.norm.pdf(z) / z
    n = int(round(k ** 2 * 2 * T))
    q = np.array([-np.quantile(rng.standard_normal(n), p_) / z for _ in range(REPS)])
    print(f"    {1 - p_:10.2%}   {n:11,d}   {1 / np.sqrt(2 * T):33.4f}   {sds.std() / sds.mean():8.4f}"
          f"   {q.std():30.4f}")
# =>   where the two errors cross. The parameter error of a Gaussian VaR is a fixed *relative* 1/sqrt(2T) at every confidence level, because the quantile multiplies both the estimate and its error; the sampling error is sqrt(p(1-p)/N)/phi(z_p) relative to z_p, which grows as the tail thins. Their crossing point is the path count beyond which a simulation is measuring its covariance rather than its own noise. 4,000 replications
#         confidence   z       relative sampling error x sqrt(N)   T=126: crossover N   T=252: crossover N   T=756: crossover N   T=2520: crossover N   
#            95.00%   1.645                               1.285                416                832              2,496              8,319   
#            99.00%   2.326                               1.605                649              1,298              3,894             12,979   
#            99.90%   3.090                               3.038              2,325              4,651             13,952             46,505   
#            99.99%   3.719                               6.792             11,626             23,253             69,758            232,527   
#
#         the check at T = 252: at the predicted crossing the two errors should be equal
#         confidence   crossover N   parameter relative sd: predicted   measured   sampling relative sd at that N
#            95.00%           832                              0.0445     0.0447                           0.0441
#            99.00%         1,298                              0.0445     0.0452                           0.0435
#            99.90%         4,651                              0.0445     0.0453                           0.0409
#            99.99%        23,253                              0.0445     0.0442                           0.0390
```

The crossings are small numbers. On a year of daily history a $99\%$ simulated VaR is parameter-dominated past $1{,}298$ paths, and even a $99.99\%$ number crosses over at $23{,}253$ — a simulation that finishes in under a second. The second panel confirms the arithmetic: at each predicted crossing the sampling relative standard deviation lands at $0.0441$, $0.0435$, $0.0409$ and $0.0390$ against a parameter relative standard deviation measured at $0.0447$, $0.0452$, $0.0453$ and $0.0442$, agreeing to within the asymptotic approximation's own accuracy — which degrades slightly at the extreme levels, exactly where the effective number of paths in the tail is smallest.

Ten years of history moves the crossings out by a factor of ten and no further. **There is no realistic combination of history length and confidence level for which a simulation with a plausible path count is limited by its paths, so the entire practice of quoting a Monte Carlo standard error alongside a risk number is a report on the wrong term.**

!!! note "The covariance estimate, the path count, the confidence level and the horizon are four inputs to a simulated risk number, and each one binds a different error"
    **The covariance estimate** is bought with calendar time and sets the floor of section 1, contributing a fixed relative $1/\sqrt{2T}$. **The path count** is bought with computation, contributes the only error the simulation reports, and stops mattering past the crossings of section 3. **The confidence level** is a choice about which question is being asked, and it moves the sampling error without moving the parameter error — which is why a $99.99\%$ number is genuinely harder to simulate and no harder to estimate. **The horizon** is the one input that changes both, because a ten-day risk number built by scaling a one-day estimate inherits the scaling rule's error on top of everything here, which is the aggregation problem [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md) treats. Only the second of the four appears in the error bar, and it is the only one a desk can change this afternoon.

## Ten Million Paths, and a Number Fixed by Two Hundred and Fifty-Two Days

Sections 2 and 3 establish where the floor is. What a simulation does when pushed well past it is the practical failure, because the diagnostic it prints keeps improving.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18095)
K, RHO, REPS, T, D, P = 10, 0.30, 400, 252, 252, 0.01
COUNTS = (1_000, 10_000, 100_000, 1_000_000)
VOLS = np.linspace(0.15, 0.25, K)
SIGMA = RHO * np.outer(VOLS, VOLS) + (1 - RHO) * np.diag(VOLS ** 2)
W = np.full(K, 1 / K)
CHOL = np.linalg.cholesky(SIGMA)
true_sd = np.sqrt(W @ SIGMA @ W / D)
true_var = -stats.norm.ppf(P) * true_sd

est = np.empty((len(COUNTS), REPS))
sds = np.empty(REPS)
for r in range(REPS):
    x = rng.standard_normal((T, K)) @ CHOL.T / np.sqrt(D)
    sds[r] = np.sqrt(W @ np.cov(x, rowvar=False) @ W)
    z = rng.standard_normal(COUNTS[-1])                # one draw, read at nested prefixes
    for i, n in enumerate(COUNTS):
        est[i, r] = -np.quantile(z[:n], P) * sds[r]

print(f"  the same book with {T} days of history behind its covariance estimate, simulated at"
      f" path counts spanning three orders of magnitude. The reported error bar is the Monte Carlo"
      f" standard error the run computes for itself; the actual error is measured against the true"
      f" {1 - P:.0%} VaR of {true_var:.4%}. {REPS} replications")
print("     paths       actual sd of VaR   reported MC error   ratio actual / reported"
      "   paths setting the quantile   floor from the covariance alone")
floor = np.std(-stats.norm.ppf(P) * sds)
for i, n in enumerate(COUNTS):
    se_rep = np.sqrt(P * (1 - P) / n) / stats.norm.pdf(stats.norm.ppf(P)) * true_sd
    print(f"    {n:10,d}   {est[i].std():20.6%}   {se_rep:17.6%}   {est[i].std() / se_rep:22.1f}"
          f"   {n * P:29,.0f}   {floor:33.6%}")
# =>   the same book with 252 days of history behind its covariance estimate, simulated at path counts spanning three orders of magnitude. The reported error bar is the Monte Carlo standard error the run computes for itself; the actual error is measured against the true 99% VaR of 1.7871%. 400 replications
#         paths       actual sd of VaR   reported MC error   ratio actual / reported   paths setting the quantile   floor from the covariance alone
#             1,000              0.113605%           0.090690%                      1.3                              10                           0.078090%
#            10,000              0.082490%           0.028679%                      2.9                             100                           0.078090%
#           100,000              0.078443%           0.009069%                      8.6                           1,000                           0.078090%
#         1,000,000              0.078413%           0.002868%                     27.3                          10,000                           0.078090%
```

The actual error falls from $0.113605\%$ to $0.078413\%$ and then stops, converging onto the covariance-only floor of $0.078090\%$ that it can never go below. The reported Monte Carlo error keeps going — $0.090690\%$, $0.028679\%$, $0.009069\%$, $0.002868\%$ — because it is a correct estimate of a quantity that stopped mattering after the first row. The ratio between the two runs $1.3$, $2.9$, $8.6$ and $27.3$, so a run at a million paths understates its own error by a factor of twenty-seven while reporting six significant figures.

This is the honest failure and it is not a bug in anyone's code. The Monte Carlo standard error answers the question it claims to answer — how much would this number move if the random draws changed — and that question stopped being the interesting one at $1{,}298$ paths. The interesting question is how much the number would move if the *history* changed, and no rerun of the simulation can answer it, because every rerun conditions on the same $\hat\Sigma$. **A precision that improves with computation, attached to a number whose error is set by the calendar, is a measurement of the machine rather than of the risk.**

## Every Repair Is More History, a Different Question, or an Error Bar That Includes the Estimate

The floor is real and only two things move it. More calendar time moves it at $1/\sqrt{2T}$, which is slow and cannot be accelerated by sampling more finely within the same window — the same statement [Geometric Brownian Motion](../part-08-stochastic-processes/09-geometric-brownian-motion.md) proves for a drift, arriving here for a second moment. Shrinkage moves it by accepting bias for variance, which is [Covariance Matrices](../part-06-multivariate-probability/02-covariance-matrices.md), and is the only lever that acts on the binding term at a fixed history.

The third repair is not to move the floor but to report it, and it costs one extra loop. Resampling the *history* — a parametric bootstrap that redraws $T$ observations, re-estimates $\hat\Sigma$, and re-runs the simulation — produces an interval that contains both components instead of one, and the machinery is identical to the one already in place. What makes this rare is not difficulty but framing: the covariance matrix arrives as an input, and inputs are not usually given error bars.

!!! warning "The error bar a risk simulation prints is a correct estimate of its smallest error, and it improves as the number gets no better"
    Every Monte Carlo risk report carries a standard error, and every one of them measures the sampling term. Section 4 shows that term at $0.002868\%$ against an actual error of $0.078413\%$ — a factor of $27.3$ — on a run whose only defect is that it used a year of data. **The free diagnostic is $1/\sqrt{2T}$, evaluated on the window the covariance was estimated from and multiplied by the risk number itself: at $T=252$ it is $4.5\%$ of the answer, at $T=756$ it is $2.6\%$, and at the $4{,}758$-day window a course lesson uses it is $1.03\%$.** It requires no simulation, no resampling and no new data, it can be written down before the model is run, and comparing it against the reported Monte Carlo error settles in one line whether the path count is worth arguing about. Where the risk number is not a Gaussian quantile the formula is a lower bound rather than an equality, because a fitted tail is estimated less precisely than a fitted variance.

## Two Errors, One Report, and the Wrong One in the Interval

This page established that a simulated risk number's variance splits by the law of total variance into a sampling term that falls like $1/N$ and a parameter term that does not contain $N$ at all, verified by quadrature at $0.111859\%$, $0.079560\%$, $0.046890\%$ and $0.026431\%$ against measured totals of $0.111752\%$, $0.079390\%$, $0.046984\%$ and $0.026197\%$; that under fixed weights the parameter term is a relative $1/\sqrt{2(T-1)}$ independent of the number of assets and of the confidence level, measured at $0.0624$, $0.0442$, $0.0257$ and $0.0139$ against $0.0632$, $0.0446$, $0.0257$ and $0.0141$, and accounting for $0.9935$ of the total variance at a hundred thousand paths on a year of history; that the two cross at $2T[\sqrt{p(1-p)}/(\varphi(z_p)z_p)]^{2}$ paths, which is $832$, $1{,}298$, $4{,}651$ and $23{,}253$ at the four standard confidence levels on a year of data, with the sampling error at each crossing landing at $0.0441$, $0.0435$, $0.0409$ and $0.0390$ against a parameter error of $0.0447$, $0.0452$, $0.0453$ and $0.0442$; and that pushing past the crossing takes the actual error to a floor of $0.078090\%$ while the reported Monte Carlo error falls to $0.002868\%$, a ratio of $27.3$.

The pairing with [Monte Carlo Option Pricing](08-monte-carlo-option-pricing.md) is exact and worth naming, because the two pages found the same structure in different quantities. There, a simulated price carried a discretization bias and a sampling error, the bias was invisible to the reported standard error, and the optimal budget split was computable and never computed. Here, a simulated risk number carries a parameter error and a sampling error, the parameter error is invisible to the reported standard error, and the crossover is computable and never computed. In both cases the error a Monte Carlo run reports is the one it can see from inside itself, which is the one that would change if the seed changed — and in both cases the error that matters comes from something fixed before the first draw. What this page has taken for granted throughout is that the risk number in question is a quantile of a Gaussian, so that a volatility determines it. Dropping that is the next page.

**A simulation reports how much its answer depends on its random numbers, which is a complete account of everything except where the answer came from.**
