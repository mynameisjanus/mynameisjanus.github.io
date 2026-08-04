# Bayesian Updating

Sequential updating is the framework's most attractive property and the one most casually misapplied. The attraction is real and provable: yesterday's posterior serves as today's prior, the order of the evidence does not matter, and a belief cannot forecast the direction of its own revision. Below, batch and sequential and randomly reordered updates over four hundred days agree to $2.060\times10^{-18}$, and the posterior mean's increments have a correlation with its level of $-0.0042$ — a martingale to the precision the simulation can measure. The misapplication is that all of it rests on the likelihood factorizing, and a research desk breaks that factorization routinely and invisibly. Updating once a day on overlapping twenty-day returns — a completely ordinary thing to do — produces a posterior standard deviation of $0.7139$ basis points where the honest figure is $3.1810$, an effective sample size of $49.4$ against the $981$ updates performed, and coverage of $0.3388$. And the coherence that makes updating attractive is also what makes a belief unable to abandon anything: a posterior that never discounts takes $498$ days to concede a regime change, while the discounting that fixes that costs a permanent error bar and wins only when the parameter moves further than the noise would have let anyone measure anyway.

This page covers sequential updating at the level of a parameter, the factorization that makes it exact, order invariance, the martingale property of the posterior mean and what it forbids, the failure of all three when observations overlap or are dependent, and exponential forgetting as the standard repair together with what it costs. It applies Bayes' rule to no event and derives no odds form, which is [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md); it proves no closure and derives no conjugate family, which is [Conjugate Priors](04-conjugate-priors.md); it constructs no prior, which is [Prior Distributions](02-prior-distributions.md); it normalizes no posterior numerically and proves no asymptotics, which is [Posterior Distributions](03-posterior-distributions.md); it derives no precision-form Gaussian update in several dimensions, which is [Conditional Gaussian](../part-06-multivariate-probability/06-conditional-gaussian.md); it builds no state-space model and derives no Kalman recursion, which is [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md); it computes no marginal likelihood, which is [Bayesian Model Comparison](06-bayesian-model-comparison.md); it forecasts no future observation, which is [Bayesian Prediction](07-bayesian-prediction.md); it applies no update to a live trading signal, which is [Bayesian Signal Updating](../part-18-quant-finance-applications/08-bayesian-signal-updating.md); it establishes no martingale convergence theorem, which is [Martingales](../part-08-stochastic-processes/10-martingales.md); and it never treats a posterior that has stopped moving as evidence that the parameter has.

The trading stake is a course lesson publishing the exact conversion this page's fourth section measures from the other direction. [Particle and Kalman Filters](../../advanced/02-particle-and-kalman-filters.md) tabulates the steady-state gain of a Kalman filter against its signal-to-noise ratio, printing `signal-to-noise q = Q/R =  0.001: steady-state gain 0.0311, EWMA lambda 0.9689, half-life  21.9 periods`, and instructs the reader to "read the table as a conversion chart between a modeling belief and a window length." That is exactly right, and section 4 supplies the other half of the chart: what each window length costs in permanent uncertainty, and which of them is actually best once the size of the change is taken into account.

## Yesterday's Posterior Is Today's Prior Because the Likelihood Factorizes, and Order Invariance Is the Same Fact Read Backwards

The sequential form is usually presented as a convenience — a way to avoid re-reading the archive every morning. It is more than that, because it is exact rather than approximate, and the exactness has a single source.

??? note "Proof that the sequential update equals the batch update and is invariant to the order of the observations, and that both follow from conditional independence rather than from Bayes' rule"

    Let $x_1,\dots,x_n$ be conditionally independent given $\theta$. Write $\pi_t(\theta)=\pi(\theta\mid x_{1:t})$ with $\pi_0=\pi$. The batch posterior is
    $$\pi_n(\theta)\ \propto\ \pi(\theta)\prod_{i=1}^{n}f(x_i\mid\theta).$$
    Proceed by induction. Suppose $\pi_{t-1}(\theta)=\pi(\theta)\prod_{i<t}f(x_i\mid\theta)/Z_{t-1}$. Treating $\pi_{t-1}$ as the prior and $x_t$ as the data,
    $$\frac{\pi_{t-1}(\theta)f(x_t\mid\theta)}{\int\pi_{t-1}(\theta')f(x_t\mid\theta')\,\mathrm{d}\theta'}=\frac{\pi(\theta)\prod_{i\le t}f(x_i\mid\theta)/Z_{t-1}}{Z_t/Z_{t-1}}=\pi_t(\theta),$$
    the constants $Z_{t-1}$ cancelling. So one-at-a-time updating reproduces the batch posterior exactly at every step, not merely at the end.

    Order invariance is the same statement. The product $\prod_i f(x_i\mid\theta)$ is over a commutative operation, so any permutation of the observations yields the identical function of $\theta$ and hence the identical posterior. For the exponential families of [Conjugate Priors](04-conjugate-priors.md) this is visible in the update rule itself, which adds $\sum_iT(x_i)$ and $n$ — both symmetric in the data. The multivariate Gaussian instance, where precisions add and the covariance recursion never sees a datum, is derived in [Conditional Gaussian](../part-06-multivariate-probability/06-conditional-gaussian.md).

    The martingale property is a separate consequence of the same setup. By the tower property of conditional expectation,
    $$\mathbb{E}\big[\mathbb{E}[\theta\mid x_{1:t+1}]\;\big|\;x_{1:t}\big]=\mathbb{E}[\theta\mid x_{1:t}],$$
    so $m_t=\mathbb{E}[\theta\mid x_{1:t}]$ is a martingale with respect to the filtration generated by the data, under the joint law of parameter and observations. **You cannot forecast the direction in which your own belief will move: a plan to "update towards a larger edge once more data arrives" is not a plan, it is an inconsistency, because if the revision were predictable it would already have been made.** The same tower property gives the variance decomposition $\mathrm{Var}(\theta)=\mathrm{Var}(m_t)+\mathbb{E}[\mathrm{Var}(\theta\mid x_{1:t})]$, so the spread a belief acquires and the spread it resolves must sum to the prior variance at every $t$.

    The load-bearing condition is the one that did all the work and is never restated: conditional independence given $\theta$. Nothing above used any property of Bayes' rule beyond the factorization of the likelihood. Where the observations are dependent — or, worse, where they are functions of one another — the product form is false, the induction fails at its first step, and the sequential update is not an approximation to the batch posterior but a different and wrong quantity.

## Three Routes to the Same Belief Agree to Eighteen Decimal Places, and the Belief Cannot Anticipate Its Own Revision

Both claims are checkable to the limits of floating point:

```python
import numpy as np

rng = np.random.default_rng(16051)
sig, tau, n, reps = 0.010, 0.0004, 400, 200_000
v0 = tau ** 2

th = rng.normal(0.0, tau, reps)
x = rng.standard_normal((reps, n)) * sig + th[:, None]

prec = 1 / v0 + n / sig ** 2                           # batch, in one step
batch = (x.sum(1) / sig ** 2) / prec
m, p = np.zeros(reps), np.full(reps, 1 / v0)           # one day at a time
for t in range(n):
    p_new = p + 1 / sig ** 2
    m = (m * p + x[:, t] / sig ** 2) / p_new
    p = p_new
perm = rng.permutation(n)                              # the same days, shuffled
ms, ps = np.zeros(reps), np.full(reps, 1 / v0)
for t in perm:
    ps_new = ps + 1 / sig ** 2
    ms = (ms * ps + x[:, t] / sig ** 2) / ps_new
    ps = ps_new

print(f"  a normal-normal update on {n} days, {reps:,} runs; the same evidence applied in"
      f" one step, in order, and in a random order")
print(f"    max |batch - sequential|          {np.abs(batch - m).max():.3e}")
print(f"    max |batch - shuffled|            {np.abs(batch - ms).max():.3e}")
print(f"    posterior precision, all three    {prec:.6f}, {p.max():.6f}, {ps.max():.6f}")

print("  the posterior mean is a martingale, so a belief cannot forecast its own revision")
print("     days t   E[m_t], bp   sd(m_t), bp   E[m_2t - m_t], bp   corr(m_2t - m_t, m_t)"
      "   Var(m_t) + E[post var]   prior var")
for t in (25, 50, 100, 200):
    pt = 1 / v0 + t / sig ** 2
    mt = (x[:, :t].sum(1) / sig ** 2) / pt
    p2 = 1 / v0 + 2 * t / sig ** 2
    m2 = (x[:, :2 * t].sum(1) / sig ** 2) / p2
    tot = mt.var() + 1 / pt
    print(f"    {t:6d}   {mt.mean() * 1e4:10.4f}   {mt.std() * 1e4:11.4f}"
          f"   {(m2 - mt).mean() * 1e4:17.4f}   {np.corrcoef(m2 - mt, mt)[0, 1]:21.4f}"
          f"   {tot * 1e8:22.4f}   {v0 * 1e8:10.4f}")
# =>   a normal-normal update on 400 days, 200,000 runs; the same evidence applied in one step, in order, and in a random order
#        max |batch - sequential|          2.060e-18
#        max |batch - shuffled|            1.952e-18
#        posterior precision, all three    10250000.000000, 10250000.000000, 10250000.000000
#      the posterior mean is a martingale, so a belief cannot forecast its own revision
#         days t   E[m_t], bp   sd(m_t), bp   E[m_2t - m_t], bp   corr(m_2t - m_t, m_t)   Var(m_t) + E[post var]   prior var
#            25       0.0019        0.7838              0.0005                 -0.0042                  15.9990      16.0000
#            50       0.0024        1.0873             -0.0032                  0.0005                  15.9971      16.0000
#           100      -0.0007        1.4838              0.0001                 -0.0016                  15.9948      16.0000
#           200      -0.0007        1.9654             -0.0027                 -0.0019                  15.9842      16.0000
```

The first block is the equivalence. Four hundred days applied in one step, applied one at a time in chronological order, and applied one at a time in a shuffled order produce posterior means differing by at most $2.060\times10^{-18}$ and $1.952\times10^{-18}$ across two hundred thousand runs — floating-point rounding and nothing else — with the posterior precision identical to six decimals in all three. There is no approximation being made and no error accumulating over four hundred sequential steps, which is worth stating because a great many iterative procedures do accumulate error and this one provably cannot.

The second block is the martingale. Over horizons doubling from twenty-five days to two hundred, the expected revision $\mathbb{E}[m_{2t}-m_t]$ measures $0.0005$, $-0.0032$, $0.0001$ and $-0.0027$ basis points against a belief whose own standard deviation is $0.7838$ to $1.9654$ basis points — zero to the resolution available. More telling is the correlation between the revision and the current level: $-0.0042$, $0.0005$, $-0.0016$, $-0.0019$. **Knowing what you currently believe tells you nothing about which way you will move next, which is a genuinely strong property and one that no informal process of "revisiting the thesis" possesses.**

The final two columns are the variance decomposition, and they are the cleanest way to see what updating actually does. The spread of the belief across runs plus the average uncertainty remaining within a run sums to $15.9990$, $15.9971$, $15.9948$ and $15.9842$ against a prior variance of $16.0000$ squared basis points. Learning is conserved: every unit of uncertainty a posterior resolves appears as a unit of variation in where the posterior ends up, and the total was fixed by the prior before any data arrived.

## When Observations Overlap, Sequential Updating Counts the Same Days Repeatedly and the Posterior Reports the Count

The condition the proof leaned on is conditional independence. The most common way to break it is not exotic dependence in the market — it is a transformation the analyst applied themselves:

```python
import numpy as np

rng = np.random.default_rng(16053)
sig, days, reps = 0.010, 1000, 20_000

print(f"  {days} days of independent returns, re-expressed as overlapping k-day sums and"
      f" updated once per day as though each window were fresh evidence, {reps:,} runs")
print("     window k   updates   nominal posterior sd   true sd   ratio   implied n_eff"
      "   honest n_eff   95% coverage")
r = rng.standard_normal((reps, days)) * sig
c = np.concatenate([np.zeros((reps, 1)), r.cumsum(1)], axis=1)
for k in (1, 5, 20, 60):
    w = (c[:, k:] - c[:, :-k]) / k                     # overlapping k-day mean returns
    n = w.shape[1]
    se = sig / np.sqrt(k)                              # sd of one window mean
    post = se / np.sqrt(n)                             # what the iid update reports
    true = w.mean(1).std()
    print(f"    {k:8d}   {n:7d}   {post * 1e4:20.4f}   {true * 1e4:7.4f}"
          f"   {true / post:5.3f}   {n:13d}   {(post / true) ** 2 * n:13.1f}"
          f"   {(np.abs(w.mean(1)) <= 1.959963985 * post).mean():13.4f}")
# =>   1000 days of independent returns, re-expressed as overlapping k-day sums and updated once per day as though each window were fresh evidence, 20,000 runs
#         window k   updates   nominal posterior sd   true sd   ratio   implied n_eff   honest n_eff   95% coverage
#               1      1000                 3.1623    3.1629   1.000            1000           999.6          0.9495
#               5       996                 1.4171    3.1648   2.233             996           199.7          0.6209
#              20       981                 0.7139    3.1810   4.456             981            49.4          0.3388
#              60       941                 0.4209    3.2252   7.663             941            16.0          0.2013
```

The returns here are independent by construction — there is no serial correlation anywhere in the data. The only thing that happens is that the analyst chooses to work with $k$-day windows and to update once per day, which every momentum and trend desk does, because a twenty-day return is the horizon the strategy trades and a daily update is the cadence the risk meeting runs on.

The first row is the control and it behaves: at $k=1$ the nominal posterior standard deviation is $3.1623$ basis points against a true $3.1629$, a ratio of $1.000$, an honest effective sample size of $999.6$ against $1000$ updates, and coverage of $0.9495$. Nothing is wrong when the windows do not overlap.

Everything is wrong as soon as they do. At $k=20$ the belief performs $981$ updates and reports a posterior standard deviation of $0.7139$ basis points; the true sampling standard deviation of that posterior mean is $3.1810$, a ratio of $4.456$, and the honest effective sample size is $49.4$. Coverage of a nominal ninety-five per cent interval is $0.3388$. At $k=60$ the ratio is $7.663$, the effective sample size is $16.0$ against $941$ updates, and coverage is $0.2013$. The honest effective sample size in each case is almost exactly $\text{days}/k$ — a thousand days divided into fifty non-overlapping twenty-day blocks — which is the correct answer and the one the analyst would have got by not overlapping.

**The failure is not that the arithmetic went wrong; it is that a day appearing in twenty windows is treated as twenty independent facts, so the posterior precision grows with the number of updates performed rather than with the amount of evidence acquired.** This is distinct from the misspecification of [Posterior Distributions](03-posterior-distributions.md), which measured the same kind of damage from dependence present in the market itself. Here the data are genuinely independent and the dependence is manufactured by the analyst in a step nobody records as a modelling decision. The tell is that the reported uncertainty *fell* when the window widened — $3.1623$ to $0.4209$ basis points — even though a longer window is unambiguously less informative per update, and no diagnostic in the sequential machinery objects.

## A Belief That Never Forgets Cannot Track a Parameter That Moves, and Forgetting Buys Adaptation at a Permanent Price

Sections 1 and 2 assumed a fixed $\theta$. A trading edge is not fixed, and the coherence that makes updating attractive is exactly what makes an old belief hard to dislodge: evidence accumulated over four years does not evaporate because the last quarter disagreed.

??? note "Proof that discounting the previous precision by $\lambda$ gives an effective window of $1/(1-\lambda)$ and a steady-state posterior standard deviation of $\sigma\sqrt{1-\lambda}$, so an adaptive belief has a floor on its own uncertainty"

    Replace the exact recursion $p_t=p_{t-1}+1/\sigma^{2}$ with the discounted one
    $$p_t=\lambda p_{t-1}+1/\sigma^{2},\qquad 0<\lambda\le1,$$
    which is what raising the previous posterior to the power $\lambda$ does to a Gaussian. Unrolling, $p_t=\sigma^{-2}\sum_{j=0}^{t-1}\lambda^{j}$, so observation $t-j$ enters with weight proportional to $\lambda^{j}$: the belief is an exponentially weighted average whose weights sum to $1/(1-\lambda)$, giving the effective window
    $$n_{\text{eff}}=\sum_{j\ge0}\lambda^{j}=\frac{1}{1-\lambda}.$$
    Taking $t\to\infty$ with $\lambda<1$ gives $p_\infty=1/(\sigma^{2}(1-\lambda))$ and therefore
    $$\mathrm{sd}(\theta\mid\text{data})\ \longrightarrow\ \sigma\sqrt{1-\lambda},$$
    a strictly positive limit. At $\lambda=1$ the sum diverges, $n_{\text{eff}}=\infty$, and the standard deviation decays as $\sigma/\sqrt t$ towards zero.

    Two consequences follow and they are in tension. **An undiscounted belief eventually claims arbitrarily high precision and will therefore require arbitrarily much contrary evidence to move, which is correct when the parameter is constant and catastrophic when it is not.** A discounted belief converges to a fixed uncertainty $\sigma\sqrt{1-\lambda}$ and never becomes more confident than that however long it runs — the price of being able to change its mind is never being allowed to be sure.

    The load-bearing observation is that $\lambda$ is not a free tuning constant but an implicit statement about how fast the parameter moves. A state-space model making that statement explicitly, with parameter innovation variance $Q$ and observation variance $R$, yields a steady-state Kalman gain that is a function of $q=Q/R$ alone, and the correspondence $\lambda=1-K$ converts one language into the other exactly — which is the conversion chart [Particle and Kalman Filters](../../advanced/02-particle-and-kalman-filters.md) prints and [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md) develops as filtering.

The tension is measurable, and which side of it wins is not a matter of taste:

```python
import numpy as np

rng = np.random.default_rng(16055)
sig, pre, post_n, reps = 0.010, 500, 500, 8_000
n = pre + post_n
LAM = (1.00, 0.99, 0.97, 0.90)


def run(x, lam):
    """Exponentially discounted normal update; returns the belief path."""
    m, p = np.zeros(len(x)), np.full(len(x), 1e-12)
    out = np.empty_like(x)
    for t in range(x.shape[1]):
        p = lam * p + 1 / sig ** 2
        m = m + (x[:, t] - m) * (1 / sig ** 2) / p
        out[:, t] = m
    return out, np.sqrt(1 / p)


print(f"  the steady state a forgetting factor buys, independent of any regime change:"
      f" {sig * 1e4:.0f}bp daily noise")
print("     lambda   effective window   steady posterior sd, bp")
for lam in LAM:
    _, sd = run(rng.standard_normal((16, n)) * sig, lam)
    win = 1 / (1 - lam) if lam < 1 else float("inf")
    print(f"    {lam:6.2f}   {win:16.1f}   {sd.mean() * 1e4:23.4f}")

print(f"  a +d edge flipping to -d at day {pre}, {reps:,} runs; tracking RMSE over the"
      f" {post_n} days after the change, and the median days to cross halfway")
print("     shift 2d, bp   RMSE 1.00   RMSE 0.99   RMSE 0.97   RMSE 0.90   best"
      "   halfway 1.00   halfway 0.99   halfway 0.97")
for d_bp in (8.0, 20.0, 40.0, 100.0):
    d = d_bp * 1e-4 / 2
    mu = np.where(np.arange(n) < pre, d, -d)
    x = rng.standard_normal((reps, n)) * sig + mu
    rms, half = [], []
    for lam in LAM:
        m, _ = run(x, lam)
        rms.append(np.sqrt(((m[:, pre:] + d) ** 2).mean()) * 1e4)
        below = m[:, pre:] < 0.0
        half.append(np.median(np.where(below.any(1), below.argmax(1), post_n)))
    b = LAM[int(np.argmin(rms))]
    print(f"    {d_bp:12.1f}   {rms[0]:9.4f}   {rms[1]:9.4f}   {rms[2]:9.4f}"
          f"   {rms[3]:9.4f}   {b:4.2f}   {half[0]:12.1f}   {half[1]:12.1f}"
          f"   {half[2]:12.1f}")
# =>   the steady state a forgetting factor buys, independent of any regime change: 100bp daily noise
#         lambda   effective window   steady posterior sd, bp
#          1.00                inf                    3.1623
#          0.99              100.0                   10.0002
#          0.97               33.3                   17.3205
#          0.90               10.0                   31.6228
#      a +d edge flipping to -d at day 500, 8,000 runs; tracking RMSE over the 500 days after the change, and the median days to cross halfway
#         shift 2d, bp   RMSE 1.00   RMSE 0.99   RMSE 0.97   RMSE 0.90   best   halfway 1.00   halfway 0.99   halfway 0.97
#                 8.0      6.7122      7.4883     12.4037     22.9259   1.00          295.0           20.0            4.0
#                20.0     14.5783      9.4746     12.8419     22.9995   0.99          452.0           48.0           11.0
#                40.0     28.6075     14.4469     14.2736     23.2547   0.97          494.0           61.0           18.0
#               100.0     70.7657     32.0941     21.6651     24.7438   0.97          498.0           67.0           22.0
```

The first table is the proof's arithmetic confirmed. A discount of $0.99$ carries an effective window of $100.0$ days and settles at a posterior standard deviation of $10.0002$ basis points; $0.97$ gives $33.3$ days and $17.3205$; $0.90$ gives $10.0$ days and $31.6228$. Each matches $\sigma\sqrt{1-\lambda}$ exactly. The $\lambda=1.00$ row is the undiscounted belief after a thousand days at $3.1623$ basis points, which is $\sigma/\sqrt{1000}$ and still falling — it has no steady state, which is the whole issue.

The second table sweeps the size of the regime change, and the best discount walks across it. When the edge flips by eight basis points, never forgetting wins outright at a tracking error of $6.7122$ basis points against $7.4883$, $12.4037$ and $22.9259$ — and this is despite taking $295$ days to concede that the sign has changed. At a twenty-basis-point flip $\lambda=0.99$ takes it at $9.4746$; at forty, $\lambda=0.97$ at $14.2736$; at a hundred, $\lambda=0.97$ at $21.6651$ against the undiscounted belief's $70.7657$.

The reason the slow filter wins the first row is that an eight-basis-point shift against a hundred basis points of daily noise is barely estimable in five hundred days at all, so a belief adapting quickly is mostly tracking noise: $\lambda=0.90$ pays a permanent $31.6228$ basis points of uncertainty to chase a signal worth four. **Forgetting is worth its price exactly when the parameter moves by more than the measurement noise would have allowed anyone to resolve, and at the signal-to-noise ratios that daily returns actually offer, that condition is met far less often than the practice of using rolling windows implies.**

The columns on the right show the other half of the trade. The undiscounted belief's median time to concede the sign change runs $295$, $452$, $494$ and $498$ days — getting *worse* as the change gets larger, because a bigger pre-change edge means more accumulated confidence pointing the wrong way. Against that, $\lambda=0.99$ concedes in $20$, $48$, $61$ and $67$ days, and $\lambda=0.97$ in $4$, $11$, $18$ and $22$. A desk that cares about the delay rather than the average error should read the right-hand columns and accept the wider interval knowingly, which is a defensible choice and a different one from minimizing tracking error.

## Coherence Is a Property of a Factorization, and Every Practical Repair Breaks It Deliberately

The results above divide cleanly. Sections 1 and 2 are theorems: exact, unconditional given conditional independence, and stronger than anything an informal review process offers. Sections 3 and 4 are what happens to those theorems in use, and the pattern is that both repairs are deliberate violations of the assumption that made the theorems true.

Discounting is the clearest case. Raising the previous posterior to a power $\lambda<1$ is not derivable from Bayes' rule — there is no prior and no likelihood whose combination produces it — and the resulting sequence of beliefs is not the posterior of any fixed model. It is a power posterior, related to the fractional-likelihood constructions [Posterior Distributions](03-posterior-distributions.md) mentions, and it is properly understood as an approximation to a state-space model in which $\theta$ genuinely evolves. Writing that model down instead makes $\lambda$ a consequence of an explicit belief about parameter drift, which is what the Kalman correspondence delivers and what [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md) sets up. Both routes lead to the same recursion; only the second one lets anybody argue about the number.

The overlap problem of section 3 has a smaller literature and a simpler fix, which is to stop overlapping. Where the overlap is unavoidable — because the horizon genuinely is twenty days and the decision genuinely is daily — the honest options are to update on non-overlapping blocks and interpolate, or to apply the long-run variance correction of [Posterior Distributions](03-posterior-distributions.md), or to divide the reported precision by the overlap factor by hand. All three are outside the framework, which is the recurring theme: the Bayesian calculation contains no term that could detect its own inputs being reused.

!!! note "Updating, filtering, smoothing, online learning and re-estimation are five things a desk calls updating, and they differ in what is assumed to be moving and what is allowed to be revised"
    **Updating** in the sense of this page holds $\theta$ fixed and accumulates evidence about it; the posterior precision only ever rises, and order does not matter. **Filtering** allows $\theta$ to evolve and computes the belief about its value *now* given data up to now, which is the Kalman and hidden-Markov setting where the precision reaches a steady state rather than diverging. **Smoothing** computes the belief about $\theta$ at each past time given *all* the data including later observations, so it revises history and is not available in real time — the course lesson measures a mean gap of $0.088$ between the two on regime probabilities. **Online learning** updates a predictor without any parameter having a distribution at all, and its guarantees are regret bounds against a comparator rather than statements about a posterior. **Re-estimation**, the most common practice, discards the previous belief and refits on a rolling window, which is exponential forgetting with a rectangular kernel and an effective window equal to the window length. Reporting a filtered quantity as though it were an updated one — claiming a precision that a steady state forbids — is the error this list exists to prevent.

!!! warning "A posterior that has stopped moving looks the same whether the evidence has been settled or the evidence has been double-counted, and the second case arrives through a step nobody logged as modelling"
    The failures on this page are silent in a specific way: the reported number moves in the direction that looks like progress. Overlapping twenty-day windows drove the posterior standard deviation from $3.1623$ down to $0.7139$ basis points, which reads as a belief converging nicely, while the true figure was $3.1810$ and coverage was $0.3388$; the sixty-day version reported $0.4209$ against a truth of $3.2252$ and covered on $0.2013$ of runs. In the other direction, an undiscounted belief's confidence rises without limit, so it took $498$ days to concede a hundred-basis-point sign flip, and at no point during those $498$ days did anything in the output indicate a problem. **The free diagnostic is to divide the number of updates you have performed by the number of genuinely non-overlapping observation blocks behind them, and to multiply your reported posterior standard deviation by the square root of that ratio before quoting it — if you update daily on $k$-day windows the factor is $\sqrt{k}$, and if the corrected interval changes a sizing decision, the belief was never entitled to the precision it printed.** For the twenty-day case that factor is $4.47$ against a measured $4.456$, so the back-of-envelope version is right to two decimals.

## Coherence Is Cheap and Conditional Independence Is Not

This page established that sequential updating reproduces the batch posterior exactly at every step and is invariant to the order of the evidence, both following from the likelihood's factorization rather than from Bayes' rule, with batch, chronological and shuffled updates over four hundred days agreeing to $2.060\times10^{-18}$ and $1.952\times10^{-18}$; that the posterior mean is a martingale, its expected revision measuring $0.0005$, $-0.0032$, $0.0001$ and $-0.0027$ basis points with correlations to its own level of $-0.0042$, $0.0005$, $-0.0016$ and $-0.0019$, and the variance decomposition holding at $15.9990$, $15.9971$, $15.9948$ and $15.9842$ against a prior variance of $16.0000$; that overlapping windows destroy all of it, a daily update on twenty-day returns reporting a posterior standard deviation of $0.7139$ basis points against a true $3.1810$, an honest effective sample size of $49.4$ against $981$ updates and coverage of $0.3388$, worsening to $7.663$, $16.0$ and $0.2013$ at sixty days, on data with no serial correlation whatsoever; and that exponential discounting gives an effective window of $1/(1-\lambda)$ and a permanent floor of $\sigma\sqrt{1-\lambda}$ on posterior uncertainty, measured at $10.0002$, $17.3205$ and $31.6228$ basis points, with the best discount running $1.00$, $0.99$, $0.97$ and $0.97$ as the regime shift grows from eight to a hundred basis points while the undiscounted belief's concession time runs $295$, $452$, $494$ and $498$ days.

The shape shared by all three exhibits is that the machinery's guarantees are conditional on a factorization that the analyst controls and rarely inspects. Order invariance and the martingale property are as strong as anything in this part, and both evaporate the moment a day appears in more than one observation — an act performed by a rolling-window transformation, not by the market. What survives is that the framework does exactly what it claims and claims nothing about the provenance of what it is handed.

Every page so far has been about the parameters of a single model. Sections 1 and 3 both used a marginal likelihood without naming it: the normalizing constant that cancelled in the induction, and the quantity by which a mixture's component weights were updated in [Conjugate Priors](04-conjugate-priors.md). Promoted from an accounting constant to the object of interest, it becomes the only device in the framework that can compare one model against another, and it turns out to depend on a prior in a way no amount of data removes. That is [Bayesian Model Comparison](06-bayesian-model-comparison.md).

**Sequential updating is exact, order-free and unable to anticipate itself, and every one of those properties is a statement about a product of densities rather than about the world the densities were fitted to.**
