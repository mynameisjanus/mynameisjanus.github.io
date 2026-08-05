# Markov Chain Monte Carlo

Markov chain Monte Carlo is usually introduced as the method that makes intractable posteriors computable, which is true and describes the wrong thing as the achievement. What it computes was already computable in the two-parameter case; what it changes is the *guarantee*. A grid is wrong by a bounded, computable amount. A chain is right in the limit and carries no finite-sample error bound at all — only diagnostics, every one of them a function of the draws the chain produced, and none of them able to see the region it never entered. Below, four chains reproduce the posterior mean published by [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) — $5.5229$ basis points against a grid answer of $5.5687$, a miss of $1.13$ Monte Carlo standard errors — at an exchange rate of $6.87$ draws for every independent one. That trade is forced rather than chosen: as dimension runs from $2$ to $50$, importance sampling's effective sample size per draw collapses from $0.692041$ to $0.000052$ and its estimate of a five-per-cent tail probability decays to $0.0285$, while the chain's falls only from $0.134915$ to $0.007023$ and its estimate stays at $0.0464$. And then the price. On a bimodal posterior with modes five units apart, four chains started from the larger mode return $\hat R=1.0001$, an effective sample size of $25{,}490$, and a posterior mean of $5.0034$ where the truth is $1.5000$ — wrong by $563.8$ Monte Carlo standard errors, with every diagnostic reporting perfect health.

This page covers the invariance contract a sampler must satisfy and the role of detailed balance in obtaining it, the ergodic average and the central limit theorem that turns integrated autocorrelation time into an exchange rate, why the trade against independent sampling becomes favourable as dimension grows, and what the standard diagnostics can and cannot detect. It builds no transition kernel and derives no acceptance rule, which is [Metropolis–Hastings](05-metropolis-hastings.md); it exploits no conditional structure, which is [Gibbs Sampling](06-gibbs-sampling.md); it derives no importance-sampling estimator and no tilting argument from first principles, which is [Importance Sampling](../part-09-monte-carlo-methods/04-importance-sampling.md); it re-derives no envelope constant and no acceptance collapse, which is [Rejection Sampling](../part-09-monte-carlo-methods/05-rejection-sampling.md); it establishes no $N^{-1/2}$ rate and no grid-cost comparison, which is [Monte Carlo Simulation](../part-09-monte-carlo-methods/03-monte-carlo-simulation.md); it proves no ergodic theorem and derives no effective sample size under dependence from scratch, which is [Random Processes](../part-08-stochastic-processes/01-random-processes.md); it establishes no spectral-gap result for a transition matrix, which is [Markov Chains](../part-08-stochastic-processes/05-markov-chains.md); it normalizes no posterior by quadrature, which is [Numerical Integration](02-numerical-integration.md); it maximizes nothing, which is [Numerical Optimization](01-numerical-optimization.md); it filters no state sequentially and resamples no particle, which is [Particle and Kalman Filters](../../advanced/02-particle-and-kalman-filters.md); and it never reads a diagnostic computed from a sample as evidence about a region that sample does not contain.

The trading stake is a course lesson that states this page's section 4 as an operational rule and then obeys it. [Particle and Kalman Filters](../../advanced/02-particle-and-kalman-filters.md) prints `particle filter (N = 2000): mean ESS 1815, resampled on 91 of 6410 days (1%)` and closes with the reason it bothers: "check the effective sample size in any particle filter — an ESS that collapses toward one means the filter has silently become a single trajectory, and its confidence intervals are fiction." The lesson is describing a diagnostic computed entirely from the particles that survived, monitored because the failure it detects is invisible in the output. Section 4 shows the same diagnostic reporting $25{,}490$ on a chain that has become a single trajectory in exactly that sense.

## Invariance Is the Entire Contract, Detailed Balance Is a Convenient Way to Get It, and Neither Says Anything About a Finite Run

A sampler is a transition kernel $P(\theta,\cdot)$, and the requirement is that the target $\pi$ be a fixed point of it: if $\theta_t\sim\pi$ then $\theta_{t+1}\sim\pi$. Everything else — how fast, from where, with what error — is outside the contract.

??? note "Proof that detailed balance implies invariance but not conversely, so reversibility is a design convenience rather than a requirement, and that invariance alone constrains no finite average"

    A kernel $P$ satisfies **detailed balance** with respect to $\pi$ if $\pi(\theta)P(\theta,\theta')=\pi(\theta')P(\theta',\theta)$ for all $\theta,\theta'$. Integrating both sides over $\theta$,
    $$\int\pi(\theta)P(\theta,\theta')\,\mathrm{d}\theta=\int\pi(\theta')P(\theta',\theta)\,\mathrm{d}\theta=\pi(\theta')\int P(\theta',\theta)\,\mathrm{d}\theta=\pi(\theta'),$$
    since $P(\theta',\cdot)$ is a probability distribution. So $\pi P=\pi$: detailed balance implies invariance. This is the discrete-time counterpart of the argument [Continuous-Time Markov Chains](../part-08-stochastic-processes/06-continuous-time-markov-chains.md) gives for generators, and it is why constructing a reversible kernel is the standard route — the condition is a pointwise identity that can be arranged one pair at a time, whereas invariance is an integral equation.

    The converse fails, and not marginally. Take the deterministic rotation on three states $1\to2\to3\to1$. It leaves the uniform distribution invariant, since a permutation of equally weighted states is uniform again, but $P(1,2)=1$ while $P(2,1)=0$, so detailed balance fails for that pair. A composition of reversible kernels is another example: each factor is reversible, the composition is invariant because invariance is preserved under composition, and the composition is generally not reversible — which is exactly the systematic-scan sampler of [Gibbs Sampling](06-gibbs-sampling.md).

    Now the negative half. Invariance is a statement about a distribution that the chain is not in. Combined with irreducibility and aperiodicity it yields the ergodic theorem — $\frac1N\sum_{t}g(\theta_t)\to\mathbb{E}_\pi[g]$ almost surely, the chain version of the result [Random Processes](../part-08-stochastic-processes/01-random-processes.md) proves — but that is an asymptotic statement with no rate attached. For any $N$, any $\varepsilon$, and any invariant kernel that mixes slowly enough, there are starting distributions under which $\lvert\frac1N\sum_t g(\theta_t)-\mathbb{E}_\pi[g]\rvert>\varepsilon$ with probability arbitrarily close to one. Nothing in the construction forbids it, because the construction only ever constrained the fixed point.

    **The load-bearing asymmetry is that correctness is a property of the kernel and accuracy is a property of the run, so a proof of invariance — which is what every sampler on the next two pages comes with — transfers no information whatever about the finite sample in front of you, and the entire practice of MCMC diagnostics exists to supply what the theorem declines to.**

## A Chain Reproduces a Published Grid Answer to One Monte Carlo Standard Error, at an Exchange Rate of Seven Draws for One

The cleanest way to establish that a sampler works is to point it at a posterior whose answer is already in print. [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) normalized a two-parameter Student-$t$ posterior two independent ways — a $601\times601$ grid and self-normalized importance sampling — and published the results. The same data, the same prior, a third method:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(17041)
n, nu, mu_t, sd_t = 250, 4, 0.0004, 0.010
s_t = sd_t / np.sqrt(nu / (nu - 2))
x = stats.t.rvs(nu, loc=mu_t, scale=s_t, size=n,
                random_state=np.random.default_rng(16031))          # Part XVI page 3


def logpost(th):
    """Student-t log-likelihood plus a Jeffreys 1/s prior, up to a constant."""
    s = np.exp(th[1])
    return -n * th[1] - 0.5 * (nu + 1) * np.log1p(((x - th[0]) / s) ** 2 / nu).sum()


def chain(start, draws, scale):
    """Random-walk Metropolis on (mu, log s); the kernel itself is built on page 05."""
    th = np.array(start, float)
    lp, acc = logpost(th), 0
    out = np.empty((draws, 2))
    jump = scale * rng.standard_normal((draws, 2))
    u = np.log(rng.random(draws))
    for i in range(draws):
        cand = th + jump[i]
        lpc = logpost(cand)
        if u[i] < lpc - lp:
            th, lp, acc = cand, lpc, acc + 1
        out[i] = th
    return out, acc / draws


def tau_int(v, M=400):
    """Integrated autocorrelation time by the initial-positive-sequence rule."""
    v = v - v.mean()
    f = np.fft.rfft(v, 2 * len(v))
    ac = np.fft.irfft(f * np.conj(f))[:M].real
    ac /= ac[0]
    s, k = 1.0, 1
    while k + 1 < M and ac[k] + ac[k + 1] > 0:
        s += 2 * ac[k]
        k += 1
    return s


draws, burn, scale = 40_000, 10_000, np.array([0.0009, 0.075])
starts = [(x.mean(), np.log(x.std())), (0.004, np.log(0.02)),
          (-0.004, np.log(0.006)), (0.0, np.log(0.03))]
paths, accs = zip(*[chain(s, draws, scale) for s in starts])
mu = np.array([p[burn:, 0] for p in paths])

N = mu.shape[1]
W = mu.var(1, ddof=1).mean()
B = mu.mean(1).var(ddof=1) * N
rhat = np.sqrt(((N - 1) / N * W + B / N) / W)
tau = float(np.mean([tau_int(m) for m in mu]))
ess = mu.size / tau
pooled, mcse = mu.mean(), np.sqrt(W / (mu.size / tau))

print(f"  four random-walk Metropolis chains on the two-parameter Student-t posterior of"
      f" Part XVI page 3: {n} returns, {nu} degrees of freedom, Jeffreys prior,"
      f" {draws:,} draws each with the first {burn:,} discarded")
print("     chain   acceptance   posterior mean of mu, bp   posterior sd, bp   P(mu>0)"
      "   integrated autocorrelation time")
for i in range(4):
    print(f"    {i + 1:6d}   {accs[i]:10.4f}   {mu[i].mean() * 1e4:23.4f}"
          f"   {mu[i].std() * 1e4:16.4f}   {(mu[i] > 0).mean():7.4f}"
          f"   {tau_int(mu[i]):31.2f}")
print(f"    pooled over {mu.size:,} draws: mean {pooled * 1e4:.4f}bp,"
      f" sd {mu.ravel().std() * 1e4:.4f}bp, P(mu>0) {(mu > 0).mean():.4f},"
      f" R-hat {rhat:.4f}, tau {tau:.2f}, ESS {ess:,.0f} ({ess / mu.size:.4f} of draws)")
print(f"    the grid answer published in Part XVI page 3 is 5.5687bp, sd 5.3801bp,"
      f" P(mu>0) 0.8457; the chain misses the mean by"
      f" {abs(pooled * 1e4 - 5.5687) / (mcse * 1e4):.2f} Monte Carlo standard errors"
      f" (MCSE {mcse * 1e4:.4f}bp)")
# =>   four random-walk Metropolis chains on the two-parameter Student-t posterior of Part XVI page 3: 250 returns, 4 degrees of freedom, Jeffreys prior, 40,000 draws each with the first 10,000 discarded
#         chain   acceptance   posterior mean of mu, bp   posterior sd, bp   P(mu>0)   integrated autocorrelation time
#             1       0.4027                    5.5503             5.3839    0.8475                              6.53
#             2       0.4039                    5.5964             5.3427    0.8544                              6.74
#             3       0.4011                    5.3830             5.3936    0.8432                              7.59
#             4       0.3989                    5.5618             5.3873    0.8519                              6.61
#        pooled over 120,000 draws: mean 5.5229bp, sd 5.3775bp, P(mu>0) 0.8492, R-hat 1.0001, tau 6.87, ESS 17,474 (0.1456 of draws)
#        the grid answer published in Part XVI page 3 is 5.5687bp, sd 5.3801bp, P(mu>0) 0.8457; the chain misses the mean by 1.13 Monte Carlo standard errors (MCSE 0.0407bp)
```

The last line is the check that matters, and it is a check against a number computed by someone else, by a different method, on a different page. The chain's pooled posterior mean of $5.5229$ basis points sits $1.13$ Monte Carlo standard errors from the grid's $5.5687$; the posterior standard deviation agrees at $5.3775$ against $5.3801$, and $P(\mu>0)$ at $0.8492$ against $0.8457$. Four chains started in wildly different places — one at the sample moments, the others at scales of $2\%$, $0.6\%$ and $3\%$ daily — end up on the same distribution.

The last column is the price, and it is the sentence [Rejection Sampling](../part-09-monte-carlo-methods/05-rejection-sampling.md) wrote in advance: a chain "gives up independence between successive draws." Integrated autocorrelation time runs $6.53$, $6.74$, $7.59$ and $6.61$ across the chains, $6.87$ pooled, which means a draw from this chain is worth $1/6.87$ of an independent draw and $120{,}000$ of them are worth $17{,}474$. **The exchange rate is not a defect to be tuned away; it is what was purchased, and section 3 is about what it bought.** Note also that the effective sample size ratio of $0.1456$ is slightly *better* than the $0.114$ importance sampling achieved on this same posterior in Part XVI — on a two-parameter problem the two methods are simply comparable, which is why nobody would reach for a chain here.

??? note "Proof that the ergodic average obeys a central limit theorem with variance $\sigma^{2}\tau/N$, so integrated autocorrelation time is exactly the exchange rate between dependent and independent draws"

    Let $g$ have finite variance $\sigma^{2}$ under $\pi$ and let the chain be stationary. The variance of the ergodic average is
    $$\mathrm{Var}\Big(\tfrac1N\sum_{t=1}^{N}g(\theta_t)\Big)=\frac{1}{N^{2}}\Big[N\sigma^{2}+2\sum_{k=1}^{N-1}(N-k)\gamma_k\Big]=\frac{\sigma^{2}}{N}\Big[1+2\sum_{k=1}^{N-1}\Big(1-\tfrac kN\Big)\rho_k\Big],$$
    with $\rho_k$ the lag-$k$ autocorrelation of $g(\theta_t)$. If $\sum_k\lvert\rho_k\rvert<\infty$ — which geometric ergodicity supplies — the bracket converges to $\tau=1+2\sum_{k\ge1}\rho_k$, the **integrated autocorrelation time**, giving asymptotic variance $\sigma^{2}\tau/N$. Under geometric ergodicity and $\mathbb{E}_\pi\lvert g\rvert^{2+\delta}<\infty$ the same quantity appears in a central limit theorem, $\sqrt{N}(\bar g_N-\mathbb{E}_\pi g)\Rightarrow N(0,\sigma^{2}\tau)$.

    Setting $\sigma^{2}\tau/N=\sigma^{2}/N_{\text{eff}}$ defines $N_{\text{eff}}=N/\tau$: the **effective sample size** is the number of independent draws that would have delivered the same precision. This is the dependent-data variance inflation of [Random Processes](../part-08-stochastic-processes/01-random-processes.md) applied to a sequence the analyst generated rather than observed, and the Monte Carlo standard error is $\sigma/\sqrt{N_{\text{eff}}}$.

    Two conditions carry the result and both are checked nowhere. Summability of $\rho_k$ fails without geometric ergodicity, which random-walk Metropolis lacks on heavy-tailed targets — the failure [Metropolis–Hastings](05-metropolis-hastings.md) measures. And $\tau$ is estimated from the same chain whose autocorrelations it summarizes, so a chain that has not explored the target estimates $\tau$ for the region it has explored.

    **The load-bearing point is that $\tau$ converts draws into information at a rate the chain itself reports, which makes the Monte Carlo standard error an estimate of the error in an average over a distribution the chain may not be sampling — correct arithmetic applied to the wrong population, with no term anywhere that would notice.**

## The Trade Is Forced by Dimension: Importance Weights Collapse Geometrically Where a Chain Degrades Like One Over d

Section 2 showed a chain matching importance sampling on two parameters, which is not an argument for using one. The argument is what happens when the parameter count is the one a hierarchical or state-space model actually has.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(17043)
DRAWS, CHAINS, BURN, C = 25_000, 4, 5_000, 1.5
TAIL = stats.norm.isf(0.05)


def tau_int(v, M=600):
    v = v - v.mean()
    f = np.fft.rfft(v, 2 * len(v))
    ac = np.fft.irfft(f * np.conj(f))[:M].real
    ac /= ac[0]
    s, k = 1.0, 1
    while k + 1 < M and ac[k] + ac[k + 1] > 0:
        s += 2 * ac[k]
        k += 1
    return s


print(f"  a standardized d-dimensional posterior. Self-normalized importance sampling"
      f" uses a proposal of the right family and centre and {C:.1f} times the width;"
      f" the chain uses a random walk scaled 2.38/sqrt(d). The target is"
      f" P(theta_1 > 1.645) = 0.0500")
print("     d   grid evaluations at 601 per axis   importance: ESS/draws"
      "   importance: P(theta_1>1.645)   chain: acceptance   chain: ESS/draws"
      "   chain: P(theta_1>1.645)")
for d in (2, 5, 10, 25, 50):
    N = DRAWS * CHAINS
    q = C * rng.standard_normal((N, d))
    lw = (-0.5 * (q ** 2).sum(1)) - (-0.5 * (q ** 2).sum(1) / C ** 2 - d * np.log(C))
    w = np.exp(lw - lw.max())
    w /= w.sum()
    ess_is = 1.0 / (w @ w)
    p_is = w[q[:, 0] > TAIL].sum()

    scale = 2.38 / np.sqrt(d)
    th = rng.standard_normal((CHAINS, d)) * 2.0
    lp = -0.5 * (th ** 2).sum(1)
    keep = np.empty((CHAINS, DRAWS))
    acc = 0
    for i in range(DRAWS):
        cand = th + scale * rng.standard_normal((CHAINS, d))
        lpc = -0.5 * (cand ** 2).sum(1)
        take = np.log(rng.random(CHAINS)) < lpc - lp
        th = np.where(take[:, None], cand, th)
        lp = np.where(take, lpc, lp)
        acc += take.sum()
        keep[:, i] = th[:, 0]
    keep = keep[:, BURN:]
    tau = float(np.mean([tau_int(k) for k in keep]))
    print(f"    {d:2d}   {601.0 ** d:32.2e}   {ess_is / N:21.6f}"
          f"   {p_is:29.4f}   {acc / (DRAWS * CHAINS):18.4f}"
          f"   {1 / tau:17.6f}   {(keep > TAIL).mean():23.4f}")
# =>   a standardized d-dimensional posterior. Self-normalized importance sampling uses a proposal of the right family and centre and 1.5 times the width; the chain uses a random walk scaled 2.38/sqrt(d). The target is P(theta_1 > 1.645) = 0.0500
#         d   grid evaluations at 601 per axis   importance: ESS/draws   importance: P(theta_1>1.645)   chain: acceptance   chain: ESS/draws   chain: P(theta_1>1.645)
#         2                           3.61e+05                0.692041                          0.0495               0.3575            0.134915                    0.0507
#         5                           7.84e+13                0.398052                          0.0500               0.2865            0.055742                    0.0497
#        10                           6.15e+27                0.158109                          0.0499               0.2610            0.033204                    0.0518
#        25                           2.96e+69                0.009939                          0.0422               0.2453            0.012213                    0.0605
#        50                          8.78e+138                0.000052                          0.0285               0.2406            0.007023                    0.0464
```

Read the table across, then down. Across at $d=2$, importance sampling is the better method by a factor of five — effective sample size per draw of $0.692041$ against the chain's $0.134915$ — and the grid is entirely feasible at $3.61\times10^{5}$ evaluations. Everything the appendix has done before this part lives in that row.

Down the importance column is a geometric collapse: $0.692041$, $0.398052$, $0.158109$, $0.009939$, $0.000052$. That is a factor of $13{,}308$ across the sweep, and the proposal being used is a good one — the right family, the right centre, uniformly fifty per cent too wide. It is the mechanism [Rejection Sampling](../part-09-monte-carlo-methods/05-rejection-sampling.md) prices as $c^{d}$, arriving in the weights instead of the acceptance rate. The consequence is in the next column: the tail probability, whose true value is $0.0500$, is estimated at $0.0495$, $0.0500$ and $0.0499$ while there are weights to work with, and then at $0.0422$ and $0.0285$ once there are not. **At fifty dimensions the estimate is $43\%$ low, and it is low rather than noisy, because the handful of draws carrying the weight are not the draws in the tail.**

Down the chain column is a decline and not a collapse: $0.134915$, $0.055742$, $0.033204$, $0.012213$, $0.007023$, a factor of $19$ over the same sweep in which importance sampling lost $13{,}308$. The tail probability holds at $0.0507$, $0.0497$, $0.0518$, $0.0605$, $0.0464$. At $d=50$ the chain's effective sample size per draw exceeds importance sampling's by a factor of $135$, and the grid needs $8.78\times10^{138}$ evaluations, which is not a large number but a meaningless one.

The acceptance column is the reason and it is the one column nobody set. Scaling the proposal as $2.38/\sqrt d$, the acceptance rate settles at $0.3575$, $0.2865$, $0.2610$, $0.2453$, $0.2406$ — converging toward $0.234$ from above, without that number appearing anywhere in the code. **A chain can hold its acceptance rate constant in any dimension by shrinking its steps, paying for it in autocorrelation at a polynomial rate; a weight or an envelope cannot, and pays exponentially. That is the whole trade, and it is forced rather than chosen.** Why $0.234$ and what it costs is [Metropolis–Hastings](05-metropolis-hastings.md).

## Every Diagnostic Is Computed From Where the Chain Has Been, So a Chain That Never Left One Mode Reports Perfect Health

Sections 2 and 3 measured a chain doing its job. The characteristic MCMC failure is not slow mixing, which is visible, but the absence of mixing between regions, which is not.

??? note "Proof that the between-chain variance estimates the target variance only when the starting values are at least as dispersed as the target, so $\hat R$ measures the initialization whenever the chain has not mixed"

    The Gelman–Rubin statistic compares the pooled variance estimate against the within-chain estimate. With $m$ chains of length $N$, writing $W$ for the mean of the within-chain variances and $B/N$ for the variance of the chain means,
    $$\hat R^{2}=\frac{\frac{N-1}{N}W+\frac{1}{N}B}{W}.$$
    Suppose the chains have equilibrated. Then each chain's draws are from $\pi$, so $W\to\mathrm{Var}_\pi$, the chain means are independent estimates of $\mathbb{E}_\pi$ with variance $\mathrm{Var}_\pi\tau/N$, so $B\to\mathrm{Var}_\pi\tau$, and $\hat R^{2}\to1+(\tau-1)/N\to1$. Convergence drives the statistic to one, as intended.

    Now suppose chain $j$ has equilibrated within a region $A_j$ and never leaves it. Then $W\to\sum_j\mathrm{Var}(\pi\mid A_j)/m$, the within-region variance, and $B\to N\cdot\mathrm{Var}_j(\mathbb{E}[\pi\mid A_j])$, the spread of the *regions' means*. So
    $$\hat R^{2}\approx1+\frac{\mathrm{Var}_j\big(\mathbb{E}[\pi\mid A_j]\big)}{\overline{\mathrm{Var}(\pi\mid A_j)}}.$$
    The numerator is a variance over the chains' *starting regions*. If every chain was started in the same region, $A_1=\dots=A_m$, the numerator is zero and $\hat R\to1$ exactly — the statistic certifies convergence to a distribution the chain never sampled. If the starts were spread across regions in proportion to $\pi$, the numerator recovers the between-region variance and $\hat R$ is large.

    So $\hat R$ tests whether the chains *agree*, which coincides with whether they have *converged* only if the initial disagreement was at least as large as the target's own spread. That condition is a property of the starting values, chosen by the analyst before any data was seen.

    **The load-bearing consequence is that $\hat R$'s power is supplied entirely by the initialization and not by the data or the chain, so overdispersed starts are not a stylistic recommendation but the entire content of the diagnostic, and a set of chains launched from one optimizer's answer has removed the only information the statistic ever had.**

The construction below is the smallest posterior that is genuinely bimodal — two explanations of the same track record, one that the edge is real and one that it is not:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(17045)
DRAWS, BURN, CHAINS, W = 50_000, 10_000, 4, 0.35
SCALE = 1.2


def logpost(t, a):
    """Two explanations of the same record: a losing regime and a winning one."""
    return np.logaddexp(np.log(W) - 0.5 * (t + a) ** 2,
                        np.log1p(-W) - 0.5 * (t - a) ** 2)


def run(a, starts):
    th = np.array(starts, float)
    lp = logpost(th, a)
    out = np.empty((CHAINS, DRAWS))
    for i in range(DRAWS):
        cand = th + SCALE * rng.standard_normal(CHAINS)
        lpc = logpost(cand, a)
        take = np.log(rng.random(CHAINS)) < lpc - lp
        th, lp = np.where(take, cand, th), np.where(take, lpc, lp)
        out[:, i] = th
    return out[:, BURN:]


def diag(k):
    n = k.shape[1]
    Wv = k.var(1, ddof=1).mean()
    B = k.mean(1).var(ddof=1) * n
    rhat = np.sqrt(((n - 1) / n * Wv + B / n) / Wv)
    f = np.fft.rfft(k[0] - k[0].mean(), 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:800].real
    ac /= ac[0]
    s, j = 1.0, 1
    while j + 1 < 800 and ac[j] + ac[j + 1] > 0:
        s += 2 * ac[j]
        j += 1
    return rhat, k.size / s, k.mean()


print(f"  a bimodal posterior for a strategy's edge: weight {W} on a mode at -a and"
      f" {1 - W:.2f} on a mode at +a, in units of the posterior's own scale."
      f" {CHAINS} chains, {DRAWS:,} draws each, first {BURN:,} discarded")
print("     separation   true mean   all chains from the right mode: R-hat   ESS"
      "   mean   error in MCSE   chains that ever crossed"
      "   dispersed starts: R-hat   mean")
for a in (1.5, 2.5, 3.5, 5.0):
    truth = (1 - 2 * W) * a
    one = run(a, [a] * CHAINS)
    r1, e1, m1 = diag(one)
    crossed = int(((one < 0).any(1)).sum())
    two = run(a, [-a, a, -a, a])
    r2, e2, m2 = diag(two)
    mcse = one.std() / np.sqrt(e1)
    print(f"    {a:10.1f}   {truth:9.4f}   {r1:37.4f}   {e1:5.0f}   {m1:6.4f}"
          f"   {abs(m1 - truth) / mcse:13.1f}   {crossed:24d}"
          f"   {r2:25.4f}   {m2:6.4f}")
# =>   a bimodal posterior for a strategy's edge: weight 0.35 on a mode at -a and 0.65 on a mode at +a, in units of the posterior's own scale. 4 chains, 50,000 draws each, first 10,000 discarded
#         separation   true mean   all chains from the right mode: R-hat   ESS   mean   error in MCSE   chains that ever crossed   dispersed starts: R-hat   mean
#               1.5      0.4500                                  1.0000   10298   0.4580             0.5                          4                      1.0000   0.4455
#               2.5      0.7500                                  1.0010    2395   0.7718             0.4                          4                      1.0004   0.7443
#               3.5      1.0500                                  1.0054     316   1.0697             0.1                          4                      1.0035   1.2428
#               5.0      1.5000                                  1.0001   25490   5.0034           563.8                          0                      3.0048   -0.3029
```

The first three rows are a chain working and visibly straining. As the modes separate, crossings become rare and the effective sample size falls from $10{,}298$ to $2{,}395$ to $316$ — the last of these on $160{,}000$ retained draws, an autocorrelation time above five hundred. All four chains still cross in every one of these rows, the estimates land within half a Monte Carlo standard error of truth, and $\hat R$ reads $1.0000$, $1.0010$, $1.0054$. A low effective sample size is the honest symptom of a hard problem.

The fourth row is the failure and every number in it is reassuring. With the modes five units apart, $0$ of $4$ chains ever crossed. $\hat R$ reads $1.0001$. The effective sample size reads $25{,}490$ — **eighty times larger than the row above it, and larger because the chain stopped attempting the hard part.** Confined to one mode the chain mixes beautifully, its autocorrelation is short, its draws look independent, and every quantity computed from them is an accurate summary of the wrong distribution. The posterior mean comes out at $5.0034$ where the truth is $1.5000$, an error of $563.8$ Monte Carlo standard errors, which is to say the reported uncertainty and the actual error are unrelated quantities.

The last two columns are the proof cashed out. Restarting the same sampler with two chains in each mode drives $\hat R$ to $3.0048$, and it does so instantly and unambiguously. Nothing about the target, the kernel, or the data changed; the only difference is where four numbers were initialized. **$\hat R$ detected the failure using information the analyst supplied and the chain could not have generated, which is why it is a diagnostic rather than a test.** It is also worth reading the dispersed run's posterior mean, $-0.3029$ against a truth of $1.5000$: the diagnostic fires correctly and the estimate remains wrong, because detecting that two modes exist is not the same as knowing their relative mass. Fixing that needs a kernel that can move between them, which is a modelling decision, not a diagnostic one.

## The Diagnostics Are Worth Running and None of Them Is a Proof

Everything above argues for scepticism rather than despair, and the working practice is well established. **Warm-up** — the modern name for burn-in, and better, since the discarded phase is also where step sizes get tuned — removes dependence on the starting point, and there is no test for how long it should be; the honest procedure is to discard half and check that the second half's conclusions do not change. **Thinning** is almost always a mistake: keeping every tenth draw reduces autocorrelation per retained draw and reduces effective sample size in absolute terms, so it is justified only when storage rather than computation is binding. Beyond $\hat R$ and effective sample size, the rank-normalized and split versions of $\hat R$ catch a chain whose two halves differ in variance rather than location, and bulk and tail effective sample sizes are reported separately because a chain can mix well in the body and badly in the tail — which is where a risk number lives.

The theory behind the practice is thinner than the practice suggests. **Geometric ergodicity** — convergence to $\pi$ at a rate $C\rho^{t}$ — is what licenses the central limit theorem in section 2's proof, and it is a property of the kernel and target jointly that is almost never verified for a real model; the spectral gap of [Markov Chains](../part-08-stochastic-processes/05-markov-chains.md) is its finite-state ancestor. What is used instead is **Hamiltonian Monte Carlo**, which exploits the gradient of the log-posterior to propose distant states with high acceptance, and its adaptive form NUTS, which is the default in every probabilistic programming system and is not built on this page. Its most useful contribution is a genuinely different diagnostic: divergent transitions report a geometry the sampler could not integrate, which is information about the target rather than about the sample. The deterministic alternative — fitting a tractable family to the posterior by minimizing a divergence rather than sampling it — is variational inference, the same bound [The EM Algorithm](03-em-algorithm.md) maximizes with the E-step relaxed, and it trades this page's asymptotic exactness for a bias that no diagnostic reports.

!!! note "The stationary distribution, the chain's distribution at step $t$, the empirical distribution of the retained draws, and the sampling distribution of the ergodic average are four different objects, and every diagnostic examines the third"
    They are conflated because in the limit the first three coincide, which is the entire theory. The **stationary distribution** $\pi$ is what the kernel was built to leave invariant, and it is the only one the correctness proof mentions. The **distribution at step $t$** is $\pi_0P^{t}$, which depends on the starting point forever at any finite $t$ and is what warm-up exists to make irrelevant; the rate at which it approaches $\pi$ is the mixing time, and it is not estimable from one run. The **empirical distribution of the draws** is the histogram in front of you, and it is what $\hat R$, effective sample size, trace plots and every posterior summary are computed from — section 4 shows it can be a faithful picture of one mode and silent about another. The **sampling distribution of the ergodic average** is the object the Monte Carlo standard error describes, and it exists only under the geometric ergodicity that licenses section 2's central limit theorem, which [Metropolis–Hastings](05-metropolis-hastings.md) shows can fail outright on heavy-tailed targets. Reading a Monte Carlo standard error as the uncertainty in a posterior quantity is what this list exists to prevent: it measures the error in an average over the third object, not the distance between the third and the first.

!!! warning "A chain that has not visited a region produces diagnostics that are silent about it, and the strongest of those diagnostics draws its power from the starting values rather than from the data"
    Nothing in the failing case looked wrong. Section 4's fourth row returned $\hat R=1.0001$, an effective sample size of $25{,}490$ — eighty times the row above, where the sampler was working — and a posterior mean in error by $563.8$ Monte Carlo standard errors. The effective sample size was high *because* the failure was total: a chain confined to one mode has short autocorrelation and looks like an excellent chain for the distribution it is actually sampling. Section 3 shows the neighbouring trap, where importance sampling's tail estimate decays to $0.0285$ against a truth of $0.0500$ while its effective sample size of $0.000052$ per draw is the only warning issued. **The free diagnostic is to initialize your chains from the several optima a randomized-restart optimizer already found — the exhibit in [Numerical Optimization](01-numerical-optimization.md) produces exactly that list as a by-product — rather than from perturbations of a single point, because a set of starts that already disagrees is the only input $\hat R$ has that the chain cannot manufacture; and where any one-dimensional marginal is cheap, integrate it on a grid and compare the answer, since a grid in one dimension costs nothing and cannot miss a mode.** Both are free, and between them they convert the page's one undetectable failure into a detectable one.

## A Guarantee About Infinity, Reported at Step Fifty Thousand

This page established that invariance is the whole of a sampler's contract and detailed balance a sufficient rather than necessary route to it, so a correctness proof transfers no information about a finite run; that a chain reproduces an independently published grid answer on a two-parameter Student-$t$ posterior, $5.5229$ basis points against $5.5687$ at a miss of $1.13$ Monte Carlo standard errors, with $\hat R=1.0001$ and an integrated autocorrelation time of $6.87$ that converts $120{,}000$ draws into $17{,}474$; that the trade is forced by dimension, importance sampling's effective sample size per draw collapsing $0.692041$, $0.398052$, $0.158109$, $0.009939$, $0.000052$ as $d$ runs $2$ to $50$ while its tail estimate decays to $0.0285$ against a truth of $0.0500$, and the chain's falling only $0.134915$ to $0.007023$ with the estimate holding at $0.0464$ and acceptance settling at $0.2406$ without being asked; and that on a bimodal posterior four chains started from one mode return $\hat R=1.0001$, an effective sample size of $25{,}490$ — eighty times the value from the harder row where mixing genuinely occurred — and a posterior mean of $5.0034$ against a truth of $1.5000$, an error of $563.8$ Monte Carlo standard errors, which dispersed starts expose immediately at $\hat R=3.0048$ without correcting the estimate.

The shape shared by all three exhibits is that every quantity a chain reports is an average over the draws it produced, and the failure mode is a discrepancy between those draws and the target that no such average can contain. In section 2 there is no discrepancy and every number is trustworthy. In section 3 the discrepancy is visible, because effective sample size falls as the weights concentrate and the warning arrives before the estimate does. In section 4 the discrepancy is invisible, because a chain confined to a region is a well-behaved chain for that region, and the one statistic that detects it is powered by a choice made before the sampler started. This is the same structural absence [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) found in a posterior's width and [Numerical Integration](02-numerical-integration.md) found in an integrator's error estimate: no term computed from the method's disagreement with what it is approximating.

What this page never did is build the kernel. Every exhibit above called a random-walk Metropolis step as a black box, tuned its scale by hand, and treated the acceptance rate as an output. That rule is one line, it is correct for any proposal whose support covers the target, and its correctness is entirely independent of the tuning parameter that section 3 showed varying the efficiency by two orders of magnitude. Both halves of that sentence deserve a proof and the second deserves a measurement, which is [Metropolis–Hastings](05-metropolis-hastings.md).

**A Markov chain guarantees that the average converges and reports how precise that average is by measuring the draws it has already taken, so the guarantee is about a limit nobody reaches and the precision is about a distribution nobody has verified is the target.**
