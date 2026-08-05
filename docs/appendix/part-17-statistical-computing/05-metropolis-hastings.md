# Metropolis–Hastings

Metropolis–Hastings is usually presented as the algorithm that works for any target known up to a constant, which is exactly true and is the least interesting thing about it. The correctness proof holds for every proposal with the right support, and it says nothing whatever about the free parameter the proposal contains — so the guarantee is uniform across settings that differ in cost by three orders of magnitude. Below, the same chain on the same posterior returns the same answer at seven proposal scales spanning five orders of magnitude, every one of them within $2.4$ Monte Carlo standard errors of a published grid result, while the effective sample size per draw runs from $0.14598$ down to $0.00019$ — a factor of $768$ between the best setting and the worst, with the two worst sitting at opposite ends and both accepting almost never or almost always. The optimum is not a free choice either: as dimension grows the best acceptance rate falls to $0.2398$ against the asymptotic $0.234$, and an isotropic proposal on a target whose inverse covariance has condition number $1{,}000$ gives up a factor of $44.8$ that dividing each coordinate by its own standard deviation recovers entirely. Then the failure, and it is not a tuning failure: an independence sampler with a Gaussian proposal on a Student-$t$ posterior accepts $0.8744$ of the time, which reads as a healthy chain, while one of its runs sat frozen at a single value for $9{,}708$ consecutive draws, its effective sample size per draw is $0.01684$, its error bar is $3.74$ times too small, and its nominal ninety-five per cent interval covers $0.5433$.

This page covers the acceptance ratio and the reason the normalizing constant cancels, the separation between correctness and efficiency and the size of the gap, optimal scaling in dimension and the preconditioning that removes an ill-conditioned target's penalty, and the condition on a proposal's tails that decides whether any finite run is usable. It establishes no invariance theory and proves no ergodic theorem, which is [Markov Chain Monte Carlo](04-markov-chain-monte-carlo.md); it exploits no full conditionals and builds no rejection-free kernel, which is [Gibbs Sampling](06-gibbs-sampling.md); it derives no importance-sampling estimator, though section 4's condition is the same one, which is [Importance Sampling](../part-09-monte-carlo-methods/04-importance-sampling.md); it constructs no envelope for independent draws, which is [Rejection Sampling](../part-09-monte-carlo-methods/05-rejection-sampling.md); it normalizes no posterior and computes no evidence, which are [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) and [Bayesian Model Comparison](../part-16-bayesian-statistics/06-bayesian-model-comparison.md); it derives no preconditioning result for an optimizer, though section 3 reuses it, which is [Numerical Optimization](01-numerical-optimization.md); it evaluates no integral by quadrature, which is [Numerical Integration](02-numerical-integration.md); it maximizes no likelihood over latent variables, which is [The EM Algorithm](03-em-algorithm.md); it derives no spectral gap for a transition matrix, which is [Markov Chains](../part-08-stochastic-processes/05-markov-chains.md); it establishes no properties of the Student-$t$ family it samples, which is [Student's t Distribution](../part-05-common-distributions/16-students-t-distribution.md); and it never reads an acceptance rate as a statement about whether a chain is working.

The trading stake is a course lesson that refuses to report a single run of a stochastic algorithm and explains why in the same breath. [Reinforcement Learning and Meta-Labeling](../../part-07-machine-learning/04-reinforcement-learning-and-meta-labeling.md) prints `2017+ Sharpe     : median +0.20, range [-0.50, +0.74], 12/20 positive` and reads the spread rather than the best number: "the difference between the best and worst 'agent' is a *lottery drawn inside the optimizer*," and "a single RL run is an anecdote." Twenty seeds were run because one would have been unfalsifiable. Section 4 is the same discipline applied to a sampler, where the spread across chains is $3.74$ times what any single chain reports as its own error.

## The Acceptance Ratio Is Built So the Normalizing Constant Cancels, and Correctness Holds for Any Proposal Whose Support Covers the Target

The algorithm is three lines. From the current state $\theta$, draw a candidate $\theta'$ from a proposal $q(\cdot\mid\theta)$; compute
$$\alpha=\min\Big(1,\ \frac{\pi(\theta')q(\theta\mid\theta')}{\pi(\theta)q(\theta'\mid\theta)}\Big);$$
with probability $\alpha$ move to $\theta'$, and otherwise stay where you are and record the current state again. The second half of that last clause is where the dependence comes from and where half of section 2's failures live.

??? note "Proof that the Metropolis–Hastings kernel satisfies detailed balance with respect to $\pi$ for any proposal with the right support, so correctness is independent of the proposal, and that the ratio needs $\pi$ only up to a constant"

    Write the off-diagonal part of the kernel as $P(\theta,\theta')=q(\theta'\mid\theta)\alpha(\theta,\theta')$ for $\theta'\ne\theta$, with the remaining mass placed on staying. For any pair, suppose without loss of generality that $\pi(\theta)q(\theta'\mid\theta)\ge\pi(\theta')q(\theta\mid\theta')$, so that $\alpha(\theta,\theta')=\pi(\theta')q(\theta\mid\theta')/[\pi(\theta)q(\theta'\mid\theta)]$ and $\alpha(\theta',\theta)=1$. Then
    $$\pi(\theta)P(\theta,\theta')=\pi(\theta)q(\theta'\mid\theta)\cdot\frac{\pi(\theta')q(\theta\mid\theta')}{\pi(\theta)q(\theta'\mid\theta)}=\pi(\theta')q(\theta\mid\theta')=\pi(\theta')P(\theta',\theta),$$
    which is detailed balance, and hence invariance by the argument in [Markov Chain Monte Carlo](04-markov-chain-monte-carlo.md). The diagonal contributes to both sides equally and is irrelevant. Nothing in the calculation constrained $q$ beyond it being a density: the $\min$ is precisely the device that forces the identity to hold whichever way the inequality ran.

    Two consequences follow immediately. First, $\pi$ enters only through the ratio $\pi(\theta')/\pi(\theta)$, so an unnormalized $q(\theta)=\pi(\theta)Z$ gives the identical ratio for every $Z>0$: the normalizing constant that [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) called "the only expensive step" is never computed, never estimated, and never needed. That page's closing instruction — "give up on $Z$ entirely and construct a chain whose stationary distribution is the posterior" — is discharged by one cancellation.

    Second, for a symmetric proposal $q(\theta'\mid\theta)=q(\theta\mid\theta')$ the proposal terms cancel too and $\alpha=\min(1,\pi(\theta')/\pi(\theta))$, which is the original Metropolis rule: always accept an uphill move, accept a downhill move with probability equal to the ratio. The requirement for irreducibility is that $q$ can eventually reach any region of positive $\pi$-mass, which every proposal below satisfies.

    **The load-bearing asymmetry is that the proof constrains the proposal only through its support, so every setting of every tuning parameter yields a correct algorithm and the proof therefore contains no information about which to choose — the entire practical content of the method lives in a quantity its correctness theorem is indifferent to.**

## Across Five Orders of Magnitude of Step Size Every Answer Is Right and the Cost of Obtaining It Varies by a Factor of Seven Hundred

The claim that correctness is independent of the proposal is checkable, and the check is more interesting when the answer is known in advance. The target is the Student-$t$ posterior [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) normalized on a grid, whose posterior mean is $5.5687$ basis points.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(17051)
n, nu, mu_t, sd_t = 250, 4, 0.0004, 0.010
s_t = sd_t / np.sqrt(nu / (nu - 2))
x = stats.t.rvs(nu, loc=mu_t, scale=s_t, size=n,
                random_state=np.random.default_rng(16031))          # Part XVI page 3
SCALE0 = np.array([0.0009, 0.075])                                  # the tuned step


def logpost(th):
    """Student-t log-likelihood plus a Jeffreys 1/s prior, up to a constant."""
    s = np.exp(th[..., 1])
    return (-n * th[..., 1]
            - 0.5 * (nu + 1) * np.log1p(((x - th[..., 0, None]) / s[..., None]) ** 2
                                        / nu).sum(-1))


def mh(scale, draws, chains=4):
    """Random-walk Metropolis: propose, compute one ratio, accept or repeat the state."""
    th = np.tile([x.mean(), np.log(x.std())], (chains, 1))
    lp = logpost(th)
    out = np.empty((chains, draws))
    acc = 0
    for i in range(draws):
        cand = th + scale * rng.standard_normal((chains, 2))
        lpc = logpost(cand)
        take = np.log(rng.random(chains)) < lpc - lp
        th = np.where(take[:, None], cand, th)
        lp = np.where(take, lpc, lp)
        acc += take.sum()
        out[:, i] = th[:, 0]
    return out, acc / (draws * chains)


def tau_int(v, M=4_000):
    v = v - v.mean()
    f = np.fft.rfft(v, 2 * len(v))
    ac = np.fft.irfft(f * np.conj(f))[:M].real
    ac /= ac[0]
    s, k = 1.0, 1
    while k + 1 < M and ac[k] + ac[k + 1] > 0:
        s += 2 * ac[k]
        k += 1
    return s


DRAWS, BURN = 30_000, 6_000
print(f"  random-walk Metropolis on the Part XVI page 3 posterior at proposal scales"
      f" spanning five orders of magnitude, 4 chains x {DRAWS:,} draws, first {BURN:,}"
      f" discarded; the grid answer for the posterior mean is 5.5687bp")
print("     step, x tuned   acceptance   tau   ESS   ESS/draws   posterior mean, bp"
      "   MCSE, bp   error in MCSE")
for mult in (0.003, 0.03, 0.3, 1.0, 3.0, 30.0, 300.0):
    out, acc = mh(SCALE0 * mult, DRAWS)
    k = out[:, BURN:]
    tau = float(np.mean([tau_int(c) for c in k]))
    ess = k.size / tau
    mcse = k.std() / np.sqrt(ess)
    err = abs(k.mean() * 1e4 - 5.5687) / (mcse * 1e4)
    print(f"    {mult:13.3f}   {acc:10.4f}   {tau:5.1f}   {ess:5.0f}"
          f"   {ess / k.size:9.5f}   {k.mean() * 1e4:18.4f}   {mcse * 1e4:8.4f}"
          f"   {err:13.2f}")
# =>   random-walk Metropolis on the Part XVI page 3 posterior at proposal scales spanning five orders of magnitude, 4 chains x 30,000 draws, first 6,000 discarded; the grid answer for the posterior mean is 5.5687bp
#         step, x tuned   acceptance   tau   ESS   ESS/draws   posterior mean, bp   MCSE, bp   error in MCSE
#                0.003       0.9948   4695.4      20     0.00021               6.3135     0.8046            0.93
#                0.030       0.9746   1586.8      60     0.00063               5.8194     0.7112            0.35
#                0.300       0.7819    26.3    3652     0.03804               5.6714     0.0884            1.16
#                1.000       0.4040     6.9   14014     0.14598               5.6772     0.0458            2.37
#                3.000       0.0897    19.4    4944     0.05150               5.4798     0.0770            1.15
#               30.000       0.0011   1212.4      79     0.00082               5.2086     0.6255            0.58
#              300.000       0.0000   5293.2      18     0.00019               1.4749     1.8367            2.23
```

The right-hand column is the theorem. Across a step size varying by a factor of a hundred thousand, the error against the published grid answer reads $0.93$, $0.35$, $1.16$, $2.37$, $1.15$, $0.58$ and $2.23$ Monte Carlo standard errors — every row consistent with the truth, including the row whose posterior mean is $1.4749$ basis points against a truth of $5.5687$, because that row's own error bar is $1.8367$ basis points and honestly says so. **A badly tuned Metropolis chain does not lie; it reports a wide interval and the interval contains the answer.**

The middle columns are the cost. Effective sample size per draw peaks at $0.14598$ and falls to $0.00021$ and $0.00019$ at the two extremes, a factor of $768$ from best to worst, and the two extremes fail for opposite reasons. At a step three thousand times too small the chain accepts $0.9948$ of proposals and goes nowhere, because each accepted move is negligible; the autocorrelation time is $4695.4$. At a step three hundred times too large it accepts $0.0000$ to four decimals and goes nowhere, because it is almost never permitted to move; the autocorrelation time is $5293.2$. **Both pathologies produce a chain that repeats itself, and the acceptance rate is the only thing that distinguishes them — which is why acceptance is the standard tuning target and why section 4 is worth reading before trusting it.**

The optimum at $0.4040$ acceptance is not a coincidence and not a property of this posterior.

## The Best Acceptance Rate Falls to 0.234 as Dimension Grows, and an Isotropic Proposal on an Ill-Conditioned Target Loses Efficiency Like the Square Root of the Condition Number

The tuning problem has a known answer in the limit, and it has a known failure mode that page 1 already measured for a different algorithm.

??? note "Proof sketch that the optimal random-walk scale is $\ell/\sqrt d$ with acceptance tending to $0.234$, so the cost per effective draw grows linearly in dimension rather than exponentially"

    Take a product target $\pi(\theta)=\prod_{i=1}^{d}f(\theta_i)$ and a proposal $\theta'=\theta+(\ell/\sqrt d)\,Z$ with $Z$ standard normal. The log acceptance ratio is a sum of $d$ independent terms,
    $$\log\frac{\pi(\theta')}{\pi(\theta)}=\sum_{i=1}^{d}\Big[\log f(\theta_i+\tfrac{\ell}{\sqrt d}Z_i)-\log f(\theta_i)\Big],$$
    each of size $O(d^{-1/2})$ with a second-order term of size $O(d^{-1})$. A central limit theorem applies: the sum converges to $N(-\tfrac{\ell^{2}I}{2},\ \ell^{2}I)$ where $I=\mathbb{E}[(\log f)'^{2}]$, a Gaussian whose mean is exactly minus half its variance — the balance that keeps the acceptance probability at an $O(1)$ limit rather than at $0$ or $1$. Taking expectations of $\min(1,e^{W})$ for such a $W$ gives an average acceptance of $2\Phi(-\ell\sqrt I/2)$.

    Each coordinate then behaves, on the time scale $t=k/d$, like a diffusion with speed $h(\ell)=\ell^{2}\cdot2\Phi(-\ell\sqrt I/2)$: the square of the step times the probability it is taken. Maximizing $h$ over $\ell$ gives $\ell^{\ast}\approx2.38/\sqrt I$ and, substituting back, an acceptance rate of $0.234$ to three places. The value is universal because $I$ cancels between the step and the rate — it is a property of the *balance* between moving far and moving often, not of the target.

    Since one unit of diffusion time costs $d$ steps, the number of iterations per effective draw grows like $d$. Set that against [Rejection Sampling](../part-09-monte-carlo-methods/05-rejection-sampling.md), where the acceptance rate itself decays like $c^{-d}$: the chain converts an exponential penalty into a linear one by allowing its steps to shrink, which is the trade [Markov Chain Monte Carlo](04-markov-chain-monte-carlo.md) calls forced.

    The result assumes the coordinates are comparably scaled. If they are not — if the target's covariance has condition number $\kappa$ — a single isotropic step cannot suit all of them, and it must be small enough for the tightest direction while the widest direction then needs many more steps to traverse. **The load-bearing quantity is the same $\kappa$ that [Numerical Optimization](01-numerical-optimization.md) showed gradient descent paying for and Newton's method ignoring, and the repair is the same change of variables, which here means proposing in units of each coordinate's own scale.**

Both halves are measurable, and the second reuses page 1's construction:

```python
import numpy as np

rng = np.random.default_rng(17053)
DRAWS, BURN, CHAINS = 20_000, 4_000, 4


def tau_int(v, M=2_000):
    v = v - v.mean()
    if not v.any():
        return float(len(v))
    f = np.fft.rfft(v, 2 * len(v))
    ac = np.fft.irfft(f * np.conj(f))[:M].real
    ac /= ac[0]
    s, k = 1.0, 1
    while k + 1 < M and ac[k] + ac[k + 1] > 0:
        s += 2 * ac[k]
        k += 1
    return s


def run(sd, step):
    """Gaussian target with marginal sds `sd` and proposal sds `step`; worst-axis ESS."""
    d = len(sd)
    th = rng.standard_normal((CHAINS, d)) * sd
    lp = -0.5 * ((th / sd) ** 2).sum(1)
    out = np.empty((CHAINS, DRAWS, d))
    acc = 0
    for i in range(DRAWS):
        cand = th + step * rng.standard_normal((CHAINS, d))
        lpc = -0.5 * ((cand / sd) ** 2).sum(1)
        take = np.log(rng.random(CHAINS)) < lpc - lp
        th = np.where(take[:, None], cand, th)
        lp = np.where(take, lpc, lp)
        acc += take.sum()
        out[:, i] = th
    k = out[:, BURN:]
    worst = max(float(np.mean([tau_int(k[c, :, j]) for c in range(CHAINS)]))
                for j in range(d))
    return acc / (DRAWS * CHAINS), 1.0 / worst


print(f"  an isotropic Gaussian target in d dimensions. The proposal scale is swept over"
      f" multiples of 2.38/sqrt(d) and the row reports the multiple with the highest ESS"
      f" per draw, against an asymptotic prediction of multiple 1 and acceptance 0.234")
print("      d   best multiple of 2.38/sqrt(d)   acceptance there   ESS per draw")
for d in (1, 2, 5, 10, 25, 50):
    ref = 2.38 / np.sqrt(d)
    best = max(((m,) + run(np.ones(d), ref * m) for m in (0.4, 0.7, 1.0, 1.4, 2.0)),
               key=lambda r: r[2])
    print(f"    {d:3d}   {best[0]:29.1f}   {best[1]:17.4f}   {best[2]:14.5f}")

print(f"  a 10-dimensional Gaussian whose inverse covariance has condition number kappa,"
      f" scored on its worst-mixing axis; both proposals are tuned by the same sweep, and"
      f" preconditioning divides each coordinate by its own standard deviation")
print("     condition number   isotropic: acceptance   isotropic: ESS/draws"
      "   preconditioned: acceptance   preconditioned: ESS/draws   gain")
d = 10
GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 1.5)
for kappa in (1.0, 10.0, 100.0, 1_000.0):
    sd = np.geomspace(1.0, np.sqrt(kappa), d)
    sd = sd / sd[-1]
    ref = 2.38 / np.sqrt(d)
    iso = max((run(sd, np.full(d, ref * m)) for m in GRID), key=lambda r: r[1])
    pre = max((run(sd, sd * ref * m) for m in (0.6, 1.0, 1.5)), key=lambda r: r[1])
    print(f"    {kappa:17,.0f}   {iso[0]:22.4f}   {iso[1]:20.6f}"
          f"   {pre[0]:27.4f}   {pre[1]:26.6f}   {pre[1] / iso[1]:8.1f}x")
# =>   an isotropic Gaussian target in d dimensions. The proposal scale is swept over multiples of 2.38/sqrt(d) and the row reports the multiple with the highest ESS per draw, against an asymptotic prediction of multiple 1 and acceptance 0.234
#          d   best multiple of 2.38/sqrt(d)   acceptance there   ESS per draw
#          1                             1.0              0.4429          0.22778
#          2                             1.0              0.3595          0.13026
#          5                             1.0              0.2903          0.05561
#         10                             1.0              0.2611          0.02613
#         25                             1.0              0.2456          0.01020
#         50                             1.0              0.2398          0.00449
#      a 10-dimensional Gaussian whose inverse covariance has condition number kappa, scored on its worst-mixing axis; both proposals are tuned by the same sweep, and preconditioning divides each coordinate by its own standard deviation
#         condition number   isotropic: acceptance   isotropic: ESS/draws   preconditioned: acceptance   preconditioned: ESS/draws   gain
#                        1                   0.1043               0.022160                        0.2651                     0.026631        1.2x
#                       10                   0.1908               0.008342                        0.2604                     0.028285        3.4x
#                      100                   0.1377               0.002267                        0.2615                     0.028183       12.4x
#                    1,000                   0.1949               0.000638                        0.2613                     0.028584       44.8x
```

The first table confirms the proof and shows how fast the asymptotics arrive. The best multiple of $2.38/\sqrt d$ is $1.0$ in every row, which is the scaling rule working exactly; acceptance at that setting runs $0.4429$, $0.3595$, $0.2903$, $0.2611$, $0.2456$, $0.2398$, approaching $0.234$ from above and effectively there by $d=25$. The one-dimensional row is worth noting separately: $0.4429$, not $0.234$, which is why a rule of thumb quoted as "aim for a quarter" is wrong for scalar problems and why the practical advice is usually given as a band from about $0.2$ to $0.5$. Efficiency falls $0.22778$, $0.13026$, $0.05561$, $0.02613$, $0.01020$, $0.00449$ — close to inversely proportional to $d$, which is the linear cost the proof predicts and the thing that makes the method usable at all.

The second table is the failure that is not about dimension. The first row is a control: with all coordinates equally scaled there is nothing to precondition, the two proposals are the same sampler, and the $1.2$ is measurement noise on an effective sample size. Below it, the isotropic proposal's worst-axis efficiency falls $0.022160$, $0.008342$, $0.002267$, $0.000638$ — roughly a factor of $3.5$ per decade of $\kappa$, so close to $\kappa^{-1/2}$ — while the preconditioned proposal holds $0.026631$, $0.028285$, $0.028183$, $0.028584$, flat to the third decimal and equal to the well-conditioned value. **Preconditioning does not improve the sampler; it makes the hard problem into the easy one, and at $\kappa=1{,}000$ that is worth a factor of $44.8$.** The information required is the target's marginal scales, which a short pilot run supplies and which adaptive Metropolis estimates on the fly.

Notice also what the isotropic rows' acceptance column does *not* do: it reads $0.1043$, $0.1908$, $0.1377$, $0.1949$ at the best setting available, never settling anywhere and never approaching $0.234$. Tuning acceptance to the textbook value would have made the ill-conditioned rows worse, because the target the rule was derived for is not the target being sampled.

## A Proposal With Lighter Tails Than the Target Accepts Most of the Time and Sits Frozen for Ten Thousand Draws

Everything so far has been a random walk, where the proposal is centred on the current state. The other classical choice ignores the current state entirely and proposes from a fixed distribution — the independence sampler, which is what a random-walk chain becomes in the limit of a well-chosen global proposal, and which behaves like importance sampling in a way that turns out to be exact.

??? note "Proof that an independence sampler is uniformly ergodic if and only if $\pi/q$ is bounded, with rate $1-1/M$ where $M=\sup\pi/q$, so a proposal with lighter tails than the target has no geometric rate at all"

    Let $q$ be the proposal, independent of the current state, and write $w(\theta)=\pi(\theta)/q(\theta)$ for the importance weight — the same object [Importance Sampling](../part-09-monte-carlo-methods/04-importance-sampling.md) builds its estimator from. The acceptance probability simplifies:
    $$\alpha(\theta,\theta')=\min\Big(1,\frac{\pi(\theta')q(\theta)}{\pi(\theta)q(\theta')}\Big)=\min\Big(1,\frac{w(\theta')}{w(\theta)}\Big),$$
    so the chain moves whenever the candidate has a larger weight and otherwise moves with probability equal to the weight ratio. Suppose $M=\sup_\theta w(\theta)<\infty$. Then for any $\theta$ and any set $A$,
    $$P(\theta,A)\ \ge\ \int_A q(\theta')\min\Big(1,\frac{w(\theta')}{w(\theta)}\Big)\mathrm{d}\theta'\ \ge\ \int_A q(\theta')\frac{w(\theta')}{M}\,\mathrm{d}\theta'=\frac{\pi(A)}{M},$$
    which is Doeblin's condition with constant $1/M$, giving uniform ergodicity with $\lVert P^{t}(\theta,\cdot)-\pi\rVert_{TV}\le(1-1/M)^{t}$ from every starting point.

    The converse is the operative half. If $w$ is unbounded, then for any $\varepsilon$ there are states with $w(\theta)>1/\varepsilon$, and from such a state the probability of moving is at most $\varepsilon\cdot\mathbb{E}_q[w]=\varepsilon$ — the chain is stuck there for a geometric number of steps with mean $1/\varepsilon$ or worse. Since the chain visits such states with positive probability, no uniform rate exists, and the expected sojourn has no finite bound. The condition $\sup\pi/q<\infty$ is exactly the support condition [Importance Sampling](../part-09-monte-carlo-methods/04-importance-sampling.md) requires for finite weight variance, arriving here as a statement about time rather than about variance.

    Concretely, if $\pi$ is Student-$t$ and $q$ is Gaussian, then $w(\theta)\propto e^{\theta^{2}/2}(1+\theta^{2}/\nu)^{-(\nu+1)/2}\to\infty$, so a single excursion into the tail freezes the chain for as long as it takes to propose something with a comparable weight, which is a very long time.

    **The load-bearing consequence is that the proposal must dominate the target's tails and that the diagnostic everyone watches is blind to the violation: a chain frozen at a high-weight state rejects everything, but it spends almost all of its remaining time in the bulk where acceptance is high, so the average acceptance rate stays comfortable while the run is worthless.**

The dichotomy is sharp enough to see in one table:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(17055)
REPS, DRAWS, BURN, NU, Q = 300, 30_000, 5_000, 3.0, 3.0
Z = stats.norm.isf(0.025)
TRUTH = float(stats.t.sf(Q, NU))


def tau_ips(v, M=4_000):
    """Integrated autocorrelation time per row, initial-positive-sequence rule."""
    n = v.shape[1]
    w = v - v.mean(1, keepdims=True)
    f = np.fft.rfft(w, 2 * n, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :M].real
    ac /= np.maximum(ac[:, :1], 1e-300)
    out = np.ones(v.shape[0])
    for r in range(v.shape[0]):
        s, k = 1.0, 1
        while k + 1 < M and ac[r, k] + ac[r, k + 1] > 0:
            s += 2 * ac[r, k]
            k += 1
        out[r] = s
    return out


def independence(logq, draw, s):
    """Independence sampler: the proposal ignores the current state entirely."""
    th = rng.standard_normal(REPS) * 0.5
    lw = stats.t.logpdf(th, NU) - logq(th / s)
    out = np.empty((REPS, DRAWS))
    stuck, worst = np.zeros(REPS), np.zeros(REPS)
    for i in range(DRAWS):
        cand = s * draw()
        lwc = stats.t.logpdf(cand, NU) - logq(cand / s)
        take = np.log(rng.random(REPS)) < lwc - lw
        th, lw = np.where(take, cand, th), np.where(take, lwc, lw)
        stuck = np.where(take, 0.0, stuck + 1)
        worst = np.maximum(worst, stuck)
        out[:, i] = th
    return out[:, BURN:], worst


PROPOSALS = (
    ("Student-t, 1 df (heavier)", lambda z: stats.t.logpdf(z, 1.0),
     lambda: stats.t.rvs(1.0, size=REPS, random_state=rng), 1.4),
    ("Student-t, 3 df (equal)", lambda z: stats.t.logpdf(z, 3.0),
     lambda: stats.t.rvs(3.0, size=REPS, random_state=rng), 1.0),
    ("Student-t, 8 df (lighter)", lambda z: stats.t.logpdf(z, 8.0),
     lambda: stats.t.rvs(8.0, size=REPS, random_state=rng), 1.1),
    ("Gaussian (much lighter)", lambda z: stats.norm.logpdf(z),
     lambda: rng.standard_normal(REPS), 1.3),
)

print(f"  an independence sampler on a Student-t posterior with {NU:.0f} degrees of"
      f" freedom, estimating P(theta > {Q:.0f}) = {TRUTH:.4f}. Every proposal below gives"
      f" a correct sampler; they differ only in the weight of their tails."
      f" {REPS} chains x {DRAWS:,} draws, {BURN:,} discarded")
print("     proposal                    acceptance   longest run stuck at one value"
      "   ESS/draws   mean estimate   spread / mean MCSE   coverage of the nominal 95%")
for name, logq, draw, s in PROPOSALS:
    k, worst = independence(logq, draw, s)
    ind = (k > Q).astype(float)
    est = ind.mean(1)
    mcse = np.sqrt(tau_ips(ind) * ind.var(1, ddof=1) / ind.shape[1])
    acc = 1.0 - (np.diff(k, axis=1) == 0).mean()
    cov = (np.abs(est - TRUTH) <= Z * mcse).mean()
    print(f"    {name:26s}   {acc:10.4f}   {worst.max():29,.0f}"
          f"   {1 / tau_ips(k).mean():9.5f}   {est.mean():13.4f}"
          f"   {est.std() / mcse.mean():18.2f}   {cov:26.4f}")
# =>   an independence sampler on a Student-t posterior with 3 degrees of freedom, estimating P(theta > 3) = 0.0288. Every proposal below gives a correct sampler; they differ only in the weight of their tails. 300 chains x 30,000 draws, 5,000 discarded
#         proposal                    acceptance   longest run stuck at one value   ESS/draws   mean estimate   spread / mean MCSE   coverage of the nominal 95%
#        Student-t, 1 df (heavier)        0.6839                              14     0.71078          0.0288                 0.94                       0.9567
#        Student-t, 3 df (equal)          1.0000                               0     0.98797          0.0289                 0.98                       0.9500
#        Student-t, 8 df (lighter)        0.9328                           1,765     0.03236          0.0285                 1.27                       0.8633
#        Gaussian (much lighter)          0.8744                           9,708     0.01684          0.0269                 3.74                       0.5433
```

The top two rows are the proof's positive half. A Student-$t$ proposal with one degree of freedom dominates the target's tails, accepts $0.6839$ of the time, never sits still for more than $14$ draws, and delivers an effective sample size of $0.71078$ per draw with coverage of $0.9567$. The exactly-matched proposal is the degenerate best case: the weight is constant, $\alpha\equiv1$, acceptance is $1.0000$, the longest stuck run is $0$, and the sampler is drawing independently — $0.98797$ effective draws per draw, which is what an independence sampler becomes when it has nothing left to correct.

The bottom two rows are the failure, and the acceptance column is the reason to keep reading. A Student-$t$ proposal with eight degrees of freedom against a three-degree-of-freedom target accepts $0.9328$ of proposals — a rate that would be read as an over-timid random walk and never as a broken chain — while one of its three hundred runs sat at a single value for $1{,}765$ consecutive draws, the effective sample size is $0.03236$ per draw, and coverage of the nominal interval has fallen to $0.8633$. The Gaussian proposal accepts $0.8744$ and freezes for $9{,}708$ draws, a third of the entire run, giving $0.01684$ effective draws per draw, an estimate biased to $0.0269$ against a truth of $0.0288$, and coverage of $0.5433$.

**The reported error bar is $3.74$ times too small, and the number that would have revealed this — the longest interval spent at a single value — is not among the quantities anyone reports.** The mechanism is the proof's: a chain that wanders into the tail acquires a weight far above anything the proposal will produce again soon and stops moving, but the frozen episodes are rare, so the *average* acceptance rate is dominated by the bulk where everything is fine. A summary statistic averaged over time cannot detect a pathology concentrated in a small fraction of it.

## The Proposal Is a Model of the Posterior, and Every Improvement Over a Random Walk Is a Better Model

The through-line of all three sections is that Metropolis–Hastings separates into a correctness argument that costs nothing and a proposal that costs everything, and that the proposal is doing the same job an approximating distribution does elsewhere in this appendix. A random walk models the posterior as locally isotropic; preconditioning models it as locally Gaussian with a known covariance; an independence sampler models it globally, and section 4 is what happens when that global model has the wrong tails.

The catalogue past this page follows the same principle. **Adaptive Metropolis** estimates the target covariance from the chain's own history and preconditions with it, which is section 3's repair performed automatically; it breaks ergodicity if done naively, since the kernel then depends on the entire past, and is repaired by *diminishing adaptation* — letting the adaptation rate fall to zero — or by simply freezing the proposal after warm-up. **Metropolis-adjusted Langevin** shifts the proposal's centre by half a step along the gradient of the log-posterior, which uses the derivative [Numerical Optimization](01-numerical-optimization.md) computes and improves the dimension scaling from $d$ to $d^{1/3}$ with an optimal acceptance rate of $0.574$. **Hamiltonian Monte Carlo** integrates a trajectory rather than taking one step, scaling like $d^{1/4}$, and its adaptive form NUTS is the default in every probabilistic programming system; none of these is built here, and all of them are the same algebra with a better proposal. **Metropolis-within-Gibbs** applies this page's rule to one coordinate at a time when a full conditional cannot be sampled directly, which is the bridge to [Gibbs Sampling](06-gibbs-sampling.md).

!!! note "Rejection sampling's acceptance rate, this page's acceptance rate, importance sampling's effective sample size ratio and a chain's effective sample size ratio are four efficiency measures on the same scale that are not comparable"
    They all live in $(0,1]$ and all sound like the fraction of work that was useful, which is why they get compared. **Rejection sampling's acceptance rate** is the fraction of proposals that become draws, and a rejected proposal produces nothing: the cost is $1/\text{rate}$ proposals per independent draw, so $0.234$ there would be poor and [Rejection Sampling](../part-09-monte-carlo-methods/05-rejection-sampling.md) prices its exponential decay in dimension. **Metropolis' acceptance rate** is the fraction of steps that move, and a rejected step still produces a draw — a duplicate — so the optimum is an interior point and $0.234$ is not merely acceptable but ideal in high dimension; the $0.9948$ and $0.0011$ rows of section 2 are symptoms of the same disease. **Importance sampling's ESS ratio** measures the concentration of weights across an independent sample, and it is a statement about the proposal alone. **A chain's ESS ratio** measures autocorrelation across a dependent sample, and section 4 shows it falling to $0.01684$ while acceptance reads $0.8744$, which is the pair that proves the two are not the same quantity. Tuning a Metropolis chain toward high acceptance because high acceptance is good in rejection sampling is the error this list exists to prevent, and section 2 prices it at a factor of $695$.

!!! warning "A Metropolis chain reports an acceptance rate averaged over time, and the failure that destroys it is concentrated in the fraction of time that average is least sensitive to"
    Nothing in the failing cases looked wrong. Section 2's extremes are at least legible — an acceptance rate of $0.9948$ or $0.0000$ is visibly wrong to anyone who knows the target band. Section 4's are not: a chain accepting $0.9328$ and one accepting $0.8744$ sit in the range a practitioner reads as "slightly over-timid, tighten the proposal a little," and both were frozen at a single value for $1{,}765$ and $9{,}708$ consecutive draws respectively, delivering coverage of $0.8633$ and $0.5433$ against a nominal $0.95$ with error bars $1.27$ and $3.74$ times too small. Section 3's ill-conditioned rows never reach the textbook acceptance rate at any setting, so tuning toward $0.234$ would have made them worse. **The free diagnostic is to record the longest run of consecutive rejections alongside the acceptance rate — one integer, computed in the loop you have already written — and to compare it against the chain length, because a chain that spent a third of its run at one value has an effective sample size that no average acceptance rate will reveal; and to run four short chains rather than one long one, since the spread of their estimates against the error bar each reports is the only check on this page that does not require knowing the answer.** The trading stake's twenty seeds are the same instrument pointed at a different algorithm.

## A Rule That Is Right Whatever You Give It

This page established that the Metropolis–Hastings kernel satisfies detailed balance for any proposal with adequate support and that the target enters only through a ratio, so the normalizing constant [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) called the only expensive step is never computed; that correctness is consequently uniform across tuning while cost is not, seven proposal scales spanning five orders of magnitude all landing within $2.37$ Monte Carlo standard errors of a published grid answer while effective sample size per draw ran $0.00021$, $0.00063$, $0.03804$, $0.14598$, $0.05150$, $0.00082$ and $0.00019$, a factor of $768$ with the two worst settings failing for opposite reasons at acceptance rates of $0.9948$ and $0.0000$; that the optimal scale is $2.38/\sqrt d$ with acceptance falling $0.4429$, $0.3595$, $0.2903$, $0.2611$, $0.2456$, $0.2398$ toward the asymptotic $0.234$ and efficiency declining like $1/d$, while an isotropic proposal on a target of condition number $1{,}000$ delivers $0.000638$ against a preconditioned $0.028584$, a factor of $44.8$ recovered by dividing each coordinate by its own scale; and that an independence sampler whose proposal has lighter tails than the target accepts $0.9328$ and $0.8744$ of proposals while freezing for $1{,}765$ and $9{,}708$ consecutive draws, delivering effective sample sizes of $0.03236$ and $0.01684$ per draw, error bars $1.27$ and $3.74$ times too small, and coverage of $0.8633$ and $0.5433$ against a nominal $0.95$.

The shape shared by all three exhibits is that the acceptance rate is the quantity everyone watches and it is a sufficient diagnostic in exactly one of them. In section 2 it identifies both failures correctly, because a random walk that is mistuned is mistuned all the time. In section 3 it is actively misleading, since the ill-conditioned target's best available setting never approaches the textbook value and tuning toward that value would cost efficiency. In section 4 it is silent, because the pathology occupies a small fraction of the run and an average over time is insensitive to exactly that. In all three cases the quantity that would have worked — the effective sample size, which is a statement about the whole trajectory rather than about one step — was computable from the draws already in memory.

The rejections are what remains. Every rejected proposal on this page was a full evaluation of the target, paid for and then discarded in favour of copying the previous state, and section 2 shows that at the optimum roughly three fifths of the work is spent that way. There is a family of proposals for which that waste is exactly zero — where the acceptance probability is identically one, not by luck or tuning but by construction, and the free parameter this page spent three sections on does not exist. The price is a different one, paid at the moment the model is written rather than when the sampler is run, and it is [Gibbs Sampling](06-gibbs-sampling.md).

**Metropolis–Hastings guarantees the right answer for any proposal and tells you nothing about which proposal to use, so the theorem is about the algorithm and every question you actually have is about the choice the theorem declines to constrain.**
