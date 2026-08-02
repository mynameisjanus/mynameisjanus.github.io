# Brownian Motion

Brownian motion is not one model among several for a price path. It is the only possible continuous-time limit of a sum of independent shocks with finite variance, which means every random walk in this course — whatever its step distribution, however fat its tails — is heading for the same object, and the only question is how slowly. What arrives in the limit is a process with continuous paths that are differentiable nowhere, a running maximum whose law is available in one line, and a first-passage time that occurs with probability one and has infinite expectation.

This page covers the four defining properties and the invariance principle that makes them inevitable, the scaling relation and the non-differentiability and quadratic variation it forces, the reflection principle and the exact law of the running maximum, the expected drawdown that a strategy with no edge is guaranteed to suffer, and the first-passage time whose median is finite and whose mean is not. It does not develop the Itô integral or Itô's lemma, which are [Stochastic Calculus](../../advanced/03-stochastic-calculus.md); it does not build the multiplicative version, which is [Geometric Brownian Motion](09-geometric-brownian-motion.md); it does not prove the central limit theorem underneath Donsker, which is [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md); it does not develop the fair-game structure it exhibits, which is [Martingales](10-martingales.md); and it prices nothing, which is [Options Pricing](../../advanced/11-options-pricing.md).

The trading stake is one of the most uncomfortable sentences in the course. [Performance Metrics and Reporting](../../part-05-backtesting-engine/04-performance-metrics-and-reporting.md) computes the drawdown anatomy of the trend book and reports that "this equity curve set a new high on only 2% of its days", adding that "it is near-universal for any positive-drift-plus-noise process, and it means the *lived experience* of running even a good strategy is losing to your own past self on ninety-eight days out of a hundred." That sentence is a theorem about the process on this page rather than an observation about `tsmom`, and the third and fourth sections derive both the $2\%$ and the depth that goes with it.

## Four Properties, and Donsker Says There Was Never Another Choice

Standard **Brownian motion** $W_t$ is defined by four requirements: $W_0=0$; increments over disjoint intervals are independent; $W_t-W_s\sim\mathcal{N}(0,t-s)$ for $s<t$; and the paths are continuous. Independence and stationarity of increments make it the continuous-time analogue of a random walk; the Gaussian marginal is the only part that looks like a choice, and it is not one.

**Donsker's invariance principle** says so precisely. Take any i.i.d. steps $X_1,X_2,\ldots$ with mean zero and variance $\sigma^{2}<\infty$, form the partial sums $S_k$, and rescale time by $1/n$ and space by $1/(\sigma\sqrt n)$. The resulting path converges in distribution — as a random *function*, not merely at a fixed time — to $W$. Nothing about the step law survives except its variance, which is [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md)'s conclusion promoted from a single marginal to the whole trajectory, and it means any functional of the path converges too: the terminal value, the maximum, the time spent positive, the first passage.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(8081)
reps = 100_000
laws = {"bernoulli": lambda k: 2.0 * (rng.random(k) < 0.5) - 1.0,
        "student t3": lambda k: rng.standard_t(3, k) / np.sqrt(3.0),
        "pareto a=3": lambda k: (rng.pareto(3.0, k) - 0.5) / np.sqrt(0.75),
        "gaussian": lambda k: rng.standard_normal(k)}
print(f"  P(running max of the rescaled walk exceeds 1), {reps} paths per cell")
print("       steps    bernoulli    student t3    pareto a=3    gaussian    brownian")
for n in (4, 16, 256, 4096):
    out = []
    for step in laws.values():
        s = np.zeros(reps)
        m = np.full(reps, -np.inf)
        for _ in range(n):
            s += step(reps)
            np.maximum(m, s, out=m)
        out.append((m / np.sqrt(n) > 1.0).mean())
    print(f"  {n:11d} {out[0]:12.4f} {out[1]:13.4f} {out[2]:13.4f} {out[3]:11.4f}"
          f" {2 * norm.sf(1.0):11.4f}")
# =>   P(running max of the rescaled walk exceeds 1), 100000 paths per cell
#           steps    bernoulli    student t3    pareto a=3    gaussian    brownian
#                4       0.1249        0.1429        0.1197      0.2146      0.3173
#               16       0.2113        0.1982        0.1674      0.2554      0.3173
#              256       0.2892        0.2785        0.2462      0.3007      0.3173
#             4096       0.3098        0.3051        0.2917      0.3139      0.3173
```

Four completely different step laws — a coin flip, a $t_3$ with infinite fourth moment, a skewed Pareto, and a Gaussian — all climb toward the same limit $0.3173$, which is $2(1-\Phi(1))$ and will be derived in the third section. At $4{,}096$ steps they read $0.3098$, $0.3051$, $0.2917$, $0.3139$. **The step distribution is not a modelling decision about the limit; it is a decision about how long the limit takes to arrive.**

The order of the columns at every row is the useful part. The Pareto, which is the most skewed of the four, is furthest from the limit at every horizon — $0.2917$ at $n=4096$ where the Gaussian is already at $0.3139$. Fat tails and skew do not escape Donsker as long as the variance is finite; they merely converge slowly, which matters because financial sample sizes are small enough for "slowly" and "not at all" to be hard to tell apart.

!!! note "Every number in this table is biased downward, and the bias is the difference between watching a path and sampling it"
    The simulation records the maximum over the $n$ integer times, and the limit $0.3173$ is the maximum over the *continuous* interval. A continuous path can exceed a level between two observations and be back below by the next one, so a discretely monitored maximum is always smaller, which is why even the Gaussian column falls short at $0.3139$ rather than landing on $0.3173$. The gap shrinks like $1/\sqrt n$ and is a genuine practical effect rather than a simulation artifact: a drawdown computed from daily marks understates the drawdown actually lived through intraday, a stop monitored on the close is a different contract from a stop monitored continuously, and a barrier option priced on discrete monitoring is worth measurably less than its continuous counterpart. The correction is standard and worth knowing exists; the mistake is not knowing the two quantities differ.

## Scaling Forces Paths With No Derivative

The variance being linear in time is the whole of the process's geometry. It gives the **scaling relation** $W_{ct}\overset{d}{=}\sqrt c\,W_t$, so the process is statistically self-similar: rescale time by $c$ and space by $\sqrt c$ and you cannot tell the difference. An increment over an interval of width $\delta$ has size of order $\sqrt\delta$, not $\delta$, and everything unusual follows from that mismatch.

??? note "Proof that Brownian paths are differentiable at no point, from the scaling relation alone"
    Consider the difference quotient over a window of width $\delta$. Its standard deviation is

    $$\operatorname{sd}\left(\frac{W_{t+\delta}-W_t}{\delta}\right)=\frac{\sqrt\delta}{\delta}=\frac{1}{\sqrt\delta}\ \longrightarrow\ \infty\quad\text{as }\delta\to0,$$

    so the quotient does not converge to anything finite; it spreads without bound. To turn divergence in distribution into a statement about paths, fix $t$ and take $\delta_k=2^{-k}$. The increments over the disjoint halves of these nested windows are independent, and a Borel–Cantelli argument on the events $\{\lvert W_{t+\delta_k}-W_t\rvert>K\delta_k\}$ — each of probability tending to one for any fixed $K$ — shows that no finite bound holds along the sequence, almost surely. Since a countable union of null sets is null, the conclusion holds simultaneously at every rational $t$, and path continuity extends it to all $t$.

    The same $\sqrt\delta$ scaling determines the **quadratic variation**. Summing squared increments over $n$ equal steps of $[0,T]$ gives a total with mean exactly $T$ for every $n$ and variance $2T^{2}/n\to0$, so the sum converges to the constant $T$: $[W]_T=T$, deterministic despite every increment being random. [Stochastic Calculus](../../advanced/03-stochastic-calculus.md) derives this in full and builds the Itô calculus on top of it; here it is enough to know that the shorthand $(dW)^{2}=dt$ abbreviates a limit that is not random.

    The load-bearing hypothesis is finite variance of the increments, which is what fixed the scaling exponent at $\tfrac12$. Give the steps infinite variance and Donsker fails, the limit is a stable process with a different exponent, and the paths acquire jumps — so "prices move unpredictably at every timescale" is not one statement but a family, and Brownian motion is the finite-variance member.

The practical residue of non-differentiability is that a Brownian path has no velocity, so "the trend right now" is not a quantity the model contains. Any estimate of it is an average over a window, its precision is set by the window length, and shrinking the window to get a more current answer makes the estimate worse at exactly the rate $1/\sqrt\delta$ above. That tension is not a limitation of estimation technique; it is a property of the object.

## The Reflection Principle Turns a Maximum Into a Marginal

The running maximum $M_T=\max_{0\leq t\leq T}W_t$ looks like it should require the whole path's law. It does not, and the reason is a symmetry argument that costs one line.

??? note "Proof that the maximum exceeds a level exactly twice as often as the endpoint does"
    Fix $a>0$ and let $\tau_a$ be the first time the path reaches $a$. Suppose $\tau_a\leq T$. By the **strong Markov property** — the process restarted at a stopping time is a fresh Brownian motion, independent of the past — the increment $W_T-W_{\tau_a}$ is symmetric about zero. So conditional on having touched $a$, the path is equally likely to finish above $a$ as below it:

    $$\mathbf{P}(W_T>a\mid\tau_a\leq T)=\tfrac12.$$

    Reflecting the path after $\tau_a$ about the level $a$ maps the paths that finish below onto those that finish above, one to one. Now note that finishing above $a$ *requires* having touched $a$, by continuity, so $\{W_T>a\}\subset\{\tau_a\leq T\}$ and

    $$\mathbf{P}(M_T\geq a)=\mathbf{P}(\tau_a\leq T)=2\,\mathbf{P}(W_T>a)=2\Bigl(1-\Phi\bigl(a/\sqrt T\bigr)\Bigr).$$

    Differentiating gives the density of $M_T$ as that of $\lvert W_T\rvert$, from which $\mathbb{E}[M_T]=\sqrt{2T/\pi}$, and for a process with volatility $\sigma$ per unit time, $\mathbb{E}[M_T]=\sigma\sqrt{2T/\pi}$.

    The load-bearing hypotheses are two and both are essential. **Continuity** is what makes $\{W_T>a\}\subset\{\tau_a\leq T\}$ — a process that can jump over $a$ breaks the containment and the factor of two with it. And the **strong Markov property** is what licenses restarting at the random time $\tau_a$; restarting at a time that peeks at the future would destroy the symmetry, which is the same measurability requirement [Martingales](10-martingales.md) is built on.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(8083)
reps, steps = 400_000, 5_040                                   # twenty years of daily marks
sd = 0.011
print(f"  the running maximum of a driftless walk, {reps} paths of {steps} steps")
print("        a/sigma    P(max > a) sim    2 P(W_T > a)    P(W_T > a)")
s = np.zeros(reps)
m = np.zeros(reps)
for _ in range(steps):
    s += sd * rng.standard_normal(reps)
    np.maximum(m, s, out=m)
scale = sd * np.sqrt(steps)
for a in (0.5, 1.0, 1.5, 2.0):
    print(f"  {a:14.1f} {(m > a * scale).mean():17.4f} {2 * norm.sf(a):15.4f}"
          f" {norm.sf(a):13.4f}")
print(f"  mean of the running max {m.mean():.4f} against sigma sqrt(2T/pi) "
      f"{scale * np.sqrt(2 / np.pi):.4f}")
# =>   the running maximum of a driftless walk, 400000 paths of 5040 steps
#            a/sigma    P(max > a) sim    2 P(W_T > a)    P(W_T > a)
#                 0.5            0.6106          0.6171        0.3085
#                 1.0            0.3132          0.3173        0.1587
#                 1.5            0.1316          0.1336        0.0668
#                 2.0            0.0448          0.0455        0.0228
#      mean of the running max 0.6167 against sigma sqrt(2T/pi) 0.6231
```

The factor of two is exact and the simulation confirms it: $0.6106$ against $0.6171$, $0.3132$ against $0.3173$, $0.1316$ against $0.1336$, $0.0448$ against $0.0455$, with the last column showing the endpoint probability that is half of each. The residual shortfall is the discrete-monitoring bias of the previous admonition, and it is the same size everywhere. The mean maximum lands at $0.6167$ against the predicted $\sigma\sqrt{2T/\pi}=0.6231$.

Read the second row as a statement about track records. A strategy with no edge whatever, run for twenty years at $1.1\%$ daily volatility, has a $31\%$ chance of showing a cumulative gain at some point of at least one full twenty-year standard deviation — and a $4.5\%$ chance of two. **The running maximum of a driftless process is not centred at zero; it is centred at $\sigma\sqrt{2T/\pi}$ and grows without bound.** Every peak in an equity curve is being read against a benchmark of zero when the correct benchmark grows like $\sqrt T$.

## A Strategy With No Edge Has a Guaranteed Expected Drawdown

The same result read from the other end is the drawdown. For a driftless process the distance below the running maximum, $M_t-W_t$, has the same law as $\lvert W_t\rvert$ at each fixed time, so its expectation is $\sigma\sqrt{2t/\pi}$ — the *expected* shortfall from the high-water mark is not zero, and it grows like the square root of the time you have been running. At the $5{,}040$ days and $1.1\%$ daily volatility above, that is $0.6231$ in log terms, or roughly $46\%$ in price terms, purely from having watched a fair game for twenty years.

This is what makes the course's $2\%$ figure a theorem rather than a finding. The set of times at which a continuous path equals its running maximum has Lebesgue measure zero — the process sits at a new high on a set of times of measure zero and strictly below it almost everywhere else. On a discrete grid the fraction of days at a new high is not zero, but it is small, it shrinks as the grid refines, and it depends far more on the observation frequency than on the strategy. So "new equity highs on $2\%$ of days" describes Brownian motion sampled daily for twenty-five years rather than diagnosing the trend book, and a strategy with three times the Sharpe would produce a number in the same neighbourhood.

!!! warning "A drawdown is evidence about a strategy only after the drawdown a fair game would have produced has been subtracted, and it almost never is"
    The reflection principle supplies the null distribution a drawdown number needs and that no tearsheet prints. Before concluding that a $-28.7\%$ drawdown means something broke, the question to answer is what the deepest drawdown of a *zero-edge* process with the same volatility over the same history would have been — and at twenty years of $1.1\%$ daily volatility its expectation alone is on the order of the number being worried about. The same applies in the flattering direction: a strategy that has never drawn down more than $5\%$ over a long history is making a claim about its volatility, not about its skill. [The Strong Law of Large Numbers](../part-07-asymptotic-theory/02-strong-law-of-large-numbers.md) runs the running-maximum argument on the Sharpe ratio instead and finds one zero-edge strategy in twenty-three touching $2.0$; this is the same phenomenon read on the equity curve, and neither is visible without simulating the null.

## A Stop Is Hit With Probability One in Infinite Expected Time

The reflection principle also settles the first-passage time. Letting $T\to\infty$ in $\mathbf{P}(\tau_a\leq T)=2(1-\Phi(a/\sqrt T))$ gives $\mathbf{P}(\tau_a<\infty)=1$: a driftless process reaches any level, however far, with certainty. The density that falls out is $f_{\tau_a}(t)=\frac{a}{\sqrt{2\pi t^{3}}}e^{-a^{2}/2t}$, whose tail decays like $t^{-3/2}$ — heavy enough that $\mathbb{E}[\tau_a]=\infty$, even though the event is certain.

```python
import numpy as np

rng = np.random.default_rng(8087)
reps, barrier = 100_000, 5
print(f"  when does a driftless walk first touch a stop {barrier} units away? {reps} paths")
print("       horizon    fraction stopped    median when stopped    mean when stopped")
pos = np.zeros(reps, dtype=np.int32)
hit = np.zeros(reps, dtype=np.int64)
for t in range(1, 10_001):
    pos += 2 * (rng.random(reps) < 0.5).astype(np.int32) - 1
    hit[(hit == 0) & (pos >= barrier)] = t
    if t in (10, 100, 1_000, 10_000):
        h = hit[(hit > 0)]
        print(f"  {t:13d} {len(h) / reps:19.4f} {np.median(h):22.1f} {h.mean():21.1f}")
# =>   when does a driftless walk first touch a stop 5 units away? 100000 paths
#           horizon    fraction stopped    median when stopped    mean when stopped
#                 10              0.1094                    7.0                   7.2
#                100              0.6181                   25.0                  32.2
#               1000              0.8737                   41.0                 116.9
#              10000              0.9598                   49.0                 386.9
```

The first column marches toward certainty exactly as the theorem promises — $0.1094$, $0.6181$, $0.8737$, $0.9598$ — and does so slowly, since the deficit decays like $1/\sqrt T$. The stop will be hit; you may wait a while.

The last two columns are the point, and they diverge. The median time-to-stop among the paths that stopped settles down: $7$, $25$, $41$, $49$, creeping toward a finite limit as the surviving paths run out of ways to avoid the barrier. The mean does not settle at all. It reads $7.2$, $32.2$, $116.9$, $386.9$, roughly tripling with each tenfold extension of the horizon, and it will keep growing for as long as the simulation is allowed to run. **The average time until a stop is hit is not a large number; it is not a number**, and every reported value is an artifact of where the observation was truncated.

The consequences are two, and both are about capital rather than probability. A spread with a stop far from its mean — the setup of [Mean Reversion and Pairs Trading](../../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md), whose Ornstein–Uhlenbeck process is this Brownian motion plus a restoring force, and which notes that the half-life "dictates the trade's entire tempo" — has a stop certain to be hit and, without that restoring force, a holding period whose average does not exist. And any capital plan built on "the average time to resolution" is planning around a statistic that is undefined; the median, or a quantile at the horizon actually intended, is the version with content.

## The Only Limit There Is, and Three Prices for It

Brownian motion earns its position by uniqueness rather than by realism. Donsker says that any finite-variance random walk converges to it as a *function*, so every functional converges too, and there is no competing continuous-time limit to argue for — the modelling debate is entirely about whether the variance is finite and how fast the convergence is, which for the Pareto column above was visibly slower than for the Gaussian.

Three things follow that a discrete intuition does not supply. Paths have no derivative, so "the current trend" is not in the model, and every estimate of it trades currency against precision at a fixed exchange rate. The running maximum has an exact law obtained by reflection, which supplies the null distribution that drawdown and high-water-mark statistics need and that tearsheets do not print — including the course's own $2\%$-of-days figure, which is a property of the process rather than of the strategy. And a barrier is reached with probability one after a wait whose mean is infinite, so questions phrased as "on average, how long until" have no answer and must be rephrased as quantiles.

All three are statements about a process with no drift, and adding drift changes each of them in ways intuition does not anticipate: the maximum acquires a finite limiting law when the drift is negative, the first-passage time acquires a finite mean when the drift points at the barrier, and the whole structure becomes the multiplicative object that actually models prices. That is [Geometric Brownian Motion](09-geometric-brownian-motion.md), where the drift turns out to be the one parameter no amount of data will pin down.
