# Martingales

A martingale is the formal object behind the sentence "there is no edge here". Its definition says only that the best forecast of tomorrow's value is today's, and from that single line follow three results a trading business runs into constantly: no betting system can make a fair game favourable, a stopping rule cannot change an expectation unless it is allowed to run forever, and a statistic watched continuously will cross any fixed threshold far more often than a statistic looked at once. The third is why backtests that stop when they look good are not evidence.

This page covers the filtration and the measurability requirement that is the formal content of point-in-time discipline, the martingale transform and the systems theorem it proves, the optional stopping theorem with its three hypotheses and a worked failure when one is dropped, Doob's maximal inequality and the martingale convergence theorem, and the inflation in false-positive rates that repeated looks at an accumulating backtest produce. It does not build the processes it is applied to, which are [Brownian Motion](08-brownian-motion.md) and [Random Walks](11-random-walks.md); it does not develop conditional expectation, which is [Conditional Expectation](../part-04-expectation-and-moments/06-conditional-expectation.md); it does not construct a test or a correction, which are [Part XII](../part-12-hypothesis-testing/index.md) and [Part XV](../part-15-multiple-testing/index.md); it develops no measure-theoretic representation theorem, which is [Stochastic Calculus](../../advanced/03-stochastic-calculus.md); and it prices nothing.

The trading stake is the null the course commits to before running a single backtest. [Momentum and Trend Following](../../part-04-strategy-development/01-momentum-and-trend-following.md) sets a falsification standard — "the regression slope of next-month return on the sign of the trailing twelve-month return should be positive for *all three* assets" — and notes in the same breath that random walks and martingales "describe the null world where every such slope is zero." This page builds that null world, and the fifth section shows that in it, a strategy reviewed monthly as its history accumulates produces a significant result $30\%$ of the time.

## A Filtration Is What Point-in-Time Discipline Actually Means

A **filtration** $\{\mathcal{F}_n\}$ is an increasing family of $\sigma$-algebras, with $\mathcal{F}_n$ representing everything knowable at time $n$. Increasing means $\mathcal{F}_n\subseteq\mathcal{F}_{n+1}$: information is never lost. A process $X_n$ is **adapted** if $X_n$ is $\mathcal{F}_n$-measurable — its value at time $n$ is determined by information available at time $n$.

That is not bookkeeping. [Probability Spaces](../part-02-probability-foundations/01-probability-spaces.md) makes the point exactly, that "a backtest that computes a signal from tomorrow's close is conditioning on a set that is not in $\mathcal{F}_t$ — the arithmetic is fine, the measure is fine, and the answer is meaningless because the question was not askable when the trade was placed", and names this page as the one whose every result is stated relative to that hypothesis. Adaptedness is lookahead-freedom given a name.

An adapted, integrable process is a **martingale** if

$$\mathbb{E}[X_{n+1}\mid\mathcal{F}_n]=X_n\quad\text{for every }n,$$

a **submartingale** if the left side is $\geq X_n$ and a **supermartingale** if it is $\leq$. By the tower property of [Conditional Expectation](../part-04-expectation-and-moments/06-conditional-expectation.md) the one-step condition iterates to $\mathbb{E}[X_m\mid\mathcal{F}_n]=X_n$ for all $m>n$, so a martingale's expectation is constant: $\mathbb{E}[X_n]=\mathbb{E}[X_0]$ at every horizon.

Three examples carry most of the weight. A symmetric [random walk](11-random-walks.md) is a martingale, and so is [Brownian motion](08-brownian-motion.md); a walk with drift $\mu$ is a sub- or supermartingale, but $S_n-n\mu$ is a martingale again. The cumulative P&L of any strategy in a market with no edge is a martingale by construction. And a likelihood ratio $L_n=\prod_{i\leq n}q(x_i)/p(x_i)$ is a martingale under $p$ regardless of what $q$ is, which is the fact underneath every sequential test.

The last object needed is a **stopping time**: a random time $\tau$ with $\{\tau\leq n\}\in\mathcal{F}_n$, so that whether you have stopped by time $n$ is decidable from what you know at time $n$. "Sell when the price first exceeds $110$" is a stopping time. "Sell at the high of the year" is not, and the entire difference between an implementable rule and a backtest artifact lives in that distinction.

## No Betting System Makes a Fair Game Favourable

Let $X_n$ be a martingale and let $H_n$ be the stake chosen for period $n$ — the size of the bet, the position, the leverage. The only requirement is that $H_n$ be **predictable**, meaning $\mathcal{F}_{n-1}$-measurable: you must size the bet before seeing its outcome. The resulting wealth is the **martingale transform**

$$(H\cdot X)_n=\sum_{k=1}^{n}H_k\,(X_k-X_{k-1}).$$

??? note "Proof that any predictable strategy applied to a fair game is itself a fair game"
    Condition one increment on the past. Since $H_{n+1}$ is $\mathcal{F}_n$-measurable it comes out of the conditional expectation as a constant:

    $$\mathbb{E}\bigl[(H\cdot X)_{n+1}-(H\cdot X)_n\mid\mathcal{F}_n\bigr]=\mathbb{E}\bigl[H_{n+1}(X_{n+1}-X_n)\mid\mathcal{F}_n\bigr]=H_{n+1}\,\mathbb{E}[X_{n+1}-X_n\mid\mathcal{F}_n]=0,$$

    the last equality being the martingale property. So $(H\cdot X)$ is itself a martingale and $\mathbb{E}[(H\cdot X)_n]=0$ for every $n$ and every predictable $H$. If $X$ is a supermartingale and $H\geq0$, the same computation gives a supermartingale.

    The load-bearing hypothesis is predictability, and it is spent in the single step where $H_{n+1}$ was pulled out of the conditional expectation. **Every betting system, martingale strategy, position-sizing rule, stop-loss, trailing stop, volatility target and regime filter is a choice of $H$**, and none of them can change the expectation, because all of them are functions of information available before the bet. A rule that could would have to depend on $X_{n+1}$, which is not a rule but a wish.

The most famous $H$ is the doubling system: bet one unit, and after each loss double the stake, so that the first win recovers everything and nets one unit. It wins with probability approaching one, and the theorem above says its expectation is zero anyway. Both statements are true simultaneously, and the block shows how.

```python
import numpy as np

rng = np.random.default_rng(8101)
reps = 2_000_000
print(f"  the doubling system on a fair coin, {reps} sessions, one unit sought")
print("      bankroll    P(win a unit)    loss if ruined    mean P&L    theory    sd of P&L")
for k in (3, 6, 10, 14):
    bank = 2 ** k - 1                                          # survives k consecutive losses
    losses = rng.geometric(0.5, reps) - 1                      # losses before the first win
    pnl = np.where(losses < k, 1.0, -float(bank))
    print(f"  {bank:14d} {(losses < k).mean():16.6f} {-bank:16d} {pnl.mean():11.4f}"
          f" {0.0:9.4f} {pnl.std(ddof=1):11.2f}")
# =>   the doubling system on a fair coin, 2000000 sessions, one unit sought
#          bankroll    P(win a unit)    loss if ruined    mean P&L    theory    sd of P&L
#                   7         0.874731               -7     -0.0022    0.0000        2.65
#                  63         0.984224              -63     -0.0096    0.0000        7.97
#                1023         0.999057            -1023      0.0339    0.0000       31.44
#               16383         0.999947           -16383      0.1316    0.0000      119.27
```

The win-rate column is what sells the system: $0.874731$, $0.984224$, $0.999057$, $0.999947$. With a bankroll of $16{,}383$ units the session ends in profit $99.9947\%$ of the time — a strategy that wins nineteen thousand sessions out of every twenty thousand.

The mean P&L column is zero at every row, and it is zero exactly rather than approximately: $-0.0022$, $-0.0096$, $0.0339$, $0.1316$ are sampling noise around a theoretical zero, which the standard-deviation column explains. **Each row trades a larger probability of winning one unit against a proportionally larger loss when it fails, and the product is invariant.** The exact statement is $(1-2^{-k})\cdot1-2^{-k}(2^{k}-1)=0$, so the arithmetic is not close to zero — it is zero.

The last column is what the win rate conceals and is the reason the estimate itself is noisy. The standard deviation of session P&L grows $2.65$, $7.97$, $31.44$, $119.27$ while the mean stays fixed, so the Sharpe ratio of the doubling system is exactly zero at every bankroll and its distribution gets more extreme with every unit of capital committed to it. Two million simulated sessions leave a standard error of $0.084$ on the last row's mean, which is why $0.1316$ appears rather than $0.0000$ — a live desk running this strategy would need decades to discover experimentally what one line of algebra says immediately.

## Optional Stopping Has Three Hypotheses and Losing One Breaks It Completely

If a system cannot change the expectation, perhaps a clever *exit* can. The **optional stopping theorem** says no, under conditions: if $X_n$ is a martingale and $\tau$ a stopping time, then $\mathbb{E}[X_\tau]=\mathbb{E}[X_0]$ provided any one of three things holds — $\tau$ is bounded, or $X_{n\wedge\tau}$ is uniformly bounded, or $\mathbb{E}[\tau]<\infty$ with bounded increments.

Those hypotheses are not decoration. Drop all three and the conclusion fails as completely as it can.

??? note "Proof that the theorem holds for a bounded stopping time, and that the bound is exactly what fails otherwise"
    Suppose $\tau\leq N$ for a constant $N$. Write the stopped process as a martingale transform with the predictable stake $H_k=\mathbf{1}\{\tau\geq k\}$ — which is predictable precisely because $\{\tau\geq k\}=\{\tau\leq k-1\}^{\mathsf{c}}\in\mathcal{F}_{k-1}$, the defining property of a stopping time. Then

    $$X_{\tau}=X_0+\sum_{k=1}^{N}\mathbf{1}\{\tau\geq k\}\,(X_k-X_{k-1})=X_0+(H\cdot X)_N,$$

    and the previous section's result gives $\mathbb{E}[(H\cdot X)_N]=0$, hence $\mathbb{E}[X_\tau]=\mathbb{E}[X_0]$. **A stopping rule is a betting system that bets one unit until it quits**, which is why the two theorems are the same theorem.

    For unbounded $\tau$ the identity $\mathbb{E}[X_{n\wedge\tau}]=\mathbb{E}[X_0]$ still holds for every finite $n$, and the question is whether the limit passes inside the expectation. The three hypotheses are three sufficient conditions for that interchange — dominated convergence in the second case, and a Wald-type argument in the third. When none holds, $X_{n\wedge\tau}\to X_\tau$ pathwise while $\mathbb{E}[X_{n\wedge\tau}]$ does not converge to $\mathbb{E}[X_\tau]$, because mass escapes to infinity.

    The load-bearing hypothesis is whichever one is available, and the failure is always the same shape: the stopped process is *not* uniformly integrable, so an event of vanishing probability carries a growing value and their product does not vanish. The block below watches exactly that product.

```python
import numpy as np

rng = np.random.default_rng(8103)
reps = 200_000
print(f"  a fair walk stopped the first time it reaches +1, {reps} paths, S0 = 0")
print("       horizon    fraction stopped    E[S at min(tau,H)]    mean of the unstopped")
pos = np.zeros(reps, dtype=np.int64)
done = np.zeros(reps, dtype=bool)
val = np.zeros(reps)
for t in range(1, 100_001):
    live = ~done
    pos[live] += 2 * (rng.random(live.sum()) < 0.5).astype(np.int64) - 1
    hit = live & (pos >= 1)
    done |= hit
    if t in (10, 100, 1_000, 10_000, 100_000):
        val = np.where(done, 1.0, pos.astype(float))
        print(f"  {t:13d} {done.mean():19.4f} {val.mean():21.4f}"
              f" {pos[~done].mean():25.2f}")
# =>   a fair walk stopped the first time it reaches +1, 200000 paths, S0 = 0
#           horizon    fraction stopped    E[S at min(tau,H)]    mean of the unstopped
#                 10              0.7531               -0.0039                     -3.07
#                100              0.9196               -0.0049                    -11.50
#               1000              0.9746               -0.0137                    -38.89
#              10000              0.9920                0.0015                   -123.50
#             100000              0.9975                0.0732                   -376.53
```

The strategy is "wait until you are one unit ahead, then stop", applied to a fair game with unlimited patience and unlimited credit. [Brownian Motion](08-brownian-motion.md) established that the barrier is reached with probability one, so $\mathbb{E}[X_\tau]=1$ while $X_0=0$: **the theorem's conclusion is false by a full unit, and the strategy appears to manufacture money from a fair game.**

The third column shows why nothing was manufactured. At every finite horizon $\mathbb{E}[X_{\tau\wedge H}]$ is zero — $-0.0039$, $-0.0049$, $-0.0137$, $0.0015$, $0.0732$, all sampling noise around zero — exactly as the bounded-stopping-time proof requires. Truncate the rule at any horizon whatever and it is worthless.

The fourth column is where the missing unit lives. The fraction still running falls $0.2469$, $0.0804$, $0.0254$, $0.0080$, $0.0025$, while the average position of those still running falls $-3.07$, $-11.50$, $-38.89$, $-123.50$, $-376.53$. The product of the two is very nearly $-1$ at every row, and stays $-1$ forever: a vanishing probability of an unbounded loss, whose contribution to the expectation does not vanish. The hypothesis that fails is $\mathbb{E}[\tau]<\infty$, which [Brownian Motion](08-brownian-motion.md) shows is infinite, and the practical translation is that the strategy requires infinite capital and infinite time to deliver its unit. Every real version of it — with a credit limit, a horizon, or a margin call — is the third column, and the third column is zero.

!!! note "The same theorem run forwards rather than backwards is Wald's identity, which is why a loss limit rescales a P&L without distorting it"
    [Poisson Processes](03-poisson-processes.md) finds that a strategy halting for the day after a large loss still satisfies $\mathbb{E}[S]=\mathbb{E}[N]\mathbb{E}[X]$ — the mean identity survives a stopping rule — and defers the reason here. It is optional stopping applied to the martingale $S_n-n\mathbb{E}[X]$: the halting time is bounded by the day's maximum trade count, so the first hypothesis holds outright and $\mathbb{E}[S_\tau]=\mathbb{E}[\tau]\mathbb{E}[X]$ exactly. That is **Wald's identity**, and its practical content is the reassuring half of this page. A loss limit written down in advance removes trades in proportion and leaves the per-trade expectation alone; it does not secretly select the bad ones, because it cannot see them. What it does change is the variance and the shape, which that page measures, and what it cannot do is manufacture the free unit the next section's third column refuses to deliver.

## Doob's Inequality Bounds the Maximum, and That Bound Is What Peeking Spends

Two further results say what a martingale can do along the way rather than at the end. **Doob's maximal inequality** states that for a non-negative submartingale and any $\lambda>0$,

$$\mathbf{P}\Bigl(\max_{k\leq n}X_k\geq\lambda\Bigr)\leq\frac{\mathbb{E}[X_n]}{\lambda},$$

with an $L^{2}$ version bounding $\mathbb{E}[\max_{k\leq n}X_k^{2}]$ by $4\mathbb{E}[X_n^{2}]$. The content is that a martingale cannot wander far above its terminal expectation without paying for it in probability — but the bound is on the *maximum over the whole path*, not on the value at one time, and the two differ by exactly the factor that the next section measures.

The **martingale convergence theorem** completes the picture: a martingale bounded in $L^{1}$ converges almost surely to a finite limit. A non-negative martingale is automatically bounded in $L^{1}$, so it converges — which for a wealth process that cannot go below zero means the P&L of a fair game settles down rather than oscillating forever. What it does *not* say is that the limit has the same expectation as the start; the doubling system's wealth converges almost surely, and its limit is one unit ahead, and the missing expectation went where the previous section's fourth column went.

## Every Peek Is a Stopping Time Nobody Declared

Put the maximal inequality together with the systems theorem and the practical consequence appears. A backtest's t-statistic, computed as data accumulates, is approximately a martingale under the null of no edge. Testing it once costs the nominal $5\%$. Testing it repeatedly asks about the *maximum* of that process, which is a different and much larger quantity — and stopping when it looks good is a stopping time that was never declared and is not accounted for anywhere in the calculation.

```python
import numpy as np
from scipy.stats import t as tdist

rng = np.random.default_rng(8107)
reps, n = 200_000, 1_260
print(f"  a zero-edge strategy tested repeatedly as its {n} days accumulate, {reps} runs")
print("       peeks    P(reject at some peek)    inflation over 5%")
r = rng.standard_normal((reps, n))
c1, c2 = np.cumsum(r, axis=1), np.cumsum(r * r, axis=1)
k = np.arange(1, n + 1)
for peeks in (1, 5, 20, 60, 252):
    at = np.unique(np.linspace(60, n, peeks).astype(int)) - 1
    m = c1[:, at] / k[at]
    v = (c2[:, at] - k[at] * m * m) / (k[at] - 1)               # unbiased sample variance
    t = m * np.sqrt(k[at] / v)
    crit = tdist.ppf(0.975, k[at] - 1)                          # the exact critical value
    ever = (np.abs(t) > crit).any(axis=1).mean()
    print(f"  {peeks:12d} {ever:25.4f} {ever / 0.05:19.2f}")
# =>   a zero-edge strategy tested repeatedly as its 1260 days accumulate, 200000 runs
#           peeks    P(reject at some peek)    inflation over 5%
#                 1                    0.0494                0.99
#                 5                    0.1618                3.23
#                20                    0.2497                4.99
#                60                    0.3012                6.02
#               252                    0.3457                6.91
```

Every history in this table has a true edge of exactly zero, and the test is correct: looked at once, at the end, it rejects $0.0494$ of the time against a nominal $0.05$, using the exact critical value at each sample size so that nothing is being blamed on small-sample approximation.

Looked at more than once, the same correct test on the same correct data falls apart. Five looks over five years — one per year, which is less scrutiny than any strategy actually receives — reject $0.1618$ of the time. Twenty looks — quarterly review — reach $0.2497$, and monthly review reaches $0.3012$. **Checking weekly reaches $0.3457$, so seven strategies in twenty with no edge whatever will at some point present a significant t-statistic**, and the researcher who stops there has a result.

The inflation is not the familiar multiple-testing problem of trying many strategies, and correcting for the number of *strategies* does not touch it. This is one strategy, one hypothesis, one data set, examined repeatedly as it grows. The looks are heavily dependent — consecutive t-statistics share almost all their data — which is why the rate climbs sub-linearly rather than like $1-0.95^{k}$, and also why no simple Bonferroni over the number of peeks is right either. The correct devices exist and are standard: fix the sample size in advance, or use a sequential boundary such as an alpha-spending function or a confidence sequence that is valid at all times simultaneously. What is not defensible is the default, which is to keep looking and stop when satisfied.

!!! warning "A stopping rule chosen after looking is not a stopping time, and no correction repairs a decision whose rule was never written down"
    The martingale machinery is unusually clear about which practices are recoverable. Stopping according to a rule fixed in advance is fine and the theorem prices it exactly. Stopping when the number looks good is a rule that depends on the realized path in a way that was not declared, so the relevant reference distribution is the one for the maximum rather than the endpoint, and the table above is the cost. **The distinguishing question is whether the rule could have been written down before the data arrived** — and it is the same question [Probability Spaces](../part-02-probability-foundations/01-probability-spaces.md) asks about lookahead, arriving from the direction of the exit rather than the entry. In practice the honest move is to record the stopping rule and the intended sample size at the start of the study, because a rule reconstructed afterwards is indistinguishable from no rule at all, and [Part XV](../part-15-multiple-testing/index.md)'s corrections address a different multiplicity and will not save this one.

## Fair Games Stay Fair, and the Ways Out Are All Accounting

The martingale property is one line and it forecloses a surprising amount. Any predictable strategy applied to a fair game is a fair game, so no sizing rule, stop, filter or system changes the expectation — the doubling system wins $99.9947\%$ of its sessions and has an expectation of exactly zero. Any stopping rule that could actually be implemented, which is to say any bounded one, leaves the expectation alone: the third column of the second block is zero at every horizon, and the apparent free unit lives entirely in a vanishing probability of an unbounded loss that no real balance sheet can carry.

What remains is not a way to beat a fair game but a way to *appear* to. A statistic watched as it accumulates crosses a fixed threshold far more often than a statistic looked at once — $0.3457$ against $0.0494$ at monthly review — because the relevant object is the maximum of the path rather than its endpoint, and Doob's inequality is the formal statement that these are different quantities. Nothing about the data, the test or the strategy is wrong in that table. What is wrong is that the stopping rule was never declared, so the reference distribution being used is the wrong one.

The practical residue is three habits, and each is one sentence. Write the sizing rule and the exit rule down before the data arrives, because that is what makes them predictable and a stopping time respectively. Write the sample size and the review schedule down too, because an undeclared stopping rule converts a $5\%$ test into a $35\%$ one. And when a fair-game argument produces something that looks like free money, find the column that is going to $-376.53$, because in a martingale it is always there. The null world these results describe is the one [Random Walks](11-random-walks.md) builds explicitly, and where its most counterintuitive property — that a fair game spends most of its time on one side — is derived.
