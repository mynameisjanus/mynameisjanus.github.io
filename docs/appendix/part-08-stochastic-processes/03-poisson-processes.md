# Poisson Processes

The Poisson process is what a Bernoulli process becomes when the slots are allowed to shrink to nothing. It is the only arrival process that treats every instant identically and remembers no instant at all, and that double emptiness is exactly why it is the right null for order flow, fills, and jumps — and exactly why the ways real markets depart from it are the interesting quantities rather than the nuisance ones.

This page covers the three definitions of the process and the arguments that make them one definition, the conditional-uniformity theorem that says a count tells you nothing about timing, superposition and thinning and why both are exact here when their discrete-time analogues were not, the scale-dependence of the equidispersion diagnostic and the clustered process that defeats it, and the compound sum whose mean survives a stopping rule while nothing else about it does. It does not develop the count distribution, which is [Poisson Distribution](../part-05-common-distributions/06-poisson-distribution.md), or the gap distribution, which is [Exponential Distribution](../part-05-common-distributions/10-exponential-distribution.md); it does not build the discrete-time ancestor, which is [Bernoulli Processes](02-bernoulli-processes.md); it allows no general gap law, which is [Renewal Processes](04-renewal-processes.md); it gives the rate no state to depend on, which is [Continuous-Time Markov Chains](06-continuous-time-markov-chains.md); and it models no market impact, which is [Part XVIII](../part-18-quant-finance-applications/index.md).

The trading stake is a promise the appendix has already made and a warning it has already issued. [Poisson Distribution](../part-05-common-distributions/06-poisson-distribution.md) defers the continuous-time construction to this part, and its own closing warning says that a dispersion ratio "is one line of code, and a Poisson assumption that has never been checked against it is an untested claim rather than a modelling convenience." That warning is correct and it is not sufficient. The fourth section builds an arrival pattern that passes the dispersion test at the horizon anyone would run it on and is not remotely Poisson, because the Poisson assumption is a claim about arrangement in time and a count is not.

## Three Definitions That Look Unrelated and Describe One Process

A **Poisson process** of rate $\lambda>0$ is a random set of points on $[0,\infty)$, described by the counting process $N(t)$ of points in $[0,t]$. There are three standard definitions and they are equivalent.

The **counting definition** asks for $N(0)=0$, independent increments — counts over disjoint intervals are independent — and $N(t+s)-N(s)$ [Poisson](../part-05-common-distributions/06-poisson-distribution.md) with mean $\lambda t$ regardless of $s$:

$$\mathbf{P}\bigl(N(t+s)-N(s)=k\bigr)=e^{-\lambda t}\frac{(\lambda t)^{k}}{k!}.$$

The **gap definition** asks for the interarrival times $T_1,T_2,\ldots$ to be independent and [exponential](../part-05-common-distributions/10-exponential-distribution.md) with rate $\lambda$, so the $k$-th arrival time $Y_k=T_1+\cdots+T_k$ is Erlang. The **infinitesimal definition** asks that over a window of width $\delta$, independently of everything outside it, exactly one arrival occurs with probability $\lambda\delta+o(\delta)$, none with probability $1-\lambda\delta+o(\delta)$, and two or more with probability $o(\delta)$.

The third is the one that shows where the process comes from. Chop $[0,t]$ into $m$ slots of width $t/m$ and run a [Bernoulli process](02-bernoulli-processes.md) with $p=\lambda t/m$. The count is binomial with mean $\lambda t$, and as $m\to\infty$ the binomial converges to the Poisson — the rare-event limit that [Poisson Distribution](../part-05-common-distributions/06-poisson-distribution.md) derives. Meanwhile the geometric gaps, measured in units of $t/m$, converge to exponential. **Every Bernoulli fact from the previous page survives this limit except one, and the third section is about the exception.**

??? note "Proof that exponential gaps and Poisson counts are the same assumption"
    Assume the gaps are independent exponentials of rate $\lambda$. The events $\{N(t)\geq k\}$ and $\{Y_k\leq t\}$ coincide, since the $k$-th arrival has happened by $t$ exactly when at least $k$ arrivals have. A sum of $k$ independent rate-$\lambda$ exponentials is Erlang with density $\lambda^{k}y^{k-1}e^{-\lambda y}/(k-1)!$, so

    $$\mathbf{P}(N(t)\geq k)=\mathbf{P}(Y_k\leq t)=\int_{0}^{t}\frac{\lambda^{k}y^{k-1}e^{-\lambda y}}{(k-1)!}\,dy,$$

    and integrating by parts $k-1$ times collapses the integral to $1-\sum_{j<k}e^{-\lambda t}(\lambda t)^{j}/j!$. Differencing at $k$ and $k+1$ leaves $\mathbf{P}(N(t)=k)=e^{-\lambda t}(\lambda t)^{k}/k!$, which is the counting definition. Independence of increments follows from the memorylessness of the exponential: conditioned on the history up to $s$, the residual wait to the next arrival is exponential of rate $\lambda$ again, so the process after $s$ is a fresh copy.

    The converse runs backwards through the same identity. $\mathbf{P}(T_1>t)=\mathbf{P}(N(t)=0)=e^{-\lambda t}$ gives the first gap exponential, and applying independent increments on $(t,\infty)$ gives the rest.

    The load-bearing hypothesis is memorylessness, and it is the *only* thing being assumed. It is spent in the step where the process after time $s$ is declared a fresh copy, and the exponential is the unique continuous law that permits it — which is why choosing Poisson counts and choosing memoryless gaps are not two modelling decisions that happen to agree, but one decision stated twice.

## A Count Tells You Nothing About When

The structural theorem of the Poisson process has no analogue among the count distributions and is what makes the process worth naming: **conditioned on $N(t)=n$, the $n$ arrival times are distributed exactly as $n$ independent uniform draws on $[0,t]$, sorted.** The rate cancels. Whatever the arrivals are, once you know how many there were, where they were is a question with the least informative possible answer.

??? note "Proof that arrival times given the count are uniform order statistics"
    Take $n=1$ first. For $s\leq t$, independence of increments and the Poisson law give

    $$\mathbf{P}\bigl(Y_1\leq s\mid N(t)=1\bigr)=\frac{\mathbf{P}(N(s)=1)\,\mathbf{P}(N(t)-N(s)=0)}{\mathbf{P}(N(t)=1)}=\frac{\lambda s e^{-\lambda s}\cdot e^{-\lambda(t-s)}}{\lambda t e^{-\lambda t}}=\frac{s}{t},$$

    the uniform CDF, with every appearance of $\lambda$ cancelling. For general $n$, partition $[0,t]$ by $0<s_1<\cdots<s_n<t$ and ask for one arrival in each infinitesimal window $ds_i$ and none elsewhere. The windows are disjoint, so the probabilities multiply:

    $$\frac{\prod_{i=1}^{n}\lambda\,ds_i\cdot e^{-\lambda t}}{e^{-\lambda t}(\lambda t)^{n}/n!}=\frac{n!}{t^{n}}\,ds_1\cdots ds_n,$$

    which is the joint density of the order statistics of $n$ independent uniforms on $[0,t]$.

    The load-bearing hypothesis is that the rate is *constant*, and the proof shows exactly where it is spent: every $\lambda$ cancels between numerator and denominator only because the same $\lambda$ appears in each window. For a non-homogeneous process with rate $\lambda(s)$ the identical argument goes through with the uniform replaced by the law with density $\lambda(s)/\int_0^t\lambda$, so a time-varying rate is not a departure from the theorem but a reweighting inside it. **Clustering, however, is a departure**, because it makes the joint law of the points fail to factor across disjoint windows at all, and the fourth section measures one.

!!! note "A deterministic intraday rate pattern is not a departure from the Poisson process, it is a change of clock"
    Order flow is not flat across the day — it has the familiar U-shape, heavy at the open and the close. That is a **non-homogeneous** Poisson process, with rate $\lambda(s)$ and counts Poisson with mean $\Lambda(t)=\int_0^t\lambda(s)\,ds$, and it is barely a new object: substituting $\tau=\Lambda(t)$ turns it back into a rate-one homogeneous process. All the structure survives the substitution, which is why the honest way to test for clustering in intraday data is to time-change by the estimated volume curve first and then look for departures. Skipping that step guarantees a finding, because the U-shape alone will register as clustering on any diagnostic. The distinction that matters is between a rate that varies *deterministically in clock time*, which is a reparameterization, and a rate that varies *randomly with the history*, which is not — and only the second is a real departure from independent increments.

Conditional uniformity is the reason superposition and thinning are so easy here. Merging independent Poisson processes of rates $\lambda_1$ and $\lambda_2$ gives a Poisson process of rate $\lambda_1+\lambda_2$ — with no inclusion–exclusion correction, because two arrivals at the same instant have probability zero. Thinning a rate-$\lambda$ process by an independent coin of probability $q$ gives a Poisson process of rate $\lambda q$, and the kept and discarded streams are **independent**. That last claim is the one that failed on the grid.

## Thinning Gives Independent Streams Here, and That Is What the Grid Was Costing

[Bernoulli Processes](02-bernoulli-processes.md) proves that splitting a discrete stream leaves the two halves correlated at exactly $-p/(2-p)$, because a slot holding one arrival cannot hold another. Shrinking the slots removes the constraint. The block below runs the same split on progressively finer grids at a fixed arrival rate, then runs it with no grid at all.

```python
import numpy as np

rng = np.random.default_rng(8031)
reps, lam = 2_000_000, 2.0
print(f"  splitting a rate-{lam} stream on the unit interval with a fair coin, {reps} runs")
print("      slots    p per slot    corr(N1, N2)    -p/(2-p)")
for m in (4, 16, 64, 256):
    p = lam / m
    n1 = np.zeros(reps, dtype=np.int32)
    n2 = np.zeros(reps, dtype=np.int32)
    for _ in range(m):
        s, c = rng.random(reps) < p, rng.random(reps) < 0.5
        n1 += s & c
        n2 += s & ~c
    print(f"  {m:9d} {p:13.5f} {np.corrcoef(n1, n2)[0, 1]:15.4f} {-p / (2 - p):11.4f}")
n = rng.poisson(lam, reps)
h = rng.binomial(n, 0.5)                                       # the same coin, no grid at all
print(f"  no grid at all {np.corrcoef(h, n - h)[0, 1]:25.4f} {0.0:12.4f}")
# =>   splitting a rate-2.0 stream on the unit interval with a fair coin, 2000000 runs
#          slots    p per slot    corr(N1, N2)    -p/(2-p)
#              4       0.50000         -0.3336     -0.3333
#             16       0.12500         -0.0670     -0.0667
#             64       0.03125         -0.0165     -0.0159
#            256       0.00781         -0.0041     -0.0039
#      no grid at all                   -0.0001       0.0000
```

The arrival rate is held at $2.0$ per unit time down every row; only the discretization changes. The correlation tracks the Bernoulli formula the whole way — $-0.3336$ against $-0.3333$, $-0.0670$ against $-0.0667$, $-0.0165$ against $-0.0159$, $-0.0041$ against $-0.0039$ — shrinking linearly in the slot width because that is how fast the slots empty out.

The last row is the process this page is about, and it reads $-0.0001$ against a true value of zero. **The independence of thinned streams is not an approximation that improves as the grid refines; it is exact once the grid is gone**, and the entire discrepancy on the rows above was an artifact of pretending time comes in pieces. That is the practical reason to work with the continuous-time object even when the data arrives on a grid: the theorems are clean, and the grid was contributing dependence that has nothing to do with the market.

The trading reading is direct. If a signal generator fires as a Poisson process and a risk filter independently vetoes a fraction of its signals, the surviving stream is Poisson at the reduced rate, and — the part that is genuinely useful — the vetoed stream carries no information about the surviving one. Counting how many signals were rejected tells you nothing about how many were taken. On a coarse grid with a high firing rate that statement is false, by an amount this table quantifies.

## A Poisson Count Is Not a Poisson Process

Equidispersion, $\mathrm{var}(N)=\mathbb{E}[N]$, is the standard Poisson diagnostic, and [Poisson Distribution](../part-05-common-distributions/06-poisson-distribution.md) is right that it is cheap and right that mixing the rate can only inflate it. But it is a test of a *count*, and the Poisson process is a claim about *arrangement*. The two come apart, and the following construction shows how far.

Both patterns below place exactly thirty arrivals in each day, so their daily count distributions are not merely similar but identical, and any dispersion ratio computed at the daily horizon is exactly zero for both. One scatters the thirty uniformly. The other places ten bursts of three.

```python
import numpy as np

rng = np.random.default_rng(8033)
days, n = 100_000, 30
print(f"  {days} days carrying exactly {n} arrivals each, so the daily count says nothing")
print("       bins    expected per bin    dispersion, uniform    1 - 1/bins    clustered")
u = rng.random((days, n))
ctr = rng.random((days, n // 3))                                # ten bursts of three
v = np.clip(np.repeat(ctr, 3, axis=1) + 0.004 * rng.standard_normal((days, n)), 0, 1)
row = np.arange(days)[:, None]
for b in (1, 4, 20, 100):
    out = []
    for a in (u, v):
        idx = np.minimum((a * b).astype(np.int64), b - 1) + b * row
        c = np.bincount(idx.ravel(), minlength=days * b)
        out.append(c.var() / c.mean())
    print(f"  {b:11d} {n / b:19.2f} {out[0]:22.4f} {1 - 1 / b:13.4f} {out[1]:12.4f}")
# =>   100000 days carrying exactly 30 arrivals each, so the daily count says nothing
#           bins    expected per bin    dispersion, uniform    1 - 1/bins    clustered
#                1               30.00                 0.0000        0.0000       0.0000
#                4                7.50                 0.7488        0.7500       2.2182
#               20                1.50                 0.9509        0.9500       2.6757
#              100                0.30                 0.9912        0.9900       2.1084
```

The first row is the indictment. At the daily horizon both processes return a dispersion of exactly $0.0000$, because the count was fixed by construction, and a desk running the recommended one-line check at the horizon it reports its data on would conclude that both series are as Poisson as anything can be.

The uniform column then does something worth pausing on: it reads $0.7488$, $0.9509$, $0.9912$ against the values $1-1/b$ printed beside it. Conditioning on a fixed total makes the sub-window counts multinomial rather than Poisson, and multinomial counts are *under*-dispersed by exactly the factor $1-1/b$. The agreement to three decimals at every row confirms the arrangement is genuinely uniform, and it also means the correct null for this statistic is $1-1/b$ rather than $1$ — a correction of $25\%$ at four bins that nobody applies.

The clustered column never comes close: $2.2182$, $2.6757$, $2.1084$, between two and three times the null at every sub-daily resolution. **The same data is equidispersed at one horizon and three times over-dispersed at another, so a dispersion ratio without a stated bin width is not a diagnostic.** Real order flow behaves like the second column — trades arrive in bursts triggered by other trades — and a Poisson model fitted to daily volume will reproduce the daily volume and understate the probability of a one-minute burst by orders of magnitude.

!!! warning "Every Poisson diagnostic is a diagnostic at one timescale, and the timescale that matters is the one the risk is taken on"
    A count is a projection, and projecting destroys arrangement. The clustered pattern above survives the equidispersion test at the daily scale, survives any test of the marginal count law at that scale, and would survive a goodness-of-fit test on daily volume with room to spare — while being wrong by a factor of two to three about exactly the quantity an execution desk cares about, which is how much arrives in the next minute. The rule is that the diagnostic must be run at the resolution at which the model will be used, and the honest report states the bin width alongside the ratio. The general form of the failure is [Conditional Distributions](../part-03-random-variables/07-conditional-distributions.md)'s point that one marginal is consistent with many joint laws; here the marginal is the count and the joint law is the timing.

## A Compound Sum Keeps Its Mean Under a Stopping Rule and Loses Everything Else

Attach a mark to each arrival — the P&L of the trade that fired — and the object of interest is the **compound Poisson** sum $S=\sum_{i=1}^{N}X_i$. When $N$ is independent of the marks, [Law of Total Expectation](../part-04-expectation-and-moments/07-law-of-total-expectation.md) and [Law of Total Variance](../part-04-expectation-and-moments/08-law-of-total-variance.md) give

$$\mathbb{E}[S]=\mathbb{E}[N]\,\mathbb{E}[X],\qquad \mathrm{var}(S)=\mathbb{E}[N]\,\mathrm{var}(X)+\mathrm{var}(N)\,\mathbb{E}[X]^{2}.$$

That independence is the hypothesis [Law of Total Expectation](../part-04-expectation-and-moments/07-law-of-total-expectation.md) flags as failing for "a strategy that halts for the day after a large loss, or that scales down after a drawdown", and it names this page for the count process and [Martingales](10-martingales.md) for the repair. The block runs both.

```python
import numpy as np

rng = np.random.default_rng(8037)
reps, lam, mu, sd, K = 200_000, 12.0, 0.02, 1.00, 60
print(f"  daily P&L as a compound sum, {reps} days, Poisson({lam}) trades of mean {mu}")
print("        rule    E[N]    E[S]    E[N]E[X]    var[S]    formula    1% worst day")
idx = np.arange(K)
x = mu + sd * rng.standard_normal((reps, K))
n = rng.poisson(lam, reps)
live = idx[None, :] < n[:, None]
c = np.where(live, x, 0.0).cumsum(axis=1)
hit = (c < -2.0) & live                                        # the day's loss limit is breached
first = np.where(hit.any(axis=1), hit.argmax(axis=1), K - 1)
keep = live & (idx[None, :] <= first[:, None])                 # halt after the breaching trade
for name, m in (("free", live), ("halt at -2", keep)):
    s, cnt = np.where(m, x, 0.0).sum(axis=1), m.sum(axis=1)
    var = cnt.mean() * sd ** 2 + cnt.var() * mu ** 2
    print(f"  {name:>10s} {cnt.mean():7.3f} {s.mean():7.4f} {cnt.mean() * mu:11.4f}"
          f" {s.var():9.4f} {var:10.4f} {np.quantile(s, 0.01):15.4f}")
# =>   daily P&L as a compound sum, 200000 days, Poisson(12.0) trades of mean 0.02
#            rule    E[N]    E[S]    E[N]E[X]    var[S]    formula    1% worst day
#            free  12.001  0.2389      0.2400   11.9939    12.0059         -8.0188
#      halt at -2   9.045  0.1804      0.1809    9.3606     9.0529         -3.8560
```

The free row is the identity working. $\mathbb{E}[S]=0.2389$ against $\mathbb{E}[N]\mathbb{E}[X]=0.2400$, and $\mathrm{var}(S)=11.9939$ against a predicted $12.0059$ — both within sampling error on $200{,}000$ days.

The halting row is more interesting than the warning it was built to illustrate. The mean identity *survives*: $0.1804$ against $0.1809$. That is not luck and it is not the independence assumption sneaking back in — it is Wald's identity, which holds because the halting time is a legitimate stopping time, decidable from the trades already taken. The general statement and its proof are [Martingales](10-martingales.md); the practical form is that **a stopping rule rescales the expected P&L in exact proportion to the trades it removes, and buys nothing at all on the mean.**

Everything else moves. The variance identity fails — $9.3606$ realized against $9.0529$ predicted, a $3.4\%$ understatement, because the formula assumes a count independent of the marks and this count runs longest exactly on the days the marks were kind. And the shape moves far more than either moment reports: the $1\%$ worst day improves from $-8.0188$ to $-3.8560$, less than half the loss.

The trap is the number a desk would actually quote. Planning assumed $12$ trades a day at $\mathbb{E}[X]=0.02$, so the budgeted daily P&L is $0.2400$; the rule delivers $0.1804$, a quarter less, because $2.955$ trades a day never happened. The tail improved, the mean fell in proportion, and the only quantity the rule ever protected is the one it was written to protect.

## The Assumption Is About Arrangement, Not About Counts

A Poisson process makes exactly one claim, in three interchangeable dialects: what happens in disjoint stretches of time is independent, and the rate is the same everywhere. Everything else on this page is a consequence — the exponential gaps, the uniform scatter given the count, the exactness of merging and thinning, the compound moments.

The consequence worth carrying is that most of the diagnostics people run test the claim's *shadow*. A count is the process integrated over a window, and integrating destroys the arrangement the assumption is actually about, which is why the fourth section's clustered pattern passes at one horizon and fails by a factor of three at another. The check with content is one that looks at the arrangement: bin at the resolution the model will be used at, compare against $1-1/b$ rather than $1$ when the total is conditioned on, and print the bin width next to the ratio.

Where the process fails, it fails in one direction. Mixing the rate inflates the variance, clustering inflates it further, and nothing in a market produces arrivals more evenly spaced than independent ones. So a fitted Poisson is systematically optimistic about bursts and never pessimistic, and the size of that optimism is not a modelling defect but an estimate of something real — the degree to which trades beget trades. The two families that put a parameter on it are [Renewal Processes](04-renewal-processes.md), which frees the gap law while keeping the restarts, and [Continuous-Time Markov Chains](06-continuous-time-markov-chains.md), which lets the rate depend on a state.
