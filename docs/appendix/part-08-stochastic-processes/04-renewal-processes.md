# Renewal Processes

A renewal process is a Poisson process with the memorylessness taken out: arrivals still restart the clock, but the gap between them is allowed any distribution at all. That single relaxation costs almost every exact formula of the previous page and buys the only class of arrival model in which the phrases "average holding period" and "trades per year" can be given honest definitions — which turns out to matter, because both are routinely computed in ways that are wrong by factors of two and six.

This page covers the definition and the renewal function, the elementary renewal theorem that fixes the long-run rate at one over the mean gap, the inspection paradox and the length-biased sampling that produces it, the renewal-reward theorem together with the annualization error it forbids, and the renewal central limit theorem that puts an error bar on a trade count. It does not assume memoryless gaps, which is [Poisson Processes](03-poisson-processes.md); it does not prove the laws of large numbers it consumes, which are [Part VII](../part-07-asymptotic-theory/index.md); it attaches no state to the process, which is [Markov Chains](05-markov-chains.md) and [Continuous-Time Markov Chains](06-continuous-time-markov-chains.md); it derives no hitting time for a price, which is [Brownian Motion](08-brownian-motion.md); and it develops no queueing theory at all.

The trading stake is the arithmetic behind every "expected annual return" a strategy has ever been sold on. A mean-reversion book holds positions for a random number of days — [Mean Reversion and Pairs Trading](../../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md) builds its whole tempo on a half-life, noting that "that single number dictates the trade's entire tempo: how long positions are held, how fast the z-score window may be" — and the figure quoted to an investor is a per-trade edge scaled up by a trade count. The fourth section shows that the usual way of performing that multiplication overstates the answer by a factor of $6.26$ on holding periods no more dispersed than a real book's.

## Freeing the Gap Law Keeps the Fresh Start at Arrivals and Loses It Everywhere Else

A **renewal process** is defined by independent, identically distributed, strictly positive interarrival times $T_1,T_2,\ldots$ with common distribution $F$, mean $\mu=\mathbb{E}[T]$ and variance $\sigma^{2}$. The arrival times are the partial sums $Y_k=T_1+\cdots+T_k$ — the **renewal epochs** — and the counting process is

$$N(t)=\max\{k:Y_k\leq t\},$$

the number of renewals by time $t$. Taking $F$ exponential recovers the [Poisson process](03-poisson-processes.md) exactly; taking $F$ degenerate at $\mu$ gives a deterministic metronome; everything else is new territory. The **renewal function** is $m(t)=\mathbb{E}[N(t)]$, and it satisfies the renewal equation $m(t)=F(t)+\int_0^t m(t-s)\,dF(s)$ — condition on the first gap, and after it the process starts over.

That "starts over" is the whole structure and it is weaker than it was. A Poisson process regenerates at *every* time $t$, because the residual wait is exponential no matter how long you have already waited. A renewal process regenerates only at the epochs $Y_k$, and at any other instant the process carries real information: how long it has been since the last arrival. **Memorylessness was never a property of arrivals; it was a property of the exponential**, and removing it means that where you are inside a gap matters. The third section is what that costs.

The one thing that survives intact is the long-run rate.

??? note "Proof that the long-run arrival rate is one over the mean gap, whatever the gap distribution"
    By definition $Y_{N(t)}\leq t<Y_{N(t)+1}$ — the last renewal before $t$ and the first after it. Divide through by $N(t)$:

    $$\frac{Y_{N(t)}}{N(t)}\leq\frac{t}{N(t)}<\frac{Y_{N(t)+1}}{N(t)+1}\cdot\frac{N(t)+1}{N(t)}.$$

    Since the gaps are strictly positive with finite mean, $N(t)\to\infty$ almost surely as $t\to\infty$. The [Strong Law of Large Numbers](../part-07-asymptotic-theory/02-strong-law-of-large-numbers.md) applied to the gaps gives $Y_n/n\to\mu$ almost surely, so both outer terms converge to $\mu$ — the right-hand one because $(N(t)+1)/N(t)\to1$ — and the sandwich forces $t/N(t)\to\mu$. Therefore

    $$\frac{N(t)}{t}\ \longrightarrow\ \frac{1}{\mu}\qquad\text{almost surely},$$

    and a separate argument using Wald's identity upgrades this to the **elementary renewal theorem** $m(t)/t\to1/\mu$ for the expectation as well.

    The load-bearing hypothesis is a finite mean gap, and it is the *only* hypothesis: no variance, no shape, no independence beyond that of the gaps themselves. That is why the result is so often over-read. **The rate depends on the mean gap alone, and essentially nothing else about the process does** — the next two sections are two quantities that get computed as though they were rates and that depend on the second moment instead.

## The Gap You Land In Is Longer Than the Average Gap

Pick a random instant and ask how long the gap containing it is. The answer is not $\mu$, and the reason is elementary once stated: long gaps cover more of the timeline than short ones, so a randomly chosen instant is more likely to fall inside a long one. Sampling by time samples gaps in proportion to their length.

??? note "Proof that the gap straddling a random instant has mean E[T squared] over E[T]"
    In the long run, gaps of length near $x$ occur with frequency $dF(x)$ and each occupies $x$ units of the timeline. The fraction of the timeline covered by such gaps is therefore $x\,dF(x)/\mu$, so a uniformly chosen instant falls inside a gap whose length has the **length-biased density**

    $$dF^{*}(x)=\frac{x\,dF(x)}{\mathbb{E}[T]}.$$

    Its mean is the quantity advertised:

    $$\mathbb{E}[T^{*}]=\int x\,dF^{*}(x)=\frac{\mathbb{E}[T^{2}]}{\mathbb{E}[T]}=\mu\Bigl(1+\frac{\sigma^{2}}{\mu^{2}}\Bigr)=\mu\,(1+c^{2}),$$

    writing $c=\sigma/\mu$ for the coefficient of variation. Since $c^{2}\geq0$ always, $\mathbb{E}[T^{*}]\geq\mu$ with equality **only** when $\sigma=0$ — a perfectly regular metronome. The exponential case has $c=1$ and gives $\mathbb{E}[T^{*}]=2\mu$: land at a random instant in a Poisson process and the gap you are inside averages twice the mean gap, which is the memorylessness of the exponential seen from outside, since the wait back to the last arrival and the wait forward to the next are both full exponentials.

    The load-bearing hypothesis is that the sampling instant is chosen by *time* rather than by *arrival*, and it is spent in the weighting $x\,dF(x)$. Choose a gap by picking a renewal index at random and the answer is $\mu$ again. The paradox is entirely a statement about which sampling scheme was used, and neither scheme is wrong — they answer different questions and are constantly swapped.

```python
import numpy as np

rng = np.random.default_rng(8041)
n, probes = 2_000_000, 2_000_000
print(f"  the gap you land in versus the average gap, {n} renewals of mean 1")
print("       gap cv    mean gap    mean sampled gap    E[T^2]/E[T]    ratio")
for k in (1e9, 4.0, 1.0, 0.25):
    t = rng.gamma(k, 1 / k, n)
    y = np.cumsum(t)
    u = rng.uniform(y[n // 10], y[-1], probes)                 # a random instant, well inside
    g = t[np.searchsorted(y, u)]                               # the gap straddling that instant
    print(f"  {1 / np.sqrt(k):12.4f} {t.mean():11.4f} {g.mean():19.4f}"
          f" {1 + 1 / k:14.4f} {g.mean() / t.mean():8.4f}")
# =>   the gap you land in versus the average gap, 2000000 renewals of mean 1
#           gap cv    mean gap    mean sampled gap    E[T^2]/E[T]    ratio
#            0.0000      1.0000              1.0000         1.0000   1.0000
#            0.5000      0.9999              1.2496         1.2500   1.2497
#            1.0000      0.9989              1.9959         2.0000   1.9981
#            2.0000      1.0017              5.0008         5.0000   4.9925
```

Every row draws gaps with mean $1$; only the dispersion changes. The mean gap column confirms it, sitting at $1.0000$, $0.9999$, $0.9989$, $1.0017$. The sampled column is what an observer arriving at a random moment measures, and it reads $1.0000$, $1.2496$, $1.9959$, $5.0008$ against the predicted $\mu(1+c^{2})$ printed beside it — agreement to three decimals at every dispersion.

The extremes are the lesson. A metronome shows no bias at all: with $c=0$ the gap you land in is the only gap there is. At $c=1$, which is the Poisson case and roughly what a well-behaved trade log looks like, **the interval you sample is exactly twice the interval that actually occurs**. At $c=2$ it is five times.

The practical form of this is a measurement error nobody suspects. Ask "how long does this strategy hold a position?" and there are two ways to answer: average the durations in the trade log, which gives $\mu$, or sample calendar days and record the duration of whatever trade was open, which gives $\mu(1+c^{2})$. The second is what a risk system that snapshots positions daily reports, and on a book with $c=1$ it will state a holding period twice the truth while both numbers are computed correctly from the same data.

!!! warning "Length-biased sampling is not confined to durations, and it is invisible because both numbers are correct"
    The same weighting appears wherever the observation scheme selects by size. Sampling trades by *volume* over-represents large trades, so an average trade size taken from a volume-weighted feed exceeds the average trade size. Sampling positions by *time held* over-represents the ones that went wrong, since losers are held longer. Sampling funds by *assets under management* over-represents the survivors. In each case the biased number is not an arithmetic error, and no amount of care in the computation will fix it, because the bias entered when the sample was drawn. The question to ask of any average is which measure it was taken under — one draw per arrival, or one per unit of time — and the gap between the two is $c^{2}$, which is a number the data will tell you.

## The Standard Way to Annualize a Trade Is Wrong by the Same Second Moment

The **renewal-reward theorem** is the reason this class of processes is worth carrying. Attach a reward $R_k$ to the $k$-th cycle, allowed to depend on that cycle's length in any way whatever, and let $C(t)$ be the total reward by time $t$. Then

$$\frac{C(t)}{t}\ \longrightarrow\ \frac{\mathbb{E}[R]}{\mathbb{E}[T]}\qquad\text{almost surely}.$$

The long-run rate is the mean reward per cycle over the mean cycle length. Note what the statement permits: $R$ and $T$ may be arbitrarily dependent — a stop-loss makes losing trades short, a trailing stop makes winners long — and the formula does not care, because numerator and denominator are per-cycle averages that the strong law delivers separately.

Note also what it forbids, which is the thing everyone does. The correct annualization is a **ratio of means**. The number computed from a trade log is usually a **mean of ratios**: take each trade's return, scale it to an annual rate by dividing by its own holding period, and average those. The two are different quantities, and by Jensen they differ in a known direction.

```python
import numpy as np

rng = np.random.default_rng(8043)
n, hold, edge = 4_000_000, 10.0, 0.004
print(f"  annualizing {n} trades of mean return {edge} and mean holding period {hold} days")
print("       duration cv    E[R]/E[T]    mean of R/T    ratio    k/(k-1)")
for k in (1e9, 4.0, 2.0, 1.2):
    d = rng.gamma(k, hold / k, n)                              # holding period, days
    r = edge + 0.02 * rng.standard_normal(n)                   # return over the whole trade
    right, naive = 252 * r.mean() / d.mean(), 252 * (r / d).mean()
    print(f"  {1 / np.sqrt(k):17.4f} {right:12.4f} {naive:14.4f}"
          f" {naive / right:8.4f} {k / (k - 1):10.4f}")
# =>   annualizing 4000000 trades of mean return 0.004 and mean holding period 10.0 days
#           duration cv    E[R]/E[T]    mean of R/T    ratio    k/(k-1)
#                 0.0000       0.1008         0.1008   1.0000     1.0000
#                 0.5000       0.1006         0.1345   1.3373     1.3333
#                 0.7071       0.1007         0.1989   1.9746     2.0000
#                 0.9129       0.1006         0.6295   6.2552     6.0000
```

The truth is constant down the table. The strategy earns $0.004$ per trade and holds for $10$ days on average, so the renewal-reward rate is $252\times0.004/10\approx0.1008$, and the third column returns $0.1008$, $0.1006$, $0.1007$, $0.1006$ regardless of how dispersed the durations are. Dispersion in holding period does not change what the book earns.

The fourth column is what gets reported. At a duration coefficient of variation of $0.5$ — a tame book — the mean-of-ratios annualization returns $0.1345$ against a truth of $0.1006$, a $34\%$ overstatement. At $0.71$ it doubles the answer. At $0.91$, still short of the exponential's $c=1$, **it reports $0.6295$ where the strategy earns $0.1006$, overstating the annual return by a factor of $6.26$.** The last column is the closed form for gamma durations, $k/(k-1)$, and the simulation tracks it at $1.0000$, $1.3373$, $1.9746$, $6.2552$ against $1.0000$, $1.3333$, $2.0000$, $6.0000$.

!!! note "The theorem tolerates any dependence between a trade's reward and its duration, which is the one thing practitioners expect it to forbid"
    Every risk rule in a real book couples $R$ and $T$ tightly, and always in the same direction. A stop-loss terminates losers early, so large negative rewards arrive with short durations. A profit target terminates winners early. A trailing stop lets winners run, so large positive rewards arrive with long durations. The instinct is that this dependence must invalidate the rate formula, and it does not: $\mathbb{E}[R]/\mathbb{E}[T]$ needs the *cycles* to be independent of one another, not the reward to be independent of the duration inside a cycle. Both expectations are ordinary per-cycle averages, and the [Strong Law of Large Numbers](../part-07-asymptotic-theory/02-strong-law-of-large-numbers.md) delivers each one separately without ever forming their joint law. What the dependence does change is the *variance* of the realized rate over a finite horizon, and that is a different question from the one the theorem answers — which is exactly the distinction [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md) draws between a converging estimator and a trustworthy error bar.

The mechanism behind the fourth column is that $\mathbb{E}[1/T]>1/\mathbb{E}[T]$, strictly, whenever $T$ varies. Dividing by the holding period gives the shortest trades enormous weight — a trade that made $0.4\%$ in one day annualizes to over $100\%$ — and averaging those inflated numbers estimates nothing the book will ever experience. The drift between the simulated $6.2552$ and the exact $6.0000$ in the last row is itself the warning: as $k\to1$ the quantity $\mathbb{E}[1/T]$ diverges, so once durations can be arbitrarily short the estimator has no finite mean at all, and its sample value is climbing rather than converging.

## A Trade Count Is a Random Variable and Usually Reported as a Constant

The elementary renewal theorem says $N(t)/t\to1/\mu$, and the first section warned that this is a statement about a mean and nothing else. The spread around it comes from the **renewal central limit theorem**: for large $t$,

$$N(t)\ \approx\ \mathcal{N}\Bigl(\frac{t}{\mu},\ \frac{t\sigma^{2}}{\mu^{3}}\Bigr),$$

which follows from applying [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md) to the arrival times and inverting the relation between $N(t)$ and $Y_k$. The variance carries $\sigma^{2}$, so the same second moment that drove the last two sections drives this one too.

```python
import numpy as np

rng = np.random.default_rng(8047)
reps, year, hold, K = 200_000, 252.0, 10.0, 400
print(f"  trades completed in {year:.0f} days when the mean holding period is {hold} days")
print("       duration cv    mean count    t/mu    sd of count    sqrt(t s^2/mu^3)")
for k in (1e9, 4.0, 1.0, 0.25):
    d = rng.gamma(k, hold / k, (reps, K))
    n = (np.cumsum(d, axis=1) <= year).sum(axis=1)             # renewals inside the year
    s2 = hold ** 2 / k
    print(f"  {1 / np.sqrt(k):17.4f} {n.mean():13.4f} {year / hold:8.4f}"
          f" {n.std(ddof=1):14.4f} {np.sqrt(year * s2 / hold ** 3):18.4f}")
# =>   trades completed in 252 days when the mean holding period is 10.0 days
#           duration cv    mean count    t/mu    sd of count    sqrt(t s^2/mu^3)
#                 0.0000       25.0000  25.2000         0.0000             0.0002
#                 0.5000       24.8206  25.2000         2.5164             2.5100
#                 1.0000       25.1972  25.2000         5.0266             5.0200
#                 2.0000       26.6991  25.2000         9.9725            10.0399
```

The theory column is reproduced closely: $2.5164$ against $2.5100$, $5.0266$ against $5.0200$, $9.9725$ against $10.0399$. A business plan that says "about twenty-five trades a year" is describing a quantity whose standard deviation is $5.03$ at exponential dispersion — so a year with $15$ trades and a year with $35$ are both entirely ordinary, and neither is evidence that anything changed.

The metronome row deserves a sentence. At $c=0$ the count is $25.0000$ with a standard deviation of exactly zero: a deterministic ten-day cycle completes exactly $25$ trades in $252$ days, every year, forever. All of the uncertainty in a trade count comes from dispersion in the gaps and none of it from the arrivals being "random" in any other sense.

The last row shows the theorem's own limits. The mean count is $26.6991$ where $t/\mu$ predicts $25.2000$ — a $6\%$ discrepancy that is not sampling error but the asymptotic approximation failing to have arrived, because at $c=2$ a year contains only about twenty-five gaps and many of them are tiny. **A rate theorem is asymptotic in the number of cycles, not in the number of days**, and a strategy that trades twenty-five times a year has a small sample by the only measure that matters here.

## Two Numbers, Two Measures, and the Second Moment Between Them

Everything on this page is one arithmetic fact wearing three costumes. The mean gap $\mu$ governs the long-run rate, and it is the only thing that does. The second moment $\mathbb{E}[T^{2}]$ governs the gap you land in, the annualization error, and the variance of the trade count — and it appears in none of the formulas anyone quotes.

The operational rule is to ask, of every average computed from a trade log, which measure it was taken under. Averaging over arrivals gives $\mu$; averaging over time gives $\mu(1+c^{2})$; averaging ratios gives something with no clean interpretation and an upward bias that grows without bound as durations disperse. The correct annualization is total reward over total time — the single ratio $\mathbb{E}[R]/\mathbb{E}[T]$ the renewal-reward theorem licenses — and it can be computed from any trade log in one line, by summing the P&L and dividing by the elapsed calendar, with no per-trade scaling anywhere in the calculation.

That construction is worth stating as the recommendation, because it is immune to all three failures at once. It never samples by time, so it cannot be length-biased. It never forms a per-trade rate, so it cannot be Jensen-inflated. And it makes the trade count irrelevant rather than assumed, which removes a standard deviation of $5.03$ from a number that was being quoted to four significant figures. The processes that come next replace the free gap distribution with a state — [Markov Chains](05-markov-chains.md) in discrete time and [Continuous-Time Markov Chains](06-continuous-time-markov-chains.md) in continuous — and the sojourn times they produce are a special case of the gaps here, with the shape no longer free.
