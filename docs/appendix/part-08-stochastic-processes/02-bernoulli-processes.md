# Bernoulli Processes

A **Bernoulli process** is a sequence of independent coin flips with success probability $p$ — the simplest arrival process there is, the discrete-time ancestor of the Poisson process, and the null model against which every claim about streaks, hot hands, and being due for a bounce has to be read. Its whole content is that nothing carries over from one slot to the next, and almost everything surprising about it comes from how much structure that assumption still manages to produce.

This page covers the three equivalent descriptions of the process and the derivations that connect them, the fresh-start property and the memorylessness of the geometric interarrival law, the distribution of the longest run and why it is so much longer than intuition allows, the downward bias in the obvious estimator of a conditional win rate, and the merging and splitting of streams together with the one place where the discrete construction differs from its continuous-time limit. It does not develop the distributions it uses, which are [Bernoulli](../part-05-common-distributions/01-bernoulli-distribution.md), [Binomial](../part-05-common-distributions/02-binomial-distribution.md), [Geometric](../part-05-common-distributions/03-geometric-distribution.md) and [Negative Binomial](../part-05-common-distributions/04-negative-binomial-distribution.md); it does not take the rare-event limit that produces continuous time, which is [Poisson Processes](03-poisson-processes.md); it allows no dependence between slots, which is [Markov Chains](05-markov-chains.md); it proves no law of large numbers, which is [Part VII](../part-07-asymptotic-theory/index.md); and it corrects no multiplicity, which is [Part XV](../part-15-multiple-testing/index.md).

The trading stake is a number the appendix has already published. [The Strong Law of Large Numbers](../part-07-asymptotic-theory/02-strong-law-of-large-numbers.md) runs zero-edge strategies for twenty-five years and finds that while only $0.0063$ of them *finish* with a Sharpe above $0.5$, "one in twenty-three touches $2.0$, a number that would end an allocator's diligence, and not one of them still has it at the end." That page explains the gap in terms of a running maximum. This page explains it in terms of the thing an allocator actually looks at — the streak — and the third section shows that a track record of $1{,}260$ trades with no edge whatsoever contains a ten-trade winning run almost half the time.

## One Process, Three Descriptions, Each Determining the Other Two

Formally, a Bernoulli process is a sequence $X_1,X_2,\ldots$ of [independent](../part-02-probability-foundations/05-independence.md) random variables with $\mathbf{P}(X_i=1)=p$ and $\mathbf{P}(X_i=0)=1-p$. Slot $i$ is an **arrival** when $X_i=1$. That is the first description, and the two that follow are not additional assumptions but consequences.

The **count** $N_t=X_1+\cdots+X_t$ of arrivals in the first $t$ slots is [binomial](../part-05-common-distributions/02-binomial-distribution.md) with parameters $t$ and $p$, so $\mathbb{E}[N_t]=tp$ and $\mathrm{var}(N_t)=tp(1-p)$. Counts over disjoint blocks of slots are independent, because they are functions of disjoint sets of independent variables — the property that will be promoted to an axiom when the process is rebuilt in continuous time.

The **interarrival times** $T_1,T_2,\ldots$, where $T_1$ is the slot of the first arrival and $T_k$ the number of slots from the $(k-1)$-th arrival to the $k$-th, are independent and [geometric](../part-05-common-distributions/03-geometric-distribution.md) with parameter $p$: $\mathbf{P}(T=k)=(1-p)^{k-1}p$ and $\mathbb{E}[T]=1/p$. The **arrival times** are their partial sums,

$$Y_k=T_1+\cdots+T_k,$$

and $Y_k$ is [Pascal](../part-05-common-distributions/04-negative-binomial-distribution.md) — negative binomial of order $k$ — with

$$p_{Y_k}(t)=\binom{t-1}{k-1}p^{k}(1-p)^{t-k},\qquad t=k,k+1,\ldots$$

??? note "Proof that the k-th arrival time has the Pascal law, counted two ways"
    The direct argument counts sequences. The event $\{Y_k=t\}$ says slot $t$ is an arrival and exactly $k-1$ of the preceding $t-1$ slots are arrivals. The first requirement costs a factor $p$. The second is a binomial event on $t-1$ independent slots with $k-1$ successes, contributing $\binom{t-1}{k-1}p^{k-1}(1-p)^{t-k}$, and the two are independent because they involve disjoint slots. Multiplying gives the stated mass function, and the position of the final success being *pinned* rather than free is the single structural difference from the binomial coefficient $\binom{t}{k}$ that would otherwise appear.

    The indirect argument never counts anything. The events $\{Y_k\leq t\}$ and $\{N_t\geq k\}$ are the same event — the $k$-th arrival has happened by slot $t$ exactly when at least $k$ arrivals have occurred by slot $t$ — so the arrival-time and count descriptions are two readings of one sequence, and the Pascal CDF is the binomial survival function. This duality is worth keeping: it is the discrete rehearsal of the identity between Erlang arrival times and Poisson counts on the [next page](03-poisson-processes.md), and it is why one process can be simulated either by flipping every slot or by drawing gaps and skipping ahead.

    The load-bearing hypothesis is independence across slots, and it is spent twice: once to multiply the two factors in the direct argument, and once to make the gaps $T_k$ independent of each other rather than merely identically distributed. Drop it and the count is still a sum of Bernoullis with mean $tp$, but its variance is no longer $tp(1-p)$ and none of the waiting-time laws survive.

!!! note "The statement that an arrival happens eventually is not a limit, and it needs an axiom that finite probability does not supply"
    With $p>0$, $\mathbf{P}(T_1>k)=(1-p)^{k}\to0$, so $\mathbf{P}(T_1<\infty)=1$: an arrival occurs, with probability one, in a finite number of slots. [Probability Axioms](../part-02-probability-foundations/02-probability-axioms.md) makes the point that this sentence is not sayable without countable additivity, since "eventually" is a countable union and finite additivity says nothing about it, and lists "everything about Bernoulli Processes" among the results resting on that axiom. The practical residue is that the complement — an infinite sequence of failures — is a perfectly legitimate outcome with probability zero, which is the [Probability Spaces](../part-02-probability-foundations/01-probability-spaces.md) distinction between impossible and null arriving in its simplest instance.

## The Fresh Start Property Says the Process Has No Idea What Time It Is

The geometric law is memoryless: for a gap $T$ and any $m,k\geq0$,

$$\mathbf{P}(T>m+k\mid T>m)=\frac{(1-p)^{m+k}}{(1-p)^{m}}=(1-p)^{k}=\mathbf{P}(T>k).$$

Having waited $m$ slots without an arrival tells you nothing about how much longer you will wait. [Geometric Distribution](../part-05-common-distributions/03-geometric-distribution.md) states this as a property of the distribution; the process version is stronger and is called the **fresh-start property**. Fix any time $t$ and look at the sequence $X_{t+1},X_{t+2},\ldots$: it is a Bernoulli process with the same $p$, independent of everything up to $t$. The process restarted at $t$ is a statistically identical copy of the original, and this remains true when $t$ is not a fixed time but the time of the $k$-th arrival, or any other time determined by the past alone.

That last qualification is the whole content, and the phrase for the times it permits is **stopping time** — a random time $\tau$ for which the event $\{\tau=t\}$ is decidable from $X_1,\ldots,X_t$. "The slot of the third arrival" qualifies. "The slot before the third arrival" does not, and the process restarted there is emphatically not a fresh Bernoulli process, since by construction the very next slot is an arrival. The distinction is the same one [Probability Spaces](../part-02-probability-foundations/01-probability-spaces.md) draws about point-in-time discipline, it is where lookahead bias lives, and it is developed properly in [Martingales](10-martingales.md).

The fresh-start property is what makes "due for a bounce" a false sentence about an i.i.d. process, and it is worth being precise about which false sentence it is. After five straight losses the probability of a win next is $p$ — unchanged, uninformed, unrepentant. What is *also* true, and is the reason the fallacy is durable, is that long runs of losses are rare, so having observed one you are entitled to doubt that the process is Bernoulli at all. Those are different inferences: one conditions within an assumed model, the other questions the model. The next two sections measure how bad both intuitions are.

## Runs Are Much Longer Than Anyone's Prior

Independence does not mean the path looks structureless. The longest run of consecutive successes in $n$ fair flips is concentrated near $\log_2 n$, which grows slowly enough that people consistently underestimate it and fast enough that any track record of realistic length contains a streak worth a paragraph in a pitch deck.

```python
import numpy as np

rng = np.random.default_rng(8021)
reps = 200_000
print(f"  longest run of consecutive wins, fair coin, {reps} histories")
print("        n    mean longest run    log2(n)    P(>=5)    P(>=8)    P(>=10)    P(>=13)")
best = np.zeros(reps, dtype=np.int32)
cur = np.zeros(reps, dtype=np.int32)
for t in range(1, 6_301):
    cur = (cur + 1) * (rng.random(reps) < 0.5)
    np.maximum(best, cur, out=best)
    if t in (60, 252, 1_260, 6_300):
        print(f"  {t:9d} {best.mean():19.4f} {np.log2(t):10.4f}"
              f" {(best >= 5).mean():9.4f} {(best >= 8).mean():9.4f}"
              f" {(best >= 10).mean():10.4f} {(best >= 13).mean():10.4f}")
# =>   longest run of consecutive wins, fair coin, 200000 histories
#            n    mean longest run    log2(n)    P(>=5)    P(>=8)    P(>=10)    P(>=13)
#             60              5.2666     5.9069    0.6219    0.1020     0.0254     0.0030
#            252              7.3095     7.9773    0.9859    0.3858     0.1111     0.0140
#           1260              9.6307    10.2992    1.0000    0.9171     0.4581     0.0726
#           6300             11.9518    12.6211    1.0000    1.0000     0.9544     0.3182
```

The second and third columns are the result. Mean longest run tracks $\log_2 n$ with a stable offset of about $0.65$ at every scale — $5.2666$ against $5.9069$, $7.3095$ against $7.9773$, $9.6307$ against $10.2992$, $11.9518$ against $12.6211$ — which is the known asymptotic $\log_2 n-\log_2\ln 2-\tfrac12$ doing its job. There is nothing to fit and nothing to estimate; the streak length is a deterministic function of how long you watched.

Now read the row a manager actually occupies. One year of daily signals is $252$ slots, and a five-day winning streak appears in $0.9859$ of zero-edge histories — it is not evidence of anything, it is the near-certain default. Five years is $1{,}260$ slots, and an *eight*-trade winning run appears in $0.9171$ of them while a **ten-trade winning run appears in $0.4581$ — almost half of strategies with no edge whatsoever**. Twenty-five years reaches $0.3182$ for a run of thirteen.

The asymmetry that makes this expensive is that the streak is discovered after the fact. Nobody specifies "I will test whether trades $811$ through $820$ are all winners"; they scan the record, find the best run, and report it. That converts a probability about one fixed window into a probability about the maximum over $n$ overlapping windows, which is what the table computes and what the intuition does not. It is the same multiplicity that [Part XV](../part-15-multiple-testing/index.md) corrects for explicitly and that a streak anecdote never does.

## The Obvious Way to Test for a Hot Hand Is Biased Against Finding One

Suppose you doubt the model rather than trusting it, and set out to measure whether wins follow wins. The obvious estimator is the obvious one: scan the record, find every slot preceded by $k$ consecutive wins, and compute the fraction of those that are wins. On a genuinely i.i.d. process the answer should be $p$. It is not.

```python
import numpy as np

rng = np.random.default_rng(8023)
reps = 200_000
print(f"  P(win | the previous k were wins), estimated the obvious way, fair coin, {reps} runs")
print("        n    k = 1    k = 2    k = 3    truth")
for n in (4, 10, 25, 100, 1_000):
    x = rng.random((reps, n)) < 0.5
    row = []
    for k in (1, 2, 3):
        prev = np.ones((reps, n - k), dtype=bool)
        for j in range(k):
            prev &= x[:, j:n - k + j]                           # the previous k were all wins
        cnt = prev.sum(axis=1)
        hit = (prev & x[:, k:]).sum(axis=1)
        ok = cnt > 0                                            # runs where the estimate exists
        row.append((hit[ok] / cnt[ok]).mean())
    print(f"  {n:9d} {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {0.5:8.4f}")
# =>   P(win | the previous k were wins), estimated the obvious way, fair coin, 200000 runs
#            n    k = 1    k = 2    k = 3    truth
#              4   0.4036   0.4165   0.4987   0.5000
#             10   0.4459   0.3755   0.3521   0.5000
#             25   0.4789   0.4307   0.3722   0.5000
#            100   0.4947   0.4838   0.4599   0.5000
#           1000   0.4995   0.4985   0.4966   0.5000
```

Every number in the table was generated by a fair coin with no memory at all, and every entry but one sits below $0.5$. At $n=10$ and $k=3$ the estimator returns $0.3521$ — a fifteen-point apparent *cold* hand manufactured entirely by the estimator, on data where the truth is exactly one half. **The obvious conditional-win-rate estimator is biased downward, so a process with no dependence looks anti-persistent, and a process with genuine persistence looks like it has none.**

The mechanism is that the estimator averages a ratio computed within each finite record, and the windows overlap. In a record of length $n$ there are only so many slots following a win, and conditioning on a win at $t$ removes that slot from the pool of candidates for the numerator elsewhere in the same sequence — so within any single finite sequence the sampled continuation rate is short of $p$, and averaging those ratios across sequences does not undo it. Averaging *counts* across sequences rather than ratios would be unbiased; averaging per-record proportions is not, and per-record proportions are what anyone computes.

The exception is diagnostic. At $n=4$ and $k=3$ the estimate is $0.4987$, essentially exact, because a four-slot record admits exactly one window in which three prior wins can be checked — and with a single window there is no ratio-averaging to be biased. The bias appears the moment there are two or more overlapping windows, peaks when the number of qualifying windows is small and variable, and decays as $n$ grows: at $n=1{,}000$ the three columns read $0.4995$, $0.4985$, $0.4966$.

!!! warning "Both intuitions about streaks are wrong, and they are wrong in opposite directions, so the errors do not cancel"
    The naive reading of a track record over-interprets runs, because $\log_2 n$ of them are free. The naive *test* of a track record under-detects persistence, because the per-record conditional rate is biased low. A desk that does both — is impressed by a ten-trade run, then reassured by a conditional win rate that comes back below $50\%$ — has made two errors that point in opposite directions and has no idea what its net position is. The repair for the first is to compute the null distribution of the maximum rather than of a fixed window; the repair for the second is to pool counts across records instead of averaging proportions, or to compare the statistic against its own simulated null, which costs the twenty lines above.

## Splitting a Stream Does Not Give Independent Streams

Two operations turn Bernoulli processes into Bernoulli processes. **Merging** superimposes independent streams with rates $p_1$ and $p_2$ and declares an arrival whenever either fires, giving a Bernoulli process with $p=p_1+p_2-p_1p_2$ — the inclusion–exclusion correction being the probability that both fire in the same slot. **Splitting** routes each arrival left or right by an independent coin with probability $q$, giving streams with rates $pq$ and $p(1-q)$.

Each output stream is individually Bernoulli, and that much is immediate. The claim usually attached to it — that the two output streams are *independent* — is false in discrete time, and the reason is worth seeing, because it is exactly the discrepancy that vanishes in the limit producing the [Poisson process](03-poisson-processes.md).

??? note "Proof that the two halves of a split stream are negatively correlated, with the exact coefficient"
    Write $S_i$ for the arrival indicator in slot $i$ and $C_i$ for the routing coin, independent of everything, with $\mathbf{P}(C_i=1)=q$. The two output indicators are $I_i=S_iC_i$ and $J_i=S_i(1-C_i)$. Their means are $\mathbb{E}[I_i]=pq$ and $\mathbb{E}[J_i]=p(1-q)$, confirming each stream's rate. But $I_iJ_i=S_i^{2}C_i(1-C_i)=0$ identically — a slot cannot send its single arrival both ways — so

    $$\mathrm{cov}(I_i,J_i)=\mathbb{E}[I_iJ_i]-\mathbb{E}[I_i]\mathbb{E}[J_i]=0-p^{2}q(1-q)=-p^{2}q(1-q).$$

    Each indicator is Bernoulli, so $\mathrm{var}(I_i)=pq(1-pq)$ and $\mathrm{var}(J_i)=p(1-q)(1-p(1-q))$. Dividing and setting $q=\tfrac12$ collapses the algebra to

    $$\mathrm{corr}(I_i,J_i)=\frac{-p^{2}/4}{(p/2)(1-p/2)}=-\frac{p}{2-p}.$$

    The counts $N_1$ and $N_2$ are sums over slots that are independent across $i$, so covariance and both variances scale by the same factor and the correlation of the counts equals the per-slot correlation exactly, with no dependence on how many slots were observed.

    The load-bearing hypothesis is that a slot holds at most one arrival, and it is spent in the single step $I_iJ_i=0$. That is a modelling artifact of discretizing time, not a fact about arrivals, and the whole discrepancy is proportional to $p$ because it is the probability of a slot being occupied at all. **The negative correlation is the price of the grid, and the Poisson process is what you get when you stop paying it.**

```python
import numpy as np

rng = np.random.default_rng(8027)
reps, slots = 2_000_000, 250
print(f"  one Bernoulli(p) stream split by a fair coin, {reps} runs of {slots} slots")
print("         p    rate 1    rate 2    corr(N1, N2)    -p/(2-p)")
for p in (0.50, 0.20, 0.05, 0.02):
    n1 = np.zeros(reps, dtype=np.int32)
    n2 = np.zeros(reps, dtype=np.int32)
    for _ in range(slots):
        s, c = rng.random(reps) < p, rng.random(reps) < 0.5
        n1 += s & c                                            # the coin sends the arrival left
        n2 += s & ~c                                           # or right, never both
    r = np.corrcoef(n1, n2)[0, 1]
    print(f"  {p:10.2f} {n1.mean() / slots:9.5f} {n2.mean() / slots:9.5f}"
          f" {r:15.4f} {-p / (2 - p):11.4f}")
# =>   one Bernoulli(p) stream split by a fair coin, 2000000 runs of 250 slots
#             p    rate 1    rate 2    corr(N1, N2)    -p/(2-p)
#            0.50   0.24997   0.25000         -0.3328     -0.3333
#            0.20   0.10000   0.10002         -0.1111     -0.1111
#            0.05   0.02501   0.02501         -0.0254     -0.0256
#            0.02   0.00999   0.01000         -0.0100     -0.0101
```

The rate columns behave: $pq$ and $p(1-q)$ are reproduced to four decimals at every $p$, so each stream really is Bernoulli with the advertised parameter. The correlation column is the point. It is negative at every $p$, and it matches the exact per-slot value $-p/(2-p)$ across two orders of magnitude — $-0.3328$ against $-0.3333$, $-0.1111$ against $-0.1111$, $-0.0254$ against $-0.0256$, $-0.0100$ against $-0.0101$.

The negativity has a one-line cause: a slot holds at most one arrival, so an arrival routed left is an arrival not routed right, and the two counts compete for a fixed supply of slots. At $p=0.5$ that competition is fierce and the correlation is $-1/3$. **The dependence is $O(p)$, so it does not vanish because the streams are "obviously unrelated" — it vanishes because the slots become empty.** Send $p\to0$ while holding the arrival rate per unit of clock time fixed by shrinking the slots, and the correlation goes to zero linearly; that limit is the Poisson process, and there the splitting theorem is exact rather than approximate.

## A Streak Is Evidence Only Against a Null Somebody Actually Computed

The Bernoulli process has no memory, no state, and no parameters beyond $p$, and this page has spent three simulations showing that a reader's intuition about it is wrong in three separate ways. Runs are longer than expected, by an amount that grows with the length of the record rather than with any property of the process. The standard estimator of persistence is biased against the thing it is looking for, by an amount that shrinks with the length of the record. And an operation that is exactly independence-preserving in continuous time is only approximately so in discrete time, by an amount proportional to how busy the slots are.

What the three failures have in common is that in each case the null distribution was assumed rather than computed, and in each case computing it was cheap. That is the practical rule this page delivers, and it does not require any of the machinery: before treating a pattern in a track record as evidence, generate the same statistic ten thousand times from a coin with the same $p$ and the same $n$, and look at where the observed value falls. A ten-trade winning run lands at the median of that distribution. A conditional win rate of $0.47$ lands above it.

The reason to install this habit here, on the simplest process in the part, is that every model that follows adds structure and therefore adds ways for the null to be non-obvious. A [Markov chain](05-markov-chains.md)'s sojourn times are geometric for the same reason these gaps are, and its runs are longer still. A [random walk](11-random-walks.md)'s time above zero has a distribution that puts its *least* likely outcome at exactly the fraction anyone would guess. In each case the arithmetic is easy and the intuition is not, which is why the appendix keeps printing the null next to the number.
