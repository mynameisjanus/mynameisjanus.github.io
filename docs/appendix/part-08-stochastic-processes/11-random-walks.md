# Random Walks

A random walk is the simplest process in this part and the one whose behaviour is furthest from anybody's intuition. It returns to its starting point with probability one and takes infinitely long about it on average. It spends most of its time on one side of zero, and the *least* likely thing it does is divide its time evenly. Every one of those facts is a statement about a fair game with no edge and no memory, which makes them the null distribution against which claims of edge have to be read — and they are read against zero instead.

This page covers the walk and the central binomial coefficient that governs its return probabilities, recurrence in one and two dimensions against transience in three, the gambler's ruin problem solved for fair and unfair games, the arcsine law for time spent above zero, and the variance ratio as the diagnostic the walk supplies for its own falsification. It does not take the continuous-time limit, which is [Brownian Motion](08-brownian-motion.md); it does not develop the fair-game structure formally, which is [Martingales](10-martingales.md); it does not prove the limit theorems it invokes, which are [Part VII](../part-07-asymptotic-theory/index.md); it does not test for a unit root or fit an ARMA model, which are [Time Series](../../part-03-statistics/03-time-series.md) and [Part XIII](../part-13-regression/index.md); and it constructs no trading rule.

The trading stake is a sentence the course writes about a real price series. [Time Series](../../part-03-statistics/03-time-series.md) observes that "the log price's 'mean' marches from 4.37 to 6.01 — it is not an estimate of anything, just a diary of where the random walk happened to wander", and names this page as the null model for that wandering. The fourth section makes the diary far stranger than it looks: a strategy whose equity curve has been above water for $90\%$ of its life is not merely consistent with having no edge, it is a *more likely* outcome under no edge than a curve that spent half its time up and half down.

## A Sum of I.I.D. Steps, and the Coefficient That Governs It

A **random walk** is the partial-sum process $S_n=X_1+\cdots+X_n$ with $S_0=0$ and the steps $X_i$ independent and identically distributed. The **simple symmetric** walk takes $X_i=\pm1$ with probability $\tfrac12$ each, and almost everything qualitative about the general case is already visible there.

The quantity that organizes the whole subject is $u_{2n}=\mathbf{P}(S_{2n}=0)$, the probability of being back at the origin after $2n$ steps — odd times are impossible by parity. It is a binomial mass function evaluated at its centre, and its asymptotic behaviour decides recurrence, first-passage, and the arcsine law together.

??? note "Proof that the return probability decays like one over the square root of n, and what that implies immediately"
    A path returns to zero at time $2n$ exactly when it takes $n$ up-steps and $n$ down-steps in some order, so

    $$u_{2n}=\binom{2n}{n}2^{-2n}.$$

    [Counting Principles](../part-01-mathematical-foundations/02-counting-principles.md) records that Stirling's approximation is used precisely to show $\binom{2n}{n}\sim4^{n}/\sqrt{\pi n}$, and names "the growth rate governing random-walk return probabilities" as its role. Substituting,

    $$u_{2n}\sim\frac{4^{n}}{\sqrt{\pi n}}\cdot4^{-n}=\frac{1}{\sqrt{\pi n}}.$$

    Two consequences follow with no further work. The series $\sum_n u_{2n}$ diverges, since $\sum1/\sqrt{n}$ does — and the expected number of visits to the origin is exactly $\sum_n u_{2n}$, so the walk visits its start infinitely often and is **recurrent**. But $u_{2n}\to0$, so the walk is nonetheless at the origin a vanishing fraction of the time; it returns infinitely often and is almost never there.

    The load-bearing fact is the exponent $\tfrac12$, and it is the same exponent that makes $\lvert S_n\rvert$ grow like $\sqrt n$. Every result on this page is that one number wearing a different hat: recurrence needs $\sum n^{-1/2}=\infty$, the infinite mean return time needs the first-passage tail $n^{-3/2}$ that differencing produces, and the arcsine law's density $1/(\pi\sqrt{x(1-x)})$ carries two square roots for the two ends of the interval.

## The Walk Returns With Probability One and Takes Forever About It

Recurrence is worth stating carefully because it is routinely over-read. The walk returns to zero with probability one, and by the fresh-start property it therefore returns infinitely often. Yet the expected time to the first return is infinite: the first-passage probabilities decay like $n^{-3/2}$, which is summable — so a return is certain — while $n\cdot n^{-3/2}=n^{-1/2}$ is not, so its mean diverges. This is the same shape [Brownian Motion](08-brownian-motion.md) finds for the first-passage time to a barrier, and the same trap: a certain event with an undefined waiting time.

Dimension changes the answer, which is **Pólya's theorem**: the simple symmetric walk is recurrent in one and two dimensions and transient in three or more. The mechanism is the same divergence test, since the return probability in $d$ dimensions decays like $n^{-d/2}$ and $\sum n^{-d/2}$ converges exactly when $d\geq3$. A three-dimensional walk visits its origin finitely many times and then wanders off forever. The financial reading is that a mean-reverting *spread* — a one-dimensional object — is guaranteed to revisit any level it has left, while a basket of three independent wandering price series has no such guarantee about its joint configuration, which is why cointegration is a statement about a specific linear combination rather than about the assets.

None of this says a walk is well behaved. Recurrence guarantees the return and says nothing about when; the typical excursion away from zero has length comparable to the time observed so far, which is why a log price wanders from $4.37$ to $6.01$ over twenty-five years without any of it being an estimate of anything.

## Gambler's Ruin Is a Boundary Value Problem

Add two absorbing barriers and the walk becomes the oldest problem in the subject. Start with $k$ units against an opponent with $N-k$, win each round with probability $p$, and play until one side has everything. Let $u_k$ be the probability of ruin.

??? note "Proof that ruin probabilities solve a two-point boundary value problem, and the answer for both fair and unfair games"
    Condition on the first round. [Law of Total Probability](../part-02-probability-foundations/06-law-of-total-probability.md) sets up exactly this recursion and defers the solution here:

    $$u_k=p\,u_{k+1}+(1-p)\,u_{k-1},\qquad u_0=1,\ u_N=0.$$

    This is a linear difference equation, so the problem is algebra rather than probability. Writing $q=1-p$ and $d_k=u_k-u_{k-1}$, the recursion rearranges to $p\,d_{k+1}=q\,d_k$, so $d_{k+1}=r\,d_k$ with $r=q/p$ and hence $d_k=r^{k-1}d_1$.

    For $p=\tfrac12$ we have $r=1$, all the increments are equal, $u_k$ is linear, and the boundary conditions give

    $$u_k=1-\frac{k}{N}.$$

    For $p\neq\tfrac12$ summing the geometric increments gives $u_k=1+d_1(r^{k}-1)/(r-1)$, and imposing $u_N=0$ fixes $d_1$:

    $$u_k=\frac{r^{k}-r^{N}}{1-r^{N}},\qquad r=\frac{1-p}{p}.$$

    The load-bearing feature is that $r$ is raised to the power of the *capital*, so a small change in $p$ enters exponentially. At $p=\tfrac12$ ruin is linear in how much you start with and the game is scale-free. Move $p$ a hair below $\tfrac12$ and $r>1$, the numerator is dominated by $r^{N}$ for any $k$ well short of $N$, and ruin becomes near-certain regardless of the starting stake. **A fair game is fragile in a way an unfair one is not: the fairness is what makes the answer linear, and nothing in the linear answer warns how fast it degrades.**

```python
import numpy as np

rng = np.random.default_rng(8113)
reps, N = 200_000, 100
print(f"  gambler's ruin on a {N}-unit table, {reps} runs, exact against simulated")
print("       start    p=0.50 exact    simulated    p=0.49 exact    simulated")
for k in (10, 25, 50, 90):
    row = [k]
    for p in (0.50, 0.49):
        if p == 0.5:
            exact = 1 - k / N
        else:
            r = (1 - p) / p
            exact = (r ** k - r ** N) / (1 - r ** N)
        pos = np.full(reps, k, dtype=np.int64)
        live = np.ones(reps, dtype=bool)
        while live.any():
            pos[live] += 2 * (rng.random(live.sum()) < p).astype(np.int64) - 1
            live &= (pos > 0) & (pos < N)
        row += [exact, (pos <= 0).mean()]
    print(f"  {row[0]:12d} {row[1]:15.4f} {row[2]:12.4f} {row[3]:15.4f} {row[4]:12.4f}")
# =>   gambler's ruin on a 100-unit table, 200000 runs, exact against simulated
#           start    p=0.50 exact    simulated    p=0.49 exact    simulated
#                10          0.9000       0.8989          0.9908       0.9908
#                25          0.7500       0.7502          0.9680       0.9680
#                50          0.5000       0.4992          0.8808       0.8810
#                90          0.1000       0.0998          0.3359       0.3356
```

The closed forms are exact and the simulation confirms them to three decimals in every cell. The fair columns are the linear answer: start with $10\%$ of the table and you are ruined $90\%$ of the time, start with $90\%$ and you are ruined $10\%$ of the time, and the relationship is a straight line.

The unfair columns are the point, and the disproportion is the lesson. Moving the win probability from $0.50$ to $0.49$ — a one-percent edge against, which is roughly what a bid-ask spread costs a strategy that trades often — changes ruin from $0.5000$ to $0.8808$ at half the table, and from $0.1000$ to $0.3359$ even when starting with ninety percent of all the money on the table. **A player holding nine times their opponent's capital, facing a game tilted by one percent, is ruined a third of the time.** The exponential in $r^{k}$ is doing that, and no amount of capital buys linear protection against it.

The trading reading is direct and is the reason costs matter more than they look. A strategy with a genuine but small edge, paying costs that consume most of it, is a walk with $p$ near but below $\tfrac12$ — and the survival probability is exponentially, not linearly, sensitive to where exactly $p$ sits. That is the same sensitivity [Mean Reversion and Pairs Trading](../../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md) is describing when it warns that every profit the pairs trade produces "is a fraction of a 23 bp leash."

## A Fair Game Spends Most of Its Time on One Side

Now the result that should change how an equity curve is read. Let $F_n$ be the fraction of time a symmetric walk spends above zero over $n$ steps. Intuition says $F_n$ concentrates near $\tfrac12$ — a fair game ought to be up about half the time. The truth is the exact opposite: $F_n$ converges to the **arcsine law**, with CDF

$$\mathbf{P}(F\leq x)=\frac{2}{\pi}\arcsin\sqrt{x},\qquad 0\leq x\leq1,$$

whose density $1/(\pi\sqrt{x(1-x)})$ is smallest at $x=\tfrac12$ and diverges at both ends. The balanced outcome is the least likely one.

```python
import numpy as np

rng = np.random.default_rng(8111)
reps, n = 400_000, 2_520
print(f"  fraction of {n} days a fair game spends above water, {reps} histories")
print("          x    P(fraction <= x)    arcsine law    (2/pi) arcsin(sqrt(x))")
prev = np.zeros(reps)
above = np.zeros(reps)
for _ in range(n):
    s = prev + (2.0 * (rng.random(reps) < 0.5) - 1.0)
    above += (prev + s) > 0                                    # the segment lies above the axis
    prev = s
f = above / n
for x in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
    a = 2 / np.pi * np.arcsin(np.sqrt(x))
    print(f"  {x:11.2f} {(f <= x).mean():18.4f} {a:14.4f} {a:26.4f}")
print(f"  P(between 0.45 and 0.55) {((f > 0.45) & (f < 0.55)).mean():.4f}"
      f"   P(outside 0.05 to 0.95) {((f < 0.05) | (f > 0.95)).mean():.4f}")
# =>   fraction of 2520 days a fair game spends above water, 400000 histories
#              x    P(fraction <= x)    arcsine law    (2/pi) arcsin(sqrt(x))
#             0.05             0.1441         0.1436                     0.1436
#             0.10             0.2047         0.2048                     0.2048
#             0.25             0.3331         0.3333                     0.3333
#             0.50             0.4996         0.5000                     0.5000
#             0.75             0.6665         0.6667                     0.6667
#             0.90             0.7951         0.7952                     0.7952
#             0.95             0.8566         0.8564                     0.8564
#      P(between 0.45 and 0.55) 0.0634   P(outside 0.05 to 0.95) 0.2864
```

The simulated CDF sits on the arcsine values to three or four decimals at every point, so the law is not an approximation being tested but a formula being confirmed. Read the third row: a quarter of fair games spend less than a quarter of their life above water, and by symmetry a quarter spend more than three-quarters of it above.

The last line is the whole page in two numbers. Over ten years of daily marks, a strategy with no edge whatever spends between $45\%$ and $55\%$ of its days above water only $6.34\%$ of the time — while spending either less than $5\%$ or more than $95\%$ of its days above water $28.64\%$ of the time. **The extreme outcome is four and a half times more likely than the balanced one.** A track record that has been in profit almost continuously and a track record that has been under water almost continuously are, jointly, the modal behaviour of a coin.

!!! warning "Time spent in profit carries almost no information about edge, and it is the statistic investors weight most heavily"
    Two facts combine badly. Under the null the fraction of time above water is arcsine-distributed, so nearly any value of it is unremarkable and the values that *look* most remarkable — always up, always down — are the likeliest. And the same $\sqrt n$ scaling means the excursions are long: a fair game that goes under water tends to stay there for a stretch comparable to the whole history observed so far, which is the [Brownian Motion](08-brownian-motion.md) result that new highs occur on a set of times of measure zero, arriving from the discrete side. The practical consequence is that "we have been profitable in eleven of the last twelve quarters" and "we have been under water for three years" are both consistent with zero edge and neither is close to significant. The statistics that carry information are the ones with a computable null — a t-statistic on the returns, corrected for the dependence [Random Processes](01-random-processes.md) measures and for the peeking [Martingales](10-martingales.md) prices — and the fraction of time in profit is not one of them.

## The Variance Ratio Is the Walk's Own Diagnostic

The walk supplies its own falsification test. Under a random walk the variance of a $q$-period change is exactly $q$ times the variance of a one-period change, since the increments are independent, so the **variance ratio**

$$VR(q)=\frac{\mathrm{var}(S_{t+q}-S_t)}{q\,\mathrm{var}(S_{t+1}-S_t)}$$

equals one for every $q$. Departures are informative in a signed way: $VR>1$ means increments reinforce each other and the series trends, $VR<1$ means they offset and it mean-reverts. The magnitude of the departure at horizon $q$ says which horizon carries the structure.

```python
import numpy as np

rng = np.random.default_rng(8117)
reps, n = 4_000, 6_300
e = rng.standard_normal((reps, n))
walk = np.cumsum(e, axis=1)
mom = np.cumsum(e + 0.15 * np.roll(e, 1, axis=1), axis=1)      # positively autocorrelated steps
sprd = np.zeros((reps, n))
for t in range(1, n):
    sprd[:, t] = 0.82 * sprd[:, t - 1] + e[:, t]               # a stationary AR(1) level


def vr(x, q):
    d1 = np.diff(x, axis=1)
    dq = x[:, q:] - x[:, :-q]
    return dq.var(axis=1) / (q * d1.var(axis=1))


print(f"  variance ratios on {reps} histories of {n} days")
print("        q    random walk    sd under the null    trending    mean-reverting spread")
for q in (2, 5, 10, 21):
    a, b, c = vr(walk, q), vr(mom, q), vr(sprd, q)
    print(f"  {q:9d} {a.mean():14.4f} {a.std(ddof=1):20.4f} {b.mean():11.4f}"
          f" {c.mean():24.4f}")
# =>   variance ratios on 4000 histories of 6300 days
#            q    random walk    sd under the null    trending    mean-reverting spread
#              2         0.9998               0.0129      1.1464                   0.9099
#              5         0.9989               0.0280      1.2333                   0.6988
#             10         0.9980               0.0432      1.2615                   0.4786
#             21         0.9972               0.0651      1.2757                   0.2602
```

The null column is flat at one — $0.9998$, $0.9989$, $0.9980$, $0.9972$ — confirming the statistic is correctly centred, and the second column is the piece that makes it usable: the standard deviation of $VR$ under the null on twenty-five years of daily data is $0.0129$ at $q=2$ and $0.0651$ at $q=21$. Without that column a measured $VR$ of $0.83$ is a number; with it, at $q=5$, it is six standard errors from one.

The third column is a series whose steps carry a lag-one autocorrelation of about $0.15$, and it reads $1.1464$ rising to $1.2757$ — the trend signature, with the ratio still climbing at three weeks because positive autocorrelation accumulates across horizons. The fourth column is a stationary AR(1) level with $\rho=0.82$, chosen because [Mean Reversion and Pairs Trading](../../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md) fits exactly that coefficient to the SPY–IVV spread. It reads $0.9099$, $0.6988$, $0.4786$, $0.2602$, falling monotonically, and the value at $q=21$ is close to the $0.21$ that lesson measures on the real spread at a five-day horizon.

!!! note "The variance ratio and the effective sample size are the same quantity, so a departure from one is a statement about how much data you have"
    Expanding the numerator of $VR(q)$ in autocovariances gives $VR(q)=1+2\sum_{k=1}^{q-1}(1-k/q)\rho_k$, which is exactly the triangular-weighted bracket [Random Processes](01-random-processes.md) derives for the variance of a sample mean and uses to define $n_{\text{eff}}=n/VR$. The two statistics are one statistic with two names, and reading them together is more informative than either alone. A trending series with $VR(21)=1.2757$ has $n_{\text{eff}}$ about $78\%$ of its nominal sample size at that horizon, so its error bars need widening by $13\%$. A mean-reverting spread with $VR(21)=0.2602$ has an effective sample nearly four times its nominal one — dependence that *helps*, which is the unusual case and is exactly what makes a stationary spread tradeable rather than merely stationary. The sign of $\log VR$ therefore says both which way the series moves and whether time is buying you information faster or slower than the calendar suggests.

The signature to notice is not the level but the *shape*. A trending series has $VR$ rising in $q$; a mean-reverting one has it falling; a random walk has it flat. Reporting a single $VR$ at a single horizon discards that, which is why the reconciliation problem [Time Series](../../part-03-statistics/03-time-series.md) works through — the same pair looking cointegrated at one lag structure and not at another, with the resolution that "tests see the horizon their lag structure selects" — is the same phenomenon read through a different statistic.

## The Null Is Stranger Than the Alternative

The random walk is the model of no edge, no memory, and no structure, and it produces behaviour that looks like all three. It returns to any level with certainty and takes an undefined length of time to do it. It is ruined by a one-percent tilt at a rate that no amount of capital buys down linearly. It spends its life lopsidedly on one side of zero, with the balanced outcome the least likely of all. And it wanders far enough that its running average is a diary rather than an estimate.

Every one of those is a null distribution that somebody's evidence is being compared against, usually implicitly and usually against zero instead. A long unbroken run of profitability is the arcsine law. A deep and lengthy drawdown is the arcsine law. A price series whose mean drifts across decades is recurrence with $\sqrt n$ excursions. A strategy that survived when a similar one blew up is the exponential in $r^{k}$. In each case the honest comparison is available in a few lines of simulation, and in each case the number that gets quoted has no null beside it.

That is the shape of the whole part, and it is worth stating once at the end. Each model here is a claim about memory — none in the arrival processes, memory through the present in the chains, none in the increments here — and each is a null rather than a description. The value of writing them down is not that markets obey them; it is that they are exactly computable, so the discrepancy between what they predict and what is measured is itself an estimate of something real. The variance ratio is the cleanest instance: it is one under the walk, and the amount by which it is not one, measured against the sampling error in the second column above, is the quantity with content.
