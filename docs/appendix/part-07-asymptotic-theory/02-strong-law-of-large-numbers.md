# The Strong Law of Large Numbers

The weak law says that at each sample size the average is probably close to the truth. The strong law says something a trader would actually want: that on almost every individual history the average eventually gets close and *stays* close, so there is a point after which the estimate never misbehaves again. The two statements sound like paraphrases and they are not, the gap between them is where every drawdown and every lucky backtest lives, and the strong law's guarantee arrives on a schedule no track record is long enough to reach.

This page covers almost-sure convergence and its definition through $\limsup$ and $\liminf$, the Borel–Cantelli argument that makes the strong law work, Kolmogorov's version and why it assumes strictly less than the weak law's proof did, the counterexample that separates the two modes, and the practical consequence that a running statistic gets many more chances to embarrass you than a fixed one. It does not re-derive the weak law, which is [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md); it does not describe the *size* of the error at finite $n$, which is [The Central Limit Theorem](03-central-limit-theorem.md); it does not develop the countable additivity axiom it leans on, which is [Probability Axioms](../part-02-probability-foundations/02-probability-axioms.md); and it treats no process with dependence, so ergodic averages of anything indexed by time are [Part VIII](../part-08-stochastic-processes/index.md).

The trading stake is a number the course produced by accident. [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) runs fifty coin-flip strategies on SPY and reports that "the best coin-flipper earned Sharpe 0.43 over a quarter century — and the expected-maximum formula says 0.45 was coming." Every one of those fifty has a true edge of exactly zero, and the strong law promises that every one of them converges to zero. The fourth section watches a single null strategy for twenty-five years and finds that it touches a Sharpe of $1.0$ along the way about thirty percent of the time — from one strategy, with no search at all.

## Almost Sure Means the limsup and the liminf Agree, With Probability One

A sequence of random variables $Y_n$ **converges almost surely** to $Y$, written $Y_n\xrightarrow{\ a.s.\ }Y$, when

$$\mathbf{P}\!\left(\left\{\omega:\lim_{n\to\infty}Y_n(\omega)=Y(\omega)\right\}\right)=1.$$

The object inside the probability is a set of *histories*, not a set of values, and the limit inside it is the ordinary deterministic limit of [Sequences and Infinite Series](../part-01-mathematical-foundations/04-sequences-and-series.md). That page supplies the vocabulary for saying when it exists: a sequence converges precisely when its $\limsup$ and $\liminf$ coincide, so the event above is exactly the event that $\limsup_n Y_n=\liminf_n Y_n=Y$, and almost-sure convergence is the requirement that this happen with probability one.

The strong law asserts it for sample means. If $X_1,X_2,\dots$ are independent and identically distributed with $\mathbb{E}[\lvert X_1\rvert]<\infty$ and mean $\mu$, then $\bar X_n\xrightarrow{\ a.s.\ }\mu$.

??? note "Proof of the strong law under a finite fourth moment, and why the mechanism is Borel–Cantelli"
    The general theorem is Kolmogorov's and its proof is long. The mechanism is visible in a version that assumes more than necessary and costs half a page. Take $\mathbb{E}[X_1]=0$ without loss of generality and assume $\mathbb{E}[X_1^{4}]=K<\infty$. Expanding $\big(\sum_{i=1}^{n}X_i\big)^{4}$ and taking expectations kills every term with a lone factor, by independence and mean zero, leaving $n$ terms of the form $\mathbb{E}[X_i^{4}]$ and $3n(n-1)$ of the form $\mathbb{E}[X_i^{2}X_j^{2}]$, so

    $$\mathbb{E}\big[\bar X_n^{4}\big]=\frac{nK+3n(n-1)\sigma^{4}}{n^{4}}\leq\frac{C}{n^{2}}$$

    for a constant $C$. Now apply Markov's inequality from [Variance](../part-04-expectation-and-moments/02-variance.md) to the fourth power: $\mathbf{P}(\lvert\bar X_n\rvert\geq\epsilon)\leq\mathbb{E}[\bar X_n^{4}]/\epsilon^{4}\leq C/(n^{2}\epsilon^{4})$.

    Here is the step the weak law never takes. Those bounds are **summable** — $\sum_n 1/n^{2}<\infty$ — and the first Borel–Cantelli lemma says that when the probabilities of a sequence of events sum to a finite number, the probability that infinitely many of them occur is zero. So for each fixed $\epsilon$, only finitely many $n$ have $\lvert\bar X_n\rvert\geq\epsilon$, on a set of histories of probability one. Intersecting over $\epsilon=1,\tfrac12,\tfrac13,\dots$ — a *countable* intersection of probability-one sets, which is again probability one — gives $\bar X_n\to0$ on almost every history.

    The load-bearing hypothesis is not the fourth moment, which is an artefact of this shortcut; Kolmogorov's theorem needs only $\mathbb{E}\lvert X_1\rvert<\infty$, which is **weaker** than the finite variance the weak law's own proof used, for a **stronger** conclusion. The genuinely indispensable ingredient is countable additivity. Both the summability step and the countable intersection over $\epsilon$ are manipulations of infinitely many events at once, and [Probability Axioms](../part-02-probability-foundations/02-probability-axioms.md) is right that every almost-sure statement in this appendix rests on that axiom. Drop it and the phrase "with probability one" stops referring to anything.

The relationship between the hypotheses is worth stating plainly, because it inverts the usual expectation. The weak law as proved on the previous page assumes a finite variance and concludes convergence in probability. The strong law assumes only a finite mean and concludes almost-sure convergence, which implies convergence in probability. The strong law is therefore not a refinement bought with extra assumptions — it is strictly better on both sides. The weak law survives in this appendix because its proof is three lines and it is the version whose finite-$n$ bound can be computed.

## In Probability Does Not Imply Almost Surely, and Here Is the Counterexample

If the strong law implies the weak law, the two modes might still coincide. They do not, and the separating example is small enough to hold in the head.

??? note "Proof that a sequence can converge in probability while converging on no history at all"
    Let $Y_1,Y_2,\dots$ be independent with $\mathbf{P}(Y_n=1)=1/n$ and $\mathbf{P}(Y_n=0)=1-1/n$.

    Convergence in probability to $0$ is immediate: for any $\epsilon\in(0,1)$, $\mathbf{P}(\lvert Y_n\rvert\geq\epsilon)=1/n\to0$. The definition on [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md) is satisfied exactly.

    Almost-sure convergence fails completely. The events $\{Y_n=1\}$ are independent and $\sum_n 1/n$ diverges, so the *second* Borel–Cantelli lemma applies and gives $\mathbf{P}(Y_n=1\ \text{infinitely often})=1$. On every history, $Y_n$ returns to $1$ forever, so $\limsup_n Y_n=1$ while $\liminf_n Y_n=0$, and by the criterion of [Sequences and Infinite Series](../part-01-mathematical-foundations/04-sequences-and-series.md) the sequence converges nowhere. Not on a small set of histories — on none of them.

    The whole difference sits in one word: **summable**. When $\sum_n\mathbf{P}(\lvert Y_n\rvert\geq\epsilon)$ converges, the first Borel–Cantelli lemma upgrades convergence in probability to almost-sure convergence and the two modes coincide. When it merely tends to zero without summing, as $1/n$ does, the individual probabilities are small and there are enough of them that something always happens later. That is the entire content of the word "strong", and it explains why the strong law's proof had to bound $\mathbf{P}$ by $C/n^{2}$ rather than the weak law's $\sigma^{2}/(n\epsilon^{2})$ — the latter is not summable, so no amount of care with it would have produced a path-wise conclusion.

The distinction is not confined to constructed examples. Applied to sample means it is the difference between asking how good an estimate is *now* and asking whether it will ever go bad *again*, and both questions have answers.

```python
import numpy as np

rng = np.random.default_rng(7211)
N, eps, reps, chunk = 25_200, 0.05, 20_000, 1_000
mu, sd = 0.075 / 252, 0.195 / np.sqrt(252)
days = np.array([252, 1_260, 6_300, 25_200])
at, ever = np.zeros(4), np.zeros(4)
for _ in range(reps // chunk):
    m = np.cumsum(mu + sd * rng.standard_normal((chunk, N)), axis=1) / np.arange(1, N + 1)
    far = np.abs(m * 252 - 0.075) >= eps                       # annualized, off by 5 points
    for j, d in enumerate(days):
        at[j] += far[:, d - 1].sum()
        ever[j] += far[:, d - 1:].any(axis=1).sum()
print(f"  running annualized mean of iid returns, mu = 0.075, sigma = 0.195, {reps} paths of 100 years")
print(f"  P(the estimate is more than {eps} away from mu) -- read at one n, then from n onward")
print("        day     years    at exactly that day     ever again after it     ratio")
for j, d in enumerate(days):
    a, e = at[j] / reps, ever[j] / reps
    print(f"  {d:9d} {d / 252:9.1f} {a:22.4f} {e:23.4f} {e / a:9.2f}")
# =>   running annualized mean of iid returns, mu = 0.075, sigma = 0.195, 20000 paths of 100 years
#      P(the estimate is more than 0.05 away from mu) -- read at one n, then from n onward
#            day     years    at exactly that day     ever again after it     ratio
#            252       1.0                 0.8006                  1.0000      1.25
#           1260       5.0                 0.5655                  0.9677      1.71
#           6300      25.0                 0.1991                  0.3926      1.97
#          25200     100.0                 0.0112                  0.0112      1.00
```

The two middle columns are the two theorems. The third column is what the weak law controls — the probability that the running estimate of the equity premium is off by five percentage points *at one nominated moment* — and it behaves impeccably, falling from $0.8006$ after a year to $0.1991$ after twenty-five and $0.0112$ after a hundred. The fourth column is what the strong law controls: the probability that the estimate is that far off at *any* point from there onward. At the twenty-five-year mark it is $0.3926$, almost exactly twice the $0.1991$ beside it.

Both columns go to zero, which is the strong law holding — these are iid draws with a finite mean, so the theorem applies and the fourth column is obliged to vanish. But it vanishes later, and the ratio between the columns is the price of asking the path-wise question. Reading the row that corresponds to the sample this course actually has: after a quarter century of data there is a one-in-five chance the estimate is badly wrong today, and a two-in-five chance it will be badly wrong at some point in the following seventy-five years. The last row's ratio of $1.00$ is an artefact of the simulation ending there, not a result.

## What the Strong Law Buys Is the Right to Stop Looking

The practical content of the distinction is about *when the estimate is inspected*. The weak law licenses one look at one pre-specified sample size. The strong law licenses a standing claim — after some finite point, which the theorem does not name, the estimate is inside any tolerance you care to set and never leaves.

Almost everything built on repeated sampling relies quietly on the strong version. A Monte Carlo integration in [Monte Carlo Simulation](../part-09-monte-carlo-methods/03-monte-carlo-simulation.md) is run until the answer stops moving, which is a path-wise stopping rule and not a fixed-$n$ statement; the practice is legitimate only because the path itself converges. The bootstrap of [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) resamples until the interval stabilizes, on the same reasoning. In both cases the number of draws is chosen by looking at the output, which the weak law says nothing about and the strong law does.

The permission is real and the way it is usually exercised is not. "Run until the answer stops moving" is an eyeball test on one path, and the strong law guarantees the path settles without saying anything about how well a *finite* stretch of it reveals that.

```python
import numpy as np

rng = np.random.default_rng(7237)
s, reps = 1.5, 2_000
truth = np.exp(s * s / 2)
marks = (1_000, 10_000, 100_000, 1_000_000)
run, seen, snap = np.zeros(reps), 0, {}
for cut in sorted({int(0.9 * m) for m in marks} | set(marks)):
    run += np.exp(s * rng.standard_normal((reps, cut - seen))).sum(axis=1)
    seen = cut
    snap[cut] = run / cut
print(f"  estimating a lognormal mean of {truth:.4f} by Monte Carlo, {reps} independent runs")
print("            n   'the last 10% moved < 1%'   actual relative RMS error   error at that rate")
for m in marks:
    look = np.mean(np.abs(snap[m] - snap[int(0.9 * m)]) / np.abs(snap[m]) < 0.01)
    rms = np.sqrt(np.mean((snap[m] / truth - 1) ** 2))
    print(f"  {m:13d} {look:26.4f} {rms:27.4f} {np.sqrt(np.e ** (s * s) - 1) / np.sqrt(m):20.4f}")
# =>   estimating a lognormal mean of 3.0802 by Monte Carlo, 2000 independent runs
#                n   'the last 10% moved < 1%'   actual relative RMS error   error at that rate
#               1000                     0.2830                      0.0930               0.0921
#              10000                     0.7065                      0.0287               0.0291
#             100000                     0.9995                      0.0090               0.0092
#            1000000                     1.0000                      0.0029               0.0029
```

The last column confirms the theorem is behaving: the actual error $0.0930$, $0.0287$, $0.0090$, $0.0029$ tracks the $\sigma/\sqrt n$ prediction $0.0921$, $0.0291$, $0.0092$, $0.0029$ on every row. Nothing is going wrong with the estimator.

The failure is in the diagnostic. At ten thousand draws, $70.65\%$ of runs pass the test "the estimate has moved less than $1\%$ over the last tenth of the sample" — while the actual relative error is $2.87\%$, nearly three times the tolerance the test appears to be enforcing. At a hundred thousand draws essentially every run passes, $0.9995$, and the error is still $0.90\%$. The eyeball test is not measuring the error; it is measuring the *increment*, and increments shrink faster than errors do because a running mean at $n$ moves by only the last tenth's worth of new information while carrying nine tenths of whatever it had already accumulated.

!!! note "The only honest convergence diagnostic for a simulation is the spread across independent runs, and it costs one extra loop"
    Every number in the table's third column required $2{,}000$ independent replications to compute, and that is precisely the point — the quantity a single run cannot see is the dispersion of the ensemble it was drawn from. The cheap version of the fix is to run the simulation a handful of times with different seeds and report the spread of the answers, which estimates the third column directly rather than the second. It is the same argument [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md) makes about the equity premium arriving from the other side: there the ensemble is inaccessible and the calendar fixes $n$, so the standard error has to be computed from a formula; here $n$ is a budget and the ensemble can simply be generated, so there is no excuse for not measuring it.

The same reasoning is what a track record cannot supply, and the asymmetry is worth naming. In a simulation the sample size is a budget: if the answer has not settled, draw more. In a market it is a calendar, and the "finite point" after which the strong law's guarantee holds is unbounded — the theorem provides no rate, by construction, since a rate would be a finite-$n$ statement and the whole content of the theorem is the limit. So the strong law is the one that says what a practitioner wants and the one whose guarantee is least checkable, while the weak law says less and at least attaches a number to it.

## A Zero-Edge Strategy Crosses Every Threshold You Will Ever Set

The gap between the two columns above has a name in trading: it is why a running performance statistic, watched continuously, produces evidence that a fixed-horizon statistic never would.

```python
import numpy as np

rng = np.random.default_rng(7229)
N, start, reps, chunk = 6_300, 252, 40_000, 4_000
sd = 0.195 / np.sqrt(252)
bars = (0.5, 1.0, 1.5, 2.0)
ever = np.zeros(4)
end = np.zeros(4)
t = np.arange(1, N + 1)
for _ in range(reps // chunk):
    r = sd * rng.standard_normal((chunk, N))                   # a genuinely zero-edge strategy
    s1, s2 = np.cumsum(r, axis=1), np.cumsum(r * r, axis=1)
    m = s1 / t
    sharpe = np.sqrt(252) * m / np.sqrt(np.maximum(s2 / t - m * m, 1e-300))
    run = sharpe[:, start - 1:]
    peak = run.max(axis=1)
    for j, b in enumerate(bars):
        ever[j] += (peak >= b).sum()
        end[j] += (run[:, -1] >= b).sum()
print(f"  {reps} strategies with a true Sharpe of exactly zero, watched daily from year 1 to year 25")
print("     Sharpe bar    reached at some point    still there at year 25    ratio")
for j, b in enumerate(bars):
    e, f = ever[j] / reps, end[j] / reps
    print(f"  {b:13.1f} {e:24.4f} {f:25.4f} {e / max(f, 1 / reps):8.1f}")
# =>   40000 strategies with a true Sharpe of exactly zero, watched daily from year 1 to year 25
#         Sharpe bar    reached at some point    still there at year 25    ratio
#                0.5                   0.6077                    0.0063     96.5
#                1.0                   0.3064                    0.0000  12256.0
#                1.5                   0.1280                    0.0000   5122.0
#                2.0                   0.0431                    0.0000   1725.0
```

Read the first row against the second column. Six percent of a percent — $0.0063$ — of these strategies finish twenty-five years with a Sharpe above $0.5$, which is what a fixed-horizon test would find and is about what it should be for something with no edge. But $0.6077$ of them, three in five, *touch* $0.5$ at some point along the way. Nearly a third touch $1.0$. One in twenty-three touches $2.0$, a number that would end an allocator's diligence, and not one of them still has it at the end.

None of these strategies has any edge whatsoever. There is no search, no overfitting, no data mining and no selection: a single pre-specified rule with a true Sharpe of exactly zero, watched daily. The published finding that "the best coin-flipper earned Sharpe 0.43" over fifty candidates is the same effect with selection added on top, and this table shows that most of the work was already done before the selection began. The strong law says every one of these paths converges to zero. It also permits every one of them to pass through $2.0$ first.

!!! warning "A statistic that is watched continuously and acted on when it crosses a threshold is being evaluated by a rule the theorem does not cover"
    The maximum of a running statistic is not the statistic. Every column of the table above is a fact about $\max_t$, and every published significance threshold — a t-statistic of $2$, a Sharpe of $1$, a p-value of $0.05$ — is calibrated for a single evaluation at a pre-specified time. Launching a strategy when its running Sharpe first crosses a bar, or killing one when its drawdown first crosses another, converts a fixed-$n$ test into a path-wise one and inflates the error rate by the ratios in the last column. The remedy is not statistical sophistication; it is pre-commitment to the evaluation date, which [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) prices at $0.29$ of deflated-Sharpe probability for the identical strategy. Where continuous monitoring is genuinely necessary, the threshold has to be set against the distribution of the running maximum rather than the distribution of the endpoint, and those are the two columns above.

## Converging Eventually Is Not a Property Any Backtest Can Exhibit

The strong law is the theorem that most closely matches what people mean when they say things average out, and that is precisely why it should be handled carefully. It is a statement about one history, which is the right object, and it is a statement about the infinite tail of that history, which is not an object any market participant has access to. It supplies no $n$, and it cannot be made to supply one, because a finite $n$ would be a different theorem.

What it does supply is a clean way to ask the right question of a long-run claim. The theorem's hypothesis is a finite mean and nothing else, so the first check is whether the quantity being averaged has one — [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md) shows what a sample mean does when it does not. The theorem's conclusion is about the whole tail of the path, so the second check is whether the number being quoted was read off at a pre-specified point or found by scanning the path for its best moment. Those two questions separate almost every misuse of "in the long run" from the legitimate uses, and neither requires any mathematics beyond knowing which of the two laws is being invoked.
