# Geometric Distribution

The geometric is the only law on the positive integers that has no memory, and that single property is what makes it the natural model for a waiting time and a ruinous one for anybody who believes a losing streak is due to end. Under it, a drawdown that has run for five hundred days is exactly as likely to finish tomorrow as one that began yesterday. Nothing accumulates, nothing is owed, and the past length of the wait is not evidence about the remaining length of it.

This page covers the waiting-time mass function, the two conventions the formulas silently choose between, the mean derived twice — once from a differentiated series and once from a single step of conditioning — memorylessness as a characterization of the family rather than a curiosity about it, the constant hazard that is the same statement in survival language, and the gap between the mean wait and the wait actually experienced. It does not cover the wait for the $r$-th success, which is [Negative Binomial Distribution](04-negative-binomial-distribution.md); it does not cover the continuous analogue, which is [Exponential Distribution](10-exponential-distribution.md); and it does not build the family that lets a hazard rise or fall with age, which is [Weibull Distribution](18-weibull-distribution.md) and is what real durations turn out to need.

The trading stake is that the index makes a new high on about $2\%$ of days, which a memoryless model turns into an average wait of $50$ days between highs. The longest observed underwater spell is $2{,}294$ days. [Performance Metrics and Reporting](../../part-05-backtesting-engine/04-performance-metrics-and-reporting.md) reports both numbers from the same series, and the last section shows they cannot both come from a geometric law — not marginally, but by eighteen orders of magnitude.

## The Waiting-Time Mass Function

Run independent trials, each succeeding with probability $p>0$, and let $X$ be the index of the first success. The event $\{X=k\}$ is a single sequence of outcomes — $k-1$ failures followed by one success — and independence multiplies their probabilities,

$$p_X(k)=(1-p)^{k-1}p,\qquad k=1,2,3,\ldots$$

There is no combinatorial factor here, unlike the binomial: only one arrangement produces a first success at trial $k$, because fixing *first* fixes the entire prefix. Summing the mass is a geometric series, which converges for every $p>0$ and gives $p\cdot\frac{1}{1-(1-p)}=1$, so the law is proper — the wait is finite with probability one however small $p$ is.

The survival function is cleaner than the mass function and is the form worth remembering, since $\{X>k\}$ is just the event that the first $k$ trials all failed,

$$\mathbf{P}(X>k)=(1-p)^k.$$

The mean and variance are

$$\mathbb{E}[X]=\frac{1}{p},\qquad \mathrm{var}(X)=\frac{1-p}{p^2}.$$

??? note "Proof of the mean, from a differentiated series and from one step of conditioning"
    The direct route treats $p$ as a variable and differentiates. Writing $q=1-p$,

    $$\mathbb{E}[X]=\sum_{k=1}^{\infty}k\,q^{k-1}p=p\,\frac{\mathrm{d}}{\mathrm{d}q}\sum_{k=1}^{\infty}q^{k}=p\,\frac{\mathrm{d}}{\mathrm{d}q}\Big(\frac{q}{1-q}\Big)=\frac{p}{(1-q)^2}=\frac{1}{p},$$

    where term-by-term differentiation is licensed inside the radius of convergence — the identities $\sum kq^{k}$ and $\sum k^2q^{k}$, and the theorem that permits differentiating a power series termwise, are [Sequences and Infinite Series](../part-01-mathematical-foundations/04-sequences-and-series.md). The same manipulation one order higher gives $\mathbb{E}[X^2]=(2-p)/p^2$ and hence the variance.

    The second route uses no series at all. Condition on the first trial: with probability $p$ the wait is $1$, and with probability $1-p$ we have spent one trial and face a fresh copy of the same problem. By the tower property of [Law of Total Expectation](../part-04-expectation-and-moments/07-law-of-total-expectation.md), $\mathbb{E}[X]=1+(1-p)\mathbb{E}[X]$, which solves to $1/p$ immediately.

    The second proof is worth more than the first, because the step "a fresh copy of the same problem" is exactly the memorylessness proved below, used before it was named. Every self-referential first-step argument in probability — hitting times on a Markov chain, the expected number of steps to ruin — is this identity in a larger setting, and it works only for laws where the remaining wait is distributed as the original.

## Two Conventions, and Which One the Formula Assumes

Half the confusion this family generates is bookkeeping. $X$ above counts *trials up to and including* the success and lives on $\{1,2,\ldots\}$; the alternative $Y=X-1$ counts *failures before* the success and lives on $\{0,1,\ldots\}$.

| Convention | Support | Mass at $k$ | Mean | Variance |
|---|---|---|---|---|
| Trials until the first success | $k\ge1$ | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ |
| Failures before the first success | $k\ge0$ | $(1-p)^{k}p$ | $(1-p)/p$ | $(1-p)/p^2$ |

The variance is the same because the two differ by a constant shift, and [Variance](../part-04-expectation-and-moments/02-variance.md) shows shifts are invisible to it. The means differ by exactly $1$. This matters in practice more than it should: `numpy`'s `geometric` uses the first convention and `scipy`'s `nbinom` the second, so a simulation and its analytic check will disagree by one unit of time if the convention is not fixed on purpose. This page uses the first throughout.

## Memorylessness Characterizes the Family

Conditioning on having already waited $s$ trials without success and asking for the chance of waiting $t$ more gives

$$\mathbf{P}(X>s+t\mid X>s)=\frac{(1-p)^{s+t}}{(1-p)^{s}}=(1-p)^{t}=\mathbf{P}(X>t).$$

The elapsed wait cancels. This is usually presented as a property the geometric happens to have; the stronger and more useful statement is that it is the *only* law on the positive integers that has it.

??? note "Proof that the geometric is the unique memoryless law on the positive integers"
    Let $X$ take values in $\{1,2,\ldots\}$ and satisfy $\mathbf{P}(X>s+t\mid X>s)=\mathbf{P}(X>t)$ for all positive integers $s,t$. Write $G(k)=\mathbf{P}(X>k)$. The condition says $G(s+t)=G(s)G(t)$, and $G(0)=1$.

    Take $s=t=1$ to get $G(2)=G(1)^2$, and induct: $G(k)=G(1)^k$ for every $k\ge0$. Set $q=G(1)$, which lies in $[0,1)$ since $X$ is finite with probability one, and let $p=1-q$. Then $\mathbf{P}(X>k)=q^k$, and differencing gives $\mathbf{P}(X=k)=G(k-1)-G(k)=q^{k-1}(1-q)=q^{k-1}p$. That is the geometric mass function, with no freedom left.

    The functional equation $G(s+t)=G(s)G(t)$ is the whole content, and it is doing the same work here that it does in the continuous case, where the same argument over the reals — with a measurability or monotonicity hypothesis added, since the rationals alone are not enough there — forces the exponential. So memorylessness is not one property among many; it *is* the family, and any evidence that a real waiting time has memory is evidence against the geometric and not against some incidental detail of it.

The same fact in survival language is that the hazard is constant. Defining the hazard at $k$ as the chance of succeeding at trial $k$ given that trial $k$ has been reached,

$$h(k)=\frac{\mathbf{P}(X=k)}{\mathbf{P}(X\ge k)}=\frac{q^{k-1}p}{q^{k-1}}=p,$$

independent of $k$. A geometric wait does not age. This is precisely the assumption that a "due for a bounce" argument denies, and it is why the sojourn time in a state of a Markov chain is geometric — [Markov Chains](../part-08-stochastic-processes/05-markov-chains.md) needs exactly this, because a chain whose transition probabilities do not depend on how long it has been in its current state cannot produce any other holding-time law.

```python
import numpy as np

p = 0.02                                                       # the index makes a new high 2% of days
rng = np.random.default_rng(13)
x = rng.geometric(p, 4_000_000)                                # trials-until-success convention
print(f"  Geom(p = {p}): mean {x.mean():7.2f} (exact {1 / p:.2f})"
      f"   sd {x.std():7.2f} (exact {np.sqrt(1 - p) / p:.2f})")
print("  the elapsed wait tells you nothing about the remaining wait")
print("     waited already   paths left   P(25 or more to go)   hazard next trial")
for s in (0, 50, 100, 200):
    tail = x[x > s]
    print(f"    {s:11d} {tail.size:12d} {(tail > s + 25).mean():18.4f}"
          f" {(tail == s + 1).mean():21.4f}")
print(f"  memoryless prediction, identical for every row:"
      f" {(1 - p) ** 25:.4f} and {p:.4f}")
# =>   Geom(p = 0.02): mean   49.95 (exact 50.00)   sd   49.42 (exact 49.50)
#      the elapsed wait tells you nothing about the remaining wait
#         waited already   paths left   P(25 or more to go)   hazard next trial
#                  0      4000000             0.6033                0.0200
#                 50      1455012             0.6030                0.0199
#                100       528929             0.6037                0.0201
#                200        69851             0.6065                0.0199
#      memoryless prediction, identical for every row: 0.6035 and 0.0200
```

The two right-hand columns are flat down the page while the column beside them falls by a factor of nearly sixty. That contrast is the point. Only one path in fifty-seven survives to trial $200$, so the population being asked about is drastically thinned — and the survivors' prospects are identical to a fresh start's: the same $0.60$ chance of twenty-five more trials, the same $0.02$ chance of finishing on the next one. The last line is the theoretical value both columns sit on, and it contains no $s$ at all. The elapsed wait is not an input to the prediction, which is the previous proof run as an experiment.

## The Mean Wait Is Not the Typical Wait

A geometric law is heavily right-skewed, so its mean sits well above its middle. The median is the smallest $k$ with $(1-p)^k\le\tfrac12$, which is about $0.693/p$ — roughly $69\%$ of the mean — and the fraction of waits that come in under the mean is

$$\mathbf{P}(X\le 1/p)=1-(1-p)^{1/p}\ \longrightarrow\ 1-e^{-1}\approx 0.632$$

as $p\to0$. Almost two thirds of waits are shorter than average, and the average is held up by a thin tail of very long ones. Reporting "the average time to recovery is fifty days" therefore describes a wait that most recoveries beat and a few miss by an enormous margin.

## Real Drawdowns Have Memory

The trading stake can now be settled. If underwater spells were geometric with $p=0.02$, the longest one in twenty-five years would be a maximum over roughly $128$ spells of mean $50$.

```python
import numpy as np

p, days, observed = 0.02, 6410, 2294                           # 2% new highs, 25y, longest spell
rng = np.random.default_rng(17)
n_spells = int(days * p)
longest = rng.geometric(p, (40_000, n_spells)).max(axis=1)
print(f"  {n_spells} geometric spells at p = {p}, over {days} days")
print(f"    median longest spell    {np.median(longest):8.0f} days")
print(f"    99th percentile         {np.quantile(longest, 0.99):8.0f} days")
print(f"    largest in 40,000 runs  {longest.max():8.0f} days")
print(f"  observed longest underwater spell {observed} days")
print(f"    P(one geometric spell exceeds it)   {(1 - p) ** observed:.3e}")
print(f"    P(any of {n_spells} spells does)          "
      f"{n_spells * (1 - p) ** observed:.3e}")
# =>   128 geometric spells at p = 0.02, over 6410 days
#        median longest spell         258 days
#        99th percentile              465 days
#        largest in 40,000 runs       718 days
#      observed longest underwater spell 2294 days
#        P(one geometric spell exceeds it)   7.458e-21
#        P(any of 128 spells does)          9.546e-19
```

The simulation and the exact calculation say the same thing from two directions. A memoryless model puts the longest spell at a median of $258$ days, exceeds $465$ only one time in a hundred, and in forty thousand simulated twenty-five-year histories never once produced a spell as long as $750$ days — let alone $2{,}294$. The closed form agrees: a single geometric spell exceeds the observed length with probability $7\times10^{-21}$, and even granting all $128$ spells a chance at it the total is under $10^{-18}$. The model is not slightly wrong. It is excluded.

!!! warning "A constant hazard is the assumption that makes a long drawdown impossible, and it is the assumption every simple recovery-time estimate makes"
    The failure is one-directional and it points the dangerous way. Real drawdowns have a *decreasing* hazard — the longer one has lasted, the longer it tends to continue, because the same regime that caused it is still in force — so the geometric understates long spells specifically. Any capital-planning or investor-communication number built on an average recovery time inherits that understatement. The repair is a family whose hazard is allowed a shape, which is [Weibull Distribution](18-weibull-distribution.md), or a model in which the success probability itself varies, which is the mixture route of [Negative Binomial Distribution](04-negative-binomial-distribution.md).

So the geometric earns its place as a null rather than as a model. Its uniqueness theorem makes it the exact hypothesis "nothing about the past matters", which is precisely what one wants to test a market claim against, and its constant hazard makes the test a one-line calculation. What it is not is a description of how long a real drawdown lasts, and the distance between those two roles is the distance between $258$ days and $2{,}294$. The practical rule is to use the geometric to compute what memorylessness would imply, and then to treat the gap between that and the data as the measurement — because the gap, not the fit, is where the information is.
