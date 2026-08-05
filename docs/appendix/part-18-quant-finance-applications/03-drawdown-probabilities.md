# Drawdown Probabilities

A drawdown is the number an investor reacts to and the number a model is least often asked about. This page supplies the two closed forms that answer it, and they separate cleanly: the depth below the running peak at a random time is exponential with mean $\sigma/2S$ — half the volatility over the Sharpe ratio — while the expected time to climb back from that depth is $1/(2S^{2})$ years, in which the volatility has cancelled entirely. Both are confirmed to within half a percentage point, $15.35\%$ against $15.02\%$ and $5.56$ years against $5.42$, and the cancellation is visible directly: doubling volatility at a fixed Sharpe of $0.65$ doubles the typical depth from $7.40\%$ to $14.26\%$ and leaves the recovery time at $1.18$ years in both rows. The maximum over a record obeys two different laws either side of a crossover at $1/S^{2}$ years — square root of the horizon below it, logarithm above — so a Sharpe-$0.30$ book needs eleven years before its drawdowns behave like a book with an edge at all. And the statistic itself decides almost nothing: over twenty years a Sharpe-$0.30$ book posts a *shallower* maximum drawdown than a Sharpe-$0.80$ book on $0.1741$ of paired records.

This page covers the stationary law of the drawdown below a running peak, the recovery time that follows from it by Wald's identity, the separation of depth from duration and what each depends on, the two horizon regimes for the maximum and the crossover between them, and what a realized maximum drawdown can and cannot establish about an edge. It derives no reflection principle and no running-maximum law, and it does not re-derive the expected shortfall from the high-water mark of a zero-edge process or the measure-zero argument behind "new equity highs on two percent of days," all of which are [Brownian Motion](../part-08-stochastic-processes/08-brownian-motion.md); it computes no probability of ever reaching a fixed multiple of starting wealth, which is [Probability of Ruin](02-probability-of-ruin.md); it proves no arcsine law for time spent above zero, which is [Random Walks](../part-08-stochastic-processes/11-random-walks.md); it derives no first-passage density, which is [Hitting and First-Passage Times](04-hitting-and-first-passage-times.md); it chooses no leverage, which is [Kelly Criterion](01-kelly-criterion.md); it resamples no real return series, which is [Bootstrap and Monte Carlo](../../part-03-statistics/05-bootstrap-and-monte-carlo.md); it fits no tail to the losses it counts, which is [Extreme Value Theory](13-extreme-value-theory.md); and it never reads a drawdown as evidence without the distribution that the same Sharpe ratio would have produced anyway.

The trading stake is a promise made by name in a course lesson. [Plotting for Research](../../part-02-python/06-plotting.md) prints that a strategy at a "healthy 0.65 Sharpe" spends **93% of all days below a previous peak**, observes that "depth, duration, and recovery are three distinct pains," and defers: "how long drawdowns *should* last for a given Sharpe is quantified in the appendix's [Drawdown Probabilities] — calibrating that expectation before going live is much cheaper than discovering it after." Section 2 is that calibration, and the answer for a Sharpe of $0.65$ is a typical depth of $7.40\%$ against a $10\%$-volatility book and a typical recovery of $1.18$ years. [Bootstrap and Monte Carlo](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) supplies the independent check at the other end of the range, reporting a median longest underwater stretch of **seven years** for a Sharpe-$0.30$ book, against the $6.75$ years section 3 measures over a twenty-year record.

## The Drawdown Below the Running Peak Has a Stationary Law, and It Is Exponential

The running maximum is not a fixed reference point, which is what makes a drawdown harder to reason about than a loss from a starting value. What rescues it is that the *gap* between a drifting process and its own running maximum is itself a well-behaved process with a limiting distribution, and the distribution is one line from the previous page.

??? note "Proof that the drawdown converges in law to an exponential with rate $2m/v^{2}$, that its mean is $\sigma/2S$, and that the expected recovery time from a depth $a$ is $a/m$"

    Let $Y_t=mt+vB_t$ be log wealth with $m>0$, and let $D_t=\max_{s\le t}Y_s-Y_t\ge0$ be the drawdown. Reversing time on $[0,t]$ maps the increments of $Y$ to those of a process with the same law, and carries $\max_{s\le t}Y_s-Y_t$ to $-\min_{s\le t}\tilde Y_s$ for the reversed path $\tilde Y$. So $D_t$ has the same distribution as the running minimum's magnitude, and letting $t\to\infty$,
    $$\mathbf{P}(D_\infty>a)=\mathbf{P}\!\left(\inf_s\tilde Y_s<-a\right)=e^{-2ma/v^{2}},$$
    the exponential rate being the one [Probability of Ruin](02-probability-of-ruin.md) obtained from the exponential martingale. So $D_\infty\sim\mathrm{Exp}(\lambda)$ with $\lambda=2m/v^{2}$ and mean $v^{2}/2m$. For a book with volatility $\sigma$ and Sharpe ratio $S$, so that $m=S\sigma$ and $v=\sigma$, the mean depth in log units is
    $$\mathbb{E}[D_\infty]=\frac{\sigma^{2}}{2S\sigma}=\frac{\sigma}{2S},$$
    the median is $\log 2$ times that, and the $95$th percentile is $\log 20$ times it.

    Recovery is the mirror image and needs no new machinery. From a depth $a$ the process must climb $a$ to set a new high, and for a Brownian motion with positive drift the first-passage time upward to a level $a$ has $\mathbb{E}[\tau_a]=a/m$ — Wald's identity applied to the stopped process, which is finite here precisely because the drift points at the barrier. Evaluated at the typical depth,
    $$\mathbb{E}[\text{recovery}]=\frac{\sigma/2S}{S\sigma}=\frac{1}{2S^{2}}\ \text{years},$$
    and every $\sigma$ has cancelled: two books at the same Sharpe ratio and different volatilities spend the *same* time underwater and differ only in how far down they are while they wait.

    **The load-bearing hypothesis is that the horizon is long compared with the relaxation time $v^{2}/m^{2}=1/S^{2}$ years. Below that the drawdown has not reached its stationary law, no mean $\sigma/2S$ exists to be measured, and the process is still behaving like the driftless one — which is the second regime of section 3 and the reason a short record cannot be read with these formulas at all.**

## Depth Scales With Volatility and Duration Does Not, So a Sharpe Ratio Alone Says How Long to Wait

The two formulas answer different questions and consume different inputs, which is the practically useful part: a desk that halves its volatility halves its drawdowns and waits exactly as long to get out of them.

```python
import numpy as np

rng = np.random.default_rng(18031)
PATHS, D, BURN, YEARS = 20_000, 252, 60, 120

print(f"  the drawdown below the running peak, sampled at a random late time. Its stationary law is"
      f" exponential with mean sigma^2/2mu, and the expected time to recover from that typical depth"
      f" is 1/(2 S^2) years -- both functions of the Sharpe ratio and the volatility only through"
      f" the combinations shown. {PATHS:,} paths x {YEARS} years, first {BURN} discarded")
print("     Sharpe   vol    mean depth: pred    meas   median: pred    meas   95th: pred    meas"
      "   P(deeper than 20%): pred    meas   recovery, yrs: pred    meas")
for S, vol in ((0.30, 0.10), (0.65, 0.10), (1.00, 0.10), (0.65, 0.20)):
    mu, sd = S * vol, vol
    lam = 2 * mu / sd ** 2                                  # rate of the stationary depth, log units
    n = YEARS * D
    y = np.cumsum(mu / D + sd / np.sqrt(D) * rng.standard_normal((PATHS, n)), axis=1)
    dd = (np.maximum.accumulate(y, axis=1) - y)[:, BURN * D:]

    # expected recovery time from the depth held at the sampling instant: a/mu by Wald
    rec = dd[:, 0] / mu
    pct = lambda a, z: -np.expm1(-np.percentile(a, z))
    print(f"    {S:6.2f}   {vol:4.0%}   {-np.expm1(-1 / lam):17.2%}   {-np.expm1(-dd.mean()):5.2%}"
          f"   {-np.expm1(-np.log(2) / lam):13.2%}   {pct(dd, 50):5.2%}"
          f"   {-np.expm1(-np.log(20) / lam):11.2%}   {pct(dd, 95):5.2%}"
          f"   {np.exp(-lam * -np.log(0.80)):24.4f}   {np.mean(dd > -np.log(0.80)):5.4f}"
          f"   {1 / (2 * S ** 2):27.2f}   {rec.mean():5.2f}")
# =>   the drawdown below the running peak, sampled at a random late time. Its stationary law is exponential with mean sigma^2/2mu, and the expected time to recover from that typical depth is 1/(2 S^2) years -- both functions of the Sharpe ratio and the volatility only through the combinations shown. 20,000 paths x 120 years, first 60 discarded
#         Sharpe   vol    mean depth: pred    meas   median: pred    meas   95th: pred    meas   P(deeper than 20%): pred    meas   recovery, yrs: pred    meas
#          0.30    10%              15.35%   15.02%          10.91%   10.54%        39.30%   39.09%                     0.2621   0.2556                          5.56    5.42
#          0.65    10%               7.40%   7.07%           5.19%   4.85%        20.58%   20.29%                     0.0550   0.0524                          1.18    1.12
#          1.00    10%               4.88%   4.54%           3.41%   3.05%        13.91%   13.60%                     0.0115   0.0109                          0.50    0.47
#          0.65    20%              14.26%   13.68%          10.11%   9.47%        36.93%   36.58%                     0.2345   0.2243                          1.18    1.12
```

Every predicted column lands within half a percentage point of its measurement, and the residual is one-signed and explainable: the running maximum is taken over a finite past rather than an infinite one, so the measured depth is always slightly the shallower of the two. The exponential shape holds across the whole distribution rather than at the mean alone — the $95$th percentile is predicted at $39.30\%$ and measured at $39.09\%$ for the Sharpe-$0.30$ book, and the probability of sitting more than $20\%$ below the peak is $0.2621$ against $0.2556$.

The last two rows are the separation. A Sharpe-$0.65$ book at $10\%$ volatility has a typical depth of $7.40\%$; the same Sharpe at $20\%$ volatility has $14.26\%$, twice as deep, because depth is $\sigma/2S$ and $\sigma$ doubled. Both recover in $1.18$ years, because recovery is $1/(2S^{2})$ and $\sigma$ is not in it. So the answer to the question [Plotting for Research](../../part-02-python/06-plotting.md) asked — how long should this last — needs only the Sharpe ratio, and the answer at $0.65$ is a bit over a year for a typical drawdown, roughly $2.4$ years for a median-to-$95$th-percentile one. **Depth is a decision about position size and duration is not, so the only lever a desk has over how long it waits is the edge itself.**

## Without Drift the Maximum Grows Like the Square Root of the Horizon and With It Like the Logarithm, and the Crossover Sits at One Over Sharpe Squared

The depth at a random time is a stationary quantity. The maximum over a whole record is not, and it obeys two completely different laws depending on whether the horizon is long enough for the drift to have asserted itself.

??? note "Proof that the maximum drawdown grows like $v\sqrt{T}$ with no drift and like $(v^{2}/2m)\log T$ with drift, and that the two regimes meet at $T\asymp1/S^{2}$"

    With $m=0$ there is no stationary law: the drawdown process is a reflected driftless Brownian motion, which is null recurrent and wanders without bound. Brownian scaling gives $\{Y_{cT}\}\overset{d}{=}\{\sqrt{c}\,Y_T\}$, and the maximum drawdown is a functional homogeneous of degree one in the path, so $\mathrm{MDD}_{cT}\overset{d}{=}\sqrt{c}\,\mathrm{MDD}_T$. Its expectation therefore grows exactly like $v\sqrt{T}$, and stretching the horizon by a factor of $50$ multiplies it by $\sqrt{50}=7.07$.

    With $m>0$ the drawdown is positive recurrent with stationary law $\mathrm{Exp}(\lambda)$, $\lambda=2m/v^{2}$, and it decorrelates over the relaxation time $v^{2}/m^{2}$. A horizon $T$ therefore contains on the order of $n\asymp Tm^{2}/v^{2}$ effectively independent excursions, and the maximum of $n$ exponential variables with rate $\lambda$ has expectation $(\log n+\gamma)/\lambda$. Hence
    $$\mathbb{E}[\mathrm{MDD}_T]\;\sim\;\frac{v^{2}}{2m}\log T+\text{const},$$
    logarithmic rather than square-root, with slope $v^{2}/2m=\sigma/2S$ — the same constant that section 1 gave as the typical depth.

    The regimes meet where the drift term $mT$ becomes comparable with the diffusive term $v\sqrt{T}$, that is at $T\asymp v^{2}/m^{2}=1/S^{2}$ years. Below the crossover the process has not yet noticed its own drift and its maximum drawdown grows like a zero-edge walk's; above it, growth is logarithmic and the record is informative. At a Sharpe of $1.0$ the crossover is one year, at $0.65$ it is $2.4$ years, and at $0.30$ it is $11$ years.

    **The load-bearing quantity is the dimensionless horizon $S^{2}T$, which is also the square of the t-statistic of the mean. So the drawdown of a record becomes readable at exactly the sample size at which the edge itself becomes statistically visible — one number governs both, and a book whose Sharpe ratio is not yet significant does not have interpretable drawdowns either.**

```python
import numpy as np

rng = np.random.default_rng(18033)
PATHS, D = 20_000, 252
HORIZONS = (1, 2, 5, 10, 20, 50)


def stats(S, vol, years):
    n = years * D
    y = np.cumsum(S * vol / D + vol / np.sqrt(D) * rng.standard_normal((PATHS, n)), axis=1)
    under = y < np.maximum.accumulate(y, axis=1)
    mdd_log = (np.maximum.accumulate(y, axis=1) - y).max(axis=1)
    longest, run = np.zeros(PATHS, int), np.zeros(PATHS, int)
    for j in range(n):
        run = np.where(under[:, j], run + 1, 0)
        longest = np.maximum(longest, run)
    return mdd_log, longest / D


print(f"  {PATHS:,} paths. Under positive drift the stationary drawdown is exponential with mean"
      f" sigma^2/2mu, so the maximum over a horizon grows like its logarithm; with no drift there"
      f" is no stationary law and the maximum grows like the square root of the horizon")
print("     Sharpe   vol    scale sigma^2/2mu   " + "".join(f"{h}y  E[maxDD]  longest   " for h in HORIZONS))
for S, vol in ((0.00, 0.10), (0.30, 0.10), (0.65, 0.10), (1.00, 0.10), (0.30, 0.16)):
    scale = vol ** 2 / (2 * S * vol) if S > 0 else float("nan")
    cells = []
    for h in HORIZONS:
        mdd, longest = stats(S, vol, h)
        cells.append(f"{-np.expm1(-mdd.mean()):9.1%}  {np.median(longest):6.2f}y   ")
    print(f"    {S:6.2f}   {vol:4.0%}   {scale:16.4f}   " + "".join(cells))

print("\n     growth in the maximum drawdown, measured in log units, against the two laws")
print("     Sharpe   vol    E[maxDD] 1y   50y   ratio   sqrt(50) = 7.07"
      "   slope per unit log-horizon   sigma^2/2mu")
for S, vol in ((0.00, 0.10), (0.30, 0.10), (0.65, 0.10), (1.00, 0.10), (0.30, 0.16)):
    m = np.array([stats(S, vol, h)[0].mean() for h in HORIZONS])
    slope = np.polyfit(np.log(HORIZONS), m, 1)[0]
    scale = vol ** 2 / (2 * S * vol) if S > 0 else float("nan")
    print(f"    {S:6.2f}   {vol:4.0%}   {m[0]:11.4f}   {m[-1]:.4f}   {m[-1] / m[0]:5.2f}"
          f"   {np.sqrt(50):16.2f}   {slope:27.4f}   {scale:11.4f}")
# =>   20,000 paths. Under positive drift the stationary drawdown is exponential with mean sigma^2/2mu, so the maximum over a horizon grows like its logarithm; with no drift there is no stationary law and the maximum grows like the square root of the horizon
#         Sharpe   vol    scale sigma^2/2mu   1y  E[maxDD]  longest   2y  E[maxDD]  longest   5y  E[maxDD]  longest   10y  E[maxDD]  longest   20y  E[maxDD]  longest   50y  E[maxDD]  longest   
#          0.00    10%                nan       11.1%    0.62y       15.5%    1.23y       24.0%    3.11y       32.3%    6.21y       42.4%   12.21y       58.6%   30.90y   
#          0.30    10%             0.1667       10.1%    0.54y       13.7%    1.02y       19.8%    2.29y       25.0%    3.98y       31.0%    6.75y       38.9%   12.27y   
#          0.65    10%             0.0769        9.0%    0.46y       11.9%    0.81y       16.3%    1.62y       19.8%    2.53y       23.4%    3.75y       28.2%    5.77y   
#          1.00    10%             0.0500        8.2%    0.39y       10.4%    0.65y       13.7%    1.18y       16.3%    1.73y       19.0%    2.38y       22.5%    3.39y   
#          0.30    16%             0.2667       15.6%    0.54y       20.9%    1.01y       29.7%    2.29y       37.1%    4.01y       44.8%    6.78y       54.6%   12.32y   
#
#         growth in the maximum drawdown, measured in log units, against the two laws
#         Sharpe   vol    E[maxDD] 1y   50y   ratio   sqrt(50) = 7.07   slope per unit log-horizon   sigma^2/2mu
#          0.00    10%        0.1179   0.8746    7.42               7.07                        0.1863           nan
#          0.30    10%        0.1065   0.4975    4.67               7.07                        0.0995        0.1667
#          0.65    10%        0.0945   0.3308    3.50               7.07                        0.0606        0.0769
#          1.00    10%        0.0852   0.2543    2.98               7.07                        0.0433        0.0500
#          0.30    16%        0.1710   0.7934    4.64               7.07                        0.1587        0.2667
```

The zero-edge row obeys its law to two decimals: stretching the horizon fifty-fold multiplies the expected maximum drawdown by $7.42$ against a predicted $\sqrt{50}=7.07$. Every row with an edge falls far short of that — $4.67$, $3.50$ and $2.98$ as the Sharpe rises — which is the logarithmic regime asserting itself, and the fitted slope per unit log-horizon climbs toward its asymptote $\sigma/2S$ from below, reaching $0.0995$ of a predicted $0.1667$ at Sharpe $0.30$, $0.0606$ of $0.0769$ at $0.65$ and $0.0433$ of $0.0500$ at $1.00$. The approach is exactly as slow as the crossover argument says: at Sharpe $0.30$ even fifty years gives $S^{2}T=4.5$, barely past the boundary, so the slope has reached only three-fifths of its limit.

Two collapses in the same table confirm the scaling from the other direction. The Sharpe-$0.30$ rows at $10\%$ and $16\%$ volatility report median longest underwater stretches of $0.54$, $1.02$, $2.29$, $3.98$, $6.75$ and $12.27$ years against $0.54$, $1.01$, $2.29$, $4.01$, $6.78$ and $12.32$ — identical at every horizon, because duration does not see volatility. Their log-space maximum drawdowns are $0.1710$ against $0.1065$ at one year, a ratio of $1.61$ against the $1.60$ the volatilities imply. And the twenty-year cell, $6.75$ years, is the independent check: [Bootstrap and Monte Carlo](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) block-bootstrapped a real Sharpe-$0.30$ strategy and reported a median longest underwater stretch of seven years.

!!! note "The depth right now, the deepest depth on record, the longest stretch underwater and the time still to go are four different drawdown numbers, and a tearsheet prints the second"
    **The depth at a random time** is the stationary quantity of section 1, exponential with mean $\sigma/2S$, and it is what an investor sees on an arbitrary Tuesday. **The maximum over a record** is an extreme-value statistic of that process, obeying section 3's two regimes, growing without bound as the record lengthens, and it is the one every tearsheet reports — which means a longer track record is penalized by the statistic that is supposed to describe its risk. **The longest stretch underwater** is a duration rather than a depth, depends on the Sharpe ratio alone, and is the number that actually determines whether a strategy survives its investors. **The time still to go** is a conditional forecast from the current depth, $a/m$ in expectation, and it is the only one of the four that is forward-looking. Confusing the second with the first inflates the perceived risk of any long record; confusing the second with the third is how a shallow, grinding, four-year drawdown gets described as milder than a sharp one twice its depth.

## A Realized Drawdown Ranks Two Books Correctly Barely More Often Than a Coin

Section 3 established that the maximum drawdown is an extreme-value statistic. Extreme-value statistics of a single record are noisy, and the practical question is what one of them can settle.

```python
import numpy as np

rng = np.random.default_rng(18035)
VOL, PATHS, D, YEARS, LIMIT = 0.10, 40_000, 252, 20, 0.30


def maxdd(S):
    y = np.cumsum(S * VOL / D + VOL / np.sqrt(D) * rng.standard_normal((PATHS, YEARS * D)), axis=1)
    return -np.expm1(-(np.maximum.accumulate(y, axis=1) - y).max(axis=1))


null = maxdd(0.0)                                    # matched zero-edge walks, same vol and horizon
ref = maxdd(0.80)
print(f"  realized maximum drawdown over {YEARS} years at {VOL:.0%} volatility, {PATHS:,} records per"
      f" Sharpe. The last three columns ask what a drawdown number decides: whether it separates this"
      f" book from a zero-edge walk, from a Sharpe-0.80 book, and whether a {LIMIT:.0%} limit fires")
print("     Sharpe   maxDD: 5th   median    95th   percentile vs zero-edge walks"
      "   P(shallower than the Sharpe-0.80 book)   P(breaches the limit)")
for S in (0.00, 0.30, 0.50, 0.80, 1.20):
    m = maxdd(S)
    med = np.median(m)
    print(f"    {S:6.2f}   {np.percentile(m, 5):10.1%}   {med:6.1%}   {np.percentile(m, 95):5.1%}"
          f"   {np.mean(null < med):29.2f}   {np.mean(m < ref):38.4f}   {np.mean(m > LIMIT):20.4f}")
# =>   realized maximum drawdown over 20 years at 10% volatility, 40,000 records per Sharpe. The last three columns ask what a drawdown number decides: whether it separates this book from a zero-edge walk, from a Sharpe-0.80 book, and whether a 30% limit fires
#         Sharpe   maxDD: 5th   median    95th   percentile vs zero-edge walks   P(shallower than the Sharpe-0.80 book)   P(breaches the limit)
#          0.00        23.7%    39.8%   62.6%                            0.50                                   0.0495                 0.8039
#          0.30        18.5%    28.9%   47.4%                            0.17                                   0.1741                 0.4502
#          0.50        16.4%    24.6%   39.3%                            0.07                                   0.2938                 0.2361
#          0.80        14.0%    20.1%   31.2%                            0.01                                   0.5004                 0.0659
#          1.20        11.8%    16.4%   24.6%                            0.00                                   0.7335                 0.0078
```

The statistic does one job well and the other badly, and the two are easy to confuse. Against the null it is informative: a Sharpe-$0.30$ book's median maximum drawdown of $28.9\%$ sits at the $17$th percentile of zero-edge walks at the same volatility, a Sharpe-$0.80$ book's $20.1\%$ at the $1$st, and a Sharpe-$1.20$ book's $16.4\%$ below anything the null produces. A drawdown that is shallow *for its volatility* is real evidence of an edge, which is the same argument [Brownian Motion](../part-08-stochastic-processes/08-brownian-motion.md) makes in the other direction about drawdowns that are not.

Ranking two books that both have edges is where it collapses. The distributions overlap almost completely — Sharpe $0.30$ spans $18.5\%$ to $47.4\%$ between its fifth and ninety-fifth percentiles, Sharpe $0.80$ spans $14.0\%$ to $31.2\%$ — and on paired twenty-year records the weaker book posts the shallower maximum drawdown $17.41\%$ of the time. At Sharpe $0.50$ against $0.80$ it is $29.38\%$, one record in three. Even a book with no edge at all beats the Sharpe-$0.80$ book on $4.95\%$ of records.

The limit column turns that into a policy cost. A $30\%$ drawdown limit — a common institutional trigger — fires on $45.02\%$ of Sharpe-$0.30$ books, which is arguably the intent, and on $6.59\%$ of Sharpe-$0.80$ books, which is not: one good manager in fifteen is dismissed over twenty years by a rule that has observed nothing except the volatility it already authorized. **A maximum drawdown is a sample maximum, its sampling distribution is as wide as the quantity itself, and no amount of care in computing it narrows a spread that is a property of the process rather than of the measurement.**

## Every Repair Is a Longer Record, a Different Statistic, or the Null Nobody Simulates

The three failures on this page have three different remedies and one of them is free. Section 3's crossover cannot be repaired at all: a record shorter than $1/S^{2}$ years is in the driftless regime and its maximum drawdown is a statement about volatility, so the only fix is more calendar time, and [Geometric Brownian Motion](../part-08-stochastic-processes/09-geometric-brownian-motion.md) proves that sampling more often within the same span does not help. Section 4's ranking failure is repaired by changing statistic rather than by collecting more of the same one: the depth at a random time and the fraction of time underwater are averages over the whole record rather than maxima of it, so they concentrate at the usual $1/\sqrt{n}$ rate while the maximum does not.

The third remedy is the one section 1 makes available and nobody uses. Both closed forms are computable from two numbers already on every tearsheet, so the expected drawdown profile of a strategy can be written down before it is traded — which is precisely what [Plotting for Research](../../part-02-python/06-plotting.md) means by "calibrating that expectation before going live is much cheaper than discovering it after."

!!! warning "A drawdown is compared against a threshold chosen by a committee and never against the distribution the strategy's own Sharpe ratio implies"
    Risk limits are set in round numbers — twenty percent, thirty percent — and a breach is read as evidence that something changed. Section 4 shows a $30\%$ limit firing on $6.59\%$ of genuinely good books over twenty years with nothing having changed at all. **The free diagnostic is the pair $\sigma/2S$ and $1/(2S^{2})$, evaluated from the Sharpe ratio and volatility the tearsheet already reports: the typical depth and the typical time to recover, with no simulation, no new data and no model beyond the one already assumed when the Sharpe ratio was annualized.** For the course's own $0.65$-Sharpe example at $10\%$ volatility they read $7.4\%$ and $1.2$ years, and the ninety-fifth percentile of depth is $20.6\%$ — so a limit set at $20\%$ would fire on roughly one drawdown in twenty by construction. A threshold that has not been checked against these two numbers is not a risk limit; it is a schedule for firing people at a rate nobody computed.

## A Number Set by Volatility and a Clock Set by Sharpe

This page established that the drawdown below the running peak converges to an exponential with rate $2m/v^{2}$, so its typical depth is $\sigma/2S$ and its expected recovery time $1/(2S^{2})$ years, verified at $15.35\%$ against $15.02\%$, $39.30\%$ against $39.09\%$ and $5.56$ years against $5.42$, with volatility cancelling out of the clock so that a Sharpe-$0.65$ book recovers in $1.18$ years at $10\%$ and at $20\%$ volatility while its depth doubles from $7.40\%$ to $14.26\%$; that the maximum over a record grows like $\sqrt{T}$ without drift, confirmed at a fifty-year multiple of $7.42$ against $7.07$, and like $\log T$ with it, at fitted slopes of $0.0995$, $0.0606$ and $0.0433$ approaching $\sigma/2S$ values of $0.1667$, $0.0769$ and $0.0500$ from below, with the crossover at $1/S^{2}$ years putting a Sharpe-$0.30$ book in the driftless regime for its first eleven; that duration is invariant to volatility at every horizon, $0.54$, $2.29$, $6.75$ and $12.27$ years against $0.54$, $2.29$, $6.78$ and $12.32$, with the twenty-year figure matching the seven-year median a course lesson block-bootstrapped from real returns; and that a realized maximum drawdown separates an edge from no edge at the $17$th, $7$th and $1$st percentiles of the null while ranking a Sharpe-$0.30$ book above a Sharpe-$0.80$ one on $0.1741$ of paired records, with a $30\%$ limit firing on $0.0659$ of the good ones.

The contrast with [Probability of Ruin](02-probability-of-ruin.md) is worth holding onto, because the two pages measure the same paths and disagree about what is knowable. Ruin from a starting value has an exact probability with a closed form in one parameter, and its whole difficulty is that the parameter is unobservable. Drawdown from a running peak has an exact stationary law in *observable* parameters — a Sharpe ratio and a volatility, both on the tearsheet — and its difficulty is that the statistic everyone reports is the maximum of that law rather than a draw from it, so the quantity is knowable and the number is not. What both pages have assumed is that the barrier, once reached, is reached by a path that arrives there in finite and predictable time, and neither has said anything about *when*. That is [Hitting and First-Passage Times](04-hitting-and-first-passage-times.md).

**A drawdown's depth is bought with position size and its duration is bought with edge, so the only way to be underwater for less time is to be better, and the only way to be less far under is to be smaller.**
