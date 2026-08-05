# Order Arrival Processes

[Poisson Processes](../part-08-stochastic-processes/03-poisson-processes.md) ends its scope fence with "it models no market impact, which is [Part XVIII]," and the reason it cannot is structural rather than a matter of coverage: a Poisson rate is a number fixed before the process starts, and order flow responds to itself. Replacing the parameter with a state gives the Hawkes process, whose stationary intensity is $\lambda_0/(1-n)$ in the branching ratio $n$ — the same hyperbolic blow-up in $1-n$ that [Queue Models](05-queue-models.md) found in $1-\rho$, verified here at $2.0000$, $3.3333$, $10.0000$ and $20.0000$ against measured $1.9963$, $3.3212$, $10.0366$ and $20.0424$ — and in which exactly a fraction $n$ of all events are the market answering itself, measured at $0.8008$ when $n=0.8$. The diagnostic consequence sharpens a warning the earlier page could only state qualitatively: the dispersion ratio of an $n=0.85$ process reads $1.643$ in windows narrower than the kernel's memory and $43.667$ in windows far wider, against a limit of $44.44$, so a single dispersion number is uninterpretable without its window and the *direction* of its drift with scale identifies the mechanism. And a Poisson fit to this flow recovers the rate to four significant figures while exceeding its own $99.9$th percentile $131.7$ times too often and understating the busiest window by a factor of four.

This page covers self-excitation as a rate that carries state, the branching ratio and the cluster representation that makes it a Galton–Watson mean, the stationarity condition and the explosion at one, the exact scale-dependence of the dispersion ratio and what its direction identifies, the random time change that turns any point process into a unit Poisson process, and what a first-moment fit to clustered flow gets right and wrong. It does not construct the Poisson process or prove superposition, thinning and conditional uniformity, and it does not re-derive the general point that a count is a projection which destroys arrangement, all of which are [Poisson Processes](../part-08-stochastic-processes/03-poisson-processes.md); it does not develop the count law or its equidispersion, which is [Poisson Distribution](../part-05-common-distributions/06-poisson-distribution.md); it does not free the gap law in a renewal sense, which is [Renewal Processes](../part-08-stochastic-processes/04-renewal-processes.md); it serves no queue and computes no waiting time, which is [Queue Models](05-queue-models.md); it estimates no impact function and derives no propagator, which is [Market Impact Models](../../advanced/05-market-impact-models.md); it schedules no execution, which is [Optimal Execution](../../advanced/04-optimal-execution-almgren-chriss.md); it quotes no spread, which is [Market Making](../../advanced/12-market-making.md); and it never reports a dispersion ratio without the window it was computed on.

The trading stake is the other half of a clustering result a course module measures. [Market Impact Models](../../advanced/05-market-impact-models.md) reports that "the sign of successive market orders exhibits long memory, with autocorrelation $C(\ell)\sim\ell^{-\gamma_c}$ and $\gamma_c$ measured around $0.5$ in equity markets, persisting over thousands of trades," and resolves the resulting paradox with a propagator whose decay is forced to $\beta_d\approx(1-\gamma_c)/2$. That result is about *which way* orders go. This page is about *when* they arrive, which is a separate process with its own memory, and the two are routinely conflated because both are called "order flow is autocorrelated." Section 2 measures the timing memory directly as the fraction of events that are endogenous, and section 4 shows what modelling it away costs.

## The Rate of a Poisson Process Is a Parameter, and the Intensity of a Hawkes Process Is a State

Every property of the Poisson process traces to one assumption: the rate at which the next event arrives does not depend on what has already happened. Relaxing exactly that, and nothing else, produces a process with a closed-form stationary intensity and a single number governing everything.

??? note "Proof that a self-exciting process has stationary intensity $\lambda_0/(1-n)$ with branching ratio $n=\int_0^\infty\phi$, that clusters have mean size $1/(1-n)$, and that both diverge at $n=1$"

    Let the conditional intensity be $\lambda(t)=\lambda_0+\int_{-\infty}^{t}\phi(t-s)\,dN(s)$ with $\phi\ge0$ and $n=\int_0^\infty\phi(u)\,du<\infty$. Taking expectations in stationarity, where $\mathbb{E}[dN(s)]=\bar\lambda\,ds$,
    $$\bar\lambda=\lambda_0+\bar\lambda\int_0^\infty\phi(u)\,du=\lambda_0+n\bar\lambda\quad\Longrightarrow\quad\bar\lambda=\frac{\lambda_0}{1-n},$$
    finite exactly when $n<1$. The algebra is identical to the queue's $\rho/(1-\rho)$ and for the same reason: a quantity feeding back into itself at gain $n$ sums a geometric series.

    The cluster representation makes $n$ interpretable. The process is equivalent to *immigrants* arriving as a Poisson process of rate $\lambda_0$, each of which produces offspring at rate $\phi$ after its own arrival, each offspring doing likewise. Offspring counts are Poisson with mean $n=\int\phi$, so a cluster is a Galton–Watson tree with offspring mean $n$, and its total progeny has expectation
    $$1+n+n^{2}+\cdots=\frac{1}{1-n},$$
    finite iff $n<1$ and infinite at $n=1$ — the extinction/explosion threshold of a branching process, arriving here as a stationarity condition. Since immigrants arrive at $\lambda_0$ and each yields $1/(1-n)$ events, the overall rate is $\lambda_0/(1-n)$ again, and the fraction of events that are *not* immigrants is exactly $n$.

    **The load-bearing quantity is $n$, and its interpretation is the useful part: it is simultaneously the gain of the feedback loop, the mean offspring count, and the fraction of all activity that is endogenous. A market with $n=0.8$ is one where four events in five happened because another event happened, and no amount of news accounts for them.**

## The Branching Ratio Is What Utilization Was, and It Says What Fraction of the Flow Is the Market Answering Itself

The cluster representation is also the fastest exact way to simulate the process, and simulating it that way makes the decomposition directly observable: every event knows whether it was an immigrant.

```python
import numpy as np

rng = np.random.default_rng(18061)
LAM0, BETA, T = 2.0, 4.0, 200_000.0                     # exogenous rate, kernel decay, horizon


def hawkes(n, horizon):
    """Exact simulation by the immigrant-and-offspring representation: exogenous events
    arrive at rate LAM0, each event spawns Poisson(n) children at Exp(BETA) delays."""
    times = rng.uniform(0, horizon, rng.poisson(LAM0 * horizon))
    out, gen = [times], times
    while len(gen):
        k = rng.poisson(n, len(gen))
        parents = np.repeat(gen, k)
        gen = parents + rng.exponential(1 / BETA, len(parents))
        gen = gen[gen < horizon]
        out.append(gen)
    return np.sort(np.concatenate(out)), len(times)


print(f"  a Hawkes process with exogenous rate {LAM0:.0f} and an exponential kernel decaying at"
      f" {BETA:.0f}, so the branching ratio n is the expected number of children per event. The"
      f" stationary intensity is lam0/(1-n) and a fraction n of all events are endogenous."
      f" horizon {T:,.0f}")
print("     branching ratio n   stationary rate: predicted   measured   endogenous share:"
      " predicted   measured   mean cluster size 1/(1-n)   measured")
for n in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9):
    ev, n_imm = hawkes(n, T)
    rate = len(ev) / T
    print(f"    {n:17.1f}   {LAM0 / (1 - n):27.4f}   {rate:8.4f}"
          f"   {n:28.4f}   {1 - n_imm / len(ev):8.4f}"
          f"   {1 / (1 - n):26.3f}   {len(ev) / n_imm:8.3f}")
# =>   a Hawkes process with exogenous rate 2 and an exponential kernel decaying at 4, so the branching ratio n is the expected number of children per event. The stationary intensity is lam0/(1-n) and a fraction n of all events are endogenous. horizon 200,000
#         branching ratio n   stationary rate: predicted   measured   endogenous share: predicted   measured   mean cluster size 1/(1-n)   measured
#                      0.0                        2.0000     1.9963                         0.0000     0.0000                        1.000      1.000
#                      0.2                        2.5000     2.4983                         0.2000     0.2000                        1.250      1.250
#                      0.4                        3.3333     3.3212                         0.4000     0.3989                        1.667      1.664
#                      0.6                        5.0000     4.9824                         0.6000     0.5991                        2.500      2.494
#                      0.8                       10.0000    10.0366                         0.8000     0.8008                        5.000      5.021
#                      0.9                       20.0000    20.0424                         0.9000     0.9004                       10.000     10.039
```

Three predictions and three matches in every row. The stationary rate tracks $\lambda_0/(1-n)$ from $1.9963$ to $20.0424$, the endogenous share tracks $n$ itself at $0.2000$, $0.3989$, $0.5991$, $0.8008$ and $0.9004$, and the mean cluster size tracks $1/(1-n)$ at $1.250$, $1.664$, $2.494$, $5.021$ and $10.039$.

The middle column is the one worth carrying to a desk. At $n=0.9$ the exogenous rate is unchanged at $2$ — the same amount of genuine news arrives — while the observed rate is $20$, and nine events in ten exist because another event did. The volume statistic and the information statistic have separated by a factor of ten, and no measurement of the rate alone can tell them apart. **The branching ratio is to a point process exactly what utilization was to a queue: a dimensionless feedback gain, entering hyperbolically through $1-n$, undefined above one, and invisible in the first moment.**

## The Dispersion Ratio Reads One at Fine Resolution and Forty-Four at Coarse, and the Direction Names the Mechanism

[Poisson Processes](../part-08-stochastic-processes/03-poisson-processes.md) warns that "every Poisson diagnostic is a diagnostic at one timescale," and exhibits a clustered pattern that *passes* at the daily scale and fails at the minute. Self-excitation with a short-memory kernel fails the same test in the opposite direction, and the closed form says by how much.

??? note "Proof that the dispersion ratio of a Hawkes count tends to $1$ for windows short against the kernel's memory and to $1/(1-n)^{2}$ for windows long against it"

    Let $N_W$ be the count in a window of width $W$. For $W$ much smaller than the kernel's memory $1/\beta$, a window contains at most one member of any cluster with probability approaching one, so the counts behave like thinned independent draws and $\mathrm{Var}(N_W)/\mathbb{E}[N_W]\to1$: the process looks Poisson at fine resolution precisely because clustering happens on a timescale the window cannot see.

    For $W$ much larger than $1/\beta$, whole clusters fall inside single windows. The cluster arrivals are Poisson at rate $\lambda_0$ and cluster sizes $C$ are i.i.d. with mean $1/(1-n)$, so $N_W$ is a compound Poisson sum and
    $$\frac{\mathrm{Var}(N_W)}{\mathbb{E}[N_W]}=\frac{\lambda_0W\,\mathbb{E}[C^{2}]}{\lambda_0W\,\mathbb{E}[C]}=\frac{\mathbb{E}[C^{2}]}{\mathbb{E}[C]}.$$
    For a Galton–Watson total progeny with Poisson offspring of mean $n$, $\mathbb{E}[C]=1/(1-n)$ and $\mathrm{Var}(C)=n/(1-n)^{3}$, giving $\mathbb{E}[C^{2}]/\mathbb{E}[C]=1/(1-n)^{2}$. So the ratio rises from $1$ to $1/(1-n)^{2}$ as the window widens through the kernel's memory, which is $2.04$ at $n=0.3$ and $44.44$ at $n=0.85$.

    The direction is the diagnostic. A process whose dispersion ratio *rises* with window width has clustering on a timescale shorter than the wide window — short-memory self-excitation. A process whose ratio *falls* with window width, which is the pattern [Poisson Processes](../part-08-stochastic-processes/03-poisson-processes.md) exhibits, has a rate varying slowly enough that wide windows average it away. The two mechanisms are opposite and a single ratio at a single scale cannot distinguish them from each other or from a genuine Poisson process.

    **The load-bearing quantity is the ratio of the window to the kernel's memory, which is a modelling choice rather than a property of the data. So the dispersion ratio is not a statistic of the process at all until a window is named, and reporting one without its window is reporting a free parameter.**

```python
import numpy as np

rng = np.random.default_rng(18063)
LAM0, BETA, T = 2.0, 4.0, 400_000.0
WINDOWS = (0.05, 0.25, 1.0, 5.0, 25.0, 200.0)


def hawkes(n, horizon):
    times = rng.uniform(0, horizon, rng.poisson(LAM0 * horizon))
    out, gen = [times], times
    while len(gen):
        k = rng.poisson(n, len(gen))
        parents = np.repeat(gen, k)
        gen = parents + rng.exponential(1 / BETA, len(parents))
        gen = gen[gen < horizon]
        out.append(gen)
    return np.sort(np.concatenate(out))


print(f"  the dispersion ratio Var(N)/E[N] of a Hawkes process, measured in windows of increasing"
      f" width. The kernel decays at rate {BETA:.0f}, so its memory is {1 / BETA:.2f} time units."
      f" Below that the process looks Poisson; far above it the ratio approaches 1/(1-n)^2."
      f" horizon {T:,.0f}")
print("     n     1/(1-n)^2   " + "".join(f"window {w:g}   " for w in WINDOWS))
for n in (0.0, 0.3, 0.5, 0.7, 0.85):
    ev = hawkes(n, T)
    cells = []
    for w in WINDOWS:
        counts = np.bincount((ev / w).astype(int), minlength=int(T / w))[:int(T / w)]
        cells.append(f"{counts.var() / counts.mean():{max(9, len(f'window {w:g}'))}.3f}   ")
    print(f"    {n:4.2f}   {1 / (1 - n) ** 2:9.2f}   " + "".join(cells))
# =>   the dispersion ratio Var(N)/E[N] of a Hawkes process, measured in windows of increasing width. The kernel decays at rate 4, so its memory is 0.25 time units. Below that the process looks Poisson; far above it the ratio approaches 1/(1-n)^2. horizon 400,000
#         n     1/(1-n)^2   window 0.05   window 0.25   window 1   window 5   window 25   window 200   
#        0.00        1.00         1.000         1.000       1.002       1.000       0.999        1.007   
#        0.30        2.04         1.070         1.293       1.690       1.959       2.017        2.008   
#        0.50        4.00         1.144         1.630       2.684       3.671       3.897        3.837   
#        0.70       11.11         1.296         2.368       5.189       9.361      10.859       11.138   
#        0.85       44.44         1.643         4.089      11.714      30.353      40.640       43.667   
```

Every row starts near one and ends at its predicted limit: $n=0.3$ runs $1.070$ to $2.008$ against $2.04$, $n=0.7$ runs $1.296$ to $11.138$ against $11.11$, and $n=0.85$ runs $1.643$ to $43.667$ against $44.44$. The transition happens across the kernel's memory of $0.25$ time units, exactly where the proof says it must.

The practical reading is uncomfortable. A desk testing this flow for Poisson behaviour at a resolution of $0.05$ finds a dispersion ratio of $1.643$ on the most violently self-exciting row in the table — mildly elevated, easily attributed to estimation noise, and passing any reasonable threshold. The same data at a resolution of $200$ gives $43.667$. Both numbers are correct, both describe the same process, and the model each supports is completely different. **A diagnostic whose answer moves by a factor of twenty-six across the range of windows a reasonable analyst might choose is not a test of the model but a measurement of the analyst's choice, and the only defence is to run it at two scales and report the direction.**

!!! note "The exogenous rate, the branching ratio, the stationary intensity and the kernel decay are four parameters of one process, and a Poisson fit returns a function of all four while naming only the third"
    **The exogenous rate** $\lambda_0$ is the arrival rate of genuine outside events — news, hedging demand, scheduled flow — and is the only one of the four that a Poisson model has a slot for. **The branching ratio** $n=\int\phi$ is the feedback gain, the mean offspring count and the endogenous share, all at once; it is dimensionless and it is what section 2 measures. **The stationary intensity** $\lambda_0/(1-n)$ is the observed rate, the product of the first two, and the only one directly visible in a trade log. **The kernel decay** sets the memory over which excitation persists and does not affect the stationary rate at all — but it decides the window at which section 3's diagnostic flips. A Poisson fit estimates the third and reports it as though it were the first, which is why a doubling of measured volume is compatible with unchanged news and with a market that has merely become more reflexive.

## A Poisson Fit Gets the Rate Right and the Busiest Window Wrong by a Factor of Four

Sections 2 and 3 both showed the first moment surviving intact while the process misbehaved. That is the failure mode worth pricing, because a fitted rate is what capacity planning, fill modelling and risk sizing all consume. The repair is a different test entirely, and it is exact.

??? note "Proof that the compensator transforms any point process into a unit-rate Poisson process, so the correct goodness-of-fit test is a time change rather than a moment ratio"

    For a point process with conditional intensity $\lambda(t)$, define the compensator $\Lambda(t)=\int_0^t\lambda(s)\,ds$ and map each event time $t_k$ to $\Lambda(t_k)$. The random time change theorem states that on the new clock the events form a **unit-rate Poisson process**, so the transformed gaps $\Lambda(t_k)-\Lambda(t_{k-1})$ are i.i.d. $\mathrm{Exp}(1)$, whatever the original process was. The intuition is that the compensator measures elapsed time in units of expected events, and by construction one event is expected per unit of that clock.

    This gives a goodness-of-fit test with no moments in it: fit a model, compute its compensator, transform the event times, and test the gaps against $\mathrm{Exp}(1)$. Under a correct model the test passes at every scale simultaneously, because the exponential law of the transformed gaps is a statement about the full distribution rather than about a ratio of its first two moments. Under the wrong model it fails, and the failure is not scale-dependent in the way section 3's ratio is.

    For the exponential kernel the compensator is computable in one pass. With $\phi(u)=n\beta e^{-\beta u}$ and $S_k=\sum_{i<k}e^{-\beta(t_k-t_i)}$ satisfying the recursion $S_k=e^{-\beta(t_k-t_{k-1})}(S_{k-1}+1)$,
    $$\Lambda(t_k)=\lambda_0t_k+n\left[(k-1)-S_k\right].$$

    **The load-bearing hypothesis is that the intensity used in the compensator is the true one. The test therefore checks a complete model rather than a single feature, which is its strength and its cost: it cannot say a process is "not Poisson" without a specific alternative to integrate, whereas a dispersion ratio needs no alternative and answers a correspondingly weaker question.**

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18065)
LAM0, BETA, T, W = 2.0, 4.0, 100_000.0, 1.0


def hawkes(n, horizon):
    times = rng.uniform(0, horizon, rng.poisson(LAM0 * horizon))
    out, gen = [times], times
    while len(gen):
        k = rng.poisson(n, len(gen))
        parents = np.repeat(gen, k)
        gen = parents + rng.exponential(1 / BETA, len(parents))
        gen = gen[gen < horizon]
        out.append(gen)
    return np.sort(np.concatenate(out))


def compensator_gaps(ev, n):
    """Lam(t_k) = lam0 t_k + n[(k-1) - S_k] with S_k = exp(-beta dt)(S_{k-1} + 1)."""
    s, prev, out = 0.0, 0.0, np.empty(len(ev))
    for k, t in enumerate(ev):
        s = np.exp(-BETA * (t - prev)) * (s + 1.0) if k else 0.0
        out[k] = LAM0 * t + n * (k - s)
        prev = t
    return np.diff(out)


print(f"  order flow generated by a Hawkes process and modelled as Poisson. The fitted rate is"
      f" unbiased in every row, so every first-moment check passes; the desk then sizes for the"
      f" Poisson 99.9th percentile of the count in a window of width {W:g}. horizon {T:,.0f}")
print("     n     events   fitted rate   true rate   Poisson 99.9th pct   actual exceedance"
      "   times nominal   busiest window: Poisson max    actual"
      "   KS vs Exp(1): as Poisson   after the Hawkes time change")
for n in (0.0, 0.3, 0.5, 0.7, 0.85):
    ev = hawkes(n, T)
    rate = len(ev) / T
    counts = np.bincount((ev / W).astype(int), minlength=int(T / W))[:int(T / W)]
    q = stats.poisson.ppf(0.999, rate * W)
    exceed = np.mean(counts > q)
    pois_max = stats.poisson.ppf(1 - 1 / len(counts), rate * W)
    ks_pois = stats.kstest(np.diff(ev) * rate, "expon").statistic
    ks_hawk = stats.kstest(compensator_gaps(ev, n), "expon").statistic
    print(f"    {n:4.2f}   {len(ev):6d}   {rate:11.4f}   {LAM0 / (1 - n):9.4f}   {q:18.0f}"
          f"   {exceed:17.5f}   {exceed / 0.001:13.1f}x   {pois_max:27.0f}   {counts.max():9d}"
          f"   {ks_pois:24.4f}   {ks_hawk:29.4f}")
# =>   order flow generated by a Hawkes process and modelled as Poisson. The fitted rate is unbiased in every row, so every first-moment check passes; the desk then sizes for the Poisson 99.9th percentile of the count in a window of width 1. horizon 100,000
#         n     events   fitted rate   true rate   Poisson 99.9th pct   actual exceedance   times nominal   busiest window: Poisson max    actual   KS vs Exp(1): as Poisson   after the Hawkes time change
#        0.00   200141        2.0014      2.0000                    8             0.00026             0.3x                            10          11                     0.0019                          0.0018
#        0.30   286080        2.8608      2.8571                    9             0.01119            11.2x                            12          21                     0.0778                          0.0011
#        0.50   402537        4.0254      4.0000                   11             0.03377            33.8x                            15          36                     0.1169                          0.0016
#        0.70   667508        6.6751      6.6667                   16             0.06992            69.9x                            20          68                     0.1478                          0.0010
#        0.85   1338307       13.3831     13.3333                   26             0.13172           131.7x                            32         129                     0.1602                          0.0005
```

The fitted rate is right in every row — $2.0014$, $2.8608$, $4.0254$, $6.6751$ and $13.3831$ against true values of $2.0000$, $2.8571$, $4.0000$, $6.6667$ and $13.3333$ — which is the whole problem, because it means every check a desk actually runs will pass. Volume forecasts, capacity plans and participation targets all consume that number and all of them are correct.

The tail is not. The Poisson $99.9$th percentile is exceeded $11.2$, $33.8$, $69.9$ and $131.7$ times too often as $n$ rises, and the busiest window over a hundred thousand windows contains $129$ events where the Poisson model's maximum is $32$ — a factor of four in the quantity that decides whether a system falls over. A capacity plan sized at three standard deviations above a correctly estimated mean is sized for a process that does not exist.

The last two columns are the repair and they are decisive. Treating the gaps as exponential under the fitted Poisson model gives Kolmogorov–Smirnov statistics of $0.0778$, $0.1169$, $0.1478$ and $0.1602$, rising monotonically with the misspecification; applying the correct compensator first gives $0.0011$, $0.0016$, $0.0010$ and $0.0005$, indistinguishable from the $0.0018$ of the genuinely Poisson row. **The information needed to detect this was in the inter-arrival times all along, and the dispersion ratio discarded it by projecting onto a count — which is [Poisson Processes](../part-08-stochastic-processes/03-poisson-processes.md)'s point about arrangement, arriving as a test that recovers what the projection lost.**

## Every Repair Is a Second Parameter, a Time Change, or a Diagnostic Run at Two Scales

The three failures have three remedies, and they cost increasing amounts of commitment. The cheapest is section 3's: run the dispersion ratio at two window widths instead of one and report the direction of travel, which needs no model and no new data and distinguishes short-memory self-excitation from a slowly-varying rate — two mechanisms that a single ratio confuses with each other and with Poisson. The next is section 1's: fit one extra parameter. A Hawkes model has two where Poisson has one, and the second is dimensionless, interpretable and bounded in $[0,1)$, so it is a cheap addition with an unusually clear meaning.

The most expensive and most complete is section 4's time change, which requires committing to a full conditional intensity before it will say anything. That is a real cost — it cannot report "not Poisson" without an alternative to integrate — and it is what buys a test that does not depend on a window. The trade is exactly the one [Poisson Processes](../part-08-stochastic-processes/03-poisson-processes.md) sets up when it observes that a count is a projection: recovering what the projection destroyed requires modelling the thing that was destroyed.

!!! warning "A dispersion ratio has a window and the window is almost never reported, so the same data supports either conclusion"
    Section 3 produces $1.643$ and $43.667$ from one process, one dataset and one statistic, differing only in a choice nobody writes down. **The free diagnostic is the same ratio computed at two widths an order of magnitude apart, with the pair reported rather than either alone: rising with width means short-memory self-excitation with $1/(1-n)^{2}$ readable off the wide end, falling with width means a slowly varying rate, and flat means the process really is Poisson at both scales.** It costs one extra line over the ratio a desk already computes, requires no model, and converts a number that can be argued either way into a statement about mechanism. The stronger version — fit the two-parameter model and test the compensated gaps — takes the Kolmogorov–Smirnov statistic from $0.1602$ to $0.0005$, but it needs a commitment the two-window check does not.

## A Rate That Answers to Itself, and the One Number That Says How Much

This page established that replacing a Poisson rate with a self-exciting intensity gives stationary rate $\lambda_0/(1-n)$, cluster size $1/(1-n)$ and endogenous share exactly $n$, all three verified — rates of $1.9963$, $3.3212$, $10.0366$ and $20.0424$ against $2.0000$, $3.3333$, $10.0000$ and $20.0000$, shares of $0.2000$, $0.5991$ and $0.9004$ against $0.2$, $0.6$ and $0.9$, cluster sizes of $1.250$, $2.494$ and $10.039$ against $1.250$, $2.500$ and $10.000$ — with the branching ratio playing the role utilization played for a queue; that the dispersion ratio runs from $1$ at windows below the kernel's memory to $1/(1-n)^{2}$ far above it, measured at $1.643$ against $43.667$ for $n=0.85$ where the limit is $44.44$, so the direction of its drift with window width identifies the mechanism where its level cannot; that the compensator transforms any point process into a unit Poisson process, giving a test with no moments in it; and that a Poisson fit recovers the rate to four significant figures while exceeding its own $99.9$th percentile $131.7$ times too often and understating the busiest of a hundred thousand windows by a factor of four, a misspecification the time change detects at a Kolmogorov–Smirnov statistic of $0.0005$ against $0.1602$.

The pairing with [Queue Models](05-queue-models.md) is closer than the shared subject matter suggests, and the symmetry is worth naming. Both pages are governed by a dimensionless feedback ratio, both blow up hyperbolically as it approaches one, and both have a first moment that stays perfectly well-behaved while the quantity anyone cares about diverges — a queue with mean service $1.000$ that waits $19.251$, a flow with rate $13.3831$ whose busiest window is four times its model's maximum. The difference is where the feedback lives: a queue's arrivals are exogenous and its congestion is a consequence, while a self-exciting process has no congestion at all and its arrivals cause each other. Two pages ago the clock was the answer and the level was fixed; here neither is, and what is fixed is a rate that turns out to be a state. Both microstructure pages have now taken the *information* content of an event for granted, treating every arrival as identical. The next page is where an arrival carries evidence and the question becomes what to believe after it.

**A Poisson model asks how often, a Hawkes model asks how often and how much of it was the market talking to itself, and the second question has an answer that the first one's estimate is entirely consistent with.**
