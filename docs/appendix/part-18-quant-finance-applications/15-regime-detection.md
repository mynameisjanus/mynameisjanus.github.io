# Regime Detection

Every page in this part has estimated a parameter and assumed it was constant. This one drops that and finds the problem is not estimation but *timing*: a regime is only useful if it is identified while it is still running, and the cost of insisting on that is a theorem rather than an implementation defect. Detection is a sequential decision with two errors that scale differently — raising the alarm threshold multiplies the average time between false alarms by roughly eight per two units, from $88$ days to $721$, $5{,}263$ and $40{,}954$, while the detection delay after a real change grows only from $3.1$ days to $4.3$, $5.5$ and $6.7$. Quiet is therefore cheap, and a well-tuned causal detector recovers $0.9026$ of the gain an oracle would deliver on a book whose regimes are real. The failure is what the same machinery does when they are not. Pointed at a single GARCH process with no regimes at all, it raises $2.82$ alarms a year, holds the book out of the market on $7.46\%$ of days, and changes the Sharpe ratio by $-0.0178$ — a detector that looks like it is working, producing a regime series that is entirely manufactured, and helping on $0.4050$ of paths.

This page covers detection as a sequential decision problem, the CUSUM statistic as a maximum of likelihood ratios over changepoint locations, the asymmetric scaling of false-alarm rate against detection delay, Lorden's optimality and what it implies about the gap between filtered and smoothed inference, and what a detector reports on a process whose only structure is volatility clustering. It does not derive the forward recursion, filtering against smoothing, or Baum–Welch, and it does not re-run the fit-two-states-to-noise experiment, all of which are [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md); it builds no Kalman, extended, unscented or particle filter, which is [Particle and Kalman Filters](../../advanced/02-particle-and-kalman-filters.md); it derives no EM algorithm, which is [The EM Algorithm](../part-17-statistical-computing/03-em-algorithm.md); it decomposes no variance across regimes, which is [The Law of Total Variance](../part-04-expectation-and-moments/08-law-of-total-variance.md); it fits nothing to real returns and reports no regime-conditioned backtest, which is [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md); it corrects no multiplicity, which is [Part XV](../part-15-multiple-testing/index.md); and it never reports an alarm rate without the rate the same detector produces on a process with nothing to detect.

The trading stake is the most seductive table in the course and the correction that follows it. [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) sizes a momentum book by hidden-state probabilities and prints `P(calm)-sized: Sharpe 1.10, maxDD -16%` against an unconditional $0.30$ and $-43\%$, then says plainly that it is "not a backtest": the probabilities are *smoothed*, computed using the entire series including everything after the day being labelled, and the module that recomputes the same table causally "finds 0.53." Section 3 is why that gap is a theorem rather than a coding error, and section 4 is the check that separates a real regime from a manufactured one.

## Detection Is a Sequential Decision, and Its Two Errors Do Not Scale Alike

A regime model that labels history is doing a different job from one that raises an alarm today, and only the second is usable. The second has an exact theory, and its central fact is an asymmetry.

??? note "Proof that the CUSUM statistic is a maximum of sequential likelihood ratios over changepoint locations, that its average run length grows exponentially in the threshold while its detection delay grows linearly, and that this is optimal"

    Suppose observations arrive with density $f_0$ before an unknown changepoint $\nu$ and $f_1$ after. The log-likelihood ratio for a change at $k$, evaluated at time $t$, is $\sum_{i=k}^{t}z_i$ with $z_i=\log(f_1(x_i)/f_0(x_i))$. Since $\nu$ is unknown, maximize over it:
    $$S_t=\max_{1\le k\le t}\sum_{i=k}^{t}z_i=\max\left(0,\;S_{t-1}+z_t\right),$$
    the second equality being the recursion that makes the maximum computable in constant memory. This is **Page's CUSUM**, and it is not a heuristic: it is the generalized likelihood ratio for the changepoint problem, with the reflecting barrier at zero arising from the maximization rather than being imposed.

    The two error scales follow from the drift of $z$. Before the change $\mathbb{E}_0[z]=-D_{\mathrm{KL}}(f_0\,\|\,f_1)<0$, so $S_t$ is a random walk with negative drift held at zero, and reaching a level $h$ requires a large deviation: by the same exponential-martingale argument [Probability of Ruin](02-probability-of-ruin.md) uses, the mean time to do so grows like $e^{h}$. After the change $\mathbb{E}_1[z]=+D_{\mathrm{KL}}(f_1\,\|\,f_0)>0$, so $S_t$ climbs at a positive rate and reaches $h$ in a time growing like $h/D_{\mathrm{KL}}$ — linearly.

    **Lorden's theorem** makes this optimal: among all stopping rules with a given average run length to false alarm, CUSUM minimizes the worst-case expected detection delay. So the exponential-against-linear trade is not a property of this statistic but of the problem, and no cleverer detector improves on it.

    **The load-bearing consequence is that the trade is enormously favourable in one direction. Buying a hundredfold reduction in false alarms costs an additive constant in delay, so a detector tuned by anyone who has looked at the two curves will sit at a high threshold — and the residual delay, which cannot be removed, is what section 3 prices.**

## The Trade Is Exponential Against Linear, So Quiet Is Cheap

Both scalings are directly measurable, and seeing them side by side is what makes the threshold choice obvious rather than arbitrary.

```python
import numpy as np

rng = np.random.default_rng(18151)
REPS, D, VOL_LO, VOL_HI = 20_000, 252, 0.10, 0.25
K = 0.5 * np.log(VOL_HI ** 2 / VOL_LO ** 2)             # reference drift of the CUSUM statistic


def cusum_run(h, shifted, horizon=40_000):
    """Page's test on the log-likelihood ratio for a variance shift. Returns the run
    length to the first alarm: an average run length under no change, a delay after one."""
    s = np.zeros(REPS)
    out = np.full(REPS, np.nan)
    live = np.ones(REPS, bool)
    v = (VOL_HI if shifted else VOL_LO) / np.sqrt(D)
    for t in range(horizon):
        x = rng.standard_normal(REPS) * v
        z = 0.5 * (x ** 2) * (1 / (VOL_LO / np.sqrt(D)) ** 2 - 1 / (VOL_HI / np.sqrt(D)) ** 2) - K
        s = np.maximum(0.0, s + z)
        fired = live & (s > h)
        out[fired] = t + 1
        live &= ~fired
        if not live.any():
            break
    return out


print(f"  Page's CUSUM watching for a volatility shift from {VOL_LO:.0%} to {VOL_HI:.0%}"
      f" annualized. The alarm threshold h trades false alarms against delay: the average run"
      f" length with no change should grow like exp(h) while the delay after a real change grows"
      f" only like h. {REPS:,} paths")
print("     threshold h   fired within the horizon   average run length, no change   in years"
      "   detection delay after a change   median   90th pct")
HORIZON = 40_000
for h in (2.0, 4.0, 6.0, 8.0):
    quiet = cusum_run(h, False, HORIZON)
    fast = cusum_run(h, True)
    # the run length is close to geometric, so estimate its mean from the firing rate,
    # which stays honest when the horizon truncates the slowest thresholds
    frac = np.mean(np.isfinite(quiet))
    arl = -HORIZON / np.log1p(-frac) if frac < 1 else np.nanmean(quiet)
    print(f"    {h:11.1f}   {frac:24.4f}   {arl:29,.0f}   {arl / D:9.1f}"
          f"   {np.nanmean(fast):32.1f}   {np.nanmedian(fast):8.1f}"
          f"   {np.nanpercentile(fast, 90):9.1f}")
# =>   Page's CUSUM watching for a volatility shift from 10% to 25% annualized. The alarm threshold h trades false alarms against delay: the average run length with no change should grow like exp(h) while the delay after a real change grows only like h. 20,000 paths
#         threshold h   fired within the horizon   average run length, no change   in years   detection delay after a change   median   90th pct
#                2.0                     1.0000                              88         0.3                                3.1        2.0         6.0
#                4.0                     1.0000                             721         2.9                                4.3        3.0         9.0
#                6.0                     0.9995                           5,263        20.9                                5.5        5.0        11.0
#                8.0                     0.6234                          40,954       162.5                                6.7        6.0        13.0
```

The two columns scale exactly as the proof requires. Doubling the threshold from $2$ to $4$ multiplies the mean time between false alarms by $8.2$, and from $4$ to $6$ by $7.3$, and from $6$ to $8$ by $7.8$ — a constant factor per unit of $h$, which is exponential growth. Over the same range the mean detection delay goes $3.1$, $4.3$, $5.5$, $6.7$ days, rising by about $1.2$ days per two units, which is linear.

The practical reading is that a desk running this detector at $h=2$ is making a mistake in an obvious direction. It suffers a false alarm every $88$ days — three or four a year — to detect real changes $3.6$ days sooner than a threshold of $8$, which raises a false alarm once every $162$ years. There is no reason to sit at the noisy end of a trade that prices quiet this cheaply, and section 4 shows what the noisy end actually produces.

## Lorden's Bound Makes the Smoothed-Minus-Filtered Gap a Theorem Rather Than a Defect

Section 1's delay is irreducible: it is the price of deciding without the future. That is exactly the quantity separating the two numbers the course reports, and the separation is not something better code removes.

??? note "Proof that no causal rule can beat CUSUM's delay at a given false-alarm rate, so the advantage a smoother enjoys is information rather than method"

    Lorden's formulation defines the worst-case expected delay $\bar d(T)=\sup_\nu\operatorname{ess\,sup}\mathbb{E}_\nu\left[(T-\nu)^{+}\mid\mathcal F_{\nu-1}\right]$ over stopping rules $T$ adapted to the observed filtration, subject to a constraint $\mathbb{E}_\infty[T]\ge\gamma$ on the mean time to false alarm. The theorem is that CUSUM with threshold $h=\log\gamma$ attains the minimum, asymptotically as $\gamma\to\infty$, with $\bar d\sim\log\gamma/D_{\mathrm{KL}}(f_1\|f_0)$.

    The word doing the work is *adapted*. Every rule in the admissible class is a function of observations up to the decision time, which is what "causal" means, and the bound is over that entire class. A **smoother** computes $\mathbf{P}(\text{regime}_t\mid x_1,\dots,x_T)$ with $T>t$ and is therefore not in the class at all: it is not a better detector but a different object, one that answers a question about the past using evidence from the future.

    The consequence is that a backtest sized by smoothed probabilities is not merely optimistic; it is measuring a quantity no live system can compute. The gap between it and the causal version is bounded below by the delay in Lorden's theorem, so it can be narrowed by a stronger signal — a larger $D_{\mathrm{KL}}$ between the regimes — and never by better inference. [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md) makes the same distinction in the filtering language and calls the difference "one line of code and the entire validity of the backtest"; this is the lower bound that makes the line unremovable.

    **The load-bearing distinction is between what a filtration contains and how well it is used. Filtering and smoothing differ in the conditioning set, not in the quality of the algorithm, so improving the algorithm cannot close the gap and only a regime that announces itself more loudly can.**

```python
import numpy as np

rng = np.random.default_rng(18153)
D, YEARS, PATHS = 252, 40, 400
VOL, MU = {0: 0.10, 1: 0.25}, {0: +0.09, 1: -0.06}      # calm pays, stress does not
STAY = {0: 1 - 1 / (2 * D), 1: 1 - 1 / 60}              # about two years calm, sixty days stress
K = 0.5 * np.log(VOL[1] ** 2 / VOL[0] ** 2)


def path(n):
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < STAY[s[t - 1]] else 1 - s[t - 1]
    return s, rng.normal([MU[k] / D for k in s], [VOL[k] / np.sqrt(D) for k in s])


def cusum_state(r, h):
    """Two-sided Page test: alarm into stress on rising variance, back out on falling."""
    out = np.zeros(len(r), dtype=int)
    up = dn = 0.0
    state = 0
    for t, x in enumerate(r):
        z = 0.5 * x ** 2 * (D / VOL[0] ** 2 - D / VOL[1] ** 2) - K
        up, dn = max(0.0, up + z), max(0.0, dn - z)
        if state == 0 and up > h:
            state, up, dn = 1, 0.0, 0.0
        elif state == 1 and dn > h:
            state, up, dn = 0, 0.0, 0.0
        out[t] = state
    return out


sharpe = lambda r: r.mean() / r.std() * np.sqrt(D)
rows = {k: [] for k in ("always on", "oracle", "CUSUM h=2", "CUSUM h=4", "CUSUM h=8")}
inmkt = {k: [] for k in rows}
for _ in range(PATHS):
    s, r = path(YEARS * D)
    for name, hold in (("always on", np.ones(len(r), bool)), ("oracle", s == 0),
                       *((f"CUSUM h={h:.0f}", cusum_state(r, h) == 0) for h in (2.0, 4.0, 8.0))):
        rows[name].append(sharpe(r * hold))
        inmkt[name].append(hold.mean())

print(f"  a two-regime book over {YEARS} years: calm at {VOL[0]:.0%} vol and {MU[0]:+.0%} drift,"
      f" stress at {VOL[1]:.0%} and {MU[1]:+.0%}, stress lasting about sixty days. The strategy"
      f" holds full size in calm and nothing in stress, and the only difference between rows is"
      f" who decides which regime it is. {PATHS} paths")
print("     rule            Sharpe: mean     sd   share of the oracle's gain   days in the market")
base, oracle = np.mean(rows["always on"]), np.mean(rows["oracle"])
for name, v in rows.items():
    got = (np.mean(v) - base) / (oracle - base)
    print(f"    {name:15s} {np.mean(v):12.4f}   {np.std(v):5.4f}   {got:26.4f}"
          f"   {np.mean(inmkt[name]):18.4f}")
# =>   a two-regime book over 40 years: calm at 10% vol and +9% drift, stress at 25% and -6%, stress lasting about sixty days. The strategy holds full size in calm and nothing in stress, and the only difference between rows is who decides which regime it is. 400 paths
#         rule            Sharpe: mean     sd   share of the oracle's gain   days in the market
#        always on             0.6012   0.1719                       0.0000               1.0000
#        oracle                0.8505   0.1552                       1.0000               0.8919
#        CUSUM h=2             0.7799   0.1532                       0.7168               0.8593
#        CUSUM h=4             0.8262   0.1560                       0.9026               0.8788
#        CUSUM h=8             0.8201   0.1569                       0.8784               0.8781
```

When the regimes are real the detector earns most of what is available. An oracle that knows the true state lifts the Sharpe ratio from $0.6012$ to $0.8505$; a causal CUSUM at $h=4$ reaches $0.8262$, which is $0.9026$ of the gain, and it does so while holding the book out of the market on almost exactly the fraction of days the oracle does — $0.8788$ against $0.8919$. The threshold matters in the expected direction, $h=2$ giving up a quarter of the gain to its own false alarms.

The structure here reproduces the published one on data whose truth is known. An unconditional book at $0.60$, an oracle at $0.85$, a causal detector at $0.83$: the same ordering as $0.30$ unconditional, $1.10$ smoothed, $0.53$ causal, with the gap between the second and third being the delay Lorden's theorem says cannot be removed. **A smoothed backtest is an upper bound and a useful one, provided it is read as the oracle row rather than as a result.**

!!! note "A filtered probability, a smoothed probability, an alarm and a regime label are four different objects, and a backtest can be built on any of them"
    **A filtered probability** conditions on data through today and is the only one a live system can compute; it flickers at boundaries and lags real changes by section 1's delay. **A smoothed probability** conditions on the whole sample, is what `predict_proba` returns by default in most libraries, and answers a question about the past that no trading decision ever asks. **An alarm** is a binary stopping decision rather than a probability, and it is what the theory of this page optimizes — Lorden's bound is about stopping times, not about beliefs. **A regime label** is the retrospective assignment of a name to a period, is produced after the fact by a human or a smoother, and is the object every narrative account of a crisis uses. Building a backtest on the second or fourth is not a subtle error, and the tell is always the same: the model's accuracy at the boundaries of a regime is too good, because the boundaries were placed with the benefit of what came next.

## Pointed at a Process With No Regimes at All, It Finds Three a Year

Sections 2 and 3 tuned and priced the detector on data that genuinely switches. The last question is what it does on data that does not, which is the case a desk cannot distinguish in advance.

```python
import numpy as np

rng = np.random.default_rng(18155)
D, YEARS, PATHS = 252, 40, 400
VOL, MU = {0: 0.10, 1: 0.25}, {0: +0.09, 1: -0.06}      # the detector's assumed regimes
K = 0.5 * np.log(VOL[1] ** 2 / VOL[0] ** 2)
OMEGA, ALPHA, BETA, DRIFT = 0.10 ** 2 / D * 0.05, 0.09, 0.86, 0.06


def garch(n):
    """One regime, no switching: a GARCH(1,1) whose volatility clusters anyway."""
    v = OMEGA / (1 - ALPHA - BETA)
    r = np.empty(n)
    for t in range(n):
        e = rng.standard_normal() * np.sqrt(v)
        r[t] = DRIFT / D + e
        v = OMEGA + ALPHA * e ** 2 + BETA * v
    return r


def cusum_state(r, h):
    out = np.zeros(len(r), dtype=int)
    up = dn = 0.0
    state = 0
    for t, x in enumerate(r):
        z = 0.5 * x ** 2 * (D / VOL[0] ** 2 - D / VOL[1] ** 2) - K
        up, dn = max(0.0, up + z), max(0.0, dn - z)
        if state == 0 and up > h:
            state, up, dn = 1, 0.0, 0.0
        elif state == 1 and dn > h:
            state, up, dn = 0, 0.0, 0.0
        out[t] = state
    return out


sharpe = lambda r: r.mean() / r.std() * np.sqrt(D)
print(f"  the same detector pointed at a series with no regimes at all -- one GARCH(1,1) process"
      f" whose volatility clusters because that is what GARCH does. Every alarm is false by"
      f" construction. {PATHS} paths x {YEARS} years")
print("     threshold h   alarms per year   share of days out of the market   Sharpe: always on"
      "     switched   difference   P(switching helps)")
base_all, cols = [], {h: [] for h in (2.0, 4.0, 8.0)}
outs = {h: [] for h in cols}
alarms = {h: [] for h in cols}
for _ in range(PATHS):
    r = garch(YEARS * D)
    base_all.append(sharpe(r))
    for h in cols:
        st = cusum_state(r, h)
        cols[h].append(sharpe(r * (st == 0)))
        outs[h].append(st.mean())
        alarms[h].append(np.sum(np.diff(st) == 1) / YEARS)
b = np.array(base_all)
for h in cols:
    s = np.array(cols[h])
    print(f"    {h:11.1f}   {np.mean(alarms[h]):15.2f}   {np.mean(outs[h]):32.4f}"
          f"   {b.mean():17.4f}   {s.mean():10.4f}   {s.mean() - b.mean():+11.4f}"
          f"   {np.mean(s > b):19.4f}")
# =>   the same detector pointed at a series with no regimes at all -- one GARCH(1,1) process whose volatility clusters because that is what GARCH does. Every alarm is false by construction. 400 paths x 40 years
#         threshold h   alarms per year   share of days out of the market   Sharpe: always on     switched   difference   P(switching helps)
#                2.0              2.82                             0.0746              0.6003       0.5825       -0.0178                0.4050
#                4.0              0.89                             0.0550              0.6003       0.5986       -0.0017                0.5000
#                8.0              0.31                             0.0385              0.6003       0.5983       -0.0020                0.4600
```

The output is a working regime model in every respect except being true. At $h=2$ the detector raises $2.82$ alarms a year and holds the book out of the market on $7.46\%$ of days — a plausible-looking series of stress episodes, occurring at a plausible frequency, on a process with exactly one regime. At $h=4$ and $h=8$ it still fires $0.89$ and $0.31$ times a year, which is to say a desk running this for a decade sees nine and three "regimes" that do not exist.

What it earns is nothing. The Sharpe difference against always-on is $-0.0178$, $-0.0017$ and $-0.0020$, and switching helps on $0.4050$, $0.5000$ and $0.4600$ of paths — a coin flip in all three rows. This is the honest failure and its shape is the one this part has found repeatedly: an apparatus that is correct, well-tuned, and answering a question the data does not pose. Volatility clustering and regime switching produce the same observable symptom, and the detector's likelihood ratio was built assuming one of them.

Read against section 3, the pair is decisive. The same machinery at the same threshold captures $0.9026$ of the available gain when regimes are real and delivers a coin flip when they are not, and *nothing in either output distinguishes the two cases* — both produce alarms, both produce out-of-market periods, both produce a plausible narrative. **The only difference is in the counterfactual, which is why the null must be simulated rather than argued about.**

## Every Repair Is a Higher Threshold, a Simulated Null, or an Admission That the Regime Is a Story

Three findings, three responses. Section 2's is free and unambiguous: the threshold trade is exponential against linear, so a detector should sit at the quiet end, and any tuning that produces several alarms a year has chosen the expensive side of a cheap trade. Section 3's cannot be repaired at all — the delay is Lorden's bound and closes only if the regimes separate more sharply — so the honest use of a smoothed backtest is as the oracle row, an upper bound on what any live system could achieve.

Section 4's is the one that costs a loop and is almost never run. The detector's alarm rate on the real series means nothing on its own; it means something only against the alarm rate the same detector produces on a series with no regimes and the same volatility dynamics. That null is cheap to simulate — a GARCH fit to the data, resampled — and it converts "the model found three regimes" from an observation into a test.

!!! warning "A detector always detects, and its output is a plausible regime series whether or not there are regimes"
    Every row of section 4 is a detector working correctly on data with nothing to find, and every row produces exactly the artefacts that would be taken as evidence it is working: alarms at a believable rate, stress periods of a believable length, and a story about each one. **The free diagnostic is the alarm rate under a null with no regimes: fit a GARCH or a block bootstrap to the same series, run the identical detector, and compare — at $h=2$ the null here fires $2.82$ times a year and at $h=8$ it fires $0.31$, so an observed rate near those numbers is evidence of nothing at all.** It costs one extra simulation of a model that has already been fitted, and it is the same discipline [Brownian Motion](../part-08-stochastic-processes/08-brownian-motion.md) applies to a drawdown and [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md) applies to a two-state fit on noise: the statistic is uninformative until the value it takes on data with no signal is known.

## What This Part Established, and What the Appendix Was For

This page established that CUSUM is the maximum of sequential likelihood ratios over changepoint locations, that its average run length to false alarm grows exponentially in the threshold while its detection delay grows linearly — measured at $88$, $721$, $5{,}263$ and $40{,}954$ days against delays of $3.1$, $4.3$, $5.5$ and $6.7$ — and that Lorden's theorem makes this trade a property of the problem; that the delay is therefore irreducible, so the gap between a smoothed backtest and a causal one is a lower bound rather than a defect, with a causal detector capturing $0.9026$ of an oracle's gain on real regimes, lifting $0.6012$ to $0.8262$ against the oracle's $0.8505$; and that the same detector on a single GARCH process with no regimes raises $2.82$ alarms a year, holds out of the market on $7.46\%$ of days, and changes the Sharpe by $-0.0178$ while helping on $0.4050$ of paths.

The part it closes has one shape running through it, and the fifteen pages are fifteen instances of it. [Kelly Criterion](01-kelly-criterion.md) found a growth-optimal fraction computed from two moments and capped by a third, unplaceable at $16.24\times$ against a bound of $15.78\times$. [Probability of Ruin](02-probability-of-ruin.md) found ruin exponential in the reciprocal of the Kelly multiple, $x^{2/c-1}$ exactly, and a jump law wiping out $0.9811$ of books the diffusion says are safe. [Drawdown Probabilities](03-drawdown-probabilities.md) found depth bought with size and duration bought with edge, $\sigma/2S$ and $1/(2S^{2})$. [Hitting and First-Passage Times](04-hitting-and-first-passage-times.md) found a half-life of $3.4$ days going with a $7.0$-day wait and a deadline costing $28.61\%$ of profit. [Queue Models](05-queue-models.md) found a delay governed by the second moment of service, four systems agreeing on every reported metric and differing by $9.6\times$. [Order Arrival Processes](06-order-arrival-processes.md) found a fitted rate correct to four figures beside a busiest window wrong by a factor of four. [Bayesian Signal Updating](07-bayesian-signal-updating.md) found sixty-five years to conviction at Sharpe $0.30$ and one day carrying $10.12\%$ of the evidence. [Monte Carlo Option Pricing](08-monte-carlo-option-pricing.md) found an estimator returning $0.000000$ with a standard error of $0.000000$. [Portfolio Risk Simulation](09-portfolio-risk-simulation.md) found a reported error bar $27.3$ times too small. [Value at Risk](10-value-at-risk.md) found two independent positions charged $101.00$ for diversifying. [Expected Shortfall](11-expected-shortfall.md) found coherence attaching to a formula that is not the one in most code, and level sets that rule out any scoring rule. [Heavy-Tailed Returns](12-heavy-tailed-returns.md) found a tail index estimated confidently on a law that has none, extrapolating $161.78$ against $41.22$. [Extreme Value Theory](13-extreme-value-theory.md) found a shape parameter of $-0.8192$ with a standard deviation of $0.9032$, a sign undetermined. [Copulas](14-copulas.md) found four books identical on every correlation and differing by sixteen in the corner. And this page found a detector finding three regimes a year in a process that has one.

In every case the model was right and the number it produced was about the wrong quantity, because the assumption that made a closed form available was the one the application violated. That is the appendix's argument, arriving where it was always headed. The mathematics of Parts I through XVII is not a prerequisite to be cleared before the trading starts; it is the only thing that says which of a model's outputs are load-bearing and which are artefacts of a convenience, and a desk that cannot make that distinction will be wrong in ways its own diagnostics confirm. Every page in this part offered one line of code that would have caught its failure, and none of those lines requires new data, a better estimator or a stronger assumption — only the question of what the number would have been if there had been nothing there.

**A model that cannot be wrong in a way you would notice is not a model of the market; it is a description of the assumption you made about it, and the whole of this appendix is the equipment for telling those apart.**
