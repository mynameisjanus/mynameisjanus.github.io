# Expected Shortfall

[Value at Risk](10-value-at-risk.md) ended on a defect that no better estimator repairs: two independent positions with individual VaRs of $-1.00$ each have a combined VaR of $99.00$, so the measure charges $101.00$ for diversifying. Averaging the tail rather than indexing into it fixes exactly that, and the fix is a theorem rather than an improvement — expected shortfall satisfies all four coherence axioms, and on the same construction it delivers $99.38$ against a sum of parts of $159.50$, a diversification benefit of $60.12$. Two things are bought with it. The first is estimation error: at the $97.5\%$ level Basel adopted, the ES estimator's standard deviation is $1.52$ times the VaR estimator's on $t(8)$ returns and $3.92$ times on $t(2.5)$, because it is an average over the region where the fewest observations are. The second is sharper and less known: ES is **not elicitable**. Its level sets are not convex — two distributions with an identical ES of $2.3378$ mix to one with $2.3646$ — so no scoring function exists whose expected value is minimized by the true ES, and the ranking of two competing ES forecasts is not a well-posed question. The measure that can be compared cannot be added, and the measure that can be added cannot be compared.

This page covers the four coherence axioms and the representation that proves them, the distinction between the coherent expected shortfall and the tail conditional expectation that is usually coded in its place, the integral representation that makes ES an average of VaRs at higher levels, the estimation cost that follows, and elicitability as a property of a functional's level sets. It does not derive the sample quantile's asymptotics, establish the pinball loss, or measure a breach-counting backtest, all of which are [Value at Risk](10-value-at-risk.md); it does not decompose a simulated risk number's error, which is [Portfolio Risk Simulation](09-portfolio-risk-simulation.md); it does not prove that a quantile fails to aggregate from margins, which is [Marginal Distributions](../part-03-random-variables/06-marginal-distributions.md); it does not catalogue which moments exist, which is [Higher-Order Moments](../part-04-expectation-and-moments/03-higher-order-moments.md); it fits no tail and extrapolates past no sample, which is [Extreme Value Theory](13-extreme-value-theory.md); it computes no ES on real returns and compares no regulatory regimes, which is [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md); and it never claims coherence for a formula without checking which of the two formulas it is.

The trading stake is a promise a course lesson makes by linking a single word to this page. [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md) writes that expected shortfall "looks past the cutoff, and it is [coherent] where VaR is not: the ES of a combined book can never exceed the sum of its parts', a guarantee VaR cannot make," and notes it is the measure Basel's FRTB regime adopted. Section 1 proves the guarantee. Section 2 then shows that the lesson's own phrase for it — "the average loss *conditional on* exceeding the threshold" — describes a different functional that does not have the guarantee, and that the two coincide exactly on the continuous return data the lesson computes with, which is why the distinction almost never surfaces and matters enormously when it does.

## Averaging the Tail Instead of Indexing Into It Buys All Four Axioms at Once

Coherence is four requirements, and three of them are easy for any sensible measure. The fourth is the one VaR fails, and the proof that ES has it is a single structural observation rather than four separate arguments.

??? note "Proof that expected shortfall is monotone, translation-equivariant, positively homogeneous and subadditive, via its representation as a supremum of expectations"

    Define, for a loss $X$ at level $p$,
    $$\mathrm{ES}_p(X)=\max\left\{\mathbb{E}[XZ]\;:\;0\le Z\le\tfrac{1}{1-p},\ \mathbb{E}[Z]=1\right\},$$
    the maximum expected loss under any reweighting of the probabilities that puts no more than $1/(1-p)$ times the original mass on any outcome. The maximizer puts full weight $1/(1-p)$ on the worst $1-p$ of outcomes and zero elsewhere, which recovers the familiar tail average and shows the two definitions agree.

    In that form all four axioms are immediate. **Monotonicity**: if $X\le Y$ pointwise then $\mathbb{E}[XZ]\le\mathbb{E}[YZ]$ for every admissible $Z$, so the maxima are ordered. **Translation equivariance**: $\mathbb{E}[(X+c)Z]=\mathbb{E}[XZ]+c$ since $\mathbb{E}[Z]=1$. **Positive homogeneity**: $\lambda>0$ scales every candidate value by $\lambda$. **Subadditivity**: for each admissible $Z$, $\mathbb{E}[(X+Y)Z]=\mathbb{E}[XZ]+\mathbb{E}[YZ]\le\mathrm{ES}_p(X)+\mathrm{ES}_p(Y)$, and taking the maximum over $Z$ on the left preserves the inequality.

    The last line is the whole content, and it is worth seeing why it works: a supremum of linear functionals is sublinear, always. The maximizing $Z$ for $X+Y$ need not be the maximizer for $X$ or for $Y$ separately, and that slack *is* the diversification benefit. VaR has no such representation — it is a quantile, not a maximum of expectations — which is why the same argument does not begin for it.

    **The load-bearing feature is that ES is defined by an optimization over scenarios rather than by a position in an ordering. That is what makes it additive-friendly, and it is also what makes section 4's problem inevitable: a functional defined as a supremum over a set that depends on the distribution does not, in general, have the convex level sets a scoring rule requires.**

## The Same Two Positions, and the Definition Usually Coded Is Not the Coherent One

The theorem is unconditional, so any counterexample must be a counterexample to something else. Running the previous page's construction through both formulas in circulation shows which.

```python
import numpy as np

rng = np.random.default_rng(18111)
P, DEFAULT, LOSS, REPS = 0.99, 0.008, 100.0, 400_000


def var_tce_es(losses, p=P):
    """VaR, the tail conditional expectation E[L | L >= VaR], and the coherent ES,
    which splits the probability atom sitting at the quantile."""
    v = np.quantile(losses, p)
    tce = losses[losses >= v].mean()
    beyond = losses[losses > v]
    es = (beyond.sum() / len(losses) + v * ((losses <= v).mean() - p)) / (1 - p)
    return v, tce, es


print(f"  the same two positions that broke VaR on the previous page: independent {DEFAULT:.1%}"
      f" default risk, paying +1 on survival and losing {LOSS:.0f} on default. Expected shortfall"
      f" averages the tail instead of indexing into it -- but only one of the two definitions in"
      f" common use is the coherent one. {REPS:,} draws")
print("     positions   VaR of sum   sum of VaRs   VaR ok?   E[L|L>=VaR] of sum   sum of them"
      "   ok?   coherent ES of sum   sum of them   ok?   ES benefit")
draws = [np.where(rng.random(REPS) < DEFAULT, LOSS, -1.0) for _ in range(5)]
v1, t1, e1 = var_tce_es(draws[0])
for k in (1, 2, 3, 5):
    v, t, e = var_tce_es(sum(draws[:k]))
    print(f"    {k:9d}   {v:10.2f}   {k * v1:12.2f}   {str(v <= k * v1):>7}"
          f"   {t:18.2f}   {k * t1:11.2f}   {str(t <= k * t1):>5}"
          f"   {e:18.2f}   {k * e1:12.2f}   {str(e <= k * e1):>5}   {k * e1 - e:10.2f}")

print("\n     on an elliptical book both measures behave, and the two ES definitions agree")
print("     correlation   VaR benefit   ES benefit   E[L|L>=VaR] of sum   coherent ES of sum"
      "   ES / VaR")
for rho in (-0.5, 0.0, 0.5, 0.99):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], REPS)
    vs, ts, es = var_tce_es(z.sum(axis=1))
    parts = [var_tce_es(z[:, i]) for i in (0, 1)]
    print(f"    {rho:11.2f}   {sum(p[0] for p in parts) - vs:11.4f}"
          f"   {sum(p[2] for p in parts) - es:10.4f}   {ts:18.4f}   {es:18.4f}   {es / vs:10.4f}")
# =>   the same two positions that broke VaR on the previous page: independent 0.8% default risk, paying +1 on survival and losing 100 on default. Expected shortfall averages the tail instead of indexing into it -- but only one of the two definitions in common use is the coherent one. 400,000 draws
#         positions   VaR of sum   sum of VaRs   VaR ok?   E[L|L>=VaR] of sum   sum of them   ok?   coherent ES of sum   sum of them   ok?   ES benefit
#                1        -1.00          -1.00      True                -0.19         -0.19    True                79.75          79.75    True         0.00
#                2        99.00          -2.00     False                99.24         -0.39   False                99.38         159.50    True        60.12
#                3        98.00          -3.00     False                98.77         -0.58   False                99.84         239.25    True       139.41
#                5        96.00          -5.00     False                97.70         -0.96   False               102.72         398.75    True       296.03
#
#         on an elliptical book both measures behave, and the two ES definitions agree
#         correlation   VaR benefit   ES benefit   E[L|L>=VaR] of sum   coherent ES of sum   ES / VaR
#              -0.50        2.3266       2.6745               2.6732               2.6732       1.1472
#               0.00        1.3675       1.5720               3.7551               3.7551       1.1441
#               0.50        0.6075       0.7063               4.6201               4.6201       1.1470
#               0.99        0.0129       0.0137               5.3112               5.3112       1.1424
```

Three columns, three different answers on the same positions. VaR fails as before. The **tail conditional expectation** $\mathbb{E}[L\mid L\ge\mathrm{VaR}]$ — the formula almost every implementation writes, and the one the phrase "average loss conditional on exceeding the threshold" describes — fails too, at $99.24$ against a sum of parts of $-0.39$. Only the coherent ES holds, at $99.38$ against $159.50$, a benefit of $60.12$ that grows to $139.41$ and $296.03$ as more independent positions are added.

The mechanism is an atom. With a $0.8\%$ default probability the loss distribution has $99.2\%$ of its mass at a single value, so the $99\%$ quantile sits *inside* that atom and the event $\{L\ge\mathrm{VaR}\}$ has probability $1.0$ rather than $0.01$: conditioning on it averages the whole distribution and returns $-0.19$. The coherent definition splits the atom, taking only the $0.01$ of probability that belongs above the quantile, which is what the supremum representation of section 1 does automatically.

The second panel is why this is nearly invisible. On continuous losses the quantile has no atom, the conditioning event has probability exactly $1-p$, and the two definitions agree to every digit printed — $2.6732$ against $2.6732$, $3.7551$ against $3.7551$, $4.6201$ and $5.3112$ likewise. **Every test on real return data agrees with every test on simulated Gaussian data that the two formulas are the same function, and they differ exactly on the discrete default-like books where VaR's failure created the demand for ES in the first place.**

## Coherence Costs a Factor of Four in Estimation Error, and the Factor Grows With the Tail

ES is an average of the worst outcomes, so it consumes strictly more of the region where data is scarcest. The cost is quantifiable in advance.

??? note "Proof that $\mathrm{ES}_p$ is the average of $\mathrm{VaR}_u$ over $u\in(p,1)$, so it inherits every quantile's estimation problem and weights the hardest ones most"

    For a continuous loss distribution, substituting $u=F(x)$ in the tail average gives
    $$\mathrm{ES}_p(X)=\frac{1}{1-p}\int_{p}^{1}\mathrm{VaR}_u(X)\,du,$$
    so ES at level $p$ is the *mean of the VaRs at every level above $p$*. This immediately explains three things. It is at least as large as $\mathrm{VaR}_p$, with equality only for a degenerate tail. It is a smooth functional of the distribution where a quantile is not, which is what makes it well-behaved under mixing in one direction and — as section 4 shows — badly behaved in another. And it is harder to estimate.

    The estimation cost follows from the levels being averaged. [Value at Risk](10-value-at-risk.md) established that the sample quantile at level $u$ has asymptotic standard deviation $\sqrt{u(1-u)/n}/f(q_u)$, which grows as $u\to1$ because the density collapses. ES averages exactly those, so its error is dominated by the highest levels — the ones with the fewest observations — rather than by the level named in its own subscript. The estimator has finite variance only when the underlying law has a finite second moment; for a tail index $\xi=1/\nu$ with $\nu\le2$ the ES estimator's variance is infinite, and for $\nu\le1$ the ES itself does not exist.

    **The load-bearing consequence is that the two measures are not on a common scale of difficulty. Moving from VaR to ES at the same confidence level is not a small refinement of the same estimate: it replaces a statistic that depends on $np$ observations with one that depends on their average, weighted toward the sparsest end, and the penalty grows with exactly the tail heaviness that motivated the change.**

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18113)
P, REPS = 0.975, 20_000

print(f"  the estimation cost of averaging the tail instead of indexing into it, at the"
      f" {P:.1%} level that Basel's FRTB adopted for expected shortfall. Student-t returns scaled"
      f" to unit variance; xi = 1/nu is the tail index, and the ES estimator has finite variance"
      f" only when nu > 2, with the variance itself infinite once nu <= 2. {REPS:,} replications")
print("     nu    1/nu    n      true VaR   true ES   ES / VaR   VaR: sd of estimate"
      "   ES: sd of estimate   ES sd / VaR sd")
for nu in (8.0, 5.0, 3.0, 2.5):
    sc = np.sqrt(nu / (nu - 2))
    tv = -stats.t.ppf(1 - P, nu) / sc
    # true ES of a unit-variance t: integrate the tail quantiles
    grid = np.linspace(1 - P, 1e-9, 400_000)
    te = -np.mean(stats.t.ppf(grid, nu)) / sc
    for n in (500, 2_000):
        x = rng.standard_t(nu, (REPS, n)) / sc
        v = -np.quantile(x, 1 - P, axis=1)
        e = np.array([-xr[xr <= -vi].mean() for xr, vi in zip(x, v)])
        print(f"    {nu:4.1f}   {1 / nu:5.3f}   {n:5d}   {tv:9.4f}   {te:7.4f}   {te / tv:8.4f}"
              f"   {v.std():19.4f}   {e.std():19.4f}   {e.std() / v.std():15.2f}")
# =>   the estimation cost of averaging the tail instead of indexing into it, at the 97.5% level that Basel's FRTB adopted for expected shortfall. Student-t returns scaled to unit variance; xi = 1/nu is the tail index, and the ES estimator has finite variance only when nu > 2, with the variance itself infinite once nu <= 2. 20,000 replications
#         nu    1/nu    n      true VaR   true ES   ES / VaR   VaR: sd of estimate   ES: sd of estimate   ES sd / VaR sd
#         8.0   0.125     500      1.9971    2.5720     1.2879                0.1519                0.2309              1.52
#         8.0   0.125    2000      1.9971    2.5720     1.2879                0.0771                0.1185              1.54
#         5.0   0.200     500      1.9912    2.7279     1.3700                0.1720                0.3105              1.81
#         5.0   0.200    2000      1.9912    2.7279     1.3700                0.0891                0.1632              1.83
#         3.0   0.333     500      1.8374    2.9107     1.5842                0.2035                0.5745              2.82
#         3.0   0.333    2000      1.8374    2.9107     1.5842                0.1041                0.2930              2.81
#         2.5   0.400     500      1.5986    2.7783     1.7379                0.2038                0.7713              3.79
#         2.5   0.400    2000      1.5986    2.7783     1.7379                0.1030                0.4041              3.92
```

The ES-to-VaR ratio rises with tail thickness — $1.2879$, $1.3700$, $1.5842$, $1.7379$ — which is the measure doing its job, reporting more risk exactly where more of it hides beyond the threshold. The cost is the last column. On $t(8)$ returns the ES estimator is $1.52$ times as noisy as the VaR estimator at the same level; on $t(5)$, $1.81$; on $t(3)$, $2.82$; on $t(2.5)$, $3.92$. Quadrupling the sample from $500$ to $2{,}000$ days halves both errors and leaves the ratio essentially unchanged at $1.54$, $1.83$, $2.81$ and $3.92$, so this is a structural penalty rather than a small-sample artefact.

Read together with section 2, the two findings are in tension in a way worth stating plainly. ES is the measure that behaves correctly on heavy-tailed and default-like books, and it is on precisely those books that its estimator is worst. **The measure is most necessary where it is least reliable, and the ratio in the final column is the exchange rate between a theoretical guarantee and a standard error.**

!!! note "Coherence, elicitability, comonotone additivity and backtestability are four things asked of a risk measure, and no measure has the first two together"
    **Coherence** is the four axioms of section 1 and is what makes a measure safe to aggregate across a book; VaR fails it, ES has it. **Elicitability** is the existence of a scoring function minimized by the true value, and is what makes two competing forecasts rankable; VaR has it, ES does not. **Comonotone additivity** — that perfectly dependent positions add exactly — is shared by both and is why neither reports a benefit for stacking the same trade twice, visible in the previous page's $\rho=0.99$ row. **Backtestability** is weaker than elicitability and is what regulators actually require: a test that a model is wrong, not a rule for ranking two models. ES is backtestable in that weak sense, jointly with VaR, which is the compromise Basel's FRTB adopts by requiring both numbers. The impossibility is not a gap in the literature but a theorem, so a desk choosing a risk measure is choosing which of the first two properties to have.

## A Functional Whose Level Sets Are Not Convex Has No Scoring Rule, and This One's Are Not

The property VaR has and ES lacks is easy to state and easy to check, and checking it is more convincing than citing it.

??? note "Proof that a functional with a strictly consistent scoring function has convex level sets, so non-convexity rules out elicitability"

    A scoring function $S(y,x)$ is **strictly consistent** for a functional $T$ if, for every distribution $F$, the expected score $\mathbb{E}_F[S(y,X)]$ is uniquely minimized at $y=T(F)$. Suppose such an $S$ exists and take two distributions $F,G$ with $T(F)=T(G)=t$. For any mixture $H=\lambda F+(1-\lambda)G$, expectation is linear in the distribution, so
    $$\mathbb{E}_H[S(y,X)]=\lambda\,\mathbb{E}_F[S(y,X)]+(1-\lambda)\,\mathbb{E}_G[S(y,X)].$$
    Each term on the right is minimized at $y=t$, so their positive combination is too, and by uniqueness $T(H)=t$. Hence the level set $\{F:T(F)=t\}$ is convex — this is **Osband's principle**, and its contrapositive is the usable form: a functional whose level sets are not convex admits no strictly consistent scoring function.

    The mean has convex level sets trivially, since it is linear in $F$. A quantile does too: if $F(v)=G(v)=p$ then $H(v)=p$ for every mixture, so $\mathrm{VaR}_p$ is constant on the mixture line — consistent with the pinball loss [Value at Risk](10-value-at-risk.md) exhibits. Expected shortfall does not, and the reason is visible in the same computation: two distributions can share an ES while having *different* VaRs, and the mixture's ES is an average over a tail whose starting point has moved.

    The repair is due to Fissler and Ziegel: the *pair* $(\mathrm{VaR}_p,\mathrm{ES}_p)$ is jointly elicitable, so a two-dimensional forecast can be scored even though neither the ES component alone can be. This is why a regime requiring expected shortfall requires the quantile alongside it, and why an ES number reported without its VaR cannot be compared with a competitor's.

    **The load-bearing distinction is between testing and ranking. Non-elicitability does not prevent asking whether an ES forecast is wrong; it prevents deciding which of two forecasts is better, because any answer depends on a scoring function whose choice changes the ordering.**

```python
import numpy as np
from scipy import optimize, stats

P = 0.975
Z, PHI = stats.norm.ppf(P), stats.norm.pdf(stats.norm.ppf(P))
K = PHI / (1 - P)                                       # ES of a standard normal, in sigmas


def mix_cdf(x, lam, a, b):
    return lam * stats.norm.cdf(x, *a) + (1 - lam) * stats.norm.cdf(x, *b)


def tail_mean(v, m, s):
    """E[X 1{X > v}] for X ~ N(m, s)."""
    u = (v - m) / s
    return m * stats.norm.sf(u) + s * stats.norm.pdf(u)


def var_es(lam, a, b):
    v = optimize.brentq(lambda x: mix_cdf(x, lam, a, b) - P, -50, 50)
    es = (lam * tail_mean(v, *a) + (1 - lam) * tail_mean(v, *b)) / (1 - P)
    return v, es


print(f"  a functional is elicitable only if its level sets are convex, so mixing two"
      f" distributions that agree on it must leave it unchanged. Below, two normals matched on one"
      f" functional are mixed and both are recomputed. All figures at the {P:.1%} level")
print("\n     matched on ES: N(0, 1) and a wider normal shifted to the same ES")
a = (0.0, 1.0)
s2 = 2.0
b = (K * (1 - s2), s2)                                  # same ES, different VaR
print(f"     component means and sds: {a} and ({b[0]:.4f}, {b[1]:.1f});"
      f" ES of each = {var_es(1.0, a, b)[1]:.4f} and {var_es(0.0, a, b)[1]:.4f}")
print("     weight on the first   VaR of the mixture   ES of the mixture   departure from the"
      " common ES")
base = var_es(1.0, a, b)[1]
for lam in (1.0, 0.75, 0.5, 0.25, 0.0):
    v, e = var_es(lam, a, b)
    print(f"    {lam:20.2f}   {v:18.4f}   {e:17.4f}   {e - base:+35.4f}")

print("\n     matched on VaR instead: same construction, solved so the VaRs agree")
c = (Z * (1 - s2), s2)                                  # same VaR, different ES
print(f"     component means and sds: {a} and ({c[0]:.4f}, {c[1]:.1f});"
      f" VaR of each = {var_es(1.0, a, c)[0]:.4f} and {var_es(0.0, a, c)[0]:.4f}")
print("     weight on the first   VaR of the mixture   ES of the mixture   departure from the"
      " common VaR")
base_v = var_es(1.0, a, c)[0]
for lam in (1.0, 0.75, 0.5, 0.25, 0.0):
    v, e = var_es(lam, a, c)
    print(f"    {lam:20.2f}   {v:18.4f}   {e:17.4f}   {v - base_v:+35.4f}")
# =>   a functional is elicitable only if its level sets are convex, so mixing two distributions that agree on it must leave it unchanged. Below, two normals matched on one functional are mixed and both are recomputed. All figures at the 97.5% level
#
#         matched on ES: N(0, 1) and a wider normal shifted to the same ES
#         component means and sds: (0.0, 1.0) and (-2.3378, 2.0); ES of each = 2.3378 and 2.3378
#         weight on the first   VaR of the mixture   ES of the mixture   departure from the common ES
#                        1.00               1.9600              2.3378                               +0.0000
#                        0.75               1.9149              2.3542                               +0.0164
#                        0.50               1.8531              2.3646                               +0.0268
#                        0.25               1.7594              2.3642                               +0.0264
#                        0.00               1.5821              2.3378                               -0.0000
#
#         matched on VaR instead: same construction, solved so the VaRs agree
#         component means and sds: (0.0, 1.0) and (-1.9600, 2.0); VaR of each = 1.9600 and 1.9600
#         weight on the first   VaR of the mixture   ES of the mixture   departure from the common VaR
#                        1.00               1.9600              2.3378                               +0.0000
#                        0.75               1.9600              2.4323                               +0.0000
#                        0.50               1.9600              2.5267                               -0.0000
#                        0.25               1.9600              2.6212                               +0.0000
#                        0.00               1.9600              2.7156                               -0.0000
```

The first panel is the counterexample and it is exact rather than statistical: two normals, $N(0,1)$ and $N(-2.3378,2)$, both with an expected shortfall of exactly $2.3378$. Mixing them in any proportion produces a distribution whose ES is *not* $2.3378$ — it rises to $2.3542$, $2.3646$ and $2.3642$ at weights of three-quarters, one-half and one-quarter. The level set is not convex, so by Osband's principle no strictly consistent scoring function for ES exists, and the mechanism is in the neighbouring column: the mixture's VaR slides from $1.9600$ to $1.5821$, moving the point from which the average is taken.

The second panel is the control and it is equally exact. Two normals matched on VaR instead of ES keep that VaR at $1.9600$ through every mixture, to the last printed digit, while their ES sweeps from $2.3378$ to $2.7156$. A quantile's level set is convex because a quantile is a statement about a single point of the distribution function, and mixing two functions that agree at a point produces a function that agrees at that point.

That is the whole asymmetry, and it has a practical edge. **A desk can hold a tournament between VaR models — score every day by the pinball loss, average, rank — and cannot hold one between ES models, because the ranking would depend on a scoring function that Osband's principle says cannot be made consistent.** What can be done is to score the pair jointly, which requires reporting the quantile that the coherent measure was supposed to replace.

## Every Repair Is the Pair, a Longer Sample, or an Honest Note About Which Formula Was Coded

The three findings have three different remedies and only one of them is free. Section 2's is: use the coherent definition. The atom-splitting formula is two lines rather than one, it agrees with the naive tail conditional expectation to every digit on continuous data, and it is the difference between having the theorem of section 1 and merely believing it. Any book containing default risk, digital payoffs, barrier options or discrete outcomes is one where the two diverge, and those are exactly the books whose aggregation the measure was chosen to handle.

Section 3's cost is irreducible at a fixed sample: ES is an average over the sparsest region and its standard error will exceed VaR's by the factor tabulated. The lever is the confidence level rather than the estimator, which is precisely what Basel's move to $97.5\%$ ES from $99\%$ VaR is — a level chosen so that the two produce comparable numbers on a Gaussian while the ES retains coherence. Section 4's is not a defect to repair but a property to work around, by reporting and scoring the pair.

!!! warning "Coherence is a property of a formula, and the formula in most code is the other one"
    The theorem in section 1 is unconditional and the guarantee it provides is real. It attaches to $\frac{1}{1-p}\int_p^1\mathrm{VaR}_u\,du$, and the expression typed in its place is almost always `losses[losses >= var].mean()`, which is a different functional that fails subadditivity at $99.24$ against $-0.39$ on the construction above. **The free diagnostic is $\mathbf{P}(L\ge\widehat{\mathrm{VaR}})$, computed on the same array: under the coherent definition it must equal $1-p$, and any material excess means an atom is sitting at the quantile and the naive average is reading the wrong conditioning event.** On the default book it reads $1.0$ against a nominal $0.01$ — a factor of a hundred, on one line, requiring no theory to notice. On continuous data it reads $1-p$ and the two formulas agree, which is why the check passes silently on every test case built from returns and fires on the first book with a digital payoff in it.

## A Measure That Adds, and Therefore Cannot Be Scored

This page established that expected shortfall is coherent, with subadditivity following from its representation as a maximum of expectations over reweighted probabilities — a supremum of linear functionals being sublinear; that the coherent definition and the tail conditional expectation $\mathbb{E}[L\mid L\ge\mathrm{VaR}]$ are different functionals that agree to every printed digit on continuous losses, at $2.6732$, $3.7551$, $4.6201$ and $5.3112$, and diverge on atomic ones, where the naive version fails subadditivity at $99.24$ against $-0.39$ while the coherent version delivers $99.38$ against $159.50$ and benefits of $60.12$, $139.41$ and $296.03$; that $\mathrm{ES}_p$ is the average of $\mathrm{VaR}_u$ over $u>p$ and therefore costs $1.52$, $1.81$, $2.82$ and $3.92$ times the VaR estimator's standard deviation as the tail index rises, a ratio unchanged by quadrupling the sample; and that ES has non-convex level sets — two laws with an ES of exactly $2.3378$ mixing to $2.3646$ — so by Osband's principle no strictly consistent scoring function exists, while the matched-VaR control holds its quantile at $1.9600$ through every mixture.

The pairing with the previous page is exact and unusually clean, because the two measures fail in complementary places and neither failure is statistical. VaR is elicitable and not coherent: it can be scored on every observation and cannot be added across a book. ES is coherent and not elicitable: it can be added and cannot be ranked. Both properties are theorems about the functional rather than facts about any estimator, so no amount of data moves either, and the practical consequence is the one Basel arrived at — report both, because each supplies what the other structurally cannot. What the three risk pages have shared is an assumption that the tail being measured is one the sample contains. Every estimator here has read the region between the threshold and the worst observation, and none has said anything about the region past it. The last four pages are about that region.

**A risk measure may be safe to add or possible to score, and the choice between those is made by the shape of its level sets rather than by anyone's preference.**
