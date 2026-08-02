# Rejection Sampling

There are exactly two things to do with a draw from the wrong distribution: keep it and correct its weight, or throw it away. The previous page did the first. This page does the second, and the trade between them is not the one intuition suggests. Discarding work sounds wasteful and reweighting sounds efficient, but the draws that survive rejection are *exact* — indistinguishable from draws taken directly from the target, usable for any functional whatever, with no weights to carry and no proposal-dependent bias to worry about. What rejection costs instead is a runtime that is random rather than fixed, and a cost constant that behaves innocently in one dimension and catastrophically in twenty.

This page covers the accept–reject algorithm and the proof that its output is exact, the envelope constant as the whole of the cost and the geometric runtime it implies, the exponential collapse of the acceptance rate with dimension, the squeezes and adaptive envelopes that make the method practical, and a direct comparison against reweighting at a fixed compute budget. It does not reweight draws to correct a proposal, which is [Importance Sampling](04-importance-sampling.md); it does not build a chain whose stationary law is the target, which is [Part XVII](../part-17-statistical-computing/index.md); it does not derive the transformations that make good envelopes available, which is [Sampling Methods](02-sampling-methods.md); it quotes no convergence rate for the resulting estimator, which is [Monte Carlo Simulation](03-monte-carlo-simulation.md); it resamples no data, which is [Bootstrap Methods](07-bootstrap-methods.md); and it fits nothing.

The trading stake is an omission the appendix flagged and postponed. [Continuous Uniform Distribution](../part-05-common-distributions/09-continuous-uniform-distribution.md) observes that the normal's "distribution function has no elementary inverse, which is why sampling it needs a different device", and the device it is pointing at is on this page — the ziggurat inside `default_rng.standard_normal` is a rejection scheme, and so is Marsaglia's polar method, and so is the alias table of [Sampling Methods](02-sampling-methods.md). The second section prices the same construction where it actually earns its keep: sampling the far tail of a normal, where the obvious approach needs three and a half million proposals per draw and the right one needs $1.0175$.

## Accept If the Uniform Lands Under the Ratio

Let $p$ be the target density and $q$ a proposal you can sample, and suppose there is a finite constant $M$ with

$$p(x)\leq M\,q(x)\quad\text{for every }x.$$

Draw $Y\sim q$ and $U\sim\mathrm{Unif}(0,1)$ independently. If $U\leq p(Y)/(M\,q(Y))$, return $Y$; otherwise discard both and start again. The picture is that $Mq$ is a curve lying above $p$ everywhere, a point is thrown uniformly into the region under $Mq$, and it is kept if it falls under $p$.

??? note "Proof that accepted draws are exactly $p$-distributed, that the acceptance probability is $1/M$, and that the number of proposals per draw is geometric"
    Write $A$ for the acceptance event. Conditioning on the proposal,

    $$\mathbf{P}(A)=\int \mathbf{P}\!\left(U\leq\frac{p(y)}{Mq(y)}\right)q(y)\,dy=\int\frac{p(y)}{Mq(y)}q(y)\,dy=\frac1M\int p(y)\,dy=\frac1M,$$

    which uses only that $p$ integrates to one and that $p/(Mq)\leq1$, the latter being the envelope condition. For the law of an accepted draw,

    $$\mathbf{P}(Y\leq x\mid A)=\frac{\mathbf{P}(Y\leq x,\,A)}{\mathbf{P}(A)}=\frac{\int_{-\infty}^{x}\frac{p(y)}{Mq(y)}q(y)\,dy}{1/M}=\int_{-\infty}^{x}p(y)\,dy,$$

    so the accepted draw has distribution function exactly $P$. There is no approximation and no asymptotics: one accepted draw is one exact draw. Since proposals are independent, the number required per acceptance is geometric with success probability $1/M$, so its mean is $M$ and its standard deviation is $\sqrt{M(M-1)}$ — a runtime whose spread is of the same order as its mean.

    The load-bearing hypothesis is that $M$ is finite, which is the requirement that $p/q$ be bounded. That is the *same* condition that made the importance-sampling weights well behaved on the previous page, and here it is not a quality criterion but an existence criterion: an unbounded $p/q$ does not give rejection sampling a large variance, it makes the algorithm undefined. **The two methods share a hypothesis, and where importance sampling degrades quietly, rejection refuses to start.**

## The Envelope Constant Is the Entire Cost, and a Good Envelope Improves With the Tail

Because $M$ is the expected number of proposals per draw, choosing an envelope is choosing a cost, and it is computable in advance from $M=\sup_x p(x)/q(x)$. The default envelope for a conditional law — sample the unconditional law and discard everything outside the conditioning set — has $M=1/\mathbf{P}(\text{set})$, which is fine for a common event and hopeless for a rare one. The repair is to pick a proposal shaped like the conditional target rather than like its parent.

```python
import numpy as np
from scipy.stats import kstest, norm

rng = np.random.default_rng(9051)
budget = 4_000_000
print(f"  sampling N(0,1) conditioned on X > t, {budget} proposals per row")
print("        t    naive M = 1/Phi(-t)    naive accepts    envelope rate    envelope M"
      "    KS p")
for t in (1, 2, 3, 5, 8):
    naive = rng.standard_normal(budget)
    a = 0.5 * (t + np.sqrt(t ** 2 + 4))                        # Robert's optimal exponential rate
    x = t - np.log(rng.random(budget)) / a                     # shifted exponential proposal
    keep = x[rng.random(budget) < np.exp(-0.5 * (x - a) ** 2)]
    rate = keep.size / budget
    cdf = 1 - norm.sf(keep[:200_000]) / norm.sf(t)             # exact truncated-normal cdf
    print(f"  {t:9d} {1 / norm.sf(t):22.4g} {int((naive > t).sum()):16d} {rate:16.4f}"
          f" {1 / rate:13.4f} {kstest(cdf, 'uniform').pvalue:7.4f}")
# =>   sampling N(0,1) conditioned on X > t, 4000000 proposals per row
#            t    naive M = 1/Phi(-t)    naive accepts    envelope rate    envelope M    KS p
#              1                  6.303           633762           0.8763        1.1412  0.9164
#              2                  43.96            90962           0.9335        1.0712  0.1368
#              3                  740.8             5575           0.9610        1.0406  0.4534
#              5              3.489e+06                1           0.9828        1.0175  0.2267
#              8              1.607e+15                0           0.9928        1.0073  0.4072
```

The naive columns collapse exactly as the previous page's crude estimator did, for the same reason. Discarding every draw below $t$ costs $1/\Phi(-t)$ proposals per acceptance: $6.3$ at $t=1$, $44$ at $t=2$, $741$ at $t=3$, and **three and a half million at $t=5$**. Four million proposals produce $633{,}762$ accepted draws at $t=1$, one at $t=5$, and none at $t=8$, where the required constant is $1.6\times10^{15}$.

The envelope columns are Robert's construction: propose from an exponential shifted to start at $t$, with rate $a=\tfrac12\big(t+\sqrt{t^{2}+4}\big)$ chosen to maximize the acceptance probability, and accept with probability $\exp\!\big(-\tfrac12(x-a)^{2}\big)$. The acceptance rate reads $0.8763$, $0.9335$, $0.9610$, $0.9828$, $0.9928$ — so the envelope constant is $1.1412$, $1.0712$, $1.0406$, $1.0175$, $1.0073$, and it *improves* as the tail gets deeper. The reason is that a normal tail becomes indistinguishable from an exponential tail as $t$ grows, so the proposal converges on the target's shape.

The last column is the check that this is sampling rather than approximating. Pushing the accepted draws through the exact truncated-normal distribution function should give uniforms, and the Kolmogorov–Smirnov $p$-values are $0.9164$, $0.1368$, $0.4534$, $0.2267$ and $0.4072$ on two hundred thousand draws apiece. **The two methods differ by a factor of $10^{15}$ in cost and by nothing at all in the law they sample**, which is the property that distinguishes rejection sampling from every other technique in this part: choosing a better envelope changes what you pay and cannot change what you get.

## In More Than a Few Dimensions the Constant Is Not a Constant

The one-dimensional intuition — pick a proposal roughly the right shape, pay a small constant — does not survive the addition of axes, and it fails for a reason that has nothing to do with the quality of the proposal.

??? note "Proof that a proposal too wide by a factor $c$ in each coordinate costs $M=c^{d}$, and that no envelope escapes an exponential in the dimension"
    Take the target $p=\mathcal{N}(0,I_d)$ and the proposal $q=\mathcal{N}(0,c^{2}I_d)$ with $c>1$ — a proposal that is correctly centred, correctly shaped, and merely $c$ times too wide per axis, which is the mildest possible misspecification. The ratio is

    $$\frac{p(x)}{q(x)}=c^{d}\exp\!\left(-\frac{\lVert x\rVert^{2}}{2}\left(1-\frac{1}{c^{2}}\right)\right),$$

    which is maximized at $x=0$ because the exponent is non-positive for $c>1$. Hence $M=c^{d}$ and the acceptance rate is $c^{-d}$, decaying geometrically in the dimension at rate $\log c$ per axis.

    The obstruction is not this particular proposal. For any $q$, $M=\sup p/q\geq\int p\,\cdot\,\sup(p/q)\big/\!\int p$, and more usefully $1/M=\mathbf{P}(A)$ is the fraction of the volume under $Mq$ that lies under $p$. In $d$ dimensions the mass of a product law concentrates on a thin shell of radius $\sqrt d$ and thickness $O(1)$, so any two product densities that are not identical put their shells in different places or at different radii, and the overlap between two shells decays exponentially in $d$. **Getting the shape approximately right stops being good enough once the volume of "approximately" grows faster than the volume of "right".**

    The load-bearing quantity is $\log c$ multiplied by $d$. At $c=1.2$ and $d=1$ the penalty is $20\%$; at $d=100$ it is $1.2^{100}=8.3\times10^{7}$. Nothing about the proposal changed. This is the exact reason [Part XVII](../part-17-statistical-computing/index.md) exists: a Markov chain gives up independence between successive draws in exchange for an acceptance rate that can be tuned to a constant in any dimension, and the trade is forced rather than chosen.

```python
import numpy as np

rng = np.random.default_rng(9053)
c, n = 1.2, 2_000_000                                          # proposal sd is 20% too wide
print(f"  accepting N(0,I_d) draws from an N(0,{c}^2 I_d) envelope, {n} proposals per row")
print("        d    envelope M = c^d    predicted rate    measured rate    proposals per draw")
for d in (1, 2, 5, 10, 20, 40, 100):
    y = rng.standard_normal((n, d)) * c
    r2 = (y ** 2).sum(axis=1)
    ratio = np.exp(-0.5 * r2 * (1 - 1 / c ** 2))               # p(y)/(M q(y)), M = c^d
    rate = float((rng.random(n) < ratio).mean())
    print(f"  {d:9d} {c ** d:19.4g} {c ** -d:17.3e} {rate:16.3e}"
          f" {(1 / rate if rate else float('inf')):22.4g}")
# =>   accepting N(0,I_d) draws from an N(0,1.2^2 I_d) envelope, 2000000 proposals per row
#            d    envelope M = c^d    predicted rate    measured rate    proposals per draw
#              1                 1.2         8.333e-01        8.337e-01                    1.2
#              2                1.44         6.944e-01        6.944e-01                   1.44
#              5               2.488         4.019e-01        4.024e-01                  2.485
#             10               6.192         1.615e-01        1.620e-01                  6.173
#             20               38.34         2.608e-02        2.608e-02                  38.35
#             40                1470         6.804e-04        6.755e-04                   1480
#            100           8.282e+07         1.207e-08        0.000e+00                    inf
```

The predicted and measured columns agree to three significant figures wherever a measurement is possible — $8.333$ against $8.337$, $1.615$ against $1.620$, $2.608$ against $2.608$, $6.804$ against $6.755$ — so the theorem is not being tested here, it is being used as a ruler. What the ruler measures is a proposal that is wrong by twenty percent per axis, which is well inside what anyone would call a good approximation.

Read the last column down. One dimension costs $1.2$ proposals per draw and is free. Ten dimensions cost $6.2$ and are cheap. Twenty cost $38$ and are annoying. Forty cost $1{,}480$ and are the difference between a simulation that finishes and one that does not. A hundred cost $8.3\times10^{7}$, and the measured rate is exactly zero because two million proposals produced no acceptances at all — the expected number was $0.024$.

**Nothing degraded except the number of axes.** The proposal at $d=100$ is the same twenty-percent-too-wide normal that was free at $d=1$, and a portfolio of a hundred assets is not an exotic object. The practical rule is blunt: rejection sampling is a one- to five-dimensional technique, it is usable up to ten or twenty when the envelope is genuinely tight, and past that the correct response is to stop looking for a better envelope and change methods.

!!! note "The ziggurat, the polar method and the alias table are all rejection sampling, and their acceptance rates are the reason they are defaults"
    [Sampling Methods](02-sampling-methods.md) measured Marsaglia's polar method at $1.273$ uniforms per normal, which is $M=4/\pi$: the proposal is uniform on the square and the target region is the inscribed disc, so the acceptance rate is the ratio of their areas. The ziggurat covers the normal density with a stack of equal-area rectangles, proposes uniformly within a randomly chosen rectangle, and needs a comparison only when the point falls in the sliver where the rectangle overhangs the density — with $256$ layers the acceptance rate exceeds $0.99$, which is why it is the library default. The alias table is the limiting case: the rejection is pre-computed away entirely, so $M=1$ and the runtime is deterministic. All three are one-dimensional problems with envelopes engineered to within a percent of the target, which is exactly the regime the previous section says the method owns, and exactly what a hundred-dimensional posterior does not offer.

## Squeezes, Adaptive Envelopes, and the Two Costs Nobody Counts

The expected number of proposals is not the expected cost, because a proposal is cheap and evaluating $p$ may not be. If $p$ involves a matrix solve, a numerical integral, or a likelihood over a long series, then the density evaluations dominate and $M$ is the wrong thing to minimize.

A **squeeze** is the standard repair: a cheap lower bound $\ell(x)\leq p(x)$ evaluated first, so that any proposal with $U\,Mq(Y)\leq\ell(Y)$ is accepted without touching $p$ at all. The expensive evaluation happens only in the gap between $\ell$ and $p$, and a good squeeze reduces the number of full evaluations by an order of magnitude while leaving the output law untouched — the accepted set is identical, so the correctness proof above is unaffected.

**Adaptive rejection sampling** goes further for log-concave densities, which include most of the exponential family. The envelope is built from tangent lines to $\log p$ at a handful of points and the squeeze from the chords between them; every rejected proposal becomes a new tangent point, so the envelope tightens as the algorithm runs and $M$ falls toward one. The construction is worth knowing because log-concavity is checkable and common, and because it turns the choice of envelope from a modelling decision into an automatic one.

The second uncounted cost is variance of runtime rather than its mean. The number of proposals per draw is geometric with mean $M$ and standard deviation $\sqrt{M(M-1)}$, so a scheme with $M=100$ will occasionally take a thousand proposals for a single draw. That is irrelevant when a million draws are wanted offline and disqualifying inside a latency budget, which is why production paths that need a bounded response time use table-driven samplers with $M=1$ and accept a larger memory footprint in exchange.

!!! warning "An acceptance rate you never measured is a runtime you cannot bound, and the geometric tail is longer than the mean suggests"
    Two failures follow from not printing the acceptance rate. The first is silent slowness: an envelope constant that drifted from $2$ to $200$ because a parameter changed shows up as a job that takes a hundred times longer, with no error and no message, and the usual diagnosis is that the machine is busy. The second is worse and is specific to conditional sampling. If the conditioning set has probability zero under the proposal — a constraint tightened past what the envelope can reach, a truncation moved beyond the proposal's support — the loop does not fail, it does not return, and there is nothing in the output to distinguish it from a slow run. Both are prevented by the same two lines: compute $M$ analytically before the loop and refuse to start if it exceeds a threshold, and count proposals and acceptances inside the loop and print the ratio beside the result. **The acceptance rate is the only number a rejection sampler produces that is not already implied by the target, and it is the one nobody reports.**

## Exact Draws Against Cheap Draws, at a Fixed Budget

The honest comparison against the previous page is not "which is more accurate" but "what does a fixed budget of random numbers buy". The two methods produce different objects, and the difference matters downstream.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(9057)
budget = 2_000_000


def row(t, name, p, p_se, e, e_se, exact):
    print(f"  {t:9d}  {name:<17} {p:>10} {p_se:>9} {e:15.4f} {e_se:10.5f} {exact:14d}")


print(f"  a fixed budget of {budget} base draws, spent three ways on the same tail")
print("        t    method              P(X > t)    rel se    E[X | X > t]     rel se"
      "    exact draws")
for t in (2, 4):
    p_ex = norm.sf(t)
    x = rng.standard_normal(budget)                            # naive rejection: keep x > t
    hit = x[x > t]
    row(t, "naive rejection", f"{hit.size / budget:.3e}",
        f"{np.sqrt(p_ex * (1 - p_ex) / budget) / p_ex:.4f}",
        hit.mean(), hit.std(ddof=1) / np.sqrt(hit.size) / hit.mean(), hit.size)

    a = 0.5 * (t + np.sqrt(t ** 2 + 4))                        # envelope rejection, no constant
    y = t - np.log(rng.random(budget)) / a
    keep = y[rng.random(budget) < np.exp(-0.5 * (y - a) ** 2)]
    row(t, "envelope", "-", "-", keep.mean(),
        keep.std(ddof=1) / np.sqrt(keep.size) / keep.mean(), keep.size)

    z = rng.standard_normal(budget) + t                        # tilted importance sampling
    w = np.where(z > t, np.exp(-t * z + 0.5 * t ** 2), 0.0)
    e_is = (w * z).sum() / w.sum()
    d = w * (z - e_is) / w.sum()
    row(t, "tilted weights", f"{w.mean():.3e}",
        f"{w.std(ddof=1) / np.sqrt(budget) / w.mean():.4f}", e_is,
        np.sqrt((d ** 2).sum()) / e_is, 0)
print(f"  exact: P = {norm.sf(2):.3e} and {norm.sf(4):.3e}, E[X|X>t] ="
      f" {norm.pdf(2) / norm.sf(2):.4f} and {norm.pdf(4) / norm.sf(4):.4f}")
# =>   a fixed budget of 2000000 base draws, spent three ways on the same tail
#            t    method              P(X > t)    rel se    E[X | X > t]     rel se    exact draws
#              2  naive rejection    2.282e-02    0.0046          2.3703    0.00067          45641
#              2  envelope                   -         -          2.3737    0.00010        1867113
#              2  tilted weights     2.272e-02    0.0011          2.3740    0.00014              0
#              4  naive rejection    3.700e-05    0.1256          4.2446    0.00718             74
#              4  envelope                   -         -          4.2255    0.00004        1949735
#              4  tilted weights     3.163e-05    0.0015          4.2262    0.00006              0
#      exact: P = 2.275e-02 and 3.167e-05, E[X|X>t] = 2.3732 and 4.2256
```

Naive rejection is the only method that answers both questions at once, because the acceptance rate *is* the tail probability. At $t=2$ that is a reasonable trade: $2.282\times10^{-2}$ with half a percent of relative error, a conditional mean of $2.3703$ against the exact $2.3732$, and $45{,}641$ exact draws in hand. At $t=4$ it falls apart on both counts — a tail probability of $3.700\times10^{-5}$ against the true $3.167\times10^{-5}$ with a $12.6\%$ relative error, a conditional mean off in the third digit, and $74$ exact draws from two million.

The envelope row abandons the normalizing constant deliberately: Robert's proposal samples the conditional law directly, so it never observes how rare the conditioning event was and cannot report $\mathbf{P}(X>t)$ at all. What it returns instead is $1{,}949{,}735$ exact, unweighted draws from the conditional distribution at $t=4$ — a rate of $97.5\%$ — and a conditional mean of $4.2255$ against the exact $4.2256$ with a relative error of $4\times10^{-5}$.

The tilted row returns the constant to $0.15\%$ and every conditional functional to about $10^{-4}$, from the same two million draws, and hands back **zero exact draws**. Everything it produces is weighted, and the weights must travel with the sample into every downstream calculation — a histogram, a quantile, a stress scenario fed to a portfolio optimizer all need weighted versions, and any code path that forgets the weights silently reports the proposal instead of the target.

**The choice is not accuracy against speed; it is a constant plus weights against a sample plus no constant.** When the deliverable is one number, importance sampling is better and usually by a lot. When the deliverable is a set of scenarios that will be reused for purposes not yet decided, exact draws are worth their acceptance rate, because an unweighted sample cannot be misused by code that does not know it is a sample from anywhere unusual.

## The Two Things to Do With a Proposal

Rejection sampling and importance sampling share one hypothesis and split on everything downstream of it. Both require $\sup_x p(x)/q(x)<\infty$; importance sampling degrades quietly when the bound fails, returning a biased number with a small error bar, while rejection sampling simply has no algorithm to run. Both are governed by how well the proposal matches the target; importance sampling converts a mismatch into variance, and rejection converts it into runtime. And both are one- to few-dimensional techniques for the same geometric reason, though only rejection makes the limit unmissable by printing an acceptance rate of zero.

Three numbers on this page are worth keeping. A naive envelope for a five-sigma normal tail costs $3.5\times10^{6}$ proposals per draw and a shaped one costs $1.0175$, so the cost of an envelope is a design decision spanning six orders of magnitude and the sampled law is identical either way. A proposal too wide by twenty percent per axis costs $1.2$ proposals in one dimension and $8.3\times10^{7}$ in a hundred, so the acceptance rate is a statement about geometry rather than about craftsmanship. And two million base draws bought either $1.9$ million exact conditional draws with no normalizing constant, or a normalizing constant to three digits with no exact draws — a genuine choice about what the output is for.

What both techniques have in common is that they change the *law* being sampled and correct for the change. The remaining route to a smaller error bar leaves the law alone and exploits structure in the estimator instead — a symmetry that lets one draw cancel another's noise, a related quantity whose answer is known in closed form, a comparison in which two runs can share their randomness. Those are cheaper than either method here, they compose with both, and their failure modes are entirely different: they do not produce wrong answers, they produce advertised savings that were never realized. That is [Variance Reduction](06-variance-reduction.md).
