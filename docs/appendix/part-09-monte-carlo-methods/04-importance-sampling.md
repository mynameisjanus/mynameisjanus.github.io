# Importance Sampling

The crude Monte Carlo estimator allocates its effort in proportion to probability, and the questions worth asking are almost never about probable events. A risk desk wants the loss that happens once a decade, a pricing desk wants the payoff of an option that expires worthless nine times in ten, and a validation desk wants the behaviour of a strategy in the regimes it has barely seen — in every case the crude estimator spends the overwhelming majority of its draws confirming that nothing happened. Importance sampling fixes this by drawing from a distribution of your choosing and correcting for the substitution exactly, which is not an approximation and does not trade accuracy for speed. What it trades is a variance problem for a modelling problem, and the modelling problem has a failure mode that produces confident, wrong answers.

This page covers the change-of-measure identity and the proposal that would make the variance zero, the rare-event arithmetic that makes the crude estimator hopeless rather than merely slow, exponential tilting and the bounded relative error it buys, the support condition and what a light-tailed proposal does when it is violated, and the self-normalized ratio estimator that a tail statistic actually requires. It does not accept or discard draws to correct a proposal, which is [Rejection Sampling](05-rejection-sampling.md); it does not exploit a known control quantity or a shared seed, which is [Variance Reduction](06-variance-reduction.md); it assumes the target law rather than estimating it from data, which is [Bootstrap Methods](07-bootstrap-methods.md); it does not iterate the proposal toward the target, which is [Part XVII](../part-17-statistical-computing/index.md); it derives no tail distribution, which is [Part XVIII](../part-18-quant-finance-applications/index.md); and it certifies nothing about a strategy.

The trading stake is a diagnostic the course already prints without deriving. [Particle and Kalman Filters](../../advanced/02-particle-and-kalman-filters.md) builds a filter whose weights are exactly the ones on this page — it "maintains $N$ weighted particles … approximating $p(x_t \mid \mathcal F_t)$" — and reports that its two-thousand-particle run "held mean ESS 1,815 and resampled on 1% of days". That number is the health of an importance-sampling scheme, and the fourth section shows what it is and is not able to see: a proposal can report an effective sample size two orders of magnitude better than the crude estimator's and still be wrong by a factor of two.

## Reweighting Is an Identity, Not an Approximation

Let $\theta=\mathbb{E}_p[f(X)]=\int f(x)p(x)\,dx$ and let $q$ be any density with $q(x)>0$ wherever $f(x)p(x)\neq0$. Multiplying and dividing by $q$,

$$\theta=\int f(x)\frac{p(x)}{q(x)}q(x)\,dx=\mathbb{E}_q\!\left[f(X)w(X)\right],\qquad w(x)=\frac{p(x)}{q(x)}.$$

Draw $Y_1,\dots,Y_N$ from $q$ and average $f(Y_i)w(Y_i)$. The **weights** $w$ are the entire mechanism, and the estimator is unbiased for every admissible $q$ — good proposals and terrible ones alike. What separates them is variance.

??? note "Proof that the reweighted estimator is unbiased for any admissible proposal, that the variance-minimizing proposal is $q^{\ast}\propto|f|p$, and that using it would require already knowing the answer"
    Unbiasedness is the display above read as an expectation, and it needs only the support condition: if $q$ vanishes on a set where $fp$ does not, the integral over that set is silently dropped and the identity fails. Under $q$ the estimator has variance $\mathrm{var}_q(fw)/N$ with

    $$\mathrm{var}_q(fw)=\int\frac{f(x)^{2}p(x)^{2}}{q(x)}\,dx-\theta^{2}.$$

    Only the first term depends on $q$. By the Cauchy–Schwarz inequality,

    $$\left(\int|f|p\right)^{2}=\left(\int\frac{|f|p}{\sqrt q}\sqrt q\right)^{2}\leq\int\frac{f^{2}p^{2}}{q}\int q=\int\frac{f^{2}p^{2}}{q},$$

    with equality exactly when $|f|p/\sqrt q$ is proportional to $\sqrt q$, that is when $q^{\ast}(x)=|f(x)|p(x)/\int|f|p$. For $f\geq0$ this gives $\int f^{2}p^{2}/q^{\ast}=\theta^{2}$ and the variance is **zero** — every draw returns exactly $\theta$.

    The load-bearing observation is the normalizing constant. The optimal proposal is $|f|p$ divided by $\int|f|p$, which for non-negative $f$ *is* $\theta$, so constructing $q^{\ast}$ requires the quantity being estimated. The result is therefore not an algorithm but a design principle, and it is a sharp one: put proposal mass where $|f|p$ is large. For a rare-event probability $f$ is an indicator, $|f|p$ is the target restricted to the rare set, and the instruction reads *sample only where the event happens and correct for having done so* — which is exactly what the next two sections implement.

## Rare Events Are Where the Crude Estimator Fails Loudly

For $\theta=\mathbf{P}(X>t)$ the crude estimator is a sample proportion with variance $\theta(1-\theta)/N$, so its **relative** standard error is $\sqrt{(1-\theta)/(N\theta)}\approx1/\sqrt{N\theta}$. Achieving even ten percent relative accuracy therefore needs $N\approx100/\theta$ draws, which is a sample size proportional to the reciprocal of the answer. The crude estimator does not degrade gracefully as events get rarer; it stops working entirely, and it stops by returning zero.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(9041)
n = 100_000
print(f"  P(Z > t) from {n} draws, crude against exponential tilting to mean t")
print("        t         exact    crude hits    crude estimate    tilted estimate"
      "    relative se    speed-up")
for t in (2, 3, 4, 5, 6):
    exact = norm.sf(t)
    z = rng.standard_normal(n)
    hits = int((z > t).sum())
    crude = hits / n
    y = rng.standard_normal(n) + t                             # proposal is N(t, 1)
    w = np.exp(-t * y + 0.5 * t ** 2) * (y > t)                # phi(y) / phi(y - t)
    tilt, rse = w.mean(), w.std(ddof=1) / np.sqrt(n) / w.mean()
    crude_rse = np.sqrt((1 - exact) / (n * exact))             # what crude would need
    print(f"  {t:9d} {exact:13.3e} {hits:13d} {crude:17.3e} {tilt:18.6e}"
          f" {rse:15.4f} {(crude_rse / rse) ** 2:11.0f}")
# =>   P(Z > t) from 100000 draws, crude against exponential tilting to mean t
#            t         exact    crude hits    crude estimate    tilted estimate    relative se    speed-up
#              2     2.275e-02          2284         2.284e-02       2.264629e-02          0.0048          18
#              3     1.350e-03           136         1.360e-03       1.356983e-03          0.0058         219
#              4     3.167e-05             2         2.000e-05       3.204342e-05          0.0067        7051
#              5     2.867e-07             0         0.000e+00       2.868747e-07          0.0075      617364
#              6     9.866e-10             0         0.000e+00       9.765628e-10          0.0083   146458420
```

Read the `crude hits` column down. At $t=2$ a hundred thousand draws produce $2{,}284$ exceedances and a perfectly good estimate. At $t=4$ they produce **two**, and the resulting $2.0\times10^{-5}$ is $37\%$ below the truth with an error bar that would be computed from a sample of size two. At $t=5$ and $t=6$ they produce none, and the crude estimate is exactly zero — not an inaccurate answer but a wrong one, and one that carries a standard error of zero and no warning whatever.

The tilted column is the same hundred thousand draws taken from $\mathcal{N}(t,1)$ instead of $\mathcal{N}(0,1)$ and reweighted by $\varphi(y)/\varphi(y-t)=e^{-ty+t^{2}/2}$. It returns $2.868747\times10^{-7}$ against the exact $2.867\times10^{-7}$ at $t=5$, and $9.765628\times10^{-10}$ against $9.866\times10^{-10}$ at $t=6$ — three significant figures on a probability of one in a billion, from a hundred thousand draws.

The relative-error column is the surprise and it is the point. It reads $0.0048$, $0.0058$, $0.0067$, $0.0075$, $0.0083$: the tilted estimator gets *slightly* worse as the event gets a billion times rarer, and slightly means a factor of $1.73$ across four orders of magnitude in $\theta$. The speed-up column converts this into paths — the number of crude draws that would be needed to match the tilted precision, divided by the hundred thousand actually used. **A hundred thousand tilted draws do the work of fourteen and a half trillion crude ones at $t=6$**, and the whole difference is a change of one line in where the sampler is centred.

## The Variance-Optimal Proposal Requires the Answer, and Tilting Is What You Do Instead

The zero-variance proposal is unusable, but the design principle it states — put mass where $|f|p$ is large — has a standard implementation for the exponential family of light-tailed laws. **Exponential tilting** replaces $p$ by

$$q_\lambda(x)=\frac{e^{\lambda x}p(x)}{M(\lambda)},\qquad M(\lambda)=\mathbb{E}_p[e^{\lambda X}],$$

which for a normal target shifts the mean to $\mu+\lambda\sigma^{2}$ at unchanged variance, and for many other exponential-family targets stays inside the same family. The tilt $\lambda$ is chosen so the proposal's mean sits on the threshold, which is the crude approximation to "sample where the event happens".

??? note "Proof that tilting the mean onto the threshold makes the relative error grow linearly in the threshold rather than exponentially, and why the measured column grows like its square root"
    Take $p=\mathcal{N}(0,1)$, $f=\mathbf{1}\{x>t\}$, $\theta=\Phi(-t)$, and the tilted proposal $q=\mathcal{N}(t,1)$. The weight is $w(y)=\varphi(y)/\varphi(y-t)=e^{-ty+t^{2}/2}$, and on the event $\{y>t\}$ it is bounded above by its value at $y=t$:

    $$w(y)\leq e^{-t^{2}+t^{2}/2}=e^{-t^{2}/2}.$$

    Hence the second moment of the summand obeys $\mathbb{E}_q[w^{2}\mathbf{1}]\leq e^{-t^{2}/2}\,\mathbb{E}_q[w\mathbf{1}]=e^{-t^{2}/2}\theta$, so the relative variance satisfies

    $$\frac{\mathrm{var}_q(w\mathbf{1})}{\theta^{2}}\leq\frac{e^{-t^{2}/2}}{\theta}-1.$$

    Mills' ratio gives $\theta\sim\varphi(t)/t=e^{-t^{2}/2}/(t\sqrt{2\pi})$, so the bound is asymptotically $t\sqrt{2\pi}$ — the relative *variance* grows linearly in $t$, so the relative *standard error* grows like $\sqrt t$. The crude estimator's relative variance is $1/\theta\sim t\sqrt{2\pi}\,e^{t^{2}/2}$, which is the same quantity multiplied by an exponential. That factor of $e^{t^{2}/2}$ is the entire speed-up column: at $t=6$ it is $e^{18}=6.6\times10^{7}$.

    The load-bearing hypothesis is that $M(\lambda)$ exists, which is a statement that $p$ has an exponential moment — that is, a tail no heavier than exponential. Every polynomial-tailed law a risk model actually uses fails it, and for those the tilting construction does not exist at all. The bound also predicts the measured numbers: $\sqrt{6/2}=1.73$ is exactly the ratio of the observed relative errors at $t=6$ and $t=2$, so a theorem stated as an asymptotic inequality is doing quantitative work at $n=100{,}000$.

## A Proposal With a Lighter Tail Than the Target Is Confidently Wrong

The support condition in the first proof is not the whole story, because a proposal can have full support and still be useless. What matters is the weight $w=p/q$: if $q$ decays faster than $p$ anywhere that matters, $w$ is unbounded there, and $\mathbb{E}_q[f^{2}w^{2}]$ can be infinite while every finite sample looks unremarkable. This is the infinite-variance failure of [Monte Carlo Simulation](03-monte-carlo-simulation.md) arriving by a route the analyst chose.

```python
import numpy as np
from scipy.stats import norm
from scipy.stats import t as tdist

rng = np.random.default_rng(9043)
nu, thr, n, reps = 3.0, 25.0, 50_000, 400
exact = tdist.sf(thr, nu)


def run(kind):
    if kind == "crude":
        y = tdist.rvs(nu, size=n, random_state=rng)            # no reweighting at all
        w = np.where(y > thr, 1.0, 0.0)
    elif kind == "pareto":                                     # proposal matches the tail index
        y = thr * rng.random(n) ** (-1.0 / nu)
        w = tdist.pdf(y, nu) / (nu * thr ** nu * y ** (-nu - 1))
    else:                                                      # proposal tail dies too fast
        y = rng.standard_normal(n) + thr
        w = np.where(y > thr, tdist.pdf(y, nu) / norm.pdf(y - thr), 0.0)
    tot, sq = w.sum(), (w ** 2).sum()
    return (w.mean(), w.std(ddof=1) / np.sqrt(n),
            tot ** 2 / sq if sq > 0 else 0.0, w.max() / w.mean() if tot > 0 else 0.0)


print(f"  P(t({nu:.0f}) > {thr:.0f}) = {exact:.6e}, estimated {reps} times from {n} draws each")
print("   proposal                estimate    reported se    actual sd    ratio"
      "        ESS    max w / mean")
for kind, label in (("crude", "no reweighting"), ("pareto", "Pareto tail, index 3"),
                    ("light", "normal at 25, sd 1")):
    out = np.array([run(kind) for _ in range(reps)])
    est, se, ess, spread = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    print(f"  {label:<22} {est.mean():11.3e} {se.mean():14.3e} {est.std(ddof=1):12.3e}"
          f" {est.std(ddof=1) / se.mean():8.2f} {ess.mean():10.1f} {spread.mean():15.1f}")
# =>   P(t(3) > 25) = 7.016569e-05, estimated 400 times from 50000 draws each
#       proposal                estimate    reported se    actual sd    ratio        ESS    max w / mean
#      no reweighting           7.080e-05      3.581e-05    3.880e-05     1.08        3.5         18805.0
#      Pareto tail, index 3     7.017e-05      7.872e-10    7.893e-10     1.00    49999.7             1.0
#      normal at 25, sd 1       3.607e-05      1.050e-05    5.847e-05     5.57      292.0          5881.1
```

The target is a Student-$t$ with three degrees of freedom — a polynomial tail, so no exponential tilt exists and the previous section's construction is unavailable. The middle row is what replaces it: a Pareto proposal on $(25,\infty)$ whose index matches the target's, so the weight $p/q$ is bounded and nearly constant. The result is $7.017\times10^{-5}$ against the exact $7.016569\times10^{-5}$, a relative standard error of $10^{-5}$, an actual-to-reported ratio of $1.00$, and an effective sample size of $49{,}999.7$ out of $50{,}000$. When the proposal has the right tail, essentially every draw counts.

The bottom row is the failure. A normal proposal centred at the threshold looks reasonable, samples the right region, and produces $3.607\times10^{-5}$ — a little over **half** the correct answer — while reporting a standard error of $1.050\times10^{-5}$ that is the same order as the correct proposal's uncertainty. Its actual run-to-run spread is $5.847\times10^{-5}$, so the reported precision understates the truth by a factor of $5.57$, and a practitioner running it once would report a number that is wrong by a factor of two with a $30\%$ error bar.

The two diagnostic columns are where the honest reading gets uncomfortable. Effective sample size, the statistic the course's particle filter monitors, reads $292$ for the broken proposal and $3.5$ for the crude estimator — so **the diagnostic ranks the badly biased scheme eighty times healthier than the unbiased one**, because it measures how concentrated the weights are and not whether they are pointed at the right place. The largest-weight column does fire: a ratio of $5{,}881$ between the biggest weight and the average one says a handful of draws are carrying the estimate. But it fires on the crude estimator too, at $18{,}805$, and there the estimator is fine.

!!! warning "A proposal with a lighter tail than its target produces an estimator that is biased, confident, and indistinguishable from a converged one by any diagnostic computed from its own draws"
    Both numeric diagnostics above are computed from the sample, and the sample is drawn from the proposal, so neither can see the region the proposal fails to reach — which is precisely the region where the missing mass lives. The check that works is analytic and takes one line: compare the tails of $p$ and $q$ symbolically and confirm $\sup_x p(x)/q(x)<\infty$. A normal proposal for a Student-$t$ target fails it, a Student-$t$ proposal for a normal target passes it trivially, and the working rule follows — **when in doubt, propose from a distribution with a heavier tail than the target, because an over-dispersed proposal costs variance and an under-dispersed one costs correctness.** The same asymmetry governs the particle filter of [Particle and Kalman Filters](../../advanced/02-particle-and-kalman-filters.md): its bootstrap proposal is the state transition itself, which is heavier-tailed than the posterior it targets, and the mean ESS of $1{,}815$ out of $2{,}000$ it reports is meaningful only because that structural condition already holds.

## Self-Normalized Weights and the Tail of a Book

Two obstacles remain between the construction above and a usable risk number. The target density is often known only up to a constant — a Bayesian posterior, a conditional law given a rare event, an empirical distribution reweighted by a stress scenario. And the quantity wanted is usually not an expectation but a *ratio* of expectations: an expected shortfall is $\mathbb{E}[X\mathbf{1}\{X>q\}]/\mathbf{P}(X>q)$, with the same weights appearing in numerator and denominator. Both are solved by the same device. The **self-normalized** estimator

$$\hat\theta_{\text{SN}}=\frac{\sum_i w(Y_i)f(Y_i)}{\sum_i w(Y_i)}$$

divides out any unknown constant in $p$ and estimates a ratio directly. It is biased at order $1/N$ rather than unbiased, because it is a ratio of two random quantities, and consistent — a trade that is almost always worth making.

```python
import numpy as np
from scipy.stats import t as tdist

rng = np.random.default_rng(9047)
nu, sd, crude_n, is_n = 4.0, 1.216, 1_000_000, 20_000          # daily vol in percent, 19.3% a year
scale = sd / np.sqrt(nu / (nu - 2))
loss = tdist.rvs(nu, size=crude_n, random_state=rng) * scale   # one million draws, drawn once

print(f"  expected shortfall of a t({nu:.0f}) book, {crude_n} crude draws against {is_n} tilted")
print("      level    exact ES    crude ES    crude rel se    tilted ES    tilted rel se"
      "    crude paths for parity")
for level in (0.99, 0.999, 0.9999):
    tq = tdist.ppf(level, nu)
    q = tq * scale
    exact = scale * tdist.pdf(tq, nu) * (nu + tq ** 2) / ((nu - 1) * (1 - level))
    tail = loss[loss > q]
    c_es = tail.mean()
    c_rse = tail.std(ddof=1) / np.sqrt(tail.size) / c_es
    y = q * rng.random(is_n) ** (-1.0 / nu)                    # Pareto proposal above the level
    w = tdist.pdf(y / scale, nu) / scale / (nu * q ** nu * y ** (-nu - 1))
    i_es = (w * y).sum() / w.sum()                             # self-normalized ratio estimator
    d = (w * (y - i_es)) / w.sum()
    i_rse = np.sqrt((d ** 2).sum()) / i_es
    print(f"  {level:9.4f} {exact:11.3f} {c_es:11.3f} {c_rse:15.4f} {i_es:12.3f} {i_rse:16.6f}"
          f" {crude_n * (c_rse / i_rse) ** 2:23.3e}")
# =>   expected shortfall of a t(4) book, 1000000 crude draws against 20000 tilted
#          level    exact ES    crude ES    crude rel se    tilted ES    tilted rel se    crude paths for parity
#         0.9900       4.489       4.491          0.0038        4.497         0.003137               1.457e+06
#         0.9990       8.329       8.274          0.0108        8.305         0.002549               1.803e+07
#         0.9999      15.001      14.990          0.0330       15.001         0.002524               1.707e+08
```

The book is a $t$ with four degrees of freedom scaled to a daily volatility of $1.216\%$, which annualizes to the $19.3\%$ that [Hypothesis Testing and Multiple Testing](../../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) measures on the course's own series, and the expected shortfall is available in closed form so both estimators can be graded. Both are correct: crude reads $4.491$, $8.274$, $14.990$ and tilted reads $4.497$, $8.305$, $15.001$ against exact values of $4.489$, $8.329$, $15.001$.

The precision columns are the content. The crude estimator's relative standard error degrades down the table — $0.0038$, $0.0108$, $0.0330$ — because a million draws leave ten thousand exceedances at the $99\%$ level and only a hundred at $99.99\%$, so the deepest number in the risk report is computed from the smallest sample in it. The tilted estimator's relative error is flat at $0.0031$, $0.0025$, $0.0025$: **the same twenty thousand draws deliver the same precision at every level, because the proposal is re-centred at each one.**

The final column prices the difference in the only currency that matters. Matching the tilted precision at the $99.99\%$ level would take $1.707\times10^{8}$ crude draws. [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md) computes its Monte Carlo VaR from exactly one million bootstrap draws — a number chosen to look generous — and one million is two orders of magnitude short of what the crude route needs at the level a stress report actually quotes. Twenty thousand tilted draws cover it, and the ratio between the two is $8{,}500$.

!!! note "The self-normalized estimator's bias is the one place where more draws are genuinely the right answer, and the order of magnitude says when to stop worrying"
    A ratio of two unbiased estimators is not unbiased. Expanding $\hat A/\hat B$ about $(\mathbb{E}\hat A,\mathbb{E}\hat B)$ gives a leading bias term of order $\big(\theta\,\mathrm{var}(\hat B)-\mathrm{cov}(\hat A,\hat B)\big)/\mathbb{E}[\hat B]^{2}$, and each of those variances carries a $1/N$, so the bias is $O(1/N)$ while the standard error is $O(N^{-1/2})$. Their ratio is therefore $O(N^{-1/2})$: at $N=20{,}000$ the bias is smaller than the standard error by a factor of about a hundred and is invisible in the table above, and at $N=100$ it would not be. The practical consequences are two. Self-normalization is safe at any sample size a simulation would actually use, and it is *not* safe inside an inner loop that runs on a few dozen particles — which is why the resampling step of a particle filter, applied when the effective sample size falls, is a bias-control device as much as a variance-control one. The bias also does not shrink with a better proposal, only with more draws, so it is the one quantity on this page that a change of measure cannot help.

## The Proposal Is a Model, and It Is the Only One

Importance sampling is the one variance-reduction technique that changes what is computable rather than merely how fast. A tail probability of $10^{-9}$ is not slow under crude Monte Carlo; it is unavailable at any budget, and returns zero with a standard error of zero. Tilting or a matched-tail proposal makes it a three-significant-figure calculation on a hundred thousand draws, with a relative error that grows like the square root of the threshold rather than the exponential of its square.

What is bought with structure is paid for with a modelling assumption, and the assumption is unusually consequential because it is invisible. The proposal is a claim about where the answer lives, and when the claim is wrong in the specific direction of being too thin, the estimator does not become noisy — it becomes biased, with a small reported standard error and an effective sample size that can look better than the crude estimator's. That is the part's recurring failure in its sharpest form: the estimate and its error bar are computed from draws taken in a region the analyst selected, so neither can report on the region that was not selected.

The practical residue is three rules and one asymmetry. Match the proposal's tail to the target's or make it heavier, never lighter. Verify $\sup p/q<\infty$ analytically, because no statistic computed from the sample will do it for you. Read the effective sample size as a measure of weight concentration and not of correctness. And when a proposal is uncertain, prefer over-dispersion, since the cost of proposing too widely is a larger error bar that is honestly reported, and the cost of proposing too narrowly is a wrong answer that is not.

The alternative to reweighting a badly-placed draw is to throw it away, which sounds wasteful and turns out to be exact, self-checking, and free of the failure mode above. It also has a cost structure that is easy to compute in advance and easy to get catastrophically wrong in more than one dimension. That is [Rejection Sampling](05-rejection-sampling.md), the other thing to do with a proposal distribution.
