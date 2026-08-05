# Bayesian Model Comparison

The Bayes factor is usually sold as the principled alternative to a p-value, and the sales pitch omits the part that decides every close call. Its virtue is genuine and mechanical: the marginal likelihood is a density over *datasets*, so it must integrate to one over everything that could have been observed, and a model spreading its bets across many possible outcomes automatically pays for the spread. Below, that integral checks out at $1.0000000000$ and the resulting Occam penalty is measured directly, growing to $0.9438$ nats. What the pitch omits is that the penalty's size is a function of the prior on the alternative, and the prior does not wash out with data — it gets *worse*. At a two-sided p-value pinned to exactly $0.05$, the same evidence yields a Bayes factor of $2.0790$ for the alternative, or $1.0172$, or $0.0341$ — odds of $29.30$ to one *for the null* — depending only on a prior width the data never sees. Even the most favourable alternative that could be constructed caps the evidence at $p=0.05$ to a Bayes factor of $6.83$, and the honest bound is $2.4560$, which at even prior odds is a posterior probability of $0.7107$ rather than anything like $0.95$. And the estimator most often used to compute the integral when it has no closed form does not converge: its error sits at $+0.5092$, $+0.1624$, $+0.1575$ and $+0.1763$ nats as the sample grows a thousandfold.

This page covers the marginal likelihood as a normalized predictive and the automatic complexity penalty that follows, the Bayes factor and posterior model odds, Lindley's paradox and the sensitivity of the comparison to a prior scale, the calibration of Bayes factors against p-values, the estimation of the evidence integral and the failure of the harmonic mean estimator, and Bayesian model averaging computed in numbers. It does not derive the Laplace approximation to the evidence or the $p\log n$ penalty, which is [Information Criteria](../part-14-model-selection/03-information-criteria.md); it does not establish the M-closed and M-open distinction or the general case for averaging, which is [Model Averaging](../part-14-model-selection/05-model-averaging.md); it derives no null distribution for a likelihood ratio, which is [Likelihood Ratio Tests](../part-12-hypothesis-testing/06-likelihood-ratio-tests.md); it constructs no prior and checks none, which is [Prior Distributions](02-prior-distributions.md); it proves no closure, which is [Conjugate Priors](04-conjugate-priors.md); it derives no importance-sampling estimator from first principles, which is [Importance Sampling](../part-09-monte-carlo-methods/04-importance-sampling.md); it builds no chain to sample a posterior, which is [Markov Chain Monte Carlo](../part-17-statistical-computing/04-markov-chain-monte-carlo.md); it splits no data and estimates no out-of-sample error, which is [Cross-Validation](../part-14-model-selection/02-cross-validation.md); it corrects nothing for the number of models examined, which is [Part XV](../part-15-multiple-testing/index.md); and it never reports a Bayes factor at one prior scale as though the scale had been implied by the data.

The trading stake is a course lesson using a marginal likelihood exactly as this page recommends and naming the mechanism precisely. [Bayesian Optimization](../../advanced/01-bayesian-optimization.md) selects a Gaussian process kernel by sweeping its length scale and reading the log marginal likelihood — `ell = 0.02: log marginal likelihood    0.47`, then `3.12`, `5.44`, `-1.66` and `-17.29` as the scale runs from $0.02$ to $4.00$ — describing the quadratic form and log-determinant as "a fit–complexity ledger" and the whole operation as "Occam's razor executed by linear algebra." That is section 1 with a kernel in place of a parameter. Section 3 is what happens to the same ledger when the models being compared differ in how *wide* their priors are rather than in how many parameters they have, which is the case a strategy comparison always presents.

## The Marginal Likelihood Is a Density Over Datasets, So Occam's Penalty Is Arithmetic Rather Than Policy

The quantity [Maximum A Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md) discards as an inconvenient constant, and which cancelled without comment in every update on the preceding five pages, is the only object in the framework capable of comparing one model with another.

??? note "Proof that $\int m(y)\,\mathrm{d}y=1$ for every model, so a model placing probability on more possible datasets must place less on each, and the complexity penalty is a consequence of normalization rather than an added term"

    For a model $M$ with prior $\pi_M$ and likelihood $f_M$, the marginal likelihood of an observed dataset $y$ is
    $$m_M(y)=\int f_M(y\mid\theta)\,\pi_M(\theta)\,\mathrm{d}\theta .$$
    Integrating over the sample space and exchanging the order,
    $$\int m_M(y)\,\mathrm{d}y=\int\pi_M(\theta)\Big(\int f_M(y\mid\theta)\,\mathrm{d}y\Big)\mathrm{d}\theta=\int\pi_M(\theta)\,\mathrm{d}\theta=1,$$
    since the inner integral is one for each $\theta$ and the prior is proper. So $m_M$ is itself a probability density over datasets — it is the prior predictive distribution — and it is *the* forecast the model made before seeing anything.

    The consequence is that models compete in a zero-sum allocation. A flexible model assigns non-negligible density to a wide range of possible datasets and, being constrained to total one, must assign less to any particular one; a rigid model concentrates its density on a narrow set and scores highly if the truth lands there and catastrophically if it does not. **No penalty term was added anywhere: the complexity charge is the normalization constraint, and it is exact rather than asymptotic, which is what distinguishes a Bayes factor from every criterion in [Information Criteria](../part-14-model-selection/03-information-criteria.md).**

    Comparison then follows from Bayes' rule applied one level up. With prior model probabilities $\pi(M_k)$,
    $$\frac{P(M_1\mid y)}{P(M_0\mid y)}=\underbrace{\frac{m_1(y)}{m_0(y)}}_{\text{Bayes factor }B_{10}}\times\frac{\pi(M_1)}{\pi(M_0)},$$
    so posterior odds are prior odds times the Bayes factor, exactly as in the odds form of [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md), with $B_{10}$ carrying everything the data contributed.

    For a normal mean with $M_0:\mu=0$ against $M_1:\mu\sim N(0,\tau^{2})$, writing $v=\sigma^{2}/n$ for the sampling variance of $\bar y$ and $g=\tau^{2}/v$, both marginals are available in closed form and
    $$\log B_{10}=-\tfrac{1}{2}\log(1+g)+\tfrac{1}{2}z^{2}\frac{g}{1+g},\qquad z=\frac{\bar y}{\sqrt v}.$$
    The load-bearing feature is the first term: it is a pure penalty depending on $g=n\tau^{2}/\sigma^{2}$ and not on the data at all, and it grows without bound in both $n$ and $\tau$. The second term is capped at $z^{2}/2$ however strong the evidence. Section 3 is the collision between those two facts.

## The Penalty Is Exact, Grows With the Sample, and Is Not What BIC Estimates It to Be

Both the normalization identity and the penalty it forces are directly checkable:

```python
import numpy as np
from scipy import stats, integrate

rng = np.random.default_rng(16061)
sig, tau, reps = 0.010, 0.0004, 200_000

# M0 says the edge is exactly zero; M1 puts a N(0, tau^2) prior on it. Both are
# conjugate, so the marginal likelihood of the sample mean is available exactly.
n0 = 250
v0 = sig ** 2 / n0
tot, _ = integrate.quad(lambda t: stats.norm.pdf(t, 0.0, np.sqrt(v0 + tau ** 2)),
                        -0.2, 0.2)
print(f"  the marginal likelihood is a density over datasets, so it integrates to one:"
      f" M1 over all possible sample means gives {tot:.10f}")
print("        n        g   Occam penalty, nats   exact log BF10 at z=2   BIC estimate"
      "   gap   crossover |z|")
for n in (25, 100, 250, 1000, 10000):
    v = sig ** 2 / n
    g = tau ** 2 / v
    zb = rng.standard_normal(reps)                     # under M0 the truth is zero
    lbf = -0.5 * np.log1p(g) + 0.5 * zb ** 2 * g / (1 + g)
    ex2 = -0.5 * np.log1p(g) + 0.5 * 4.0 * g / (1 + g)
    bic = 0.5 * 4.0 - 0.5 * np.log(n)
    zc = np.sqrt(np.log1p(g) * (1 + g) / g)
    print(f"    {n:5d}   {g:6.3f}   {-lbf.mean():19.4f}   {ex2:21.4f}   {bic:12.4f}"
          f"   {ex2 - bic:5.4f}   {zc:13.4f}")
# =>   the marginal likelihood is a density over datasets, so it integrates to one: M1 over all possible sample means gives 1.0000000000
#            n        g   Occam penalty, nats   exact log BF10 at z=2   BIC estimate   gap   crossover |z|
#           25    0.040                0.0004                  0.0573         0.3906   -0.3332          1.0098
#          100    0.160                0.0049                  0.2017        -0.3026   0.5042          1.0373
#          250    0.400                0.0255                  0.4032        -0.7607   1.1639          1.0852
#         1000    1.600                0.1680                  0.7530        -1.4539   2.2069          1.2461
#        10000   16.000                0.9438                  0.4657        -2.6052   3.0709          1.7350
```

The first line is the identity. Integrating $M_1$'s marginal likelihood over every sample mean it could have produced returns $1.0000000000$, confirming that the object being maximized in a model comparison is a forecast that was fully committed before the data arrived. This is the sense in which a Bayes factor is a genuine out-of-sample quantity computed without splitting the sample — the comparison [Cross-Validation](../part-14-model-selection/02-cross-validation.md) makes by holding data back, made instead by requiring the model to have distributed its probability in advance.

The `Occam penalty` column is the average log evidence *against* the richer model when the simpler one is true, and it grows $0.0004$, $0.0049$, $0.0255$, $0.1680$ and $0.9438$ nats as the sample runs from twenty-five days to ten thousand. Nothing was added to produce it. The richer model is being charged for the range of sample means it was willing to entertain, and the charge rises with $n$ because a larger sample makes the data more informative and the unused breadth more wasteful.

The `BIC estimate` column is the reason this page exists separately from [Information Criteria](../part-14-model-selection/03-information-criteria.md). BIC approximates $\log B_{10}$ by $\tfrac{1}{2}z^{2}-\tfrac{1}{2}\log n$, dropping an $O(1)$ term, and the size of that dropped term is visible: the gap between the exact log Bayes factor at $z=2$ and BIC's estimate runs $-0.3332$, $0.5042$, $1.1639$, $2.2069$ and $3.0709$ nats. A gap of three nats is a factor of twenty in the odds. **BIC is consistent, which means the gap does not grow relative to the terms that diverge, and it is not accurate, which means a BIC difference should never be exponentiated and quoted as an odds ratio at any finite sample size.**

The last column is where the trouble starts. The threshold $|z|$ at which the evidence exactly balances is not fixed at $1.96$ or anywhere near it — it runs $1.0098$, $1.0373$, $1.0852$, $1.2461$ and $1.7350$, rising with the sample. A Bayes factor and a fixed-level test are not two roads to one verdict, and the divergence grows with the evidence available.

## At a Fixed P-Value the Bayes Factor Can Be Driven to Either Verdict by a Prior the Data Never Sees

The two terms in section 1's formula move in opposite directions as the prior widens, and the resulting behaviour is the sharpest known separation between the two schools.

??? note "Proof that at fixed $z$ the Bayes factor tends to zero as the prior scale or the sample size grows, so the same p-value can be made to support the null arbitrarily strongly, which is Lindley's paradox"

    Hold $z=\bar y/\sqrt v$ fixed — that is, hold the p-value fixed — and let $g=n\tau^{2}/\sigma^{2}$ vary. From section 1,
    $$\log B_{10}(g)=-\tfrac12\log(1+g)+\tfrac12 z^{2}\frac{g}{1+g}.$$
    As $g\to\infty$ the second term rises to its ceiling $z^{2}/2$ while the first falls without bound, so $\log B_{10}\to-\infty$: the comparison eventually favours the null with unbounded strength. Since $g$ is proportional to both $n$ and $\tau^{2}$, this happens either by widening the prior on the alternative or by collecting more data while keeping the p-value at its threshold.

    Differentiating, $\partial_g\log B_{10}=\tfrac{1}{2(1+g)^{2}}(z^{2}-1-g)$, so the Bayes factor is maximized at $g^{*}=z^{2}-1$ and the maximum attainable value over all prior scales is
    $$\max_g B_{10}=\frac{e^{(z^{2}-1)/2}}{|z|}\qquad (|z|>1),$$
    which at $z=1.96$ is about $2.46$. A yet more favourable alternative — a point mass exactly at the observed $\bar y$, which no honest analyst may choose in advance — gives $e^{z^{2}/2}$, about $6.83$ at the same $z$. Sellke, Bayarri and Berger sharpen this into the calibration $B_{10}\le 1/(-e\,p\log p)$ for a wide class of alternatives.

    The frequentist reading of the same algebra is that a fixed-level test has power tending to one against any fixed alternative, so at large $n$ a rejection at $p=0.05$ is produced by effects too small to matter, and the Bayes factor is registering that the observed $\bar y$ is closer to the null than to anything the alternative predicted. **The two procedures do not disagree about the data; they disagree about whether an alternative that has to specify where the effect might be should be charged for how vaguely it specified it.** The load-bearing consequence is that a Bayes factor is not a prior-free summary of evidence and cannot be made into one, since the limit that removes the prior's influence is exactly the limit in which the answer becomes degenerate.

Every quantity in that proof is a number:

```python
import numpy as np
from scipy import stats

z = stats.norm.isf(0.025)
print(f"  the two-sided p-value is pinned at exactly 0.05 in every row below, so z ="
      f" {z:.4f} throughout and the data never changes its verdict")
print("        n   prior sd tau, bp   g = n tau^2/sigma^2   BF10     BF01   verdict")
sig = 0.010
for n in (100, 1000, 10000):
    for tau_bp in (2.0, 20.0, 200.0):
        tau = tau_bp * 1e-4
        g = n * tau ** 2 / sig ** 2
        lbf = -0.5 * np.log1p(g) + 0.5 * z ** 2 * g / (1 + g)
        bf = np.exp(lbf)
        v = ("for the alternative" if bf > 3 else
             "for the null" if bf < 1 / 3 else "worth nothing")
        print(f"    {n:5d}   {tau_bp:15.1f}   {g:19.4f}   {bf:6.4f}   {1 / bf:6.2f}"
              f"   {v}")

print("  the most favourable alternative that exists, and the ceiling it puts on a p-value")
print("      p-value   z    max BF10 over all priors   Sellke bound -e p ln p"
      "   posterior P(H1) at even odds")
for p in (0.05, 0.01, 0.005, 0.001):
    zz = stats.norm.isf(p / 2)
    mx = np.exp(0.5 * zz ** 2)                         # attained by a point mass at z
    sb = 1 / (-np.e * p * np.log(p))
    print(f"    {p:9.3f}   {zz:5.3f}   {mx:22.2f}   {sb:22.4f}"
          f"   {sb / (1 + sb):28.4f}")

print("  and the p-value at which the Bayes factor is exactly one, which is not a constant")
print("      n = 100    n = 1000   n = 10000   n = 100000")
row = []
for n in (100, 1000, 10000, 100000):
    g = n * (20.0 * 1e-4) ** 2 / sig ** 2
    zc = np.sqrt(np.log1p(g) * (1 + g) / g)
    row.append(2 * stats.norm.sf(zc))
print("    " + "   ".join(f"{r:9.5f}" for r in row))
# =>   the two-sided p-value is pinned at exactly 0.05 in every row below, so z = 1.9600 throughout and the data never changes its verdict
#            n   prior sd tau, bp   g = n tau^2/sigma^2   BF10     BF01   verdict
#          100               2.0                0.0400   1.0558     0.95   worth nothing
#          100              20.0                4.0000   2.0790     0.48   worth nothing
#          100             200.0              400.0000   0.3392     2.95   worth nothing
#         1000               2.0                0.4000   1.4631     0.68   worth nothing
#         1000              20.0               40.0000   1.0172     0.98   worth nothing
#         1000             200.0             4000.0000   0.1079     9.27   for the null
#        10000               2.0                4.0000   2.0790     0.48   worth nothing
#        10000              20.0              400.0000   0.3392     2.95   worth nothing
#        10000             200.0            40000.0000   0.0341    29.30   for the null
#      the most favourable alternative that exists, and the ceiling it puts on a p-value
#          p-value   z    max BF10 over all priors   Sellke bound -e p ln p   posterior P(H1) at even odds
#            0.050   1.960                     6.83                   2.4560                         0.7107
#            0.010   2.576                    27.59                   7.9884                         0.8887
#            0.005   2.807                    51.40                  13.8867                         0.9328
#            0.001   3.291                   224.48                  53.2560                         0.9816
#      and the p-value at which the Bayes factor is exactly one, which is not a constant
#          n = 100    n = 1000   n = 10000   n = 100000
#          0.15608     0.05106     0.01423     0.00397
```

The first table holds the p-value at exactly $0.05$ in all nine rows. The data is identically significant throughout — the same $z=1.9600$, the same tail probability, the same verdict from every frequentist test ever devised. The Bayes factor reads $1.0558$, $2.0790$, $0.3392$, $1.4631$, $1.0172$, $0.1079$, $2.0790$, $0.3392$ and $0.0341$. At ten thousand observations with a two-hundred-basis-point prior it is $29.30$ to one *for the null* on data a frequentist calls significant. Nothing in that column is a property of the returns; it is a property of $g=n\tau^{2}/\sigma^{2}$, and $\tau$ was chosen by the analyst.

The second table is the calibration, and it is the most useful thing on this page for anyone who has to read other people's results. The largest Bayes factor obtainable at $p=0.05$ by *any* alternative — including the inadmissible one that places all its prior mass exactly where the data landed — is $6.83$. The Sellke bound, which is attainable by a legitimate alternative, gives $2.4560$; starting from even prior odds that is a posterior probability of $0.7107$ for the alternative. At $p=0.01$ the bound is $7.9884$ and the posterior probability $0.8887$; at $p=0.001$, $53.2560$ and $0.9816$. **A result significant at five per cent, read as generously as the mathematics permits, supports the alternative at roughly seven-to-three — not nineteen-to-one — and the gap between those two readings is the single largest source of misplaced confidence in applied statistics.**

The third line finishes the argument about thresholds. The p-value at which the Bayes factor is exactly one, for a fixed prior width, runs $0.15608$, $0.05106$, $0.01423$ and $0.00397$ as $n$ goes from a hundred to a hundred thousand. There is no p-value that means the same thing at every sample size, and the coincidence at $n=1000$ — where the balance point is $0.05106$, almost exactly the conventional threshold — is a coincidence about this $\tau$ and this $\sigma$.

## The Evidence Is an Integral, and Its Most-Used Estimator Does Not Converge

Everything above assumed the marginal likelihood was available. Outside conjugate families it is an integral over the whole parameter space — the same object [Posterior Distributions](03-posterior-distributions.md) found expensive, now required rather than cancelled.

??? note "Proof that the harmonic mean of the likelihood over posterior draws is unbiased for $1/m(y)$ and has infinite variance under weak conditions, so it converges too slowly to be usable"

    Start from the identity, valid whenever the prior is proper,
    $$\int\frac{1}{f(y\mid\theta)}\,\pi(\theta\mid y)\,\mathrm{d}\theta=\int\frac{1}{f(y\mid\theta)}\cdot\frac{f(y\mid\theta)\pi(\theta)}{m(y)}\,\mathrm{d}\theta=\frac{1}{m(y)},$$
    which suggests estimating $m(y)$ by the harmonic mean of the likelihood evaluated at posterior draws. The estimator requires only posterior samples, which any sampler already produces, and needs no proposal — which is why it is popular.

    Its variance is the problem. The variance of the summand is governed by
    $$\int\frac{1}{f(y\mid\theta)^{2}}\,\pi(\theta\mid y)\,\mathrm{d}\theta=\frac{1}{m(y)}\int\frac{\pi(\theta)}{f(y\mid\theta)}\,\mathrm{d}\theta,$$
    and the remaining integral diverges whenever the prior places mass on regions where the likelihood is arbitrarily small — which it does for essentially every model with an unbounded parameter space, since $1/f(y\mid\theta)$ grows without bound in the tails while $\pi$ decays only polynomially or exponentially. The estimator is therefore unbiased with infinite variance: it is dominated by the rare draw landing in a low-likelihood region, obeys no central limit theorem, and its running value drifts upward in occasional jumps rather than converging.

    Importance sampling repairs it by choosing the proposal rather than inheriting it. Writing $m(y)=\mathbb{E}_q[f(y\mid\theta)\pi(\theta)/q(\theta)]$ for any $q$ with suitable support, the variance is finite when $q$ has heavier tails than the integrand — the condition [Importance Sampling](../part-09-monte-carlo-methods/04-importance-sampling.md) establishes. Taking $q=\pi$ recovers the naive prior-proposal estimator, which is finite-variance but wasteful when the posterior is much tighter than the prior; taking $q$ to be an over-dispersed approximation to the posterior is the standard advice and the basis of bridge and path sampling. The requirement that the proposal be over-dispersed is the same condition an independence sampler needs to be usable at all, priced in [Metropolis–Hastings](../part-17-statistical-computing/05-metropolis-hastings.md).

    **The load-bearing distinction is that the harmonic mean estimator fails silently: it returns a finite number, that number moves as draws are added, and the movement looks like ordinary Monte Carlo noise rather than like an estimator with no second moment.**

The failure is visible at any sample size once the exact answer is known:

```python
import numpy as np
from scipy import stats, special

rng = np.random.default_rng(16065)
sig, tau, n = 0.010, 0.0020, 250
v = sig ** 2 / n
x = rng.standard_normal(n) * sig + 0.0006
xb = x.mean()

exact = stats.norm.logpdf(xb, 0.0, np.sqrt(v + tau ** 2))
pv = 1 / (1 / tau ** 2 + 1 / v)
pm = pv * xb / v
print(f"  a conjugate normal model whose log evidence is known exactly to be"
      f" {exact:.6f}; three estimators are scored against it")
print("        draws   harmonic mean   error   prior-proposal IS   error"
      "   over-dispersed IS   error")
sd_q = 1.5 * np.sqrt(pv)                               # the standard advice: fatten it
for S in (1_000, 10_000, 100_000, 1_000_000):
    th_post = rng.normal(pm, np.sqrt(pv), S)
    ll = stats.norm.logpdf(xb, th_post, np.sqrt(v))
    hm = -(special.logsumexp(-ll) - np.log(S))         # harmonic mean of the likelihood
    th_pri = rng.normal(0.0, tau, S)
    ip = special.logsumexp(stats.norm.logpdf(xb, th_pri, np.sqrt(v))) - np.log(S)
    th_q = rng.normal(pm, sd_q, S)
    lw = (stats.norm.logpdf(xb, th_q, np.sqrt(v))
          + stats.norm.logpdf(th_q, 0.0, tau)
          - stats.norm.logpdf(th_q, pm, sd_q))
    br = special.logsumexp(lw) - np.log(S)
    print(f"    {S:9,d}   {hm:13.6f}   {hm - exact:+6.4f}   {ip:17.6f}   {ip - exact:+6.4f}"
          f"   {br:25.6f}   {br - exact:+6.4f}")

print("  posterior model probabilities over three nested candidates, and what averaging buys")
print("     candidate        log evidence   posterior weight   forecast, bp"
      "   BMA contribution, bp")
cands = (("edge is zero", 0.0), ("weak prior, 5bp", 0.0005), ("wide prior, 50bp", 0.0050))
le, fc = [], []
for nm, t in cands:
    le.append(stats.norm.logpdf(xb, 0.0, np.sqrt(v + t ** 2)))
    fc.append((t ** 2 / (t ** 2 + v)) * xb if t else 0.0)
le, fc = np.array(le), np.array(fc)
w = np.exp(le - le.max())
w /= w.sum()
for (nm, _), l, ww, f in zip(cands, le, w, fc):
    print(f"    {nm:16s}   {l:12.4f}   {ww:16.4f}   {f * 1e4:12.4f}   {ww * f * 1e4:20.4f}")
print(f"    the model-averaged forecast is {(w @ fc) * 1e4:.4f}bp against the best single"
      f" model's {fc[int(np.argmax(w))] * 1e4:.4f}bp and the sample mean's {xb * 1e4:.4f}bp")
# =>   a conjugate normal model whose log evidence is known exactly to be 5.075194; three estimators are scored against it
#            draws   harmonic mean   error   prior-proposal IS   error   over-dispersed IS   error
#            1,000        5.584422   +0.5092            5.126273   +0.0511                    5.062847   -0.0123
#           10,000        5.237553   +0.1624            5.073804   -0.0014                    5.075612   +0.0004
#          100,000        5.232730   +0.1575            5.069481   -0.0057                    5.073971   -0.0012
#        1,000,000        5.251491   +0.1763            5.074819   -0.0004                    5.074589   -0.0006
#      posterior model probabilities over three nested candidates, and what averaging buys
#         candidate        log evidence   posterior weight   forecast, bp   BMA contribution, bp
#        edge is zero             4.5459             0.2903         0.0000                 0.0000
#        weak prior, 5bp          5.0343             0.4731         4.7431                 2.2439
#        wide prior, 50bp         4.3415             0.2366        12.1379                 2.8721
#        the model-averaged forecast is 5.1160bp against the best single model's 4.7431bp and the sample mean's 12.3321bp
```

The exact log evidence is $5.075194$ and the three estimators are scored against it directly. The harmonic mean's error is $+0.5092$ at a thousand draws and $+0.1763$ at a million — it improved by a factor of three while the sample grew by a factor of a thousand, and it is not obviously heading anywhere. A correctly behaved Monte Carlo estimator improves by a factor of about thirty-two over that range. Prior-proposal importance sampling does: $+0.0511$, $-0.0014$, $-0.0057$, $-0.0004$. The over-dispersed proposal does slightly better again at $-0.0123$, $+0.0004$, $-0.0012$, $-0.0006$, and it does so with a proposal that has to be constructed rather than inherited, which is the whole of the extra work.

The second block prices model averaging in the currency a desk cares about. Three candidates — the edge is exactly zero, a five-basis-point prior, a fifty-basis-point prior — have log evidences of $4.5459$, $5.0343$ and $4.3415$, giving posterior weights of $0.2903$, $0.4731$ and $0.2366$. No model dominates, which is the ordinary situation at two hundred and fifty days, and the concentration [Model Averaging](../part-14-model-selection/05-model-averaging.md) describes has not begun. The model-averaged forecast is $5.1160$ basis points against the best single model's $4.7431$ and the raw sample mean's $12.3321$. The averaging did two things at once: it shrank the sample mean by more than half, and it did so by a factor that was computed rather than chosen — which is the strongest argument for the whole apparatus, and is worth exactly as much as the three priors it was built from.

## Everything on This Page Requires a Proper Prior, Which Is Why the Comparison Cannot Be Made Prior-Free

The dependence measured in section 3 is not a defect to be engineered away, and the attempts are instructive. An improper prior on the alternative makes the Bayes factor undefined rather than uninformative: multiplying an improper prior by an arbitrary constant multiplies $m_1$ by that constant and hence $B_{10}$ too, so the comparison has no value at all. This is the sharpest practical difference from parameter estimation, where [Prior Distributions](02-prior-distributions.md) showed improper priors are usually harmless because the normalizer cancels — here it is the answer.

The standard repairs each buy something and pay for it. Unit-information priors set $g=1$ by construction, which is what makes BIC an approximation to a Bayes factor at all, and the choice is a convention rather than a belief. Zellner's $g$-prior parameterizes the scale explicitly and then requires a value of $g$; putting a hyperprior on $g$ — the mixture-of-$g$-priors approach — is a genuine improvement, and by section 3's algebra it is also a mixture of the columns in that first table, so it does not escape the dependence, it averages over it with weights someone chose. Fractional and intrinsic Bayes factors use part of the data to convert an improper prior into a proper one and the rest to compare, which works and reintroduces a choice of how much data to spend.

The honest position, and the one [Model Averaging](../part-14-model-selection/05-model-averaging.md) takes when it distinguishes M-closed from M-open, is that a Bayes factor answers a well-posed question — which of these fully specified models, priors included, predicted this dataset better — and that the question is only worth asking when the priors were specified for reasons that survive scrutiny. Where they were not, the sensitivity is the finding, and the report should be the curve rather than the point.

!!! note "A Bayes factor, a likelihood ratio, a likelihood-ratio test statistic, a BIC difference and a posterior model probability are five numbers built from the same two fits, and only two of them are comparable across sample sizes"
    [Likelihood Ratio Tests](../part-12-hypothesis-testing/06-likelihood-ratio-tests.md) distinguishes the first four and this page supplies the fifth. A **likelihood ratio** compares maximized likelihoods and always favours the larger model, since a nested maximum cannot be smaller. A **likelihood-ratio test statistic** is twice its logarithm and is the only member with a null distribution, which is what lets it be turned into a p-value. A **Bayes factor** integrates each likelihood against its prior rather than maximizing, so it is not a function of the maxima at all, it penalizes automatically by section 1's normalization argument, and it depends on the priors in the way section 3 measures. A **BIC difference** is a Laplace approximation to twice the log Bayes factor under a unit-information prior, accurate to $O(1)$ — a gap measured here at up to $3.0709$ nats — and should be used for ranking rather than exponentiated into odds. A **posterior model probability** is a Bayes factor combined with prior model odds and normalized across the candidate set, so it changes when a candidate is added even though no Bayes factor between the original pair changed. Quoting a posterior model probability without stating the candidate set, or an odds ratio derived from a BIC difference, are the two errors this list exists to prevent.

!!! warning "A Bayes factor computed at one prior scale looks exactly like a Bayes factor computed at the right one, and the sample size that should settle the question makes the dependence stronger rather than weaker"
    Every verdict in section 3 came from identical data. Nine rows at a p-value of exactly $0.05$ returned Bayes factors from $2.0790$ for the alternative to $0.0341$ — odds of $29.30$ to one for the null — driven entirely by a prior width nobody in the analysis had to defend, and the spread *widens* with $n$ because $g=n\tau^{2}/\sigma^{2}$ grows. The estimation side fails just as quietly: the harmonic mean estimator returned $5.584422$, $5.237553$, $5.232730$ and $5.251491$ against an exact $5.075194$, four plausible-looking numbers that never converge, while a proposal-based estimator on the same draws reached $-0.0004$ of the truth. **The free diagnostic is to recompute every Bayes factor you intend to report across a grid of prior scales spanning two orders of magnitude either side of your choice, print $\log B_{10}$ against $\log\tau$, and quote the range rather than the point whenever the verdict changes sign anywhere on that grid — and if you estimated the evidence numerically, run the same estimator on a conjugate simplification of your model where the answer is known, because an estimator with no second moment cannot be detected from its own output.** The first check is four lines and the second is the only way to find out that the harmonic mean has been lying.

## A Verdict That Is a Function of a Choice Nobody Recorded

This page established that the marginal likelihood integrates to one over datasets, verified at $1.0000000000$, so a model's complexity penalty is the normalization constraint rather than an added term, and posterior odds are prior odds times the Bayes factor; that the penalty is exact rather than asymptotic and grows with the sample, measured at $0.0004$, $0.0049$, $0.0255$, $0.1680$ and $0.9438$ nats, while BIC's approximation to the same quantity leaves an $O(1)$ gap running $-0.3332$ to $3.0709$ nats and the balance point $|z|$ moves from $1.0098$ to $1.7350$; that at a p-value pinned to exactly $0.05$ the Bayes factor reads anywhere from $2.0790$ for the alternative to $0.0341$ against it, an odds of $29.30$ to one for the null, as the prior width and sample size vary, with the balance-point p-value itself running $0.15608$, $0.05106$, $0.01423$ and $0.00397$; that the most favourable alternative in existence caps the evidence at $p=0.05$ to a Bayes factor of $6.83$ and the honest bound to $2.4560$, a posterior probability of $0.7107$ at even prior odds; and that the harmonic mean estimator of the evidence has infinite variance, its error sitting at $+0.5092$, $+0.1624$, $+0.1575$ and $+0.1763$ nats across a thousandfold increase in draws where proposal-based estimators reached $-0.0004$.

The shape shared by all three exhibits is the part's recurring one arriving in its most consequential form. Sections 1 and 2 are a genuine achievement: an exact complexity penalty derived from normalization, requiring no data splitting and no asymptotic argument, which is more than any criterion in [Part XIV](../part-14-model-selection/index.md) can claim. Section 3 shows that the same exactness is what makes the answer a function of a prior scale, since the penalty term $-\tfrac12\log(1+g)$ has no other source. The virtue and the vulnerability are the same equation, and the output displays neither.

What remains is the question none of the preceding pages asked. Every posterior and every model comparison so far has been about parameters — quantities that are never observed and never settle anything. What a desk actually needs is a distribution over the next return, the next month's winning trades, the next drawdown, and the machinery for producing one is the last construction in this part. That is [Bayesian Prediction](07-bayesian-prediction.md).

**A Bayes factor charges a model for the breadth of its predictions, which is exactly right, and the width it charges for was set by a prior chosen for convenience — so the most defensible comparison in statistics is decided by the least defended number in the analysis.**
