# Exponential Families

The previous page ended on a question it could not answer: sufficiency guarantees losslessness but says nothing about dimension, and the order statistics are sufficient for every iid model while compressing almost nothing. So which models admit a summary whose size stays fixed as the sample grows? The answer is unusually sharp for statistics — essentially one class, defined by a single functional form — and the reason it deserves a page of its own is not the form itself but everything that turns out to be equivalent to it. Fixed-dimension sufficiency, a concave likelihood, moments available by differentiation, and a conjugate prior are not four conveniences that happen to co-occur. They are one structural fact seen from four directions, which is why they arrive together and, more to the point, why they leave together.

This page covers the exponential-family form and the named distributions it absorbs, the log-partition function as a moment generator and the convexity that follows from it, the sufficient statistic read directly off the exponent with a dimension that does not grow with $n$, the Pitman–Koopman–Darmois theorem and the support condition it actually needs, and what is lost for the models a trader fits that sit outside the class. It defines sufficiency and proves the factorization theorem elsewhere, which is [Statistics and Sufficiency](05-statistics-and-sufficiency.md); it derives the density, mean and variance of no individual named law, which is [Part V](../part-05-common-distributions/index.md); it maximizes no likelihood and runs no optimizer, which is [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md) and [Part XVII](../part-17-statistical-computing/index.md); it constructs no conjugate prior and updates no posterior, which is [Conjugate Priors](../part-16-bayesian-statistics/04-conjugate-priors.md); it fits no generalized linear model, which is [Generalized Linear Models](../part-13-regression/03-generalized-linear-models.md); it establishes no property of an estimator, which is [Part XI](../part-11-parameter-estimation/index.md); and it recommends no distribution for returns.

The trading stake is the model the course actually fits and the conveniences it silently forfeits. [Time Series Analysis](../../part-03-statistics/03-time-series.md) reports a GARCH(1,1) at `omega 0.0252  alpha 0.126  beta 0.856  persistence 0.982` and then a GJR extension at `GJR alpha 0.000  gamma 0.186 (p = 1.3e-10)`, concluding that "For equity indexes, volatility does not respond to surprise — it responds to *bad* surprise, entirely." Neither model is an exponential family in its parameters. Every guarantee established below — a summary of fixed size, a likelihood with one optimum, a moment obtained by differentiating — is unavailable for exactly the model the lesson needed, and the fifth section measures what that costs on a mixture where the maximum-likelihood estimate does not exist at all.

## One Functional Form Covers Most of the Named Distributions

A family $\{p_\theta\}$ is an **exponential family** if its densities can be written

$$p_\theta(x)=h(x)\,\exp\!\big\{\eta(\theta)^{\top}T(x)-A(\theta)\big\},$$

with $T(x)$ the **sufficient statistic**, $\eta(\theta)$ the **natural parameter**, $A(\theta)$ the **log-partition function** that normalizes the density, and $h(x)$ a carrier free of $\theta$. Reparameterizing by $\eta$ itself gives the **natural form** $p_\eta(x)=h(x)\exp\{\eta^{\top}T(x)-A(\eta)\}$, and everything below is cleanest there.

The form is less restrictive than it looks. A normal with known variance is $\eta=\mu$, $T(x)=x$, $A(\eta)=\eta^2/2$. A Poisson is $\eta=\log\lambda$, $T(x)=x$, $A(\eta)=e^{\eta}$. An exponential with rate $\lambda$ is $\eta=-\lambda$, $T(x)=x$, $A(\eta)=-\log(-\eta)$. A Bernoulli is $\eta=\log\frac{p}{1-p}$, $T(x)=x$, $A(\eta)=\log(1+e^{\eta})$ — the log-odds, arriving as the natural parameter rather than as a modelling choice, which is where logistic regression's link function comes from. The normal with both parameters unknown, the gamma, the beta, the chi-square and the multinomial all fit, with $T$ two-dimensional where two parameters are free. [Part V](../part-05-common-distributions/index.md) treats these individually; the point here is that one calculation covers them at once.

What does *not* fit is worth naming immediately, because it is the more interesting list. A Student-$t$ with unknown degrees of freedom is not an exponential family. A uniform on $[0,\theta]$ is not. A mixture of two normals is not. A GARCH process is not. The boundary between the two lists is the subject of the fourth section.

## The Log-Partition Function Generates the Moments and Makes the Likelihood Concave

$A$ exists only to normalize the density, which makes it look like bookkeeping. It is not: differentiating it produces the moments of $T$, and its second derivative being a variance is what makes the whole class computationally tractable.

??? note "Proof that $\nabla A(\eta)=\mathbb{E}[T(X)]$ and $\nabla^{2}A(\eta)=\mathrm{var}(T(X))$, so $A$ is convex and the log-likelihood is concave"
    Normalization requires $\int h(x)\exp\{\eta^{\top}T(x)\}\,dx=e^{A(\eta)}$. Differentiating both sides with respect to $\eta$ and exchanging derivative and integral,

    $$\int T(x)h(x)e^{\eta^{\top}T(x)}dx=e^{A(\eta)}\nabla A(\eta)\;\Longrightarrow\;\nabla A(\eta)=\int T(x)\,p_\eta(x)\,dx=\mathbb{E}_\eta[T(X)].$$

    Differentiating a second time and using the same exchange,

    $$\nabla^{2}A(\eta)=\mathbb{E}_\eta\!\left[T T^{\top}\right]-\mathbb{E}_\eta[T]\,\mathbb{E}_\eta[T]^{\top}=\mathrm{var}_\eta\big(T(X)\big),$$

    which is a covariance matrix and therefore positive semi-definite at every $\eta$. So $A$ is convex on the natural parameter space. The log-likelihood of an iid sample is

    $$\ell(\eta)=\eta^{\top}\sum_i T(x_i)-n A(\eta)+\text{const},$$

    a linear function minus $n$ times a convex one, hence concave — with strict concavity wherever $\mathrm{var}(T)$ is positive definite. A concave function on a convex set has no local maximum that is not global, and the first-order condition $\nabla A(\hat\eta)=\frac1n\sum_i T(x_i)$ says the fitted model matches the observed average of $T$ exactly.

    The load-bearing step is the exchange of differentiation and integration, valid on the interior of the set where the integral converges, and the load-bearing consequence is the convexity. **Concavity is what makes maximum likelihood a solved problem inside this class and an open one outside it** — inside, any hill-climbing algorithm from any starting point reaches the same answer, and the phrase "the optimizer converged" carries information; outside, it reports only that the algorithm stopped.

```python
import numpy as np

rng = np.random.default_rng(10061)
draws, h = 400_000, 1e-5                                       # central differences on A(eta)


def check(name, A, eta, sample):
    d1 = (A(eta + h) - A(eta - h)) / (2 * h)
    d2 = (A(eta + h) - 2 * A(eta) + A(eta - h)) / h ** 2
    t = sample(draws)
    print(f"  {name:<22} {eta:8.4f} {d1:12.6f} {t.mean():12.6f} {d2:12.6f} {t.var(ddof=1):12.6f}")


print("   family                      eta       A'(eta)        E[T]      A''(eta)      var(T)")
check("normal, sigma = 1", lambda e: e ** 2 / 2, 0.8,
      lambda m: 0.8 + rng.standard_normal(m))
check("Poisson", lambda e: np.exp(e), np.log(3.5),
      lambda m: rng.poisson(3.5, m).astype(float))
check("exponential", lambda e: -np.log(-e), -2.0,
      lambda m: rng.exponential(0.5, m))
check("Bernoulli", lambda e: np.log1p(np.exp(e)), np.log(0.3 / 0.7),
      lambda m: (rng.random(m) < 0.3).astype(float))
# =>    family                      eta       A'(eta)        E[T]      A''(eta)      var(T)
#      normal, sigma = 1        0.8000     0.800000     0.800038     1.000000     1.001880
#      Poisson                  1.2528     3.500000     3.497848     3.500005     3.509679
#      exponential             -2.0000     0.500000     0.499173     0.249999     0.249601
#      Bernoulli               -0.8473     0.300000     0.299812     0.210000     0.209925
```

Four families, four different sample spaces — the real line, the non-negative integers, the positive half-line, and two points — and one calculation. The first derivative of $A$, computed by central differences on a function written down from the density, matches the simulated mean of $T$ in every row: $0.800000$ against $0.800038$, $3.500000$ against $3.497848$, $0.500000$ against $0.499173$, $0.300000$ against $0.299812$. The agreement is to the Monte Carlo error of four hundred thousand draws, which is what it should be.

The second derivative does the same for the variance: $1.000000$ against $1.001880$, $3.500005$ against $3.509679$, $0.249999$ against $0.249601$, $0.210000$ against $0.209925$. The Poisson row is the recognizable one — $A''=A'=\lambda$ is the familiar statement that a Poisson's variance equals its mean, arriving here as a property of $A$ rather than as a separate fact to memorize. The Bernoulli's $0.210000$ is $p(1-p)$ at $p=0.3$, obtained without ever writing $p(1-p)$ down.

**Every moment of $T$ in this class is a derivative of one scalar function**, and the same function's convexity is what guarantees the likelihood has a single peak. That is a great deal of machinery to get from a normalizing constant, and the fourth section explains why nothing outside the class gets any of it.

## The Sufficient Statistic Is Read Off the Exponent and Its Dimension Never Grows

The factorization theorem of the previous page says the sufficient statistic is whatever the parameter touches. In the natural form the parameter touches exactly $T(x)$, so sufficiency can be read off by inspection.

??? note "Proof that $\sum_i T(x_i)$ is sufficient for an iid sample and that its dimension is the same for every $n$"
    For an iid sample from an exponential family the joint density is a product,

    $$p_\eta(x_{1:n})=\Big(\prod_i h(x_i)\Big)\exp\!\Big\{\eta^{\top}\sum_{i=1}^{n}T(x_i)-nA(\eta)\Big\},$$

    because the exponents add. This is exactly the factorization $g_\eta(S(x))\,h(x)$ with $S(x)=\sum_i T(x_i)$, so $S$ is sufficient. It is also minimal whenever the family is of full rank, since the likelihood ratio between two samples is free of $\eta$ precisely when their values of $S$ agree.

    The dimension is the point. $T$ maps each observation into $\mathbb{R}^{k}$ where $k$ is fixed by the family, and summing does not change the dimension, so $S(x)\in\mathbb{R}^{k}$ whether $n$ is ten or ten million. An iid exponential-family sample of any length is losslessly represented by $k$ numbers, and because $S$ is a sum, those numbers can be accumulated in a single pass, merged across partitions, and updated online without revisiting a single observation.

    The load-bearing step is the additivity of the exponent, which is what converts a product over observations into a sum inside one exponential. **The reason a streaming system can hold two accumulators and lose nothing is a property of the family and not of the estimator**, which is the point the previous page's risk system got wrong by keeping the accumulators after leaving the family.

```python
import numpy as np

rng = np.random.default_rng(10063)
n = 12                                                         # two samples, one value of (sum, sum sq)

x = 0.4 + 1.7 * rng.standard_normal(n)
q = rng.standard_normal(n)
q = q - q.mean()
y = x.mean() + q * (x.std(ddof=0) / q.std(ddof=0))
print(f"  both samples: sum {x.sum():.8f} / {y.sum():.8f},"
      f" sum of squares {(x ** 2).sum():.8f} / {(y ** 2).sum():.8f}")


def norm_ll(mu, s, d):
    return float(-d.size * np.log(s) - ((d - mu) ** 2).sum() / (2 * s ** 2))


def t_ll(mu, s, nu, d):
    return float(-d.size * np.log(s) - (nu + 1) / 2 * np.log1p(((d - mu) / s) ** 2 / nu).sum())


print("     model                    loglik(A)     loglik(B)          gap")
for mu, s in ((0.4, 1.7), (0.0, 2.0)):
    print(f"  {'normal mu=' + str(mu) + ' sigma=' + str(s):<24} {norm_ll(mu, s, x):12.6f}"
          f" {norm_ll(mu, s, y):13.6f} {abs(norm_ll(mu, s, x) - norm_ll(mu, s, y)):12.2e}")
for nu in (3.0, 8.0):
    a, b = t_ll(0.4, 1.7, nu, x), t_ll(0.4, 1.7, nu, y)
    print(f"  {'t(' + str(nu) + ') mu=0.4 sigma=1.7':<24} {a:12.6f} {b:13.6f} {abs(a - b):12.2e}")
# =>   both samples: sum 9.38134228 / 9.38134228, sum of squares 40.41590033 / 40.41590033
#         model                    loglik(A)     loglik(B)          gap
#      normal mu=0.4 sigma=1.7    -12.393634    -12.393634     1.78e-15
#      normal mu=0.0 sigma=2.0    -13.369754    -13.369754     1.78e-15
#      t(3.0) mu=0.4 sigma=1.7    -12.457957    -12.132881     3.25e-01
#      t(8.0) mu=0.4 sigma=1.7    -12.356498    -12.167848     1.89e-01
```

Two samples are constructed to share a sum of $9.38134228$ and a sum of squares of $40.41590033$ while being different data. Under the normal model their log-likelihoods agree to $1.78\times10^{-15}$ at both parameter settings — floating-point identity, the same result the previous page obtained, and here it is visibly a consequence of $(\sum x,\sum x^{2})$ being the exponent's $T$.

The $t$ rows are the same two samples under a family that is not exponential. At three degrees of freedom the log-likelihoods differ by $0.325$; at eight, by $0.189$. **The pair that was a complete summary one line earlier is now demonstrably lossy**, and it is lossy in a way that has a direction: the $t$ density's $\log(1+z^2/\nu)$ term weights each observation individually, so it can tell apart samples that agree on both moments, which is precisely the information a running sum discards. Note also which way the gap moves — it is larger at $\nu=3$ than at $\nu=8$, so the heavier the tail the more the reduction costs.

## Fixed-Dimension Sufficiency Is Essentially Exclusive to This Class

The converse is what makes the class worth naming rather than merely convenient: outside it, no fixed-size summary exists.

??? note "Proof that a $\theta$-dependent support blocks the exponential form, and what Pitman–Koopman–Darmois does and does not claim"
    The theorem states that for a family of densities with **support not depending on the parameter**, satisfying mild smoothness, a sufficient statistic whose dimension is bounded independently of $n$ exists only if the family is exponential. The support condition is not a technicality and the standard counterexample shows why.

    Take the uniform family on $[0,\theta]$, with density $\theta^{-1}\mathbf{1}\{0\le x\le\theta\}$. The joint density of an iid sample is

    $$p_\theta(x_{1:n})=\theta^{-n}\,\mathbf{1}\{x_{(n)}\le\theta\}\,\mathbf{1}\{x_{(1)}\ge0\},$$

    which factors with $T(x)=x_{(n)}=\max_i x_i$. So a **one-dimensional** sufficient statistic exists at every $n$, and the family is emphatically not exponential — an exponential family's density is positive wherever $h$ is, so its support cannot move with $\theta$, while this one's does. Pitman–Koopman–Darmois is not contradicted, because its hypothesis is exactly the condition this family violates.

    The load-bearing hypothesis is therefore the fixed support, and it is the reason the theorem is routinely misquoted as "only exponential families have low-dimensional sufficient statistics" when the correct statement carries a clause. **The practical content survives the correction intact: among families with fixed support — which is every model anyone fits to returns — a summary of bounded size means an exponential family and nothing else**, and the Student-$t$, the mixture and the GARCH all have fixed support and all fail, so for them the minimal sufficient statistic is the full order statistics and grows with the sample.

!!! note "Conjugate priors, constant-memory accumulators and concave likelihoods are three faces of one fact, which is why they arrive and depart together"
    A conjugate prior is a prior whose functional form is preserved by the likelihood, and in the natural parameterization the construction is immediate: a prior proportional to $\exp\{\eta^{\top}\tau-\kappa A(\eta)\}$ multiplied by the likelihood $\exp\{\eta^{\top}\sum_i T(x_i)-nA(\eta)\}$ returns a density of the same shape with $\tau\mapsto\tau+\sum_i T(x_i)$ and $\kappa\mapsto\kappa+n$. Bayesian updating is therefore *addition* on the sufficient statistic, which is the same additivity that gave the constant-memory accumulator, which is the same exponent structure that gave the concavity. The relationship is not analogy — it is one property described three ways, and [Conjugate Priors](../part-16-bayesian-statistics/04-conjugate-priors.md) develops the Bayesian side. The consequence for practice is a reliable early warning: **the moment a model stops admitting a closed-form conjugate update, expect the constant-memory summary and the unique optimum to be gone as well**, because all three were the same structural fact and none of them survives alone.

## Most Models a Trader Fits Are Outside the Class

Every model in the course that does real work is outside. A GARCH's conditional variance depends recursively on the whole history, so the likelihood does not factor into a fixed exponent. A Student-$t$ with unknown $\nu$ was shown lossy above. A hidden Markov model and a normal mixture are sums over unobserved labels, and a sum of exponential-family terms is not an exponential family. Here is what is forfeited.

```python
import numpy as np

rng = np.random.default_rng(10067)
n, starts = 600, 40                                            # a two-component normal mixture

z = rng.random(n) < 0.35
d = np.where(z, rng.normal(-1.5, 0.6, n), rng.normal(0.8, 1.1, n))


def mix_ll(w, m1, s1, m2, s2):
    a = w * np.exp(-((d - m1) / s1) ** 2 / 2) / s1
    b = (1 - w) * np.exp(-((d - m2) / s2) ** 2 / 2) / s2
    return float(np.log(a + b + 1e-300).sum())


def em(w, m1, s1, m2, s2, iters=400):
    for _ in range(iters):
        a = w * np.exp(-((d - m1) / s1) ** 2 / 2) / s1
        b = (1 - w) * np.exp(-((d - m2) / s2) ** 2 / 2) / s2
        r = a / (a + b + 1e-300)
        w = min(max(r.mean(), 1e-6), 1 - 1e-6)
        m1, m2 = (r * d).sum() / r.sum(), ((1 - r) * d).sum() / (1 - r).sum()
        s1 = max(np.sqrt((r * (d - m1) ** 2).sum() / r.sum()), 1e-8)
        s2 = max(np.sqrt(((1 - r) * (d - m2) ** 2).sum() / (1 - r).sum()), 1e-8)
    return w, m1, s1, m2, s2


mu, sg = d.mean(), d.std(ddof=0)
one = -n * np.log(sg) - ((d - mu) ** 2).sum() / (2 * sg ** 2)
print(f"  one normal on {n} points, closed form: mu {mu:+.6f}  sigma {sg:.6f}  loglik {one:.3f}")
print("   a component collapsing onto a single observation")
print("      sigma_1      loglik")
for s1 in (1e-1, 1e-2, 1e-3, 1e-5, 1e-8):
    print(f"  {s1:11.0e} {mix_ll(1.0 / n, d[0], s1, mu, sg):11.2f}")

lls, m1s = [], []
for _ in range(starts):
    p = em(rng.uniform(0.2, 0.8), rng.uniform(-3, 3), rng.uniform(0.3, 2.0),
           rng.uniform(-3, 3), rng.uniform(0.3, 2.0))
    lls.append(mix_ll(*p))
    m1s.append(p[1])
lls, m1s = np.array(lls), np.array(m1s)
print(f"   EM from {starts} random starts on the same data")
print(f"     interior optimum {lls.max():.4f}, distinct values"
      f" {len(np.unique(np.round(lls, 4)))}, guarded away from the spike")
print(f"     fitted mu_1 lands near {np.round(np.unique(np.round(m1s, 3)), 2)}"
      f" in {np.mean(m1s < 0):.0%} / {np.mean(m1s > 0):.0%} of starts")
# =>   one normal on 600 points, closed form: mu +0.014880  sigma 1.457521  loglik -526.042
#       a component collapsing onto a single observation
#          sigma_1      loglik
#            1e-01     -525.67
#            1e-02     -524.88
#            1e-03     -523.62
#            1e-05     -521.12
#            1e-08     -514.22
#       EM from 40 random starts on the same data
#         interior optimum -497.5702, distinct values 1, guarded away from the spike
#         fitted mu_1 lands near [-1.44  0.77] in 35% / 65% of starts
```

The first line is the inside of the class, for contrast. A single normal on six hundred points has its maximum-likelihood estimate in closed form — $\hat\mu=+0.014880$, $\hat\sigma=1.457521$ — reached by no iteration, dependent on no starting value, and unique because the likelihood is concave. There is nothing to tune and nothing to check.

The sweep is the outside, and it is worse than multimodality. Placing one component on a single observation and shrinking its width drives the log-likelihood to $-525.67$, $-524.88$, $-523.62$, $-521.12$ and $-514.22$ as $\sigma_1$ falls from $10^{-1}$ to $10^{-8}$, with no bound: the density at that one point grows like $1/\sigma_1$ while the other component continues to explain everything else. **The supremum of the mixture likelihood is $+\infty$ and the maximum-likelihood estimate does not exist**, which is not a numerical difficulty but a statement about the model.

The last two lines say what an implementation actually returns. Forty random starts all reach the same interior optimum at $-497.5702$, which looks like reassuring stability until one notices why: the variance floor of $10^{-8}$ in the update is what stops the algorithm walking off toward the spike, so the reported answer is a property of a guard rather than of the model. And the labelling is arbitrary in the way [Statistical Models](04-statistical-models.md) predicted — the fitted $\mu_1$ lands near $-1.44$ in $35\%$ of starts and near $+0.77$ in the other $65\%$, so averaging $\hat\mu_1$ across runs converges to a number describing neither component.

!!! warning "An optimizer that needs a variance floor, a starting value or a convergence tolerance to return an answer is reporting a property of those choices, and no amount of tuning converts that back into a guarantee"
    Inside an exponential family the phrase "the fit converged" is informative, because concavity means any interior stationary point is the global optimum. Outside it the phrase means only that the iteration stopped moving, and three separate things may be true: the optimum found is local, the global supremum may be infinite and unattainable, and the parameters reported may be one of several labellings of the same law. The free diagnostics are the ones from the previous page's warning plus one specific to this class: **before fitting, ask whether the model has a fixed-dimension sufficient statistic — equivalently, whether a conjugate prior exists for it — and if the answer is no, treat every fitted parameter as conditional on the start and the guard until refits from separated starting values say otherwise.** The corollary matters for engineering rather than for statistics: any pipeline that summarizes and discards on the strength of "these numbers are sufficient" has made an exponential-family assumption, and that assumption should be written next to the accumulator rather than inferred later from a breach count.

## A Class Worth Knowing By Its Boundary

Three things were established and they are one thing. The log-partition function generates the moments of $T$ by differentiation and is convex, so the log-likelihood is concave and has a single peak. The exponent's additivity makes $\sum_i T(x_i)$ sufficient at a dimension that never grows, so an arbitrarily long sample compresses to a fixed handful of numbers computable in one pass. And the same structure makes a conjugate prior an addition on that statistic. Pitman–Koopman–Darmois closes the circle: among families with fixed support, nothing outside the class has the second property, and the first and third go with it.

The symmetry with the previous page is what makes the boundary worth memorizing rather than the form. [Statistics and Sufficiency](05-statistics-and-sufficiency.md) showed that a reduction derived under one model is lossy under another, and measured the loss on a risk number. This page identifies exactly which models admit a useful reduction in the first place, and the answer partitions the world in a way that lines up almost perfectly with a practitioner's intuition about which models are pleasant: the pleasant ones are the exponential families, and the pleasantness is one theorem rather than a series of lucky coincidences. The corollary is the uncomfortable half — every model in the course that captures something real about returns, from GARCH's persistence of $0.982$ to the GJR asymmetry at $\gamma=0.186$ to any mixture over regimes, sits outside, so the guarantees are absent exactly where the modelling is interesting.

That is not an argument against those models. It is an argument for knowing which guarantees are in force, because the failure modes on either side of the boundary are different failures. Inside, the remaining question is how far a correctly computed estimate sits from the truth, and that error splits into two pieces with opposite cures. That is [Bias and Variance](07-bias-and-variance.md).

**The exponential family is the set of models where the convenient thing and the correct thing coincide, and knowing its boundary is worth more than knowing its form, because the boundary is where every guarantee stops without announcing it.**
