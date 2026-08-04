# Conjugate Priors

Conjugacy is usually presented as an algebraic convenience that computation has made obsolete, and both halves of that are wrong. It is not an accident of algebra — it is the closure property of the exponential family, and the prior's parameters are literally a count of imaginary observations and their sufficient statistic. And it has not been made obsolete, because the assumption it forces is still made constantly by people who would deny holding it: a desk that shrinks a covariance matrix toward a target is using an inverse-Wishart prior it never wrote down. What conjugacy costs is not flexibility in the middle of the distribution but rigidity in the tail, and the tail is what decides how a posterior answers something surprising. Below, a normal prior confronted with an observation twenty standard errors away holds its ground forever, dragging the posterior mean $2.0000$ standard errors off the data and reporting the same standard deviation of $0.9487$ it reports when nothing is wrong; a Student-$t$ prior of identical scale pulls hardest at $0.3778$ around four standard errors and then progressively lets go, falling to $0.1893$. Meanwhile the two smallest entries in the catalogue are worth more than their reputation: treating a volatility as known rather than estimated costs $0.8765$ coverage where the Student-$t$ posterior delivers $0.9495$, and one pseudo-observation converts a transition probability of $-\infty$ in log terms into $-2.3979$.

This page covers conjugacy as closure for exponential families, the update as addition of sufficient statistics, the catalogue and what its two least glamorous entries buy, mixtures of conjugate priors and the nonlinearity they restore, and the tail behaviour that decides how a conjugate posterior resolves a conflict between prior and data. It does not construct the exponential-family conjugate prior from scratch, which is [Exponential Families](../part-10-statistics-foundations/06-exponential-families.md); it does not re-derive the beta–Bernoulli or gamma–Poisson updates, which are [Beta Distribution](../part-05-common-distributions/14-beta-distribution.md) and [Gamma Distribution](../part-05-common-distributions/13-gamma-distribution.md); it does not derive the multivariate Gaussian sequential update, which is [Conditional Gaussian](../part-06-multivariate-probability/06-conditional-gaussian.md); it argues for no prior and checks none against its implications, which is [Prior Distributions](02-prior-distributions.md); it normalizes no posterior numerically and proves no asymptotics, which is [Posterior Distributions](03-posterior-distributions.md); it minimizes no expected loss, which is [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md); it applies no update sequentially and derives no forgetting, which is [Bayesian Updating](05-bayesian-updating.md); it computes no Bayes factor, which is [Bayesian Model Comparison](06-bayesian-model-comparison.md); it derives no shrinkage estimator from a frequentist risk argument, which is [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md); and it never presents closure under updating as evidence that the family is the right one.

The trading stake is the course's most comprehensively negative portfolio result, whose central number is a conjugate prior nobody in the lesson calls one. [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) shrinks a sample covariance toward a scalar target and finds that `LW-MVO scores 0.115 against plain MVO's 0.377`, with the Ledoit–Wolf intensity measured at $1.85\%$ — "doing its job perfectly", the lesson says, and "the job is nearly vacuous here." Shrinking a covariance toward a structured target is exactly the posterior mean under an inverse-Wishart prior centred on that target, with the intensity playing the part of the prior's degrees of freedom; section 5 makes the identification precise. The lesson's finding is that the repair was aimed at the wrong input, and this page adds what it was silently assuming about the tails while it aimed.

## Conjugacy Is the Closure Property of the Exponential Family, and the Update Is Addition

The definition usually given — a prior is conjugate if the posterior lies in the same family — sounds like a coincidence one hunts for family by family. It is a theorem, and it holds for exactly one class of models.

??? note "Proof that an exponential-family likelihood admits a prior closed under updating, that the update adds the observed sufficient statistic and increments a count, and that the hyperparameters are therefore pseudo-observations"

    Let the model be an exponential family in natural form,
    $$f(x\mid\eta)=h(x)\exp\big\{\eta^{\top}T(x)-A(\eta)\big\},$$
    whose construction and log-partition properties are established in [Exponential Families](../part-10-statistics-foundations/06-exponential-families.md). For $n$ conditionally independent observations the likelihood is $\exp\{\eta^{\top}\sum_iT(x_i)-nA(\eta)\}$ times a factor free of $\eta$, so the data enter only through $\sum_iT(x_i)$ and $n$ — which is sufficiency, and is why the argument works at all.

    Take the prior $\pi(\eta\mid\tau,\kappa)\propto\exp\{\eta^{\top}\tau-\kappa A(\eta)\}$ over the set where this is integrable. Multiplying,
    $$\pi(\eta\mid x)\ \propto\ \exp\Big\{\eta^{\top}\Big(\tau+\sum_i T(x_i)\Big)-(\kappa+n)A(\eta)\Big\},$$
    which is the same family at $\tau\mapsto\tau+\sum_iT(x_i)$ and $\kappa\mapsto\kappa+n$. Closure holds, the update is addition, and the arithmetic is $O(1)$ in the number of observations already seen.

    The interpretation of $(\tau,\kappa)$ falls out of the form: the prior is what the posterior would have been starting from nothing and observing $\kappa$ imaginary data points with total sufficient statistic $\tau$. That is exactly the pseudo-count reading [Beta Distribution](../part-05-common-distributions/14-beta-distribution.md) gives $\alpha$ and $\beta$ and [Gamma Distribution](../part-05-common-distributions/13-gamma-distribution.md) gives its shape and rate. **A conjugate prior's strength is measured in observations, which is the one currency that makes prior and data commensurable, and it is why "how many days of evidence is this belief worth" is always an answerable question for a conjugate prior and rarely for any other.**

    One further consequence constrains everything below. Diaconis and Ylvisaker showed that within this family the posterior mean of the mean-parameter is exactly linear in the data,
    $$\mathbb{E}[\mu\mid x]=\frac{\kappa}{\kappa+n}\,\mu_0+\frac{n}{\kappa+n}\,\bar T,$$
    and that this linearity *characterizes* conjugate priors among all priors for a regular exponential family. The load-bearing consequence is that a conjugate posterior mean is a fixed convex combination whose weights were determined before any data arrived: it cannot respond more to a surprising observation than to an unsurprising one, because the weight is not a function of what was seen. Sections 3 and 4 are both about escaping that sentence.

## The Catalogue Is Short, and Its Two Least Glamorous Entries Are Where the Interval and the Zero Come From

The named conjugate pairs number about a dozen. Two of them account for most of the practical damage done by ignoring the family:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(16041)
sig, reps = 0.010, 40_000

print(f"  the mean of {sig * 1e4:.0f}bp returns, with the volatility known and with it"
      f" estimated; the second posterior is Student-t and the first pretends it is not")
print("        n   t crit   normal crit   width ratio   coverage, sigma known"
      "   coverage, sigma estimated")
for n in (5, 10, 30, 100, 250):
    x = rng.standard_normal((reps, n)) * sig
    xb, s = x.mean(1), x.std(1, ddof=1)
    tc, zc = stats.t.isf(0.025, n - 1), stats.norm.isf(0.025)
    known = (np.abs(xb) <= zc * s / np.sqrt(n)).mean()
    est = (np.abs(xb) <= tc * s / np.sqrt(n)).mean()
    print(f"    {n:5d}   {tc:6.4f}   {zc:11.4f}   {tc / zc:11.4f}   {known:21.4f}"
          f"   {est:26.4f}")

print("  a three-regime transition row observed 8 times from a rarely visited state;"
      " the Dirichlet prior's parameters are pseudo-counts and the MLE has none")
print("     counts      MLE row                 Dir(1,1,1) posterior mean"
      "        log-lik of a held-out calm->crisis step")
for counts in ([6, 2, 0], [5, 3, 0], [8, 0, 0], [4, 3, 1]):
    c = np.array(counts)
    mle = c / c.sum()
    pos = (c + 1) / (c.sum() + 3)
    ll = -np.inf if mle[2] == 0 else np.log(mle[2])
    print(f"    {str(counts):10s}  {np.array2string(mle, precision=4, floatmode='fixed')}"
          f"   {np.array2string(pos, precision=4, floatmode='fixed')}"
          f"   {ll:10.4f} against {np.log(pos[2]):8.4f}")
# =>   the mean of 100bp returns, with the volatility known and with it estimated; the second posterior is Student-t and the first pretends it is not
#            n   t crit   normal crit   width ratio   coverage, sigma known   coverage, sigma estimated
#            5   2.7764        1.9600        1.4166                  0.8765                       0.9495
#           10   2.2622        1.9600        1.1542                  0.9181                       0.9500
#           30   2.0452        1.9600        1.0435                  0.9415                       0.9510
#          100   1.9842        1.9600        1.0124                  0.9480                       0.9505
#          250   1.9695        1.9600        1.0049                  0.9508                       0.9517
#      a three-regime transition row observed 8 times from a rarely visited state; the Dirichlet prior's parameters are pseudo-counts and the MLE has none
#         counts      MLE row                 Dir(1,1,1) posterior mean        log-lik of a held-out calm->crisis step
#        [6, 2, 0]   [0.7500 0.2500 0.0000]   [0.6364 0.2727 0.0909]         -inf against  -2.3979
#        [5, 3, 0]   [0.6250 0.3750 0.0000]   [0.5455 0.3636 0.0909]         -inf against  -2.3979
#        [8, 0, 0]   [1.0000 0.0000 0.0000]   [0.8182 0.0909 0.0909]         -inf against  -2.3979
#        [4, 3, 1]   [0.5000 0.3750 0.1250]   [0.4545 0.3636 0.1818]      -2.0794 against  -1.7047
```

The first table is the normal–inverse-gamma pair, which is what one gets by refusing to pretend the volatility is known. Integrating the unknown scale out of the joint posterior leaves a Student-$t$ marginal for the mean with $n-1$ degrees of freedom, and the entire practical content of that is the critical value: $2.7764$ against the normal's $1.9600$ at five observations, a $1.4166$ ratio, narrowing to $1.0435$ by thirty and $1.0049$ by two hundred and fifty. Coverage tells the same story from the other side. An analyst who plugs the sample volatility into a normal posterior — which is what a known-variance conjugate update does — achieves $0.8765$, $0.9181$, $0.9415$, $0.9480$ and $0.9508$ against a nominal $0.95$, while the Student-$t$ posterior delivers $0.9495$, $0.9500$, $0.9510$, $0.9505$ and $0.9517$. **Estimating a volatility and then treating it as known is the single most common way a correctly specified Bayesian calculation is made wrong, and at the sample sizes on which new strategies are judged it costs eight points of coverage.**

The second table is the Dirichlet–multinomial pair and it is about zero rather than about width. A transition row estimated by maximum likelihood from eight visits to a rare state assigns probability exactly zero to any transition not yet observed. In a hidden Markov model — the construction [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md) develops and which the course lesson fits to regimes — that zero is not a small number, it is an absorbing claim: the log-likelihood of any future path containing that transition is $-\infty$, and the model has asserted that a move from calm to crisis is not merely unlikely but impossible. A $\text{Dir}(1,1,1)$ prior, worth three pseudo-observations in total, converts the row $[6,2,0]$ into $[0.6364,0.2727,0.0909]$ and the $-\infty$ into $-2.3979$. The row $[8,0,0]$, which the MLE reads as a state that never leaves, becomes $[0.8182,0.0909,0.0909]$.

That $0.0909$ is not a claim to have observed anything. It is one imaginary observation, and the pseudo-count reading of section 1 is what makes it defensible: an analyst who says "I have never seen this state go to crisis, but I have only watched it eight times" has said something precise, and the Dirichlet parameter is the arithmetic of that sentence. The MLE's alternative is to say the same evidence proves impossibility.

## A Mixture of Conjugate Priors Is Conjugate, Which Restores the Nonlinearity the Family Is Accused of Lacking

Section 1 closed on a genuine limitation: a conjugate posterior mean shrinks by a factor fixed before the data arrived. The escape is cheaper than it looks and is itself conjugate.

??? note "Proof that a finite mixture of conjugate priors is conjugate, with component weights updated by their own marginal likelihoods, so the posterior mean becomes a nonlinear function of the data"

    Let $\pi(\theta)=\sum_{k}w_k\pi_k(\theta)$ with $\sum_k w_k=1$ and each $\pi_k$ conjugate to $f(\cdot\mid\theta)$. Then
    $$\pi(\theta\mid x)\ \propto\ \sum_k w_k\,\pi_k(\theta)f(x\mid\theta)=\sum_k w_k\,m_k(x)\,\pi_k(\theta\mid x),\qquad m_k(x)=\int\pi_k(\theta)f(x\mid\theta)\,\mathrm{d}\theta,$$
    where each $\pi_k(\cdot\mid x)$ is the conjugate update of component $k$. Normalizing,
    $$\pi(\theta\mid x)=\sum_k w_k'\,\pi_k(\theta\mid x),\qquad w_k'=\frac{w_k m_k(x)}{\sum_j w_j m_j(x)}.$$
    The posterior is a mixture of the same number of components from the same family, so closure holds; the component parameters update by the addition rule of section 1, and the weights update in proportion to each component's marginal likelihood — the same quantity that [Bayesian Model Comparison](06-bayesian-model-comparison.md) uses to compare models, here comparing explanations inside one model.

    The consequence for the posterior mean is the point:
    $$\mathbb{E}[\theta\mid x]=\sum_k w_k'(x)\,\mathbb{E}_k[\theta\mid x],$$
    a weighted average of linear shrinkages whose *weights depend on the data*. That is not linear in $x$, so the Diaconis–Ylvisaker characterization is not contradicted — a mixture of conjugate priors is not itself a conjugate prior in their sense — and the family has bought exactly the behaviour it is usually said to preclude. **Finite mixtures of conjugate priors are dense in the space of priors on the natural parameter, so conjugacy restricts the shape of a belief not at all and restricts only how many components one is willing to carry, which is a computational budget rather than a modelling assumption.**

    The load-bearing distinction is between the two ways of paying for flexibility: numerical integration pays in evaluations at every summary, as [Posterior Distributions](03-posterior-distributions.md) measures, while a mixture pays a fixed cost of $K$ closed-form updates and keeps every summary exact.

Two components are enough to change the behaviour completely:

```python
import numpy as np

sig, n, w, t1, t2 = 0.010, 250, 0.90, 0.0002, 0.0030
se = sig / np.sqrt(n)

print(f"  a spike-and-slab prior on a daily edge: {w:.0%} weight on N(0, {t1 * 1e4:.0f}bp)"
      f" and {1 - w:.0%} on N(0, {t2 * 1e4:.0f}bp), against the single normal with the"
      f" same prior variance; {n} days, standard error {se * 1e4:.4f}bp")
tot = np.sqrt(w * t1 ** 2 + (1 - w) * t2 ** 2)
lam_s = se ** 2 / (se ** 2 + tot ** 2)
print(f"    the matched single normal has sd {tot * 1e4:.4f}bp and shrinks by a constant"
      f" {1 - lam_s:.4f} whatever it sees")
print("     observed, bp   t-stat   single normal, bp   mixture, bp   slab weight"
      "   mixture shrinkage")
for xb_bp in (0.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    xb = xb_bp * 1e-4
    single = (1 - lam_s) * xb
    v = np.array([t1, t2]) ** 2 + se ** 2
    lw = np.log([w, 1 - w]) - 0.5 * np.log(v) - 0.5 * xb ** 2 / v
    p = np.exp(lw - lw.max())
    p /= p.sum()
    post = (p * (np.array([t1, t2]) ** 2 / v) * xb).sum()
    print(f"    {xb_bp:12.1f}   {xb / se:6.3f}   {single * 1e4:17.4f}"
          f"   {post * 1e4:11.4f}   {p[1]:11.4f}   {post / xb if xb else 0.0:17.4f}")
# =>   a spike-and-slab prior on a daily edge: 90% weight on N(0, 2bp) and 10% on N(0, 30bp), against the single normal with the same prior variance; 250 days, standard error 6.3246bp
#        the matched single normal has sd 9.6747bp and shrinks by a constant 0.7006 whatever it sees
#         observed, bp   t-stat   single normal, bp   mixture, bp   slab weight   mixture shrinkage
#                 0.0    0.000              0.0000        0.0000        0.0235              0.0000
#                 2.0    0.316              1.4012        0.2243        0.0245              0.1121
#                 4.0    0.632              2.8024        0.4600        0.0278              0.1150
#                 8.0    1.265              5.6048        1.0453        0.0459              0.1307
#                16.0    2.530             11.2096        5.3068        0.2779              0.3317
#                32.0    5.060             22.4192       30.6207        0.9994              0.9569
```

The two priors carry the same total variance, so any comparison of "how much information the prior contains" scores them equally. They behave nothing alike. The single normal retains a constant $0.7006$ of whatever it is shown — an observed edge of two basis points becomes $1.4012$, and one of thirty-two becomes $22.4192$, the same fraction in both cases, which is the linearity the first proof guarantees.

The mixture retains $0.1121$ at an observed two basis points and $0.9569$ at thirty-two. At small signals it is far more skeptical than the single normal, because the spike component dominates and the spike says edges are tiny; at large signals it is far more permissive, because the slab's marginal likelihood has overtaken the spike's and the weight has moved from $0.0235$ to $0.9994$. The transition is visible in the middle rows: at a $t$-statistic of $2.530$ the slab weight is $0.2779$ and the mixture retains $0.3317$, so it is still shrinking hard at a signal a frequentist would call significant.

**That profile — crush the small, pass the large, and be undecided in between — is what a research desk means when it says it wants to be skeptical without being blind, and it is available in closed form at the cost of carrying two numbers instead of one.** It is also the honest version of the skeptical prior [Prior Distributions](02-prior-distributions.md) examined: a single tight normal applies its skepticism uniformly, including to the one strategy in a hundred that genuinely works.

## Conjugacy Fixes the Tail, and the Tail Decides How a Posterior Answers a Surprise

The remaining cost is not about the centre of the prior at all. It is about what happens when the prior and the data disagree, which is the situation every risk system is built for.

??? note "Proof that a normal prior's posterior mean is pulled from the data by an amount growing without bound in the conflict, while a heavy-tailed prior's posterior mean converges to the observation, which is O'Hagan's conflict resolution"

    Take one observation $x$ with known error scale $\sigma$ and a prior for $\mu$ centred at zero with scale $\tau$. Under a normal prior the posterior mean is $\mathbb{E}[\mu\mid x]=(1-\lambda)x$ with $\lambda=\sigma^{2}/(\sigma^{2}+\tau^{2})$, so the *pull*
    $$x-\mathbb{E}[\mu\mid x]=\lambda x$$
    is proportional to $x$ and diverges as the conflict grows. The posterior variance is $(1-\lambda)\sigma^{2}$, a constant: it does not depend on $x$ at all, so the model reports identical confidence whether the observation agrees with the prior or contradicts it by twenty standard errors.

    Now let the prior have regularly varying tails, $\pi(\mu)\sim c|\mu|^{-(\nu+1)}$ as $|\mu|\to\infty$, which a Student-$t$ with $\nu$ degrees of freedom satisfies. The unnormalized posterior is $\pi(\mu)\phi((x-\mu)/\sigma)$. For large $x$, the mass concentrates where the likelihood is appreciable, $\mu=x+O(\sigma)$, and there $\pi(\mu)\approx\pi(x)(1+O(\sigma/x))$ is nearly constant across the likelihood's support because a power law is slowly varying at that scale. The prior therefore acts locally like a flat prior, and
    $$\mathbb{E}[\mu\mid x]-x\longrightarrow 0,\qquad \mathrm{Var}(\mu\mid x)\longrightarrow\sigma^{2}.$$
    The prior has been discarded. This is Dawid's and O'Hagan's result on outlier-proneness: whichever of the two information sources has the lighter tails is the one that wins a conflict, so a normal prior against a normal likelihood splits the difference forever, a heavy-tailed prior yields to a sharp likelihood, and a heavy-tailed likelihood — the returns model of [Posterior Distributions](03-posterior-distributions.md) — yields to a sharp prior by discounting the outlier instead.

    **The load-bearing fact is that a prior's tail is not a detail of its shape but a specification of how much conflict it will tolerate before conceding, and conjugate families fix that behaviour as a side effect of being chosen for their algebra.** Nothing about wanting closed-form updates implies wanting a belief that never yields.

The two behaviours diverge at conflicts a market produces regularly:

```python
import numpy as np
from scipy import stats

tau, se, nu = 3.0, 1.0, 3
g = np.linspace(-400.0, 400.0, 4_000_001)

print(f"  one observation with standard error {se:.0f}, against a normal prior and a"
      f" Student-t prior with {nu} degrees of freedom, both centred at 0 with scale"
      f" {tau:.0f}; every entry is in standard errors")
print("     observation   normal posterior mean   normal sd   t posterior mean   t sd"
      "   normal pull   t pull")
for x in (1.0, 2.0, 4.0, 6.0, 10.0, 20.0):
    lam = se ** 2 / (se ** 2 + tau ** 2)
    nm, nsd = (1 - lam) * x, np.sqrt((1 - lam) * se ** 2)
    lw = stats.t.logpdf(g, nu, scale=tau) + stats.norm.logpdf(x, g, se)
    p = np.exp(lw - lw.max())
    p /= p.sum()
    m = p @ g
    print(f"    {x:11.1f}   {nm:21.4f}   {nsd:9.4f}   {m:16.4f}"
          f"   {np.sqrt(p @ g ** 2 - m ** 2):5.4f}   {x - nm:11.4f}   {x - m:6.4f}")
# =>   one observation with standard error 1, against a normal prior and a Student-t prior with 3 degrees of freedom, both centred at 0 with scale 3; every entry is in standard errors
#         observation   normal posterior mean   normal sd   t posterior mean   t sd   normal pull   t pull
#                1.0                  0.9000      0.9487             0.8832   0.9424        0.1000   0.1168
#                2.0                  1.8000      0.9487             1.7805   0.9531        0.2000   0.2195
#                4.0                  3.6000      0.9487             3.6521   0.9816        0.4000   0.3479
#                6.0                  5.4000      0.9487             5.6222   1.0010        0.6000   0.3778
#               10.0                  9.0000      0.9487             9.6790   1.0091        1.0000   0.3210
#               20.0                 18.0000      0.9487            19.8107   1.0042        2.0000   0.1893
```

The two priors are centred identically and scaled identically. Below four standard errors they are barely distinguishable — posterior means of $0.9000$ against $0.8832$ and $1.8000$ against $1.7805$ — which is the general rule that tail choices are invisible until something happens.

Then they separate, and the direction is the one that matters. The normal prior's pull grows without limit: $0.1000$, $0.2000$, $0.4000$, $0.6000$, $1.0000$ and $2.0000$ standard errors. At an observation twenty standard errors from the prior centre it still insists the truth is at $18.0000$, having moved the answer two full standard errors on the strength of a belief the data has comprehensively refuted. The Student-$t$ prior's pull rises to a maximum of $0.3778$ near six standard errors and then *falls* — $0.3210$ at ten and $0.1893$ at twenty — as the prior concedes and the posterior mean converges to the observation.

The standard deviation columns say the same thing about confidence. The normal prior reports $0.9487$ at every row, including the row where it is wrong by two standard errors: it never widens, because its variance does not depend on the data. The Student-$t$ posterior widens from $0.9424$ to $1.0091$, converging on the likelihood's own $1.0000$ — the correct behaviour, which is to stop claiming the prior added information once the prior has been overruled. **A conjugate normal prior is a commitment to split every future disagreement in a ratio fixed today, and the one situation in which that is certainly wrong is the situation in which the disagreement is large.**

## What a Conjugate Prior Assumes Is Rarely What Its User Would Defend

The identification promised at the top can now be made exact. Shrinking a sample covariance $S$ toward a target $F$ as $\hat\Sigma=\delta F+(1-\delta)S$ is the posterior mean of $\Sigma$ under an inverse-Wishart prior centred on $F$, with prior degrees of freedom $\nu_0$ satisfying $\delta=\nu_0/(\nu_0+n)$ to first order — the same $\kappa/(\kappa+n)$ weight section 1 derived, with $\nu_0$ the pseudo-observation count. The course lesson's measured intensity of $1.85\%$ at five hundred and four observations therefore corresponds to a prior worth roughly nine and a half imaginary days, which is a very weak belief, and the lesson's own diagnosis — the estimator "is doing its job perfectly and the job is nearly vacuous here" — is exactly what a nine-day prior against five hundred days of data must produce. Ledoit and Wolf choose $\delta$ analytically to minimize expected squared error rather than by asserting a belief, which is the empirical-Bayes route [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md) treats, but the object being chosen is a conjugate prior's strength either way.

The assumption riding along is the one nobody would defend if asked. An inverse-Wishart prior is conjugate to a *Gaussian* likelihood, so the whole construction assumes returns whose tails [Posterior Distributions](03-posterior-distributions.md) measured to be badly wrong — and the failure lands on covariance estimates during exactly the stress episodes when correlations move and the estimate matters. The lesson's finding that constraints beat shrinkage, the Jagannathan–Ma equivalence taking Sharpe from $0.377$ to $0.444$, is consistent with this: a no-short constraint implies a shrinkage that is not derived from a Gaussian conjugate argument and does not inherit its tail assumption.

Where genuine conjugacy is unavailable, semi-conjugacy usually is. Placing independent normal and inverse-gamma priors on a mean and a variance is not jointly conjugate, but each full conditional is, which is precisely the structure a Gibbs sampler consumes — the connection developed in [Gibbs Sampling](../part-17-statistical-computing/06-gibbs-sampling.md). Conjugacy therefore does not disappear when models get realistic; it retreats from the joint distribution into the conditionals, and remains the reason the computation is tractable at all.

!!! note "Conjugate, semi-conjugate, mixture-of-conjugates, reference and empirical-Bayes priors are five constructions with the same closed-form arithmetic and five different sources for the numbers in them"
    They are routinely reported as one thing. A **conjugate** prior is closed under updating for the whole parameter vector and its hyperparameters are pseudo-observations, which is the only reading on this list that makes a prior's strength commensurable with a sample size. A **semi-conjugate** prior is conjugate in each full conditional and not jointly, so it keeps the closed forms and loses the single-step update — the structure Gibbs sampling exists to exploit. A **mixture of conjugates** is closed under updating with data-dependent weights, and section 3 shows it escapes the linearity that characterizes the first entry while remaining exact. A **reference** prior such as Jeffreys' is chosen for invariance and is sometimes conjugate by coincidence, as $\pi(\sigma)\propto1/\sigma$ is the limit of an inverse-gamma; the coincidence is not a justification, as [Prior Distributions](02-prior-distributions.md) argues. An **empirical-Bayes** prior has the conjugate form with hyperparameters estimated from the same data they will be applied to, which is what Ledoit–Wolf does and what the hierarchical construction of [Prior Distributions](02-prior-distributions.md) does honestly by giving the estimate a standard error. Quoting a conjugate prior's pseudo-count as evidence of modesty when the hyperparameters were fitted to the data is the error this list exists to prevent.

!!! warning "A conjugate family is chosen for the tractability of its centre and silently specifies its tail, so the assumption that decides every conflict is the one that was never part of the decision"
    The choices measured here were all made for algebraic reasons and all bind somewhere else. Treating an estimated volatility as known keeps the normal–normal update in closed form and delivers coverage of $0.8765$ at five observations against the Student-$t$ posterior's $0.9495$. A maximum likelihood transition row keeps the arithmetic prior-free and asserts a log-likelihood of $-\infty$ for a transition seen zero times in eight visits, where one pseudo-observation gives $-2.3979$. A normal prior keeps the posterior mean linear and, at an observation twenty standard errors out, holds the answer $2.0000$ standard errors from the data while reporting the same $0.9487$ standard deviation it reports when nothing is wrong — where a Student-$t$ prior of identical scale pulls $0.1893$ and widens to $1.0091$. **The free diagnostic is to refit every conjugate posterior you intend to publish under a heavy-tailed prior of the same location and scale — a Student-$t$ with three or four degrees of freedom in place of the normal, an inverse-gamma with a smaller shape in place of a tight one — and to report both posterior means whenever they differ by more than a tenth of a posterior standard deviation, because that difference is the size of the conflict your conjugate family resolved without telling you.** For a mixture or a one-dimensional grid it costs one extra closed-form update.

## Closed Form Is a Property of the Algebra and Not of the Belief

This page established that conjugacy is the closure property of the exponential family rather than a coincidence, with the update adding sufficient statistics and incrementing a count so hyperparameters read as pseudo-observations; that Diaconis and Ylvisaker's linearity characterizes the family, fixing the shrinkage weight before any data arrives; that the two least glamorous entries in the catalogue carry most of the practical value, the normal–inverse-gamma pair raising the critical value from $1.9600$ to $2.7764$ at five observations and coverage from $0.8765$ to $0.9495$, and one Dirichlet pseudo-count converting a $-\infty$ log-likelihood into $-2.3979$ on a transition never observed; that a mixture of conjugate priors is conjugate with weights updated by marginal likelihood, retaining $0.1121$ of an observed two-basis-point edge and $0.9569$ of a thirty-two-basis-point one where a single normal of identical total variance retains $0.7006$ of both, with the slab weight travelling from $0.0235$ to $0.9994$; and that tail behaviour decides conflicts, a normal prior's pull growing $0.1000$, $0.2000$, $0.4000$, $0.6000$, $1.0000$, $2.0000$ standard errors at a fixed posterior standard deviation of $0.9487$, while a Student-$t$ prior's pull peaks at $0.3778$ and falls to $0.1893$ as its posterior standard deviation widens to $1.0091$.

The shape shared by all three exhibits is that a conjugate family answers a question about tractability and is then relied on for an answer about belief. The pseudo-count reading makes prior strength honest and readable, which is more than any other construction in this part manages; the linearity and the tail behaviour that come attached were never chosen, are never printed, and decide the cases that matter. That is the part's recurring failure in its most concrete form: the input doing the work is the one nobody selected.

Everything so far has treated the data as a single batch. The pseudo-count reading of section 1 suggests something stronger — that the update should be applicable one observation at a time, with yesterday's posterior serving as today's prior, and that the order should not matter. It is true, it is a theorem, and the conditions under which it fails are the conditions under which a trading desk actually operates. That is [Bayesian Updating](05-bayesian-updating.md).

**A conjugate prior states how many observations your belief is worth, which is the most honest thing any prior on this list does, and then quietly states how much evidence it will take before that belief gives way — a number nobody chose and the output never shows.**
