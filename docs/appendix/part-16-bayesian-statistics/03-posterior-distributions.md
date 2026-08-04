# Posterior Distributions

A posterior is usually described as what you believe after seeing the data, and the description is right in a way that hides the two things worth knowing about it. The first is arithmetic: the posterior is available immediately up to a multiplicative constant and the constant is the only expensive part, which is why the cheapest summary — the mode — is the one everyone computes and the one that answers no question anybody asked. Below, a two-parameter posterior with no closed form is normalized two entirely different ways, by quadrature over $361{,}201$ grid evaluations and by importance sampling, and they agree on the posterior mean to $5.5687$ against $5.5682$ basis points; the same grid in ten dimensions would need more evaluations than there are atoms in the observable universe. The second is that concentration is not accuracy. A posterior for a strategy's mean built on serially correlated days reports a standard deviation $2.032$ times too small and covers the truth on $0.6626$ of runs instead of $0.95$; a Gaussian model asked for a one-per-cent daily loss on Student-$t$ returns answers $-2.2823\%$ when the truth is $-2.6216\%$, with a posterior standard deviation of $0.0847\%$ — a bias four times its own stated uncertainty — and covers the truth $0.1542$ of the time while looking exactly as confident as the correctly specified case that covers it $0.9480$ of the time.

This page covers the posterior as an unnormalized object and the $\propto$ convention that exploits it, what the normalizing constant costs and how it is obtained when no closed form exists, the Bernstein–von Mises theorem and the sense in which a posterior becomes a sampling distribution, and what happens to all of it when the model is wrong. It constructs no prior and argues for none, which is [Prior Distributions](02-prior-distributions.md); it proves no closure and derives no conjugate family, which is [Conjugate Priors](04-conjugate-priors.md); it takes no posterior mode and derives no penalized-likelihood identity, which is [Maximum A Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md); it minimizes no expected loss and selects no summary, which is [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md); it quotes no credible interval as an inferential object beside a confidence interval, which is [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md); it builds no chain whose stationary distribution is the posterior, which is [Markov Chain Monte Carlo](../part-17-statistical-computing/04-markov-chain-monte-carlo.md); it derives no importance-sampling estimator from first principles, which is [Importance Sampling](../part-09-monte-carlo-methods/04-importance-sampling.md); it updates nothing sequentially, which is [Bayesian Updating](05-bayesian-updating.md); it compares no models and computes no Bayes factor, which is [Bayesian Model Comparison](06-bayesian-model-comparison.md); and it never treats the width of a posterior as evidence about anything except the model that produced it.

The trading stake is a promotion gate catching precisely this failure and naming it in one clause. [Production ML](../../part-07-machine-learning/05-production-ml.md) scores a challenger against a champion and prints `challenger-2020H1 (through 2020-06): AUC 0.581, Brier 0.2660 -> HOLD — fails 'Brier no worse than champion'` — a model that beats the incumbent on ranking by five points and loses on calibration, which the lesson reads as "its probabilities are more confident than its accuracy justifies (a model that drank 2020's chaos and came out swaggering)." That is section 4 in operational dress. The gate exists because a discrimination metric and a calibration metric can disagree, and a posterior reports only the analogue of the first.

## The Posterior Is Known Up to a Constant, and Recovering That Constant Is the Only Expensive Step

The single most useful fact about a posterior is that the interesting part is free. Prior times likelihood is a product of two things already in hand; the denominator is an integral over the whole parameter space, and it is the only object in Bayesian inference that ever requires real computational effort.

??? note "Proof that the posterior mode requires no normalizing constant while every expectation, probability and quantile does, so the cheapest summary is the one that answers no decision problem"

    Write the unnormalized posterior as $q(\theta)=\pi(\theta)f(x\mid\theta)$ and the normalizer as $Z=\int q(\theta)\,\mathrm{d}\theta$, so that $\pi(\theta\mid x)=q(\theta)/Z$. The notation $\pi(\theta\mid x)\propto\pi(\theta)f(x\mid\theta)$, whose general convention [Mathematical Notation](../part-01-mathematical-foundations/03-mathematical-notation.md) sets out and which [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md) uses at the level of events, is exactly the statement that $Z$ has been suppressed because it does not depend on $\theta$.

    For the mode, $\arg\max_\theta q(\theta)/Z=\arg\max_\theta q(\theta)$ for any $Z>0$, since dividing by a positive constant preserves the location of a maximum. The mode is therefore computable by optimization on $q$ alone, which is the observation [Maximum A Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md) is built on.

    For anything else the constant survives. An expectation is
    $$\mathbb{E}[g(\theta)\mid x]=\frac{\int g(\theta)q(\theta)\,\mathrm{d}\theta}{\int q(\theta)\,\mathrm{d}\theta},$$
    a ratio of two integrals over the same space; a probability is the case $g=\mathbb{1}_A$; a quantile inverts the function $t\mapsto\int_{-\infty}^{t}q/Z$, which requires $Z$ to know where the mass ends. Note that the ratio form means one *may* avoid computing $Z$ separately — any method delivering both integrals with the same weights gives the answer, which is what self-normalized importance sampling exploits and why [Importance Sampling](../part-09-monte-carlo-methods/04-importance-sampling.md) treats targets known only up to a constant as the normal case rather than a special one.

    The cost is dimensional. Evaluating $Z$ on a product grid with $m$ points per axis costs $m^{d}$ likelihood evaluations, so a resolution that is generous in two dimensions is impossible in ten: at $m=601$ the counts are $3.6\times10^{5}$ and $6.5\times10^{27}$. Rejection sampling fails for the reason [Rejection Sampling](../part-09-monte-carlo-methods/05-rejection-sampling.md) gives, importance sampling degrades as the proposal and target separate, and what remains is to give up on $Z$ entirely and construct a chain whose stationary distribution is the posterior — which is [Markov Chain Monte Carlo](../part-17-statistical-computing/04-markov-chain-monte-carlo.md) and is not derived here.

    **The load-bearing asymmetry is that the summary requiring no integral is the summary minimizing no expected loss, so the computational convenience of the mode and its decision-theoretic emptiness are the same fact viewed from two sides.**

## Quadrature and Self-Normalized Importance Sampling Agree to Four Decimals in Two Dimensions, and Neither Survives Ten

A Student-$t$ likelihood with unknown location and scale has no conjugate prior and no closed-form posterior, which makes it the smallest realistic test of the machinery — and heavy tails are the correct model for daily returns rather than a pathological choice:

```python
import numpy as np
from scipy import stats, special

rng = np.random.default_rng(16031)
n, nu, mu_t, sd_t = 250, 4, 0.0004, 0.010
s_t = sd_t / np.sqrt(nu / (nu - 2))
x = stats.t.rvs(nu, loc=mu_t, scale=s_t, size=n, random_state=rng)

mu_g = np.linspace(-0.0060, 0.0060, 601)
ls_g = np.linspace(np.log(0.0010), np.log(0.0400), 601)
dmu, dls = mu_g[1] - mu_g[0], ls_g[1] - ls_g[0]


def loglik(mu, ls):
    """Student-t log-likelihood; a Jeffreys 1/s prior is flat in log s, so it adds nothing."""
    return stats.t.logpdf(x[:, None], nu, loc=mu, scale=np.exp(ls)).sum(0)


M, L = np.meshgrid(mu_g, ls_g, indexing="ij")
lp = loglik(M.ravel(), L.ravel()).reshape(M.shape)
w = np.exp(lp - lp.max())
Z = w.sum() * dmu * dls
post = w / w.sum()
mmu = (post.sum(1) * mu_g).sum()
vmu = (post.sum(1) * mu_g ** 2).sum() - mmu ** 2
grid = (mmu, np.sqrt(vmu), post[mu_g > 0].sum(), np.log(Z) + lp.max())

print(f"  {n} returns from a Student-t with {nu} degrees of freedom, no conjugate form;"
      f" the posterior is known up to a constant and every summary but the mode needs it")
print(f"    grid, {mu_g.size} x {ls_g.size} = {mu_g.size * ls_g.size:,} evaluations"
      f"     mean {grid[0] * 1e4:7.4f}bp   sd {grid[1] * 1e4:6.4f}bp"
      f"   P(mu>0) {grid[2]:.4f}   log Z {grid[3]:10.4f}")
print("     importance draws   mean, bp    sd, bp   P(mu>0)      log Z   ESS   ESS/draws")
for d in (1_000, 10_000, 100_000, 1_000_000):
    q_mu = rng.normal(x.mean(), 2 * x.std() / np.sqrt(n), d)
    q_ls = rng.normal(np.log(x.std()), 0.35, d)
    lq = (stats.norm.logpdf(q_mu, x.mean(), 2 * x.std() / np.sqrt(n))
          + stats.norm.logpdf(q_ls, np.log(x.std()), 0.35))
    lw = loglik(q_mu, q_ls) - lq
    ww = np.exp(lw - lw.max())
    p = ww / ww.sum()
    m = p @ q_mu
    print(f"    {d:16,d}   {m * 1e4:8.4f}  {np.sqrt(p @ q_mu ** 2 - m ** 2) * 1e4:8.4f}"
          f"   {p[q_mu > 0].sum():7.4f}   {special.logsumexp(lw) - np.log(d):10.4f}"
          f"   {1 / (p @ p):5.0f}   {1 / (p @ p) / d:9.4f}")
# =>   250 returns from a Student-t with 4 degrees of freedom, no conjugate form; the posterior is known up to a constant and every summary but the mode needs it
#        grid, 601 x 601 = 361,201 evaluations     mean  5.5687bp   sd 5.3801bp   P(mu>0) 0.8457   log Z   817.3645
#         importance draws   mean, bp    sd, bp   P(mu>0)      log Z   ESS   ESS/draws
#                   1,000     5.3099    5.1793    0.8472     817.4549     115      0.1147
#                  10,000     5.7091    5.3678    0.8530     817.3987    1172      0.1172
#                 100,000     5.5309    5.3772    0.8484     817.3641   11367      0.1137
#               1,000,000     5.5682    5.3800    0.8507     817.3662   113751      0.1138
```

The two routes share nothing but the unnormalized posterior. Quadrature lays a deterministic lattice over the parameter space and sums; importance sampling draws from a convenient proposal and reweights by the ratio of target to proposal, never forming a grid at all. Their agreement is therefore a real check rather than a restatement: posterior mean $5.5687$ against $5.5682$ basis points at a million draws, standard deviation $5.3801$ against $5.3800$, $P(\mu>0)$ of $0.8457$ against $0.8507$, and a log normalizing constant of $817.3645$ against $817.3662$.

The convergence is the ordinary Monte Carlo rate and the diagnostic column says why one should not be complacent about it. Effective sample size runs $115$, $1{,}172$, $11{,}367$ and $113{,}751$ against nominal draw counts of a thousand up to a million — a stable efficiency near $0.114$, meaning roughly nine draws are needed for every one that counts. That ratio is a property of how well the proposal matches the target, and it is respectable here because the target is two-dimensional and nearly Gaussian. It degrades geometrically as dimension rises, for the reason [Importance Sampling](../part-09-monte-carlo-methods/04-importance-sampling.md) gives: the weights become dominated by a handful of draws and the estimator's variance is carried by samples the proposal was never going to produce.

**Both methods are exact in the limit and both are unusable past a handful of parameters, which is the entire reason the next part exists.** The honest summary of this section is that the posterior of a two-parameter model is a solved problem by at least two independent routes, and that a hierarchical model with fifty group means — the construction [Prior Distributions](02-prior-distributions.md) recommended — has fifty-two parameters and is solved by neither.

## With Enough Data the Posterior Becomes the Sampling Distribution of the Estimator, Which Is Why the Paradigms Stop Disagreeing Numerically Long Before They Stop Disagreeing

The convergence noted at the end of [The Bayesian Framework](01-bayesian-framework.md) has a precise form, and its precise form is also where the trouble in section 4 enters.

??? note "Proof that the posterior tends to $N(\hat\theta, I(\theta_0)^{-1}/n)$ in total variation and the prior's contribution decays at rate $1/n$, which is the Bernstein–von Mises theorem"

    Expand the log-likelihood around the maximum likelihood estimate $\hat\theta$. Writing $\ell_n(\theta)=\sum_i\log f(x_i\mid\theta)$ and using $\ell_n'(\hat\theta)=0$,
    $$\ell_n(\theta)=\ell_n(\hat\theta)-\tfrac{1}{2}(\theta-\hat\theta)^{\top}\big(-\ell_n''(\hat\theta)\big)(\theta-\hat\theta)+R_n,$$
    with $-\ell_n''(\hat\theta)=nI(\theta_0)+o_P(n)$ by the law of large numbers and $R_n$ of order $n\|\theta-\hat\theta\|^{3}$. On the scale that matters, $\|\theta-\hat\theta\|=O_P(n^{-1/2})$, the quadratic term is $O(1)$ and the remainder is $O_P(n^{-1/2})$. Therefore
    $$\pi(\theta\mid x)\ \propto\ \pi(\theta)\exp\!\big\{-\tfrac{n}{2}(\theta-\hat\theta)^{\top}I(\theta_0)(\theta-\hat\theta)\big\}\big(1+o_P(1)\big).$$
    A prior continuous and positive at $\theta_0$ is locally constant on a neighbourhood of width $n^{-1/2}$, since $\pi(\hat\theta+u/\sqrt n)=\pi(\theta_0)(1+O(n^{-1/2}))$, so it factors out. What remains is a Gaussian kernel, and the theorem states the convergence in total variation:
    $$\big\|\pi(\cdot\mid x)-N\big(\hat\theta,\ I(\theta_0)^{-1}/n\big)\big\|_{TV}\xrightarrow{\ P\ }0.$$

    Two readings follow and both matter. Numerically, the posterior and the sampling distribution of the maximum likelihood estimator become the same Gaussian, so a credible interval and a Wald interval coincide to the order of the approximation — which is why the course lesson's `[0.5281, 0.5530]` and `[0.5282, 0.5530]` agree. Interpretively, they remain statements about different things, and the theorem is silent on that: it equates two densities, not two meanings.

    The prior's influence is $O(1/n)$ rather than $O(n^{-1/2})$, because a smooth prior shifts the posterior mean by approximately $\pi'(\theta_0)/(n\pi(\theta_0))\cdot I(\theta_0)^{-1}$ while the posterior's own width is $O(n^{-1/2})$; the ratio of shift to width is therefore $O(n^{-1/2})$ and vanishes. **The conditions are the load-bearing part: the model must contain the truth, the parameter must be interior and identified, the information matrix must be non-singular, and the prior must be positive at the truth.** A prior assigning zero density to a region assigns zero posterior density there forever, no matter what the data says, and a truth outside the model class is what section 4 is about.

The rate and the conditions are both checkable:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(16033)
theta, reps = 0.54, 4_000
g = np.linspace(1e-6, 1 - 1e-6, 20_001)
dg = g[1] - g[0]
PRIORS = (("flat Beta(1,1)", 1.0, 1.0), ("Jeffreys Beta(.5,.5)", 0.5, 0.5),
          ("skeptical Beta(60,60)", 60.0, 60.0))

print(f"  hit rate with a true value of {theta}; the posterior is compared with the normal"
      f" Bernstein-von Mises approximation N(theta-hat, theta-hat(1-theta-hat)/n),"
      f" {reps:,} datasets")
print("         n   total variation   prior shift, bp   shift x n   P(>1/2) flat"
      "   P(>1/2) skeptical   P(>1/2) normal")
for n in (25, 100, 1000, 10000):
    k = rng.binomial(n, theta, reps)
    th = k / n
    nrm = stats.norm(th[:, None], np.sqrt(th * (1 - th) / n)[:, None])
    means, tails = {}, {}
    for nm, a, b in PRIORS:
        po = stats.beta(a + k[:, None], b + n - k[:, None])
        means[nm] = ((a + k) / (a + b + n)).mean()
        tails[nm] = po.sf(0.5).mean()
        if nm == "flat Beta(1,1)":
            tv = 0.5 * np.abs(po.pdf(g) - nrm.pdf(g)).sum(1).mean() * dg
    shift = abs(means["skeptical Beta(60,60)"] - means["flat Beta(1,1)"]) * 1e4
    print(f"    {n:6d}   {tv:15.4f}   {shift:15.2f}   {shift * n:9.0f}"
          f"   {tails['flat Beta(1,1)']:12.4f}   {tails['skeptical Beta(60,60)']:18.4f}"
          f"   {nrm.sf(0.5).mean():14.4f}")
# =>   hit rate with a true value of 0.54; the posterior is compared with the normal Bernstein-von Mises approximation N(theta-hat, theta-hat(1-theta-hat)/n), 4,000 datasets
#             n   total variation   prior shift, bp   shift x n   P(>1/2) flat   P(>1/2) skeptical   P(>1/2) normal
#            25            0.0293            292.81        7320         0.6075               0.5593           0.6101
#           100            0.0082            218.34       21834         0.7209               0.6790           0.7221
#          1000            0.0015             42.63       42628         0.9656               0.9616           0.9658
#         10000            0.0004              4.66       46572         1.0000               1.0000           1.0000
```

The total variation distance between the exact Beta posterior and the Gaussian the theorem promises runs $0.0293$, $0.0082$, $0.0015$ and $0.0004$ as the sample grows from twenty-five to ten thousand — the $O(n^{-1/2})$ decay, and small enough by a thousand observations that no decision distinguishes them. The right-hand columns show the same thing in a quantity a desk would report: $P(\text{hit rate}>1/2)$ under a flat prior and under the Gaussian approximation agree at $0.6075$ against $0.6101$, then $0.7209$ against $0.7221$, then $0.9656$ against $0.9658$, and finally at $1.0000$ throughout.

The prior column is where the asymptotic licence should be read carefully. A skeptical $\text{Beta}(60,60)$ — a hundred and twenty pseudo-observations centred on a coin — moves the posterior mean by $292.81$ basis points at $n=25$, then $218.34$, $42.63$ and $4.66$. The decay is eventually the promised $1/n$, and the `shift x n` column shows it stabilizing only in the last two rows, at $42{,}628$ and $46{,}572$. **The rate is asymptotic in units of the prior's own strength, so a prior worth a hundred and twenty observations is still doing visible work at a thousand and has not begun to vanish at a hundred.** At twenty-five observations the flat and skeptical priors disagree about $P(\text{hit rate}>1/2)$ by $0.6075$ against $0.5593$, which is the difference between a lean and nothing at all.

## Where the Model Is Wrong the Posterior Concentrates Just as Fast on a Different Number, and Reports a Narrower Interval as It Does

Every condition in section 3's proof is a statement about the model containing the truth. Drop that and the theorem does not fail gracefully — it is replaced by a different theorem with the same shape and a different variance, and nothing in the output records which one applied.

??? note "Proof that under misspecification the posterior concentrates on the Kullback–Leibler minimizer with curvature $H$ while the estimator's sampling variance is the sandwich $H^{-1}JH^{-1}$, so the two coincide only when the information equality holds"

    Let the data be drawn from $p_0$, which need not lie in $\{f(\cdot\mid\theta)\}$. The law of large numbers gives $n^{-1}\ell_n(\theta)\to\mathbb{E}_{p_0}[\log f(X\mid\theta)]$, maximized at
    $$\theta^{*}=\arg\min_\theta \mathrm{KL}\big(p_0\,\|\,f(\cdot\mid\theta)\big),$$
    the pseudo-true value. Repeating section 3's expansion around $\hat\theta$ — which now converges to $\theta^{*}$ rather than to a true parameter — the posterior is asymptotically $N(\hat\theta,\ H^{-1}/n)$ with
    $$H=-\mathbb{E}_{p_0}\big[\nabla^{2}\log f(X\mid\theta^{*})\big],$$
    the curvature of the fitted log-likelihood. Meanwhile the sampling distribution of $\hat\theta$ is governed by the central limit theorem applied to the score, whose variance under $p_0$ is $J=\mathrm{Var}_{p_0}[\nabla\log f(X\mid\theta^{*})]$, giving $\sqrt n(\hat\theta-\theta^{*})\Rightarrow N(0,H^{-1}JH^{-1})$.

    When the model is correct, $H=J=I(\theta_0)$ — the information equality — and the sandwich collapses to $I^{-1}$, recovering section 3. When it is not, $H$ and $J$ are unrelated, and the posterior reports $H^{-1}$ regardless. Two failures follow. Dependence between observations inflates $J$ without touching $H$, since $H$ is an expectation of a second derivative that never sees the joint law; the posterior is then too narrow by the ratio of long-run to short-run variance, which for an AR(1) is $(1+\rho)/(1-\rho)$. Shape misspecification moves $\theta^{*}$ itself, so a functional of $\theta^{*}$ — a quantile, a tail probability — converges to the wrong number at the usual $\sqrt n$ rate.

    **The posterior's spread is computed from the curvature of the model being fitted and never from the data's disagreement with it, so a posterior cannot narrow in response to evidence that its own model is wrong — it has no term that could.** The load-bearing consequence is that concentration measures sample size and model rigidity, not proximity to the truth, and the two are indistinguishable from inside.

Both failure modes are large at sample sizes a desk would consider ample:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(16035)
n, reps, sig = 500, 20_000, 0.010
z = stats.norm.isf(0.025)

print(f"  {n} returns, flat prior, posterior for the mean computed as though the days were"
      f" independent, {reps:,} datasets")
print("      rho   posterior sd   true sd   ratio   95% credible coverage   with the"
      " long-run variance")
for rho in (0.0, 0.2, 0.4, 0.6):
    e = rng.standard_normal((reps, n)) * sig
    x = np.empty((reps, n))
    x[:, 0] = e[:, 0] / np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    xb, s = x.mean(1), x.std(1, ddof=1)
    psd = s / np.sqrt(n)
    lrv = psd * np.sqrt((1 + rho) / (1 - rho))          # oracle sandwich correction
    print(f"    {rho:5.2f}   {psd.mean() * 1e4:12.4f}   {xb.std() * 1e4:7.4f}"
          f"   {xb.std() / psd.mean():5.3f}   {(np.abs(xb) <= z * psd).mean():21.4f}"
          f"   {(np.abs(xb) <= z * lrv).mean():21.4f}")

m, draws = 4_000, 2_000
zq = stats.norm.ppf(0.01)
print(f"  {n} returns from a Student-t fitted with a Gaussian, {m:,} datasets and {draws:,}"
      f" posterior draws each; the question is the 1% daily loss the model implies")
print("       df   true 1% quantile   posterior mean   posterior sd   95% credible coverage"
      "   mean shortfall")
for df in (3, 4, 6, 12, 1000):
    sc = sig / np.sqrt(df / (df - 2))
    x = stats.t.rvs(df, scale=sc, size=(m, n), random_state=rng)
    truth = stats.t.ppf(0.01, df, scale=sc)
    S = ((x - x.mean(1, keepdims=True)) ** 2).sum(1)
    v = stats.invgamma.rvs((n - 1) / 2, scale=S[:, None] / 2, size=(m, draws),
                           random_state=rng)
    q = x.mean(1)[:, None] + np.sqrt(v) * (rng.standard_normal((m, draws)) / np.sqrt(n) + zq)
    lo, hi = np.quantile(q, 0.025, axis=1), np.quantile(q, 0.975, axis=1)
    print(f"    {df:5d}   {truth * 1e2:16.4f}%   {q.mean() * 1e2:14.4f}%"
          f"   {q.std(1).mean() * 1e2:12.4f}%   {((lo <= truth) & (truth <= hi)).mean():21.4f}"
          f"   {(q.mean() - truth) * 1e2:13.4f}%")
# =>   500 returns, flat prior, posterior for the mean computed as though the days were independent, 20,000 datasets
#          rho   posterior sd   true sd   ratio   95% credible coverage   with the long-run variance
#         0.00         4.4702    4.4784   1.002                  0.9484                  0.9484
#         0.20         4.5604    5.5903   1.226                  0.8925                  0.9467
#         0.40         4.8692    7.4814   1.536                  0.7985                  0.9468
#         0.60         5.5691   11.3149   2.032                  0.6626                  0.9458
#      500 returns from a Student-t fitted with a Gaussian, 4,000 datasets and 2,000 posterior draws each; the question is the 1% daily loss the model implies
#           df   true 1% quantile   posterior mean   posterior sd   95% credible coverage   mean shortfall
#            3            -2.6216%          -2.2823%         0.0847%                  0.1542          0.3393%
#            4            -2.6495%          -2.3192%         0.0860%                  0.1510          0.3303%
#            6            -2.5660%          -2.3243%         0.0862%                  0.2760          0.2417%
#           12            -2.4474%          -2.3262%         0.0862%                  0.6945          0.1212%
#         1000            -2.3278%          -2.3273%         0.0864%                  0.9480          0.0004%
```

The first table is the dependence failure and it is pure spread. The posterior's standard deviation for the mean barely moves as autocorrelation rises — $4.4702$, $4.5604$, $4.8692$ and $5.5691$ basis points — while the actual sampling standard deviation of the posterior mean runs $4.4784$, $5.5903$, $7.4814$ and $11.3149$. The ratio reaches $2.032$ at $\rho=0.6$, and coverage of a nominal ninety-five per cent credible interval falls $0.9484$, $0.8925$, $0.7985$, $0.6626$. Substituting the long-run variance restores coverage to $0.9484$, $0.9467$, $0.9468$ and $0.9458$, which is the sandwich of the proof doing its job — and it is worth being clear that this repair is not a Bayesian operation. It replaces the posterior's own variance with one estimated from the data's disagreement with the model, an act the Bayesian machinery has no way to request.

The second table is worse, because there the point estimate moves too. A Gaussian model fitted to Student-$t$ returns and asked for the one-per-cent daily loss answers $-2.2823\%$ at three degrees of freedom when the truth is $-2.6216\%$, a shortfall of $0.3393\%$ of capital — on a hundred-million-dollar book, three hundred and thirty-nine thousand dollars of loss placed outside the interval the model says it should be inside. The posterior standard deviation is $0.0847\%$, so the bias is four times the uncertainty the model admits to. Coverage of the nominal ninety-five per cent credible interval for that quantile is $0.1542$, then $0.1510$, $0.2760$ and $0.6945$ as the tails lighten, and $0.9480$ once the data is genuinely Gaussian.

**The posterior standard deviation is essentially constant across that entire sweep — $0.0847\%$, $0.0860\%$, $0.0862\%$, $0.0862\%$, $0.0864\%$ — while coverage runs from $0.1542$ to $0.9480$, so the one number the analyst would look at to judge confidence is exactly the number that carries no information about whether the answer is right.** That is the Brier score in the course lesson's gate, arriving from the other direction: discrimination and calibration are different quantities, and a posterior reports only the width of its own belief.

## The Repairs Are Real, and Every One of Them Imports Something From Outside the Posterior

The failures of section 4 are well known and each has an established response, which is worth stating because the responses share a feature. The sandwich or robust posterior rescales the credible region by $H^{-1}JH^{-1}$ estimated from residual behaviour, exactly as the second column of the first table does. A block bootstrap of the sort [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) develops estimates the same correction without naming $J$. Power posteriors raise the likelihood to a fractional power $\eta<1$, widening the posterior by $1/\eta$ and acknowledging that $n$ dependent observations are worth fewer than $n$ independent ones. Each is a genuine repair and none of them is derivable from the prior and likelihood alone: they all require a quantity — the data's disagreement with the model — that the Bayesian calculation does not contain.

The constructive response is posterior predictive checking, which is the only diagnostic here internal to the framework: simulate replicate datasets from the fitted posterior, compute a statistic on each, and compare with the observed value. A Gaussian fit to $t_3$ returns produces replicates whose kurtosis is nowhere near the sample's, and that discrepancy is visible without knowing the truth. [Bayesian Prediction](07-bayesian-prediction.md) treats the predictive distribution properly and notes the check's known conservatism, since the data is used both to fit and to test.

The theoretical position is due to Berk, Huber and White, and it is not pessimistic so much as deflationary: the posterior converges to the projection of the truth onto the model class, and that projection is the best available answer within the assumptions made. The failure is not that the posterior lies. It is that the object converged to is defined relative to a model, the report contains no reference to the model, and a reader who has forgotten which assumptions were made cannot recover them from the number.

!!! note "The posterior, the normalized likelihood, the sampling distribution of the estimator and the bootstrap distribution are four densities over the same axis, and misspecification separates them in different directions"
    Under correct specification and large $n$ all four converge to the same Gaussian, which is why they are casually interchanged. The **posterior** $\pi(\theta\mid x)$ is a belief about $\theta$ given this dataset, with width $H^{-1}/n$ set by the curvature of the fitted model. The **normalized likelihood** is the posterior under a flat prior in the current coordinates and inherits the coordinate dependence [Prior Distributions](02-prior-distributions.md) measures. The **sampling distribution of the estimator** describes $\hat\theta$ over repeated datasets, with width $H^{-1}JH^{-1}/n$, and coincides with the posterior only when the information equality $H=J$ holds — that is, only when the model is right. The **bootstrap distribution** of [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) estimates the third by resampling and therefore tracks it under misspecification, which is precisely why a bootstrap standard error and a posterior standard deviation disagreeing is informative rather than an inconsistency to be reconciled. Reporting a posterior standard deviation as though it were a standard error is the error this list exists to prevent, and section 4 prices it at a factor of $2.032$.

!!! warning "A posterior narrows at the same rate whether it is approaching the truth or a projection of it, so concentration is evidence about sample size and model rigidity and never about correctness"
    Nothing in the failing cases looked wrong. The dependence exhibit produced posterior standard deviations of $4.4702$ to $5.5691$ basis points, a smooth and plausible progression, while true sampling variability ran to $11.3149$ and coverage fell to $0.6626$. The tail exhibit produced posterior standard deviations of $0.0847\%$ to $0.0864\%$ — flat to three decimal places across models whose coverage ran from $0.1542$ to $0.9480$ — and a one-per-cent loss estimate wrong by four times its own stated uncertainty. In both cases the failure is invisible from inside because the posterior's width is a function of the model's curvature and the sample size, and neither of those quantities knows anything about the model being wrong. **The free diagnostic is to draw a few hundred replicate datasets from your fitted posterior, compute on each the one statistic your model was never asked to match — the lag-one autocorrelation if you assumed independence, the kurtosis or the count of four-sigma days if you assumed a Gaussian — and report where the observed value falls among the replicates, because a posterior predictive p-value below $0.01$ on a statistic you did not fit means the interval you are about to publish has coverage you have not measured.** It costs one simulation loop and it is the only check on this page that the framework can perform on itself.

## A Distribution That Sharpens Whether or Not It Is Getting Closer

This page established that a posterior is available immediately up to a constant and that the constant is the only expensive step, so the mode is free and every expectation, probability and quantile is not; that two independent routes to that constant agree closely on a non-conjugate two-parameter problem, quadrature over $361{,}201$ evaluations and importance sampling returning posterior means of $5.5687$ and $5.5682$ basis points, standard deviations of $5.3801$ and $5.3800$ and log normalizers of $817.3645$ and $817.3662$, at an effective sample size ratio near $0.114$ that decays geometrically with dimension; that Bernstein–von Mises makes the posterior and the sampling distribution of the estimator the same Gaussian, with total variation falling $0.0293$, $0.0082$, $0.0015$, $0.0004$ from twenty-five observations to ten thousand while a prior worth a hundred and twenty pseudo-observations still shifts the posterior mean by $42.63$ basis points at $n=1000$; and that dropping the assumption the model contains the truth replaces that theorem with one of the same shape and a different variance, the posterior reporting standard deviations of $4.4702$ to $5.5691$ basis points where the truth was $4.4784$ to $11.3149$, coverage falling to $0.6626$ and restored to $0.9458$ only by importing the long-run variance, and a Gaussian model's one-per-cent daily loss landing at $-2.2823\%$ against a truth of $-2.6216\%$ with a posterior standard deviation of $0.0847\%$ and coverage of $0.1542$.

The shape shared by all three exhibits is that the posterior's width is computed from the model and the sample size and from nothing else. That is a feature in section 3, where it delivers the exact asymptotic behaviour the theorem promises, and it is the whole failure in section 4, where the same formula keeps producing confident numbers about a model class the data has already contradicted. There is no term anywhere in the calculation that could respond, which is why every repair on this page is an import.

What has been assumed throughout is that the normalizer is an obstacle to be overcome by computation. For one family of models it is not an obstacle at all: the integral is available in closed form, the posterior stays inside the family it started in, and updating becomes arithmetic on a handful of numbers. That family is small, its convenience has consequences for exactly the tail behaviour section 4 found so expensive, and it is [Conjugate Priors](04-conjugate-priors.md).

**A posterior reports how tightly a model can be tuned rather than how close that model is to the world, and the two are the same number only when the assumption nobody printed happens to hold.**
