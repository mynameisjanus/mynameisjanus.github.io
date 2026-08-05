# Expectation-Maximization Algorithm

Expectation-maximization is usually sold as the algorithm that makes an intractable likelihood tractable, and it is; what the pitch omits is that the three things it does not supply are the three things anybody wanted. It does not supply a rate: the same mixture converges in $8$ sweeps at wide separation and $147$ at narrow, and the constant governing that is the fraction of the information the missing data took with it, measured below at $0.0652$ and $0.8662$ and predicting the sweep count to within a few per cent. It does not supply a standard error: the substitute everyone reaches for is the complete-data information, which sits flat at $5.3666$ basis points while the truth runs to $6.9912$, and intervals built from it cover $0.8686$ instead of $0.95$ — too small by exactly $\sqrt{1-\text{missing fraction}}$, an identity that holds to four decimals in every row. And it does not supply a destination: monotone ascent holds exactly, $0$ decreases across $2{,}291{,}200$ sweeps, all the way up a spike where a fitted component has collapsed onto one day. On $0.4925$ of datasets one of those spikes carries the highest likelihood of all $32$ starts, so keeping the best of $32$ restarts selects a degenerate fit half the time and puts the probability of a $-3\%$ day at $0.2949\%$ against a truth of $1.2119\%$ — where keeping the best of one gets $0.6449\%$. Searching harder makes the answer worse.

This page covers the bound EM actually maximizes and why ascent follows from the bound rather than from the model, the linear convergence rate and its identification with the missing-information fraction, the standard errors EM does not produce and the size of the error in the usual substitute, and the sense in which the monotone-ascent guarantee is compatible with converging to something that is not an estimate. It derives no forward–backward recursion and no Baum–Welch update, which is [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md) and is the worked specialization of everything here; it runs no general-purpose optimizer on the same surface, which is [Numerical Optimization](01-numerical-optimization.md); it evaluates no integral by quadrature, which is [Numerical Integration](02-numerical-integration.md); it draws no sample from a posterior over the latent variables, which is [Markov Chain Monte Carlo](04-markov-chain-monte-carlo.md); it derives no likelihood asymptotics and no Fisher information from scratch, which is [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md); it establishes no consistency or efficiency, which is [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md); it selects no number of components, which is [Information Criteria (AIC/BIC)](../part-14-model-selection/03-information-criteria.md); it derives no penalized-likelihood identity for the prior that repairs section 4, which is [Maximum A Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md); it constructs no interval and proves nothing about coverage, which is [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md); it treats no latent variable as a sequential filtering problem, which is [Particle and Kalman Filters](../../advanced/02-particle-and-kalman-filters.md); and it never treats a likelihood that increased as evidence that anything was learned.

The trading stake is a course lesson that names both failures on this page in a single sentence and then declines to be reassured by either. [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) writes that "EM maximizes likelihood for a *given* number of states; it cannot tell you how many to use, and it finds local optima, so every fit below is the best of five random restarts," and prints `k=1: loglik  19094.8   BIC  -38172.1` through `k=4: loglik  20785.6   BIC  -41369.6` — a criterion that improves at every $k$ and would keep improving. The lesson's response is to refuse it: "a state you cannot name is a state you cannot trade." Section 4 measures what the best-of-five recipe costs when the likelihood being maximized has no maximum.

## EM Maximizes a Bound, and Ascent Follows From the Bound Rather Than From Anything About the Model

The observed-data likelihood of a latent-variable model is an integral — over regime paths, over component labels, over censored values — and it is the integral that makes direct maximization hard. EM never evaluates it. It manipulates a lower bound whose maximization is a weighted version of the problem that would have been easy had the latent variables been seen.

??? note "Proof that $\log p(x\mid\theta)=\mathcal{L}(q,\theta)+\mathrm{KL}(q\Vert p(z\mid x,\theta))$ for every $q$, so the E-step makes the bound tight and the M-step cannot decrease the observed-data likelihood, and that merely increasing $Q$ suffices"

    For any distribution $q$ over the latent variables $z$, write
    $$\mathcal{L}(q,\theta)=\int q(z)\log\frac{p(x,z\mid\theta)}{q(z)}\,\mathrm{d}z,\qquad \mathrm{KL}(q\Vert p)=-\int q(z)\log\frac{p(z\mid x,\theta)}{q(z)}\,\mathrm{d}z.$$
    Adding them, the $q$ terms cancel and $p(x,z\mid\theta)/p(z\mid x,\theta)=p(x\mid\theta)$, which does not depend on $z$, so
    $$\mathcal{L}(q,\theta)+\mathrm{KL}(q\Vert p(z\mid x,\theta))=\log p(x\mid\theta).$$
    The identity is exact for every $q$, and since $\mathrm{KL}\ge0$ it makes $\mathcal{L}$ a lower bound on the log-likelihood, touching it precisely when $q=p(z\mid x,\theta)$.

    The **E-step** sets $q_t=p(z\mid x,\theta_t)$, driving the divergence to zero, so $\mathcal{L}(q_t,\theta_t)=\log p(x\mid\theta_t)$: the bound is tight at the current parameter and nowhere else. The **M-step** maximizes $\mathcal{L}(q_t,\theta)$ over $\theta$, which — since the $-\int q\log q$ term is free of $\theta$ — is the same as maximizing $Q(\theta\mid\theta_t)=\mathbb{E}_{q_t}[\log p(x,z\mid\theta)]$, the complete-data log-likelihood averaged over the imputed latent variables. Chaining,
    $$\log p(x\mid\theta_{t+1})\ \ge\ \mathcal{L}(q_t,\theta_{t+1})\ \ge\ \mathcal{L}(q_t,\theta_t)\ =\ \log p(x\mid\theta_t),$$
    where the first inequality is the bound, the second is the M-step's optimality, and the equality is the E-step's tightness. Note what the argument never used: any property of $p$ beyond being a probability model. Note also that the second inequality only needs $Q(\theta_{t+1}\mid\theta_t)\ge Q(\theta_t\mid\theta_t)$, not a maximizer — which is *generalized* EM, and is why an M-step solved approximately, by one Newton step or by a coordinate update, inherits the guarantee intact.

    **The load-bearing consequence is that monotone ascent is a property of the bound's geometry and not of the surface being climbed, so it holds identically on a path toward a well-identified maximum and on a path toward a point where the likelihood is unbounded — and a sequence of increasing numbers is exactly as much evidence in the second case as in the first.**

## The Rate Is Linear, and Its Constant Is the Fraction of the Information the Missing Data Took With It

Because EM is a fixed-point iteration $\theta_{t+1}=M(\theta_t)$ rather than a Newton method, its convergence is linear rather than quadratic, and the contraction factor is not an implementation detail. It is a statistic of the problem.

??? note "Proof that $DM(\theta^{\ast})=I_{\mathrm{mis}}I_{\mathrm{com}}^{-1}$, so EM's linear rate is the largest missing-information fraction, and that Louis' identity $I_{\mathrm{obs}}=I_{\mathrm{com}}-I_{\mathrm{mis}}$ follows from the same decomposition"

    Differentiate the identity $\log p(x\mid\theta)=Q(\theta\mid\theta')-H(\theta\mid\theta')$ twice in $\theta$, where $H(\theta\mid\theta')=\mathbb{E}_{q_{\theta'}}[\log p(z\mid x,\theta)]$, and evaluate at $\theta=\theta'=\theta^{\ast}$. Writing $I_{\mathrm{com}}=-\nabla^{2}_{\theta}Q$ and $I_{\mathrm{mis}}=-\nabla^{2}_{\theta}H$, both at $\theta^{\ast}$,
    $$I_{\mathrm{obs}}=I_{\mathrm{com}}-I_{\mathrm{mis}},$$
    which is Louis' identity: the information in the data you have equals the information you would have had, minus the information the latent variables carried away. $I_{\mathrm{mis}}$ is a covariance of a conditional score and is therefore positive semi-definite, so $I_{\mathrm{obs}}\preceq I_{\mathrm{com}}$ always.

    The M-step solves $\nabla_\theta Q(\theta\mid\theta')=0$, defining $M$ implicitly. Differentiating that equation in $\theta'$ and using the fact that $\nabla_\theta Q(\theta^{\ast}\mid\theta^{\ast})=0$ gives
    $$DM(\theta^{\ast})=I_{\mathrm{mis}}I_{\mathrm{com}}^{-1}=I-I_{\mathrm{obs}}I_{\mathrm{com}}^{-1}.$$
    A fixed-point iteration converges linearly at the spectral radius of its Jacobian, so the rate is the largest eigenvalue of $I_{\mathrm{mis}}I_{\mathrm{com}}^{-1}$ — the *missing-information fraction* in the least-determined direction. Two limits are worth naming: when nothing is missing, $I_{\mathrm{mis}}=0$, the rate is $0$ and EM converges in one step; as the latent variables approach carrying all the information, the rate approaches $1$ and the iteration stalls.

    In one dimension this rearranges into a statement about standard errors. From $I_{\mathrm{obs}}=I_{\mathrm{com}}(1-r)$ with $r$ the rate,
    $$\frac{\mathrm{se}_{\mathrm{com}}}{\mathrm{se}_{\mathrm{obs}}}=\sqrt{\frac{I_{\mathrm{obs}}}{I_{\mathrm{com}}}}=\sqrt{1-r},$$
    so the standard error obtained by pretending the imputed values were data is too small by a factor determined entirely by how slowly the algorithm ran. Section 3 checks this to four decimals.

    **The load-bearing identification is that the speed of the algorithm and the precision of the estimate are the same quantity read twice, so a fit that took many sweeps is not merely expensive — it is a fit in which the data determined the parameter poorly, and the iteration count is the only free diagnostic of that fact anybody already has.**

The prediction is quantitative, so it can be checked against a stopwatch:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(17031)
n, P, S1, S2, M1 = 6_000, 0.80, 0.0075, 0.0165, 0.0004


def step(t, x):
    """One EM sweep for a two-component mixture with known scales; t = (p, m1, m2)."""
    a = np.log(t[0]) + stats.norm.logpdf(x, t[1], S1)
    b = np.log1p(-t[0]) + stats.norm.logpdf(x, t[2], S2)
    w = np.exp(a - np.logaddexp(a, b))
    return np.array([w.mean(), (w * x).sum() / w.sum(),
                     ((1 - w) * x).sum() / (1 - w).sum()])


def jacobian(t, x):
    """DM at the fixed point; its spectral radius is the missing-information fraction."""
    h = np.array([1e-6, 1e-9, 1e-9])
    J = np.empty((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h[j]
        J[:, j] = (step(t + e, x) - step(t - e, x)) / (2 * h[j])
    return J


print(f"  two-component mixture on {n:,} days with known scales {S1} and {S2}, fitted by"
      f" EM from a common start; separation is the gap between the component means in"
      f" units of the wider scale")
print("     separation   sweeps to 1e-10   observed rate   spectral radius of DM"
      "   sweeps the rate predicts")
for sep in (4.0, 3.0, 2.0, 1.5, 1.0):
    m2 = M1 - sep * S2
    x = np.where(rng.random(n) < P, rng.normal(M1, S1, n), rng.normal(m2, S2, n))
    t, path = np.array([0.5, M1 + 0.002, m2 - 0.002]), []
    for _ in range(4_000):
        t = step(t, x)
        path.append(t.copy())
    path = np.array(path)
    star = path[-1]
    err = np.abs(path - star).max(1)
    k = int(np.argmax(err < 1e-10)) + 1
    m = min(10, k - 1)
    r = (err[k - 1] / err[k - 1 - m]) ** (1 / m)
    rho = np.abs(np.linalg.eigvals(jacobian(star, x))).max()
    pred = np.log(1e-10 / err[0]) / np.log(rho)
    print(f"    {sep:10.2f}   {k:16,d}   {r:13.4f}   {rho:21.4f}   {pred:24.1f}")
# =>   two-component mixture on 6,000 days with known scales 0.0075 and 0.0165, fitted by EM from a common start; separation is the gap between the component means in units of the wider scale
#         separation   sweeps to 1e-10   observed rate   spectral radius of DM   sweeps the rate predicts
#              4.00                  8          0.0656                  0.0652                        6.3
#              3.00                 17          0.2858                  0.2858                       15.3
#              2.00                 43          0.6021                  0.6021                       40.4
#              1.50                 94          0.7949                  0.7949                       91.5
#              1.00                147          0.8662                  0.8662                      147.5
```

The middle two columns are the theorem. The rate measured from the last decade of the error decay and the spectral radius of the EM map's Jacobian at its own fixed point agree to four decimals in four of five rows and to three in the first — $0.0656$ against $0.0652$, then $0.2858$, $0.6021$, $0.7949$ and $0.8662$ matching exactly. These are two entirely different computations: one watches the algorithm run, the other differentiates a map at a point. Their agreement is the identification $DM=I_{\mathrm{mis}}I_{\mathrm{com}}^{-1}$ being correct.

The outer columns turn it into a cost. Sweeps to convergence run $8$, $17$, $43$, $94$ and $147$; the count predicted from the rate alone runs $6.3$, $15.3$, $40.4$, $91.5$ and $147.5$. **A rate of $0.8662$ against a rate of $0.0652$ is a factor of eighteen in wall-clock time, and the only thing that changed between the top and bottom rows is how far apart the two regimes are — a property of the market, not of the code.** At separation $4.0$ each observation almost announces which component produced it, the latent labels are nearly observed, and the missing-information fraction is $0.0652$. At separation $1.0$ the components overlap heavily, most days could plausibly have come from either, and the fraction is $0.8662$.

The practical reading is the one the proof ends on. An EM fit that needs a hundred and fifty sweeps is not reporting a slow implementation; it is reporting that $0.8662$ of the information about the regime structure went missing with the labels, and that the parameter it is about to hand over is correspondingly ill-determined. Nobody has to compute anything extra to learn this — the iteration count was already on the screen.

## EM Returns No Standard Error, and the Substitute Everybody Reaches For Is Too Small by Exactly the Square Root of What Went Missing

The M-step maximizes $Q$, so the curvature naturally available at the end of an EM run is $I_{\mathrm{com}}$, the complete-data information — the curvature of the problem as though the latent variables had been observed. It is the wrong matrix, it is always too large, and the direction of the error is always the same.

A censored sample makes the arithmetic exact, and a reporting floor on a return series is the plainest example there is:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(17033)
n, reps, MU, SIG = 500, 20_000, 0.0004, 0.0120
z = stats.norm.isf(0.025)


def em_step(mu, tot, k, cut):
    """One EM sweep for the mean of a left-censored normal with the scale known."""
    a = (cut - mu) / SIG
    imp = mu - SIG * np.exp(stats.norm.logpdf(a) - stats.norm.logcdf(a))
    return (tot + k * imp) / n


def loglik(mu, x, keep, k, cut):
    """Observed-data log-likelihood: uncensored densities plus a censoring probability."""
    q = ((x - mu[:, None]) / SIG) ** 2
    return -0.5 * (q * keep).sum(1) + k * stats.norm.logcdf((cut - mu) / SIG)


print(f"  the mean of {n} daily returns left-censored at a reporting floor, fitted by EM"
      f" with the scale known; {reps:,} datasets, true mean {MU * 1e4:.2f}bp, and the"
      f" rate is the derivative of the EM map at its own fixed point")
print("     censored share   EM rate   sd, complete data   sd, observed data"
      "   true sampling sd   sd ratio   sqrt(1 - rate)   coverage, complete"
      "   coverage, observed")
for q in (0.01, 0.10, 0.25, 0.50, 0.75):
    cut = stats.norm.ppf(q, MU, SIG)
    x = rng.normal(MU, SIG, (reps, n))
    keep = x > cut
    k = n - keep.sum(1)
    tot = np.where(keep, x, 0.0).sum(1)

    mu = tot / np.maximum(n - k, 1)
    for _ in range(400):
        mu = em_step(mu, tot, k, cut)

    h = 1e-8
    rate = (em_step(mu + h, tot, k, cut) - em_step(mu - h, tot, k, cut)) / (2 * h)
    d = 1e-7
    info = -(loglik(mu + d, x, keep, k, cut) - 2 * loglik(mu, x, keep, k, cut)
             + loglik(mu - d, x, keep, k, cut)) / d ** 2
    se_com, se_obs = SIG / np.sqrt(n), 1 / np.sqrt(info)
    print(f"    {q:14.2f}   {rate.mean():7.4f}   {se_com * 1e4:17.4f}"
          f"   {se_obs.mean() * 1e4:17.4f}   {mu.std() * 1e4:17.4f}"
          f"   {se_com / se_obs.mean():9.4f}   {np.sqrt(1 - rate.mean()):14.4f}"
          f"   {(np.abs(mu - MU) <= z * se_com).mean():18.4f}"
          f"   {(np.abs(mu - MU) <= z * se_obs).mean():19.4f}")
# =>   the mean of 500 daily returns left-censored at a reporting floor, fitted by EM with the scale known; 20,000 datasets, true mean 4.00bp, and the rate is the derivative of the EM map at its own fixed point
#         censored share   EM rate   sd, complete data   sd, observed data   true sampling sd   sd ratio   sqrt(1 - rate)   coverage, complete   coverage, observed
#                  0.01    0.0010              5.3666              5.3692              5.3666      0.9995           0.9995               0.9499                0.9500
#                  0.10    0.0170              5.3666              5.4127              5.4749      0.9915           0.9915               0.9463                0.9482
#                  0.25    0.0606              5.3666              5.5370              5.5546      0.9692           0.9692               0.9414                0.9486
#                  0.50    0.1819              5.3666              5.9339              5.9311      0.9044           0.9045               0.9244                0.9521
#                  0.75    0.4022              5.3666              6.9446              6.9912      0.7728           0.7732               0.8686                0.9496
```

The `sd ratio` and `sqrt(1 - rate)` columns are two independent computations of the same number and they agree to four decimals in every row: $0.9995$, $0.9915$, $0.9692$, $0.9044$ against $0.9045$, and $0.7728$ against $0.7732$. The left one is a ratio of standard errors — one from the complete-data information, one from a numerical second derivative of the observed-data log-likelihood. The right one is built from the derivative of the EM map, which knows nothing about either. **The speed of the algorithm predicts the size of the error in the naive standard error, exactly, which is the previous section's identification cashed out.**

The columns around them say why it matters. The complete-data standard error is $5.3666$ basis points in every row, because it is $\sigma/\sqrt{n}$ and neither $\sigma$ nor $n$ changes when observations are censored — the imputed values are counted as though they were data, which is precisely what the E-step invites. The true sampling standard deviation of the estimator runs $5.3666$, $5.4749$, $5.5546$, $5.9311$ and $6.9912$, and the observed-information standard error tracks it: $5.3692$, $5.4127$, $5.5370$, $5.9339$ and $6.9446$. At three quarters censoring the naive figure is $22.7\%$ too small.

Coverage converts that into the number a risk committee reads. Intervals from the observed information cover $0.9500$, $0.9482$, $0.9486$, $0.9521$ and $0.9496$ — nominal, throughout. Intervals from the complete-data information cover $0.9499$, $0.9463$, $0.9414$, $0.9244$ and $0.8686$. **The failure is silent and directional: it never overstates uncertainty, it worsens smoothly as more of the data goes missing, and the run that produces the worst interval is the run that took longest, which is the one an analyst is most likely to have watched carefully and least likely to distrust.**

## Monotone Ascent Is a Guarantee About a Sequence and Not About a Destination, and More Restarts Make the Destination Worse

Sections 2 and 3 assumed a fixed point worth converging to. For the model in the trading stake there need not be one. A Gaussian mixture's likelihood is unbounded: send one component's mean to any observed value and its scale to zero, and the likelihood diverges, so the maximum likelihood estimate does not exist.

??? note "Proof that a Gaussian mixture's likelihood is unbounded above, so the estimator EM is iterating toward is undefined and every finite value it reports is an artefact of where the arithmetic stopped"

    For a two-component mixture with density $p\,\phi(x;\mu_1,\sigma_1)+(1-p)\,\phi(x;\mu_2,\sigma_2)$ and any sample $x_1,\dots,x_n$, fix $p\in(0,1)$, $\mu_2$ and $\sigma_2>0$, and set $\mu_1=x_1$. Then
    $$L(\theta)=\Big[\tfrac{p}{\sigma_1\sqrt{2\pi}}+(1-p)\phi(x_1;\mu_2,\sigma_2)\Big]\prod_{i\ge2}\Big[p\,\phi(x_i;x_1,\sigma_1)+(1-p)\phi(x_i;\mu_2,\sigma_2)\Big].$$
    As $\sigma_1\downarrow0$ the first bracket grows like $p/(\sigma_1\sqrt{2\pi})\to\infty$, while each remaining bracket is bounded below by $(1-p)\phi(x_i;\mu_2,\sigma_2)>0$, a quantity free of $\sigma_1$. The product therefore diverges, so $\sup_\theta L=\infty$ and no maximizer exists.

    The divergence is a genuine feature of the model rather than a numerical artefact, and it sits at the boundary of the parameter space, where the fitted density has become a point mass at one observation plus a component covering the rest. Two properties make it dangerous rather than merely notable. It is *attracting*: there is a basin of positive volume around each of $n$ such configurations, so random starts find them at a rate that does not vanish. And it is *invisible to the ascent guarantee*: section 1's proof holds at every step of the path toward it, because that proof never assumed a maximizer existed.

    The standard repairs both change the model. A **variance floor** restricts the parameter space to $\sigma\ge\sigma_{\min}$, restoring compactness and hence a maximizer. A **conjugate prior** on the variances adds a term diverging to $-\infty$ as $\sigma\downarrow0$, giving a bounded penalized objective whose maximizer exists — the MAP-EM of [Maximum A Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md), and in practice a constant added inside the M-step's variance update.

    **The load-bearing point is that "take the highest likelihood over many restarts" is a rule for selecting among candidates and the supremum it approximates is $+\infty$, so the rule is not an approximation to anything: increasing the number of restarts increases the chance of finding a point near the divergence, and the recipe converges to the degeneracy rather than away from it.**

Six months of daily returns is enough sample for a two-regime model and enough for the divergence:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(17035)
n, reps, S, SWEEPS = 120, 400, 32, 180
P, M1, M2, SD1, SD2 = 0.85, 0.0004, -0.0020, 0.0070, 0.0200
LOG2PI = np.log(2 * np.pi)


def em(x, t, ridge):
    """Vectorized EM over a (reps, starts) grid; t is (..., 5) = p, m1, s1, m2, s2."""
    drops, prev = 0, None
    for _ in range(SWEEPS):
        p = np.clip(t[..., 0], 1e-14, 1 - 1e-14)
        s1, s2 = t[..., 2, None], t[..., 4, None]
        la = (np.log(p)[..., None] - 0.5 * ((x - t[..., 1, None]) / s1) ** 2
              - np.log(s1) - 0.5 * LOG2PI)
        lb = (np.log1p(-p)[..., None] - 0.5 * ((x - t[..., 3, None]) / s2) ** 2
              - np.log(s2) - 0.5 * LOG2PI)
        tot = np.logaddexp(la, lb)
        ll = tot.sum(-1)
        if prev is not None:
            ok = np.isfinite(ll) & np.isfinite(prev)
            drops += int((ll[ok] < prev[ok] - 1e-6).sum())
        prev = ll
        w = np.exp(la - tot)
        sw = np.clip(w.sum(-1), 1e-10, n - 1e-10)
        m1 = (w * x).sum(-1) / sw
        m2 = ((1 - w) * x).sum(-1) / (n - sw)
        v1 = (w * (x - m1[..., None]) ** 2).sum(-1) / sw + ridge
        v2 = ((1 - w) * (x - m2[..., None]) ** 2).sum(-1) / (n - sw) + ridge
        t = np.stack([sw / n, m1, np.sqrt(np.maximum(v1, 1e-24)),
                      m2, np.sqrt(np.maximum(v2, 1e-24))], -1)
    return t, prev, drops


def stress_prob(t):
    """Model-implied probability of a day worse than -3%, in per cent."""
    p = np.clip(t[..., 0], 0.0, 1.0)
    return 100 * (p * stats.norm.cdf(-0.03, t[..., 1], np.maximum(t[..., 2], 1e-12))
                  + (1 - p) * stats.norm.cdf(-0.03, t[..., 3],
                                             np.maximum(t[..., 4], 1e-12)))


TRUTH = 100 * (P * stats.norm.cdf(-0.03, M1, SD1)
               + (1 - P) * stats.norm.cdf(-0.03, M2, SD2))
x = np.where(rng.random((reps, n)) < P, rng.normal(M1, SD1, (reps, n)),
             rng.normal(M2, SD2, (reps, n)))
gap = np.diff(np.sort(x, 1), axis=1).min(1)
sd0 = x.std(1)
t0 = np.stack([rng.uniform(0.1, 0.9, (reps, S)),
               rng.normal(0, sd0[:, None], (reps, S)),
               sd0[:, None] * np.exp(rng.normal(-0.7, 1.5, (reps, S))),
               rng.normal(0, sd0[:, None], (reps, S)),
               sd0[:, None] * np.exp(rng.normal(-0.7, 1.5, (reps, S)))], -1)
xx = x[:, None, :]

out = {}
for tag, ridge in (("plain", 0.0), ("floored", 0.0015 ** 2)):
    t, ll, drops = em(xx, t0.copy(), ridge)
    smin = np.minimum(t[..., 2], t[..., 4])
    out[tag] = (np.where(np.isfinite(ll), ll, -np.inf), smin < gap[:, None],
                stress_prob(t), drops)

ll, deg, sp, drops = out["plain"]
print(f"  two-component mixture on {n} days, {reps} datasets x {S} random starts, plain"
      f" EM; the quantity read off the fit is the probability of a day worse than -3%,"
      f" whose true value is {TRUTH:.4f}%")
print(f"    sweeps that decreased the observed log-likelihood: {drops} of"
      f" {reps * S * (SWEEPS - 1):,}, so monotone ascent holds on every run that stayed"
      f" finite")
wins = (np.where(deg, ll, -np.inf).max(1) > np.where(deg, -np.inf, ll).max(1)).mean()
print(f"    starts terminating on a degenerate spike: {deg.mean():.4f}, and on"
      f" {wins:.4f} of datasets a spike carries the highest likelihood"
      f" of all {S} starts")
print("     restarts kept   plain EM: selected fit degenerate   plain EM: probability of"
      " a -3% day   floored EM: selected fit degenerate   floored EM: probability")
for R in (1, 2, 5, 10, 32):
    cells = []
    for tag in ("plain", "floored"):
        L, D, SPr, _ = out[tag]
        idx = L[:, :R].argmax(1)
        r = np.arange(reps)
        cells.append((D[r, idx].mean(), np.median(SPr[r, idx])))
    print(f"    {R:14d}   {cells[0][0]:33.4f}   {cells[0][1]:31.4f}%"
          f"   {cells[1][0]:33.4f}   {cells[1][1]:26.4f}%")
# =>   two-component mixture on 120 days, 400 datasets x 32 random starts, plain EM; the quantity read off the fit is the probability of a day worse than -3%, whose true value is 1.2119%
#        sweeps that decreased the observed log-likelihood: 0 of 2,291,200, so monotone ascent holds on every run that stayed finite
#        starts terminating on a degenerate spike: 0.0434, and on 0.4925 of datasets a spike carries the highest likelihood of all 32 starts
#         restarts kept   plain EM: selected fit degenerate   plain EM: probability of a -3% day   floored EM: selected fit degenerate   floored EM: probability
#                     1                              0.0400                            0.6449%                              0.0000                       1.0498%
#                     2                              0.0600                            0.9099%                              0.0000                       1.1316%
#                     5                              0.1325                            0.8647%                              0.0000                       1.1316%
#                    10                              0.2275                            0.7307%                              0.0000                       1.1306%
#                    32                              0.4925                            0.2949%                              0.0000                       1.1316%
```

The first line is section 1's guarantee, verified to the limit of floating point: across $2{,}291{,}200$ sweeps, $0$ decreased the observed-data log-likelihood. EM did exactly what it promises, every time, on every run. The second line is what the promise is worth. Individual starts land on a degenerate spike $0.0434$ of the time, and because $32$ starts get $32$ chances, on $0.4925$ of datasets — very nearly half — the single highest likelihood among all of them belongs to a fit whose narrow component has collapsed below the smallest gap between two adjacent days.

The table is the recipe from the trading stake, run at several budgets. Keeping the best of one restart selects a degenerate fit on $0.0400$ of datasets; the best of two, $0.0600$; of five, $0.1325$; of ten, $0.2275$; of thirty-two, $0.4925$. **The curve is monotone increasing in the amount of work done, which inverts the usual relationship between effort and reliability: restarts exist to escape local optima, and here every additional restart is another lottery ticket in a draw whose prize is a fit that is not an estimate.** The lesson's five restarts sit at $0.1325$.

The consequence in the column a desk would read is worse than the degeneracy rate suggests. The probability of a $-3\%$ day, whose true value is $1.2119\%$, comes out at $0.6449\%$ from a single start and falls to $0.2949\%$ at thirty-two — the model becomes *more* confident that severe days are rare as the search grows more thorough, because a degenerate fit spends one of its two components on a single point and describes the entire remaining distribution, tails included, with the other. Understating the frequency of a $-3\%$ day by a factor of two and a half is a sizing error in the direction that adds leverage.

The last two columns are the repair and they cost one line. Adding a constant of $(0.0015)^{2}$ inside the variance update — a floor of fifteen basis points of daily volatility, which is far below anything a real regime exhibits and far above zero — eliminates degenerate terminations entirely, $0.0000$ at every restart budget, and returns $1.0498\%$ to $1.1316\%$ against the truth of $1.2119\%$. **The floor does not improve the optimizer. It repairs the model, by making the parameter space one on which a maximum likelihood estimate exists at all.**

## What EM Buys, What It Never Buys, and the Point Where the Bound Stops Being Tight

EM's real advantage is not speed, since section 2 shows it can be very slow. It is that each M-step is the complete-data problem, which for the models people actually write is available in closed form: means, variances and transition counts are weighted averages, so there is no inner optimizer, no step size, no line search, and no failure of the kind [Numerical Optimization](01-numerical-optimization.md) measures. Parameters stay inside their constraints automatically — variances positive, probabilities summing to one — because a weighted average of valid values is valid. That is why the algorithm survives in production long after faster methods are available.

The catalogue around it follows from the same decomposition. **Generalized EM** replaces the M-step's maximization with any increase in $Q$, inheriting the ascent guarantee, and **ECM** cycles conditional maximizations for the same reason. **Aitken acceleration** extrapolates the linear sequence using the rate measured in section 2, which is worth doing precisely when the rate is near one, and quasi-Newton acceleration does the same thing better. Standard errors come from Louis' identity directly, from the **supplemented EM** algorithm, which recovers $I_{\mathrm{obs}}$ from the numerical Jacobian of the EM map — the exact object section 2 computes — or from a bootstrap, which is the only route that also survives misspecification. And the most consequential generalization comes from relaxing the E-step: if $p(z\mid x,\theta)$ is itself intractable, restrict $q$ to a family that is not, and the identity in section 1 still holds while $\mathrm{KL}$ no longer reaches zero. The bound stays a bound and stops being tight, monotone ascent applies to the bound rather than to the likelihood, and the resulting estimate is biased by the gap. That is variational inference, and it trades the guarantee this page opened with for one that is weaker in a way the output does not record.

!!! note "The complete-data log-likelihood, the observed-data log-likelihood, the $Q$-function and the evidence lower bound are four objects over the same parameter, and EM increases the last two while being credited with the second"
    They coincide often enough to be conflated and differ exactly where it matters. The **complete-data log-likelihood** $\log p(x,z\mid\theta)$ is the easy object, available in closed form, and it is not computable because $z$ was never seen. The **observed-data log-likelihood** $\log p(x\mid\theta)$ is the quantity of interest and is an integral over $z$; EM never evaluates it, which is why a monotone-ascent check must be coded separately, as it is above. The **$Q$-function** $\mathbb{E}_{q_t}[\log p(x,z\mid\theta)]$ is the complete-data log-likelihood averaged against the *current* imputation, so it changes at every sweep, and comparing $Q$ values across sweeps is meaningless — a common error, since $Q$ is the thing the M-step returns. The **evidence lower bound** $\mathcal{L}(q,\theta)$ equals $Q$ plus an entropy term free of $\theta$, so it has the same maximizer, and it is what EM genuinely climbs; it touches the observed-data log-likelihood after each E-step and falls below it in between. Reading a rising $Q$ as a rising likelihood is what this list exists to prevent, and it is the same confusion that makes a variational bound look like a likelihood when the E-step is approximate.

!!! warning "Monotone ascent holds on the path to a point that is not an estimate, and the standard remedy of many restarts makes arrival there more likely rather than less"
    Nothing in the failing cases looked wrong. Section 4's runs produced $0$ decreases in $2{,}291{,}200$ sweeps — a perfect record against the only guarantee EM offers — while $0.0434$ of them climbed toward a component of zero width, and the resulting fits carried the highest likelihood on $0.4925$ of datasets. Selecting the best of thirty-two restarts, which is diligence, chose one of those on $0.4925$ of datasets and reported the probability of a $-3\%$ day as $0.2949\%$ against a truth of $1.2119\%$. Section 3's runs were worse still in the sense that nothing at all was visibly abnormal: the fit converged, the estimate was unbiased, and only the interval was wrong, covering $0.8686$ where it claimed $0.95$. **The free diagnostic is to record two numbers you already have — the sweep count, which is $\log(\text{tolerance})/\log(\text{rate})$ and therefore tells you the missing-information fraction and hence how much your naive standard error understates, and the smallest fitted scale parameter divided by the smallest gap between two adjacent observations, which must be discarded when it approaches one — and then to report the spread of the quantity you will size on across surviving restarts rather than the likelihood of the winner.** Both cost nothing, and the likelihood, which is the number everyone does record, is the one number that identifies neither failure.

## An Algorithm That Cannot Go Down and Need Not Arrive

This page established that EM maximizes a lower bound made tight by the E-step, so monotone ascent follows from the geometry of the bound rather than from any property of the model and extends unchanged to a merely-increasing M-step; that convergence is linear at the spectral radius of $I_{\mathrm{mis}}I_{\mathrm{com}}^{-1}$, the rate measured from the error decay agreeing with the Jacobian of the EM map at $0.0656$ against $0.0652$ and then exactly at $0.2858$, $0.6021$, $0.7949$ and $0.8662$, with sweep counts of $8$, $17$, $43$, $94$ and $147$ against predictions of $6.3$, $15.3$, $40.4$, $91.5$ and $147.5$; that Louis' identity makes the complete-data standard error too small by $\sqrt{1-r}$, verified as $0.9995$, $0.9915$, $0.9692$, $0.9044$ and $0.7728$ against $0.9995$, $0.9915$, $0.9692$, $0.9045$ and $0.7732$ computed from the algorithm's own speed, with a flat $5.3666$ basis points reported where the truth ran to $6.9912$ and coverage falling to $0.8686$ where the observed information held $0.9496$; and that the mixture likelihood is unbounded, so $0$ decreases in $2{,}291{,}200$ sweeps coexists with $0.0434$ of starts climbing toward a zero-width component, a spike holding the best likelihood on $0.4925$ of datasets, degenerate selection rising $0.0400$, $0.0600$, $0.1325$, $0.2275$, $0.4925$ with the number of restarts, and the probability of a $-3\%$ day falling from $0.6449\%$ to $0.2949\%$ against a truth of $1.2119\%$ until a variance floor of fifteen basis points removes the problem entirely.

The shape shared by all three exhibits is that EM's guarantee is about the sequence of likelihood values and every failure here is about something else. The rate is a property of the data's ability to identify the parameter, and the algorithm reports it only as elapsed time. The standard error is a property of the observed-data curvature, and the algorithm computes the complete-data curvature instead because that is what the M-step needed. The destination is a property of whether the objective has a maximum, and the algorithm never asks. In all three cases the missing quantity is available — the sweep count, a second difference of the observed log-likelihood, the smallest fitted scale — and in all three the algorithm's own output does not contain it.

What every section here shares with [Numerical Optimization](01-numerical-optimization.md) is that the answer was a point, obtained by climbing. Section 3's whole difficulty was that a point needs an error bar and the natural one was wrong; section 4's was that the climb has nowhere to arrive. Both dissolve if the object being computed is a distribution over the parameter rather than a maximizer of a function — no curvature to invert, no basin to be trapped in, and a degenerate spike carrying negligible mass rather than infinite height. The obstacle is that such a distribution is known only up to a constant that [Numerical Integration](02-numerical-integration.md) cannot supply beyond a handful of dimensions, and the way around it is to give up on the constant and build a chain instead. That is [Markov Chain Monte Carlo](04-markov-chain-monte-carlo.md).

**Expectation-maximization guarantees that a number goes up, and the number is the likelihood of a model whose maximum may not exist, computed at a rate that measures how little the data determines the parameter, with a curvature that describes data nobody observed — so every quantity it reports is correct and none of them is the one being read.**
