# Gibbs Sampling

Gibbs sampling is usually sold as the sampler with no tuning parameter, and the description is accurate and misleading in the same breath. There is no dial, so the three sections [Metropolis–Hastings](05-metropolis-hastings.md) spent on step sizes do not apply and nothing can be mistuned at run time. What replaces the dial is a decision taken when the model is written and never revisited — which coordinates are updated together — and its cost is set by a quantity nobody looks at before pressing run. Below, on the semi-conjugate model [Conjugate Priors](../part-16-bayesian-statistics/04-conjugate-priors.md) named in advance as "precisely the structure a Gibbs sampler consumes," the sampler accepts every proposal by construction and returns $0.9796$ effective draws per draw against a well-tuned random walk's $0.1209$. Then the bill. For a return regression on two factors, the integrated autocorrelation time is exactly $(1+\rho^{2})/(1-\rho^{2})$ in the posterior correlation, measured at $1.03$, $1.69$, $9.73$, $89.87$ and $733.92$ against predictions of $1.00$, $1.68$, $9.45$, $107.37$ and $1067.15$ — so two factors correlated at $0.999$ need $25{,}292{,}185{,}213$ draws for a standard error of one hundredth of a basis point, where drawing both loadings at once needs $34{,}847{,}581$. And on a hierarchy of fifty strategy Sharpe ratios, the textbook parameterization delivers an effective sample size of $644$ where an algebraically identical rewrite delivers $37{,}735$, at $\hat R$ values of $1.0042$ and $1.0000$ that no reader would separate.

This page covers the full-conditional update and its identity with a Metropolis step that never rejects, the semi-conjugate structure that makes it available, the exact relationship between posterior correlation and cost, the reparameterizations that change that cost without changing the model, and the case where every conditional is proper and the joint distribution does not exist. It establishes no invariance theory and no ergodic average, which is [Markov Chain Monte Carlo](04-markov-chain-monte-carlo.md); it derives no general acceptance ratio and tunes no proposal, which is [Metropolis–Hastings](05-metropolis-hastings.md); it constructs no conjugate family and proves no closure, which is [Conjugate Priors](../part-16-bayesian-statistics/04-conjugate-priors.md); it argues for no prior, which is [Prior Distributions](../part-16-bayesian-statistics/02-prior-distributions.md); it normalizes no posterior by quadrature, which is [Numerical Integration](02-numerical-integration.md); it maximizes no likelihood over latent variables, though section 5 notes the correspondence, which is [The EM Algorithm](03-em-algorithm.md); it preconditions no optimizer, which is [Numerical Optimization](01-numerical-optimization.md); it derives no forward–backward recursion for a state sequence, which is [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md); it estimates no covariance matrix and shrinks nothing by a fixed rule, which is [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md); and it never treats the absence of a tuning parameter as the absence of a choice.

The trading stake is a course lesson measuring the correlation regime a desk actually operates in, which turns out to be the regime this page prices. [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) prints `2020 COVID Feb-Apr                0.878       89.3%    1.12        66.6%` — average pairwise correlation of $0.878$ across nine sectors, a first principal component holding $89.3\%$ of the variance, and an effective breadth of $1.12$ independent assets out of nine. The lesson's point is that diversification evaporates exactly when it is needed. Section 3's point is that a componentwise sampler's efficiency is governed by the same number, so a posterior fitted during that regime is the one a coordinate-at-a-time sampler handles worst.

## A Full-Conditional Update Is a Metropolis Step Whose Acceptance Probability Is Identically One

The algorithm cycles: hold every coordinate but one fixed, draw that one from its conditional distribution given the rest and the data, move to the next. Nothing is rejected and nothing is tuned. Both facts have the same one-line explanation.

??? note "Proof that a full-conditional update is a Metropolis–Hastings step with acceptance identically one, and that a composition of invariant kernels is invariant while a composition of reversible kernels need not be reversible"

    Take the proposal that changes only block $j$, drawing $\theta_j'\sim\pi(\cdot\mid\theta_{-j})$ and leaving $\theta_{-j}$ alone. Substituting into the acceptance ratio of [Metropolis–Hastings](05-metropolis-hastings.md) and writing $\pi(\theta)=\pi(\theta_j\mid\theta_{-j})\pi(\theta_{-j})$,
    $$\alpha=\frac{\pi(\theta_j'\mid\theta_{-j})\pi(\theta_{-j})}{\pi(\theta_j\mid\theta_{-j})\pi(\theta_{-j})}\cdot\frac{\pi(\theta_j\mid\theta_{-j})}{\pi(\theta_j'\mid\theta_{-j})}=1,$$
    since the proposal density for the reverse move is the same conditional evaluated at the old value. Every factor cancels. So Gibbs is not an alternative to Metropolis–Hastings; it is the case where the proposal *is* the target's own conditional, and the acceptance ratio — the entire subject of the previous page — collapses to a constant.

    Each single-block update leaves $\pi$ invariant, since drawing $\theta_j$ from $\pi(\cdot\mid\theta_{-j})$ while $\theta_{-j}\sim\pi(\theta_{-j})$ reproduces the joint by definition of a conditional. Invariance is preserved under composition: if $\pi P_1=\pi$ and $\pi P_2=\pi$ then $\pi(P_1P_2)=(\pi P_1)P_2=\pi P_2=\pi$. So the **systematic scan** — update block $1$, then $2$, then $3$, in fixed order — is invariant.

    Reversibility is not preserved. Each $P_j$ satisfies detailed balance, but $P_1P_2$ generally does not: the probability of the path $\theta\to\theta'$ through an intermediate state differs from the reverse path because the blocks are visited in a fixed order, and reversing time reverses that order. This is a concrete instance of the gap [Markov Chain Monte Carlo](04-markov-chain-monte-carlo.md) proves in the abstract — detailed balance is sufficient for invariance, not necessary — and it is the reason the systematic scan needs the general argument above rather than the pairwise one. The **random scan**, which picks $j$ uniformly at each step, is a mixture of reversible kernels and therefore reversible, which is why theoretical results are usually stated for it and practical code usually is not.

    **The load-bearing consequence is that Gibbs has no free parameter because the proposal was determined by the model's factorization, so the choice a Metropolis user makes at run time was made here at specification time — by the person who decided which quantities were parameters and which conditionals could be sampled — and is not adjustable afterwards.**

## On the Model Part XVI Named in Advance, Gibbs Rejects Nothing and Delivers Eight Times the Effective Sample

Full conditionals are worth having only when they can be sampled. [Conjugate Priors](../part-16-bayesian-statistics/04-conjugate-priors.md) identified where that happens and predicted this page: placing independent normal and inverse-gamma priors on a mean and a variance "is not jointly conjugate, but each full conditional is, which is precisely the structure a Gibbs sampler consumes."

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(17061)
n, MU_T, SIG_T = 500, 0.00042, 0.0115
x = rng.normal(MU_T, SIG_T, n)
M0, T0, A0, B0 = 0.0, 0.0010, 3.0, 3.0 * 0.010 ** 2      # independent normal / inv-gamma
DRAWS, BURN, CHAINS = 20_000, 4_000, 4


def tau_int(v, M=2_000):
    v = v - v.mean()
    f = np.fft.rfft(v, 2 * len(v))
    ac = np.fft.irfft(f * np.conj(f))[:M].real
    ac /= ac[0]
    s, k = 1.0, 1
    while k + 1 < M and ac[k] + ac[k + 1] > 0:
        s += 2 * ac[k]
        k += 1
    return s


def gibbs():
    """Each full conditional is conjugate even though the joint prior is not."""
    mu = np.full(CHAINS, x.mean())
    v = np.full(CHAINS, x.var())
    out = np.empty((CHAINS, DRAWS, 2))
    for i in range(DRAWS):
        prec = 1 / T0 ** 2 + n / v
        m = (M0 / T0 ** 2 + x.sum() / v) / prec
        mu = m + rng.standard_normal(CHAINS) / np.sqrt(prec)
        ss = ((x - mu[:, None]) ** 2).sum(1)
        v = stats.invgamma.rvs(A0 + n / 2, scale=B0 + ss / 2, size=CHAINS,
                               random_state=rng)
        out[:, i] = np.stack([mu, np.sqrt(v)], 1)
    return out[:, BURN:]


def logpost(mu, sd):
    """The same posterior up to a constant, for the random-walk comparison."""
    v = sd ** 2
    return (-0.5 * ((x - mu[..., None]) ** 2).sum(-1) / v - n * np.log(sd)
            - 0.5 * (mu - M0) ** 2 / T0 ** 2 - (A0 + 1) * np.log(v) - B0 / v)


def rwm(scale):
    th = np.stack([np.full(CHAINS, x.mean()), np.full(CHAINS, x.std())], 1)
    lp = logpost(th[:, 0], th[:, 1])
    out = np.empty((CHAINS, DRAWS, 2))
    acc = 0
    for i in range(DRAWS):
        cand = th + scale * rng.standard_normal((CHAINS, 2))
        ok = cand[:, 1] > 0
        lpc = np.where(ok, logpost(cand[:, 0], np.abs(cand[:, 1])), -np.inf)
        take = (np.log(rng.random(CHAINS)) < lpc - lp) & ok
        th = np.where(take[:, None], cand, th)
        lp = np.where(take, lpc, lp)
        acc += take.sum()
        out[:, i] = th
    return out[:, BURN:], acc / (DRAWS * CHAINS)


g = gibbs()
best = max((rwm(np.array([5.2e-4, 3.6e-4]) * m) for m in (0.5, 1.0, 1.6)),
           key=lambda r: -np.mean([tau_int(r[0][c, :, 0]) for c in range(CHAINS)]))
r, acc = best

print(f"  a semi-conjugate normal model on {n} daily returns: an independent normal prior"
      f" on the mean and an inverse-gamma on the variance, which is conjugate in each"
      f" full conditional and not jointly. {CHAINS} chains x {DRAWS:,} draws,"
      f" {BURN:,} discarded")
print("     sampler                     acceptance   mean, bp   daily vol, bp"
      "   tau(mean)   tau(vol)   ESS/draws   posterior correlation")
for name, s, a in (("Gibbs", g, 1.0), ("random-walk Metropolis", r, acc)):
    tm = float(np.mean([tau_int(s[c, :, 0]) for c in range(CHAINS)]))
    tv = float(np.mean([tau_int(s[c, :, 1]) for c in range(CHAINS)]))
    flat = s.reshape(-1, 2)
    print(f"    {name:26s}   {a:10.4f}   {flat[:, 0].mean() * 1e4:8.4f}"
          f"   {flat[:, 1].mean() * 1e4:13.4f}   {tm:9.2f}   {tv:8.2f}"
          f"   {1 / max(tm, tv):9.4f}   {np.corrcoef(flat.T)[0, 1]:21.4f}")
# =>   a semi-conjugate normal model on 500 daily returns: an independent normal prior on the mean and an inverse-gamma on the variance, which is conjugate in each full conditional and not jointly. 4 chains x 20,000 draws, 4,000 discarded
#         sampler                     acceptance   mean, bp   daily vol, bp   tau(mean)   tau(vol)   ESS/draws   posterior correlation
#        Gibbs                            1.0000     4.5351        110.7589        1.00       1.02      0.9796                 -0.0206
#        random-walk Metropolis           0.3358     4.6687        110.6855        7.39       8.27      0.1209                 -0.0002
```

Both samplers agree on the posterior — a mean of $4.5351$ against $4.6687$ basis points and a daily volatility of $110.7589$ against $110.6855$ — which is the check that the Gibbs conditionals were derived correctly. What differs is everything else. Gibbs accepts $1.0000$ of its proposals because there is nothing to reject, and its integrated autocorrelation times are $1.00$ and $1.02$: the draws are effectively independent, $0.9796$ per draw. The random walk, tuned by the same sweep the previous page recommends and landing at an acceptance rate of $0.3358$ squarely inside the textbook band, delivers $0.1209$ — a factor of $8.1$.

The last column is why, and it is the whole of section 3 in advance. The posterior correlation between the mean and the volatility is $-0.0206$, essentially zero, because a symmetric distribution's location tells you almost nothing about its scale. A Gibbs sampler that updates the mean given the volatility and the volatility given the mean is, on a posterior with no correlation, drawing from the joint directly. **The rejection-free property is free; the near-independence is not, and it was a property of this posterior rather than of the algorithm.**

## The Efficiency Is the Posterior Correlation, Exactly, and Two Factors at 0.999 Cost Twenty-Five Billion Draws

The dependence between the coordinates being cycled is not merely the reason Gibbs was fast above. For the Gaussian case it is a closed form, which makes the cost of a badly parameterized model predictable before any sampler is run.

??? note "Proof that componentwise Gibbs on a bivariate normal has lag-one autocorrelation exactly $\rho^{2}$, hence $\tau=(1+\rho^{2})/(1-\rho^{2})$ and $\mathrm{ESS}/N=(1-\rho^{2})/(1+\rho^{2})$"

    Let $(\theta_1,\theta_2)$ be standard bivariate normal with correlation $\rho$. The conditionals are $\theta_1\mid\theta_2\sim N(\rho\theta_2,1-\rho^{2})$ and symmetrically. One systematic sweep from $\theta_1^{(t)}$ draws
    $$\theta_2^{(t+1)}=\rho\,\theta_1^{(t)}+\sqrt{1-\rho^{2}}\,Z_1,\qquad \theta_1^{(t+1)}=\rho\,\theta_2^{(t+1)}+\sqrt{1-\rho^{2}}\,Z_2,$$
    with $Z_1,Z_2$ independent standard normals. Substituting, $\theta_1^{(t+1)}=\rho^{2}\theta_1^{(t)}+\text{noise}$, so the $\theta_1$ marginal chain is an AR(1) with coefficient exactly $\rho^{2}$ — and being an AR(1), its lag-$k$ autocorrelation is $\rho^{2k}$.

    Summing the geometric series in the definition of integrated autocorrelation time,
    $$\tau=1+2\sum_{k\ge1}\rho^{2k}=1+\frac{2\rho^{2}}{1-\rho^{2}}=\frac{1+\rho^{2}}{1-\rho^{2}},$$
    so $\mathrm{ESS}/N=(1-\rho^{2})/(1+\rho^{2})$, which is $1$ at $\rho=0$ and diverges as $\lvert\rho\rvert\to1$. At $\rho=0.9$ it is $0.1050$; at $\rho=0.99$, $0.0100$; at $\rho=0.999$, $0.0010$.

    Two structural readings follow. The cost depends on $\rho$ only through $\rho^{2}$, so the sign is irrelevant and a strongly negative posterior correlation is exactly as expensive as a strongly positive one — which matters because regression loadings on positively correlated factors have *negatively* correlated posteriors. And updating both coordinates jointly from the exact bivariate conditional gives independent draws, $\tau=1$, at no additional cost per sweep: the blocked sampler is available whenever the joint conditional can be sampled, which for a Gaussian block is always.

    **The load-bearing quantity is a posterior correlation, which is not an input to the model, is not chosen by anyone, and is estimable from the draws the sampler has already produced — so the entire cost of a componentwise sampler is knowable from its own first thousand iterations and is almost never computed.**

Two factors that a desk would call near-duplicates put the formula to work:

```python
import numpy as np

rng = np.random.default_rng(17063)
n, SIG, DRAWS, BURN, CHAINS = 1_500, 0.010, 40_000, 8_000, 4
B_TRUE = np.array([0.0006, 0.0004])


def tau_int(v, M=8_000):
    v = v - v.mean()
    f = np.fft.rfft(v, 2 * len(v))
    ac = np.fft.irfft(f * np.conj(f))[:M].real
    ac /= ac[0]
    s, k = 1.0, 1
    while k + 1 < M and ac[k] + ac[k + 1] > 0:
        s += 2 * ac[k]
        k += 1
    return s


print(f"  a return regression on two factors with correlation r, {n:,} days, flat priors"
      f" and known noise scale, so the posterior for the two loadings is bivariate normal"
      f" with correlation rho. {CHAINS} chains x {DRAWS:,} draws, {BURN:,} discarded")
print("     factor correlation   posterior rho   predicted tau   measured tau"
      "   predicted ESS/draws   measured ESS/draws   blocked, measured"
      "   draws for 0.01bp   blocked draws for 0.01bp")
for r in (0.0, 0.5, 0.9, 0.99, 0.999):
    L = np.array([[1.0, 0.0], [r, np.sqrt(1 - r ** 2)]])
    F = rng.standard_normal((n, 2)) @ L.T
    y = F @ B_TRUE + rng.normal(0, SIG, n)
    A = F.T @ F / SIG ** 2                                  # posterior precision
    C = np.linalg.inv(A)
    bhat = C @ (F.T @ y) / SIG ** 2
    rho = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])

    b = np.tile(bhat, (CHAINS, 1))
    out = np.empty((CHAINS, DRAWS))
    for i in range(DRAWS):
        for j in (0, 1):
            k = 1 - j
            m = bhat[j] - A[j, k] / A[j, j] * (b[:, k] - bhat[k])
            b[:, j] = m + rng.standard_normal(CHAINS) / np.sqrt(A[j, j])
        out[:, i] = b[:, 0]
    keep = out[:, BURN:]
    tau = float(np.mean([tau_int(c) for c in keep]))

    R = np.linalg.cholesky(C)                               # both loadings at once
    blk = (bhat + rng.standard_normal((CHAINS, DRAWS, 2)) @ R.T)[:, BURN:, 0]
    tau_b = float(np.mean([tau_int(c) for c in blk]))

    pred = (1 + rho ** 2) / (1 - rho ** 2)
    unit = (np.sqrt(C[0, 0]) * 1e4 / 0.01) ** 2
    print(f"    {r:18.3f}   {rho:13.4f}   {pred:13.2f}   {tau:12.2f}"
          f"   {1 / pred:20.5f}   {1 / tau:18.5f}   {1 / tau_b:17.5f}"
          f"   {unit * tau:18,.0f}   {unit * tau_b:24,.0f}")
# =>   a return regression on two factors with correlation r, 1,500 days, flat priors and known noise scale, so the posterior for the two loadings is bivariate normal with correlation rho. 4 chains x 40,000 draws, 8,000 discarded
#         factor correlation   posterior rho   predicted tau   measured tau   predicted ESS/draws   measured ESS/draws   blocked, measured   draws for 0.01bp   blocked draws for 0.01bp
#                     0.000         -0.0277            1.00           1.03                0.99847              0.96962             0.99775               68,089                     66,169
#                     0.500         -0.5024            1.68           1.69                0.59693              0.59053             0.99396              147,252                     87,485
#                     0.900         -0.8992            9.45           9.73                0.10584              0.10277             0.99411            3,365,245                    347,902
#                     0.990         -0.9907          107.37          89.87                0.00931              0.01113             0.97740          308,686,011                  3,514,056
#                     0.999         -0.9991         1067.15         733.92                0.00094              0.00136             0.98893       25,292,185,213                 34,847,581
```

The prediction and the measurement are the same number. Integrated autocorrelation time comes out at $1.03$, $1.69$, $9.73$, $89.87$ and $733.92$ against a closed form giving $1.00$, $1.68$, $9.45$, $107.37$ and $1067.15$, the last two rows running low because an autocorrelation time near a thousand cannot be estimated cleanly from a window of this length — an understatement in the direction that flatters the sampler. Note the sign: correlating the factors at $+0.999$ produces a posterior correlation of $-0.9991$ between their loadings, because when two regressors are nearly identical the data determines their sum and says almost nothing about their difference. The formula depends on $\rho^{2}$, so the sign changes nothing.

The last two columns convert autocorrelation into work. To pin the first loading down to a Monte Carlo standard error of one hundredth of a basis point, a componentwise sampler needs $68{,}089$ draws at zero correlation, $3{,}365{,}245$ at $0.9$, and $25{,}292{,}185{,}213$ — twenty-five billion — at $0.999$. Drawing both loadings jointly from their bivariate conditional needs $66{,}169$, $347{,}902$ and $34{,}847{,}581$ respectively, and the measured blocked efficiency is $0.99775$, $0.99396$, $0.99411$, $0.97740$ and $0.98893$ per draw regardless of the correlation. **Blocking costs one Cholesky factorization of a two-by-two matrix and buys a factor of $9.7$ at the crisis correlation the trading stake reports and $726$ at $0.999$.** Nothing about the model, the prior or the data changed; the two samplers target the identical posterior and differ only in which coordinates move together.

## The One Decision Gibbs Makes Is Made When the Model Is Written, and Costs a Factor of Fifty-Nine That No Diagnostic Attributes

Blocking is the visible lever because it is a change to the sampler. The invisible one is a change to the *coordinates*, made when the model was specified, algebraically irrelevant, and worth as much.

The hierarchical construction is the standard case, and it is the one [Prior Distributions](../part-16-bayesian-statistics/02-prior-distributions.md) recommended and [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) said was "solved by neither" quadrature nor importance sampling:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(17065)
J, DRAWS, BURN, CHAINS = 50, 40_000, 8_000, 4
SE, MU_T = 0.25, 0.35                      # standard error of each Sharpe; true centre


def rhat(v):
    n = v.shape[1]
    W = v.var(1, ddof=1).mean()
    B = v.mean(1).var(ddof=1) * n
    return np.sqrt(((n - 1) / n * W + B / n) / W)


def tau_int(v, M=8_000):
    v = v - v.mean()
    f = np.fft.rfft(v, 2 * len(v))
    ac = np.fft.irfft(f * np.conj(f))[:M].real
    ac /= ac[0]
    s, k = 1.0, 1
    while k + 1 < M and ac[k] + ac[k + 1] > 0:
        s += 2 * ac[k]
        k += 1
    return s


def centered(y):
    """theta_j drawn given tau, then tau given theta: the textbook parameterization."""
    mu = np.full(CHAINS, y.mean())
    tau = np.full(CHAINS, y.std())
    out = np.empty((CHAINS, DRAWS))
    for i in range(DRAWS):
        prec = 1 / SE ** 2 + 1 / tau[:, None] ** 2
        m = (y / SE ** 2 + mu[:, None] / tau[:, None] ** 2) / prec
        th = m + rng.standard_normal((CHAINS, J)) / np.sqrt(prec)
        mu = th.mean(1) + tau * rng.standard_normal(CHAINS) / np.sqrt(J)
        S = ((th - mu[:, None]) ** 2).sum(1)
        tau = np.sqrt(stats.invgamma.rvs((J - 1) / 2, scale=S / 2, size=CHAINS,
                                         random_state=rng))
        out[:, i] = tau
    return out[:, BURN:]


def noncentered(y):
    """theta_j = mu + tau * eta_j, so eta and tau are drawn without seeing each other."""
    mu = np.full(CHAINS, y.mean())
    tau = np.full(CHAINS, y.std())
    out = np.empty((CHAINS, DRAWS))
    for i in range(DRAWS):
        prec = tau[:, None] ** 2 / SE ** 2 + 1.0
        m = (tau[:, None] * (y - mu[:, None]) / SE ** 2) / prec
        eta = m + rng.standard_normal((CHAINS, J)) / np.sqrt(prec)
        mu = ((y - tau[:, None] * eta).mean(1)
              + SE * rng.standard_normal(CHAINS) / np.sqrt(J))
        p = (eta ** 2).sum(1) / SE ** 2
        mt = (eta * (y - mu[:, None])).sum(1) / SE ** 2 / p
        tau = np.abs(mt + rng.standard_normal(CHAINS) / np.sqrt(p))
        out[:, i] = tau
    return out[:, BURN:]


print(f"  {J} strategy Sharpe ratios, each estimated with a standard error of {SE},"
      f" sharing an unknown cross-sectional dispersion tau. The shrinkage a desk applies"
      f" is B = SE^2/(SE^2+tau^2). {CHAINS} chains x {DRAWS:,} draws, {BURN:,} discarded")
print("     true tau   true shrinkage   centered: median tau   smallest tau reached"
      "   R-hat   ESS   implied shrinkage   non-centered: median tau   R-hat   ESS"
      "   implied shrinkage")
for tau_t in (0.00, 0.05, 0.10, 0.30):
    theta = MU_T + tau_t * rng.standard_normal(J)
    y = theta + SE * rng.standard_normal(J)
    truth = SE ** 2 / (SE ** 2 + tau_t ** 2)
    row = []
    for f in (centered, noncentered):
        t = f(y)
        row.append((np.median(t), t.min(), rhat(t),
                    t.size / float(np.mean([tau_int(c) for c in t])),
                    SE ** 2 / (SE ** 2 + np.median(t) ** 2)))
    c, nc = row
    print(f"    {tau_t:8.2f}   {truth:14.4f}   {c[0]:20.4f}   {c[1]:20.4f}"
          f"   {c[2]:6.4f}   {c[3]:5.0f}   {c[4]:17.4f}   {nc[0]:25.4f}"
          f"   {nc[2]:6.4f}   {nc[3]:5.0f}   {nc[4]:18.4f}")
# =>   50 strategy Sharpe ratios, each estimated with a standard error of 0.25, sharing an unknown cross-sectional dispersion tau. The shrinkage a desk applies is B = SE^2/(SE^2+tau^2). 4 chains x 40,000 draws, 8,000 discarded
#         true tau   true shrinkage   centered: median tau   smallest tau reached   R-hat   ESS   implied shrinkage   non-centered: median tau   R-hat   ESS   implied shrinkage
#            0.00           1.0000                 0.0546                 0.0000   1.0042     644              0.9544                      0.0608   1.0000   37735               0.9442
#            0.05           0.9615                 0.1007                 0.0001   1.0003     683              0.8603                      0.1007   1.0000   25201               0.8603
#            0.10           0.8621                 0.0848                 0.0007   1.0039    1091              0.8968                      0.0821   1.0000   29070               0.9027
#            0.30           0.4098                 0.3145                 0.1095   1.0000   27038              0.3871                      0.3137   1.0000   37143               0.3885
```

The two parameterizations describe the same model. Writing $\theta_j$ directly with prior $N(\mu,\tau^{2})$ and writing $\theta_j=\mu+\tau\eta_j$ with $\eta_j\sim N(0,1)$ define identical joint distributions, and the tables confirm it: median $\tau$ of $0.0546$ against $0.0608$, then $0.1007$ against $0.1007$, $0.0848$ against $0.0821$, $0.3145$ against $0.3137$, and implied shrinkage factors agreeing to two decimals throughout. **Both samplers are right, which is what makes the comparison a fair one.**

The effective sample sizes are $644$ and $37{,}735$ in the first row, $683$ and $25{,}201$ in the second, $1{,}091$ and $29{,}070$ in the third — a factor of $59$, $37$ and $27$ — and then $27{,}038$ against $37{,}143$ in the last, where the gap almost closes. The pattern is the funnel: when $\tau$ is genuinely small the strategies' true Sharpe ratios are nearly identical, the conditional for each $\theta_j$ given $\tau$ is extremely tight, and the conditional for $\tau$ given the $\theta_j$ is pinned by their observed spread. Neither coordinate can move without the other, so the centered sampler takes small steps down a narrowing neck. Writing $\eta_j$ instead makes $\tau$ and $\eta$ *a priori* independent, and the neck disappears.

The last row is the reassuring one and the one that makes the failure hard to catch: at $\tau=0.30$, where the strategies genuinely differ, the centered sampler is nearly as good and a practitioner who tested there would conclude the parameterization does not matter. It matters exactly in the regime a desk cares about — fifty backtests whose apparent dispersion might be entirely noise, which is the case for deciding whether to shrink them all to the mean.

**$\hat R$ reads $1.0042$, $1.0003$, $1.0039$ and $1.0000$ for the centered sampler and $1.0000$ throughout for the non-centered, so the convergence diagnostic separates them by four thousandths and no reader would act on that.** The effective sample size does show the difference and is the honest signal, but it reports only that this chain is slow — it does not report that an algebraically identical rewrite, costing four lines, is fifty-nine times faster. That inference is not available from any output the sampler produces.

??? note "Proof that every full conditional can be proper while the joint distribution does not exist, so a Gibbs sampler can run indefinitely, pass every diagnostic, and be sampling from nothing"

    Consider the pair of conditionals on the positive quadrant
    $$f(x\mid y)=y\,e^{-xy},\qquad f(y\mid x)=x\,e^{-xy},\qquad x,y>0.$$
    Each is a perfectly proper exponential density — the first is $\mathrm{Exp}(y)$ in $x$, the second $\mathrm{Exp}(x)$ in $y$ — so both steps of a Gibbs sampler are well defined and easy to implement. Any joint density compatible with them must satisfy $f(x,y)\propto e^{-xy}$, and
    $$\int_0^{\infty}\!\!\int_0^{\infty}e^{-xy}\,\mathrm{d}x\,\mathrm{d}y=\int_0^{\infty}\frac{1}{y}\,\mathrm{d}y=\infty,$$
    which diverges at both ends. No joint distribution exists, so there is nothing for the chain to be invariant with respect to and no stationary distribution to converge to.

    The chain nevertheless runs. It produces numbers, they have an empirical distribution, and that distribution drifts without settling — slowly enough that a finite run looks like a slowly mixing but functional sampler. Every diagnostic is computed from the draws, so $\hat R$ across chains started nearby will be near one, and the effective sample size will merely be small. This is the same blindness [Markov Chain Monte Carlo](04-markov-chain-monte-carlo.md) measures for a chain trapped in one mode, in its most extreme form: there is no target at all.

    The practical route in is an improper prior. Placing a flat prior on a variance component in a hierarchical model can leave every full conditional proper while the posterior is improper, which is why the condition is checked analytically and not by running the sampler. **The load-bearing point is that Gibbs only ever consults conditionals, so it cannot detect a property — the existence of the joint — that no conditional contains, and the check has to happen on paper before the model is coded.**

## Blocking, Collapsing and Reparameterizing Are the Levers, and All Three Rewrite the Model Rather Than the Sampler

Section 3's repair and section 4's repair look different and are the same move: change what the coordinates are so that the conditionals stop fighting. That is the whole of Gibbs tuning, and it happens in the model file.

**Blocking** updates correlated coordinates jointly, which section 3 prices at a factor of $726$ and which is available whenever the joint conditional is tractable — always for a Gaussian block, and for regression loadings, factor exposures and transition-matrix rows in practice. **Collapsing** integrates a parameter out analytically and samples the rest from the marginal; Liu's theorem guarantees it never increases the autocorrelation, and it is why a conjugate scale parameter is usually integrated away rather than sampled. **Reparameterizing** is section 4, and the standard advice inverts with the data: the non-centred form wins when the likelihood is weak relative to the prior, the centred form when it is strong, and the interpolation between them is a modelling decision made by whoever writes the file. **Metropolis-within-Gibbs** covers the case where one conditional cannot be sampled directly — a single [Metropolis–Hastings](05-metropolis-hastings.md) step for that block, with the rest of the sweep unchanged, inheriting invariance from the composition argument in section 1.

Two structural connections are worth naming because Part XVIII will need them. **Data augmentation** introduces latent variables precisely so that the conditionals become tractable, which is the same construction [The EM Algorithm](03-em-algorithm.md) uses for the same reason — the E-step's imputation and the Gibbs sampler's latent draw are the same object, averaged in one case and sampled in the other. And **forward-filtering, backward-sampling** draws a whole latent state sequence in one block by running the forward recursion of [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md) and then sampling backwards, which is blocking applied to a time series and the reason a regime model's transition matrix can be given a posterior rather than a point estimate.

!!! note "Blocking, collapsing, reparameterizing and thinning are the four responses to a slow Gibbs chain, and three of them help"
    They are grouped together in practice and only three of them are doing anything. **Blocking** updates correlated coordinates jointly, removing the $\rho^{2}$ autocorrelation of section 3 entirely at the cost of sampling a joint conditional; it strictly reduces autocorrelation and is the first thing to try. **Collapsing** integrates a parameter out and samples the smaller model, which by Liu's theorem never does worse than the componentwise sweep and usually does better, at the cost of an analytic integral. **Reparameterizing** leaves the model's distribution unchanged and changes only which quantities are named, which section 4 prices at a factor of $59$ and which costs nothing but the algebra. **Thinning** keeps every $k$-th draw; it raises the effective sample size *per retained draw*, which is what makes it feel like a repair, and lowers it in absolute terms, because discarding $k-1$ of every $k$ draws discards information the estimator would have used. It is justified when storage rather than computation is the binding constraint and essentially never otherwise. Treating thinning as a cure for autocorrelation is the error this list exists to prevent.

!!! warning "Gibbs has no dial and therefore no tuning diagnostic, so its one decision is invisible in the output and its cost is a posterior correlation nobody computed"
    Nothing in the failing cases looked wrong. Section 3's componentwise sampler on factors correlated at $0.999$ produced perfectly valid draws with a correct posterior mean and an integrated autocorrelation time of $733.92$, requiring $25{,}292{,}185{,}213$ draws for a target precision that blocking reaches in $34{,}847{,}581$. Section 4's centred hierarchy returned $\hat R$ values of $1.0042$, $1.0003$, $1.0039$ and $1.0000$ — indistinguishable from the non-centred sampler's $1.0000$ — while running $59$ times slower on the regime that matters, and it looked identical to its faster twin on the regime where the parameterization is irrelevant. And the improper-posterior case runs forever, reports a small effective sample size, and is sampling from a distribution that does not exist. **The free diagnostic is to compute the posterior correlation matrix of your parameters from the draws you already have and read the ceiling straight off it: any pair above $0.9$ in absolute value caps your effective sample size per draw at $(1-\rho^{2})/(1+\rho^{2})$ — $0.1050$ at $0.9$ and $0.0010$ at $0.999$ — so block that pair or reparameterize before spending another draw, and check on paper that the posterior is proper, because no output the sampler produces can tell you it is not.** The correlation matrix costs one line on a sample already in memory, and it is the only number on this page that converts a slow chain into a specific instruction.

## A Sampler With No Dial and One Decision

This page established that a full-conditional update is a Metropolis–Hastings step whose acceptance probability is identically one, making Gibbs the special case where the proposal is the target's own conditional, and that a composition of invariant kernels is invariant while a composition of reversible ones need not be — so the systematic scan is correct and not reversible; that on the semi-conjugate model [Conjugate Priors](../part-16-bayesian-statistics/04-conjugate-priors.md) predicted, Gibbs accepts $1.0000$ of proposals and delivers $0.9796$ effective draws per draw against a well-tuned random walk's $0.1209$, because the posterior correlation between mean and volatility is $-0.0206$; that the cost is the posterior correlation exactly, $\tau=(1+\rho^{2})/(1-\rho^{2})$ measured at $1.03$, $1.69$, $9.73$, $89.87$ and $733.92$ against predictions of $1.00$, $1.68$, $9.45$, $107.37$ and $1067.15$, so factors correlated at $0.999$ need $25{,}292{,}185{,}213$ draws where blocking needs $34{,}847{,}581$; and that an algebraically identical rewrite of a fifty-strategy hierarchy changes the effective sample size from $644$ to $37{,}735$ while $\hat R$ moves from $1.0042$ to $1.0000$, with every full conditional capable of being proper while the joint distribution fails to exist at all.

The shape shared by all three exhibits is that Gibbs moved the tuning problem from run time to specification time and thereby made it invisible. A Metropolis user who mistunes a step size sees an acceptance rate outside the expected band; a Gibbs user who parameterizes badly sees a chain that is simply slow, with no indication that the slowness is a property of the coordinates rather than of the problem. In every case the diagnostic that would have identified the cause — the posterior correlation matrix — is computable from draws already in memory, and in every case the algorithm has no reason to compute it, because the algorithm never needed the joint distribution for anything.

That absence is where this part ends. Nothing in the last three pages required knowing the normalizing constant, which is precisely what made them work, and nothing in them supplies a probability, a loss, a position size or a price. What has been built is a way to put a distribution on a parameter that no closed form reaches: a regime model's transition matrix rather than its point estimate, a stochastic volatility path rather than a filtered mean, a tail quantile with its own uncertainty attached rather than a plug-in. Turning those into a limit, a hedge ratio, a capital charge or a bet size is the application, and it is [Part XVIII](../part-18-quant-finance-applications/index.md).

**Gibbs sampling removes the tuning parameter by fixing the proposal to the model's own conditionals, so the choice does not disappear — it moves to the moment the model is written, where it is made by someone thinking about parameters rather than about mixing, and where nothing downstream will report what it cost.**
