# Prior Distributions

A prior is usually defended as the weak link that vanishes with enough data, and the defence concedes the wrong point. The problem is not that a prior is subjective; it is that the priors chosen to *avoid* being subjective are the most opinionated ones available, and they are opinionated about quantities nobody inspected. Below, three analysts who each refuse to assume anything — one working in volatility, one in variance, one in log volatility — report posterior median volatilities differing by $35.62\%$ on the same five observations, because "flat" names a coordinate system rather than a state of ignorance. Jeffreys' rule is the only one of the three that survives the change of variables, and it is nearly unbiased where the others run $13.86\%$ and $35.62\%$ high. Worse, the prior a course lesson calls "agnostic" turns out, when pushed through the model it belongs to, to assert a median absolute annualized Sharpe of $10.7282$ and an $0.8498$ probability that the strategy's Sharpe exceeds three in magnitude. The one construction here that gets better rather than worse as the problem grows is the one that stops pretending: a hierarchical prior whose width is estimated from the family, which beats both fixed alternatives by $0.5936$ once the family is genuinely heterogeneous.

This page covers what a prior has to be, why flatness is not neutrality, Jeffreys' rule and the invariance that motivates it, prior predictive checking as the only reliable way to read a prior, hierarchical priors whose width the data supplies, and improper priors together with the propriety their use requires. It computes no posterior summary and minimizes no expected loss, which is [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md); it takes no posterior mode and derives no ridge or lasso correspondence, which is [Maximum A Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md); it proves no closure and derives no conjugate family, which is [Conjugate Priors](04-conjugate-priors.md); it normalizes no posterior and establishes no asymptotics, which is [Posterior Distributions](03-posterior-distributions.md); it updates nothing sequentially, which is [Bayesian Updating](05-bayesian-updating.md); it computes no Bayes factor and no marginal likelihood for a model, which is [Bayesian Model Comparison](06-bayesian-model-comparison.md); it forecasts no observable it has not already integrated, which is [Bayesian Prediction](07-bayesian-prediction.md); it proves no inadmissibility and does not re-derive James–Stein, which is [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md); it derives no Fisher information from first principles, which is [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md); and it never describes a prior as uninformative without saying informative about what.

The trading stake is a course lesson publishing its prior sensitivity, drawing the right conclusion, and using one word this page is going to contest. [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) runs three priors over one momentum series and prints `skeptical  N(0, 1bp):  post ann mean +1.7%, P(edge > 0) 0.79`, `agnostic   N(0, 100bp):  post ann mean +5.8%, P(edge > 0) 0.93` and `optimistic N(4bp, 2bp):  post ann mean +7.4%, P(edge > 0) 0.99`, concluding that "when the sensitivity table disagrees this much, the data has not settled the question, and *that* is the finding." The conclusion is right. The word is `agnostic`, and section 3 pushes that prior through the model to find it the most extravagant of the three by a wide margin — not the neutral row in the table, but the row asserting the strategy is almost certainly one of the greatest ever built or one of the worst.

## There Is No Non-Informative Prior, Because Flatness Is a Property of a Coordinate System Rather Than of a Belief

The intuition behind a flat prior is that assigning equal density everywhere expresses no preference. The intuition is coherent for a finite parameter set, where equal probability on each of $k$ values is genuinely symmetric and survives any relabelling. For a continuous parameter it fails immediately, and the failure is not subtle.

??? note "Proof that a flat density is preserved only under affine reparameterization, so uniformity expresses a commitment to a coordinate system rather than an absence of belief"

    Let $\theta$ have density $\pi_\theta$ and let $\varphi=g(\theta)$ for a smooth strictly monotone $g$. The change-of-variables formula of [Change of Variables](../part-03-random-variables/09-change-of-variables.md) gives the density of $\varphi$ as
    $$\pi_\varphi(\varphi)=\pi_\theta\big(g^{-1}(\varphi)\big)\left|\frac{\mathrm{d}g^{-1}}{\mathrm{d}\varphi}\right|.$$
    Suppose $\pi_\theta\equiv c$ is flat. Then $\pi_\varphi(\varphi)=c\,|\mathrm{d}g^{-1}/\mathrm{d}\varphi|$, which is constant in $\varphi$ if and only if $g^{-1}$ has constant derivative — that is, if and only if $g$ is affine. For any nonlinear reparameterization the flat prior becomes non-flat, and the direction is set by the curvature of $g$: a prior flat in $\sigma$ puts density proportional to $1/(2\sqrt{v})$ on $v=\sigma^{2}$, favouring small variances, while a prior flat in $v$ puts density proportional to $2\sigma$ on $\sigma$, favouring large volatilities.

    There is no coordinate-free notion of "uniform" on a non-compact continuum, because uniformity is defined relative to a measure and the parameter space carries no canonical one. **An analyst who writes down a flat prior has not declined to express a belief; they have expressed the belief that their particular parameterization is the one in which ignorance looks symmetric, which is a strong claim about a coordinate system chosen for algebraic convenience.**

    The load-bearing consequence is that the choice is invisible in the output. A posterior computed under a flat prior on $\sigma$ and one computed under a flat prior on $\sigma^{2}$ are both reported as "flat prior" results, both are normalized distributions over the same physical quantity, and nothing in either display records which convention produced it.

The repair Jeffreys proposed is to build the prior out of something that already transforms correctly, and the Fisher information of [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md) is exactly such an object.

??? note "Proof that $\pi_J(\theta)\propto\sqrt{\det I(\theta)}$ transforms as a density under any smooth reparameterization, so it selects the same measure whatever coordinates are used, which is Jeffreys' rule"

    Fisher information for a scalar parameter is $I(\theta)=\mathbb{E}_\theta[(\partial_\theta\log f(X\mid\theta))^{2}]$. Under $\varphi=g(\theta)$ the chain rule gives $\partial_\varphi\log f=\partial_\theta\log f\cdot(\mathrm{d}\theta/\mathrm{d}\varphi)$, so
    $$I(\varphi)=I(\theta)\left(\frac{\mathrm{d}\theta}{\mathrm{d}\varphi}\right)^{2},\qquad\text{hence}\qquad \sqrt{I(\varphi)}=\sqrt{I(\theta)}\left|\frac{\mathrm{d}\theta}{\mathrm{d}\varphi}\right|.$$
    The right-hand side is precisely how a density must transform. Therefore if $\pi_J(\theta)\propto\sqrt{I(\theta)}$ is asserted in $\theta$-coordinates, the induced density on $\varphi$ is proportional to $\sqrt{I(\varphi)}$ — the same rule, stated in the new coordinates. In $p$ dimensions the same argument with the Jacobian matrix $J$ gives $I(\varphi)=J^{\top}I(\theta)J$ and $\sqrt{\det I(\varphi)}=\sqrt{\det I(\theta)}\,|\det J|$, which is again the transformation law for a density.

    The rule is therefore a *procedure* that commutes with reparameterization rather than a distribution that happens to look symmetric in one chart. Two consequences are worth having. For a location parameter $I$ is constant and Jeffreys returns the flat prior, which is why the flat prior's bad reputation is undeserved in the one case people usually meet it. For the scale of a mean-zero normal with $n$ observations, $\log f=-n\log\sigma-\sum x_i^{2}/(2\sigma^{2})+\text{const}$ gives $I(\sigma)=2n/\sigma^{2}$, so
    $$\pi_J(\sigma)\propto 1/\sigma,$$
    which is flat in $\log\sigma$ and is the unique choice among the three the next section measures that gives the same answer in all three charts.

    **Jeffreys' rule buys invariance and nothing else: it is not the prior of least information in any decision-theoretic sense, it is frequently improper, and in several dimensions it produces estimators poor enough that Jeffreys himself recommended against applying it coordinate-wise.** The load-bearing fact is that invariance is a property one can check, whereas non-informativeness is not, and a rule delivering a checkable property is worth more than a rule delivering a comforting adjective.

## Three Analysts Who Each Decline to Assume Anything, in Three Different Coordinates, Report Volatilities That Differ by a Third

The disagreement is not a limiting curiosity. It is largest exactly where volatility estimation is hardest, which is where a new strategy lives:

```python
import numpy as np
from scipy import stats, special

rng = np.random.default_rng(16021)
sig, reps = 0.010, 20_000

# Volatility from n mean-zero returns. A prior flat in one coordinate has density
# proportional to sigma^-a in sigma, and the posterior for sigma^2 is then inverse-gamma
# with shape (n + a - 1)/2 and scale S/2. Only a = 1, Jeffreys, is coordinate-free.
COORD = (("flat in sigma", 0), ("flat in sigma-squared", -1), ("Jeffreys, flat in log sigma", 1))

print(f"  volatility of a mean-zero return series, true sigma = {sig * 1e4:.0f}bp,"
      f" {reps:,} datasets; every summary is divided by the truth")
print("        n   prior                          median      mean   P(sigma > 1.5 true)"
      "   median vs Jeffreys")
for n in (5, 10, 30, 100):
    x = rng.standard_normal((reps, n)) * sig
    S = (x ** 2).sum(1)
    row = []
    for nm, a in COORD:
        al = (n + a - 1) / 2
        v = stats.invgamma(al, scale=S / 2)
        emean = np.exp(0.5 * np.log(S / 2) + special.gammaln(al - 0.5) - special.gammaln(al))
        row.append((nm, np.sqrt(v.ppf(0.5)).mean() / sig, emean.mean() / sig,
                    v.sf((1.5 * sig) ** 2).mean()))
    ref = row[-1][1]
    for nm, med, mean, tail in row:
        print(f"    {n:5d}   {nm:28s}  {med:7.4f}  {mean:8.4f}   {tail:18.4f}"
              f"   {med / ref - 1:+18.2%}")
# =>   volatility of a mean-zero return series, true sigma = 100bp, 20,000 datasets; every summary is divided by the truth
#            n   prior                          median      mean   P(sigma > 1.5 true)   median vs Jeffreys
#            5   flat in sigma                  1.1575    1.3289               0.2921              +13.86%
#            5   flat in sigma-squared          1.3787    1.6920               0.4257              +35.62%
#            5   Jeffreys, flat in log sigma    1.0166    1.1280               0.1952               +0.00%
#           10   flat in sigma                  1.0683    1.1255               0.1521               +5.82%
#           10   flat in sigma-squared          1.1386    1.2085               0.2088              +12.78%
#           10   Jeffreys, flat in log sigma    1.0095    1.0574               0.1089               +0.00%
#           30   flat in sigma                  1.0190    1.0343               0.0193               +1.75%
#           30   flat in sigma-squared          1.0375    1.0536               0.0254               +3.59%
#           30   Jeffreys, flat in log sigma    1.0015    1.0160               0.0145               +0.00%
#          100   flat in sigma                  1.0063    1.0106               0.0000               +0.51%
#          100   flat in sigma-squared          1.0115    1.0158               0.0000               +1.02%
#          100   Jeffreys, flat in log sigma    1.0012    1.0054               0.0000               +0.00%
```

Every row is a defensible analyst. The first has a parameter called volatility and puts a flat prior on it; the second has a parameter called variance and puts a flat prior on that; the third works in log volatility, which is what Jeffreys' rule prescribes for a scale. None of them has looked at the data before choosing, and none believes they have assumed anything.

At five observations the posterior median volatility comes out at $1.1575$, $1.3787$ and $1.0166$ times the truth. The variance-coordinate analyst is high by $35.62\%$ and the volatility-coordinate analyst by $13.86\%$, and both are high for the reason the first proof gives: the Jacobian carrying a flat prior from one chart to the other tilts density towards larger values, and the tilt is not corrected by five observations. The decision-relevant column is worse than the point estimate. Asked for the probability that true volatility exceeds one and a half times its actual value — a risk-limit question — the three report $0.2921$, $0.4257$ and $0.1952$. One analyst thinks that event is within easy reach and another thinks it is a one-in-five tail, from identical data and identical models.

The disagreement decays, and the rate is the honest part of the table. The gap against Jeffreys runs $+13.86\%$ and $+35.62\%$ at $n=5$, then $+5.82\%$ and $+12.78\%$, then $+1.75\%$ and $+3.59\%$, and finally $+0.51\%$ and $+1.02\%$ at a hundred observations. It falls like $1/n$, which is the rate the asymptotic argument of [The Bayesian Framework](01-bayesian-framework.md) promises for any prior's influence. **The coordinate problem is therefore a small-sample problem, and small samples are exactly the regime in which a desk is asked whether a new strategy's volatility is under control.** Jeffreys' column is the only one whose median tracks the truth throughout, at $1.0166$, $1.0095$, $1.0015$ and $1.0012$, and it earns that not by being uninformative but by being the same measure in every chart.

## A Prior Is a Statement About Everything the Parameter Implies, and the Only Way to Read One Is to Push It Through the Model

Section 2 compared priors that were all trying to say nothing. The more common failure is a prior deliberately widened to seem harmless, whose consequences were never examined because the parameter it constrains is not a quantity anyone has intuitions about. A daily mean return of one hundred basis points is a number without an obvious scale. The Sharpe ratio it implies is not:

```python
import numpy as np

rng = np.random.default_rng(16023)
sig, days, reps = 0.010, 252, 200_000

# A prior on the daily mean is a prior on everything the daily mean implies. Push each
# one through a year of the model and read the observables it predicts.
PRIORS = (("skeptical  N(0, 1bp)", 0.0, 0.0001),
          ("weak       N(0, 3bp)", 0.0, 0.0003),
          ("agnostic   N(0, 100bp)", 0.0, 0.0100),
          ("optimistic N(4bp, 2bp)", 0.0004, 0.0002))

print(f"  each prior on the daily mean pushed through {days} days at {sig * 1e4:.0f}bp"
      f" of daily volatility, {reps:,} draws; every column is an observable")
print("     prior                    median |S|   P(|S|>1)   P(|S|>3)   P(|S|>10)"
      "   1% annual   99% annual")
for nm, m, s in PRIORS:
    mu = rng.normal(m, s, reps)
    r = rng.standard_normal((reps, days)) * sig + mu[:, None]
    tot = r.sum(1)
    sh = r.mean(1) / r.std(1, ddof=1) * np.sqrt(days)
    print(f"    {nm:24s}   {np.median(np.abs(sh)):10.4f}   {(np.abs(sh) > 1).mean():8.4f}"
          f"   {(np.abs(sh) > 3).mean():8.4f}   {(np.abs(sh) > 10).mean():9.4f}"
          f"   {np.quantile(tot, 0.01):+9.2%}   {np.quantile(tot, 0.99):+10.2%}")
# =>   each prior on the daily mean pushed through 252 days at 100bp of daily volatility, 200,000 draws; every column is an observable
#         prior                    median |S|   P(|S|>1)   P(|S|>3)   P(|S|>10)   1% annual   99% annual
#        skeptical  N(0, 1bp)           0.6844     0.3237     0.0033      0.0000     -37.48%      +37.29%
#        weak       N(0, 3bp)           0.7475     0.3653     0.0071      0.0000     -40.95%      +40.95%
#        agnostic   N(0, 100bp)        10.7282     0.9502     0.8498      0.5302    -589.13%     +583.96%
#        optimistic N(4bp, 2bp)         0.8463     0.4250     0.0132      0.0000     -28.98%      +48.81%
```

This is a prior predictive check, and it consists entirely of taking the prior seriously enough to ask what it predicts. Draw a mean from the prior, simulate a year at that mean, and look at what comes out. No data is used and none is needed — the check is available before the first observation arrives, which is what makes it free.

The lesson's `skeptical` prior implies a median absolute annualized Sharpe of $0.6844$, a $0.3237$ chance of exceeding one in magnitude, and annual outcomes whose central ninety-eight per cent span $-37.48\%$ to $+37.29\%$. Those are the beliefs of somebody who has seen strategies before. The `weak` row, a three-basis-point prior, is barely different at $0.7475$ and $0.3653$, which is the useful discovery that weakly informative priors are cheap: tightening from a hundred basis points to three costs almost nothing in expressed open-mindedness.

The `agnostic` row is the one the word does not fit. A hundred-basis-point prior standard deviation on a daily mean implies a median absolute Sharpe of $10.7282$, a probability of $0.9502$ that the magnitude exceeds one, $0.8498$ that it exceeds three and $0.5302$ that it exceeds ten. Its central ninety-eight per cent of annual outcomes runs from $-589.13\%$ to $+583.96\%$. No such strategy has ever existed, and an analyst asked directly whether they believed a coin-flip chance of a Sharpe above ten would have said no. They asserted it anyway, because the assertion was made in units where it was unreadable, and the posterior it produced — the lesson's `+5.8%` with `P(edge > 0) 0.93` — carries no trace of it.

**A prior is not a statement about a parameter; it is a statement about every observable the parameter determines, and the only reliable way to read one is to generate from it.** The lesson's instinct to publish the sensitivity table is exactly right, and this section adds one line to that practice: publish what each row implies about a quantity the reader has intuitions about. The `optimistic` prior, which sounds like the tendentious one, turns out to be modest — a median absolute Sharpe of $0.8463$ and a $0.0132$ chance of exceeding three — and is by that measure the most defensible of the four.

## A Hierarchical Prior Lets the Family Set the Width, Which Is the Only Construction Here That Improves as the Problem Gets Larger

Every prior so far was asserted. When a desk is estimating not one edge but fifty related ones, the width of the prior is a quantity the data speaks to directly, and the resulting construction is the one place on this page where the analyst's judgment is genuinely displaced by measurement.

??? note "Proof that a hierarchical prior's posterior mean is a precision-weighted average whose weight is identified by the spread of the family, so the prior's width is estimated rather than asserted"

    Let variant $j$ have true edge $\mu_j$ and an observed mean $y_j$ with $y_j\mid\mu_j\sim N(\mu_j,v)$, where $v=\sigma^{2}/n$ is the known sampling variance of a mean over $n$ days. Place the hierarchical prior $\mu_j\sim N(m,\tau^{2})$, independently across $j$. Conditioning on $\tau$ and $m$, the normal–normal algebra of [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md) gives
    $$\mathbb{E}[\mu_j\mid y_j]=\lambda m+(1-\lambda)y_j,\qquad \lambda=\frac{v}{v+\tau^{2}},$$
    so each estimate is pulled towards the family centre by a weight fixed entirely by the ratio of within-variant noise to between-variant spread.

    What makes this different from asserting $\tau$ is that $\tau$ is identified by the marginal law. Integrating out $\mu_j$ gives $y_j\sim N(m,v+\tau^{2})$ unconditionally, so the sample variance of the observed variant means estimates $v+\tau^{2}$:
    $$\mathbb{E}\Big[\frac{1}{J-1}\sum_j (y_j-\bar y)^{2}\Big]=v+\tau^{2}\quad\Longrightarrow\quad \hat\tau^{2}=\max\Big(0,\ \tfrac{1}{J-1}\textstyle\sum_j(y_j-\bar y)^{2}-v\Big).$$
    The truncation at zero is not cosmetic. The moment estimator is unbiased for $\tau^{2}$ on the whole line but $\tau^{2}$ cannot be negative, so the truncation introduces an upward bias exactly when the truth sits at the boundary — when every variant genuinely has the same edge. **The hierarchical construction converts the prior's width from an assertion into an estimate, and the price is that the estimate is worst precisely in the case where the strongest pooling would have been correct.**

    The load-bearing distinction is between this and the shrinkage of [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md), which reaches the same arithmetic from James–Stein's frequentist argument without any prior at all. The formulas coincide; the accounts of where $\lambda$ came from do not, and only the hierarchical account gives $\lambda$ a standard error.

The cost and the benefit are both measurable, and they trade against the one quantity nobody knows in advance:

```python
import numpy as np

rng = np.random.default_rng(16025)
J, n, sig, reps = 50, 500, 0.010, 4_000
v = sig ** 2 / n                                       # sampling variance of one variant's mean

print(f"  {J} strategy variants, {n} days each at {sig * 1e4:.0f}bp daily volatility,"
      f" {reps:,} families; the prior's width is estimated from the family itself")
print("     true tau, bp   true lambda   est lambda    no pooling   complete   partial"
      "   partial vs best")
for tau_bp in (0.0, 0.5, 1.0, 2.0, 4.0):
    tau = tau_bp * 1e-4
    mu = rng.normal(0.0004, tau, (reps, J))            # the family's true edges
    yb = mu + rng.standard_normal((reps, J)) * np.sqrt(v)
    gm = yb.mean(1, keepdims=True)
    between = ((yb - gm) ** 2).sum(1, keepdims=True) / (J - 1)
    tau2 = np.maximum(between - v, 0.0)                # method-of-moments prior variance
    lam = v / (v + tau2)
    est = (gm + (1 - lam) * (yb - gm), yb, np.broadcast_to(gm, yb.shape))
    mse = [(((e - mu) * 1e4) ** 2).mean() for e in est]
    print(f"    {tau_bp:12.1f}   {v / (v + tau ** 2):11.4f}   {lam.mean():10.4f}"
          f"   {mse[1]:11.4f}   {mse[2]:8.4f}   {mse[0]:7.4f}"
          f"   {mse[0] / min(mse[1], mse[2]):15.4f}")
# =>   50 strategy variants, 500 days each at 100bp daily volatility, 4,000 families; the prior's width is estimated from the family itself
#         true tau, bp   true lambda   est lambda    no pooling   complete   partial   partial vs best
#                 0.0        1.0000       0.9339       20.1562     0.4028    0.7468            1.8540
#                 0.5        0.9877       0.9301       20.0055     0.6373    0.9611            1.5081
#                 1.0        0.9524       0.9135       20.0560     1.3667    1.6561            1.2118
#                 2.0        0.8333       0.8386       20.0038     4.3289    4.1109            0.9496
#                 4.0        0.5556       0.5768       20.0386    16.1430    9.5828            0.5936
```

The `no pooling` column is flat at about $20.05$ squared basis points across every row, which is the sanity check: taking each variant's own mean ignores the family entirely, so its error is the sampling variance $\sigma^{2}/n$ and cannot depend on how heterogeneous the truth is. Everything interesting happens in the other two.

When the variants are genuinely identical, complete pooling is unbeatable — $0.4028$ against no pooling's $20.1562$, a fiftyfold reduction from the obvious act of averaging fifty estimates of one number. Partial pooling gets $0.7468$, which is $1.8540$ times worse, and the reason is in the proof: the estimated weight comes out at $0.9339$ rather than the correct $1.0000$, because a variance estimator truncated at zero cannot quite reach the boundary. That penalty is the honest cost of not knowing $\tau$ in advance.

The ordering reverses as the family spreads out. At a true $\tau$ of two basis points partial pooling is already ahead at $4.1109$ against complete pooling's $4.3289$, and at four basis points it wins decisively — $9.5828$ against $16.1430$ and $20.0386$, a ratio of $0.5936$ against the better of the two fixed rules. Across the sweep the estimated weight tracks the true one closely once it is away from the boundary, $0.8386$ against $0.8333$ and $0.5768$ against $0.5556$.

**This is what the course lesson measured without naming it.** [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) sweeps a shrinkage weight over fifty momentum variants and finds "error falls monotonically all the way to $\lambda = 1$, meaning the variants' individual histories contained *no* usable information beyond the family average," printing `lambda 0.0: OOS MSE 6.51`, `0.5: 4.40`, `1.0: 3.42`. A monotone descent to complete pooling is the top row of this table: the family's estimated $\tau$ was indistinguishable from zero, so the correct prior was the degenerate one, and the fifty variants were fifty measurements of a single number. The lesson's result is not evidence that shrinkage is always maximal; it is evidence about that particular grid, and the table above is what the same measurement returns when the grid is genuinely diverse.

## Improper Priors Are Legitimate Where the Posterior Is Proper, and the Check Is Not Optional

Jeffreys' prior for a scale, $\pi(\sigma)\propto1/\sigma$, integrates to infinity. So does a flat prior on the whole real line. These are improper — they are not probability distributions at all — and they are used constantly, because the posterior they produce is often perfectly well defined. The rule is exact: an improper prior is admissible as a formal device whenever $\int\pi(\theta)f(x\mid\theta)\,\mathrm{d}\theta<\infty$, and the resulting normalized function is a genuine posterior. For the volatility problem of section 2 that integral converges for every $n\ge1$ under Jeffreys, which is why the third column of that table exists at all.

The failure mode is that propriety is a property of the prior *and* the likelihood together, so it must be checked per model rather than per prior. Hierarchical models are where this bites: a flat prior on the group-level scale $\tau$ produces an improper posterior in the standard normal hierarchy, and the impropriety is invisible to any sampler, which will happily return draws that look like a converged answer. [Maximum A Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md) makes the related point that the flat-prior limit collapses the posterior mode onto the maximum likelihood estimate, and that there is no such thing as no prior.

Reference priors, developed by Bernardo and Berger, replace Jeffreys' invariance argument with an explicit information-theoretic one: choose the prior maximizing the expected Kullback–Leibler divergence between prior and posterior, so the data is asked to contribute as much as possible. In one dimension the answer coincides with Jeffreys'. In several it does not, and the reference construction's ordering of parameters into interest and nuisance groups is what repairs the multivariate behaviour Jeffreys' rule handles badly. Both are attempts to automate a judgment, and the honest summary is that automation buys a defensible default rather than an absence of choice.

!!! note "Informative, weakly informative, reference, improper and hierarchical priors are five things called the prior, and only one of them has a width the data determines"
    The vocabulary conceals a real ranking. An **informative** prior encodes a specific external belief — the lesson's `optimistic N(4bp, 2bp)` — and is defensible exactly to the extent its source can be named. A **weakly informative** prior is deliberately wider than any belief anyone holds but still tight enough to exclude the absurd, which section 3 shows is a much narrower band than practitioners assume: three basis points behaves almost identically to a hundred on every readable quantity while excluding Sharpe ratios above ten. A **reference** prior, of which Jeffreys' is the one-dimensional case, is a rule chosen for an invariance or information property rather than for expressing a belief, and section 1 shows what it buys is checkable and what it does not buy is neutrality. An **improper** prior is not a distribution and is a formal device whose legitimacy depends on the likelihood it meets. A **hierarchical** prior is the only one on this list whose width is an estimated parameter with a sampling distribution, which is why section 4's construction adapts and the other four cannot. Reporting any of the first four as though it were the last — describing a fixed wide prior as "letting the data decide" — is the error the list exists to prevent.

!!! warning "A prior chosen to be harmless is inspected in the units it was written in and never in the units it constrains, so the most extravagant assertion on the page is usually the one nobody argued about"
    Every number in section 3 was available before any data arrived and none of them was computed. The prior a careful lesson calls `agnostic` asserts a median absolute Sharpe of $10.7282$, a $0.8498$ probability of exceeding three in magnitude and annual outcomes spanning $-589.13\%$ to $+583.96\%$, and it produced a posterior — `+5.8%`, `P(edge > 0) 0.93` — that looks like the neutral row of a sensitivity table. The same failure runs through section 2, where declining to assume anything about volatility in three different charts produced medians of $1.1575$, $1.3787$ and $1.0166$ times the truth and tail probabilities of $0.2921$, $0.4257$ and $0.1952$ for a risk-limit breach. **The free diagnostic is to draw a few thousand parameter values from your prior, push each through the model for one realistic horizon, and print the median and the ninety-ninth percentile of a quantity you have professional intuitions about — a Sharpe ratio, an annual return, a maximum drawdown — before looking at any data, and to tighten the prior whenever those numbers describe a strategy you have never seen.** It costs four lines and it is the only check on this page that requires no theory at all.

## A Belief You Declined to Write Down Is Usually an Extravagant One

This page established that a flat density survives only affine reparameterization, so uniformity commits to a coordinate system rather than expressing ignorance; that Jeffreys' $\sqrt{\det I(\theta)}$ transforms exactly as a density and therefore selects one measure in every chart, giving $\pi(\sigma)\propto1/\sigma$ for a scale; that three analysts declining to assume anything in three coordinates reported posterior median volatilities of $1.1575$, $1.3787$ and $1.0166$ times the truth at five observations, with risk-limit probabilities of $0.2921$, $0.4257$ and $0.1952$, the gap against Jeffreys falling $+35.62\%$, $+12.78\%$, $+3.59\%$ and $+1.02\%$ as $n$ ran $5$ to $100$; that a prior is a statement about every observable the parameter implies, so the `agnostic` prior of a course lesson asserts a median absolute Sharpe of $10.7282$, probabilities of $0.9502$, $0.8498$ and $0.5302$ of exceeding one, three and ten, and annual outcomes from $-589.13\%$ to $+583.96\%$, while a three-basis-point prior differs from a hundred-basis-point one by $0.7475$ against $10.7282$ on that same reading; and that a hierarchical prior estimates its own width, costing $1.8540$ times the best fixed rule when the family is genuinely homogeneous and returning a weight of $0.9339$ instead of $1.0000$, then paying that back at $0.5936$ once the family spreads to four basis points.

The shape shared by all three exhibits is that the damage is done by the quantity nobody printed. The coordinate convention, the implied Sharpe distribution and the assumed between-variant spread are all inputs, all consequential, and none of them appears anywhere in a posterior. That is the failure [The Bayesian Framework](01-bayesian-framework.md) named at the level of the framework, arriving here with numbers attached, and it is why the free diagnostic above is a prior predictive check rather than an argument.

What has not yet been examined is the object all of this feeds. Sections 2 and 4 computed posteriors freely, using conjugate forms and a normalizing integral that happened to be available, and said nothing about what to do when it is not — or about what a posterior's concentration actually establishes once the model producing it is wrong. That is [Posterior Distributions](03-posterior-distributions.md).

**Every prior is informative about something, the only question is whether it is informative about a quantity anyone checked, and the priors chosen to avoid commitment are reliably the ones committing to the most.**
