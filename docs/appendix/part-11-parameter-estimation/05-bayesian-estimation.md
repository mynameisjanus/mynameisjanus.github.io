# Bayesian Estimation

The two estimators before this one converge on a number and report how tightly they converged. A posterior does something categorically different: it is a distribution over the parameter, and a distribution is not an estimate. Something has to collapse it to one number before a position can be sized, and the thing that does the collapsing is a loss function — the same object [Point Estimation](01-point-estimation.md) showed was the only way to rank two rules, now doing the same job one level down. That is the whole content of Bayesian *estimation* as distinct from Bayesian inference: the posterior is the inference, and the estimate is a decision made about it. What falls out is that every Bayes estimator is a shrinkage rule whose weight is computable before any data arrives, that its frequentist risk beats the unbiased estimator's over a wide range and loses badly outside it, and that the shrinkage constant nobody could compute two parts ago is exactly what a prior variance is a guess at.

This page covers the posterior as an object that requires a summary, the loss functions that select the mean, the median or a quantile, the precision-weighted form every Bayes estimator takes and its identity with the optimal shrinkage the earlier theorem declined to identify, the frequentist risk function of a Bayes rule, and James–Stein as the estimator that shrinks correctly without a prior at all. It constructs no prior and argues for none, which is [Prior Distributions](../part-16-bayesian-statistics/02-prior-distributions.md); it derives no conjugate family and proves no closure, which are [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) and [Conjugate Priors](../part-16-bayesian-statistics/04-conjugate-priors.md); it updates nothing sequentially, which is [Bayesian Updating](../part-16-bayesian-statistics/05-bayesian-updating.md); it computes no marginal likelihood and compares no models, which is [Bayesian Model Comparison](../part-16-bayesian-statistics/06-bayesian-model-comparison.md); it forecasts nothing, which is [Bayesian Prediction](../part-16-bayesian-statistics/07-bayesian-prediction.md); it takes no posterior mode, which is [Maximum A Posteriori Estimation](06-maximum-a-posteriori-estimation.md); it samples no posterior it cannot integrate, which is [Markov Chain Monte Carlo](../part-17-statistical-computing/04-markov-chain-monte-carlo.md); it quotes no interval, which is [Confidence Intervals](07-confidence-intervals.md); and it never claims a posterior is more honest than a likelihood merely for being a distribution.

The trading stake is the strongest shrinkage verdict in the course and the section that qualifies it. [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) estimates fifty momentum variants on one period, scores them on the next, and finds that "error falls monotonically all the way to $\lambda = 1$, meaning the variants' individual histories contained *no* usable information beyond the family average," with the sweep printing `lambda 0.0: OOS MSE 6.51`, `0.5: 4.40`, `1.0: 3.42` and the in-sample champion turning a $+10.5\%$ estimate into a realized $-1.8\%$. The fifth section prices that verdict and locates exactly where it stops being right.

## A Posterior Is a Distribution and an Estimate Is a Summary, So a Loss Function Has to Choose Which One

Bayes' rule turns a prior density $\pi(\theta)$ and a likelihood $f(x\mid\theta)$ into the **posterior**

$$\pi(\theta\mid x)=\frac{\pi(\theta)f(x\mid\theta)}{\int\pi(\theta')f(x\mid\theta')\,d\theta'},$$

and the denominator is the object [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md) identifies as "the constant that turns a likelihood back into a probability distribution over hypotheses." That constant is what maximum likelihood discards and what makes the difference between an object you can maximize and an object you can integrate.

Having an integrable object changes the question. A likelihood has a peak and nothing else, so $\arg\max$ is the only available summary; a posterior has a mean, a median, a mode, a variance and every quantile, and there is no mathematical reason to prefer one. The choice is a decision problem, and the machinery is the one already built: pick a loss $L(\theta,a)$, form the **posterior expected loss** $\mathbb{E}[L(\theta,a)\mid x]$, and report the $a$ that minimizes it. The result is the **Bayes estimator** for that loss.

Two things are worth stating before the algebra. First, this construction collapses the partial order [Point Estimation](01-point-estimation.md) proved could not be totalized: averaging the frequentist risk $R(\theta,\hat\theta)$ against $\pi$ produces a single number, the **Bayes risk**, and a scalar can always be minimized. The reason "best estimator" now has a referent is that a weighting over $\theta$ was supplied, and supplying it is the entire cost. Second, the loss and the prior enter at different points and do different jobs — the prior decides what the posterior is, the loss decides which number comes out of it — so two desks with identical beliefs and different cost structures should report different estimates from the same data, and usually do not.

## The Posterior Mean Minimizes Quadratic Risk, the Median Minimizes Absolute Risk, and an Asymmetric Loss Picks a Quantile

The three standard losses give three standard summaries, and the derivation is short enough that the pattern is visible.

??? note "Proof that the posterior mean minimizes expected squared error, the posterior median expected absolute error, and an asymmetric linear loss the $c_u/(c_u+c_o)$ quantile"
    For squared error, expand about the posterior mean $\bar\theta=\mathbb{E}[\theta\mid x]$:

    $$\mathbb{E}\big[(\theta-a)^{2}\mid x\big]=\mathrm{var}(\theta\mid x)+(\bar\theta-a)^{2},$$

    which is minimized at $a=\bar\theta$ with minimum equal to the posterior variance. The estimator and its own uncertainty come out of the same identity.

    For absolute error the objective is not differentiable at $a=\theta$, so differentiate the integral instead:

    $$\frac{d}{da}\mathbb{E}\big[|\theta-a|\mid x\big]=\mathbf{P}(\theta<a\mid x)-\mathbf{P}(\theta>a\mid x),$$

    which is zero exactly when both probabilities equal $\tfrac12$ — the posterior median.

    Generalize to an asymmetric linear loss charging $c_u$ per unit of underestimate and $c_o$ per unit of overestimate, $L=c_u(\theta-a)^{+}+c_o(a-\theta)^{+}$. The same differentiation gives

    $$\frac{d}{da}\mathbb{E}[L\mid x]=c_o\,\mathbf{P}(\theta<a\mid x)-c_u\,\mathbf{P}(\theta>a\mid x)=0\ \Longleftrightarrow\ \mathbf{P}(\theta\le a\mid x)=\frac{c_u}{c_u+c_o},$$

    so the optimal report is the $c_u/(c_u+c_o)$ posterior quantile. Squared error is the special case that happens to return a moment; every other loss in the family returns a quantile.

    The load-bearing step is the derivative of the posterior expected loss in $a$ — linear with one root under $L^{2}$, a difference of probabilities with a level set under $L^{1}$, and the same difference reweighted under asymmetry. **A posterior does not contain an estimate; a loss function extracts one, and reporting a posterior mean is an assertion that your costs are symmetric and quadratic, which no trading book's are.**

The asymmetry is not a refinement. Underestimating a volatility sizes a position too large in the regime the estimate was wrong about, and overestimating it costs foregone return — two different sums of money, and the ratio between them is a fact about the book rather than about the data.

```python
import numpy as np

rng = np.random.default_rng(11051)
ann, draws, sets = np.sqrt(252), 40_000, 200
sd = 0.195 / ann                                               # a 19.5% asset, the course's SPY
grid = ((1, 1), (2, 1), (5, 1), (10, 1))

print("     n    under:over    optimal quantile    vol at optimum    posterior mean vol"
      "    excess loss from using the mean")
for n in (21, 63, 252):
    a = np.zeros((len(grid), sets))
    mu = np.zeros(sets)
    ex = np.zeros((len(grid), sets))
    for j in range(sets):
        x = sd * rng.standard_normal(n)
        v = 100 * ann * np.sqrt((x ** 2).sum() / rng.chisquare(n, draws))
        mu[j] = v.mean()
        for i, (cu, co) in enumerate(grid):
            a[i, j] = np.quantile(v, cu / (cu + co))
            f = lambda z: (cu * np.maximum(v - z, 0) + co * np.maximum(z - v, 0)).mean()
            ex[i, j] = f(mu[j]) / f(a[i, j]) - 1
    for i, (cu, co) in enumerate(grid):
        print(f"  {n:4d} {f'{cu}:{co}':>13} {cu / (cu + co):19.3f} {a[i].mean():17.3f}"
              f" {mu.mean():21.3f} {ex[i].mean():34.4f}")
# =>      n    under:over    optimal quantile    vol at optimum    posterior mean vol    excess loss from using the mean
#        21           1:1               0.500            19.325                19.733                             0.0087
#        21           2:1               0.667            20.705                19.733                             0.0449
#        21           5:1               0.833            22.662                19.733                             0.4107
#        21          10:1               0.909            24.184                19.733                             1.0413
#        63           1:1               0.500            19.583                19.715                             0.0028
#        63           2:1               0.667            20.363                19.715                             0.0652
#        63           5:1               0.833            21.407                19.715                             0.4871
#        63          10:1               0.909            22.176                19.715                             1.2054
#       252           1:1               0.500            19.596                19.628                             0.0007
#       252           2:1               0.667            19.978                19.628                             0.0804
#       252           5:1               0.833            20.472                19.628                             0.5406
#       252          10:1               0.909            20.823                19.628                             1.3196
```

The quantile column is the theorem and the volatility column is its consequence in reportable units. On twenty-one days of a $19.5\%$ asset the symmetric answer is $19.325\%$, the two-to-one answer is $20.705\%$, the five-to-one answer is $22.662\%$ and the ten-to-one answer is $24.184\%$. Nothing about the data changed between those rows; the only input that moved was the ratio of what an understatement costs to what an overstatement costs, and the reported volatility moved by five points.

The last column prices the habit of reporting the mean anyway. Under symmetric loss the mean is almost exactly right, costing $0.87\%$, $0.28\%$ and $0.07\%$ of avoidable expected loss at the three sample sizes — which is why the habit survives, since the case everybody tests is the case where it works. At five-to-one it costs between $41\%$ and $54\%$ more than the optimum, and at ten-to-one it costs $104\%$ to $132\%$ more: **the wrong summary of the right posterior more than doubles the expected loss.**

The direction of the last column is the part worth pausing on. The penalty *rises* with sample size, from $1.0413$ at $n=21$ to $1.3196$ at $n=252$, because the posterior tightens around the truth faster than the achievable loss falls, so the fixed relative error of using the wrong summary occupies a larger share of a smaller total. **Choosing the wrong summary is a bias in the sense of [Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md) — it survives more data, and no amount of history repairs it.**

## Every Bayes Estimator Is a Shrinkage Rule and the Prior Variance Is the Number the Shrinkage Theorem Refused to Supply

The mean, the median and the quantiles all move toward the prior as the data thins, and in the conjugate normal case the movement has an exact algebraic form that connects this page to a theorem two parts back.

??? note "Proof that the normal–normal posterior mean is a precision-weighted average, so a prior variance is a numerical guess at the squared distance the shrinkage theorem could not compute"
    Take $X_1,\dots,X_n$ independent $\mathcal{N}(\theta,\sigma^{2})$ with $\sigma^{2}$ known, and a prior $\theta\sim\mathcal{N}(\mu_0,\tau^{2})$. Multiplying the two exponentials and completing the square in $\theta$ gives a normal posterior with

    $$\mathbb{E}[\theta\mid x]=\frac{\tau^{2}}{\tau^{2}+\sigma^{2}/n}\,\bar x+\frac{\sigma^{2}/n}{\tau^{2}+\sigma^{2}/n}\,\mu_0,\qquad \mathrm{var}(\theta\mid x)=\Big(\frac{1}{\tau^{2}}+\frac{n}{\sigma^{2}}\Big)^{-1}.$$

    Precisions add and the mean is their weighted average. Writing $v=\sigma^{2}/n$ for the sampling variance, the shrinkage toward the prior centre is $\lambda=v/(v+\tau^{2})$.

    Now compare with the shrinkage theorem of [Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md), which proved that shrinking an unbiased estimator toward a constant $c$ has mean squared error $(1-\lambda)^{2}v+\lambda^{2}(\theta-c)^{2}$, minimized at

    $$\lambda^{\star}=\frac{v}{v+(\theta-c)^{2}},$$

    and then declined to identify $\lambda^{\star}$ because $(\theta-c)^{2}$ involves the unknown. The two expressions are the same expression with $\tau^{2}$ standing in for $(\theta-c)^{2}$.

    The load-bearing quantity is that substitution. A prior variance is a numerical answer to the question the shrinkage theorem refused: how far, in squared units, do you expect the truth to sit from the point you are shrinking toward. **Choosing a prior is not adding information to the data; it is being made to state the one number that the risk calculation needed and could not obtain, and the honesty of the estimate is exactly the honesty of that number.**

The identification is worth holding onto because it removes the mystique from both sides. Anyone who has ever chosen a shrinkage intensity, a regularization strength, an exponential decay factor or a blending weight between a model and a benchmark has chosen a prior variance, whether or not the word appeared. The Bayesian version's only distinguishing feature is that it forces the number into the open where it can be argued about.

## A Bayes Estimator Has a Frequentist Risk Function Too, and It Is Better Almost Everywhere and Worse Somewhere

A Bayes estimator is a function of the data like any other, so it has a frequentist risk function $R(\theta,\hat\theta_\pi)$ evaluated at each fixed $\theta$ — a quantity the Bayesian framework has no use for and the practitioner very much does, since the truth is one number rather than a distribution. Reading that risk function is the only way to see what a prior costs when it is wrong.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(11057)
n, reps = 6_158, 200_000
s2 = (0.193 / np.sqrt(252)) ** 2                               # the course's momentum strategy
priors = (("skeptical  N(0, 1bp)", 0.0, 1e-4),
          ("agnostic   N(0, 100bp)", 0.0, 1e-2),
          ("optimistic N(4bp, 2bp)", 4e-4, 2e-4))

def post(xbar, mu0, tau):
    t2 = 1 / (1 / tau ** 2 + n / s2)
    return t2 * (mu0 / tau ** 2 + n * xbar / s2), t2

for lab, mu0, tau in priors:                                   # replay the lesson's own numbers
    m, t2 = post(0.058 / 252, mu0, tau)
    print(f"  {lab}:  weight {tau ** 2 / (tau ** 2 + s2 / n):.3f}, post ann mean {252 * m:+.1%},"
          f"  P(edge > 0) {1 - norm.cdf(0, m, np.sqrt(t2)):.2f}")
print("  true ann mu    risk(no prior)    risk(skeptical)    risk(agnostic)    risk(optimistic)"
      "    winner")
for a in (0.0, 0.02, 0.039, 0.058, 0.15, 0.30):
    xb = rng.normal(a / 252, np.sqrt(s2 / n), reps)
    r = [252 ** 2 * ((xb - a / 252) ** 2).mean()]
    r += [252 ** 2 * ((post(xb, m0, t)[0] - a / 252) ** 2).mean() for _, m0, t in priors]
    who = ("no prior", "skeptical", "agnostic", "optimistic")[int(np.argmin(r))]
    print(f"  {a:11.3f} {r[0]:17.6f} {r[1]:18.6f} {r[2]:17.6f} {r[3]:19.6f}    {who}")
# =>   skeptical  N(0, 1bp):  weight 0.294, post ann mean +1.7%,  P(edge > 0) 0.79
#      agnostic   N(0, 100bp):  weight 1.000, post ann mean +5.8%,  P(edge > 0) 0.93
#      optimistic N(4bp, 2bp):  weight 0.625, post ann mean +7.4%,  P(edge > 0) 0.99
#      true ann mu    risk(no prior)    risk(skeptical)    risk(agnostic)    risk(optimistic)    winner
#            0.000          0.001531           0.000132          0.001530            0.002027    skeptical
#            0.020          0.001522           0.000330          0.001521            0.001515    skeptical
#            0.039          0.001528           0.000893          0.001527            0.001129    skeptical
#            0.058          0.001522           0.001805          0.001521            0.000854    optimistic
#            0.150          0.001520           0.011344          0.001519            0.000934    optimistic
#            0.300          0.001524           0.044968          0.001523            0.006167    agnostic
```

The three header lines reproduce the lesson's published table exactly — $+1.7\%$ with $\mathbf{P}=0.79$, $+5.8\%$ with $0.93$, $+7.4\%$ with $0.99$ — and add the column the lesson did not print. The shrinkage weights are $0.294$, $1.000$ and $0.625$, and every one of them is computable from $n$, $\sigma^{2}$ and $\tau$ before a single return is loaded. **The spread of conclusions the lesson calls a sensitivity analysis is fixed in advance by three numbers a practitioner types, and the data moves it not at all.**

The risk sweep says what those weights cost. The skeptical prior's risk is $0.000132$ at $\mu=0$ against the unshrunk $0.001531$ — better by a factor of $11.6$ — and $0.044968$ at $\mu=0.30$ against $0.001524$, worse by a factor of $29.5$. That shape is the whole of Bayesian point estimation in one row pair: a prior buys a large improvement in the region it believes in and pays an unbounded penalty outside it, and the size of both is set by $\tau$.

The remaining columns close the argument in two directions. The agnostic prior is indistinguishable from no prior at every $\mu$ — $0.001530$ against $0.001531$, $0.001519$ against $0.001520$ — because its weight is $1.000$ to three decimals, so a prior wide enough to be uncontroversial is a prior that does nothing, and calling an analysis Bayesian on its strength is a labelling exercise. And no rule wins everywhere: the skeptical prior takes the first three rows, the optimistic one takes the middle, the near-flat one takes the last. **Three defensible priors produce three different estimators, each of them the best available over a range of truths, and the data cannot say which range it is in** — which is the crossing theorem of [Point Estimation](01-point-estimation.md) reappearing with the priors in the role of the rules.

!!! warning "A posterior mean reported without the prior that produced it is a point estimate with an undeclared shrinkage target, and the target is where the number goes when the data is thin"
    The failure is not that priors are subjective. It is that the shrinkage weight is a deterministic function of $\tau^{2}$, $\sigma^{2}$ and $n$ that nobody recomputes when $n$ changes, so a prior calibrated as mild on twenty years of daily data becomes dominant on a new strategy with six months of history — the same $\tau=1$ basis point that pulls $70.6\%$ of the way to zero above pulls $98.7\%$ of the way on two hundred observations, and the reported edge is then a restatement of the prior with the data as decoration. The free diagnostic is to publish the weight rather than the estimate: **compute $\lambda=(\sigma^{2}/n)/(\tau^{2}+\sigma^{2}/n)$ and print it beside every posterior mean, then recompute the analysis under a prior centred at zero and one centred at the family average and publish the spread.** The lesson's own three priors move the probability of an edge from $0.79$ to $0.99$ on identical data, and that spread is the finding — a research note reporting only the middle one has reported a choice as a result.

## James–Stein Shrinks Three or More Parameters With No Prior at All and Knows When to Stop

The previous section leaves an awkward position: shrinkage helps a great deal when $\tau$ is right and hurts a great deal when it is not, and $\tau$ is unknown. For a single parameter there is no escape. For three or more estimated together there is, and it is one of the genuinely surprising results in statistics.

??? note "Proof that the sample mean is inadmissible in three or more dimensions, and that the James–Stein weight estimates the shrinkage the previous section had to assume"
    Estimate $K$ means $\theta_1,\dots,\theta_K$ from independent $y_i\sim\mathcal{N}(\theta_i,\sigma^{2})$ under total squared-error loss. Suppose the $\theta_i$ were themselves drawn from $\mathcal{N}(\bar\theta,\tau^{2})$. Then marginally $y_i\sim\mathcal{N}(\bar\theta,\tau^{2}+\sigma^{2})$, so

    $$\mathbb{E}\Big[\frac{(K-3)\sigma^{2}}{\sum_j(y_j-\bar y)^{2}}\Big]=\frac{\sigma^{2}}{\tau^{2}+\sigma^{2}},$$

    using the mean of an inverse chi-squared on $K-1$ degrees of freedom. The right-hand side is precisely the Bayes shrinkage weight $\lambda$ of the previous section, so the observable quantity on the left is an unbiased estimate of the unobservable weight. Substituting it gives the **positive-part James–Stein estimator**

    $$\hat\theta_i^{\mathrm{JS}}=\bar y+\Big(1-\frac{(K-3)\sigma^{2}}{\sum_j(y_j-\bar y)^{2}}\Big)^{+}(y_i-\bar y).$$

    Stein's theorem is that for $K\ge3$ this dominates the raw $y$ — strictly lower total risk at *every* $\theta$, with no prior assumed and no distributional assumption on the $\theta_i$ needed for the domination itself. The sample mean is therefore inadmissible in three or more dimensions, and by the admissibility of proper-prior Bayes rules the estimator that dominates it is essentially always a Bayes or empirical-Bayes rule.

    The load-bearing quantity is the cross-sectional spread $\sum_j(y_j-\bar y)^{2}$, which is the only thing in the formula the analyst does not choose. It is large when the units genuinely differ, and the rule then shrinks little; it is small when they do not, and the rule then shrinks almost everything. **James–Stein is the previous section's prior with $\tau^{2}$ read off the data instead of asserted, which is why it needs three parameters — two are not enough to measure a spread.**

The course's shrinkage experiment is exactly this problem with $K=50$, and its verdict was that full shrinkage minimized error. The question the verdict leaves open is whether that is a fact about shrinkage or a fact about that particular family.

```python
import numpy as np

rng = np.random.default_rng(11053)
K, se, reps = 50, 0.20, 4_000                                  # 50 variants, 25 years of Sharpe

print(f"  {K} variants, Sharpe standard error {se}, truth centred at 0.20")
print("     tau    MSE(no shrink)    MSE(James-Stein)    MSE(full shrink)    winner"
      "    in-sample best: estimated    realized")
for tau in (0.00, 0.02, 0.05, 0.10, 0.20):
    th = 0.20 + tau * rng.standard_normal((reps, K))
    y = th + se * rng.standard_normal((reps, K))
    yb = y.mean(axis=1, keepdims=True)
    s = ((y - yb) ** 2).sum(axis=1, keepdims=True)
    js = yb + np.maximum(0.0, 1 - (K - 3) * se ** 2 / s) * (y - yb)
    m = [((e - th) ** 2).mean() for e in (y, js, np.broadcast_to(yb, y.shape))]
    i = y.argmax(axis=1)
    print(f"  {tau:6.2f} {m[0]:17.5f} {m[1]:19.5f} {m[2]:19.5f}"
          f" {('no shrink', 'James-Stein', 'full shrink')[int(np.argmin(m))]:>12}"
          f" {y[np.arange(reps), i].mean():27.4f}"
          f" {th[np.arange(reps), i].mean():11.4f}")
# =>   50 variants, Sharpe standard error 0.2, truth centred at 0.20
#         tau    MSE(no shrink)    MSE(James-Stein)    MSE(full shrink)    winner    in-sample best: estimated    realized
#        0.00           0.04009             0.00172             0.00080  full shrink                      0.6513      0.2000
#        0.02           0.03995             0.00203             0.00117  full shrink                      0.6506      0.2047
#        0.05           0.03998             0.00402             0.00325  full shrink                      0.6647      0.2281
#        0.10           0.04014             0.00970             0.01064  James-Stein                      0.7033      0.3007
#        0.20           0.04012             0.02128             0.04004  James-Stein                      0.8389      0.5224
```

The unshrunk column is flat at $0.0401$ regardless of $\tau$, as it must be — the raw estimate's error is the sampling error and knows nothing about the cross-section. Every other column is a fraction of it. At zero true dispersion, shrinking all the way scores $0.00080$ and James–Stein scores $0.00172$, both roughly fifty times better than doing nothing, so the course's verdict is not merely correct at its own family but correct by two orders of magnitude.

The crossover is at $\tau=0.10$, which is half a standard error. Below it full shrinkage wins — $0.00325$ against James–Stein's $0.00402$ at $\tau=0.05$ — and at $\tau=0.10$ the ordering reverses, $0.00970$ against $0.01064$, with the gap widening to $0.02128$ against $0.04004$ at $\tau=0.20$. **The course's "shrink all the way" is right for families whose true dispersion is under about half a standard error and wrong above it, and James–Stein gets both regimes right without being told which one it is in** — at a cost of being slightly worse than full shrinkage where full shrinkage is optimal, which is the price of not having been told.

The last two columns are the winner's curse with the same date stamp the course gave it. At zero dispersion the best-looking of fifty variants shows an estimated Sharpe of $0.6513$ and its true Sharpe is $0.2000$ — the family average, exactly, because there is nothing else for it to be. The gap of $0.45$ is the maximum of fifty standard normals scaled by the standard error, a quantity that depends on the size of the search and not at all on the data, which is why the lesson's champion could report $+10.5\%$ and realize $-1.8\%$. Even at $\tau=0.20$, where genuine differences exist, the winner is overstated by $0.3165$. Charging correctly for the size of that search is [Part XV](../part-15-multiple-testing/index.md); shrinking so that the search returns a less extreme winner in the first place is this page.

!!! note "Shrinkage, empirical Bayes and James–Stein produce the same arithmetic from three different accounts of where the weight came from, and only the account differs"
    All three compute $\hat\theta_i=\bar y+(1-\lambda)(y_i-\bar y)$ and differ in the provenance of $\lambda$. Shrinkage in the sense of [Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md) treats $\lambda$ as a tuning constant, proves a favourable trade exists at the margin, and cannot say what it is. Bayesian estimation obtains it from an asserted prior variance, $\lambda=v/(v+\tau^{2})$, and stands or falls on that assertion. Empirical Bayes, of which James–Stein is the classical instance, estimates $\tau^{2}$ from the cross-section and therefore needs the units to be genuinely exchangeable — fifty variants of one strategy are, fifty unrelated assets are not, and applying the rule across a heterogeneous set pulls a genuinely different unit toward a mean it has no business near. The distinction that matters operationally is not Bayesian against frequentist but *asserted against estimated*: an asserted weight is auditable and wrong in a stated direction, an estimated one is self-correcting and needs $K\ge3$ comparable units to estimate from. Ledoit–Wolf covariance shrinkage is the same idea with a matrix target, and [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) measures it choosing an intensity of $1.85\%$ — a correct estimate of a weight that was never the problem.

## The Prior Is an Input and the Loss Is a Choice, and Only One of Them Gets Written Down

This page established that a posterior is an inference and an estimate is a decision about it, so a loss function has to select the summary; that squared error returns the mean, absolute error the median and an asymmetric linear loss the $c_u/(c_u+c_o)$ quantile, with a ten-to-one cost ratio moving a reported volatility from $19.6\%$ to $20.8\%$ and the mean costing $132\%$ in excess expected loss at $n=252$ and *more* as the sample grows; that every conjugate Bayes estimator is a precision-weighted average whose weight is the shrinkage the earlier theorem proved existed and declined to compute; that a prior buys a factor of $11.6$ in risk where it is right and costs a factor of $29.5$ where it is not, while a prior wide enough to be uncontroversial is one that does nothing; and that James–Stein recovers the weight from the cross-section, matching full shrinkage below half a standard error of dispersion and beating it above.

Two inputs entered this page and they were treated very differently. The prior is argued about, published, subjected to sensitivity analysis, and used as the reason to distrust the whole apparatus. The loss function is chosen silently by whoever wrote the reporting code, is almost never stated, and — on the numbers above — moves the reported estimate further than the choice between a skeptical and an agnostic prior does. That asymmetry of attention is the practical failure this page documents, and it is the same failure [Point Estimation](01-point-estimation.md) identified from the frequentist side, arriving here by a different road.

What the page does not do is take the cheap route through the posterior. Every summary above required an integral — a mean, a median, a quantile, an expected loss — and integrals over a posterior are the expensive part of Bayesian work, which is why the machinery of [Markov Chain Monte Carlo](../part-17-statistical-computing/04-markov-chain-monte-carlo.md) exists. There is exactly one summary that needs no integral, and it is the one every practitioner actually computes when the posterior is not conjugate. It is also the only summary on these two pages that minimizes no expected loss at all, and it moves when you change coordinates. That is [Maximum A Posteriori Estimation](06-maximum-a-posteriori-estimation.md).

**A prior is the number the shrinkage theorem refused to supply, and writing one down is the only way to be wrong about it out loud.**
