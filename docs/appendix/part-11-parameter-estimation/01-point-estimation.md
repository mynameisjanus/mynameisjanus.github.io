# Point Estimation

A point estimate is a single number, and the single number is the least interesting thing about it. What makes estimation a subject rather than an arithmetic exercise is that the number was produced by a rule, that the rule has a distribution the number does not, and that the choice between two rules cannot be made by inspecting the two numbers they returned on the one dataset anybody has. This page builds the frame that choice is made in — an estimand, a rule, a loss, a risk — and the frame's first substantive result is a negative one. Risk does not order estimators. Two rules cross, each is better than the other somewhere, and the rule that ignores the data entirely cannot be beaten everywhere by anything. Every remaining page in this part is a different device for restricting the question until it has an answer, and knowing which restriction you accepted is the difference between choosing an estimator and inheriting one.

This page covers the separation of an estimand from an estimator from an estimate, the plug-in principle as the default construction together with the curvature charge it silently incurs, loss functions and the risk they induce, the failure of risk to be a total order and the admissibility of the rule that ignores the data, and the standard error as the one part of a reported estimate a reader cannot reconstruct. It splits no error into a squared bias and a variance, which is [Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md); it catalogues no property, so unbiasedness, consistency, efficiency and the information bound are [Properties of Estimators](02-properties-of-estimators.md); it maximizes no likelihood, which is [Maximum Likelihood Estimation](03-maximum-likelihood-estimation.md); it matches no moments, which is [Method of Moments](04-method-of-moments.md); it averages risk against no prior, which is [Bayesian Estimation](05-bayesian-estimation.md); it derives no sampling distribution from a family, which is [Sampling Distributions](../part-10-statistics-foundations/03-sampling-distributions.md); it turns no standard error into an interval, which is [Confidence Intervals](07-confidence-intervals.md); it resamples nothing, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it optimizes no portfolio, which is [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md); and it never tells you which loss function your desk is actually paid on.

The trading stake is a machine that is provably optimal and provably ruinous, with the two proofs about different arguments. [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) states it in one breath: "Mean-variance optimization is correct. Given the true $\mu$ and $\Sigma$ it produces the best possible portfolio, and no method in the table beats it. Fed two-year estimates it produced 11.81× gross exposure, 293.9% volatility, a −99.9% drawdown, and a Sharpe below a portfolio that requires no estimation whatsoever." The published Sharpe is $0.377$ against $1/N$'s $0.450$. The fifth section prices that exactly, by running the same optimizer against training windows from six months to twenty years and watching every input converge while the output does not.

## An Estimator Is a Rule and an Estimate Is a Number, and Only the Rule Has a Distribution

Three objects wear the same name in ordinary speech and have to be separated before anything on this page can be stated. The **estimand** is a functional $\theta=T(F)$ of the population law — a mean, a variance, a quantile, a tail index, a Sharpe ratio. It is a fixed number attached to the world rather than to the data, and [Population vs Sample](../part-10-statistics-foundations/01-population-vs-sample.md) is where the argument that such a law exists at all was made. The **estimator** is a function $\hat\theta=g(X_1,\dots,X_n)$ of the data and nothing else, which makes it a random variable with a law of its own, induced by the sampling. The **estimate** is the value that function took on the one sample you drew. It is a number. It has no distribution, no bias, no variance and no standard error, and every property this part discusses belongs to the rule that produced it rather than to it.

The distinction is pedantic right up to the moment somebody reports a Sharpe ratio of $0.30$ and is asked how confident they are. There is no answer to that question about the number $0.30$. There is an answer about the *rule* — draw a fresh twenty-five years, apply the same arithmetic, see how far the answer moves — and that answer exists only because the rule can be applied to samples that were never drawn. **Every uncertainty statement in statistics is a statement about a procedure evaluated on data nobody has**, which is why replication is the natural language of this part and why all three code blocks below simulate rather than compute.

A consequence follows immediately. Because the estimator is a function of the data, any question about it is a question about a distribution, and distributions are compared by their whole shape rather than by one number. Two rules can have identical means and wildly different tails. A rule can be right on average and never right. The sampling law of $\hat\theta$ is precisely the object [Sampling Distributions](../part-10-statistics-foundations/03-sampling-distributions.md) constructs, which is why that page is a prerequisite for this one rather than a companion to it.

## The Plug-In Principle Turns Any Population Functional Into an Estimator and Charges Curvature Times Variance

If the estimand is a functional of the law, there is an obvious rule: estimate the law, then apply the functional. Replace $F$ by the empirical distribution $\hat F_n$ that puts mass $1/n$ on each observation and read off $\hat\theta=T(\hat F_n)$. That single move manufactures the sample mean from the population mean, the sample quantile from the population quantile, the sample covariance matrix from the population one and — composed twice — the sample Sharpe ratio and the sample Kelly fraction. It is the default construction in this part and the one every other page is measured against.

The plug-in principle is free when $T$ is linear and is not free otherwise, and the charge has a closed form.

??? note "Proof that a plug-in estimator of a nonlinear functional is biased at order $1/n$ with a sign fixed by curvature, and that for an inverse variance the charge is exactly $(n-1)/(n-3)$"
    Let $\hat\theta$ be an unbiased estimator of $\theta$ with variance $v$, and let $g$ be smooth. Expanding about $\theta$,

    $$g(\hat\theta)=g(\theta)+g'(\theta)(\hat\theta-\theta)+\tfrac12 g''(\theta)(\hat\theta-\theta)^{2}+\cdots,$$

    and taking expectations kills the linear term because $\hat\theta$ is unbiased, leaving

    $$\mathbb{E}\big[g(\hat\theta)\big]=g(\theta)+\tfrac12 g''(\theta)\,v+O(n^{-2}).$$

    The bias is *curvature times variance*, its sign is the sign of $g''$, and both are known before any data arrives — for convex $g$ the plug-in estimate runs high and for concave $g$ it runs low, which is Jensen's inequality with a rate attached.

    The approximation becomes an identity in the case that matters most on a trading desk. Under normality $(n-1)s^{2}/\sigma^{2}\sim\chi^{2}_{n-1}$, and an inverse chi-squared with $\nu$ degrees of freedom has mean $1/(\nu-2)$, so

    $$\mathbb{E}\Big[\frac{1}{s^{2}}\Big]=\frac{n-1}{\sigma^{2}}\cdot\frac{1}{n-3}=\frac{n-1}{n-3}\cdot\frac{1}{\sigma^{2}},$$

    exactly, for every $n>3$ and with no expansion anywhere. Since $g(x)=1/x$ has $g''>0$ the sign agrees with the general result, and the magnitude exceeds the first-order term because the expansion has not converged at small $n$. Note also what the identity requires: at $n\le3$ the expectation is infinite, so the estimator every risk system computes has no mean at all on a week of data.

    The load-bearing step is the second-order term surviving an expectation that annihilated the first, which is what makes the bias a deterministic function of curvature rather than a property of the data. **Every plug-in estimate of a nonlinear quantity inherits the curvature of the map times the variance of its input, and matrix inversion is the most curved map on a trading desk** — which is the mechanism, stated three sections early, behind the failure the fifth section measures.

The inverse variance is not an academic example. It is the entire content of a Kelly fraction $f=\mu/\sigma^{2}$, of any inverse-variance weighting scheme, and of the $\Sigma^{-1}$ at the front of every mean-variance portfolio. [Kelly, Volatility Targeting, and Leverage](../../part-08-portfolio-management/02-kelly-vol-targeting-leverage.md) sizes positions with exactly this quantity, so the charge is worth measuring in the units it will be paid in.

```python
import numpy as np

rng = np.random.default_rng(11011)
reps, ann = 400_000, np.sqrt(252)
mu, sd = 0.075 / 252, 0.195 / ann                              # the course's SPY calibration
kelly = mu / sd ** 2

print("      n    E[1/s2]*sigma2    exact (n-1)/(n-3)    median f-hat/f    P(f-hat < 0)"
      "    P(f-hat > 2f)")
for n in (5, 21, 63, 252, 1260):
    x = mu + sd * rng.standard_normal((reps, n))
    f = x.mean(axis=1) / x.var(axis=1, ddof=1)
    print(f"  {n:5d} {(sd ** 2 / x.var(axis=1, ddof=1)).mean():17.5f}"
          f" {(n - 1) / (n - 3):20.5f} {np.median(f) / kelly:17.5f}"
          f" {(f < 0).mean():15.5f} {(f > 2 * kelly).mean():16.5f}")
# =>       n    E[1/s2]*sigma2    exact (n-1)/(n-3)    median f-hat/f    P(f-hat < 0)    P(f-hat > 2f)
#          5           2.00605              2.00000           1.01578         0.47777          0.47904
#         21           1.11206              1.11111           0.99193         0.45612          0.45548
#         63           1.03280              1.03333           0.99846         0.42425          0.42372
#        252           1.00793              1.00803           1.01201         0.34874          0.35214
#       1260           1.00163              1.00159           0.99596         0.19556          0.19466
```

The first two columns are the identity confirmed at every sample size: $2.00605$ against $2$, $1.11206$ against $1.11111$, $1.03280$ against $1.03333$, $1.00793$ against $1.00803$, $1.00163$ against $1.00159$. On a month of data the inverse variance a Kelly sizer consumes is overstated by $11\%$ before a single error enters the numerator, and the overstatement is a property of the arithmetic rather than of the market.

The third column is where the story turns. The *median* plug-in Kelly fraction is within a percent of the truth at every $n$ — $1.01578$, $0.99193$, $0.99846$, $1.01201$, $0.99596$ — so a practitioner watching typical outcomes sees a rule that looks unbiased. It is not: its mean carries the $11\%$ above and much more at $n=5$. **The estimator is roughly median-unbiased and badly mean-biased, and which of those two facts is the relevant one is a question about the loss function rather than about the estimator.**

The last two columns are what a bias calculation cannot reach and what a desk actually pays. On a year of daily data the plug-in Kelly fraction is *negative* $34.874\%$ of the time — a third of the time, a year of data instructs a Kelly sizer to short an asset with a positive drift — and it exceeds twice the correct size $35.214\%$ of the time. Five years brings both to about one in five, and even then the rule doubles the correct leverage in one deployment out of five. The $0.8\%$ mean bias at $n=252$ is a rounding error beside a rule that misses the sign a third of the time, and no debiasing touches the second number. **The curvature charge is exactly computable and is not the problem; the dispersion is the problem, and the two become comparable only under a criterion that scores them together.**

## A Loss Function Is the Only Thing That Makes One Estimator Better Than Another

Both readings of the table above are legitimate. Choosing between them, or between either and something else entirely, requires a statement of what an error costs — a **loss function** $L(\theta,a)\ge0$ giving the penalty for reporting $a$ when the truth is $\theta$, with $L(\theta,\theta)=0$. The loss is a modelling decision rather than a mathematical one, and it is the input that decides every ranking on this page.

Averaging the loss over the sampling distribution gives the **risk function**

$$R(\theta,\hat\theta)=\mathbb{E}_{\theta}\big[L\big(\theta,\hat\theta(X)\big)\big],$$

which is a function of $\theta$ — one number for each possible state of the world, not one number. Under squared-error loss $L(\theta,a)=(\theta-a)^{2}$ the risk is the mean squared error and the split into squared bias plus variance that [Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md) proves applies without change. Under absolute loss it is the mean absolute error and the split does not apply at all, which is worth knowing before quoting a bias–variance argument about a rule that will be graded on absolute error.

Squared-error loss is the default in this appendix for two reasons, and neither is that it describes trading. It is differentiable, so risks can be minimized in closed form; and it makes the risk decompose, so the two error sources separate. What it asserts about a desk is that an error of two units costs four times an error of one and that overestimating costs exactly what underestimating costs — and a leverage decision, a stop distance and a hedge ratio all violate the second clause. [Bayesian Estimation](05-bayesian-estimation.md) makes the asymmetry explicit and shows what it does to the reported number.

!!! note "The word risk on this page means expected loss under repeated sampling and the word risk on a trading desk means the dispersion of a realized profit and loss, and the two are different functions of different arguments"
    Decision-theoretic risk $R(\theta,\hat\theta)=\mathbb{E}_\theta[L]$ averages over samples that were never drawn, at a fixed unknown $\theta$; it is a property of an estimation procedure and it is not observable. Financial risk — volatility, value at risk, expected shortfall — is a functional of the distribution of a book's return at a fixed set of positions; it is a property of the book and it is estimated from data like anything else, in [Value at Risk](../part-18-quant-finance-applications/10-value-at-risk.md) and [Expected Shortfall](../part-18-quant-finance-applications/11-expected-shortfall.md). The collision is not merely verbal, because the two point in opposite directions on the same question: a rule with low estimation risk can produce a book with high financial risk, which is what the fifth section measures, and cutting financial risk by shrinking positions raises estimation risk for the quantity being shrunk. When both words appear in one sentence, the only safe reading is to substitute "expected loss" for the first and "dispersion of outcomes" for the second and check that the sentence still says something.

## Risk Functions Cross, So the Uniformly Best Estimator Exists Only When the Parameter Set Is a Point

With a loss chosen, "better" has a definition. $\hat\theta_1$ **dominates** $\hat\theta_2$ if $R(\theta,\hat\theta_1)\le R(\theta,\hat\theta_2)$ for every $\theta$, with strict inequality somewhere; an estimator dominated by nothing is **admissible**. Domination is the strongest comparison available and it is a partial order rather than a total one — most pairs of estimators are simply incomparable — and this is not a defect of the theory but the central fact about it.

??? note "Proof that no estimator has minimum risk at every parameter value, and that the constant rule which ignores the data entirely is admissible under squared-error loss"
    Suppose $\hat\theta^{\star}$ achieved the minimum risk at every $\theta\in\Theta$ simultaneously. Fix any $c\in\Theta$ and compare it with the constant rule $\hat\theta_c\equiv c$, which ignores the data. That rule has risk

    $$R(\theta,\hat\theta_c)=(\theta-c)^{2},$$

    which is zero at $\theta=c$. Minimality forces $R(c,\hat\theta^{\star})\le0$, hence $R(c,\hat\theta^{\star})=0$, hence $\hat\theta^{\star}=c$ almost surely under $\mathbf{P}_c$. Since $c$ was arbitrary, $\hat\theta^{\star}$ would have to equal every point of $\Theta$ almost surely, which is impossible unless $\Theta$ is a single point.

    The same computation shows $\hat\theta_c$ is admissible. Any competitor $\hat\theta$ dominating it must satisfy $R(c,\hat\theta)\le R(c,\hat\theta_c)=0$, so $\hat\theta=c$ almost surely under $\mathbf{P}_c$ — and for the models in use here, an estimator pinned to $c$ on a set of samples of full measure under $\mathbf{P}_c$ has no freedom left to be strictly better anywhere else. So the rule that discards the sample, costs nothing to compute and is wrong almost everywhere cannot be beaten uniformly by any rule whatever.

    The load-bearing quantity is the constant rule's zero risk at one point, which buys immunity to uniform domination at the price of unbounded risk everywhere else. **"Best estimator" has no referent, and the only two ways to manufacture one are to shrink the class of competitors until a best element exists — restrict to unbiased rules, which is [Properties of Estimators](02-properties-of-estimators.md) — or to collapse the risk function to a scalar by averaging it against a weighting over $\theta$, which is [Bayesian Estimation](05-bayesian-estimation.md).** Every later page in this part takes one of those two roads, and the choice is usually made by habit rather than by argument.

The theorem is abstract and the crossing it describes is not. Take the most-quoted estimate in the course — the equity premium, reported by [Probability and Random Variables](../../part-03-statistics/01-probability-and-random-variables.md) as `ann mean 0.075 +/- 0.039` with a $t$-statistic "of about 1.9, not even two standard errors from zero" — and race the sample mean against the constant rule $\hat\mu\equiv0$ and against a rule that splits the difference.

```python
import numpy as np

rng = np.random.default_rng(11013)
n, reps, ann = 6_158, 40_000, np.sqrt(252)
sd = 0.195 / ann                                               # 19.5% annualized, the course's SPY
se = 252 * sd / np.sqrt(n)

print(f"  n = {n}, one standard error = {se:.5f} annualized")
print("  true ann mu    risk(sample mean)    risk(constant 0)    risk(half-shrunk)"
      "    best lambda    winner")
for m in (0.0, 0.02, 0.039, 0.075, 0.126, 0.20):
    hat = 252 * (m / 252 + sd * rng.standard_normal((reps, n))).mean(axis=1)
    r = [((hat - m) ** 2).mean(), m ** 2, ((0.5 * hat - m) ** 2).mean()]
    who = ("sample mean", "constant 0", "half-shrunk")[int(np.argmin(r))]
    print(f"  {m:11.3f} {r[0]:20.6f} {r[1]:19.6f} {r[2]:20.6f}"
          f" {se ** 2 / (se ** 2 + m ** 2):14.3f}    {who}")
# =>   n = 6158, one standard error = 0.03945 annualized
#      true ann mu    risk(sample mean)    risk(constant 0)    risk(half-shrunk)    best lambda    winner
#            0.000             0.001552            0.000000             0.000388          1.000    constant 0
#            0.020             0.001552            0.000400             0.000491          0.796    constant 0
#            0.039             0.001570            0.001521             0.000776          0.506    half-shrunk
#            0.075             0.001556            0.005625             0.001796          0.217    sample mean
#            0.126             0.001522            0.015876             0.004339          0.089    sample mean
#            0.200             0.001561            0.040000             0.010385          0.037    sample mean
```

The first column is flat by construction: the sample mean's risk is $\mathrm{se}^{2}$ whatever the truth is, reading $0.001552$, $0.001552$, $0.001570$, $0.001556$, $0.001522$, $0.001561$ across a tenfold range of $\mu$. That flatness is the whole appeal of an unbiased estimator — it assumes nothing about the parameter and pays the same price everywhere. The second column is the constant rule's $\mu^{2}$, zero where it happens to be right and unbounded where it is not.

They cross, and they cross at exactly one standard error. At $\mu=0.039$ the two risks are $0.001570$ and $0.001521$, indistinguishable, and the crossing point is $\mu=\mathrm{se}=0.03945$ by construction, since $\mu^{2}=\mathrm{se}^{2}$ there. **The course's $\pm0.039$ is not only an error bar; it is the boundary at which a rule that never looks at the data starts losing to one that does**, and the published estimate of $0.075$ sitting less than two of these away is the same $t$-statistic of $1.9$ read as a decision rather than as a test.

The last two columns are where the page stops being a taxonomy. The half-shrunk rule wins at the crossing point and loses at both ends, so three rules produce three regions and no rule wins everywhere — the theorem, in a table. And the optimal shrinkage $\lambda^{\star}=\mathrm{se}^{2}/(\mathrm{se}^{2}+\mu^{2})$ runs $1.000$, $0.796$, $0.506$, $0.217$, $0.089$, $0.037$, so at the course's own published figure the risk-optimal move is to shrink the equity premium $21.7\%$ toward zero. **That number cannot be computed, because it depends on the $\mu$ being estimated** — a dependence [Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md) proved is unavoidable, and the one [Bayesian Estimation](05-bayesian-estimation.md) resolves by making the practitioner guess it out loud.

## The Standard Error Is the Report, and It Is the Only Part of the Estimate the Reader Cannot Reconstruct

Since the estimator has a distribution and the estimate does not, a reported estimate is incomplete without a summary of the rule's dispersion. The **standard error** is the standard deviation of the estimator's sampling distribution, $\mathrm{se}(\hat\theta)=\sqrt{\mathrm{var}(\hat\theta)}$, itself estimated from the same sample. It is the minimal honest report, and it is the only component of a published number a reader cannot recompute: given the estimate and the method anyone can reproduce the arithmetic, and nobody can reproduce the sampling variability without either the raw data or a stated standard error.

What a standard error does not do is propagate through a decision, and that is the failure that costs money. Every input to a portfolio optimizer can carry a standard error shrinking like $1/\sqrt n$ while the optimizer's *output* refuses to converge, because the map from inputs to positions is precisely the high-curvature inversion of the second section. The optimizer is not merely passing the error through; it is selecting on it, allocating most aggressively to whichever asset's mean was overestimated by the most.

```python
import numpy as np

rng = np.random.default_rng(11017)
k, rho, vol, ann, trials = 9, 0.58, 0.18, np.sqrt(252), 2_000
S = vol ** 2 * ((1 - rho) * np.eye(k) + rho * np.ones((k, k))) / 252
L, mu = np.linalg.cholesky(S), (0.06 + 0.010 * rng.standard_normal(k)) / 252
orc = np.linalg.solve(S, mu)
orc /= orc.sum()
eq = np.full(k, 1 / k)

print("  training T    MAE(mu-hat) ann    rel err Sigma    med OOS Sharpe MVO    1/N    oracle"
      "    med gross    P(SR < 0)")
for T in (126, 252, 504, 1260, 5040):
    sr, gr, em, es = [], [], [], []
    for _ in range(trials):
        x = mu + rng.standard_normal((T, k)) @ L.T
        mh, Sh = x.mean(axis=0), np.cov(x, rowvar=False)
        w = np.linalg.solve(Sh, mh)
        w /= w.sum()
        y = mu + rng.standard_normal((252, k)) @ L.T
        sr.append([ann * (y @ v).mean() / (y @ v).std() for v in (w, eq, orc)])
        gr.append(np.abs(w).sum())
        em.append(252 * np.abs(mh - mu).mean())
        es.append(np.linalg.norm(Sh - S) / np.linalg.norm(S))
    sr = np.array(sr)
    print(f"  {T:10d} {np.mean(em):18.5f} {np.mean(es):16.5f} {np.median(sr[:, 0]):21.3f}"
          f" {np.median(sr[:, 1]):6.3f} {np.median(sr[:, 2]):9.3f} {np.median(gr):12.2f}"
          f" {(sr[:, 0] < 0).mean():12.3f}")
# =>   training T    MAE(mu-hat) ann    rel err Sigma    med OOS Sharpe MVO    1/N    oracle    med gross    P(SR < 0)
#             126            0.20455          0.15387                 0.116  0.452     0.507        11.57        0.458
#             252            0.14467          0.11103                 0.109  0.445     0.497        10.66        0.453
#             504            0.10220          0.07705                 0.130  0.455     0.495        10.50        0.449
#            1260            0.06248          0.04931                 0.178  0.489     0.521         7.84        0.431
#            5040            0.03213          0.02454                 0.289  0.480     0.555         4.36        0.385
```

Columns two and three are estimation theory working perfectly. The mean absolute error of the estimated drift falls from $20.455$ annualized percentage points at six months to $3.213$ at twenty years, a factor of $6.4$ against the $\sqrt{40}=6.3$ the rate predicts. The covariance matrix's relative error falls from $0.15387$ to $0.02454$. Every input is consistent, every standard error shrinks on schedule, and nothing in either column hints at trouble.

Column four is the output. The plug-in optimizer's median out-of-sample Sharpe rises from $0.116$ to $0.289$ across a fortyfold increase in training data and never reaches the equal-weight portfolio's $0.480$, let alone the oracle's $0.555$ — and the oracle is the *same optimizer* handed the true $\mu$ and $\Sigma$, so the machinery is not at fault and the estimation is. After twenty years of daily history the gap between what the optimizer delivers with perfect inputs and what it delivers with excellent ones is still wider than the entire premium being harvested. **Consistency of every input is compatible with the output being beaten by a rule that reads none of them.**

The last two columns say what the loss actually was. Median gross exposure is $11.57\times$ at six months, against the $11.81\times$ the course published on two-year windows, so the leverage that lesson reported is not a quirk of its universe but a generic property of inverting an estimated covariance matrix against an estimated mean. And the optimizer loses money out of sample $45.8\%$ of the time at six months and still $38.5\%$ of the time after twenty years, so the decision's own error rate falls by seven percentage points while its inputs improve sixfold. **A standard error describes the wobble of an input and says nothing whatever about the wobble of a decision built from it, and only the second one trades.**

!!! warning "An estimator chosen because it is the natural formula has had its loss function chosen for it by whoever wrote the formula, and that loss is almost never the one the book is marked against"
    The mechanism is a default nobody notices making a decision nobody reviewed. Squared-error loss is built into the sample mean, into least squares, into every $R^{2}$ and every mean-squared-error column in this appendix, and it asserts symmetry between overestimating and underestimating together with a quadratic penalty for size. A leverage decision is not symmetric: an overestimate compounds into a drawdown and an underestimate costs foregone return, and the two are separated by a bankruptcy boundary squared error cannot see. A volatility forecast is not symmetric either, since underestimating it sizes a position too large in exactly the regime the forecast was wrong about. The tell is a research process that argues about estimators for weeks and has never once written down $L$. The free diagnostic is to reverse the order: **write the loss in the currency the book is denominated in, simulate the two candidate rules against a truth you set yourself, and read the risk in dollars — and if the two cannot be separated in dollars, the choice between them is not an estimation question and should stop consuming research time.** The second table is that diagnostic run once, and its answer is that the sample mean and the rule that ignores the data are separated nowhere except in a band one standard error wide, which is a finding about how little the choice matters and how much the crossing point does.

## The Estimator Is a Bet on a Loss Function Nobody Wrote Down

This page established that an estimand, an estimator and an estimate are three objects and only the middle one has a distribution; that the plug-in principle manufactures a rule from any functional and charges curvature times variance for it, exactly $(n-1)/(n-3)$ for the inverse variance a Kelly sizer consumes and $11\%$ on a month of data, with a dispersion that flips the sign of the recommendation a third of the time on a year; that a loss function is the only thing making one rule better than another, and the risk it induces is a function on the parameter space rather than a number; that risk functions cross, so no estimator is uniformly best and the rule ignoring the data cannot be uniformly beaten; and that the standard error of every input to a decision can converge while the decision does not.

The negative result is the useful one, because it says where the rest of this part comes from. There are exactly two ways to extract a winner from a partial order — delete competitors until a maximum exists, or collapse the order to a scalar — and the seven pages that follow are those two moves applied over and over. Restricting to unbiased estimators and asking which has least variance produces the information bound and the estimators attaining it. Restricting to a functional form produces maximum likelihood and the method of moments. Averaging risk against a prior produces the Bayes estimators, and maximizing the posterior instead of averaging it produces the penalized fits every desk already runs under a different name. None of these is more principled than the others. Each is a different admission about what was not known, and the admission is the part that goes unrecorded.

What survives all of them is the observation the fifth section measured. An optimizer whose inputs converge and whose output does not is not badly estimated; it is well estimated and badly posed, because the quantity it needed was never $\hat\mu$ or $\hat\Sigma$ but the decision built from them, and no standard error was ever computed for that. The discipline this part teaches is to name the estimand the decision actually consumes, put the loss on *that*, and treat everything upstream as machinery. Narrowing the class of rules far enough for anything to be promised at all is what [Properties of Estimators](02-properties-of-estimators.md) does next, and the promises turn out to be cheaper than they sound.

**A number is an answer, an estimator is a rule for producing answers, and the reason so many rules are wrong is that the question they answer was chosen by whoever wrote the formula and has not been read by anyone since.**
