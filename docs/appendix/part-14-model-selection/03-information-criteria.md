# Information Criteria (AIC/BIC)

Two formulas differing in one constant invite the reading that one is a stricter version of the other, and the usual gloss — BIC penalizes complexity more, so use it when you want a simpler model — treats the choice as a matter of taste. It is not. The two criteria are estimates of two different quantities, derived from two different arguments, and each is provably better than the other at the job it was built for and provably worse at the job it was not. Below, on a truth with two real predictors and six exact zeros, BIC identifies the true model $99.8\%$ of the time while AIC stalls at $72.5\%$ no matter how much data arrives — and BIC's predictions are twice as accurate. On a truth with no finite correct model, the ordering reverses at every sample size and AIC's prediction error is $0.0096$ against BIC's $0.0167$. Neither result is a defect. What is a defect is that the score is a density and therefore carries units: comparing a model of $y$ against a model of $\log y$ without the change-of-variable term picks the log model $1.0000$ of the time regardless of which one generated the data, and BIC's consistency, which section 2 proves, collapses to a $0.5\%$ hit rate when the rows are autocorrelated at $0.98$.

This page covers AIC as an estimate of expected Kullback–Leibler divergence with $2p$ arising as the optimism of a maximised log-likelihood, BIC as a Laplace approximation to a marginal likelihood and therefore an estimate of a posterior model probability, the theorem that no criterion is both efficient and consistent, the asymptotic equivalence of AIC and leave-one-out, and the three conditions under which any of these numbers stops meaning anything. It does not decompose prediction error into bias, variance and a floor, which is [Bias–Variance Tradeoff](01-bias-variance-tradeoff.md); it splits no data, which is [Cross-Validation](02-cross-validation.md); it searches over no unordered candidate sets, which is [Feature Selection](04-feature-selection.md); it combines no models, which is [Model Averaging](05-model-averaging.md); it computes no Bayes factor and no posterior over parameters, which is [Part XVI](../part-16-bayesian-statistics/index.md); it attaches no error rate to a nested comparison, which is [Likelihood Ratio Tests](../part-12-hypothesis-testing/06-likelihood-ratio-tests.md); it corrects nothing for the number of candidates examined, which is [Part XV](../part-15-multiple-testing/index.md); and it never treats a difference in score as a difference that has been tested.

The trading stake is a criterion declining to make a decision, in public. [Bayesian Methods and HMMs](../../part-03-statistics/06-bayesian-methods-and-hmms.md) fits Gaussian hidden Markov models to SPY returns and reports that "BIC improves at every $k$, and would keep improving past four", with increments of $2{,}614$ points from one state to two, $518$ from two to three, and $65$ from three to four, concluding that "the statistical criterion ranks fit; it does not make the modeling decision." Sections 3 and 5 are why that is the correct reading rather than a failure of the tool: BIC is consistent for a true model *within the candidate set*, a small Gaussian mixture is not the truth about returns, and the row count is not the effective sample size of a series with that much dependence. [Returns and Distributions](../../part-03-statistics/02-returns-and-distributions.md) supplies the companion case, where AIC crowns the normal-inverse-Gaussian at $-40115.1$ over a Student-$t$ at $-40062.7$ and the lesson overrules it on simplicity.

## AIC Estimates a Divergence, and Its Penalty Is the Optimism of a Maximised Likelihood Rather Than a Preference for Small Models

The quantity a predictive modeller wants is the Kullback–Leibler divergence from the truth $g$ to a fitted model — equivalently, since the $\mathbb{E}_g[\log g]$ term is common to all candidates, the expected log-likelihood of the fitted model on a *fresh* draw. That expectation cannot be evaluated, and the natural substitute, the maximised log-likelihood on the data at hand, is systematically too large by an amount that turns out not to depend on the truth.

??? note "Proof that the maximised log-likelihood overstates the expected log-likelihood on fresh data by $p$ to first order, so subtracting $p$ debiases it and $-2\hat\ell+2p$ estimates twice the expected divergence"

    Let $\hat\theta$ maximise $\ell(\theta)=\sum_i\log f(y_i\mid\theta)$ over a $p$-dimensional parameter, and let $\theta_0$ minimise the divergence within the model class. The target is $\mathbb{E}_{\tilde y}[\ell(\hat\theta;\tilde y)]$ for an independent replicate $\tilde y$, and the available quantity is $\ell(\hat\theta;y)$. Expand both to second order about $\theta_0$. On the training data, $\ell(\hat\theta;y)-\ell(\theta_0;y)\approx\tfrac{1}{2}(\hat\theta-\theta_0)^\top J(\hat\theta-\theta_0)$, where $J$ is the observed information; on the replicate the same expansion loses the term that made $\hat\theta$ a maximiser, giving $\mathbb{E}[\ell(\hat\theta;\tilde y)]-\mathbb{E}[\ell(\theta_0;\tilde y)]\approx-\tfrac{1}{2}(\hat\theta-\theta_0)^\top J(\hat\theta-\theta_0)$. The two differ by the whole quadratic form, twice, and since $\sqrt{n}(\hat\theta-\theta_0)$ is asymptotically normal with covariance $J^{-1}KJ^{-1}$ for $K$ the score covariance,
    $$\mathbb{E}\big[\ell(\hat\theta;y)-\ell(\hat\theta;\tilde y)\big]\approx\operatorname{tr}(J^{-1}K).$$
    When the model is correctly specified the information equality gives $J=K$ and the trace collapses to $p$, the parameter count. Multiplying by $-2$ for the deviance scale yields $\mathrm{AIC}=-2\hat\ell+2p$.

    Three things are worth extracting. The penalty is not a judgement about parsimony — it is a bias correction, and its size was computed rather than chosen, which is why $2$ and not $3$ or $1.5$. The collapse of $\operatorname{tr}(J^{-1}K)$ to $p$ needs the model to be right or close to it, and when it is not the correct penalty is the trace, which is the Takeuchi criterion and is almost never used because estimating $K$ well is harder than the original problem. And the expansion is asymptotic in $n/p$, which is why the small-sample correction $\mathrm{AICc}=\mathrm{AIC}+2p(p+1)/(n-p-1)$ exists and why it matters below about forty observations per parameter.

    The load-bearing fact is that the target is prediction, evaluated at the same sample size. **AIC does not ask which model is true; it asks which model will score best on a fresh draw of the same size, and a model does not have to be true to win that contest.**

The optimism this proof isolates is the same quantity [Bias–Variance Tradeoff](01-bias-variance-tradeoff.md) computed geometrically as $2\sigma^{2}\operatorname{tr}(S)/n$ for a linear smoother. Mallows' $C_p$ is that version written for squared error, and for Gaussian models with known variance $C_p$ and AIC rank candidates identically. The three arrive at one penalty from a projection, a likelihood and a divergence.

## BIC Comes From a Laplace Approximation to a Marginal Likelihood, So It Estimates a Posterior Probability and Not an Error

BIC's formula differs from AIC's only in replacing $2$ with $\log n$, and the derivation shares nothing with section 1's.

??? note "Proof that $-2\log\int f(y\mid\theta)\pi(\theta)\,d\theta=-2\hat\ell+p\log n+O(1)$, so BIC ranks candidates by posterior probability under equal priors"

    The Bayesian evidence for a model is $m(y)=\int f(y\mid\theta)\pi(\theta)\,d\theta$. Write the integrand as $\exp\{\ell(\theta)\}\pi(\theta)$ and expand $\ell$ about its maximum: $\ell(\theta)\approx\hat\ell-\tfrac{n}{2}(\theta-\hat\theta)^\top \bar J(\theta-\hat\theta)$ with $\bar J$ the average observed information. The Gaussian integral evaluates to
    $$m(y)\approx e^{\hat\ell}\,\pi(\hat\theta)\,(2\pi)^{p/2}n^{-p/2}\lvert\bar J\rvert^{-1/2},$$
    so $-2\log m(y)=-2\hat\ell+p\log n-p\log(2\pi)+\log\lvert\bar J\rvert-2\log\pi(\hat\theta)$. Every term after $p\log n$ is $O(1)$ in $n$ and identical across candidates only to that order, so dropping them gives $\mathrm{BIC}=-2\hat\ell+p\log n$. Under equal prior model probabilities, differences in BIC therefore approximate twice the log posterior odds.

    Consistency follows directly. Comparing a true model against one with $q$ extra parameters, the likelihood-ratio term $2(\hat\ell_{\text{big}}-\hat\ell_{\text{true}})$ converges in distribution to $\chi^{2}_{q}$ — bounded in probability — while the penalty difference $q\log n$ diverges. So the probability of preferring the larger model goes to zero. Against a model missing a real predictor, the likelihood term grows like $n$ and beats $\log n$, so underfitting also vanishes. Both errors go to zero and BIC selects the true model with probability tending to one, *provided the true model is among the candidates*.

    That proviso is the whole content of the difference between the two criteria. AIC's derivation never mentions a true model in the candidate set and its target is defined whether or not one exists; BIC's target is the posterior probability of a discrete hypothesis, and if none of the hypotheses is true then the quantity being approximated is the probability of something false.

    The load-bearing asymmetry is that $\log n\to\infty$ while $2$ does not. **BIC's penalty grows with the data because it is buying certainty about which hypothesis is true, and AIC's does not because it is buying accuracy on the next observation, and no amount of data makes those the same purchase.**

## AIC Is Efficient, BIC Is Consistent, and No Criterion Is Both, So Their Disagreement Is a Property of the World Rather Than of the Analyst

The two derivations predict a specific pattern: BIC should win where a true finite model exists, AIC should win where one does not, and the loser in each case should lose on the metric it was not built for. The prediction is testable on the same candidate set, changing only the truth behind the data:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(14031)
P, sig, reps = 8, 1.0, 3000
sparse = np.array([1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])     # true size 2
taper = 1.2 / np.arange(1.0, P + 1.0) ** 1.5                    # no finite truth


def run(beta, n):
    Xv = rng.standard_normal((20_000, P))
    yv = Xv @ beta
    pick = np.zeros((2, P + 1), int)
    err = np.zeros(2)
    for _ in range(reps):
        X = rng.standard_normal((n, P))
        y = X @ beta + rng.normal(0.0, sig, n)
        Q, R = np.linalg.qr(X)
        g = Q.T @ y
        rss = np.sum(y**2) - np.cumsum(g**2)                    # nested, all at once
        k = np.arange(1, P + 1) + 1.0                           # params incl. sigma
        crit = (n * np.log(rss / n) + 2 * k, n * np.log(rss / n) + k * np.log(n))
        for c, sc in enumerate(crit):
            j = int(np.argmin(sc)) + 1
            b = np.linalg.solve(R[:j, :j], g[:j])
            pick[c, j] += 1
            err[c] += np.mean((yv - Xv[:, :j] @ b) ** 2)
    return pick, err / reps


for tag, beta in (("sparse truth: 2 real predictors, 6 exactly zero", sparse),
                  ("tapering truth: 8 predictors with decaying coefficients", taper)):
    print(f"  {tag}")
    print("        n    AIC size   BIC size   P(AIC = true)   P(BIC = true)"
          "    AIC OOS    BIC OOS")
    for n in (50, 200, 1000, 5000):
        pick, err = run(beta, n)
        tot = pick.sum(1)
        sz = [(pick[c] * np.arange(P + 1)).sum() / tot[c] for c in (0, 1)]
        hit = [pick[c, 2] / tot[c] for c in (0, 1)]
        star = "" if beta is sparse else "*"
        print(f"    {n:5d}    {sz[0]:8.3f}   {sz[1]:8.3f}   {hit[0]:13.3f}{star}"
              f"   {hit[1]:12.3f}{star}   {err[0]:8.4f}   {err[1]:8.4f}")
    if beta is not sparse:
        print("      * no true model exists here; the column is P(size = 2)")

n, over = 4000, 0                                   # one surplus parameter only
for _ in range(20_000):
    X = rng.standard_normal((n, 3))
    y = X @ sparse[:3] + rng.normal(0.0, sig, n)
    g = np.linalg.qr(X)[0].T @ y
    rss = np.sum(y**2) - np.cumsum(g**2)
    over += n * np.log(rss[2] / n) + 6 < n * np.log(rss[1] / n) + 4
print(f"  with exactly one surplus parameter at n = {n}, AIC takes it "
      f"{over / 20_000:.4f} of the time")
print(f"  against P(chi-squared with 1 df > 2) = {stats.chi2.sf(2.0, 1):.4f}")
# =>   sparse truth: 2 real predictors, 6 exactly zero
#            n    AIC size   BIC size   P(AIC = true)   P(BIC = true)    AIC OOS    BIC OOS
#           50       2.833      2.116           0.676          0.924     0.0959     0.0557
#          200       2.684      2.028           0.709          0.976     0.0202     0.0111
#         1000       2.634      2.009           0.730          0.991     0.0039     0.0021
#         5000       2.630      2.002           0.725          0.998     0.0008     0.0004
#      tapering truth: 8 predictors with decaying coefficients
#            n    AIC size   BIC size   P(AIC = true)   P(BIC = true)    AIC OOS    BIC OOS
#           50       3.963      2.514           0.223*          0.444*     0.1640     0.1691
#          200       5.512      3.599           0.008*          0.121*     0.0466     0.0629
#         1000       7.367      5.715           0.000*          0.000*     0.0096     0.0167
#         5000       7.990      7.782           0.000*          0.000*     0.0016     0.0021
#          * no true model exists here; the column is P(size = 2)
#      with exactly one surplus parameter at n = 4000, AIC takes it 0.1520 of the time
#      against P(chi-squared with 1 df > 2) = 0.1573
```

The top block is BIC's theorem, discharged. Its probability of recovering the true model runs $0.924$, $0.976$, $0.991$, $0.998$ — converging to one, exactly as the Laplace argument requires — and its mean selected size settles onto $2.002$ against a truth of $2$. AIC's hit rate runs $0.676$, $0.709$, $0.730$, $0.725$ and stops. It is not converging slowly; it is not converging. Its mean size settles at $2.630$, so it carries roughly two thirds of a surplus parameter forever, and the last two lines say why: with exactly one surplus candidate available, AIC takes it $0.1520$ of the time against the asymptotic $P(\chi^{2}_{1}>2)=0.1573$. The threshold that AIC applies to a likelihood-ratio statistic is the fixed number $2$, so its over-selection probability is a fixed tail area, and no quantity of data changes a fixed tail area.

The prediction columns in that same block are the part that stops the argument from being a criticism. BIC's out-of-sample error is $0.0557$, $0.0111$, $0.0021$ and $0.0004$ against AIC's $0.0959$, $0.0202$, $0.0039$ and $0.0008$ — roughly half, at every sample size. When a true sparse model exists, the criterion that finds it also predicts better, and AIC's extra parameters are pure variance.

The lower block reverses every column. With coefficients that decay rather than vanish there is no true finite model, so both criteria are chasing a moving approximation, and AIC's error of $0.1640$, $0.0466$, $0.0096$ and $0.0016$ beats BIC's $0.1691$, $0.0629$, $0.0167$ and $0.0021$ at every sample size — by a factor of $1.7$ at $n=1000$. BIC's growing penalty, which was exactly right when it was buying certainty about a discrete hypothesis, is now refusing to spend parameters on real signal: its mean size lags at $5.715$ where AIC has reached $7.367$, and the coefficients BIC declines to estimate are not zero.

**The two criteria do not disagree about a model; they disagree about whether the truth is in the room, and that is a question about the world which no amount of data internal to the comparison can settle.**

## AIC and Leave-One-Out Are the Same Rule Arrived at From Opposite Directions

Section 1 estimated the optimism of a likelihood analytically; [Cross-Validation](02-cross-validation.md) measured the same optimism by withholding data. The two should agree, and Stone's theorem says they do asymptotically. How fast is a measurement:

```python
import numpy as np

rng = np.random.default_rng(14033)
P, sig, reps = 8, 1.0, 3000
taper = 1.2 / np.arange(1.0, P + 1.0) ** 1.5

print(f"  AIC against leave-one-out on the same nested candidates, {reps:,} datasets")
print("        n   agree   AIC size   LOO size   BIC size    AIC OOS    LOO OOS"
      "    BIC OOS")
for n in (40, 100, 400, 2000):
    Xv = rng.standard_normal((20_000, P))
    yv = Xv @ taper
    agree, size, err = 0, np.zeros(3), np.zeros(3)
    for _ in range(reps):
        X = rng.standard_normal((n, P))
        y = X @ taper + rng.normal(0.0, sig, n)
        Q, R = np.linalg.qr(X)
        g = Q.T @ y
        rss = np.sum(y**2) - np.cumsum(g**2)
        hc = np.cumsum(Q**2, axis=1)                    # leverage of each nested fit
        press = np.array([np.sum(((y - Q[:, :j] @ g[:j]) / (1 - hc[:, j - 1])) ** 2)
                          for j in range(1, P + 1)])
        k = np.arange(1, P + 1) + 1.0
        crit = (n * np.log(rss / n) + 2 * k, press, n * np.log(rss / n) + k * np.log(n))
        js = [int(np.argmin(c)) + 1 for c in crit]
        agree += js[0] == js[1]
        for c, j in enumerate(js):
            size[c] += j
            err[c] += np.mean((yv - Xv[:, :j] @ np.linalg.solve(R[:j, :j], g[:j])) ** 2)
    print(f"    {n:5d}  {agree / reps:6.3f}" + "".join(f"{v / reps:11.3f}" for v in size)
          + "".join(f"{v / reps:11.4f}" for v in err))
# =>   AIC against leave-one-out on the same nested candidates, 3,000 datasets
#            n   agree   AIC size   LOO size   BIC size    AIC OOS    LOO OOS    BIC OOS
#           40   0.804      3.798      3.491      2.426     0.2051     0.1989     0.2007
#          100   0.870      4.695      4.536      2.976     0.0868     0.0867     0.1024
#          400   0.954      6.443      6.418      4.390     0.0237     0.0238     0.0360
#         2000   0.993      7.816      7.814      6.747     0.0043     0.0043     0.0078
```

The agreement column runs $0.804$, $0.870$, $0.954$ and $0.993$: two procedures with nothing in common operationally — one computes a penalty from a parameter count, the other refits the model $n$ times — pick the identical model out of eight candidates on $99.3\%$ of datasets by $n=2000$. The selected sizes converge with it, $3.798$ against $3.491$ at $n=40$ closing to $7.816$ against $7.814$, and the out-of-sample errors are indistinguishable from $n=100$ onward at $0.0868$ against $0.0867$, $0.0237$ against $0.0238$ and $0.0043$ against $0.0043$.

The small-sample column is where they differ and it favours resampling: at $n=40$, leave-one-out achieves $0.1989$ against AIC's $0.2051$, choosing a slightly smaller model at $3.491$ against $3.798$. This is section 1's asymptotic expansion showing its age at five observations per parameter, and it is the regime AICc was invented for. BIC, meanwhile, is a different rule throughout — $2.426$, $2.976$, $4.390$, $6.747$ — and pays for it here in exactly the way section 3 predicts.

**A criterion and a resampling scheme are not alternatives between which one chooses on philosophical grounds; on this class of problem they are the same estimator, one of which costs one fit and the other $n$, and the choice between them is about assumptions and arithmetic rather than about outlook.**

## The Score Is a Density and the Sample Size Is Not a Row Count, So Two Routine Operations Void the Comparison Silently

Everything above assumed the candidates were densities for the same random variable, fitted to independent observations. Both assumptions are broken casually and neither breakage announces itself:

```python
import numpy as np

rng = np.random.default_rng(14035)
n, reps = 500, 3000


def gauss_ll(v, m):                       # maximised Gaussian log-likelihood
    return -0.5 * m * (np.log(2 * np.pi * v) + 1.0)


print(f"  a likelihood is a density for a particular random variable, n = {n}")
print("    truth      AIC on y   AIC on log y, naive   Jacobian   corrected"
      "   naive picks log   corrected picks log")
for tag, gen in (("y Gaussian ", lambda: rng.normal(100.0, 10.0, n)),
                 ("log y Gauss", lambda: np.exp(rng.normal(4.6, 0.1, n)))):
    naive_wins = corr_wins = 0
    for _ in range(reps):
        y = gen()
        a = 2 * 2 - 2 * gauss_ll(y.var(), n)
        b = 2 * 2 - 2 * gauss_ll(np.log(y).var(), n)
        naive_wins += b < a
        corr_wins += b + 2 * np.sum(np.log(y)) < a
    y = gen()
    a = 2 * 2 - 2 * gauss_ll(y.var(), n)
    b = 2 * 2 - 2 * gauss_ll(np.log(y).var(), n)
    jac = 2 * np.sum(np.log(y))
    print(f"    {tag}  {a:8.1f}   {b:19.1f}   {jac:8.1f}   {b + jac:9.1f}"
          f"   {naive_wins / reps:15.4f}   {corr_wins / reps:19.4f}")

print("  the n in BIC's log(n) is an effective sample size, not a row count")
print("    autocorrelation   BIC size   P(BIC = true)   rows   effective rows")
P, beta = 8, np.array([1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
for phi in (0.0, 0.5, 0.9, 0.98):
    size = hit = 0
    for _ in range(reps):
        E = rng.standard_normal((n, P + 1))
        for t in range(1, n):
            E[t] = phi * E[t - 1] + np.sqrt(1 - phi**2) * E[t]
        X, y = E[:, :P], E[:, :P] @ beta + E[:, P]
        g = np.linalg.qr(X)[0].T @ y
        rss = np.sum(y**2) - np.cumsum(g**2)
        k = np.arange(1, P + 1) + 1.0
        j = int(np.argmin(n * np.log(rss / n) + k * np.log(n))) + 1
        size += j
        hit += j == 2
    neff = n * (1 - phi) / (1 + phi)
    print(f"    {phi:15.2f}   {size / reps:8.3f}   {hit / reps:13.3f}   {n:4d}"
          f"   {neff:14.1f}")
# =>   a likelihood is a density for a particular random variable, n = 500
#        truth      AIC on y   AIC on log y, naive   Jacobian   corrected   naive picks log   corrected picks log
#        y Gaussian     3701.4                -889.8     4598.6      3708.8            1.0000                0.0827
#        log y Gauss    3774.6                -828.4     4600.9      3772.5            1.0000                0.9050
#      the n in BIC's log(n) is an effective sample size, not a row count
#        autocorrelation   BIC size   P(BIC = true)   rows   effective rows
#                   0.00      2.016           0.986    500            500.0
#                   0.50      2.095           0.935    500            166.7
#                   0.90      5.484           0.170    500             26.3
#                   0.98      7.406           0.005    500              5.1
```

The first block's last two columns are the finding. The naive comparison picks the log model $1.0000$ of the time in *both* rows — when the data is Gaussian in levels and when it is Gaussian in logs. It is not comparing models at all. Taking logs of data centred near $100$ compresses the spread by a factor of about a hundred, and a density concentrated on a hundredth of the range is a hundred times taller, so the log-scale likelihood is larger by $n\log 100$ whatever the shape of the distribution. The gap is $3701.4$ against $-889.8$: four and a half thousand AIC points of pure unit conversion, in a setting where practitioners argue over differences of two.

The Jacobian term $2\sum_i\log y_i$ is the correction, at $4598.6$ and $4600.9$ — nearly identical in the two rows, as it must be since it depends on the data's scale and not on which model is right. With it applied the comparison starts working: the levels-Gaussian row picks the log model only $0.0827$ of the time and the log-Gaussian row picks it $0.9050$ of the time. **A likelihood is a density with respect to a measure, so a criterion built from likelihoods can only compare models of the same random variable, and transforming the response is not preprocessing but a change of the question.**

The second block breaks BIC's consistency without touching BIC. The data-generating process is unchanged — two real predictors, six exact zeros, $500$ rows — and only the serial dependence of the rows moves. At $\phi=0$ BIC behaves as section 2 promises, selecting size $2.016$ and hitting the truth $0.986$ of the time. At $\phi=0.5$ it holds at $0.935$. At $\phi=0.9$ it collapses to $0.170$ and takes $5.484$ predictors; at $\phi=0.98$ it takes $7.406$ of $8$ and finds the true model $0.5\%$ of the time. The final column is the diagnosis: Kish's effective sample size $n(1-\phi)/(1+\phi)$ reads $500$, $166.7$, $26.3$ and $5.1$. BIC is charging $\log 500 = 6.2$ per parameter for information worth $\log 5.1 = 1.6$, so the penalty is nearly four times too *small* relative to the evidence, and spurious predictors clear it easily.

!!! note "AIC, AICc, BIC, Mallows' $C_p$ and the adjusted $R^{2}$ are five penalties applied to one fit, and they are estimating four different quantities"
    The family resemblance conceals the disagreement. **AIC** estimates expected Kullback–Leibler divergence and its $2p$ is the optimism derived in section 1. **AICc** is AIC with the finite-sample term $2p(p+1)/(n-p-1)$, the same target with a better expansion, and it matters below roughly forty observations per parameter — section 4's $n=40$ row is where its absence costs AIC $0.2051$ against leave-one-out's $0.1989$. **BIC** estimates a log marginal likelihood and therefore a posterior model probability, a different target reached by section 2's Laplace argument, which is why its penalty grows with $n$ and AIC's does not. **Mallows' $C_p$** is AIC's optimism written for squared error with $\sigma^{2}$ estimated from a large working model, so it ranks Gaussian candidates identically to AIC and inherits the working model's problems, which [Bias–Variance Tradeoff](01-bias-variance-tradeoff.md) measured as an implied squared bias of $-0.2463$. **Adjusted $R^{2}$** penalizes by $(n-1)/(n-p-1)$ and corresponds to selecting on the residual variance estimate; it is far weaker than any of the others and is the only one on the list not derived as an estimate of anything. Reporting "the information criterion" without naming it therefore states a fit and withholds the question.

!!! warning "A criterion returns a ranking whether or not any of its derivation's conditions hold, and the conditions are properties of the data rather than of the candidates"
    Nothing in the output is conditional. Section 1's $2p$ needed the information equality, which needs a roughly correct model; section 2's consistency needed the true model to be among the candidates, which on financial data it never is; both needed independent observations, and section 5 shows the row count standing in for an effective sample size that was $5.1$; and both needed every candidate to be a density for the same variable, which a single `np.log` silently voids at a cost measured above as $4{,}598.6$ points. The course's own encounter is the honest version: [Bayesian Methods and HMMs](../../part-03-statistics/06-bayesian-methods-and-hmms.md) watches BIC improve at every state count and declines to follow it, on the grounds that "a state you cannot name is a state you cannot trade". **The free diagnostic is to check the criterion's implied error rate before believing a ranking: a difference of $\Delta$ between nested candidates differing by $q$ parameters corresponds to a likelihood-ratio statistic of $\Delta+2q$ for AIC, so compare it against the $\chi^{2}_{q}$ distribution and read off the tail area — if $\Delta=2$ on one parameter you are acting on a $p$-value of $0.157$, which is a number no one would report as evidence, and if the same comparison run on a block-bootstrapped resample of your own data flips the winner more than a fifth of the time then the ranking is describing your sample's dependence rather than your models.**

## A Criterion Prices Complexity Without Splitting Data, and the Price Is a Set of Assumptions the Output Cannot Display

This page established that AIC's $2p$ is the measured optimism of a maximised log-likelihood, equal to $\operatorname{tr}(J^{-1}K)$ in general and to $p$ under correct specification, so the penalty was computed rather than chosen; that BIC's $p\log n$ comes from a Laplace approximation to a marginal likelihood and therefore targets a posterior probability, giving consistency whenever the true model is among the candidates; that the two are provably optimal for different worlds, BIC recovering a sparse truth at $0.924$, $0.976$, $0.991$ and $0.998$ while AIC stalled at $0.676$, $0.709$, $0.730$ and $0.725$ and predicted twice as badly at $0.0959$ against $0.0557$, with every column reversing under a tapering truth where AIC's $0.0096$ beat BIC's $0.0167$; that AIC's refusal to converge is a fixed tail area, taking one surplus parameter $0.1520$ of the time against $P(\chi^{2}_{1}>2)=0.1573$; that AIC and leave-one-out select identically on $0.804$, $0.870$, $0.954$ and $0.993$ of datasets with out-of-sample errors agreeing to four decimals from $n=100$; and that the comparison voids silently under two routine operations, a change of response variable moving the score by $4{,}598.6$ points and driving the naive verdict to the log model $1.0000$ of the time whichever model was true, and serial dependence at $\phi=0.98$ cutting BIC's hit rate from $0.986$ to $0.005$ against an effective sample size of $5.1$ rows out of $500$.

The trade against [Cross-Validation](02-cross-validation.md) is now explicit. A criterion costs one fit where resampling costs $K$, needs no partition and therefore imports none of a partition's properties, and section 4 shows it reaching the same answer. It pays for that with assumptions — a likelihood, an asymptotic expansion, a correctly specified model for the information equality, a true model in the room for consistency, independent rows for both — where cross-validation assumed only that the held-out data was independent of the training data. Neither is the safe choice. They fail on different inputs, which is the argument for computing both and treating disagreement as information rather than as a tie to be broken.

What both share is a candidate set someone else assembled. Every number on this page ranked eight nested models that arrived already ordered, and the ranking was cheap precisely because the ordering was given. Remove it — let the candidates be an unordered collection of subsets of $p$ predictors, of which there are $2^{p}$ — and the criterion still evaluates each one correctly while the act of searching acquires a cost that no criterion charges for. That is [Feature Selection](04-feature-selection.md).

**An information criterion converts a model comparison into arithmetic on a single fit, and every assumption that made the conversion valid is discharged before the number is printed and recorded nowhere on it.**
