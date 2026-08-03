# Maximum A Posteriori Estimation

Every summary on the previous page required an integral. The mean, the median, the quantiles and the expected losses that select them are all integrals over the posterior, and outside the conjugate families integrals are the expensive part of Bayesian work. The posterior mode is the exception. It needs no normalizing constant, no quadrature and no sampler — only a maximization, which is the operation every fitting library already performs — and that single fact explains why it is the Bayesian estimate practitioners actually compute, usually without calling it Bayesian at all. Every ridge regression, every $L^{1}$ penalty, every weight-decay term in a neural network is a maximum a posteriori estimate under a prior the person who tuned it never wrote down. What the cheapness costs is stated plainly here: the mode minimizes no expected loss, and it moves when you change coordinates.

This page covers the posterior mode as the summary that needs no integral, its identity with penalized maximum likelihood, the correspondence between a Gaussian prior and ridge and between a Laplace prior and lasso with the penalty weight fixed by a ratio of variances, the failure of the mode to survive reparameterization while the median does, and the flat-prior limit in which the whole apparatus collapses back to maximum likelihood. It computes no posterior mean and minimizes no expected loss, which is [Bayesian Estimation](05-bayesian-estimation.md); it argues for no prior, which is [Prior Distributions](../part-16-bayesian-statistics/02-prior-distributions.md); it derives no posterior in closed form, which are [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) and [Conjugate Priors](../part-16-bayesian-statistics/04-conjugate-priors.md); it fits no penalized regression and tunes no penalty by resampling, which are [Regularization](../part-13-regression/05-regularization.md) and [Cross-Validation](../part-14-model-selection/02-cross-validation.md); it maximizes no unpenalized likelihood, which is [Maximum Likelihood Estimation](03-maximum-likelihood-estimation.md); it runs no optimizer over a non-concave surface, which is [Numerical Optimization](../part-17-statistical-computing/01-numerical-optimization.md); it derives no Jacobian, which is [Change of Variables](../part-03-random-variables/09-change-of-variables.md); and it never treats the mode of a posterior as a summary of the posterior.

The trading stake is a prior that a course lesson identifies with a multiple-testing correction. [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) argues that a tight prior centred at zero "encodes 'almost every backtest that reaches my desk has no edge,' which is lesson four's multiple-testing correction re-expressed as a belief" and concludes that "**A desk that runs skeptical priors is running Bonferroni continuously, without the ceremony.**" Its skeptical `N(0, 1bp)` prior turns a $+5.8\%$ estimated edge into $+1.7\%$ with `P(edge > 0) 0.79`. The second and third sections price that: the prior is a ridge penalty of $\lambda=14781.3$, its shrinkage weight of $0.7059$ is fixed before any return is loaded, and the same idea expressed as a Laplace prior deletes forty-seven of fifty strategies outright.

## The Posterior Mode Is the One Bayesian Summary That Needs No Integral

The **maximum a posteriori** estimate is

$$\hat\theta_{\mathrm{MAP}}=\arg\max_\theta\ \pi(\theta\mid x)=\arg\max_\theta\ \pi(\theta)f(x\mid\theta),$$

and the second equality is the whole story. The posterior's denominator, $\int\pi(\theta')f(x\mid\theta')\,d\theta'$, does not depend on $\theta$, so it cannot move the location of the maximum and can be dropped entirely. That is the same observation [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md) makes about maximum likelihood — "maximizing the likelihood over $A$ ignores that constant entirely, which is legitimate" — applied one level up, and it is what makes the mode computable on any model where the prior and the likelihood can be evaluated pointwise.

The consequence is a sharp asymmetry in cost. Computing a posterior mean over a fifty-parameter model requires either conjugacy or a sampler; computing the mode requires calling an optimizer on a function you can already write down. In practice that means MAP is available whenever maximum likelihood is available, at the cost of one extra additive term, and every other summary is not.

!!! note "The posterior mode is a Bayesian object computed by a frequentist operation, and it is the only summary on these two pages that minimizes no expected loss"
    [Bayesian Estimation](05-bayesian-estimation.md) derived every summary from a loss: squared error gives the mean, absolute error gives the median, an asymmetric linear loss gives a quantile. There is no loss function on a continuous parameter whose Bayes rule is the mode. The usual rescue is a zero–one loss that charges nothing inside a window of width $\epsilon$ around the truth and one outside it, whose Bayes rule is the centre of the most probable window — and that does tend to the mode as $\epsilon\to0$, but the limit is exactly where the invariance of the fourth section dies, because a window of fixed width in $\sigma$ is not a window of fixed width in $\sigma^{2}$. So the mode is a summary chosen for computability and then retrofitted with a decision-theoretic story that holds only in a limit that reintroduces the coordinate dependence. Ownership is worth stating too, because the names collide across parts: the posterior mode and its penalized-likelihood identity are this page; the posterior mean as the $L^{2}$-optimal estimate is [Bayesian Estimation](05-bayesian-estimation.md); and the construction of priors, the derivation of posteriors and the mechanics of updating are [Part XVI](../part-16-bayesian-statistics/index.md), whose [The Bayesian Framework](../part-16-bayesian-statistics/01-bayesian-framework.md) sets the vocabulary these two pages spend.

## MAP Is Penalized Maximum Likelihood and the Penalty Is the Log Prior

Taking logs of the product turns the maximization into a familiar object.

??? note "Proof that maximizing a posterior is minimizing a penalized negative log-likelihood, with a Gaussian prior giving ridge and a Laplace prior lasso, and that the penalty weight is a ratio of variances rather than a tuning parameter"
    Take logs of $\pi(\theta)f(x\mid\theta)$ and negate:

    $$\hat\theta_{\mathrm{MAP}}=\arg\min_\theta\ \big[-\ell(\theta)-\log\pi(\theta)\big],$$

    so the log prior is an additive penalty on the negative log-likelihood and nothing else.

    With $x_i\sim\mathcal{N}(\theta,\sigma^{2})$ and $\theta\sim\mathcal{N}(0,\tau^{2})$, the objective is

    $$\frac{1}{2\sigma^{2}}\sum_i(x_i-\theta)^{2}+\frac{\theta^{2}}{2\tau^{2}}\ \propto\ \sum_i(x_i-\theta)^{2}+\lambda\theta^{2},\qquad \lambda=\frac{\sigma^{2}}{\tau^{2}},$$

    which is ridge regression with the penalty weight fixed by the ratio of the noise variance to the prior variance. Solving gives $\hat\theta=n\bar x/(n+\lambda)$, identical to the posterior mean because a normal posterior is symmetric.

    With a Laplace prior $\pi(\theta)\propto\exp(-|\theta|/b)$ the penalty is $|\theta|\,\sigma^{2}/b$, which is lasso, and the solution is the soft threshold

    $$\hat\theta=\mathrm{sign}(\bar x)\Big(|\bar x|-\frac{\sigma^{2}}{nb}\Big)^{+},$$

    exactly zero whenever the evidence falls short of the threshold. A prior with a kink at the origin produces an estimator with a flat region at the origin, and a prior that is smooth there cannot.

    The load-bearing step is that $\int\pi f$ is constant in $\theta$ and drops out. That is what makes MAP computable without an integral, and it is also what makes it silent about model comparison: the quantity it discards is the marginal likelihood, which is the entire content of [Bayesian Model Comparison](../part-16-bayesian-statistics/06-bayesian-model-comparison.md). **Every ridge penalty a desk tunes by cross-validation is a Gaussian prior whose variance the desk declined to state, and stating it is strictly more informative than tuning it** — because $\tau=\sigma/\sqrt\lambda$ is a number in the units of the effect being estimated, and a practitioner can say whether it is plausible.

```python
import numpy as np

rng = np.random.default_rng(11061)
n, reps = 6_158, 200_000
s2 = (0.193 / np.sqrt(252)) ** 2                               # the course's momentum strategy
truth = 0.058 / 252                                            # the flat-prior answer, +5.8% a year
sim = rng.normal(truth, np.sqrt(s2 / n), reps)

print("  prior sd tau    implied ridge lambda    weight on the prior    MAP ann mean"
      "    max |MAP - ridge|    E[MAP] at that truth    sd(MAP)")
for tau in (1e-4, 2e-4, 5e-4, 1e-3, 1e-2):
    w = (s2 / n) / (tau ** 2 + s2 / n)
    lam = s2 / tau ** 2
    m, ridge = (1 - w) * truth, n * truth / (n + lam)          # argmin sum (x-th)^2 + lam th^2
    f = (1 - w) * sim
    print(f"  {tau:12.1e} {lam:23.4f} {w:22.4f} {252 * m:15.4%} {abs(m - ridge):20.2e}"
          f" {252 * f.mean():23.4%} {252 * f.std():10.4%}")
# =>   prior sd tau    implied ridge lambda    weight on the prior    MAP ann mean    max |MAP - ridge|    E[MAP] at that truth    sd(MAP)
#           1.0e-04              14781.3492                 0.7059         1.7057%             0.00e+00                 1.7047%    1.1451%
#           2.0e-04               3695.3373                 0.3750         3.6248%             0.00e+00                 3.6226%    2.4335%
#           5.0e-04                591.2540                 0.0876         5.2919%             0.00e+00                 5.2888%    3.5527%
#           1.0e-03                147.8135                 0.0234         5.6640%             2.71e-20                 5.6607%    3.8025%
#           1.0e-02                  1.4781                 0.0002         5.7986%             2.71e-20                 5.7952%    3.8928%
```

The fifth column is the identity and it reads machine zero at every row. The Bayesian calculation and the penalized least-squares calculation are the same arithmetic, so a desk that has never used the word "prior" and a desk that uses nothing else are running identical code whenever the prior is Gaussian.

The first three columns say what the translation buys. The lesson's skeptical prior, $\tau=1$ basis point, is a ridge penalty of $\lambda=14781.3$; its agnostic prior is a penalty of $1.4781$; and priors a practitioner would describe as similarly reasonable span four orders of magnitude in the units a regularization routine actually consumes. The shrinkage weight runs $0.7059$, $0.3750$, $0.0876$, $0.0234$, $0.0002$, and the MAP estimates it produces — $1.7057\%$ and $5.7986\%$ at the two ends — reproduce the lesson's published $+1.7\%$ and $+5.8\%$ to the decimal. **All of it is fixed by $n$, $\sigma^{2}$ and $\tau$ before any data is loaded, so the estimate's dependence on the data is a factor of $1-w$ decided in advance.**

The last two columns are the trade in frequentist terms, at a truth equal to the flat-prior answer. The skeptical prior returns $1.7047\%$ on average against a truth of $5.8\%$ — a bias of more than four percentage points — while cutting the estimator's standard deviation from $3.8928\%$ to $1.1451\%$, a factor of $3.4$. That is a favourable trade if the truth is near zero and a ruinous one if it is not, which is the risk function [Bayesian Estimation](05-bayesian-estimation.md) traced; what this page adds is that the same trade is being made, unnamed and unexamined, every time a penalty strength is set.

## A Gaussian Prior Is Ridge and a Laplace Prior Is Lasso, With the Penalty Weight Fixed by a Ratio of Variances

The Laplace case is where the mode stops being a convenience and starts changing the answer, because the kink at the origin produces exact zeros and the posterior it came from produces none.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(11067)
K, real, se, reps = 50, 5, 0.20, 4_000                         # 45 dead variants, 5 with an edge
th = np.zeros((reps, K))
th[:, :real] = 0.10
y = th + se * rng.standard_normal((reps, K))

def post_mean(y, b):                                           # Laplace prior, closed form
    mp, mm = y - se ** 2 / b, y + se ** 2 / b
    lp, lm = -y / b, y / b
    z = np.maximum(lp, lm)
    wp, wm = np.exp(lp - z), np.exp(lm - z)
    ap, am = norm.cdf(mp / se), norm.cdf(-mm / se)
    num = wp * (mp * ap + se * norm.pdf(mp / se)) + wm * (mm * am - se * norm.pdf(mm / se))
    return num / (wp * ap + wm * am)

print("  rule                       b    total MSE    zeroed of 50    kept of 5    kept of 45"
      "    MSE on the 5 real")
def report(name, b, e):
    z = np.abs(e) < 1e-12
    print(f"  {name:<20} {b:7} {((e - th) ** 2).mean():12.5f} {z.mean() * K:15.2f}"
          f" {(~z[:, :real]).mean() * real:12.3f} {(~z[:, real:]).mean() * (K - real):13.3f}"
          f" {((e[:, :real] - 0.10) ** 2).mean():21.5f}")
report("no shrink", "-", y)
for b in (0.10, 0.20, 0.40):
    report("lasso MAP", f"{b:.2f}", np.sign(y) * np.maximum(np.abs(y) - se ** 2 / b, 0.0))
    report("posterior mean", f"{b:.2f}", post_mean(y, b))
# =>   rule                       b    total MSE    zeroed of 50    kept of 5    kept of 45    MSE on the 5 real
#      no shrink            -            0.04009            0.00        5.000        45.000               0.04032
#      lasso MAP            0.10         0.00141           47.57        0.359         2.068               0.00992
#      posterior mean       0.10         0.00424            0.00        5.000        45.000               0.00897
#      lasso MAP            0.20         0.00671           33.85        1.864        14.285               0.01264
#      posterior mean       0.20         0.01259            0.00        5.000        45.000               0.01519
#      lasso MAP            0.40         0.01719           19.00        3.287        27.717               0.02055
#      posterior mean       0.40         0.02260            0.00        5.000        45.000               0.02400
```

The zeroed column is the page's central claim and it is exact. At $b=0.10$ the mode of the posterior sets $47.57$ of $50$ coefficients to precisely zero; the *mean of the same posterior*, under the same prior, on the same data, sets $0.00$ of them to zero — and the same holds at $b=0.20$ and $b=0.40$. Nothing about the prior changed between those row pairs and nothing about the belief changed; only which feature of the resulting distribution was reported. **Sparsity is a property of the mode, not of the prior, and it disappears the moment you integrate rather than maximize.**

The total-MSE column shows the mode winning, and the reason is worth naming rather than celebrating. Lasso-MAP scores $0.00141$ against the posterior mean's $0.00424$ and the unshrunk $0.04009$, but it achieves that by zeroing ninety-five percent of the coefficients in a problem where ninety percent of the truth is genuinely zero. The mode is not detecting sparsity; it is imposing it, and the imposition happens to be right here.

The last two columns price what the imposition costs where it is wrong. At $b=0.10$ the rule keeps only $0.359$ of the five variants that genuinely have an edge — it deletes ninety-three percent of the signal along with the noise — and the posterior mean under the identical prior does better on those five, $0.00897$ against $0.00992$. At weaker priors the ordering on the real five reverses and the mode wins there too, so the honest summary is narrow: **the mode's aggregate advantage comes from a hard decision the posterior never made, and the fraction of true effects it deletes is invisible in every column a practitioner looks at.** Charging correctly for the size of the search that produced fifty candidates is a different repair and is [Bonferroni Correction](../part-15-multiple-testing/02-bonferroni-correction.md) and the rest of [Part XV](../part-15-multiple-testing/index.md); the lesson's claim that a skeptical prior does the same job "without the ceremony" is right about the shrinkage and silent about the deletions.

!!! warning "A regularization strength chosen by cross-validation is a prior variance chosen by looking at the data, and it makes every standard error computed afterwards too small"
    Two distinct problems ride together here. The first is that tuning $\lambda$ on the same data used to fit means the reported estimate is the winner of a search, so its standard error understates the dispersion by the amount [Part XV](../part-15-multiple-testing/index.md) exists to charge for. The second is subtler and is this page's: because $\lambda$ was chosen numerically, the prior it corresponds to was never inspected, and priors that sound alike differ by four orders of magnitude in $\lambda$. The free diagnostic reverses the translation: **convert the penalty back into its prior, $\tau=\sigma/\sqrt\lambda$, and say the implied prior standard deviation out loud in the units of the effect — and if it is smaller than the effect you claim to be detecting, the penalty is deciding the answer and the data is a formality.** For the table above, $\lambda=14781.3$ implies $\tau=1$ basis point a day, or about $2.5\%$ a year, against an edge the strategy is claimed to earn of $5.8\%$: a prior that considers the claimed effect a two-sigma event, applied to the data meant to test it. That is a defensible position and it should be a stated one.

## The Mode Moves When the Coordinates Do and Only the Median Does Not

Maximum likelihood survives any reparameterization, as [Maximum Likelihood Estimation](03-maximum-likelihood-estimation.md) proved with no Jacobian appearing anywhere. Adding a prior destroys that, and the reason is exactly the term the earlier proof did not need.

??? note "Proof that the posterior mode is not invariant under reparameterization while the posterior median is, and that the culprit is a Jacobian the mode cannot ignore"
    A posterior is a density on the parameter, so under a smooth one-to-one $\eta=g(\theta)$ it transforms by [Change of Variables](../part-03-random-variables/09-change-of-variables.md):

    $$p_H(\eta\mid x)=p_\Theta\big(g^{-1}(\eta)\mid x\big)\cdot\Big|\frac{dg^{-1}}{d\eta}\Big|.$$

    The argmax of a product is not the image of the argmax of the first factor unless the second factor is constant, so $\arg\max_\eta p_H\ne g(\arg\max_\theta p_\Theta)$ in general. A likelihood carries no such factor, because it is a function of $\theta$ rather than a density in $\theta$ — which is precisely the distinction [Maximum Likelihood Estimation](03-maximum-likelihood-estimation.md) draws in its own note.

    Quantiles are immune. For increasing $g$, $\mathbf{P}(\eta\le g(m)\mid x)=\mathbf{P}(\theta\le m\mid x)$, so the median in the new coordinate is the image of the median in the old one, exactly, with no correction at all.

    The concrete case is a volatility. With $x_i\sim\mathcal{N}(0,\sigma^{2})$, $S=\sum x_i^{2}$, and a flat prior on $\sigma$, the posterior is $p(\sigma\mid x)\propto\sigma^{-n}e^{-S/2\sigma^{2}}$. Its mode in $\sigma$ is at $\sqrt{S/n}$. Rewriting the same posterior as a density in $u=\sigma^{2}$ multiplies by $|d\sigma/du|=1/(2\sqrt u)$ and moves the mode to $u=S/(n+1)$; rewriting it in $\log\sigma$ multiplies by $\sigma$ and moves it to $\sigma^{2}=S/(n-1)$. Three reports of one posterior:

    $$\sqrt{\frac{S}{n+1}}\ <\ \sqrt{\frac{S}{n}}\ <\ \sqrt{\frac{S}{n-1}}.$$

    The load-bearing quantity is the Jacobian, and it is exactly the term maximum likelihood is entitled to ignore because it is identical for every candidate model on fixed data. **Maximum likelihood is invariant because the Jacobian is a constant across the models being compared, and MAP is not because the Jacobian is a function of the parameter being maximized over, and the entire difference between the two is one prior.**

```python
import numpy as np

rng = np.random.default_rng(11063)
ann, draws, sets = np.sqrt(252), 40_001, 400
sd = 0.195 / ann                                               # one 19.5% asset, mean known zero

print("  one posterior from one flat prior on sigma, its mode read in three coordinates")
print("     n    mode in sigma    mode in sigma^2    mode in log sigma    posterior median"
      "    posterior mean    median minus sqrt(median of sigma^2)")
for n in (21, 63, 252, 1260):
    a = np.zeros((6, sets))
    for j in range(sets):
        S = ((sd * rng.standard_normal(n)) ** 2).sum()
        u = S / rng.chisquare(n - 1, draws)                    # p(u) from a flat prior on sigma
        v = 100 * ann * np.sqrt(u)
        a[:, j] = [100 * ann * np.sqrt(S / n), 100 * ann * np.sqrt(S / (n + 1)),
                   100 * ann * np.sqrt(S / (n - 1)), np.median(v), v.mean(),
                   np.median(v) - 100 * ann * np.sqrt(np.median(u))]
    m = a.mean(axis=1)
    print(f"  {n:4d} {m[0]:16.3f} {m[1]:18.3f} {m[2]:20.3f} {m[3]:19.3f} {m[4]:17.3f}"
          f" {np.abs(a[5]).max():39.1e}")
# =>   one posterior from one flat prior on sigma, its mode read in three coordinates
#         n    mode in sigma    mode in sigma^2    mode in log sigma    posterior median    posterior mean    median minus sqrt(median of sigma^2)
#        21           19.271             18.828               19.747              20.083            20.529                                 0.0e+00
#        63           19.484             19.331               19.640              19.746            19.881                                 0.0e+00
#       252           19.518             19.480               19.557              19.584            19.616                                 0.0e+00
#      1260           19.480             19.472               19.487              19.493            19.499                                 0.0e+00
```

Columns two through four are one posterior read three ways. On twenty-one days of a $19.5\%$ asset the maximum a posteriori volatility is $19.271\%$ if the mode is taken in $\sigma$, $18.828\%$ if it is taken in the variance and square-rooted, and $19.747\%$ if it is taken in the log-volatility and exponentiated. The data is identical, the prior is identical, the posterior is identical, and the answer moves by $0.919$ volatility points depending on which coordinate somebody happened to code the optimizer in. At a $10\%$ volatility target that is a five-percent difference in position size, produced by a decision nobody recorded.

The fifth and sixth columns are the alternatives. The posterior median is $20.083\%$ and the posterior mean is $20.529\%$, both above all three modes because the posterior for a volatility is right-skewed and a mode sits below a median sits below a mean. Neither is more correct in the abstract; they are the answers to the absolute-error and squared-error questions of [Bayesian Estimation](05-bayesian-estimation.md), and the modes are the answers to no question at all.

The last column is the contrast that makes the point exact rather than rhetorical: the median computed in volatility units minus the square root of the median computed in variance units is $0.0$, identically, at every sample size. **A quantile is a statement about probability mass and mass does not care what the axis is called, while a mode is a statement about density height and density height is measured per unit of whatever axis you chose.** The spread across coordinates falls from $0.919$ points at $n=21$ to $0.015$ at $n=1260$, so this is an $O(1/n)$ effect — negligible for a long history, and exactly the size of the estimate for the short ones where regularization is actually reached for.

## As the Prior Flattens MAP Becomes Maximum Likelihood, Which Is Both the Reassurance and the Warning

The obvious escape from the previous section is to use a prior so flat that it cannot matter. The escape works, and closes a circle rather than opening one.

??? note "Proof that MAP converges to the maximum likelihood estimate at rate $\sigma^{2}/(n\tau^{2})$ as the prior flattens, and that the flat limit is not a probability distribution"
    In the conjugate normal case the weight on the prior is $w=(\sigma^{2}/n)/(\tau^{2}+\sigma^{2}/n)$, so

    $$\hat\theta_{\mathrm{MAP}}-\bar x=-w\,(\bar x-\mu_0),\qquad w\approx\frac{\sigma^{2}}{n\tau^{2}}\ \ \text{for large }\tau,$$

    and the estimate approaches the maximum likelihood estimate at rate $\sigma^{2}/(n\tau^{2})$ — in $\tau$ at fixed $n$, and equally in $n$ at fixed $\tau$. Any prior with a bounded continuous density gives the same limit, since the log prior contributes $O(1)$ to an objective whose likelihood term grows like $n$.

    The limit object is not a distribution. A density flat on the whole real line integrates to infinity, so "no prior" is an **improper** prior, and improper priors need not produce proper posteriors. Worse for this page, flatness is coordinate-dependent in exactly the way the fourth section describes: a prior flat in $\sigma$ is not flat in $\sigma^{2}$, and the three MAP estimates in the table above are the three answers "I used a flat prior" can mean.

    The load-bearing word is *improper*. **There is no such thing as no prior; there is only a prior that is flat in a coordinate somebody chose, and MAP reads that choice back to you as an estimate** — which is why the flat-prior limit is a reassurance about the influence of beliefs and a warning about the influence of parameterizations, and the same fact is both.

The circle this closes is worth stating explicitly. Maximum likelihood was invariant and had no prior; MAP has a prior and is not invariant; and flattening the prior recovers the invariance by removing the thing that broke it. The trade is exact, and it says that coordinate-dependence is not a defect in the mode's implementation but the price of admitting information about the parameter at all — because information about a parameter is a density on the parameter, and a density is a thing per unit of axis.

Two practical consequences follow. First, on long samples the whole issue evaporates: at $n=1260$ the three modes above differ by $0.015$ volatility points, well inside any reasonable tolerance, so a desk fitting on twenty years of data may use whichever coordinate is convenient. Second, on short samples it does not, and short samples are precisely where a prior gets reached for — a new strategy, a new regime, a thinly traded name. **The estimator is coordinate-dependent exactly in the regime it was introduced to handle**, and the repair, if the ambiguity matters, is not to argue about coordinates but to report the median instead, which costs one integral and settles the question permanently.

## The Estimate That Costs No Integral and Charges for It in Coordinates

This page established that the posterior mode is the one Bayesian summary requiring no normalizing constant, which is why it is the one practitioners compute; that maximizing a posterior is minimizing a penalized negative log-likelihood with the log prior as the penalty, so a Gaussian prior is ridge at $\lambda=\sigma^{2}/\tau^{2}$ and a Laplace prior is lasso, with the lesson's skeptical `N(0, 1bp)` corresponding to $\lambda=14781.3$ and a shrinkage weight of $0.7059$ fixed before any data arrives; that the mode of a Laplace posterior zeroes $47.57$ of $50$ coefficients while the mean of the identical posterior zeroes none, so sparsity belongs to the summary and not the belief; that one posterior read in three coordinates gives $18.828\%$, $19.271\%$ and $19.747\%$ for the same volatility while the median is identical in all three to machine precision; and that flattening the prior recovers invariance by deleting the term that destroyed it.

The pattern across the two Bayesian pages is a single trade made twice. The posterior contains everything, and every way of extracting a number from it gives something up. The mean and the median give up computability and buy a loss-function justification and coordinate independence. The mode gives up both and buys an optimizer call. Which is the right trade depends on the sample size, the geometry of the model, and whether anyone will ever ask why the answer changed when the code was refactored from variance units to volatility units — and the last of those is not a statistical question until it is.

What both pages have in common with the three before them is that they end with one number and no statement of how far it could be wrong. Every estimator in this part has produced a point and, at best, a standard error attached to it by an asymptotic argument. Turning that into a statement with a stated failure rate — an interval, and the exact sense in which it is or is not a probability about the parameter — is [Confidence Intervals](07-confidence-intervals.md), where the credible interval this page's posterior would produce and the confidence interval a frequentist would produce turn out to answer different questions and, on realistic data, to disagree about which one is doing better.

**The mode is where the density is highest and the estimate is where the loss is lowest, and no theorem says those are the same place.**
