# Maximum Likelihood Estimation

Maximum likelihood is one move. Write down the density the data supposedly came from, read it as a function of the parameter instead of the observation, take logs, differentiate, set to zero. Everything the method is famous for follows from that move and so does everything it is dangerous for. It attains the information bound in the limit because the score is the direction of steepest ascent in the likelihood; it hands over a standard error for free because the curvature at the peak is the information; it survives any change of units because a maximum is a location and locations transform; and it converges tightly, reproducibly and with a shrinking standard error on the wrong answer whenever the density it maximized is not the density the data came from — because nothing in the procedure ever consults the data about that.

This page covers the likelihood as a density read backwards, the score equations and the observed information that comes with them, the invariance of the argmax and the non-invariance of every likelihood level, the estimator's asymptotic efficiency alongside its finite-sample bias, and the Kullback–Leibler projection a misspecified fit converges to. It proves no central limit theorem, which is [The Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md); it defines sufficiency nowhere, which is [Statistics and Sufficiency](../part-10-statistics-foundations/05-statistics-and-sufficiency.md); it establishes no efficiency bound in general, which is [Properties of Estimators](02-properties-of-estimators.md); it matches no moments, which is [Method of Moments](04-method-of-moments.md); it multiplies the likelihood by no prior, which is [Maximum A Posteriori Estimation](06-maximum-a-posteriori-estimation.md); it compares no two likelihoods as a test, which is [Likelihood Ratio Tests](../part-12-hypothesis-testing/06-likelihood-ratio-tests.md); it penalizes no likelihood for parameter count, which is [Information Criteria](../part-14-model-selection/03-information-criteria.md); it runs no optimizer and implements no expectation-maximization, which are [Numerical Optimization](../part-17-statistical-computing/01-numerical-optimization.md) and [The EM Algorithm](../part-17-statistical-computing/03-em-algorithm.md); and it never checks whether the density it maximized is the one the data came from.

The trading stake is a single fitted parameter that reorganizes an entire risk framework. [Returns and Their Distributions](../../part-03-statistics/02-returns-and-distributions.md) fits four families to twenty-five years of daily SPY returns and reports that "the fitted t degrees of freedom is **2.65**: the data places daily SPY in a regime where kurtosis is infinite and even the variance is only barely finite. That single parameter is the stylized facts compressed into one number." The published log-likelihoods are $19095.0$ for the normal against $20034.3$ for the Student-$t$ — a gap of $939.3$ points — with AIC moving from $-38185.9$ to $-40062.7$. The fourth section prices the precision of that $2.65$, and finds the published gap sits below the fifth percentile of what a genuine $t(2.65)$ sample produces.

## The Likelihood Is the Density Read Backwards and the Log Is What Makes It Arithmetic

A density $f(x;\theta)$ is normally read forward: fix $\theta$, vary $x$, and integrate to one. The **likelihood** is the same function read backward — fix the observed $x$, vary $\theta$ — and it is not a density in $\theta$, does not integrate to one, and has units that depend on the units of $x$. For independent observations it multiplies,

$$L(\theta)=\prod_{i=1}^{n}f(x_i;\theta),$$

and the product is the reason nobody ever computes it. With $n=6{,}410$ daily returns each of density order $40$, the product overflows a double before the two-hundredth observation. Taking logs converts it to a sum,

$$\ell(\theta)=\sum_{i=1}^{n}\log f(x_i;\theta),$$

which is numerically stable, differentiable term by term, and additive across independent blocks of data — the three properties every fitting routine in this appendix relies on. The logarithm is not a convenience here but the only representation in which the object exists at scale, which is the same argument [Exponentials, Logarithms, and Growth](../part-01-mathematical-foundations/07-exponentials-logarithms-growth.md) makes when it routes the forward algorithm's rescaling through this page.

The **maximum likelihood estimator** is $\hat\theta=\arg\max_\theta \ell(\theta)$. The distinction [Mathematical Notation](../part-01-mathematical-foundations/03-mathematical-notation.md) insists on is load-bearing: "$\max_\theta \ell(\theta)$ is the peak height of the log-likelihood, a number nobody cares about, while $\hat\theta = \arg\max_\theta \ell(\theta)$ is the parameter estimate, which is the entire output." The third section shows that the two objects behave differently under the most routine operation in data work, and that the one nobody cares about is the one every model-comparison table is built from.

!!! note "The likelihood is a function of the parameter and not a distribution over it, which is why it does not integrate to one and why the phrase 'the most likely parameter' has no frequentist referent"
    [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md) states the asymmetry exactly: a conditional density "is a probability over $B$ for each fixed $A$, so it sums to one across the possible data. Read the other way — as a function of $A$ for fixed observed $B$ — it is a **likelihood**, and it does not sum to one across hypotheses." Three consequences follow and all three are routinely violated in writing. The likelihood's absolute value is meaningless, since it carries the units of $f$ and can be rescaled by changing the units of $x$; only ratios at fixed data are interpretable. There is no such thing as the probability that $\theta$ lies in an interval under a likelihood, because there is no measure on $\theta$ to integrate — an object that *is* a distribution over $\theta$ requires a prior and is [Bayesian Estimation](05-bayesian-estimation.md). And $\hat\theta$ is not "the most probable parameter" but the parameter under which the observed data was least surprising, which is a statement about the data with $\theta$ as an index. The same denominator that Bayes' rule divides by to restore a distribution is the constant maximum likelihood discards, which is why the two methods can agree numerically and disagree completely about what the number means.

## The Score Equations Are the First-Order Condition and the Hessian Is the Standard Error

Maximizing a smooth function means setting its gradient to zero. The gradient of the log-likelihood is the **score** $S(\theta)=\nabla_\theta\ell(\theta)$, and the **score equations** $S(\hat\theta)=0$ are the first-order condition. [Calculus Essentials](../part-01-mathematical-foundations/06-calculus-essentials.md) routes here for exactly this: "Applying it to a log-likelihood gives the score equations of Maximum Likelihood Estimation, and the Hessian of the log-likelihood is (minus) the observed information, which is where the estimator's standard errors come from."

For a normal with known variance the score equations return the sample mean; for the normal's variance they return the sample variance with divisor $n$, not $n-1$, which is where the maximum likelihood estimator's finite-sample bias enters and why [Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md) has a correction to discuss at all. For a Student-$t$ there is no closed form, and the equations are solved numerically. The important structural point is that the same derivative that locates the maximum also measures how sharp it is, and sharpness is precision.

??? note "Proof that the inverse observed information is the estimator's variance, and that the identity behind it holds only when the model is correct"
    Expand the score about the truth $\theta_0$ and evaluate at the root $\hat\theta$:

    $$0=S(\hat\theta)\approx S(\theta_0)+\ell''(\theta_0)\,(\hat\theta-\theta_0),\qquad\text{so}\qquad \hat\theta-\theta_0\approx\frac{-S(\theta_0)}{\ell''(\theta_0)}.$$

    The numerator is a sum of $n$ independent mean-zero terms, so the central limit theorem gives $S(\theta_0)\approx\mathcal{N}(0,nI_1)$ where $I_1=\mathbb{E}[s_i^{2}]$ is the information in one observation; the denominator concentrates at $-nJ_1$ with $J_1=-\mathbb{E}[\partial_\theta^{2}\log f]$ by the law of large numbers. Hence

    $$\sqrt n(\hat\theta-\theta_0)\Rightarrow\mathcal{N}\big(0,\ J_1^{-1}I_1J_1^{-1}\big).$$

    Now the **information identity**. Differentiating $\int f\,dx=1$ twice under the integral gives $\mathbb{E}[\partial_\theta^{2}\log f]+\mathbb{E}[(\partial_\theta\log f)^{2}]=0$, that is $J_1=I_1$, whereupon the sandwich collapses to $1/I_1$ and the estimator attains the Cramér–Rao bound of [Properties of Estimators](02-properties-of-estimators.md) asymptotically. The practical form is that the **observed information** $J(\hat\theta)=-\ell''(\hat\theta)$, a quantity the optimizer already computed, delivers

    $$\hat{\mathrm{se}}(\hat\theta)=1/\sqrt{J(\hat\theta)}$$

    with no extra work and no simulation.

    The identity is where the model enters. Both expectations are taken under $f_\theta$, so if the data came from some other law $F$ then $J_1\ne I_1$, the sandwich does not collapse, and the correct variance is $J^{-1}KJ^{-1}$ with $K$ the empirical variance of the scores. The free standard error is the special case of that sandwich in which the two matrices were assumed equal.

    The load-bearing step is the information identity, and it is an assumption about the world dressed as an algebraic identity. **The Hessian hands you a standard error for free and it is the right one only if the density is right; when it is not, the free number is too small, and the repair is a second matrix nobody computes.**

## The Argmax Is Invariant Under Reparameterization and Every Likelihood Level Is Not

If $\hat\theta$ maximizes $\ell$ and $g$ is any one-to-one map, then $g(\hat\theta)$ maximizes the likelihood written in the new coordinate, because $\ell^{\ast}(\eta)=\ell(g^{-1}(\eta))$ takes the same set of values in the same order. Fit a variance and take the square root, or fit a log-volatility and exponentiate — the answer is identical, and no Jacobian appears anywhere. This is a property of maxima, not of likelihoods: it holds because a maximum is a *location*, and it is exactly the property [Maximum A Posteriori Estimation](06-maximum-a-posteriori-estimation.md) loses the moment a prior density is multiplied in.

??? note "Proof that the maximum likelihood estimate is invariant under any reparameterization, and that no comparison of likelihood levels across differently scaled data is"
    For a one-to-one $g$ with inverse $h$, the log-likelihood in the new parameter is $\ell^{\ast}(\eta)=\ell(h(\eta))$. If $\hat\theta$ maximizes $\ell$ then for every $\eta$ we have $\ell^{\ast}(\eta)=\ell(h(\eta))\le\ell(\hat\theta)=\ell^{\ast}(g(\hat\theta))$, so $g(\hat\theta)$ maximizes $\ell^{\ast}$. No derivative and no Jacobian is used, because the parameter is not being integrated over — which is precisely the difference from a density, where [Change of Variables](../part-03-random-variables/09-change-of-variables.md) shows the Jacobian is unavoidable.

    Transforming the *data* is a different matter. Let $y_i=h(x_i)$ for a smooth increasing $h$. The density of $y$ under any model is $f_Y(y;\theta)=f_X(h^{-1}(y);\theta)\,|(h^{-1})'(y)|$, so

    $$\ell_Y(\theta)=\ell_X(\theta)-\sum_{i=1}^{n}\log|h'(x_i)|.$$

    The subtracted term does not involve $\theta$ and does not involve the model, so it is the same constant for every candidate family. Two things follow, and they point opposite ways. The argmax is untouched, and so is every *difference* of log-likelihoods or of information criteria computed on one dataset — model comparison is safe. But the level is shifted by a quantity that can be made anything at all by choosing units, so a log-likelihood, an AIC or a BIC quoted as an absolute number is a statement about the units of the data and nothing else.

    The load-bearing word is *argmax*. **Invariance is a property of where the maximum is, not of how high it is, which is why maximum likelihood survives a change of units and why every criterion built on likelihood levels is comparable only within one fixed representation of one fixed dataset.**

```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm, norminvgauss, t as tdist

rng = np.random.default_rng(11033)
n, nu0 = 6_410, 2.65
r = (0.195 / np.sqrt(252)) * tdist.rvs(nu0, size=n, random_state=rng) / np.sqrt(nu0 / (nu0 - 2))
tol = {"xatol": 1e-13}

def fit_t(x):
    def neg(lv):
        v, m, s2 = np.exp(lv), x.mean(), x.var()
        for _ in range(60):
            w = (v + 1) / (v + (x - m) ** 2 / s2)
            m = (w * x).sum() / w.sum()
            s2 = (w * (x - m) ** 2).mean()
        return -tdist.logpdf(x, v, m, np.sqrt(s2)).sum()
    return -minimize_scalar(neg, bounds=(np.log(1.05), np.log(60.0)), method="bounded",
                            options=tol).fun

def scale_mle(x, link):
    f, lo, hi = {"sigma": (lambda u: u, 1e-6, 1.0),
                 "sigma^2": (np.sqrt, 1e-12, 1.0),
                 "log sigma": (np.exp, -14.0, 0.0)}[link]
    g = minimize_scalar(lambda u: -norm.logpdf(x, x.mean(), f(u)).sum(),
                        bounds=(lo, hi), method="bounded", options=tol)
    return f(g.x)

print("  one normal fit, three coordinates for the scale:  "
      + "  ".join(f"{k} -> {scale_mle(r, k):.10f}" for k in ("sigma", "sigma^2", "log sigma")))
print(f"  n log 100 = {n * np.log(100):.1f}, the Jacobian every model pays for rescaling the data")
print("  model      k    loglik(r)    loglik(100r)    shift    AIC gap vs normal on r    on 100r")
ref = None
for name, k in (("normal", 2), ("student-t", 3), ("NIG", 4)):
    ll = []
    for x in (r, 100 * r):
        if name == "normal":
            ll.append(norm.logpdf(x, x.mean(), x.std()).sum())
        elif name == "student-t":
            ll.append(fit_t(x))
        else:
            ll.append(norminvgauss.logpdf(x, *norminvgauss.fit(x)).sum())
    aic = np.array([2 * k - 2 * v for v in ll])
    ref = aic if ref is None else ref
    print(f"  {name:<9} {k:4d} {ll[0]:12.1f} {ll[1]:15.1f} {ll[0] - ll[1]:8.1f}"
          f" {ref[0] - aic[0]:24.1f} {ref[1] - aic[1]:10.1f}")
# =>   one normal fit, three coordinates for the scale:  sigma -> 0.0106894805  sigma^2 -> 0.0106894804  log sigma -> 0.0106894807
#      n log 100 = 29519.1, the Jacobian every model pays for rescaling the data
#      model      k    loglik(r)    loglik(100r)    shift    AIC gap vs normal on r    on 100r
#      normal       2      19996.4         -9522.8  29519.1                      0.0        0.0
#      student-t    3      21032.5         -8486.6  29519.1                   2070.4     2070.4
#      NIG          4      21032.3         -8486.8  29519.1                   2067.9     2067.9
```

The first line is invariance in practice. The same normal fitted by numerically maximizing over the scale, over the variance, and over the log-scale returns $0.0106894805$, $0.0106894804$ and $0.0106894807$, agreeing to nine decimal places with the disagreement sitting at the optimizer's tolerance rather than in the mathematics. **A modelling choice that changes nothing is worth confirming changes nothing**, because the next section's estimator does not have this property and the difference between them is one prior.

The middle columns are the trap. Every log-likelihood falls by exactly $29519.1$ when the returns are quoted in percent instead of decimals, and $29519.1$ is $n\log 100$ to the digit. The normal goes from $+19996.4$ to $-9522.8$, changing sign; the Student-$t$ from $+21032.5$ to $-8486.6$. Nothing about the data or the fit changed — only the units did.

The last two columns are what survives. The AIC gap against the normal is $2070.4$ on the decimal scale and $2070.4$ on the percent scale, and $2067.9$ against $2067.9$ for the four-parameter family; the shift cancels exactly in every difference because it does not depend on the model. So the course's comparison of $-38185.9$ against $-40062.7$ is meaningful and would be meaningful in any units, while either number quoted alone is not. **Model selection reads differences and is safe; the practice of recording an absolute AIC in a research log and comparing it to next quarter's is a comparison of unit conventions.**

## The Estimator Attains the Information Bound in the Limit and Is Biased at Every Finite Sample Size

The proof in the second section delivers the headline guarantee: under regularity and a correct model, $\hat\theta$ is consistent, asymptotically normal, and asymptotically efficient, meeting the Cramér–Rao bound that [Properties of Estimators](02-properties-of-estimators.md) derived. What none of those adjectives covers is any finite $n$. Maximum likelihood is biased in general — the normal's variance MLE is the clearest case, being low by a factor $(n-1)/n$ — and for a parameter entering the density nonlinearly the bias has no closed form and the sampling distribution is not symmetric. The course's $2.65$ is such a parameter.

```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm, t as tdist

rng = np.random.default_rng(11031)
nu0, vol = 2.65, 0.195 / np.sqrt(252)                          # the course's fitted SPY tail

def fit_t(x):
    def neg(lv):
        v, m, s2 = np.exp(lv), x.mean(), x.var()
        for _ in range(60):
            w = (v + 1) / (v + (x - m) ** 2 / s2)
            m = (w * x).sum() / w.sum()
            s2 = (w * (x - m) ** 2).mean()
        return -tdist.logpdf(x, v, m, np.sqrt(s2)).sum()
    r = minimize_scalar(neg, bounds=(np.log(1.05), np.log(60.0)), method="bounded")
    return np.exp(r.x), -r.fun

print("       n    reps    median nu-hat    sd    5th pct    95th pct"
      "    loglik(t) - loglik(normal): 5th    median    95th")
for n, reps in ((252, 400), (1260, 400), (6410, 200)):
    v, gap = np.empty(reps), np.empty(reps)
    for i in range(reps):
        x = vol * tdist.rvs(nu0, size=n, random_state=rng) / np.sqrt(nu0 / (nu0 - 2))
        v[i], ll = fit_t(x)
        gap[i] = ll - norm.logpdf(x, x.mean(), x.std()).sum()
    q = np.quantile(gap, [0.05, 0.5, 0.95])
    print(f"  {n:6d} {reps:7d} {np.median(v):16.3f} {v.std():5.3f} {np.quantile(v, 0.05):10.3f}"
          f" {np.quantile(v, 0.95):11.3f} {q[0]:33.1f} {q[1]:9.1f} {q[2]:9.1f}")
# =>        n    reps    median nu-hat    sd    5th pct    95th pct    loglik(t) - loglik(normal): 5th    median    95th
#         252     400            2.731 0.605      2.046       3.915                              13.6      40.9     121.0
#        1260     400            2.677 0.234      2.334       3.100                             152.0     260.5     574.9
#        6410     200            2.662 0.107      2.508       2.852                            1116.6    1570.4    2952.5
```

At the course's sample size the estimator is very good. The median $\hat\nu$ is $2.662$ against a truth of $2.65$, with a standard deviation of $0.107$ and a ninety-percent interval of $[2.508,\ 2.852]$ — narrow enough that the qualitative claim the lesson builds on it, that the fourth moment does not exist, is settled by the data rather than by the estimator. The finite-sample bias is upward and small, $0.012$ against a standard deviation eight times larger.

At one year of data the same estimator is nearly useless for that purpose. The median is $2.731$, the standard deviation is $0.605$ — five and a half times wider — and the ninety-percent interval runs $[2.046,\ 3.915]$, which straddles the $\nu=4$ boundary where the kurtosis becomes finite. **A year of daily returns cannot distinguish a market with infinite kurtosis from one with finite kurtosis, and the estimate it returns will look like a number rather than like a coin flip.** The bias is upward at every $n$ and shrinks as $0.081$, $0.027$, $0.012$, so the estimator errs on the side of thinner tails when data is short — the wrong direction for a risk system.

The last three columns are where the simulation disagrees with the course, informatively. On genuine $t(2.65)$ data of the course's own length the Student-$t$ beats the normal by a median of $1570.4$ log-likelihood points, with a fifth percentile of $1116.6$. The published gap on the real series is $939.3$, which sits *below* the fifth percentile of what an exact $t(2.65)$ sample delivers. The fit is not wrong — the same likelihood picked $2.65$ in both cases — but the real returns give the Student-$t$ a smaller advantage than a real Student-$t$ would, which is the signature of a tail that is partly produced by volatility clustering rather than by a static heavy-tailed law. **The parameter is estimated precisely and the family is still not the family**, which is the subject of the next section.

## A Misspecified Fit Converges Tightly on the Kullback–Leibler Projection and Reports Its Tightness as Precision

Nothing in the score equations asks whether $f_\theta$ contains the truth. If it does not, the estimator still converges — to a specific, identifiable, wrong answer.

??? note "Proof that a misspecified maximum likelihood estimator converges to the Kullback–Leibler projection of the truth onto the model, so it is consistent for something and that something is not the parameter"
    Let the data come from $F$ with density $f^{\ast}$, and let the fitted family be $\{f_\theta\}$ with $f^{\ast}\notin\{f_\theta\}$. By the law of large numbers,

    $$\frac1n\ell(\theta)=\frac1n\sum_i\log f_\theta(x_i)\ \longrightarrow\ \mathbb{E}_{F}\big[\log f_\theta(X)\big]=-H(F)-\mathrm{KL}\big(F\,\Vert\,f_\theta\big),$$

    where $H(F)=-\mathbb{E}_F[\log f^{\ast}]$ does not involve $\theta$. Maximizing the left side is therefore minimizing $\mathrm{KL}(F\Vert f_\theta)$, and $\hat\theta\to\theta^{\ast}=\arg\min_\theta\mathrm{KL}(F\Vert f_\theta)$, the **Kullback–Leibler projection** of the truth onto the model.

    Everything downstream survives in modified form. The estimator is consistent for $\theta^{\ast}$, asymptotically normal about it, and its standard error shrinks at the usual rate — but the asymptotic variance is the sandwich $J^{-1}KJ^{-1}$ of the second section rather than $1/I$, and the target is a projection rather than a parameter. The projection depends on the whole shape of $F$, so it moves when the tail moves even if the quantity being reported does not change at all under the truth.

    The load-bearing quantity is $\mathrm{KL}(F\Vert f_\theta)$, an object with no dependence on any true parameter because there is no true parameter — the model does not contain one. **A wrong model does not fail loudly; it converges, tightly and reproducibly, on the closest wrong answer, and the standard error it prints describes the tightness of that convergence rather than the distance to the truth.**

The cleanest demonstration is the fit every risk system actually runs: a normal, maximized against returns that are not normal.

```python
import numpy as np
from scipy.stats import norm, t as tdist

rng = np.random.default_rng(11037)
reps = 20_000

print("    nu      n    Hessian se    sandwich se    true sd    Hessian/true"
      "    fitted normal 1% VaR    true 1% VaR")
for nu in (8.0, 6.0, 4.5, 3.4):
    q = tdist.ppf(0.01, nu) / np.sqrt(nu / (nu - 2))            # unit-variance t, so sigma = 1
    for n in (252, 1260, 6410):
        x = tdist.rvs(nu, size=(reps, n), random_state=rng) / np.sqrt(nu / (nu - 2))
        d = x - x.mean(axis=1, keepdims=True)
        v = (d ** 2).mean(axis=1)
        hes, san = v * np.sqrt(2 / n), np.sqrt(((d ** 4).mean(axis=1) - v ** 2) / n)
        var1 = norm.ppf(0.01, x.mean(axis=1), np.sqrt(v))
        print(f"  {nu:5.1f} {n:6d} {hes.mean():13.5f} {san.mean():14.5f} {v.std():10.5f}"
              f" {hes.mean() / v.std():15.3f} {var1.mean():23.3f} {q:14.3f}")
# =>     nu      n    Hessian se    sandwich se    true sd    Hessian/true    fitted normal 1% VaR    true 1% VaR
#        8.0    252       0.08878        0.11268    0.11723           0.757                  -2.319         -2.508
#        8.0   1260       0.03979        0.05189    0.05258           0.757                  -2.324         -2.508
#        8.0   6410       0.01766        0.02327    0.02327           0.759                  -2.326         -2.508
#        6.0    252       0.08878        0.12738    0.13979           0.635                  -2.317         -2.566
#        6.0   1260       0.03977        0.05997    0.06200           0.641                  -2.323         -2.566
#        6.0   6410       0.01766        0.02736    0.02799           0.631                  -2.326         -2.566
#        4.5    252       0.08889        0.15697    0.20278           0.438                  -2.314         -2.629
#        4.5   1260       0.03981        0.07832    0.10148           0.392                  -2.323         -2.629
#        4.5   6410       0.01766        0.03742    0.04309           0.410                  -2.326         -2.629
#        3.4    252       0.08889        0.22358    0.64634           0.138                  -2.293         -2.657
#        3.4   1260       0.03976        0.12517    0.28391           0.140                  -2.314         -2.657
#        3.4   6410       0.01765        0.06866    0.15668           0.113                  -2.322         -2.657
```

The Hessian column is the free standard error and it is the same number in every tail: $0.08878$, $0.03979$, $0.01766$ at the three sample sizes, identical down the table because $\hat\sigma^{2}\sqrt{2/n}$ knows only the variance and the variance was standardized to one. The true column is what the estimator's dispersion actually is, and it grows with the tail: $0.02327$, $0.02799$, $0.04309$, $0.15668$ at $n=6410$. **The reported precision is blind to the one feature of the data that determines the real precision.**

The ratio column prices the error. At $\nu=8$ the free standard error is $0.757$ of the truth — already a twenty-five percent understatement at a tail thickness no risk committee would flag. At $\nu=6$ it is $0.631$, at $\nu=4.5$ about $0.41$, and at $\nu=3.4$ it is $0.138$, $0.140$, $0.113$ — a seven- to ninefold understatement that *worsens as the sample grows*, because the fourth moment the true variance depends on does not exist and the empirical version keeps climbing. The sandwich repairs most of this where the fourth moment exists, matching the truth to three digits at $\nu=8$ ($0.02327$ against $0.02327$) and closely at $\nu=6$, and it fails alongside everything else at $\nu=3.4$, reporting $0.06866$ against $0.15668$.

The last two columns are the Kullback–Leibler projection with a price tag. The fitted normal's one-percent quantile converges beautifully — $-2.319$, $-2.324$, $-2.326$ — on the value $-2.326$, which is the normal quantile and is exactly what the projection predicts. The truth is $-2.508$ at $\nu=8$ and $-2.657$ at $\nu=3.4$. **The estimate converges, its reported precision improves with every added observation, and the quantity it converges to understates the loss threshold by fourteen percent forever**, which is the same failure [Sampling Distributions](../part-10-statistics-foundations/03-sampling-distributions.md) measured from the coverage side when a nominal ninety-five percent variance interval held $0.4566$ of the time.

!!! warning "A fit that converged, reported a small standard error and reproduced across restarts has established that the model is internally consistent and nothing whatever about whether it is the right model"
    Every diagnostic a fitting routine emits is computed under the fitted model. Convergence says the optimizer found the peak of the surface it was given; the standard error is a curvature of that same surface; a tight confidence interval means the surface is sharply peaked; and stability across random starts means the surface has one peak. None of these consults the data about whether the surface was the right one, and the previous table shows all four can look excellent while the standard error is understated eightfold and an implied loss threshold is wrong by a seventh. The free diagnostic is a single simulation: **draw a sample of your own $n$ from your own fitted parameters, refit it, and compare the real data's fitted log-likelihood and one shape statistic — a sample kurtosis, a count of five-sigma days — against the ensemble the simulation produces; if the real data sits outside the simulated range, the standard error you are quoting is a sandwich you have not computed.** The fourth section is that check run against the course's own table, and the published gap of $939.3$ falling below the fifth percentile of $1116.6$ is exactly the signal it exists to raise. The machinery for turning this into a formal statement is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); the version above costs four lines and is worth running before any of it.

## One Recipe, Four Guarantees, and All Four Conditional on the Density Being Right

This page established that the likelihood is a density read backward and is not a distribution over the parameter; that the score equations are the first-order condition and the observed information is a standard error the optimizer has already computed, valid only through an identity that holds when the model is correct; that the argmax survives any reparameterization while every likelihood level shifts by $n\log 100$ under a change of units, so differences are comparable and absolute values are not; that at the course's sample size $\hat\nu$ lands at $2.662\pm0.107$ while at one year it lands at $2.731\pm0.605$ and cannot tell infinite kurtosis from finite; and that a misspecified fit converges on the Kullback–Leibler projection with a standard error that can be an eighth of the truth and improving.

The four guarantees — efficiency, invariance, a free standard error, and asymptotic normality — all descend from the same move, and so does the single failure mode. Because the estimator only ever asks which member of the family fits best, it is silent by construction about whether the family was worth searching. That silence is not a gap the method could be extended to fill; it is what "maximum likelihood" means. Every check that could raise the alarm lives outside the procedure: a goodness-of-fit comparison, a sandwich variance, a simulation from the fitted model, a held-out sample.

The most useful way to hold the method is therefore as a projection rather than as an inference. It reports the closest point of the model to the data in the one specific sense of Kullback–Leibler divergence, and it reports the curvature of the model at that point. Both are correct statements about the model. Neither is a statement about the world unless the model contains it, and the estimator will never say which case it is in. A method that avoids the density entirely, buys consistency from the delta method alone, and fails in an entirely different place is [Method of Moments](04-method-of-moments.md) — the natural control for everything on this page, fitted to the same $2.65$ with the opposite outcome.

**Maximum likelihood answers the question of which member of this family would have been least surprised by this data, and it answers it just as confidently when the data came from somewhere else.**
