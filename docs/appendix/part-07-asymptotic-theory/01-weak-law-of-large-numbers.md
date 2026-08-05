# The Weak Law of Large Numbers

Average enough independent draws and the average lands near the truth. The theorem that says so is the shortest in this part — one inequality applied to one quantity, then a limit — and it is the most over-read result in quantitative finance, because it promises that the sample mean converges and says absolutely nothing about when. Every appeal to "with enough data" is an appeal to this theorem, and the theorem has no opinion about whether the data you have is enough.

This page covers the statement of the weak law, its proof as Chebyshev's inequality applied to an average, the definition of convergence in probability that the word "weak" refers to, the same theorem read as a statement about the empirical distribution rather than about a mean, and the two ways it fails — slowly, when the variance is large, and completely, when there is no mean to converge to. It does not prove Chebyshev's inequality, which is [Variance](../part-04-expectation-and-moments/02-variance.md); it does not establish the path-wise version, which is [The Strong Law of Large Numbers](02-strong-law-of-large-numbers.md); it says nothing about the *shape* of the error, which is [The Central Limit Theorem](03-central-limit-theorem.md); it does not define consistency or any other property of an estimator, which is [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md); and it develops no resampling machinery, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md).

The trading stake is the single number this course keeps returning to. [Probability and Random Variables](../../part-03-statistics/01-probability-and-random-variables.md) estimates the equity premium from twenty-five years of daily data and reports an annualized mean of $0.075$ with a standard error of $0.039$ — a t-statistic of about $1.9$, "not even two standard errors from zero" — and concludes that "vol is estimable; the mean barely is." The weak law guarantees that estimate converges. The third section computes what it has converged to after twenty-five years, and the answer is a number with a $2.8\%$ chance of having the wrong sign.

## The Proof Is Chebyshev and a Limit and Nothing Else

Let $X_1,X_2,\dots$ be independent and identically distributed with mean $\mu$ and finite variance $\sigma^2$, and write $\bar X_n=\tfrac1n\sum_{i=1}^{n}X_i$. The weak law states that for every $\epsilon>0$,

$$\mathbf{P}\!\left(\lvert \bar X_n-\mu\rvert\geq\epsilon\right)\longrightarrow 0\qquad\text{as}\qquad n\to\infty.$$

The proof is two facts already established elsewhere in this appendix, put next to each other.

??? note "Proof that the sample mean converges in probability to the population mean"
    [Variance](../part-04-expectation-and-moments/02-variance.md) proves Chebyshev's inequality — for any random variable $Y$ with mean $\nu$ and finite variance, $\mathbf{P}(\lvert Y-\nu\rvert\geq c)\leq\mathrm{var}(Y)/c^{2}$ — and, in the section on the variance of a sum, that $\mathrm{var}(\bar X_n)=\sigma^{2}/n$ when the terms are uncorrelated. Neither is re-derived here.

    Apply the first to $Y=\bar X_n$, whose mean is $\mu$ by linearity of expectation, with $c=\epsilon$:

    $$\mathbf{P}\!\left(\lvert \bar X_n-\mu\rvert\geq\epsilon\right)\leq\frac{\mathrm{var}(\bar X_n)}{\epsilon^{2}}=\frac{\sigma^{2}}{n\epsilon^{2}}.$$

    The right-hand side is a constant divided by $n$. Fix $\epsilon$, let $n$ grow, and it goes to zero. That is the entire argument, and the source page is right to call it "Chebyshev plus a limit and nothing else."

    Exactly one hypothesis is doing structural work, and it is not the one the statement advertises. Finite variance appears in the bound, but it is *sufficient and not necessary*: Khinchine's version of the theorem assumes only that the mean exists, and the conclusion is identical. Independence is likewise stronger than needed — the proof used only that the terms are pairwise **uncorrelated**, since that is all $\mathrm{var}(\bar X_n)=\sigma^2/n$ requires, and [Independence](../part-02-probability-foundations/05-independence.md) is careful that uncorrelated is the weaker condition. What is genuinely non-negotiable is that $\mu$ exist at all. Without a finite mean there is no $\mu$ for the left-hand side to reference, the statement cannot even be written down, and the fifth section shows what the sample mean does instead.

## Convergence in Probability Is a Statement About Each n, Not About a Path

The mode of convergence in the theorem has a name. A sequence of random variables $Y_n$ **converges in probability** to $Y$, written $Y_n\xrightarrow{\ p\ }Y$, when $\mathbf{P}(\lvert Y_n-Y\rvert\geq\epsilon)\to0$ for every $\epsilon>0$. Read the quantifiers carefully, because the whole of this part's structure is in them: for each fixed $n$, a probability is computed across the ensemble of possible samples, and *that number* is required to shrink.

Nothing in the definition follows any single sample as $n$ grows. It is entirely consistent with the definition that every realized sequence $\bar X_1(\omega),\bar X_2(\omega),\dots$ wanders away from $\mu$ infinitely often, provided the set of samples on which it is far away at any given $n$ is shrinking. That is not a pathological reading — it is exactly what the counterexample in [The Strong Law of Large Numbers](02-strong-law-of-large-numbers.md) exhibits, and it is why the strong law is a separate theorem rather than a restatement.

This distinction has a practical edge that survives translation out of the mathematics. A trader running one account observes one path. The weak law is a statement about the ensemble of accounts that could have existed, and licenses a claim about *this* account only in the sense that it was drawn from that ensemble. The path-wise claim, which is the one that sounds like what was meant, requires the countable additivity axiom of [Probability Axioms](../part-02-probability-foundations/02-probability-axioms.md) and is the subject of the next page.

## The Theorem Promises Everything and Schedules Nothing

The bound from the proof is not merely qualitative — it is a number, computable for any $n$, and worth computing at the sample sizes finance actually has.

```python
import numpy as np

rng = np.random.default_rng(7101)
mu, sigma, reps = 0.075, 0.195, 200_000                        # the published equity premium
print(f"  iid returns at mu = {mu}, sigma = {sigma}; {reps} independent histories each")
print("        days    years   sd(mean)   Chebyshev P(|err| >= mu)   simulated   P(mean < 0)")
for n in (252, 1_260, 6_300, 63_000, 630_000):
    se = sigma / np.sqrt(n / 252)
    m = mu + se * rng.standard_normal(reps)
    print(f"  {n:10d} {n / 252:8.1f} {se:10.4f} {min(1.0, (se / mu) ** 2):23.4f}"
          f" {np.mean(np.abs(m - mu) >= mu):11.4f} {np.mean(m < 0):13.4f}")
# =>   iid returns at mu = 0.075, sigma = 0.195; 200000 independent histories each
#            days    years   sd(mean)   Chebyshev P(|err| >= mu)   simulated   P(mean < 0)
#             252      1.0     0.1950                  1.0000      0.6995        0.3503
#            1260      5.0     0.0872                  1.0000      0.3899        0.1938
#            6300     25.0     0.0390                  0.2704      0.0551        0.0279
#           63000    250.0     0.0123                  0.0270      0.0000        0.0000
#          630000   2500.0     0.0039                  0.0027      0.0000        0.0000
```

The third row is the published estimate. Twenty-five years of daily data produces a standard error of $0.0390$ on an annualized mean of $0.075$ — the course's $0.039$, reproduced from nothing but $\sigma/\sqrt{\text{years}}$ — and the simulated probability that such an estimate lands on the wrong side of zero is $0.0279$. One history in thirty-six, over a quarter century, reports a negative equity premium.

The Chebyshev column shows the bound working exactly as advertised and exactly as loosely. At twenty-five years it certifies that the estimate misses by a full $\mu$ with probability at most $0.2704$; the truth is $0.0551$, so the bound is about five times too wide. That gap is not a defect. [Variance](../part-04-expectation-and-moments/02-variance.md) makes the point that Chebyshev is the best bound true of *all* distributions simultaneously, and the slack measures what assuming a shape would buy — which is what [The Central Limit Theorem](03-central-limit-theorem.md) sells.

The column that should end the argument is the first. To halve the standard error from $0.039$ to $0.0195$ requires four times the data: one hundred years. To reach $0.0039$, a tenth of the estimate, requires $2{,}500$ years. The weak law is true along every row of that table and useless along all of them, because convergence at rate $1/\sqrt{n}$ against a calendar that supplies $252$ observations a year is a promise denominated in a currency nobody trades in.

!!! note "The published decade means are not evidence that the equity premium changed; they are the standard error being displayed three times"
    [Probability and Random Variables](../../part-03-statistics/01-probability-and-random-variables.md) reports annualized means of $-0.009$ for the 2000s, $+0.126$ for the 2010s and $+0.134$ for the 2020s, and observes that "the 2000s mean was *negative*." At ten years per bucket the table above puts the standard error near $0.062$, so three draws spanning $0.143$ is close to what an unchanging $\mu$ would produce on its own. The honest reading is not that the premium moved; it is that a decade is not long enough to tell whether it did — and no rearrangement of the same twenty-five years can answer the question, because the standard error depends on calendar span rather than on how finely the span is sampled.

## The Bootstrap Is This Theorem Applied to the Distribution Itself

Nothing in the proof required $X_i$ to be a return. Apply it to the indicator $\mathbf 1\{X_i\leq x\}$, whose mean is $F(x)$ and whose variance is $F(x)(1-F(x))\leq\tfrac14$, and the conclusion is that the **empirical distribution function** $\hat F_n(x)=\tfrac1n\sum_i\mathbf 1\{X_i\leq x\}$ converges in probability to $F(x)$, at every $x$, with a bound that does not depend on $F$ at all.

That is the license [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) invokes when it says the plug-in principle "works at all" because "$\hat F$ converges to $F$". The uniform strengthening — that the *worst* gap over all $x$ vanishes almost surely — is stated on [Cumulative Distribution Functions](../part-03-random-variables/02-cumulative-distribution-functions.md); what the weak law supplies underneath it is the pointwise version, one $x$ at a time, from an argument about a Bernoulli average.

```python
import numpy as np

rng = np.random.default_rng(7109)
reps = 20_000
print(f"  empirical CDF against the law that generated it, {reps} samples at each n")
print("           n   mean sup|Fhat - F|   x sqrt(n)   Fhat(q05): mean        sd    sd / 0.05")
for n in (10, 100, 1_000, 10_000, 100_000):
    u = np.sort(rng.random((reps, n)), axis=1)                 # F(X) is uniform, so work there
    i = np.arange(1, n + 1)
    sup = np.maximum((i / n - u).max(axis=1), (u - (i - 1) / n).max(axis=1))
    tail = np.mean(u <= 0.05, axis=1)                          # Fhat at the 5% quantile
    print(f"  {n:10d} {sup.mean():18.4f} {sup.mean() * np.sqrt(n):11.4f}"
          f" {tail.mean():18.4f} {tail.std():9.4f} {tail.std() / 0.05:12.4f}")
# =>   empirical CDF against the law that generated it, 20000 samples at each n
#               n   mean sup|Fhat - F|   x sqrt(n)   Fhat(q05): mean        sd    sd / 0.05
#              10             0.2594      0.8203             0.0498    0.0689       1.3785
#             100             0.0853      0.8533             0.0500    0.0220       0.4403
#            1000             0.0273      0.8619             0.0500    0.0069       0.1374
#           10000             0.0087      0.8671             0.0500    0.0022       0.0437
#          100000             0.0028      0.8703             0.0500    0.0007       0.0137
```

The middle column is the convergence, and the column beside it is its rate. The worst gap between $\hat F_n$ and $F$ anywhere on the line falls from $0.2594$ at ten observations to $0.0028$ at a hundred thousand, and multiplying by $\sqrt{n}$ flattens it onto $0.8203$, $0.8533$, $0.8619$, $0.8671$, $0.8703$ — converging to a constant, which is the same $1/\sqrt{n}$ the sample mean obeyed, now governing an entire function at once.

The last two columns are where the bootstrap's practical limit lives. At the $5\%$ quantile, $\hat F_n$ is unbiased at every sample size — $0.0500$ on four of the five rows — but its standard deviation *relative to the quantity being estimated* is $1.3785$ at $n=10$ and only falls below $5\%$ somewhere past ten thousand observations. A resampling scheme inherits this directly: the centre of the distribution is pinned down long before the tail is, so a bootstrap confidence interval for a median is trustworthy at sample sizes where a bootstrap interval for a $1\%$ VaR is still mostly noise. The published practice of "checking stability by deleting the worst day" is a crude test of exactly this column.

## Without a Mean There Is Nothing to Converge To

The weak law's real hypothesis, established in the first proof, is that $\mu$ exists. When it does not, the sample mean does not converge slowly. It does not converge.

??? note "Proof that the sample mean of Cauchy draws has the same distribution as a single draw"
    The standard Cauchy density is $f(x)=1/\big(\pi(1+x^{2})\big)$, which is the Student's $t$ of [Student's t Distribution](../part-05-common-distributions/16-students-t-distribution.md) at $\nu=1$. Its mean does not exist: $\int\lvert x\rvert f(x)\,dx$ diverges logarithmically, so the defining integral is not merely infinite but undefined, and [Higher-Order Moments](../part-04-expectation-and-moments/03-higher-order-moments.md) is the page that catalogues when a family runs out of moments this way.

    Its characteristic function is $\varphi(t)=\mathbb{E}[e^{itX}]=e^{-\lvert t\rvert}$. For independent draws the characteristic function of a sum is the product, so $\bar X_n=\tfrac1n\sum_iX_i$ has

    $$\varphi_{\bar X_n}(t)=\prod_{i=1}^{n}\varphi(t/n)=\big(e^{-\lvert t\rvert/n}\big)^{n}=e^{-\lvert t\rvert},$$

    which is $\varphi$ again. The average of $n$ Cauchy draws is exactly Cauchy, for every $n$: same location, same scale, same tails. Averaging a million of them produces a random variable statistically indistinguishable from the first observation.

    The load-bearing hypothesis is visible in the arithmetic. The proof of the weak law needed $\mathrm{var}(\bar X_n)=\sigma^{2}/n$, a division by $n$ that came from a finite second moment; here the scale parameter divides by $n$ and then the $n$ independent contributions multiply it straight back, because the Cauchy's tail is heavy enough that the sum is dominated by its largest term rather than built from all of them. That is the general mechanism, not a Cauchy quirk: for a power-law tail with index $\alpha\leq1$ the maximum of $n$ draws is the same order as their sum, so an average is a report on its own worst observation. [Heavy-Tailed Returns](../part-18-quant-finance-applications/12-heavy-tailed-returns.md) is where that regime is measured rather than assumed.

```python
import numpy as np

rng = np.random.default_rng(7127)
reps = 200_000
print(f"  interquartile range of the sample mean, {reps} histories at each n")
print("           n      normal    t(2.65)    t(1) = Cauchy      normal x sqrt(n)   Cauchy x sqrt(n)")


def iqr(a):
    lo, hi = np.quantile(a, [0.25, 0.75])
    return hi - lo


for n in (10, 100, 1_000, 10_000):
    g = rng.standard_normal((reps, n)).mean(axis=1)
    t = rng.standard_t(2.65, (reps, n)).mean(axis=1)
    c = rng.standard_cauchy((reps, n)).mean(axis=1)
    print(f"  {n:10d} {iqr(g):11.4f} {iqr(t):10.4f} {iqr(c):16.4f}"
          f" {iqr(g) * np.sqrt(n):21.4f} {iqr(c) * np.sqrt(n):18.4f}")
# =>   interquartile range of the sample mean, 200000 histories at each n
#               n      normal    t(2.65)    t(1) = Cauchy      normal x sqrt(n)   Cauchy x sqrt(n)
#              10      0.4282     0.6793           2.0006                1.3541             6.3265
#             100      0.1353     0.2466           2.0014                1.3527            20.0142
#            1000      0.0427     0.0824           2.0054                1.3490            63.4153
#           10000      0.0134     0.0267           2.0003                1.3449           200.0264
```

The interquartile range is used rather than the standard deviation because the Cauchy has no standard deviation to report, which is itself the point: the ordinary diagnostic is unavailable exactly where it is most needed.

The Cauchy column is $2.0006$, $2.0014$, $2.0054$, $2.0003$. Three decades of sample size and the spread of the sample mean does not move — it is the theoretical value $2$ at every row, precisely as the proof requires. Multiplying by $\sqrt{n}$, which flattens the normal column onto about $1.35$, sends the Cauchy column to $6.3265$, $20.0142$, $63.4153$, $200.0264$: growing like $\sqrt{n}$, because there is no convergence for the scaling to expose.

The middle column is the case that actually occurs. [Returns and Their Distributions](../../part-03-statistics/02-returns-and-distributions.md) fits daily SPY returns to a Student's $t$ with $\nu=2.65$ and separately to a stable law with $\alpha=1.53$. The $t_{2.65}$ column converges — $0.6793$ down to $0.0267$, a factor of $25$ where $\sqrt{n}$ would give $31.6$ — so under that fit the weak law applies, with a mean, a variance, and a rate slightly worse than root-$n$ at these sample sizes. Under the stable fit with $\alpha=1.53<2$ there is no variance, and under a hypothetical $\alpha\leq1$ there would be no mean. **The two fits to the same data disagree about whether the theorem on this page applies**, and no amount of additional data from that market settles it, because the disagreement is about the tail and the tail is where data is scarcest by construction.

!!! warning "Every claim that a strategy's edge will emerge with more data is a bet that the fourth column of that table is not the one you are in"
    The failure mode is quiet. A Cauchy sample mean computed on a million observations returns a number, formatted to four decimals, with a plausible magnitude; nothing in the output distinguishes it from a converged estimate. The diagnostic that does distinguish them costs almost nothing: recompute the statistic on nested subsamples — the first tenth, the first half, all of it — and read the sequence. A converging estimate settles and stays settled; a non-converging one hops, and the hops are the same size at every scale. That plot belongs beside any long-run average whose underlying distribution has not been checked for a tail index, which in practice means all of them.

## A Consistent Estimator Is a Promise About a Sample You Will Not Collect

The weak law is the ancestor of consistency, and consistency is the weakest property an estimator can have that is worth naming. It says the estimator is not wrong in the limit. It says nothing about bias at any finite $n$, nothing about the spread at the $n$ you have, and nothing about how the error is distributed — those are [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md), this page's third section, and [The Central Limit Theorem](03-central-limit-theorem.md) respectively.

So consistency is never the interesting question, and an argument that reaches for it has usually skipped the interesting question. When someone says an estimate will be fine with enough data, there are exactly two things worth asking, in order. Does a limit exist — is there a finite mean for the average to approach, which is a claim about the tail and not about the sample. And if so, what is the standard error at the sample size actually available, which the first table computes in one line from $\sigma$ and the calendar. The published $0.075\pm0.039$ is the answer to the second question for the most-studied series in finance, after the longest sample most practitioners will ever assemble. It is what convergence looks like on the way there, and it is where every decision has to be made.
