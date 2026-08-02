# Continuous Mapping Theorem

A continuous function passes a limit through unchanged, in all three of this part's modes, and that single fact is why a converging variance estimator is a converging volatility estimator with nothing further to prove. The hypothesis is usually quoted as "$g$ is continuous", which is not quite it: what is needed is that $g$ be continuous on a set the limit lands in with probability one. That distinction is invisible for the smooth transforms of a normal that fill textbooks, and it is decisive for the thresholds, stops, indicators and quantiles that fill trading systems, every one of which is discontinuous by design.

This page covers the statement for almost-sure, in-probability and in-distribution convergence, the proof in each mode, the volatility estimator as the worked consequence and the contrast between what does and does not pass through a transform, the counterexample at a discontinuity, the refinement from "continuous" to "continuous where the limit lives", and the two separate continuity conditions that are easily conflated. It does not prove the limit theorems that supply its inputs, which are [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md), [The Strong Law of Large Numbers](02-strong-law-of-large-numbers.md) and [The Central Limit Theorem](03-central-limit-theorem.md); it does not linearize anything, which is [The Delta Method](04-delta-method.md); it does not define continuity, which is [Calculus Essentials](../part-01-mathematical-foundations/06-calculus-essentials.md); and it establishes no property of any estimator, which is [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md).

The trading stake is a statistic the course reports and then disowns. [Drawdowns, Tail Risk, and Stress Testing](../../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md) fits an extreme-value shape parameter of $\xi=+0.327$ to a book's losses, notes that "moments exist only up to order $1/0.327=3.1$, so the book's *kurtosis does not exist* in the limit", and concludes that the reported sample kurtosis of $29.2$ "is not an estimate of a population quantity — it is a number that grows with sample size." That is a plug-in failing, and the sixth section shows it is not this theorem's fault: the map is perfectly continuous and the input never converged. Distinguishing that failure from the other kind is what the page is for.

## Continuity Preserves All Three Limits, With One Qualification

Let $g$ be a function whose set of discontinuity points is $D_g$, and suppose $\mathbf{P}(X\in D_g)=0$ — the limit almost never lands where $g$ misbehaves. Then

$$X_n\xrightarrow{\ a.s.\ }X\ \Rightarrow\ g(X_n)\xrightarrow{\ a.s.\ }g(X),\qquad X_n\xrightarrow{\ p\ }X\ \Rightarrow\ g(X_n)\xrightarrow{\ p\ }g(X),\qquad X_n\Longrightarrow X\ \Rightarrow\ g(X_n)\Longrightarrow g(X).$$

Three theorems with one hypothesis, and the mode is carried along untouched.

??? note "Proof of the mapping theorem in each of the three modes"
    **Almost surely.** Let $A=\{\omega:X_n(\omega)\to X(\omega)\}$ and $B=\{\omega:X(\omega)\notin D_g\}$. By hypothesis $\mathbf{P}(A)=\mathbf{P}(B)=1$, so $\mathbf{P}(A\cap B)=1$. Fix any $\omega\in A\cap B$: then $X_n(\omega)\to X(\omega)$ is an ordinary convergent sequence of numbers, and $g$ is continuous at the point $X(\omega)$, so $g(X_n(\omega))\to g(X(\omega))$ by the deterministic limit law of [Sequences and Infinite Series](../part-01-mathematical-foundations/04-sequences-and-series.md). That holds on a set of probability one, which is the conclusion. This mode is not an analogue of the deterministic result; it *is* the deterministic result, applied one history at a time.

    **In probability.** Fix $\epsilon>0$ and $\delta>0$. Let $B_\eta$ be the set of $x\notin D_g$ at which continuity holds with modulus $\eta$ — that is, $\lvert x-y\rvert<\eta$ implies $\lvert g(x)-g(y)\rvert<\epsilon$. The sets $B_\eta$ increase to a set of probability one as $\eta\downarrow0$, so choose $\eta$ with $\mathbf{P}(X\notin B_\eta)<\delta/2$. Then

    $$\mathbf{P}\big(\lvert g(X_n)-g(X)\rvert\geq\epsilon\big)\leq\mathbf{P}\big(\lvert X_n-X\rvert\geq\eta\big)+\mathbf{P}(X\notin B_\eta),$$

    because if $X$ lands in $B_\eta$ and $X_n$ is within $\eta$ of it, the images are within $\epsilon$. The first term vanishes as $n\to\infty$ and the second is below $\delta/2$, so the limsup is under $\delta$ for every $\delta$.

    **In distribution.** Here the portmanteau characterization does the work: $X_n\Longrightarrow X$ if and only if $\mathbb{E}[h(X_n)]\to\mathbb{E}[h(X)]$ for every bounded continuous $h$. Take any such $h$; then $h\circ g$ is bounded, and it is continuous everywhere $g$ is, so its discontinuity set is contained in $D_g$ and therefore null under the limit. A small extension of the portmanteau statement — that convergence in distribution gives $\mathbb{E}[f(X_n)]\to\mathbb{E}[f(X)]$ for bounded $f$ whose discontinuities are $X$-null — applies to $f=h\circ g$ and yields $\mathbb{E}[h(g(X_n))]\to\mathbb{E}[h(g(X))]$. Since $h$ was arbitrary, $g(X_n)\Longrightarrow g(X)$.

    The load-bearing hypothesis is $\mathbf{P}(X\in D_g)=0$ — that the **discontinuity set is null under the limit law**, not that $g$ is continuous everywhere. Each proof spent it in the same place: the almost-sure argument needed continuity at the point $X(\omega)$, the in-probability argument needed $X$ to land in $B_\eta$, and the in-distribution argument needed $h\circ g$'s discontinuities to be $X$-null. Without it every conclusion fails, and the fourth section shows that it fails outright rather than approximately. The failure is silent, because $g(X_n)$ is a number that gets computed either way.

## A Converging Variance Estimator Is a Converging Volatility Estimator for Free

The consequence [Sequences and Infinite Series](../part-01-mathematical-foundations/04-sequences-and-series.md) advertises is immediate. The sample variance satisfies $\hat v_n\xrightarrow{\ p\ }\sigma^{2}$ by [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md); the map $x\mapsto\sqrt x$ is continuous on $(0,\infty)$; the limit $\sigma^{2}$ lies there whenever $\sigma>0$, which is with probability one since it is a constant. Therefore $\hat\sigma_n\xrightarrow{\ p\ }\sigma$. No new moment condition, no new sample-size requirement, no new proof — the theorem is doing all the work, and it does the same for $1/\hat\sigma$, for $\log\hat\sigma$, for the annualization $\sqrt{252}\,\hat\sigma$ and for any other continuous rescaling.

What does *not* transfer is worth the same emphasis.

```python
import numpy as np
from scipy.special import gammaln

rng = np.random.default_rng(9311)
sigma, reps = 0.20, 200_000
print(f"  Bessel's correction is exact for the variance and cannot survive a square root")
print("        n   E[v_hat]/sigma^2   E[sd_hat]/sigma   c4(n) exact   bias %   relative SE %")
for n in (5, 10, 21, 63, 252, 1_260):
    v = sigma ** 2 * rng.chisquare(n - 1, reps) / (n - 1)
    c4 = np.exp(0.5 * np.log(2 / (n - 1)) + gammaln(n / 2) - gammaln((n - 1) / 2))
    print(f"  {n:9d} {v.mean() / sigma ** 2:18.4f} {np.sqrt(v).mean() / sigma:17.4f}"
          f" {c4:13.4f} {100 * (c4 - 1):8.2f} {100 / np.sqrt(2 * n):15.2f}")
# =>   Bessel's correction is exact for the variance and cannot survive a square root
#            n   E[v_hat]/sigma^2   E[sd_hat]/sigma   c4(n) exact   bias %   relative SE %
#              5             1.0007            0.9406        0.9400    -6.00           31.62
#             10             0.9995            0.9723        0.9727    -2.73           22.36
#             21             1.0009            0.9881        0.9876    -1.24           15.43
#             63             0.9997            0.9958        0.9960    -0.40            8.91
#            252             0.9996            0.9988        0.9990    -0.10            4.45
#           1260             0.9998            0.9997        0.9998    -0.02            1.99
```

The first numeric column is Bessel's correction doing exactly what it was designed to do: $1.0007$, $0.9995$, $1.0009$, $0.9997$, $0.9996$, $0.9998$ — unbiased at every sample size, to within the Monte Carlo error of two hundred thousand replications.

The second column is the same estimator after a square root, and it is biased at every sample size: $0.9406$ at $n=5$, matching the exact constant $c_4(n)=\sqrt{2/(n-1)}\,\Gamma(n/2)/\Gamma((n-1)/2)$ printed beside it at $0.9400$. The reason is Jensen's inequality, not sampling error — $\sqrt{\cdot}$ is strictly concave, so $\mathbb{E}[\sqrt{\hat v}]<\sqrt{\mathbb{E}[\hat v]}$, permanently and by an amount no correction to $\hat v$ can fix. **Consistency passes through a continuous map and unbiasedness does not**, and there is no version of Bessel's correction for the standard deviation that works for all distributions.

The last two columns are why nobody corrects it. At $n=252$ the bias is $-0.10\%$ against a relative standard error of $4.45\%$ — a forty-fifth of the noise, invisible in any application. At $n=5$ the bias is $-6.00\%$ against a standard error of $31.62\%$, a fifth of the noise, and still not the reason a five-day volatility estimate is useless. The theorem's conclusion is asymptotic and correct; the bias is a finite-sample artefact that the same theorem guarantees will vanish.

!!! note "The mapping theorem licenses the square root of 252 and says nothing whatever about which returns it is applied to"
    Annualization is $g(x)=\sqrt{252}\,x$, about as continuous as a function gets, so the theorem transfers convergence through it without conditions. [Performance Metrics and Reporting](../../part-05-backtesting-engine/04-performance-metrics-and-reporting.md) nonetheless records that applying $\sqrt{252}$ to monthly returns makes a $0.30$ strategy print $1.64$, "an error of a factor of $\sqrt{21}$ that has appeared in real fund marketing more than once" — and separately that even done correctly the monthly Sharpe is $0.36$ rather than $0.30$, because the $\sqrt n$ rule assumes independence across time. In both cases the map was continuous, the estimator converged, and the answer was wrong. The theorem certifies that a limit survives a transformation; it has no opinion about whether the transformation is the right one, and neither of those two errors is detectable by anything on this page.

## The Mapping Theorem Is the Deterministic Limit Law With Bookkeeping

[Sequences and Infinite Series](../part-01-mathematical-foundations/04-sequences-and-series.md) says that the probabilistic descendants of "continuity preserves limits" look far more sophisticated than they are, and the three proofs above are the evidence. The almost-sure case is literally the deterministic theorem evaluated pointwise. The in-probability case is the same $\epsilon$–$\delta$ argument with a set of small probability set aside. Only the in-distribution case needs new machinery, and that is because convergence in distribution is a statement about laws rather than about values, so there are no pointwise limits to push a function through.

It is worth admitting the ordering that follows. This page is logically **prior** to the two before it: [Slutsky's Theorem](05-slutskys-theorem.md) is the mapping theorem applied to $(x,y)\mapsto x+y$, $xy$ or $x/y$ once a constant limit has upgraded marginal convergence to joint convergence, and [The Delta Method](04-delta-method.md) is Slutsky applied to a Taylor expansion. The appendix numbers them in the opposite order because that is the order in which a practitioner meets them — the Sharpe ratio's standard error is needed weekly and the portmanteau lemma is needed never. Nothing in the mathematics justifies the file order, and a reader who wants the logical development should take this page first.

## At a Discontinuity the Theorem Is Not Merely Silent, It Is False

Where the hypothesis fails, the conclusion does not degrade gracefully. It is simply untrue, and the cleanest counterexample needs no randomness whatsoever.

??? note "Proof that a discontinuity at the limit point destroys the conclusion completely"
    Let $X_n\equiv1/n$, a deterministic sequence viewed as a sequence of random variables, and $X\equiv0$. Then $X_n\to X$ almost surely, in probability, and in distribution — every mode, with no error term, because the sequence is not random at all.

    Let $g(x)=\mathbf 1\{x>0\}$. Then $g(X_n)=1$ for every $n$, so $g(X_n)\to1$ in every mode, while $g(X)=g(0)=0$. The limit of the images is $1$ and the image of the limit is $0$: the two differ by the entire range of $g$, at every sample size, forever. There is no $n$ at which the approximation is good.

    The hypothesis fails in the most concentrated way possible. $D_g=\{0\}$, a single point, of Lebesgue measure zero and utterly negligible under any continuous law — but the limit $X$ is the constant $0$, so $\mathbf{P}(X\in D_g)=1$ rather than $0$. **A null set under the wrong measure is not a null set.** This is exactly the situation of an estimator converging to a parameter that sits on a threshold, and the discontinuity carries the limit's entire mass by construction rather than by accident.

    Read through the other lens, the same example shows why convergence in distribution is defined only at continuity points of the limiting CDF. Here $F_n(0)=\mathbf{P}(1/n\leq0)=0$ for every $n$, while $F(0)=\mathbf{P}(0\leq0)=1$. So $F_n(0)\not\to F(0)$, and the sequence would fail to converge in distribution at all under a naive definition requiring convergence everywhere — even though it converges to $0$ in the strongest possible sense. The restriction to continuity points is what makes the definition usable, and $0$ is precisely a discontinuity point of $F$.

The example looks contrived and its instances are not. A hit rate is $\mathbf 1\{r>0\}$ averaged. A stop is an indicator of a threshold crossing. A regime flag is $\mathbf 1\{\hat v>v_0\}$. "Did the strategy beat its benchmark" is $\mathbf 1\{\hat\mu_A>\hat\mu_B\}$. A historical VaR is a quantile, and a quantile is discontinuous exactly where $F$ is flat — which is why [Cumulative Distribution Functions](../part-03-random-variables/02-cumulative-distribution-functions.md) finds a strategy that is flat half the time reporting "the same VaR at the 30% confidence level as at the 70% one", and is right to call that the correct answer rather than a defect.

!!! warning "Every stop, threshold, flag and ranking is a discontinuous function of an estimate, and the theorem that licenses smooth transforms says nothing about any of them"
    The practical consequence is that convergence of an input buys nothing for these quantities and has to be re-argued from scratch. Two rules cover most cases. First, a discontinuous statistic inherits no guarantee from its input's consistency, so its sampling behaviour must be simulated rather than deduced — which costs the twenty lines of the last section on this page. Second, the guarantee is recoverable when the limit is known to sit *away* from the jump, which converts an asymptotic question into a finite-sample distance check: the flag $\mathbf 1\{\hat v>v_0\}$ is trustworthy exactly when $\lvert\sigma^{2}-v_0\rvert$ is several standard errors, and worthless otherwise. Neither rule requires new theory, and neither is applied by default.

## The Continuity Set Only Has to Catch the Limit

[Calculus Essentials](../part-01-mathematical-foundations/06-calculus-essentials.md) states the hypothesis as "$g$ is continuous", which is the special case $D_g=\varnothing$ and is true of every transform that page uses. The general condition is strictly weaker, and the extra room it buys is used constantly.

The reciprocal $g(x)=1/x$ is discontinuous at the origin, and the theorem applies to $1/\hat\sigma_n$ without qualification because the limit is $\sigma>0$ and $\mathbf{P}(\sigma\in\{0\})=0$. That is the step [Slutsky's Theorem](05-slutskys-theorem.md) needs for every quotient it forms, and it would be unavailable under the naive hypothesis. The absolute value $g(x)=\lvert x\rvert$ is continuous everywhere but differentiable nowhere at the origin, so the mapping theorem applies where [The Delta Method](04-delta-method.md) does not — continuity is strictly weaker than differentiability, and buys a strictly larger class of transforms at the price of a weaker conclusion. And a stop payoff $g(x)=\max(x,c)$ is continuous everywhere despite its kink, so convergence passes through it cleanly even though it manufactures an atom in the output distribution, exactly as the transform table of [Functions of Random Variables](../part-03-random-variables/08-functions-of-random-variables.md) records.

The two qualifications on this page are easy to conflate and are about different objects. The mapping theorem's condition is about where **$g$** jumps, and asks that the limit avoid those points. The definition of convergence in distribution carries its own condition, about where the limiting **$F$** jumps, and requires $F_n(x)\to F(x)$ only at points where $F$ is continuous. The counterexample above satisfies neither, which is why it can be read both ways, but they are independent requirements: a smooth $g$ applied to a limit with an atom engages the second and not the first, and an indicator applied to a limit with a continuous law engages the first and not the second.

## A Plug-In Is Only as Good as the Map at the Point It Lands

Every plug-in statistic has the form $g(\hat\theta)$, and there are exactly two ways for it to fail. Either $\hat\theta$ does not converge — a question about moments, settled by [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md) and [The Central Limit Theorem](03-central-limit-theorem.md) — or it converges and $g$ is discontinuous at the limit, a question about geometry, settled here. The two are unrelated, and in the output they are indistinguishable, because in both cases what comes back is a number.

```python
import numpy as np

rng = np.random.default_rng(9323)
sigma, reps = 0.195, 400_000
print("  panel A: the sign of an estimated edge, a discontinuous map of a converging estimate")
print("         days     years    sd(mean)   P(mean > 0): SR = 0    SR = 0.02    SR = 0.30")
for n in (252, 2_520, 25_200, 252_000):
    se = sigma / np.sqrt(n / 252)
    z = se * rng.standard_normal(reps)
    p = [np.mean(sr * sigma + z > 0) for sr in (0.0, 0.02, 0.30)]
    print(f"  {n:13d} {n / 252:9.1f} {se:11.4f} {p[0]:21.4f} {p[1]:12.4f} {p[2]:12.4f}")
print("  panel B: three maps of the same converging variance estimate")
print("            n   E[sqrt(v)]/sigma   E[sigma/sqrt(v)]   P(v > sigma^2)   P(v > 1.1 sigma^2)")
for n in (21, 63, 252, 1_260):
    v = rng.chisquare(n - 1, reps) / (n - 1)                   # in units of sigma^2
    print(f"  {n:13d} {np.sqrt(v).mean():18.4f} {(1 / np.sqrt(v)).mean():18.4f}"
          f" {np.mean(v > 1.0):16.4f} {np.mean(v > 1.1):20.4f}")
# =>   panel A: the sign of an estimated edge, a discontinuous map of a converging estimate
#             days     years    sd(mean)   P(mean > 0): SR = 0    SR = 0.02    SR = 0.30
#                252       1.0      0.1950                0.5005       0.5084       0.6180
#               2520      10.0      0.0617                0.5006       0.5261       0.8285
#              25200     100.0      0.0195                0.5000       0.5789       0.9987
#             252000    1000.0      0.0062                0.4999       0.7363       1.0000
#      panel B: three maps of the same converging variance estimate
#                n   E[sqrt(v)]/sigma   E[sigma/sqrt(v)]   P(v > sigma^2)   P(v > 1.1 sigma^2)
#                 21             0.9874             1.0398           0.4576               0.3400
#                 63             0.9957             1.0126           0.4753               0.2745
#                252             0.9990             1.0030           0.4878               0.1331
#               1260             0.9999             1.0005           0.4960               0.0072
```

Panel A is the second failure in its purest form. The estimator converges superbly — the standard deviation of the estimated mean falls from $0.1950$ to $0.0062$, a factor of $31.5$ across a thousandfold increase in data — while the probability that its *sign* is positive under a true edge of exactly zero reads $0.5005$, $0.5006$, $0.5000$, $0.4999$. A thousand years of daily data and the answer to "is the edge positive" is still a coin flip. Nothing is broken: the limit is sitting on the jump of $\mathbf 1\{x>0\}$, so the theorem does not apply and the sign of a converging estimate simply never converges.

The two columns beside it show what "eventually" buys and how slowly. At a true Sharpe of $0.02$ — small but genuinely non-zero, so the limit is off the jump and the theorem does apply — the probability climbs $0.5084$, $0.5261$, $0.5789$, $0.7363$, reaching only three-quarters after a thousand years. At a Sharpe of $0.30$ it reaches $0.9987$ after a century. Both converge to $1$, as they must, and the schedule is the same one [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md) computes: the distance from the jump has to exceed several standard errors, and the standard error falls like $1/\sqrt n$.

Panel B applies three maps to one converging variance estimate. The two continuous maps behave: $\mathbb{E}[\sqrt{\hat v}]/\sigma$ runs $0.9874\to0.9999$ and $\mathbb{E}[\sigma/\sqrt{\hat v}]$ runs $1.0398\to1.0005$, both approaching one, with the reciprocal biased in the opposite direction because $1/\sqrt{\cdot}$ is convex where $\sqrt{\cdot}$ is concave. The regime flag set exactly at the truth gives $0.4576$, $0.4753$, $0.4878$, $0.4960$ — converging to $\tfrac12$, not to the indicator's value at the limit, which is $0$. The flag set $10\%$ above the truth is the same map with the limit moved off the jump, and it converges properly: $0.3400$, $0.2745$, $0.1331$, $0.0072$. Same estimator, same function, two threshold choices, and only one of them has an asymptotic guarantee.

That gives the diagnostic this part closes on. For any plug-in, ask the two questions in order — does the input converge, and is the map continuous where it converges — and then check which of the two failed by recomputing the statistic on nested subsamples and reading it against $n$. A converged plug-in flattens out. A first-mode failure drifts without settling, which is what the published sample kurtosis of $29.2$ does under $\xi=+0.327$, and what a Cauchy sample mean does under no tail index at all. A second-mode failure hops between a small number of values and never lands, which is what a threshold flag does when the threshold is near the truth. Three shapes, one plot, and it costs less than the number it is checking.
