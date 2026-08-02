# Slutsky's Theorem

Every limiting distribution in statistics is stated in terms of parameters nobody knows. The central limit theorem gives $\sqrt n(\bar X_n-\mu)\Longrightarrow\mathcal{N}(0,\sigma^{2})$, and using it requires dividing by a $\sigma$ that has to be estimated from the same data. Slutsky's theorem is the permission slip for that substitution: replace a nuisance parameter by anything that converges in probability to it and the limiting distribution is unchanged. It is the reason a t-statistic works, it costs one line, and it certifies only that the replacement converges — not that it converges to the right number.

This page covers the statement and its three arithmetic forms, the proof written as a discarded $o_p(1)$ remainder, the t-statistic as the theorem's canonical use, what happens when a plug-in estimator is consistent for the wrong quantity, and what happens when the thing being plugged in has a random limit rather than a constant one. It does not prove the central limit theorem that supplies its input, which is [The Central Limit Theorem](03-central-limit-theorem.md); it does not prove the continuous mapping theorem its own proof leans on, which is [Continuous Mapping Theorem](06-continuous-mapping-theorem.md); it does not linearize any transform, which is [The Delta Method](04-delta-method.md); and it constructs no test and no critical region, which is [Part XII](../part-12-hypothesis-testing/index.md).

The trading stake is the most alarming pair of numbers in the course. [Hypothesis Testing and Multiple Testing](../../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) takes a strategy with a Sharpe of $0.30$, tests it on overlapping twenty-one-day returns, and gets a naive t-statistic of $7.40$ — "a seven-sigma discovery" — which a HAC correction deflates to $1.73$, because "each day's return is counted twenty-one times and the test believes there are 6,138 independent observations when there are effectively 335." Slutsky's hypotheses are satisfied throughout. The fourth section shows that this is exactly the point.

## It Combines Two Different Kinds of Limit, and Only One of Them May Be Random

Let $X_n\Longrightarrow X$ in distribution and $Y_n\xrightarrow{\ p\ }c$ for a **constant** $c$. Then

$$X_n+Y_n\ \Longrightarrow\ X+c,\qquad X_nY_n\ \Longrightarrow\ cX,\qquad \frac{X_n}{Y_n}\ \Longrightarrow\ \frac{X}{c}\ \ (c\neq0).$$

The shape of the statement is a descendant of something entirely deterministic. [Sequences and Infinite Series](../part-01-mathematical-foundations/04-sequences-and-series.md) proves the limit laws for ordinary sequences — sums of convergent sequences converge to the sum of the limits, products to the product — and notes that the probabilistic versions "add bookkeeping about *how* convergence is measured, not a new idea." Slutsky's theorem is that sentence made precise. The new content is not the arithmetic; it is that the two sequences are allowed to converge in two *different senses*, and the price of that permission is that one of the two limits must be degenerate.

## The Proof Is a Remainder That Is o_p(1) and Therefore Discardable

The obstacle is that convergence in distribution says nothing about joint behaviour. Knowing the marginal limits of $X_n$ and $Y_n$ generally tells you nothing about the limit of $X_n+Y_n$, since the pair could be arranged with any dependence at all. The whole theorem is the observation that a constant limit removes that freedom.

??? note "Proof that a constant limit makes marginal convergence into joint convergence, and the rest is the mapping theorem"
    First the sum, in the notation of [Mathematical Notation](../part-01-mathematical-foundations/03-mathematical-notation.md). Write $Y_n=c+R_n$ where $R_n=Y_n-c$ satisfies $R_n=o_p(1)$ by hypothesis. Then

    $$X_n+Y_n=(X_n+c)+R_n.$$

    The first bracket converges in distribution to $X+c$, being a fixed shift of a converging sequence. For the second, fix any $\epsilon>0$ and any continuity point $x$ of the limiting CDF. Conditioning on whether $\lvert R_n\rvert<\epsilon$ gives

    $$\mathbf{P}(X_n+Y_n\leq x)\leq\mathbf{P}(X_n+c\leq x+\epsilon)+\mathbf{P}(\lvert R_n\rvert\geq\epsilon),$$

    and the same argument from below with $x-\epsilon$. The second probability vanishes because $R_n=o_p(1)$; letting $n\to\infty$ and then $\epsilon\to0$, using continuity of the limiting CDF at $x$, squeezes $\mathbf{P}(X_n+Y_n\leq x)$ onto $\mathbf{P}(X+c\leq x)$. **An $o_p(1)$ term can be added to a sequence converging in distribution without changing the limit** — that is the whole theorem, and the products and quotients follow from it.

    For the product, write $X_nY_n=cX_n+X_nR_n$. The first term converges to $cX$. The second is $O_p(1)\cdot o_p(1)=o_p(1)$, since $X_n$ converges in distribution and is therefore bounded in probability, so the sum converges to $cX$ by the paragraph above. For the quotient with $c\neq0$, note $1/Y_n\xrightarrow{\ p\ }1/c$ by [Continuous Mapping Theorem](06-continuous-mapping-theorem.md) — the map $y\mapsto1/y$ is continuous at $c$, which is where the hypothesis $c\neq0$ is spent — and apply the product form.

    Stated at the right level of generality, all three cases are one case: $(X_n,Y_n)\Longrightarrow(X,c)$ jointly, and then $g(x,y)=x+y$, $xy$ or $x/y$ is continuous at every point of the limit's support, so the mapping theorem finishes it.

    The load-bearing hypothesis is that $c$ is a **constant**, and it is not a technicality. Take $Z\sim\mathcal{N}(0,1)$ and set $X_n=Z$ for every $n$. If $Y_n=Z$ then $X_nY_n=Z^{2}\Longrightarrow\chi^{2}_{1}$; if instead $Y_n=-Z$ then $X_nY_n=-Z^{2}$, which is the negative of a chi-square. The marginal limits are identical in the two cases — both $X_n$ and $Y_n$ converge in distribution to a standard normal either way — and the products converge to distributions that do not even share a support. With a random limit, marginal convergence determines nothing, and the fifth section shows what that costs when the random object is a denominator.

## Every t-Statistic You Have Ever Read Is This Theorem

The canonical application is so routine it is usually invisible. The central limit theorem delivers a normal limit for a standardized mean, but the standardization uses $\sigma$; what gets computed uses $\hat\sigma$. Factor the computed statistic into the two pieces the theorem wants:

$$\frac{\sqrt n\,(\bar X_n-\mu)}{\hat\sigma}=\underbrace{\frac{\sqrt n\,(\bar X_n-\mu)}{\sigma}}_{\Longrightarrow\ \mathcal{N}(0,1)}\cdot\underbrace{\frac{\sigma}{\hat\sigma}}_{\xrightarrow{\ p\ }1}.$$

The first factor converges in distribution by [The Central Limit Theorem](03-central-limit-theorem.md). The second converges in probability to the constant $1$, because $\hat\sigma^{2}\xrightarrow{\ p\ }\sigma^{2}$ by [The Weak Law of Large Numbers](01-weak-law-of-large-numbers.md) applied to squared deviations and then $\sqrt{\cdot}$ and $\sigma/\cdot$ are continuous at $\sigma>0$. Slutsky multiplies them and the limit is $\mathcal{N}(0,1)$: the estimated denominator costs nothing asymptotically.

```python
import numpy as np
from scipy.stats import t as tdist

rng = np.random.default_rng(7507)
z, reps = 1.959963984540054, 400_000
print(f"  actual size of a nominal 5% two-sided test, {reps} samples, normal data")
print("        n   sigma known   sigma estimated   exact t(n-1)   the t critical value")
for n in (5, 10, 30, 252, 6_300):
    x = rng.standard_normal((reps, n))
    known = np.mean(np.abs(x.mean(axis=1) * np.sqrt(n)) > z)
    est = np.mean(np.abs(x.mean(axis=1) * np.sqrt(n) / x.std(axis=1, ddof=1)) > z)
    print(f"  {n:9d} {known:13.4f} {est:17.4f} {2 * tdist.sf(z, n - 1):14.4f}"
          f" {tdist.ppf(0.975, n - 1):22.4f}")
# =>   actual size of a nominal 5% two-sided test, 400000 samples, normal data
#            n   sigma known   sigma estimated   exact t(n-1)   the t critical value
#              5        0.0496            0.1208         0.1216                 2.7764
#             10        0.0496            0.0814         0.0816                 2.2622
#             30        0.0494            0.0592         0.0597                 2.0452
#            252        0.0502            0.0511         0.0511                 1.9695
#           6300        0.0500            0.0500         0.0500                 1.9603
```

The theorem's promise is the last row and it is kept exactly: at $6{,}300$ observations the known-$\sigma$ and estimated-$\sigma$ columns both print $0.0500$, and the critical value that would have been correct, $1.9603$, is indistinguishable from the normal's $1.9600$. Substituting the estimate cost nothing measurable.

The first row is what "asymptotically" is hiding. At five observations the same substitution takes the actual size of a nominal $5\%$ test from $0.0496$ to $0.1208$ — the test rejects a true null one time in eight — and the simulated column agrees with the exact $t_4$ value of $0.1216$ to within sampling error, confirming that nothing is wrong except the sample size. Slutsky is silent about all of this. It says the two columns coincide in the limit, and the finite-sample repair is Gosset's exact $t$ distribution, which is available here only because the data was normal.

That is worth stating as a general lesson rather than a curiosity about small samples. **Slutsky guarantees the asymptotic size and provides no rate**, so the question of when $n$ is large enough for a plug-in denominator has to be answered separately, by exact theory when it exists and by simulation otherwise. By $n=252$ the discrepancy here is $0.0511$ against $0.0500$ and can be ignored. The next section is the case where it cannot be ignored at any $n$.

## A Consistent Variance Estimator Is Not the Same as a Correct One

The theorem's hypothesis is that $\hat\sigma^{2}$ converges in probability to a constant. It does not require that constant to be the variance the limiting distribution needs. When returns are dependent, the ordinary sample variance converges perfectly well — to a number that is not the right one — and every hypothesis of Slutsky's theorem is satisfied while the conclusion is worthless.

```python
import numpy as np

rng = np.random.default_rng(7523)
z, reps, days = 1.959963984540054, 4_000, 6_200
print(f"  a zero-edge strategy, {reps} histories of {days} days, tested on h-day overlapping returns")
print("       h    lag-1 rho    mean naive t    mean HAC t    size, naive    size, HAC    eff n / n")
d = rng.standard_normal((reps, days))
for h in (1, 5, 21, 63):
    c = np.cumsum(d, axis=1)
    x = c[:, h:] - c[:, :-h] if h > 1 else d                   # overlapping h-day sums
    n = x.shape[1]
    e = x - x.mean(axis=1, keepdims=True)
    g0 = (e * e).mean(axis=1)
    lrv = g0.copy()
    for k in range(1, h):                                      # Newey-West at the true bandwidth
        lrv += 2 * (1 - k / h) * (e[:, k:] * e[:, :-k]).mean(axis=1)
    rho = (e[:, 1:] * e[:, :-1]).mean(axis=1) / g0
    t_naive = x.mean(axis=1) * np.sqrt(n / g0)
    t_hac = x.mean(axis=1) * np.sqrt(n / np.maximum(lrv, 1e-12))
    print(f"  {h:6d} {rho.mean():12.4f} {np.abs(t_naive).mean():15.4f} {np.abs(t_hac).mean():13.4f}"
          f" {np.mean(np.abs(t_naive) > z):14.4f} {np.mean(np.abs(t_hac) > z):12.4f}"
          f" {(g0 / lrv).mean():12.4f}")
# =>   a zero-edge strategy, 4000 histories of 6200 days, tested on h-day overlapping returns
#           h    lag-1 rho    mean naive t    mean HAC t    size, naive    size, HAC    eff n / n
#           1      -0.0005          0.8033        0.8033         0.0490       0.0490       1.0000
#           5       0.7996          1.7981        0.9757         0.3795       0.1057       0.2944
#          21       0.9520          3.6984        0.9899         0.6807       0.1090       0.0716
#          63       0.9837          6.4597        1.0031         0.8173       0.1180       0.0241
```

Every history in this table has a true edge of exactly zero, and the daily returns are genuinely independent — the dependence is manufactured entirely by the overlap. At $h=21$ the lag-one autocorrelation is $0.9520$, against the $0.94$ the course measures on real overlapping returns, so the setup is a faithful synthetic version of the published catastrophe.

The naive column is the failure. The mean absolute t-statistic rises from $0.8033$ at daily frequency — the correct value for a standard normal — to $3.6984$ at $h=21$ and $6.4597$ at $h=63$, and the actual size of a nominal $5\%$ test reaches $0.6807$: **two-thirds of genuinely worthless strategies are declared significant**. The published $7.40$ is one draw from the $h=21$ row's distribution, and it is not an unlucky draw.

The last column is the mechanism and the diagnostic. The ratio of the ordinary variance to the long-run variance is $0.0716$ at $h=21$, which on $6{,}179$ overlapping observations is about $442$ independent ones — the same order as the course's "effectively 335". The ordinary sample variance is converging, in probability, to a constant, exactly as Slutsky requires; the constant is the one-period variance $\gamma_0$, and the quantity the central limit theorem for dependent sequences actually needs is the long-run variance $\gamma_0+2\sum_{k\geq1}\gamma_k$. The two differ by a factor of fourteen, the estimator is consistent for the wrong one, and nothing anywhere in the calculation is aware of it.

The HAC column shows what the repair buys and what it does not. Estimating the long-run variance directly brings the mean absolute t-statistic back to $0.9757$, $0.9899$, $1.0031$ — the standard normal's value, recovered at every overlap — and that is Slutsky's theorem working as designed with a *correctly targeted* plug-in. But the actual test size only falls to $0.1057$, $0.1090$, $0.1180$, which is still more than twice nominal. A HAC estimator is a sum of many autocovariances each estimated with error, so it is consistent and badly biased at finite $n$, and the residual over-rejection is that bias. **HAC is a large improvement and not a fix**, and reporting a HAC t-statistic of $1.73$ as though it were exact overstates the case in the same direction, just less.

!!! warning "Consistency identifies the limit an estimator has, and says nothing about whether that limit is the quantity the formula needs"
    The two variance estimators in that table are both consistent, both computed from the same data, and differ by a factor of fourteen. That is not a paradox: they converge to different constants because they are estimating different things, and only one of them appears in the theorem being invoked. The pattern recurs everywhere a plug-in is used. A sample variance estimates the one-period variance and a t-statistic on dependent data needs the long-run variance. A pooled standard deviation estimates a within-group quantity and a difference-in-means test needs the between-group one. An in-sample residual variance estimates a fitted quantity and a forecast interval needs the predictive one. The question to ask of any plug-in is never "is this consistent" — it almost always is — but "consistent *for what*", and the answer has to be compared against the symbol it is being substituted for.

## A Random Limit in the Denominator Breaks It Outright

The previous section had a denominator converging to the wrong constant. The remaining failure is a denominator that does not converge to a constant at all, which is where the theorem's one structural hypothesis fails rather than merely being applied carelessly.

This happens whenever the quantity being plugged in is computed from a *fixed* amount of data while the numerator's precision grows. A risk model calibrated once on a six-month window and never refreshed, a volatility estimate frozen at a reference date, a scaling factor set from a pilot study — in each case $\hat\sigma$ is a random variable whose distribution does not depend on $n$, so $\hat\sigma\Longrightarrow$ something random and $\hat\sigma\xrightarrow{\ p\ }$ nothing.

```python
import numpy as np

rng = np.random.default_rng(7537)
z, reps = 1.959963984540054, 1_000_000
print(f"  nominal 5% test; the denominator is either the whole sample or a frozen calibration window")
print("        n    denominator grows with n    frozen w = 6    frozen w = 21    frozen w = 60")
num = rng.standard_normal(reps)                                # sqrt(n) * xbar, already standard
frozen = {w: np.sqrt(rng.chisquare(w - 1, reps) / (w - 1)) for w in (6, 21, 60)}
for n in (252, 1_260, 6_300, 25_200):
    grow = np.sqrt(rng.chisquare(n - 1, reps) / (n - 1))
    row = [np.mean(np.abs(num / grow) > z)] + [np.mean(np.abs(num / s) > z) for s in frozen.values()]
    print(f"  {n:9d} {row[0]:26.4f} {row[1]:15.4f} {row[2]:16.4f} {row[3]:16.4f}")
# =>   nominal 5% test; the denominator is either the whole sample or a frozen calibration window
#            n    denominator grows with n    frozen w = 6    frozen w = 21    frozen w = 60
#            252                     0.0506          0.1068           0.0636           0.0540
#           1260                     0.0497          0.1068           0.0636           0.0540
#           6300                     0.0496          0.1068           0.0636           0.0540
#          25200                     0.0495          0.1068           0.0636           0.0540
```

The first column is Slutsky's theorem: a denominator built from all $n$ observations converges in probability to $\sigma$, and the actual size sits at $0.0506$, $0.0497$, $0.0496$, $0.0495$ — nominal, and getting slightly better as $n$ grows.

The three frozen columns are byte-identical down every row. $0.1068$, $0.0636$ and $0.0540$ at $n=252$; the same three numbers to four decimals at $n=25{,}200$, a hundredfold more data. **The size error does not improve with sample size, because the sample size is not what it depends on.** The statistic's limiting distribution is a $t$ with $w-1$ degrees of freedom no matter how large $n$ becomes, since the numerator converges to a standard normal and the denominator never converges to anything — the theorem's conclusion is unavailable and the object it would have produced does not exist.

What makes this failure mode dangerous is that it is invisible in the usual diagnostics. The point estimate is fine and improves with $n$ exactly as expected. The standard error shrinks like $1/\sqrt n$ exactly as expected. Only the *calibration* is wrong, permanently, by an amount fixed by a window size nobody is thinking about. The remedy is not statistical: it is to make the denominator's data grow with the numerator's, or to use the $t_{w-1}$ critical value that honestly reflects the frozen window, which at $w=6$ is $2.5706$ rather than $1.9600$.

## The Theorem Certifies the Substitution and Never the Substitute

Slutsky's theorem is a piece of permission, and the useful way to hold it is by what it does not say. It does not say the plug-in is accurate — the first table's $0.1208$ at $n=5$. It does not say the plug-in targets the right quantity — the second table's factor of fourteen. It does not say anything at all when the plug-in fails to settle on a constant — the third table's four identical rows. It says only that if the replacement converges in probability to the parameter it replaces, the limiting distribution survives.

So the checklist is three questions, and each of the last three sections is one of them. Does the plug-in converge to a constant at all, or is it computed from a window that does not grow — a question about the estimator's construction, not its formula. If it converges, is the constant the one appearing in the limiting distribution, which for anything with time dependence means comparing a one-period variance against a long-run variance and is where the largest errors in published backtests live. And is the sample size large enough that the asymptotic answer has arrived, which the theorem refuses to address and simulation answers in twenty lines.

The published deflation from $7.40$ to $1.73$ is a correct answer to the second question and a partial answer to the third. That single correction moved a result from seven-sigma to insignificant without touching the strategy, the data, or the point estimate. Every one of those three numbers was right the whole time; only the denominator was wrong, and it was wrong in the way that is hardest to see, because it was consistent.
