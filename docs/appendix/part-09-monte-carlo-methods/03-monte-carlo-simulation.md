# Monte Carlo Simulation

Monte Carlo is usually introduced as a numerical method, which is the wrong category and causes most of the trouble that follows. A numerical method has an error you can bound; this has an error that is itself a random variable, so the output of a simulation is never a number but always a number and an interval, and quoting the first without the second is not a rounding of the truth but a change of its type. Everything difficult about the subject follows from one further fact: the interval is estimated from the same draws that produced the number, so a simulation is a procedure that grades its own homework, and it will report high confidence in a wrong answer whenever the property making the answer wrong is also the property making the grading wrong.

This page covers the estimator and the dimension-free convergence rate that makes it worth having, the standard error and the four-fold rule that governs what more computing buys, the almost-sure convergence that licenses running a simulation until the answer stops moving, the second moment an error bar quietly assumes and the failure when it does not exist, and the construction of a null distribution as the cheapest statistic available. It does not build the generator or the transformations underneath, which are [Random Number Generation](01-random-number-generation.md) and [Sampling Methods](02-sampling-methods.md); it does not change the sampling law in order to shrink the variance, which are [Importance Sampling](04-importance-sampling.md), [Rejection Sampling](05-rejection-sampling.md) and [Variance Reduction](06-variance-reduction.md); it draws from a specified law rather than from data, which is [Bootstrap Methods](07-bootstrap-methods.md); it proves none of the limit theorems it uses, which are [Part VII](../part-07-asymptotic-theory/index.md); it builds no test and applies no multiplicity correction, which are [Part XII](../part-12-hypothesis-testing/index.md) and [Part XV](../part-15-multiple-testing/index.md); and it prices nothing, which is [Options Pricing](../../advanced/11-options-pricing.md).

The trading stake is a licence the appendix has already issued and never justified. [The Strong Law of Large Numbers](../part-07-asymptotic-theory/02-strong-law-of-large-numbers.md) observes that a Monte Carlo integration "is run until the answer stops moving, which is a path-wise stopping rule and not a fixed-$n$ statement", and that the practice "is legitimate only because the path itself converges". That is correct, and the third and fourth sections price it: the strong law holds under a hypothesis strictly weaker than the one an error bar needs, so there is a whole class of problems on which the practice is licensed, the answer does converge, and every standard error the run prints is wrong by a factor that grows as the run gets longer.

## An Integral Is an Expectation and an Expectation Is an Average

Every Monte Carlo problem has the same shape. Write the quantity of interest as an expectation $\theta=\mathbb{E}[f(X)]$ for a law of $X$ you can sample and a function $f$ you can evaluate, draw $X_1,\dots,X_N$ independently, and report

$$\hat\theta_N=\frac1N\sum_{i=1}^{N}f(X_i).$$

The estimator is unbiased for any $f$ with a finite mean and consistent by the law of large numbers. What makes the method worth the trouble is not that — a grid rule is also consistent — but the rate.

??? note "Proof that the Monte Carlo error rate is $N^{-1/2}$ in every dimension, and that this is what it is bought for"
    Assume $\sigma^{2}=\mathrm{var}(f(X))<\infty$. Independence gives $\mathrm{var}(\hat\theta_N)=\sigma^{2}/N$ exactly, with no approximation and no appeal to a limit, so the root-mean-square error is $\sigma/\sqrt N$. Nothing in that calculation mentions the dimension of $X$. Dimension enters only through $\sigma$, which is a property of the integrand rather than of the space it lives in.

    Compare a deterministic rule. A product grid with $m$ points per axis in $d$ dimensions uses $N=m^{d}$ evaluations. A one-dimensional rule of order $p$ — $p=2$ for the trapezoid rule, $p=4$ for Simpson's — extends to the product grid with error $O(m^{-p})=O(N^{-p/d})$. The exponent carries $d$ in the denominator, so the rate degrades with every axis added, and the two methods cross where $p/d=1/2$: the trapezoid rule loses at $d=5$, Simpson's at $d=9$, and past that no amount of quadrature sophistication recovers the ground, because the deficiency is the exponential growth of the grid rather than the quality of the rule.

    The load-bearing hypothesis is $\sigma^{2}<\infty$, and it is the only one. Its role is easy to miss because it is stated as a regularity condition and behaves as a modelling assumption: the entire apparatus of error bars, sample-size planning and convergence diagnostics is a claim about a second moment, and the fourth section is what happens when the claim is false. [Sequences and Series](../part-01-mathematical-foundations/04-sequences-and-series.md) puts the same arithmetic in the units that matter, noting that the gap between geometric and $p$-series decay "decides whether a Monte Carlo estimator is usable at $10^4$ paths or needs $10^8$" — and $N^{-1/2}$ is the slowest useful rate there is.

## The Standard Error Is Reported or the Estimate Is Not

Because $\mathrm{var}(\hat\theta_N)=\sigma^{2}/N$, the estimator's own draws supply its precision: the sample standard deviation $s$ of the $f(X_i)$ estimates $\sigma$, and $s/\sqrt N$ is the standard error. That is the whole reporting discipline, and it has one immediate consequence usually called the **four-fold rule** — halving the error costs four times the paths, so a simulation that is one decimal place short is not a small amount of work away from being right.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(9031)
s0, k, r, sigma, T = 100.0, 105.0, 0.03, 0.20, 1.0
d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
exact = s0 * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))

print(f"  a European call worth {exact:.4f} exactly, priced by simulation at seven sample sizes")
print("        paths      estimate    standard error    error in se    se x sqrt(paths)")
for n in (1_000, 10_000, 100_000, 1_000_000, 2_000_000, 10_000_000, 100_000_000):
    tot = sq = 0.0
    left = n
    while left:                                                # chunked, so 1e8 paths fit in ram
        m = min(left, 10_000_000)
        z = rng.standard_normal(m)
        pay = np.exp(-r * T) * np.maximum(s0 * np.exp((r - 0.5 * sigma ** 2) * T
                                                      + sigma * np.sqrt(T) * z) - k, 0.0)
        tot += pay.sum()
        sq += (pay ** 2).sum()
        left -= m
    est = tot / n
    se = np.sqrt((sq / n - est ** 2) * n / (n - 1) / n)
    print(f"  {n:11d} {est:13.4f} {se:17.5f} {(est - exact) / se:14.2f} {se * np.sqrt(n):19.3f}")
# =>   a European call worth 7.1281 exactly, priced by simulation at seven sample sizes
#            paths      estimate    standard error    error in se    se x sqrt(paths)
#             1000        7.4335           0.41426           0.74              13.100
#            10000        7.2066           0.12757           0.62              12.757
#           100000        7.1470           0.03972           0.48              12.560
#          1000000        7.1173           0.01253          -0.86              12.535
#          2000000        7.1336           0.00887           0.62              12.545
#         10000000        7.1328           0.00397           1.19              12.540
#        100000000        7.1289           0.00125           0.68              12.532
```

The setup is the risk-neutral expectation of [Stochastic Calculus](../../advanced/03-stochastic-calculus.md), whose two-million-path run prices this call at $7.1341\pm0.0174$ against the analytic $7.1281$; the row at two million here reads $7.1336\pm0.00887$, and $1.96\times0.00887=0.0174$ is the same interval on a different seed. Reading the estimate column alone would suggest a method converging erratically — $7.4335$, then $7.2066$, then $7.1470$, then $7.1173$, which is on the *other* side of the truth.

The fourth column is what makes that pattern uninteresting. Every error, measured in units of its own standard error, lies between $-0.86$ and $1.19$. The estimates are not converging erratically; they are converging exactly as fast as they claim to, and the apparent wobble is what a sequence of independent draws from a normal distribution looks like. **An estimate that has moved by less than its standard error has not moved.**

The last column is the theorem in one line. The quantity $s\sqrt N$ estimates $\sigma$, the integrand's own standard deviation, and it is flat at about $12.53$ across five orders of magnitude in $N$ — $13.100$, $12.757$, $12.560$, $12.535$, $12.545$, $12.540$, $12.532$. That flatness *is* the $N^{-1/2}$ rate: the standard error falls from $0.414$ to $0.00125$, a factor of $331$, while $N$ rises by a factor of $10^{5}$, and $\sqrt{10^{5}}=316$. The price of the last decimal is visible in the same column read the other way. Going from $10^{6}$ to $10^{8}$ paths — a hundredfold increase in compute — bought the estimate one digit.

!!! note "Reporting the number of paths and reporting the standard error are not the same disclosure, and only one of them is a claim"
    A simulation described as "two million paths" has stated its cost. A simulation described as "$7.1336\pm0.0174$ at $95\%$" has stated what it found, and the difference matters because the map between them is the unknown $\sigma$. The two-million-path call above and a two-million-path barrier option with a $1\%$ knock-out probability have identical cost and standard errors differing by an order of magnitude, because almost every path of the second contributes nothing to the estimate. The convention worth adopting is the one [Stochastic Calculus](../../advanced/03-stochastic-calculus.md) uses without comment: print the interval, then print the discrepancy from any known reference in units of that interval, so that agreement becomes a number instead of an impression. A Monte Carlo price quoted to four decimals with no interval has communicated three digits of noise as though they were data.

## Convergence Is Almost Sure, Which Is What Licenses Running Until It Stops Moving

The practice everyone follows is to run a simulation until the answer stabilizes and then stop. This is not covered by the weak law, which says that for each fixed $N$ the estimator is probably close and says nothing about the behaviour of the *sequence* — a rule that watches the running answer and decides when to halt is a statement about the whole path $\hat\theta_1,\hat\theta_2,\dots$, and the weak law permits that path to leave any neighbourhood infinitely often.

??? note "Proof that a path-wise stopping rule is licensed by almost-sure convergence and by nothing weaker, and that the licence does not extend to the error bar"
    The strong law states that $\hat\theta_N\to\theta$ almost surely whenever $\mathbb{E}|f(X)|<\infty$. Fix $\varepsilon>0$. Almost-sure convergence says that with probability one there is a finite $N_0(\omega)$, depending on the realized path, beyond which $|\hat\theta_N-\theta|<\varepsilon$ for every $N\geq N_0$. A rule that halts once the running mean has been stable to within $\varepsilon$ is therefore examining a sequence which, on almost every path, is eventually and permanently inside the target band; the rule terminates with probability one and returns a value within tolerance.

    Under the weak law alone none of that follows. Convergence in probability is compatible with $|\hat\theta_N-\theta|>\varepsilon$ occurring for infinitely many $N$ on every path, provided the fraction of such $N$ thins out. A stability rule applied to such a sequence would still halt — it would halt during one of the quiet stretches — and the value it returned would carry no guarantee whatever.

    The load-bearing asymmetry is between the two hypotheses. The strong law needs only $\mathbb{E}|f(X)|<\infty$, a first-moment condition. The standard error, and every confidence interval built from it, needs $\mathbb{E}[f(X)^{2}]<\infty$, a second-moment condition. Nothing observable in the run distinguishes them, because the sample variance of a finite sample is finite whether or not the population variance is. **The stopping rule is licensed under a hypothesis strictly weaker than the one the error bar assumes, and the gap between the two hypotheses is populated by exactly the integrands a trading application produces.**

## An Error Bar Is a Claim About a Second Moment That May Not Exist

That gap is not a technicality about pathological functions. Losses on a fat-tailed book, waiting times until a barrier is touched, payoffs of levered or path-dependent structures, and any ratio whose denominator can approach zero all produce integrands with a finite mean and an infinite variance. The appendix has already met one: [Brownian Motion](../part-08-stochastic-processes/08-brownian-motion.md) shows that a stop-loss is hit with probability one and that "the average time until a stop is hit is not a large number; it is not a number", so a simulation estimating that average will return a value, a standard error, and no target.

```python
import numpy as np

rng = np.random.default_rng(9033)
reps = 1_000
s0, k, r, sigma, T = 100.0, 105.0, 0.03, 0.20, 1.0


def call(m):                                                   # finite mean, finite variance
    z = rng.standard_normal(m)
    return np.exp(-r * T) * np.maximum(s0 * np.exp((r - 0.5 * sigma ** 2) * T
                                                   + sigma * np.sqrt(T) * z) - k, 0.0)


def pareto(alpha):
    return lambda m: rng.random(m) ** (-1.0 / alpha)           # P(X > x) = x^-alpha on [1, inf)


print(f"  {reps} independent runs at each size, three integrands, error bars read against truth")
print("   integrand           truth          N     estimate    reported se    actual sd    ratio")
for name, f, truth in (("call payoff", call, "7.128"), ("Pareto(1.5)", pareto(1.5), "3.000"),
                       ("Pareto(0.8)", pareto(0.8), "  inf")):
    for n in (1_000, 10_000, 100_000):
        est = np.empty(reps)
        se = np.empty(reps)
        for j in range(reps):
            x = f(n)
            est[j], se[j] = x.mean(), x.std(ddof=1) / np.sqrt(n)
        print(f"  {name:<16} {truth:>7} {n:10d} {est.mean():12.3f} {se.mean():14.4f}"
              f" {est.std(ddof=1):12.4f} {est.std(ddof=1) / se.mean():8.2f}")
# =>   1000 independent runs at each size, three integrands, error bars read against truth
#       integrand           truth          N     estimate    reported se    actual sd    ratio
#      call payoff        7.128       1000        7.134         0.3964       0.3994     1.01
#      call payoff        7.128      10000        7.130         0.1254       0.1239     0.99
#      call payoff        7.128     100000        7.131         0.0396       0.0369     0.93
#      Pareto(1.5)        3.000       1000        2.998         0.3262       0.7188     2.20
#      Pareto(1.5)        3.000      10000        2.999         0.1613       0.3313     2.05
#      Pareto(1.5)        3.000     100000        2.997         0.0750       0.3210     4.28
#      Pareto(0.8)          inf       1000      197.696       176.4536    2694.5806    15.27
#      Pareto(0.8)          inf      10000      541.113       499.3152    5025.7894    10.07
#      Pareto(0.8)          inf     100000      381.811       305.2580    2379.1385     7.79
```

The first three rows are the control and they behave. For the call payoff the reported standard error and the actual run-to-run spread agree to within a few percent — ratios of $1.01$, $0.99$ and $0.93$ — which is what a finite second moment looks like from the outside.

The middle block is the one to sit with. A Pareto law with tail index $1.5$ has mean exactly $3$ and infinite variance. The estimate is *right at every sample size*: $2.998$, $2.999$, $2.997$, indistinguishable from the truth, exactly as the strong law promises, because the first moment exists and a first moment is all the strong law needs. Every convergence diagnostic anyone runs would pass. And the reported standard error falls dutifully from $0.3262$ to $0.1613$ to $0.0750$ — a factor of $4.3$, close enough to the $10$ a well-behaved integrand would deliver that nothing looks amiss — while the *actual* standard deviation of the estimate across a thousand independent runs goes $0.7188$, $0.3313$, $0.3210$ and then stops improving. The ratio of what is reported to what is true is $2.20$, then $2.05$, then $4.28$. **The estimate converges, the error bar shrinks, and the error bar becomes more wrong the longer the run goes on.**

The last block is the same disease with the first moment gone as well. A Pareto with index $0.8$ has no mean, so there is nothing to converge to, and the estimates read $197.7$, $541.1$ and $381.8$ — moving by hundreds while reporting standard errors in the hundreds and producing actual spreads in the thousands. A practitioner watching a single run of this would see a number that wanders, conclude it needs more paths, and obtain a different wandering number. There is no value of $N$ at which this run is finished, and nothing printed by it says so.

!!! warning "A standard error that shrinks like one over the square root of N is evidence that N grew, and is not evidence that the estimate improved"
    The sample standard deviation of a finite sample is finite, so $s/\sqrt N$ always exists, always decreases, and always decreases at the advertised rate — whether or not the population it is estimating has a variance at all. That makes the most-watched convergence diagnostic in simulation completely uninformative about the one failure it would need to detect. Two checks do work and neither is expensive. Run the whole simulation a handful of times on independent seeds and compare the spread of the answers against the standard error each run reported, which is what the table above does and what turns a hidden $4.28$ into something visible. And watch the integrand's own tail: track the running maximum of $|f(X_i)|$ against $N$, and if it keeps setting records at a rate that does not decay, the second moment is in doubt no matter what the sample variance says. The repair, once the diagnosis is confirmed, is not more paths — it is a different estimator, which is [Importance Sampling](04-importance-sampling.md), or a different summary, such as a median or a trimmed mean whose sampling behaviour survives the tail.

## Simulating the Null Is the Cheapest Statistic You Own

The most valuable thing a simulation produces is often not an estimate of something unknown but the distribution of something known to be worthless. A null distribution costs a few lines, requires no theory, and converts a number that looks impressive into a number with a scale beside it. The canonical case in trading is selection: a research process that tries $N$ variants and reports the best has computed a maximum, and the maximum of $N$ noise draws is not zero.

```python
import numpy as np

rng = np.random.default_rng(9037)
reps, years, sd_s = 1_000, 25, 0.20                            # se of a Sharpe over 25 years
tail = {}
print(f"  best Sharpe of a family of N worthless variants, {years} years each, {reps} families")
print("        N    independent    rho = 0.5    rho = 0.9    sqrt(2 ln N) x 0.20")
for n in (10, 100, 1_000, 10_000, 100_000):
    row = []
    for rho in (0.0, 0.5, 0.9):
        best = np.empty(reps)
        for j in range(reps):
            g = rng.standard_normal()
            best[j] = (sd_s * (np.sqrt(rho) * g
                               + np.sqrt(1 - rho) * rng.standard_normal(n))).max()
        row.append(best.mean())
        tail[(n, rho)] = np.quantile(best, 0.95)
    print(f"  {n:9d} {row[0]:14.3f} {row[1]:12.3f} {row[2]:12.3f}"
          f" {sd_s * np.sqrt(2 * np.log(n)):22.3f}")

T, n = years * 252, 1_000
best = np.empty(40)
for j in range(40):
    ret = rng.standard_normal((n, T)) * 0.01                   # pure noise, by construction
    best[j] = (np.sqrt(252) * ret.mean(axis=1) / ret.std(axis=1)).max()
print(f"  simulating {n} strategies as returns rather than as Sharpes gives {best.mean():.3f}")
print(f"  95th percentile of the best of 10000: independent {tail[(10_000, 0.0)]:.3f}"
      f"   rho = 0.9 {tail[(10_000, 0.9)]:.3f}")
# =>   best Sharpe of a family of N worthless variants, 25 years each, 1000 families
#            N    independent    rho = 0.5    rho = 0.9    sqrt(2 ln N) x 0.20
#             10          0.304        0.216        0.102                  0.429
#            100          0.508        0.361        0.158                  0.607
#           1000          0.645        0.448        0.204                  0.743
#          10000          0.770        0.547        0.238                  0.858
#         100000          0.877        0.615        0.286                  0.960
#      simulating 1000 strategies as returns rather than as Sharpes gives 0.673
#      95th percentile of the best of 10000: independent 0.887   rho = 0.9 0.545
```

The independent column reproduces the table [Distributed Backtesting](../../advanced/09-distributed-backtesting.md) computes from the closed form — $0.31$, $0.51$, $0.65$, $0.77$, $0.88$ there against $0.304$, $0.508$, $0.645$, $0.770$, $0.877$ here — and its reading stands: ten thousand dead strategies buy an expected best Sharpe of $0.77$ by luck alone, a number most investors would fund. The last column is the asymptotic $\sigma_s\sqrt{2\ln N}$, which runs about $0.1$ high at every $N$ because it is the leading term of an expansion whose next term is negative and decays only like $1/\sqrt{\ln N}$. It has the right shape and the wrong level, which is itself a good reason to simulate a null rather than quote its asymptotics.

The two middle columns are why the simulated null is worth more than the formula. A real variant grid is not independent: fifty momentum lookbacks on one asset share almost all of their returns. Impose a common factor and the expected best falls hard — at $N=10{,}000$, from $0.770$ independent to $0.547$ at correlation $0.5$ and $0.238$ at correlation $0.9$ — because selection can only exploit the idiosyncratic part of a family, and a highly correlated family barely has one. Deflating a correlated grid with the independent formula therefore *over*-corrects, by a factor of three in the last column.

The final line stops that from becoming a comfortable conclusion. The $95$th percentile of the best-of-ten-thousand is $0.887$ when the family is independent and $0.545$ when it is correlated at $0.9$ — more than twice the correlated *mean* of $0.238$. A common factor lowers the average of the maximum and raises its dispersion, because the whole family moves together, so the occasional family that draws well on the common factor produces a champion far above its own expectation. **A correlated grid is less likely to manufacture a good-looking winner and more likely to manufacture a spectacular one**, which is why the honest instrument is a resampling of the actual family rather than any formula, and is what [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) buys when its Reality Check resamples "the entire family's history … preserving each variant's correlation with its siblings".

One detail inside the block is worth naming, because it is this page's method applied to itself. Simulating a thousand strategies as actual return series and computing their Sharpes gives $0.673$, against the $0.645$ obtained by treating the Sharpes directly as normal draws. The gap is small, real, and in the expected direction: an estimated Sharpe is a ratio of two estimates and has slightly heavier tails than a normal, so the maximum of a thousand of them runs a little high. The normal shortcut is a good approximation and it is an approximation, and the only reason that is known is that both were run.

## The Number and Its Error Bar Are One Object

Three results here fit into a single working rule. The convergence rate is $N^{-1/2}$ regardless of dimension, so Monte Carlo is the only method that does not degrade as a problem grows and the slowest useful method for problems that are small. That rate is bought with a hypothesis — a finite second moment — which the simulation cannot check, and when the hypothesis fails the estimate can be perfectly correct while its stated precision is wrong by a factor of four and getting worse. And the same machinery that estimates an unknown quantity will, for the same cost, produce the distribution of a quantity known to be worthless, which is usually the more valuable output of the two.

The through-line is the one the whole part keeps returning to. A simulation reports $\hat\theta_N$ and $s/\sqrt N$, both computed from the same draws, so any defect large enough to corrupt the first is generally large enough to corrupt the second in the same direction. The arithmetic-seeding failure of [Random Number Generation](01-random-number-generation.md) had exactly that shape — a correct estimate with an error bar four times too small — and the infinite-variance failure here has it again, with the same factor arriving by an entirely different mechanism. In both cases the diagnostic that would have caught it required running the whole procedure more than once, which is the one thing a report of a single run cannot do.

What remains is the question the four-fold rule makes urgent. If halving the error costs four times the paths, then buying accuracy with compute has sharply diminishing returns, and the alternative is to buy it with structure — to sample from a different law and correct for the difference, to discard the draws that carry no information, or to lean on a related quantity whose answer is already known. That is the next three pages, and the first of them is the one that turns an intractable problem into a tractable one rather than merely a faster one: [Importance Sampling](04-importance-sampling.md), where the point is that the naive estimator spends its effort where the probability is, and the question almost never is.
