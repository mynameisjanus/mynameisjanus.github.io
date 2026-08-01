# Exponentials, Logarithms, and Growth

Compounding is multiplication, and logarithms turn multiplication into addition. That single sentence explains why log returns exist, why likelihoods are maximized in log space, why volatility drag is subtracted rather than ignored, and why prices are modeled as exponentials of normal variables rather than as normal variables. This page derives those results from the function $e^x$ and its inverse, and then applies them to the quantities the course reports constantly: annualized returns, volatility scaled by $\sqrt{252}$, half-lives, and compound growth.

Almost everything here is a consequence of the Taylor expansions in [Calculus Essentials](06-calculus-essentials.md) and the geometric series in [Sequences and Infinite Series](04-sequences-and-series.md).

## The Number $e$

Three definitions, all the same number:

$$e = \lim_{n\to\infty}\left(1+\frac{1}{n}\right)^{n} = \sum_{k=0}^{\infty}\frac{1}{k!} = \text{the unique } a>0 \text{ with } \frac{d}{dx}a^x = a^x.$$

The first is the compounding definition — one unit of interest at 100%, split into $n$ payments and reinvested. The second is the Taylor series of $e^x$ at $x=1$ and is how the number is actually computed, converging fast enough to give 15 digits in 18 terms. The third is the property that makes $e$ the natural base for calculus: it is the exponential that is its own derivative, so nothing extra appears when it is differentiated.

The limit definition converges slowly, which is worth seeing because it is the mathematical form of "compounding more frequently buys less than you would think":

```python
import math

for n in (1, 10, 100, 10_000, 1_000_000):
    print(f"n={n:<9} (1 + 1/n)^n = {(1 + 1 / n) ** n:.10f}")
print(f"{'e':<11} = {math.e:.10f}")
# => n=1         (1 + 1/n)^n = 2.0000000000
#    n=10        (1 + 1/n)^n = 2.5937424601
#    n=100       (1 + 1/n)^n = 2.7048138294
#    n=10000     (1 + 1/n)^n = 2.7181459268
#    n=1000000   (1 + 1/n)^n = 2.7182804691
#    e           = 2.7182818285
```

Each factor of 100 in compounding frequency buys about two more correct digits — first-order convergence, exactly the $O(1/n)$ rate of the truncated series.

### Exponential and Logarithm Laws

| Exponential | Logarithm |
|---|---|
| $e^{x+y} = e^xe^y$ | $\log(xy) = \log x + \log y$ |
| $e^{x-y} = e^x/e^y$ | $\log(x/y) = \log x - \log y$ |
| $(e^x)^a = e^{ax}$ | $\log(x^a) = a\log x$ |
| $e^0 = 1$ | $\log 1 = 0$ |
| $e^x>0$ always | $\log x$ defined only for $x>0$ |

The logarithm is the inverse: $\log(e^x) = x$ and $e^{\log x} = x$ for $x>0$. Bases convert by $\log_b x = \log x/\log b$. As stated in [Mathematical Notation](03-mathematical-notation.md), $\log$ in this book means the natural logarithm unless a base is written.

The row that matters most is the first. Turning products into sums is not an algebraic nicety — it is the only way to compute with many small numbers at once.

### Why Likelihoods Live in Log Space

A likelihood is a product of thousands of probabilities or densities. Doubles underflow to zero long before the product is finished:

```python
import numpy as np

rng = np.random.default_rng(11)
p = rng.uniform(0.05, 0.5, 5000)      # 5000 per-observation likelihood factors

print("direct product:", np.prod(p))
print(f"sum of logs:    {np.log(p).sum():.4f}")
# => direct product: 0.0
#    sum of logs:    -7186.5329
```

The product is not small; it is $e^{-7186.5}$, a number about $10^{-3121}$ that no float can hold. The log-likelihood is an ordinary number that any optimizer can work with, and since $\log$ is strictly increasing, maximizing it maximizes the likelihood itself — the argmax is unchanged, only the arithmetic is survivable. This is why every estimation routine in the book, from [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md) to the forward algorithm in [Hidden Markov Models](../part-08-stochastic-processes/07-hidden-markov-models.md), works with sums of logs. When probabilities must be added rather than multiplied in log space, the log-sum-exp trick does it without leaving: $\log\sum_i e^{a_i} = a^* + \log\sum_i e^{a_i - a^*}$ with $a^* = \max_i a_i$.

## Compounding

Invest $P$ at annual rate $R$, compounded $m$ times a year for $t$ years:

$$A = P\left(1+\frac{R}{m}\right)^{mt}.$$

Letting $m\to\infty$ and applying the limit definition of $e$ gives **continuous compounding**:

$$A = Pe^{Rt}.$$

The **effective annual yield** is what a rate actually pays over a year once compounding is included, $\text{APY} = (1+R/m)^m - 1$, against the quoted $\text{APR} = R$:

```python
apr = 0.06

for m, label in ((1, "annual"), (12, "monthly"), (365, "daily")):
    print(f"{label:<11} {(1 + apr / m) ** m - 1:.8f}")
print(f"{'continuous':<11} {math.exp(apr) - 1:.8f}")
# => annual      0.06000000
#    monthly     0.06167781
#    daily       0.06183131
#    continuous  0.06183655
```

Continuous compounding is the ceiling, not a different regime: the gap between daily and continuous is half a basis point on a 6% rate. The practical significance is that a continuously compounded rate is a *log* rate — $A = Pe^{Rt}$ means $R = \frac{1}{t}\log(A/P)$ — so the natural way to quote growth is already the log return of the next section.

## Simple Returns and Log Returns

Two definitions of "the return", both correct, neither interchangeable:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1 \qquad\text{(simple)},$$

$$\ell_t = \log\frac{P_t}{P_{t-1}} = \log(1 + r_t)\qquad\text{(log)}.$$

Each is convenient for exactly one thing, and the choice is dictated by which:

|  | Simple returns | Log returns |
|---|---|---|
| Aggregate across **time** | Multiplicative: $\prod_t(1+r_t) - 1$ | **Additive**: $\sum_t \ell_t$ |
| Aggregate across **assets** | **Additive**: $r_p = \sum_i w_i r_i$ | Not additive |
| Range | $\ge -1$ | Any real number |
| Symmetry | $-50\%$ needs $+100\%$ to recover | $-0.69$ needs $+0.69$ |

Portfolio arithmetic requires simple returns: a portfolio's simple return is the weighted average of its constituents' simple returns, and no such identity holds for logs. Time aggregation requires log returns: ten years of daily log returns sum to the total log return, so cumulative performance is a `cumsum` rather than a `cumprod`.

```python
rng = np.random.default_rng(3)
r = rng.normal(0.0004, 0.012, 2520)          # 10 years of daily simple returns
log_r = np.log1p(r)

print(f"terminal wealth, prod(1+r)   {np.prod(1 + r):.6f}")
print(f"terminal wealth, exp(sum l)  {math.exp(log_r.sum()):.6f}")
print(f"sum of simple returns        {r.sum():.6f}")
print(f"sum of log returns           {log_r.sum():.6f}")
# => terminal wealth, prod(1+r)   4.811004
#    terminal wealth, exp(sum l)  4.811004
#    sum of simple returns        1.751414
#    sum of log returns           1.570906
```

The first two lines are the same number, as they must be. The last two are the trap: adding up ten years of *simple* daily returns gives 175%, while the position actually gained 381%. The sum of simple returns is not any return at all — it is a quantity with no financial meaning that nonetheless appears in reports, because summing a column is easy.

Use `np.log1p(r)` rather than `np.log(1 + r)`. For the small returns that dominate daily data, forming $1+r$ first discards precision in exactly the digits that matter, and `log1p` is built to avoid it.

### The Approximation and Where It Breaks

From the Taylor series, for small $r$,

$$\log(1+r) = r - \frac{r^2}{2} + \frac{r^3}{3} - \cdots \approx r - \frac{r^2}{2}.$$

The error table in [Calculus Essentials](06-calculus-essentials.md) quantifies it: at $r = 1\%$ daily the first-order approximation $\ell\approx r$ is off by 5 basis points *of the return*, at 5% it is off by 12 bp, and at 20% it is off by 1.8 percentage points. So the two conventions are interchangeable for daily equity data and are not interchangeable for monthly data, for crypto, for leveraged strategies, or for anything with fat tails. The safest habit is to convert explicitly at the boundary — store simple returns, convert to logs for time aggregation, convert back for reporting — and to never let a plot's axis label leave the convention ambiguous. [Returns and Distributions](../../part-03-statistics/02-returns-and-distributions.md) works through the empirical consequences.

## Volatility Drag

The second-order term in that expansion is not a rounding error. It is a systematic cost, and it is the most consequential piece of arithmetic on this page.

Take log returns $\ell_t$ that are iid with mean $m$ and variance $s^2$. Over $T$ periods the total log return is $\sum_t\ell_t$, so the **compound growth rate** is $m$ — the mean of the *logs*. The **arithmetic mean of simple returns** is a different number: taking expectations of $1 + r = e^{\ell}$ with $\ell$ normal gives $\mathbb{E}[1+r] = e^{m + s^2/2}$, so

$$\mu \approx m + \frac{\sigma^2}{2} \quad\Longleftrightarrow\quad \underbrace{g}_{\text{compound}} \approx \underbrace{\mu}_{\text{arithmetic}} - \frac{\sigma^2}{2}.$$

The gap $\sigma^2/2$ is **volatility drag**: the amount by which a volatile investment's compound growth falls short of its average return. It is not a fee or a friction — it is the mathematical fact that gains and losses compound multiplicatively while averages add. A $-50\%$ followed by a $+50\%$ averages zero and leaves you down 25%.

At a 40% annualized volatility, the drag is $0.4^2/2 = 8$ percentage points a year:

```python
mu, sigma = 0.10, 0.40
years, paths = 30, 5_000
dt = 1 / 252

rng = np.random.default_rng(42)
r = rng.normal(mu * dt, sigma * math.sqrt(dt), (paths, years * 252))
log_growth = np.log1p(r).sum(axis=1) / years

print(f"arithmetic mean, annualized  {r.mean() / dt: .4f}")
print(f"median compound growth       {np.median(log_growth): .4f}")
print(f"mu - sigma^2/2               {mu - sigma ** 2 / 2: .4f}")
print(f"paths that lost money in 30y {(log_growth < 0).mean(): .4f}")
# => arithmetic mean, annualized   0.1003
#    median compound growth        0.0198
#    mu - sigma^2/2                0.0200
#    paths that lost money in 30y  0.3884
```

A strategy with a genuine 10% expected return compounds at 2%, and after thirty years, 39% of its paths have lost money. Nothing about the simulation is adversarial — the returns are normal, the mean is positive every single day, and there are no costs. The volatility alone eats four fifths of the edge.

Three consequences run through the rest of the course:

- **Reported averages must say which mean.** An arithmetic mean of monthly returns overstates what an investor experienced; the geometric mean (CAGR) is what compounded. The gap is $\sigma^2/2$, and it widens quadratically with volatility.
- **Leverage has an optimum.** Scaling a strategy by $k$ multiplies $\mu$ by $k$ and $\sigma^2$ by $k^2$, so growth is $k\mu - k^2\sigma^2/2$ — a downward parabola maximized at $k^* = \mu/\sigma^2$. That is the [Kelly criterion](../part-18-quant-finance-applications/01-kelly-criterion.md), derived here as a one-line consequence of drag, and it is why [Kelly, Volatility Targeting, and Leverage](../../part-08-portfolio-management/02-kelly-vol-targeting-leverage.md) treats leverage past $k^*$ as strictly value-destroying rather than merely riskier.
- **Volatility reduction is return enhancement.** Halving volatility at constant arithmetic mean raises compound growth by $\tfrac{3}{8}\sigma^2$, which is why minimum-variance and volatility-targeted portfolios can beat higher-returning ones on realized wealth.

## Square-Root-of-Time Scaling

Volatility is quoted annually and estimated daily, and the bridge is the $\sqrt{T}$ rule:

$$\sigma_{\text{annual}} = \sigma_{\text{daily}}\sqrt{252}.$$

The derivation is short and the assumption it rests on is strict. If daily returns $\ell_1,\ldots,\ell_T$ are **uncorrelated** with common variance $\sigma^2$, then the variance of their sum is the sum of variances:

$$\mathrm{var}\left(\sum_{t=1}^{T}\ell_t\right) = T\sigma^2 \implies \mathrm{sd}\left(\sum_{t=1}^{T}\ell_t\right) = \sigma\sqrt{T}.$$

Variance scales linearly with time; standard deviation scales with its square root. Mean return scales linearly too, which is why the Sharpe ratio picks up a factor $\sqrt{T}$ rather than $T$ — and why annualizing a monthly Sharpe with $\sqrt{252}$, an error documented in [Performance Metrics and Reporting](../../part-05-backtesting-engine/04-performance-metrics-and-reporting.md), inflates it by $\sqrt{21}$.

Autocorrelation breaks the rule, because the cross terms in the variance of a sum no longer vanish. For an AR(1) process with coefficient $\phi$, the long-horizon variance is inflated by roughly $(1+\phi)/(1-\phi)$:

```python
def agg(x, k):
    return x[: len(x) // k * k].reshape(-1, k).sum(axis=1)

rng = np.random.default_rng(5)
n = 252 * 100
iid = rng.normal(0, 0.01, n)

eps = rng.normal(0, 0.01, n)
ar = np.empty(n)
ar[0] = eps[0]
for i in range(1, n):
    ar[i] = 0.3 * ar[i - 1] + eps[i]          # positive autocorrelation

for name, x in (("iid", iid), ("AR(1), phi=0.3", ar)):
    realized = agg(x, 21).std(ddof=1)
    scaled = math.sqrt(21) * x.std(ddof=1)
    print(f"{name:<15} realized 21d vol {realized:.5f}   sqrt(21) rule {scaled:.5f}"
          f"   ratio {realized / scaled:.3f}")
print(f"theory for AR(1): {math.sqrt((1 + 0.3) / (1 - 0.3)):.3f}")
# => iid             realized 21d vol 0.04588   sqrt(21) rule 0.04594   ratio 0.999
#    AR(1), phi=0.3  realized 21d vol 0.06331   sqrt(21) rule 0.04766   ratio 1.328
#    theory for AR(1): 1.363
```

For iid returns the rule is exact to three decimals. With mild positive autocorrelation the true monthly volatility is 33% higher than the rule reports — a risk understatement of the kind that does not show up until it matters. Positive autocorrelation (trending, illiquid, or stale-priced assets) means $\sqrt{T}$ *understates* long-horizon risk; negative autocorrelation (mean-reverting series) means it overstates. The diagnostic is free: compare realized multi-day volatility against the scaled figure, and treat a large ratio as evidence about the return process rather than as a nuisance. The same comparison appears as a Sharpe-ratio discrepancy in [Performance Metrics and Reporting](../../part-05-backtesting-engine/04-performance-metrics-and-reporting.md) and as an autocorrelation test in [Time Series](../../part-03-statistics/03-time-series.md).

## Half-Lives and Decay

Exponential decay is the mirror of exponential growth, and it parameterizes every "how fast does this forget" question in the course. A quantity decaying as $x_k = x_0\lambda^k$ with $0<\lambda<1$ reaches half its value after

$$k_{1/2} = \frac{\log(1/2)}{\log\lambda},$$

obtained by solving $\lambda^k = 1/2$. The relationships worth having memorized, all derived from the geometric series in [Sequences and Infinite Series](04-sequences-and-series.md):

| Quantity | In terms of $\lambda$ |
|---|---|
| Half-life | $\log(0.5)/\log\lambda$ |
| Centre of mass (effective window) | $\lambda/(1-\lambda)$ |
| Weight retained after $k$ periods | $\lambda^{k}$ |

In continuous time the same statement is $x(t) = x_0e^{-\kappa t}$ with half-life $\log 2/\kappa$, and $\kappa$ is the mean-reversion speed of an Ornstein–Uhlenbeck process. Estimating $\kappa$ from a fitted AR(1) coefficient — $\kappa = -\log\phi$ per period — is the standard route to "this spread mean-reverts with a half-life of eleven days", the number that decides whether a pairs trade is tradeable after costs ([Mean Reversion and Pairs Trading](../../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md)).

## Why Prices Are Lognormal

Put the pieces together. Prices evolve multiplicatively, so

$$P_T = P_0\prod_{t=1}^{T}(1+r_t) = P_0\exp\left(\sum_{t=1}^{T}\ell_t\right).$$

If the log returns $\ell_t$ are iid with finite variance, the [Central Limit Theorem](../part-07-asymptotic-theory/03-central-limit-theorem.md) makes their sum approximately normal — so the *log price* is approximately normal, and the price itself is **lognormal**. This is not merely a convenient assumption; it is what multiplicative accumulation plus a CLT delivers.

The consequences line up with what prices actually do. A lognormal price is positive by construction, since $e^x>0$ for every $x$ — the property a normal model for prices notoriously lacks. Its distribution is right-skewed, matching the asymmetry between an asset that can double many times over and one that can only fall to zero. Its median is $P_0e^{mT}$ while its mean is $P_0e^{(m+s^2/2)T}$, and the gap between them is volatility drag appearing for the third time on this page: the average outcome is dragged upward by a thin tail of very large values that the typical path never sees.

The continuous-time limit of this construction is geometric Brownian motion, $dP = \mu P\,dt + \sigma P\,dW$, whose solution $P_t = P_0\exp\big((\mu-\sigma^2/2)t + \sigma W_t\big)$ carries the drag term explicitly in its drift. That is developed in [Geometric Brownian Motion](../part-08-stochastic-processes/09-geometric-brownian-motion.md), and the distribution itself in [Lognormal Distribution](../part-05-common-distributions/17-lognormal-distribution.md). The real market deviates from the model in ways the course spends considerable time on — returns are fat-tailed, volatility clusters, and the iid assumption fails at every horizon — but the deviations are best understood as departures from this baseline, which is why the baseline is worth deriving carefully.
