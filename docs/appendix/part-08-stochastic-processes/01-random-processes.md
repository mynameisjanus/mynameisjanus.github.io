# Random Processes

A stochastic process is not a sequence of random variables. It is a single random variable whose value is an entire function of time, and the distinction stops being pedantic the moment you notice that finance hands you exactly one draw from it — one path, one history, one realized sequence of prices — so that every statistic anyone computes is an average *along* that single draw rather than *across* the ensemble every theorem is written about.

This page covers the definition of a process through its finite-dimensional distributions, strict and covariance stationarity and the gap between them, the autocovariance function and the one property that makes a function an autocovariance at all, the ergodic theorem as the licence to read a time average as an ensemble average, the variance of a sample mean under dependence and the effective sample size it defines, and the reason a series can be indistinguishable from noise in its levels and violently dependent in its squares. It does not prove a limit theorem for independent draws, which is [Part VII](../part-07-asymptotic-theory/index.md); it does not fit, select, or test a dependence model, which is [Part XIII](../part-13-regression/index.md) and [Time Series](../../part-03-statistics/03-time-series.md); it does not simulate a path in order to estimate an integral, which is [Part IX](../part-09-monte-carlo-methods/index.md); it assumes no Markov structure, which is [Markov Chains](05-markov-chains.md); and it takes no continuous-time limit, which is [Brownian Motion](08-brownian-motion.md).

The trading stake is a single pinned number in the course. [Hypothesis Testing and Multiple Testing](../../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) tests a strategy on overlapping twenty-one-day returns and prints `effective n 335 of 6138` — a data set that presents itself as six thousand observations and behaves like three hundred — then reports a naive t-statistic of $7.40$, "a seven-sigma discovery", which deflates to $1.73$ once the dependence is accounted for. [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md) certifies that the substitution producing $7.40$ was legal. This page is where the missing $5{,}803$ observations went, and the fourth section computes the loss exactly.

## A Process Is One Draw From a Space of Functions

A **stochastic process** is a family $\{X_t\}_{t\in T}$ of random variables on a common [probability space](../part-02-probability-foundations/01-probability-spaces.md), indexed by a set $T$. When $T=\{0,1,2,\ldots\}$ the process is discrete-time and the index counts bars; when $T=[0,\infty)$ it is continuous-time and the index is a clock. Fix an outcome $\omega$ and the whole family collapses to an ordinary function $t\mapsto X_t(\omega)$, called a **sample path**. Fix a time $t$ instead and you are left with an ordinary random variable. Those two readings are the entire subject.

What specifies a process is not any one of its marginals but the collection of its **finite-dimensional distributions**: the joint law of $(X_{t_1},\ldots,X_{t_k})$ for every finite set of times. Kolmogorov's extension theorem says that any family of such laws which is *consistent* — marginalizing $(X_{t_1},X_{t_2})$ down to $X_{t_1}$ gives the same answer whichever pair you started from — is the finite-dimensional family of some process. That is a licence to define processes the way every model in this part does: by writing down how consecutive values relate, and never once exhibiting the underlying space.

The reason to insist on the path reading is that it is the only one the data supports. An ensemble statement — "the standard deviation of tomorrow's return is $1\%$" — is a claim about the marginal law at one time, and the market will show you exactly one realization of it. Every estimate that gets computed instead runs along the time axis: a sample mean of $6{,}300$ consecutive daily returns, a rolling volatility, a drawdown. Whether those two kinds of average agree is not a technicality to be waved through. It is a property called ergodicity, it is the subject of the third section, and it can fail.

## Stationarity Is Two Assumptions and Only the Weaker One Is Ever Checked

A process is **strictly stationary** if shifting the clock changes nothing: for every $k$, every set of times $t_1<\cdots<t_k$, and every lag $h$, the joint law of $(X_{t_1+h},\ldots,X_{t_k+h})$ equals that of $(X_{t_1},\ldots,X_{t_k})$. This is a statement about the whole finite-dimensional family and is essentially never verifiable.

A process is **covariance stationary** (or weakly stationary) if only the first two moments are shift-invariant: $\mathbb{E}[X_t]=\mu$ for all $t$, and $\mathrm{cov}(X_t,X_{t+k})=\gamma_k$ depends on the lag $k$ alone. The function $k\mapsto\gamma_k$ is the **autocovariance function**, and $\rho_k=\gamma_k/\gamma_0$ the **autocorrelation function**. Strict stationarity plus a finite second moment implies covariance stationarity; the converse is false, and the gap is where most of this part lives.

Not every function of $k$ can be an autocovariance. Symmetry $\gamma_{-k}=\gamma_k$ and the bound $\lvert\gamma_k\rvert\leq\gamma_0$ are immediate from [Covariance](../part-04-expectation-and-moments/04-covariance.md), but they are not sufficient. The real constraint is that $\gamma$ must be **positive semidefinite**, which is the autocovariance function's version of the requirement that a [covariance matrix](../part-06-multivariate-probability/02-covariance-matrices.md) be positive semidefinite — and it is the same requirement, applied to every finite window at once.

??? note "Proof that an autocovariance function must be positive semidefinite, and that this is its only constraint"
    Take any times $t_1,\ldots,t_n$ and any real weights $a_1,\ldots,a_n$, and form the scalar $Z=\sum_i a_i X_{t_i}$. A variance cannot be negative, so

    $$0\leq\mathrm{var}(Z)=\sum_{i=1}^{n}\sum_{j=1}^{n}a_ia_j\,\mathrm{cov}(X_{t_i},X_{t_j})=\sum_{i=1}^{n}\sum_{j=1}^{n}a_ia_j\,\gamma_{t_i-t_j}.$$

    That is exactly the statement that the matrix $\Gamma_n=[\gamma_{i-j}]_{i,j\leq n}$ — a symmetric Toeplitz matrix, constant along each diagonal — is positive semidefinite for every $n$. Herglotz's theorem supplies the converse: every positive semidefinite sequence is the autocovariance function of some covariance-stationary process, namely the one whose spectral measure is the Fourier transform of $\gamma$. So positive semidefiniteness is not one constraint among several; it is the complete characterization.

    The load-bearing hypothesis is stationarity itself, and it is spent in the step where $\mathrm{cov}(X_{t_i},X_{t_j})$ was written as $\gamma_{t_i-t_j}$. Without it the covariance depends on both times separately, the matrix is not Toeplitz, and nothing about lag $k$ can be estimated by pooling the pairs that are $k$ apart — which is what every autocorrelation estimator does. **Stationarity is what converts one path into many observations of the same quantity**, and every diagnostic downstream inherits the assumption.

!!! note "Covariance stationarity is strict stationarity only for Gaussian processes, and that equivalence is assumed far more often than it is earned"
    A Gaussian process is determined by its mean and covariance, so shift-invariance of those two determines shift-invariance of everything, and the two definitions coincide. For any other process they do not, and the difference is precisely the structure this part exists to model: a series can have a constant mean, a constant variance, and zero autocorrelation at every lag while its conditional variance wanders enormously. [Time Series](../../part-03-statistics/03-time-series.md) measures this on index returns and reports the decade standard deviations differing "by half (0.9% vs 1.4%), so strict stationarity fails for returns too" — while every covariance-based diagnostic on the same data passes. The fifth section reproduces that pattern from a generating process where the answer is known.

## A Time Average Is an Ensemble Average Only If the Process Is Ergodic

The [weak law of large numbers](../part-07-asymptotic-theory/01-weak-law-of-large-numbers.md) says a sample mean of independent draws converges to their common expectation. Nothing in it survives contact with dependence, and the replacement is not automatic: what needs proving is that the *time* average $\bar X_n=\frac1n\sum_{t=1}^{n}X_t$ along one path converges to the *ensemble* mean $\mu=\mathbb{E}[X_t]$. A process for which it does is **mean-ergodic**. The criterion is remarkably clean.

??? note "Proof that a time average converges in mean square exactly when the autocovariances vanish on average"
    Assume covariance stationarity with mean $\mu$ and autocovariances $\gamma_k$. Expanding the variance of the sample mean and counting how many pairs sit at each lag,

    $$\mathrm{var}(\bar X_n)=\frac{1}{n^{2}}\sum_{s=1}^{n}\sum_{t=1}^{n}\gamma_{s-t}=\frac{1}{n}\sum_{k=-(n-1)}^{n-1}\Bigl(1-\frac{\lvert k\rvert}{n}\Bigr)\gamma_k,$$

    because the diagonal $s=t$ contributes $n$ terms of $\gamma_0$, each off-diagonal $\lvert k\rvert$ contributes $n-\lvert k\rvert$ terms of $\gamma_k$, and dividing by $n^{2}$ produces the triangular weight. Since $\mathbb{E}[\bar X_n]=\mu$ exactly, $\mathbb{E}[(\bar X_n-\mu)^{2}]=\mathrm{var}(\bar X_n)$, and $\bar X_n\to\mu$ in mean square if and only if that expression tends to zero. The right-hand side is $1/n$ times a Cesàro average of the $\gamma_k$, so the condition is

    $$\frac{1}{n}\sum_{k=0}^{n-1}\gamma_k\ \longrightarrow\ 0.$$

    Two sufficient conditions follow immediately and cover every model in this part: $\gamma_k\to0$, and the stronger $\sum_k\lvert\gamma_k\rvert<\infty$, under which $n\,\mathrm{var}(\bar X_n)$ converges to the **long-run variance** $\gamma_0+2\sum_{k\geq1}\gamma_k$ rather than merely to zero.

    The load-bearing hypothesis is that the autocovariances decay *at all*, and the failure mode is visible in the formula rather than hidden in it. If $\gamma_k\to c>0$ — every pair of times, however far apart, retaining a fixed common component — the Cesàro average converges to $c$, the variance of the sample mean converges to $c$ instead of to zero, and the time average never settles on $\mu$ no matter how long the path is run.

That failure is not a pathology invented to make a theorem look sharp. It is what happens whenever a process contains a component drawn once and then held fixed forever, which is the mathematical shape of a structural break, a permanently altered fee schedule, or a regime the market entered and never left. Two processes with *identical* $\mathcal{N}(0,1)$ marginals — indistinguishable in any histogram, any QQ plot, any moment test — can differ entirely in whether their time averages converge to the right number.

```python
import numpy as np

rng = np.random.default_rng(8013)
reps, chunk, tau = 20_000, 1_000, np.sqrt(0.5)
print(f"  two processes with identical N(0,1) marginals, {reps} paths, running sample mean")
print("          n    ergodic sd(xbar)    1/sqrt(n)    non-ergodic sd(xbar)    sd of frozen level")
level = rng.standard_normal(reps) * tau                        # drawn once per path, then frozen
se, sn, t = np.zeros(reps), np.zeros(reps), 0
for _ in range(20):
    z = rng.standard_normal((reps, chunk))
    se += z.sum(axis=1)                                        # ergodic: pure iid noise
    sn += np.sqrt(1 - tau ** 2) * z.sum(axis=1) + chunk * level
    t += chunk
    if t in (1_000, 2_000, 5_000, 10_000, 20_000):
        print(f"  {t:10d} {(se / t).std(ddof=1):19.5f} {1 / np.sqrt(t):12.5f}"
              f" {(sn / t).std(ddof=1):23.5f} {tau:23.5f}")
# =>   two processes with identical N(0,1) marginals, 20000 paths, running sample mean
#              n    ergodic sd(xbar)    1/sqrt(n)    non-ergodic sd(xbar)    sd of frozen level
#            1000             0.03157      0.03162                 0.70680                 0.70711
#            2000             0.02243      0.02236                 0.70655                 0.70711
#            5000             0.01413      0.01414                 0.70638                 0.70711
#           10000             0.01002      0.01000                 0.70631                 0.70711
#           20000             0.00709      0.00707                 0.70625                 0.70711
```

The ergodic column does what twenty years of statistical training expects. Its spread falls $0.03157\to0.02243\to0.01413\to0.01002\to0.00709$, tracking $1/\sqrt n$ to three decimals at every checkpoint, and a path of twenty thousand observations pins the mean to within a hundredth of a standard deviation.

The non-ergodic column does not move. $0.70680$, $0.70655$, $0.70638$, $0.70631$, $0.70625$ — five values across a twentyfold increase in sample size, all of them sitting on the standard deviation of the frozen level, $0.70711$, and none of them showing the faintest tendency to shrink. **The sample mean is converging perfectly well; it is converging to the level this particular path drew, which is not the ensemble mean and never will be.** Its autocovariance is $\gamma_k=\tau^{2}=0.5$ for every $k\geq1$, so the Cesàro average converges to $0.5$ rather than $0$, and the proof above predicted the number $\sqrt{0.5}=0.70711$ that the last column prints.

The uncomfortable part is the diagnostic. From a *single* path — which is all anyone has — the non-ergodic process looks like a stationary series that happens to have a nonzero mean. Nothing in its histogram, its autocorrelation estimate, or its unit-root test distinguishes "this market's long-run mean is $0.4$" from "this market's long-run mean is $0$ and this path drew $0.4$." Ergodicity is the assumption that makes the first reading illegal, it is never tested, and where it fails it fails silently.

## Dependence Does Not Bias the Mean, It Deletes Observations

Grant ergodicity and the time average converges. The next question is how fast, and the proof of the second section already answered it: with summable autocovariances, $\mathrm{var}(\bar X_n)\approx\frac{\gamma_0}{n}\bigl(1+2\sum_{k\geq1}\rho_k\bigr)$. The bracket is a pure multiplier on the independent-case answer, so it is natural to move it into the sample size and define the **effective sample size**

$$n_{\text{eff}}=\frac{n}{1+2\sum_{k\geq1}\rho_k},$$

the number of independent observations that would have delivered the same precision. The denominator is the integrated autocorrelation time, and where the series is a sampler's output rather than a market's it becomes the exchange rate every MCMC diagnostic is quoted in, which is [Markov Chain Monte Carlo](../part-17-statistical-computing/04-markov-chain-monte-carlo.md). For an AR(1) with $X_t=\rho X_{t-1}+\varepsilon_t$ the autocorrelations are $\rho_k=\rho^{k}$, the geometric series of [Sequences and Infinite Series](../part-01-mathematical-foundations/04-sequences-and-series.md) sums to $1+2\rho/(1-\rho)=(1+\rho)/(1-\rho)$, and the ratio collapses to a single number worth memorizing:

$$\frac{n_{\text{eff}}}{n}=\frac{1-\rho}{1+\rho}.$$

```python
import numpy as np

rng = np.random.default_rng(8011)
reps, n = 20_000, 1_260
print(f"  sample mean of a stationary AR(1), {reps} paths of {n} steps, unit marginal variance")
print("       rho    sd(xbar)    iid sd    inflation      n_eff    n_eff / n    (1-rho)/(1+rho)")
for rho in (0.00, 0.20, 0.50, 0.80, 0.95):
    x = rng.standard_normal((reps, n))
    x[:, 1:] *= np.sqrt(1 - rho ** 2)                          # innovations of the stationary law
    for t in range(1, n):
        x[:, t] += rho * x[:, t - 1]
    m = x.mean(axis=1)
    sd, iid = m.std(ddof=1), 1 / np.sqrt(n)
    print(f"  {rho:8.2f} {sd:11.5f} {iid:9.5f} {sd / iid:12.4f} {n * (iid / sd) ** 2:10.1f}"
          f" {(iid / sd) ** 2:12.4f} {(1 - rho) / (1 + rho):18.4f}")
# =>   sample mean of a stationary AR(1), 20000 paths of 1260 steps, unit marginal variance
#           rho    sd(xbar)    iid sd    inflation      n_eff    n_eff / n    (1-rho)/(1+rho)
#          0.00     0.02821   0.02817       1.0015     1256.3       0.9970             1.0000
#          0.20     0.03465   0.02817       1.2298      833.1       0.6612             0.6667
#          0.50     0.04896   0.02817       1.7381      417.1       0.3310             0.3333
#          0.80     0.08466   0.02817       3.0051      139.5       0.1107             0.1111
#          0.95     0.17471   0.02817       6.2016       32.8       0.0260             0.0256
```

The last two columns are the theory and the simulation, and they agree to the third decimal at every row: $0.6612$ against $0.6667$, $0.3310$ against $0.3333$, $0.1107$ against $0.1111$, $0.0260$ against $0.0256$. Five years of daily data is $1{,}260$ observations. At $\rho=0.95$ it is worth $32.8$.

Read the inflation column as the thing it actually is — the factor by which every standard error on the page is wrong if the dependence is ignored. At $\rho=0.5$, a modest and entirely plausible persistence, error bars are too narrow by $1.7381$, so a $t$ of $2.0$ is really a $t$ of $1.15$ and a nominal $5\%$ test rejects far more often than it claims. At $\rho=0.8$ the factor is $3.0051$. **Dependence does not push the estimate anywhere; the sample mean stays unbiased at every row. It removes observations, and it removes them from the denominator of a standard error where nobody is looking.**

This is the whole mechanism behind the course's `effective n 335 of 6138`. Twenty-one-day overlapping returns are a moving sum, so consecutive observations share twenty of their twenty-one daily terms and the lag-one autocorrelation is close to $0.95$ by construction — the last row of this table. The published ratio $335/6138=0.0546$ sits between this table's $0.0260$ and the next persistence rung, exactly where a triangular-weighted sum of twenty-one overlapping terms puts it. Nothing about that data set was fraudulent or unusual. It was counted as though the fourth column read $1.0000$.

## The Memory Is in the Magnitude, So the Error Bar That Breaks Is the Risk One

The AR(1) above puts the dependence in the level, which makes it easy to see and easy to test. Financial returns do something more awkward: they are close to uncorrelated in the level and strongly dependent in the square. A process built that way passes every linear diagnostic and violates independence comprehensively, and the practical consequence is not the one people expect.

```python
import numpy as np

rng = np.random.default_rng(8017)
reps, n, w, a, b = 2_000, 6_300, 0.01, 0.09, 0.90
print(f"  GARCH(1,1) with alpha {a}, beta {b}; {reps} histories of {n} days")
print("       lag    mean acf of r    mean acf of r^2")
r = np.empty((reps, n))
h = np.full(reps, w / (1 - a - b))
z = rng.standard_normal((reps, n))
for t in range(n):
    r[:, t] = np.sqrt(h) * z[:, t]
    h = w + a * r[:, t] ** 2 + b * h


def acf(y, k):                                                 # per-history lag-k autocorrelation
    e = y - y.mean(axis=1, keepdims=True)
    return ((e[:, k:] * e[:, :-k]).mean(axis=1) / (e * e).mean(axis=1)).mean()


def frac(y):                                                   # n_eff / n from 100 lags
    return 1 / (1 + 2 * sum(acf(y, k) for k in range(1, 101)))


for k in (1, 2, 5, 10, 20, 50):
    print(f"  {k:8d} {acf(r, k):16.4f} {acf(r ** 2, k):18.4f}")
print(f"  n_eff / n for the mean {frac(r):.4f}, for the variance {frac(r ** 2):.4f}")
# =>   GARCH(1,1) with alpha 0.09, beta 0.9; 2000 histories of 6300 days
#           lag    mean acf of r    mean acf of r^2
#             1           0.0000             0.2493
#             2          -0.0006             0.2476
#             5           0.0000             0.2344
#            10           0.0000             0.2175
#            20          -0.0003             0.1876
#            50          -0.0002             0.1181
#      n_eff / n for the mean 1.0263, for the variance 0.0373
```

The middle column is a wall of zeros. $0.0000$, $-0.0006$, $0.0000$, $0.0000$, $-0.0003$, $-0.0002$ — the returns are serially uncorrelated to four decimals at every lag out to fifty, which is what the generating process guarantees, since each day's return is a scale times an independent standard normal. Any autocorrelation test, any Ljung–Box statistic, any variance-ratio test on the levels will pass this data.

The right column is the same process seen through its squares, and it is not close: $0.2493$ at lag one, still $0.1181$ at lag fifty, decaying at the $\alpha+\beta=0.99$ rate that gives volatility clustering its name. This is the exact pattern [Time Series](../../part-03-statistics/03-time-series.md) measures on real index returns — "returns clear the ACF significance band only at lag one and unstably; squared returns are massively autocorrelated at every lag" — reproduced here from a generator where there is no doubt about what is true.

The last line is the payoff and it runs in the opposite direction to intuition. For the **mean**, $n_{\text{eff}}/n$ is $1.0263$: the dependence costs nothing at all, because the mean's precision depends on the autocorrelation of the levels, and there isn't any. For the **variance**, $n_{\text{eff}}/n$ is $0.0373$ — $6{,}300$ days of data carry the information of $235$ independent ones, and every standard error on a volatility, a VaR, or a covariance matrix estimated from this series is too narrow by a factor of $\sqrt{1/0.0373}=5.2$. **The one statistic that finance can measure precisely is the one whose error bar this process quietly destroys**, and it destroys it in a way no diagnostic run on the returns themselves can see.

!!! warning "A passing test for autocorrelation in returns is not evidence of independence, and the two are conflated in almost every backtest report"
    Uncorrelatedness constrains the second joint moment and nothing else. Independence constrains every joint moment, and the gap between them is exactly the volatility clustering above — the single most robust empirical fact about asset returns, and the one that makes a Ljung–Box statistic on returns a test of almost nothing. The practical rule is that the diagnostic has to be run on the transform that the statistic being defended actually depends on: on the levels if you are quoting a mean or a Sharpe, on the squares if you are quoting a volatility or a VaR, on the products if you are quoting a correlation. [Returns and Their Distributions](../../part-03-statistics/02-returns-and-distributions.md) documents the empirical version; [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md) shows what happens downstream when the wrong long-run variance is substituted anyway.

## Every Statistic Computed Along a Path Has Two Sample Sizes

The number of rows in the data frame is not a sample size. It is an upper bound on one, and every model in the rest of this part is a specific proposal for how much smaller the real one is: a Markov chain says the effective count is governed by the second-largest eigenvalue of its transition matrix, a Poisson process says the counts in disjoint windows really are independent so the two numbers coincide, and a random walk says the level carries no information at all and only the increments count.

Three questions separate the failures, and each of the last three sections is one of them. Does the time average converge to the ensemble quantity at all, or does the process carry a component drawn once and frozen — a question about the model's construction that no single path can answer, and the one place where more data is genuinely no help. If it converges, how many independent observations is the path worth, which is $1+2\sum_k\rho_k$ and is usually the difference between a result and a rounding error. And is the autocorrelation being measured the autocorrelation of the quantity whose error bar is being quoted, since a series can be clean in the level and hopeless in the square.

None of this changes a point estimate. That is what makes it dangerous. A mean, a volatility, and a Sharpe ratio computed on dependent data are all still consistent, still converging, still correct in the limit the theory describes; only the *precision* attached to them is fictitious, and precision is what every decision downstream is actually made on. The course's $7.40$ was never a statement about a strategy. It was a statement about a denominator, computed as though six thousand numbers had been six thousand facts.
