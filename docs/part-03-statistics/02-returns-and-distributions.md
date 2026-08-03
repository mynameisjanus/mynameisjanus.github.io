# Returns and Their Distributions

In August 2007, Goldman Sachs's CFO explained a bad week by saying the firm was seeing "25-standard-deviation moves, several days in a row." Under a normal distribution, a single 25-sigma event has probability around $10^{-137}$ — you would not expect one in the lifetime of the universe, let alone several before Friday. The moves were real; the thing that failed was the model that called them 25-sigma. Quoting sigma counts from a Gaussian fitted to fat-tailed data is not a description of the market, it is a confession about your distributional assumptions.

This lesson makes those assumptions explicit and then tests them. It pins down the two definitions of return and where each one is the correct choice, documents the stylized facts of real returns as measured numbers rather than folklore, quantifies exactly how badly the normal fails on twenty-five years of SPY, and fits the heavier-tailed families — Student-t, normal inverse Gaussian, stable — that fail less. Every block loads the cache built in [Probability and Random Variables](01-probability-and-random-variables.md).

## Two definitions of return

There are two returns for every price move, and they answer different questions:

$$
R_t = \frac{P_t}{P_{t-1}} - 1,
\qquad
r_t = \ln\!\frac{P_t}{P_{t-1}} = \ln(1 + R_t).
$$

The simple return $R_t$ is what your account statement means by a return — the fractional change in money. The log return $r_t$ is the continuously-compounded rate that produces the same move. For small moves they are nearly identical ($\ln(1+x) \approx x$); for violent days they visibly part ways. The appendix derives the relationship, the size of the approximation error, and the volatility drag that separates arithmetic from compound returns in [Exponentials, Logarithms, and Growth](../appendix/part-01-mathematical-foundations/07-exponentials-logarithms-growth.md).

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
spy = px["SPY"]
R = spy.pct_change().dropna()
r = np.log(spy).diff().dropna()

big = R.idxmax()
print(f"largest up day {big.date()}: simple {R[big]:+.4f}, log {r[big]:+.4f}")
# => largest up day 2008-10-13: simple +0.1452, log +0.1356
med = (R.abs() - R.abs().median()).abs().idxmin()
print(f"median-size day {med.date()}: simple {R[med]:+.6f}, log {r[med]:+.6f}")
# => median-size day 2015-01-23: simple -0.005483, log -0.005498
```

On the median day the two agree to the fourth decimal; on the biggest day of 2008 they differ by a full percentage point. Neither is "the" return — the mistake is using one where the other's algebra applies, which is what the next section is about.

## Aggregation: time compounds, portfolios sum

The entire reason both definitions survive is that each makes a different aggregation exact. Log returns **add across time**: a monthly return is the sum of its daily log returns, exponentiated. Simple returns **add across a portfolio**: a portfolio's return is the value-weighted sum of its holdings' simple returns. Each identity fails if you swap definitions:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
spy = px["SPY"]
R = spy.pct_change().dropna()
r = np.log(spy).diff().dropna()

monthly_simple = (1 + R).resample("ME").prod() - 1
monthly_log = np.exp(r.resample("ME").sum()) - 1
print(np.allclose(monthly_simple.iloc[1:], monthly_log.iloc[1:]))  # => True

Rb = px[["SPY", "TLT"]].dropna().pct_change().dropna()
rb = np.log(px[["SPY", "TLT"]].dropna()).diff().dropna()
d = "2008-10-13"
print(f"true 60/40 return    {0.6 * Rb.loc[d, 'SPY'] + 0.4 * Rb.loc[d, 'TLT']:+.4f}")
print(f"weighted log returns {0.6 * rb.loc[d, 'SPY'] + 0.4 * rb.loc[d, 'TLT']:+.4f}")
# => true 60/40 return    +0.0820
#    weighted log returns +0.0762
```

The `allclose` confirms the time identity to machine precision — sum-of-logs and product-of-simples are the same monthly number, which is why [NumPy and Vectorization](../part-02-python/01-numpy-and-vectorization.md) pushed compounding into log space. The 60/40 example shows the portfolio identity breaking the other way: weighting *log* returns understates the true portfolio return by 58 basis points on one day, because the log of a sum is not the sum of logs (the general machinery for what happens to distributions under transformations is in [Functions of Random Variables](../appendix/part-03-random-variables/08-functions-of-random-variables.md), and the density-level version — where the exponential's Jacobian is exactly what separates the median price ratio from the mean one — is [Change of Variables](../appendix/part-03-random-variables/09-change-of-variables.md)). The working rules:

| Operation | Use simple | Use log |
|---|---|---|
| Compound one asset over time | — | sum, then `exp` |
| Aggregate a portfolio on one day | value-weighted sum | — |
| Statistical modeling of one series | — | additivity plays well with theory |
| Back-of-envelope on small moves | either | either — they agree to $O(x^2)$ |

## Stylized facts, measured not asserted

Across markets, decades, and asset classes, daily returns keep exhibiting the same three regularities — the *stylized facts*. On the cached SPY series they are one code block, not a literature review:

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

print(f"skew {stats.skew(r):+.2f}  excess kurt {stats.kurtosis(r):.1f}")
# => skew -0.20  excess kurt 11.4
a_r, a_abs = acf(r, nlags=20), acf(r.abs(), nlags=20)
for lag in [1, 5, 20]:
    print(f"lag {lag:2d}: acf(r) {a_r[lag]:+.3f}   acf(|r|) {a_abs[lag]:+.3f}")
# => lag  1: acf(r) -0.086   acf(|r|) +0.294
#    lag  5: acf(r) -0.011   acf(|r|) +0.339
#    lag 20: acf(r) +0.004   acf(|r|) +0.224
```

Fact one, **fat tails**: excess kurtosis over 11 against the normal's 0 — the previous lesson's headline number. Fact two, **mild asymmetry**: skew around −0.2, real but small, and the least stable moment across subperiods. Fact three, **volatility clustering**, and this is the pair of columns to stare at: returns themselves are nearly uncorrelated (−0.086 at lag one, noise beyond), but their *magnitudes* are strongly and persistently correlated — still +0.22 twenty days out. Yesterday tells you almost nothing about the direction of today, and quite a lot about the size of today. This is the "split personality" from lesson one made precise, and it is the single fact that the iid assumption in everything below quietly ignores — [Time Series Analysis](03-time-series.md) is where it stops being ignored.

## The normal benchmark, and the size of its failure

If daily returns were normal with the sample mean and standard deviation, the expected number of days beyond $k$ sigma in a sample of $n$ is

$$
\mathbb{E}[\#\{|z_t| > k\}] = n \cdot 2\,\Phi(-k),
$$

where $\Phi$ is the standard normal CDF (the distribution's full anatomy is in [The Gaussian Distribution](../appendix/part-05-common-distributions/11-gaussian-distribution.md)). That formula turns "the normal underestimates tails" from a slogan into a scoreboard:

```python
import numpy as np
import pandas as pd
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
z = (r - r.mean()) / r.std()

for k in [3, 4, 5]:
    obs = (z.abs() > k).sum()
    exp = len(r) * 2 * stats.norm.sf(k)
    print(f"|z| > {k}: observed {obs:4d}   normal-expected {exp:8.3f}")
# => |z| > 3: observed  100   normal-expected   17.306
#    |z| > 4: observed   41   normal-expected    0.406
#    |z| > 5: observed   18   normal-expected    0.004
```

Read the last row slowly. The Gaussian model expected 0.004 five-sigma days in twenty-five years — roughly one per seven thousand years of trading. SPY delivered eighteen, one every seventeen months: the model is off by a factor of about 4,500, and the error *grows* with $k$, which is exactly where risk lives. This is the same fat tail the log-density histogram in [Plotting for Research](../part-02-python/06-plotting.md) made visible by eye; now it has a number. A risk system, an option price, or a position size built on the normal is not slightly wrong in the tails — it is wrong by orders of magnitude, precisely on the days that decide whether the fund survives.

## Heavier-tailed families

The families that fail less share one design idea: polynomial rather than exponential tail decay. The Student-t with $\nu$ degrees of freedom has density tails

$$
f(x) \;\sim\; |x|^{-(\nu + 1)},
$$

so $\nu$ directly indexes tail weight — moments exist only up to order $\nu$, meaning a t with $\nu < 4$ has no finite kurtosis and a t with $\nu < 2$ no finite variance ([Student's t-Distribution](../appendix/part-05-common-distributions/16-students-t-distribution.md) has the derivations). The **normal inverse Gaussian** (NIG) reaches heavy tails by a different route: it is a normal whose variance is itself random, which is exactly what volatility clustering suggests the market is doing — the same construction that makes the $t$ itself, with a different mixing law, as [Student's t Distribution](../appendix/part-05-common-distributions/16-students-t-distribution.md) derives — and its four parameters buy separate control of tail weight and asymmetry. The **stable** family is the theoretical aristocracy — closed under addition, tail index $\alpha \le 2$, beloved of Mandelbrot — but for $\alpha < 2$ it has *infinite variance*, a strong claim about markets that the data gets to veto below. The finance-side survey of all three lives in [Heavy-Tailed Returns](../appendix/part-18-quant-finance-applications/13-heavy-tailed-returns.md).

## Fitting by likelihood, choosing by information criteria

Maximum likelihood ([Maximum Likelihood Estimation](../appendix/part-11-parameter-estimation/03-maximum-likelihood-estimation.md)) fits each family; AIC ([Information Criteria](../appendix/part-14-model-selection/03-information-criteria.md)) compares them with a penalty for parameter count, so a family only wins by earning its extra flexibility:

```python
import numpy as np
import pandas as pd
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna().values

fits = {"normal": (stats.norm, stats.norm.fit(r)),
        "student-t": (stats.t, stats.t.fit(r)),
        "NIG": (stats.norminvgauss, stats.norminvgauss.fit(r))}
for name, (dist, p) in fits.items():
    ll = dist.logpdf(r, *p).sum()
    print(f"{name:10s} k={len(p)}  loglik {ll:8.1f}  AIC {2 * len(p) - 2 * ll:9.1f}")
# => normal     k=2  loglik  19095.0  AIC  -38185.9
#    student-t  k=3  loglik  20034.3  AIC  -40062.7
#    NIG        k=4  loglik  20061.6  AIC  -40115.1

print(f"fitted t df = {fits['student-t'][1][0]:.2f}")  # => fitted t df = 2.65
```

Two results matter. First, the gap between the normal and everything else is enormous — nearly a thousand log-likelihood points, a magnitude of evidence that closes the question. Second, the fitted t degrees of freedom is **2.65**: the data places daily SPY in a regime where kurtosis is infinite and even the variance is only barely finite. That single parameter is the stylized facts compressed into one number, and it is pinned this tightly only because the sample is this long — on one year of data the same fit returns $2.73 \pm 0.61$ and cannot tell infinite kurtosis from finite ([Maximum Likelihood Estimation](../appendix/part-11-parameter-estimation/03-maximum-likelihood-estimation.md)), while the cheaper estimator that matches the sample kurtosis instead converges on 4.13 and never on 2.65 ([Method of Moments](../appendix/part-11-parameter-estimation/04-method-of-moments.md)). NIG's fourth parameter (asymmetry) earns its keep on AIC, making it the formal winner, but the t is within sight and much simpler — on a desk, t-versus-NIG is a taste decision, normal-versus-either is not a decision at all.

The stable family gets its own block because its likelihood has no closed form and the honest cost should be visible:

```python
import numpy as np
import pandas as pd
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna().values

# stable MLE is genuinely expensive — about five minutes even on half the sample
alpha, beta, loc, scale = stats.levy_stable.fit(r[::2])
print(f"alpha = {alpha:.2f}, beta = {beta:.2f}")  # => alpha = 1.53, beta = -0.23
```

The fitted $\alpha = 1.53$ asserts infinite variance — and on a same-sample AIC comparison the stable finishes *behind* both t and NIG. The empirical verdict on daily index returns is consistent across decades of literature and reproduced here: tails are far heavier than Gaussian but lighter than stable. Mandelbrot was right about the disease and wrong about the cure.

## QQ plots: reading the tails by eye

A QQ plot ranks the data, ranks the model's quantiles, and plots one against the other; a good fit is a straight line, and every departure has a diagnosis. It is the sixty-second check that should precede any formal fitting:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
stats.probplot(r, dist="norm", plot=axes[0])
axes[0].set_title("SPY daily returns vs normal")
df = stats.t.fit(r)[0]
stats.probplot(r, dist=stats.t, sparams=(df,), plot=axes[1])
axes[1].set_title(f"vs Student-t (df = {df:.2f})")
plt.show()
```

Against the normal, SPY draws the classic S-curve: the extreme empirical quantiles are far larger in magnitude than the theoretical line, both tails peeling away — fat tails on sight. Against the fitted t, the body and most of the tail sit on the line, with only the last few points wandering. The reading vocabulary is small: an **S-curve** means fat tails; a **tilted or bowed line** means skew; a **kink** in one tail usually means one regime (a crash cluster) that the rest of the sample does not resemble; points peeling off only beyond the 99.9th percentile mean the family is fine except for the handful of days that extreme-value theory exists for. Fit formally afterward — but the plot is where you learn which families are even candidates.

!!! warning "There are no 25-sigma days, only wrong models"
    Sigma-counting is model-relative arithmetic: under the fitted Student-t, 2008 and 2020 were merely very bad days, entirely within the distribution's expectations. When a risk report announces an 8-sigma event, the correct response is not awe at the market but suspicion of the covariance model that produced the number — and the same suspicion applies to your own backtest the moment it assumes normality to compute anything.

!!! abstract "Key takeaways"
    - Simple returns measure money, log returns measure compounding; on calm days they agree to four decimals and on 2008-10-13 they differ by a full point.
    - Log returns sum across time and simple returns sum across positions — swapping the two cost 58 basis points on one day of a 60/40 portfolio.
    - The stylized facts are one code block: excess kurtosis 11.4, mild negative skew, and magnitude autocorrelation near +0.3 while sign autocorrelation is noise.
    - The fitted normal expected 0.004 five-sigma days in 25 years and SPY produced 18 — an error factor of thousands that grows exactly where risk lives.
    - Heavy-tailed families differ in mechanism: t buys polynomial tails with one parameter, NIG randomizes the variance, stable asserts infinite variance.
    - By AIC, NIG narrowly beats Student-t and both annihilate the normal; the fitted t df of 2.65 is the fat-tail headline, and the stable's α = 1.53 overshoots what data supports.
    - QQ plots read the tails by eye — S-curve fat tails, tilt skew, kinks a foreign regime — and belong before any formal fit.

## Where this goes next

Every fit in this lesson treated the sample as independent draws from one urn, and the stylized-facts block already proved that false: return magnitudes remember themselves for weeks. The next step is to model the time axis instead of assuming it away — stationarity, autocorrelation, volatility dynamics, and the long-run relationships between prices. That is [Time Series Analysis](03-time-series.md).
