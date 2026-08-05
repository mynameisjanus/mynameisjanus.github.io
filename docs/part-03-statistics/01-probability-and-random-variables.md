# Probability and Random Variables

Ask a trader for the probability that tomorrow is a down day and you will get a number without hesitation; ask where the number came from and the conversation gets quiet. Most market intuitions are base rates nobody ever measured — "bounces follow selloffs," "bonds hedge stocks" — repeated until they feel like data. This lesson is about replacing that folklore with counted frequencies: framing market questions as probability statements, choosing the right random-variable model for each quantity, estimating moments with error bars attached, and measuring dependence in ways that survive a crisis.

The formal machinery — events as sets, conditioning, densities — lives in the appendix, starting at [Sets and Functions](../appendix/part-01-mathematical-foundations/01-sets-and-functions.md) and [Conditional Probability](../appendix/part-02-probability-foundations/03-conditional-probability.md); this lesson assumes it and spends the time on data.

All of Part III runs on one real dataset, downloaded once and cached. Every code block loads the same file, so every printed number in this part reproduces on your machine exactly.

## One dataset, cached once

The dataset is twenty-five years of daily adjusted closes for four ETFs: SPY (US equities), IVV (a second S&P 500 tracker, which earns its place in the [Time Series Analysis](03-time-series.md) lesson), TLT (long-duration Treasuries), and GLD (gold). One download, one file:

```python
# one-time download — requires a network connection
import yfinance as yf

px = yf.download(["SPY", "IVV", "TLT", "GLD"], start="2000-01-01",
                 end="2025-07-01", auto_adjust=True)["Close"]
px = px[["SPY", "IVV", "TLT", "GLD"]]
px.to_parquet("data/prices.parquet")
```

From here on, every block in Part III opens the same way — the cache is the Part III ritual, the way `default_rng(42)` was Part II's:

```python
import pandas as pd

px = pd.read_parquet("data/prices.parquet")

print(px.shape)                                       # => (6411, 4)
print(px.index[0].date(), "->", px.index[-1].date())  # => 2000-01-03 -> 2025-06-30
print(px.apply(lambda s: s.first_valid_index().date()))
# => Ticker
#    SPY    2000-01-03
#    IVV    2000-05-19
#    TLT    2002-07-30
#    GLD    2004-11-18
#    dtype: object
```

That last print is the first act of data hygiene in this part: the four funds were born on different dates, so any cross-asset panel has a ragged start. Whenever a block needs all columns it will `dropna()` and print the surviving row count — silently analyzing a panel where one column is one-third missing is how correlation studies go wrong before any statistics happen. The cache is also deliberately frozen: dividend adjustments rewrite price history retroactively, so re-downloading next year shifts old numbers. The file, not the vendor, is the source of truth for everything printed in this part — the same point-in-time discipline that [SQL and Data Storage](../part-02-python/05-sql-and-data-storage.md) made about research databases.

!!! note "Versions"
    Part III assumes Python 3.12+, NumPy 2.x, pandas 3.x, SciPy 1.18+, statsmodels 0.14+, arch 8.x, hmmlearn 0.3.x, and yfinance 1.x (with pyarrow for parquet); the examples were verified with NumPy 2.5, pandas 3.0, SciPy 1.18, statsmodels 0.14.6, arch 8.0, and hmmlearn 0.3.3. The cached window ends 2025-06-30 — yfinance's `end` is exclusive.

## Events, conditioning, and base rates

An event is a set of outcomes, and its probability is estimated by counting: the share of days on which the event occurred. The conditional probability

$$
P(A \mid B) \;=\; \frac{P(A \cap B)}{P(B)}
$$

is the same count restricted to the days where $B$ held — and conditioning is the entire trade, because an edge is precisely a conditional probability that differs from its base rate. Three folklore claims, counted:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
spy = px["SPY"]
r = np.log(spy).diff().dropna()
down = r < 0
prev = down.shift(1)

print(f"P(down)             {down.mean():.3f}")               # => 0.452
print(f"P(down | down yest) {down[prev == True].mean():.3f}")  # => 0.439
print(f"P(down | up yest)   {down[prev == False].mean():.3f}") # => 0.462

dd = spy / np.maximum.accumulate(spy) - 1.0       # drawdown from running peak
in_dd = (dd.shift(1) <= -0.05).reindex(r.index).fillna(False)
print(f"P(down | >5% dd)    {down[in_dd].mean():.3f}")         # => 0.454
print(f"share of days in >5% drawdown: {in_dd.mean():.3f}")    # => 0.550
```

The base rate is the first surprise: SPY fell on only 45% of days, so "the market goes up more often than not" is true daily, not just annually. Conditioning on yesterday moves the needle a little — down days follow down days *less* often than they follow up days, a mild daily mean reversion worth about 2.3 points of probability. And the drawdown conditioning is a genuine negative result: being 5% off the high tells you almost nothing about tomorrow's direction, even though 55% of all days — a majority of this 25-year bull market — were spent there. The habit to build is running exactly this computation before believing any conditional claim, because two of these three intuitions just failed on data.

## Pick the model the quantity deserves

A random variable is a number attached to an outcome (the formal construction is in [Random Variables](../appendix/part-03-random-variables/01-random-variables.md)), and the modeling decision is which distribution family fits the quantity's actual structure — discrete families for counts and signs, continuous densities ([Probability Density Functions](../appendix/part-03-random-variables/04-probability-density-functions.md)) for returns and prices.

| Market quantity | Natural model | Why |
|---|---|---|
| Sign of a day's return | Bernoulli($p$) | Two outcomes; $p$ is the base rate above |
| Down days in a month | Binomial — if days were iid | The "if" is testable, and fails informatively |
| Trades arriving per second | Poisson, to first order | Counts of rare-ish independent events |
| A daily return | Continuous, fat-tailed | The subject of the next lesson |

The second row is the interesting one, because it tests the first: if daily signs were an iid Bernoulli coin, the count of down days in a month would be Binomial, and extreme months would occur at a computable rate.

```python
import numpy as np
import pandas as pd
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

p = (r < 0).mean()
print(f"p = {p:.3f}, SE = {np.sqrt(p * (1 - p) / len(r)):.4f}")  # => p = 0.452, SE = 0.0062

monthly = (r < 0).resample("ME").agg(["sum", "count"])
monthly = monthly[monthly["count"] >= 15]          # drop partial months
extreme = (monthly["sum"] >= 14).sum()
expected = sum(stats.binom.sf(13, int(c), p) for c in monthly["count"])
print(f"months with >=14 down days: observed {extreme}, iid-expected {expected:.1f}")
# => months with >=14 down days: observed 8, iid-expected 13.3
```

The iid coin *overpredicts* bad months — eight observed against thirteen expected — because the mild mean reversion from the previous section stabilizes the count of signs. Hold that against what the next section shows about magnitudes: the sign process of daily returns is tamer than independence suggests, while the size process is far wilder. That split personality — direction nearly patternless, magnitude strongly structured — is the single most important empirical fact in this part, and lessons [two](02-returns-and-distributions.md) and [three](03-time-series.md) are each half of it.

## The first four moments of a real return series

Moments are the standard compression of a distribution: mean and variance for location and scale, then the standardized third and fourth moments

$$
\text{skew} = \frac{\mathbb{E}\big[(X-\mu)^3\big]}{\sigma^3},
\qquad
\text{kurt} = \frac{\mathbb{E}\big[(X-\mu)^4\big]}{\sigma^4} - 3
$$

for asymmetry and tail weight, with the $-3$ making the normal distribution's kurtosis the zero point (derivations in [Higher-Order Moments](../appendix/part-04-expectation-and-moments/03-higher-order-moments.md)).

```python
import numpy as np
import pandas as pd
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

print(f"annualized mean {252 * r.mean():.3f}, vol {np.sqrt(252) * r.std():.3f}")
# => annualized mean 0.075, vol 0.195
print(f"skewness {stats.skew(r):.2f}, excess kurtosis {stats.kurtosis(r):.2f}")
# => skewness -0.20, excess kurtosis 11.41
print(f"worst day {r.min():+.3f} ({r.idxmin().date()}), z = {(r.min() - r.mean()) / r.std():.1f}")
# => worst day -0.116 (2020-03-16), z = -9.4
```

In trading terms: the mean says equities paid about 7.5% a year for a quarter century; the vol says the ride cost about 19.5% a year of standard deviation; the skew says the bad surprises lean slightly larger than the good ones. The number that should stop you is the kurtosis. A normal distribution has excess kurtosis zero; SPY's daily returns sit above eleven, and the worst day in the sample is nine standard deviations from the mean — an event a Gaussian world would essentially never produce. Nothing else in Part III makes sense until that number is felt: it is why the next lesson exists, why naive t-tests overstate confidence, and why every risk model that assumed normality has the same failure story. There is a sharper reading of the eleven, too: [Higher-Order Moments](../appendix/part-04-expectation-and-moments/03-higher-order-moments.md) shows that at the tail index the next lesson fits, the population fourth moment does not exist at all, so the sample kurtosis is not converging on eleven — it grows with the sample, and the honest statement is that the tail is heavy enough to break the statistic being used to describe it.

## Standard errors, or which moment you can trust

Every number above is an estimate, and an estimate without a standard error is a rumor — the law of the estimate itself, and why a drift's precision is fixed by calendar span while a volatility's is fixed by observation count, is [Sampling Distributions](../appendix/part-10-statistics-foundations/03-sampling-distributions.md). For the mean of $n$ observations,

$$
\operatorname{SE}(\bar{r}) = \frac{\sigma}{\sqrt{n}},
$$

and with $n = 6{,}410$ daily observations that machinery produces the least comfortable number in this lesson:

```python
import numpy as np
import pandas as pd
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

se = 252 * r.std() / np.sqrt(len(r))
print(f"ann mean {252 * r.mean():.3f} +/- {se:.3f} (1 SE)")  # => ann mean 0.075 +/- 0.039

for a, b in [("2000", "2009"), ("2010", "2019"), ("2020", "2025")]:
    x = r.loc[a:b]
    print(f"{a}-{b}  mean {252 * x.mean():+.3f}  vol {np.sqrt(252) * x.std():.3f}  "
          f"skew {stats.skew(x):+.2f}  kurt {stats.kurtosis(x):.1f}")
# => 2000-2009  mean -0.009  vol 0.224  skew +0.09  kurt 9.6
#    2010-2019  mean +0.126  vol 0.147  skew -0.51  kurt 4.5
#    2020-2025  mean +0.134  vol 0.215  skew -0.54  kurt 12.7
```

Read the first line twice: after twenty-five years of daily data, the equity premium estimate is 7.5% with a standard error of 3.9% — a t-statistic of about 1.9, not even two standard errors from zero. That 3.9% is not this estimator's error bar but the floor for every unbiased estimator at any sampling frequency, because the Fisher information for a drift is $T/\sigma^2$ and the frequency cancels ([Properties of Estimators](../appendix/part-11-parameter-estimation/02-properties-of-estimators.md)). The decade table shows why more data barely helps: the 2000s mean was *negative*, and the skew flipped sign between decades. The mean of returns is genuinely, almost irreducibly hard to estimate, because its standard error shrinks with calendar time, not observation count — sampling the same decade more finely adds almost no information about it. Volatility is the opposite: its decade-to-decade movement (22%, 15%, 22%) reflects real regimes, not estimation noise. **Vol is estimable; the mean barely is.** That asymmetry governs everything downstream — it is why volatility models in lesson three work, why strategy t-tests in [lesson four](04-hypothesis-testing-and-multiple-testing.md) disappoint, and why the Bayesian machinery of [lesson six](06-bayesian-methods-and-hmms.md) still shows wide intervals after twenty years of updating.

## Dependence: covariance, correlation, and ranks

With two assets the object of interest is dependence: covariance for the raw co-movement, Pearson correlation for its scale-free version, and Spearman's rank correlation, which asks only whether the assets move in the same *order* — making it immune to the outliers that fat-tailed returns produce in bulk (the formal definitions are in [Covariance](../appendix/part-04-expectation-and-moments/04-covariance.md) and [Correlation](../appendix/part-04-expectation-and-moments/05-correlation.md)).

```python
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = px[["SPY", "TLT", "GLD"]].dropna().pct_change().dropna()

print(len(rets), "common days from", rets.index[0].date())
# => 5184 common days from 2004-11-19
print(rets.corr().round(2))
# => Ticker   SPY   TLT   GLD
#    SPY     1.00 -0.31  0.05
#    TLT    -0.31  1.00  0.16
#    GLD     0.05  0.16  1.00
print(rets.corr(method="spearman").round(2))
# => Ticker   SPY   TLT   GLD
#    SPY     1.00 -0.26  0.06
#    TLT    -0.26  1.00  0.18
#    GLD     0.06  0.18  1.00
```

The `dropna()` prints its receipt first: the panel starts where its youngest member (GLD) starts, in late 2004, and the row count says exactly what sample the matrices describe. The full-sample story is the classic one — stocks and long bonds negatively correlated at −0.31, gold near zero against both, the diversification everyone's 60/40 is built on. Pearson and Spearman broadly agree here, which is itself information: the negative SPY–TLT relationship is not an artifact of a few extreme days. When the two *disagree* — Pearson large, Spearman small — a handful of outliers is doing the work, and rank correlation is the five-second diagnostic that catches it. [Correlation](../appendix/part-04-expectation-and-moments/05-correlation.md) quantifies how little it takes: one contaminated observation in five thousand moves Pearson to $+0.74$ between two independent series while Spearman stays at $+0.01$, because Pearson averages products and a single large pair contributes a term that scales with the square of its size.

## When correlation lies

A single full-sample correlation quietly averages over two decades of changing relationships, and the average is not where the money is. Two decompositions matter: what happens on the bad days specifically, and what happens across time.

```python
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = px[["SPY", "TLT", "GLD"]].dropna().pct_change().dropna()

stress = rets[rets["SPY"] <= rets["SPY"].quantile(0.10)]
q45, q55 = rets["SPY"].quantile([0.45, 0.55])
middle = rets[rets["SPY"].between(q45, q55)]
print(f"SPY-TLT corr: full {rets['SPY'].corr(rets['TLT']):+.2f}, "
      f"stress {stress['SPY'].corr(stress['TLT']):+.2f}, "
      f"middle {middle['SPY'].corr(middle['TLT']):+.2f}")
# => SPY-TLT corr: full -0.31, stress -0.27, middle -0.02

roll = rets["SPY"].rolling(252).corr(rets["TLT"])
for d in ["2008-12-31", "2020-06-30", "2022-12-30", "2025-06-30"]:
    print(d, f"{roll.asof(d):+.2f}")
# => 2008-12-31 -0.48
#    2020-06-30 -0.50
#    2022-12-30 +0.08
#    2025-06-30 +0.06
```

The conditional cut is reassuring at first: on SPY's worst-decile days the hedge held (−0.27), and on ordinary days the two assets barely interact at all — the full-sample −0.31 is mostly a tail phenomenon. Then the rolling series delivers the correction. Through 2008 and 2020 the trailing-year correlation was near −0.50 and Treasuries were the crisis hedge of legend; by end-2022 it was *positive* — stocks and long bonds fell together all year, the 60/40 portfolio had one of its worst years on record, and anyone sized to the historical correlation discovered they owned two copies of the same inflation bet. Averages over history are not promises about regimes; the pattern where assets decouple in calm markets and re-correlate in stress is common enough that it has its own formal machinery, sketched in the appendix page on [Copulas](../appendix/part-18-quant-finance-applications/14-copulas.md).

The same common-sample data, compressed into the ledger this part returns to:

| Asset | Ann. return | Ann. vol | Skew | Excess kurtosis |
|---|---|---|---|---|
| SPY | +11.7% | 19.2% | 0.00 | 15.3 |
| TLT | +4.3% | 14.8% | +0.08 | 3.4 |
| GLD | +10.9% | 17.6% | −0.17 | 5.9 |

Every asset class is fat-tailed — equities spectacularly so — and note that SPY's *full-common-sample* skew is zero even though the 2010s alone showed −0.51: skew is the least stable of the four moments, and quoting it without a window is meaningless.

!!! warning "A correlation is a season, not a law"
    Any correlation quoted without a date range and a standard error is a story about the past wearing the costume of a parameter. SPY–TLT was −0.50 when it mattered in 2020 and +0.08 when it mattered in 2022; a hedge sized to the former was destroyed by the latter. Before any dependence number enters a position size, ask what window produced it, what its stress-conditional version looks like, and what happens to the portfolio if its sign flips.

!!! abstract "Key takeaways"
    - Part III runs on one cached real dataset — SPY, IVV, TLT, GLD daily closes, 2000–2025 — downloaded once; the frozen file, not the vendor, is the source of truth for every printed number.
    - An edge is a conditional probability that differs from its base rate, and counting is how you check one: two of three pieces of market folklore failed on 25 years of data.
    - Match the distribution family to the quantity's structure — and test the match: monthly down-day counts are *tamer* than an iid Bernoulli coin predicts, while return magnitudes are far wilder.
    - SPY's daily returns have excess kurtosis above 11 and a nine-sigma worst day — the single number that motivates the rest of this part.
    - The annualized mean carries a standard error half its own size after 25 years: volatility is estimable, the mean barely is, and that asymmetry shapes every later lesson.
    - Compute Pearson and Spearman together; agreement says the relationship is broad-based, disagreement says outliers are driving it.
    - Full-sample correlations average over regimes — SPY–TLT lived near −0.50 in two crises and flipped positive in 2022, which is the difference between a hedge and a second exposure.

## Where this goes next

Four moments compress a distribution; they do not describe it. Excess kurtosis of eleven says the tails are heavy but not what shape they are, and shape is where option prices, risk limits, and position sizes actually live. [Returns and Their Distributions](02-returns-and-distributions.md) puts full distributions to the data — measures exactly how badly the normal fails, and fits the families that fail less.
