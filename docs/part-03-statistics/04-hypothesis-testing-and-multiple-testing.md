# Hypothesis Testing and Multiple Testing

Every backtest ends with the same implicit claim: *this would have made money, and not by luck*. That is a hypothesis test, whether or not anyone writes it down — and it is a test conducted by someone with a conflict of interest, using data that violates the test's assumptions, usually after trying enough variants that something was guaranteed to look good. This lesson runs that trial properly, on a real strategy, and lets the evidence land where it lands.

The defendant throughout is twelve-month time-series momentum on SPY — long after an up year, short after a down year — chosen because it is public-domain, plausible, and four lines of code. The formal framework (nulls, alternatives, test statistics, what a p-value is and is not) lives in the appendix's [Hypothesis Testing Framework](../appendix/part-12-hypothesis-testing/01-hypothesis-testing-framework.md), [Test Statistics](../appendix/part-12-hypothesis-testing/02-test-statistics.md), and [p-values](../appendix/part-12-hypothesis-testing/03-p-values.md) pages; this lesson applies it to strategy returns, where every assumption earns a cross-examination.

## The question a backtest is actually asking

The null hypothesis is boredom: $H_0\!: \mu = 0$ — the strategy's true mean return is zero and the backtest profit is luck. The test statistic is the distance of the sample mean from zero, in standard-error units:

$$
t \;=\; \frac{\bar{x}}{\hat\sigma / \sqrt{n}} .
$$

First, the defendant:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

pos = np.sign(r.rolling(252).sum()).shift(1)   # +1 if trailing year up, else -1
strat = (pos * r).dropna()

print(f"n {len(strat)}, ann ret {252 * strat.mean():.1%}, "
      f"ann vol {np.sqrt(252) * strat.std():.1%}, "
      f"Sharpe {np.sqrt(252) * strat.mean() / strat.std():.2f}")
# => n 6158, ann ret 5.9%, ann vol 19.3%, Sharpe 0.30
print(f"share of days long: {(pos == 1).mean():.2f}")  # => 0.73
```

The `shift(1)` is the single most important character in the block — the position acts on *yesterday's* signal, because a signal that trades on the day it is computed is lookahead, and lookahead is the backtesting sin that Part V builds an entire engine to prevent. The strategy made 5.9% a year at a 0.30 Sharpe over 24 years, long about three-quarters of the time. The question this lesson exists to answer: is that evidence of an edge, or a coin that happened to land well?

## The t-test's fine print

Run the standard test first, then read the terms and conditions:

```python
import numpy as np
import pandas as pd
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
strat = (np.sign(r.rolling(252).sum()).shift(1) * r).dropna()

t, p = stats.ttest_1samp(strat, 0)
print(f"t = {t:.2f}, p = {p:.3f}")   # => t = 1.50, p = 0.135
print(f"kurtosis {stats.kurtosis(strat):.1f}, lag-1 autocorr {strat.autocorr(1):+.3f}")
# => kurtosis 12.2, lag-1 autocorr -0.061
```

The verdict is already unimpressive — p = 0.135 fails the conventional bar before any correction. But the diagnostics beneath it matter for every backtest you will ever test, including the ones that *do* clear the bar:

| Assumption | Daily strategy returns | Consequence |
|---|---|---|
| Normality | violated — kurtosis 12 | nearly harmless: at n ≈ 6,000 the [CLT](../appendix/part-07-asymptotic-theory/03-central-limit-theorem.md) makes the *mean* nearly normal anyway |
| Finite variance | holds, without much room — lesson two fitted a t with df 2.65 | the CLT's non-negotiable entry fee |
| Independence | violated, and it is load-bearing | the standard error itself is computed wrong |

The order is deliberate. Normality of returns — the assumption everyone worries about — is the one the central limit theorem largely repairs for you at this sample size. Independence — the assumption everyone forgets — feeds directly into $\hat\sigma/\sqrt{n}$: if returns are correlated across days, $n$ observations do not contain $n$ observations' worth of information, and the denominator of every t-statistic in every backtest report is a small lie.

## Autocorrelation and the honest sample size

With autocorrelated data the effective sample size is

$$
n_{\text{eff}} \;=\; \frac{n}{1 + 2\sum_{k \ge 1} \rho_k},
$$

and the practical repair is a HAC (heteroskedasticity-and-autocorrelation-consistent, Newey-West) standard error — a regression of the returns on a constant, with the covariance estimator doing the correcting:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
strat = (np.sign(r.rolling(252).sum()).shift(1) * r).dropna()
n = len(strat)

naive = sm.OLS(strat.values, np.ones(n)).fit()
hac = sm.OLS(strat.values, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": 21})
print(f"naive t {naive.tvalues[0]:.2f}   HAC t {hac.tvalues[0]:.2f}")
# => naive t 1.50   HAC t 1.61
print(f"effective n {n * (hac.tvalues[0] / naive.tvalues[0]) ** 2:.0f} of {n}")
# => effective n 7101 of 6158
```

A surprise worth savoring: the correction went *up*. This strategy's daily returns are mildly negatively autocorrelated (−0.061), so consecutive days partially cancel, each observation carries slightly more than one observation of information, and the naive test was — for once — conservative. The correction is a correction, not a penalty; the data decides its direction. Now watch the direction that destroys careers, using the same strategy summed into overlapping 21-day returns — the shape monthly-report tables and multi-day-horizon signals naturally produce:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
strat = (np.sign(r.rolling(252).sum()).shift(1) * r).dropna()

m = strat.rolling(21).sum().dropna()          # overlapping 21-day returns
print(f"lag-1 autocorr {m.autocorr(1):.2f}")  # => 0.94
naive = sm.OLS(m.values, np.ones(len(m))).fit()
hac = sm.OLS(m.values, np.ones(len(m))).fit(cov_type="HAC", cov_kwds={"maxlags": 42})
print(f"naive t {naive.tvalues[0]:.2f}   HAC t {hac.tvalues[0]:.2f}")
# => naive t 7.40   HAC t 1.73
print(f"effective n {len(m) * (hac.tvalues[0] / naive.tvalues[0]) ** 2:.0f} of {len(m)}")
# => effective n 335 of 6138
```

Same strategy, same information — and the naive t-statistic reads **7.40**, a seven-sigma discovery, because each day's return is counted twenty-one times and the test believes there are 6,138 independent observations when there are effectively 335. The HAC estimator deflates it back to 1.73, in line with the daily-frequency truth. The theorem that licenses substituting *any* estimated variance into a limiting normal is [Slutsky's Theorem](../appendix/part-07-asymptotic-theory/05-slutskys-theorem.md), and it is worth reading for what it does not certify: both estimators here are consistent, they simply converge to two different constants that differ by a factor of fourteen, and only one of them appears in the formula being used. Any time a backtest's t-statistic looks miraculous, the first suspect is not the strategy but the counting: overlapping windows, smoothed marks, illiquid prices, and monthly aggregation of persistent positions all manufacture exactly this inflation.

## Sharpe ratios have standard errors too

The industry reports Sharpe ratios, not t-statistics, so put the error bar where the industry looks. Lo's approximation for iid-ish returns — derived, along with the skew and kurtosis terms it drops, in [The Delta Method](../appendix/part-07-asymptotic-theory/04-delta-method.md) — gives the annualized Sharpe a standard error of

$$
\operatorname{SE}(\widehat{SR}_{\text{ann}}) \;\approx\; \sqrt{\frac{1 + \widehat{SR}_d^2 / 2}{n}} \cdot \sqrt{252},
$$

where $\widehat{SR}_d$ is the daily Sharpe:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
strat = (np.sign(r.rolling(252).sum()).shift(1) * r).dropna()
n = len(strat)

sr_d = strat.mean() / strat.std()
sr = np.sqrt(252) * sr_d
se = np.sqrt((1 + 0.5 * sr_d**2) / n) * np.sqrt(252)
print(f"Sharpe {sr:.2f} +/- {se:.2f}, 95% CI [{sr - 1.96 * se:.2f}, {sr + 1.96 * se:.2f}]")
# => Sharpe 0.30 +/- 0.20, 95% CI [-0.09, 0.70]
```

This may be the most sobering print in Part III. Twenty-four years of daily data — a track record longer than most funds survive — and the Sharpe ratio is 0.30 ± 0.20, a confidence interval spanning from slightly-money-losing to genuinely good. It is the same arithmetic as lesson one's standard error on the mean wearing industry clothing: Sharpe precision grows with *calendar time*, and no realistic track record is long enough to pin a moderate Sharpe down tightly. When an allocator treats the difference between a 0.8 and a 1.0 Sharpe as decision-relevant information from a three-year track record, the standard error says that difference is noise.

## Manufacturing significance: a grid of variants

Nobody backtests one variant. The 252-day lookback above was a choice, and the honest researcher admits the neighboring choices were available — so run all of them, and count what trying fifty variants buys:

```python
import numpy as np
import pandas as pd
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

rows = []
for lb in range(10, 501, 10):
    s = (np.sign(r.rolling(lb).sum()).shift(1) * r).dropna()
    rows.append((lb, np.sqrt(252) * s.mean() / s.std(), stats.ttest_1samp(s, 0).pvalue))
grid = pd.DataFrame(rows, columns=["lookback", "sharpe", "p"])

sig = grid[grid.p < 0.05]
print(f"{len(grid)} variants, {len(sig)} with p < 0.05, {0.05 * len(grid):.1f} expected by luck")
# => 50 variants, 1 with p < 0.05, 2.5 expected by luck
print(sig.round(3).to_string(index=False))
# =>  lookback  sharpe     p
#          180   0.424 0.035
```

Under a global null — fifty worthless strategies — chance alone hands you 2.5 "significant" results (see [Multiple Comparisons](../appendix/part-15-multiple-testing/01-multiple-comparisons.md) for why this arithmetic is exact in expectation). The grid produced *one*: a 180-day lookback at p = 0.035, flanked on both sides by lookbacks that show nothing. That isolation is diagnostic. A real effect bends the whole neighborhood of the parameter space around it — 160, 180, 200 should all catch it — while a fluke is a lone bright cell, which is precisely the overfitting signature the parameter-heatmap discussion in [Plotting for Research](../part-02-python/06-plotting.md) taught you to distrust by eye. Here the same signature arrives as arithmetic: fewer discoveries than luck predicts, and the one discovery has no neighbors. The literature on what unreported search does to reported results goes by [Data-Snooping Bias](../appendix/part-15-multiple-testing/04-data-snooping-bias.md), and it is the quiet explanation for most published backtests that never worked live.

## Familywise error, or the price of certainty

The classical repair is to control the **familywise error rate** — the probability that even one of the fifty rejections is false. The union bound makes Bonferroni almost embarrassingly simple: test each variant at $\alpha/m$ instead of $\alpha$, and the family-level error stays below $\alpha$ ([Bonferroni Correction](../appendix/part-15-multiple-testing/02-bonferroni-correction.md)). For fifty variants, the per-test bar drops to 0.001 — a t-statistic near 3.3 rather than 2.0. The price is power: real but modest edges, the only kind markets leave lying around, systematically fail a bar set that high. Bonferroni answers the question "am I certain nothing here is a fluke?" — the right question when a single approval is irreversible (a drug trial, a one-shot allocation), and usually the wrong one for a research pipeline, whose actual question is "of the ideas I advance to further work, what fraction is junk?"

## False discovery rate: the pipeline's question

That question has its own criterion — the **false discovery rate** — and the Benjamini-Hochberg procedure controls it: sort the p-values, find the largest $k$ with $p_{(k)} \le kq/m$, reject that many ([False Discovery Rate](../appendix/part-15-multiple-testing/03-false-discovery-rate.md)). At $q = 0.10$ you accept that one in ten of your advanced ideas will be a dud, in exchange for far more power than Bonferroni. The full accounting on the momentum grid:

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

pvals = [stats.ttest_1samp((np.sign(r.rolling(lb).sum()).shift(1) * r).dropna(), 0).pvalue
         for lb in range(10, 501, 10)]

for method, alpha in [("bonferroni", 0.05), ("fdr_bh", 0.10)]:
    rejected = multipletests(pvals, alpha=alpha, method=method)[0]
    print(f"{method} at {alpha}: {rejected.sum()} survivors")
# => bonferroni at 0.05: 0 survivors
#    fdr_bh at 0.1: 0 survivors
```

Zero survivors, under either philosophy — even the forgiving one. The honest conclusion of this lesson's trial: daily SPY sign-momentum, tested across its whole lookback family with multiplicity accounted for, offers no statistically defensible edge in this sample. That a five-decade literature finds time-series momentum across dozens of assets is not contradicted by one index failing one family of tests — but *this* backtest, on *this* data, does not clear the bar, and saying so plainly is the skill this lesson teaches. The choice between the two corrections, for the record:

| Procedure | Controls | Cost | Reach for it when |
|---|---|---|---|
| None | each test alone | a 50-variant grid manufactures ~2.5 flukes | never, for grids |
| Bonferroni | P(any false positive) | brutal power loss as $m$ grows | one irreversible decision rides on it |
| Benjamini-Hochberg | share of false positives among discoveries | a known, chosen dud rate | ranking ideas for further research |

One honest caveat closes the loop: these corrections treat the tests as roughly independent, and fifty overlapping momentum variants are anything but. The tools built specifically for correlated strategy families — White's Reality Check and Hansen's SPA test ([the appendix covers both](../appendix/part-15-multiple-testing/05-whites-reality-check.md)) — resample the whole grid at once, and they are exactly what [Validation and Overfitting](../part-04-strategy-development/08-validation-and-overfitting.md) deploys when Part IV industrializes this lesson.

!!! warning "Every parameter you tried is a test you ran"
    The grid counted fifty lookbacks because fifty were coded. It did not count the variants you ran last month and discarded, the entry rules you mentally rejected after one glance at an equity curve, or the nine ideas that died before this one was born. Every one of those was a draw from the null, and no correction formula can include tests it never saw. The only defenses are procedural: register what you try, decide evaluation rules before looking, and treat any result whose search history you cannot reconstruct as unreviewed evidence.

!!! abstract "Key takeaways"
    - A backtest is a hypothesis test with $H_0\!: \mu = 0$; the twelve-month momentum defendant posts a 0.30 Sharpe over 24 years, and `shift(1)` is what keeps the test free of lookahead.
    - The naive t-test's fragile assumption is not normality — the CLT repairs that at n ≈ 6,000 — but independence, which corrupts the standard error itself.
    - HAC standard errors correct in whichever direction the autocorrelation points: mildly negative here (effective n *above* n), and catastrophically positive for overlapping returns, where a fake t of 7.4 deflated to 1.7.
    - An annualized Sharpe of 0.30 carries a ±0.20 standard error after 24 years — Sharpe precision grows with calendar time, and most track records cannot statistically separate mediocre from good.
    - Fifty lookback variants yielded one nominal discovery against 2.5 expected by luck, and the discovery had no significant neighbors — the lone-bright-cell signature of noise.
    - Bonferroni controls the probability of any false positive and is priced accordingly; Benjamini-Hochberg controls the dud rate among advanced ideas, which is the question research pipelines actually ask.
    - Both corrections returned zero survivors: this strategy family, honestly accounted, shows no defensible edge on this sample — and stating that cleanly is the deliverable.

## Where this goes next

Every interval and p-value in this lesson leaned on a formula — Lo's approximation, HAC asymptotics, the CLT — and each formula holds under conditions that fat-tailed, dependent strategy returns strain. When the statistic is more exotic than a mean (a maximum drawdown, a hit-rate difference, a Sharpe of a Sharpe), the formulas run out entirely. [Bootstrap and Monte Carlo Methods](05-bootstrap-and-monte-carlo.md) replaces derivation with resampling: let the data write its own sampling distribution, and find out what error bars look like when no textbook supplies them.
