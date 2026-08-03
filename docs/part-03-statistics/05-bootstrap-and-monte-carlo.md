# Bootstrap and Monte Carlo Methods

The previous lesson could put an error bar on a Sharpe ratio because Andrew Lo derived a formula for one. Now try the statistics a risk committee actually asks about: the maximum drawdown you should expect, the odds of a five-year losing stretch, the confidence interval on a hit-rate difference between two signal versions. There are no formulas. The classical toolkit hands you asymptotic standard errors for means and little else, and strategy evaluation runs almost entirely on statistics the textbook never derived.

Resampling is the way out: let the data stand in for the distribution that generated it, and simulate the sampling variability you cannot derive. This lesson builds the bootstrap from its iid form to the block form that respects the serial dependence lesson three documented, assembles a reusable Sharpe-interval pipeline, turns Monte Carlo loose on drawdown distributions, and closes with permutation tests — plus the honest list of places where resampling quietly lies. The strategy under the microscope is still lesson four's twelve-month momentum rule. Underneath all of it is one construction — every simulated path in this lesson starts as a stream of uniforms pushed through an inverse distribution function, and why that works for *any* law, including the empirical one being resampled here, is [Change of Variables](../appendix/part-03-random-variables/09-change-of-variables.md) and [Sampling Methods](../appendix/part-09-monte-carlo-methods/02-sampling-methods.md).

## When the formula runs out

The bootstrap rests on one idea, the *plug-in principle*: the empirical distribution $\hat{F}$ of your sample is your best estimate of the unknown $F$, so the sampling variability of any statistic under $F$ is estimated by its variability under $\hat{F}$ — which you can measure directly, by redrawing samples from your own data:

$$
\operatorname{SE}_{F}\big(\hat\theta\big) \;\approx\; \operatorname{SE}_{\hat F}\big(\hat\theta\big).
$$

Why this works at all is the [Weak Law of Large Numbers](../appendix/part-07-asymptotic-theory/01-weak-law-of-large-numbers.md) — $\hat F$ converges to $F$ — and the general theory is the appendix's [Bootstrap Methods](../appendix/part-09-monte-carlo-methods/07-bootstrap-methods.md) page. The practical promise is what matters: the same twenty lines of code produce an error bar for *any* statistic — Sharpe, drawdown, hit rate, skew of the worst decade — with no derivation, ever. The rest of the lesson is about earning that promise honestly.

## The iid bootstrap

Draw $n$ observations from the sample *with replacement*, compute the statistic, repeat ten thousand times; the spread of the results is the sampling distribution. For the momentum strategy's Sharpe:

```python
import numpy as np
import pandas as pd
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
strat = (np.sign(r.rolling(252).sum()).shift(1) * r).dropna().values
n = len(strat)

def sharpe(x, axis=None):
    return np.sqrt(252) * x.mean(axis=axis) / x.std(axis=axis)

rng = np.random.default_rng(42)
boot = sharpe(strat[rng.integers(0, n, (10_000, n))], axis=1)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"Sharpe {sharpe(strat):.2f}, iid bootstrap 95% CI [{lo:.2f}, {hi:.2f}]")
# => Sharpe 0.30, iid bootstrap 95% CI [-0.09, 0.71]

res = stats.bootstrap((strat,), sharpe, n_resamples=9_999, method="BCa",
                      random_state=42)
print(f"BCa 95% CI [{res.confidence_interval.low:.2f}, {res.confidence_interval.high:.2f}]")
# => BCa 95% CI [-0.10, 0.70]
```

The percentile interval simply reads the 2.5th and 97.5th percentiles of the bootstrap distribution; BCa additionally corrects for bias and skew in that distribution, and for a well-behaved statistic like the Sharpe the two agree to the second decimal (the taxonomy of interval constructions is in [Bootstrap Confidence Intervals](../appendix/part-11-parameter-estimation/08-bootstrap-confidence-intervals.md)). Note what else agrees: Lo's *analytic* interval from the last lesson was [−0.09, 0.70]. When a formula exists, the bootstrap reproduces it — that agreement is the sanity check, and the bootstrap's value is everywhere the formula does not exist. Be precise about what the interval claims: it quantifies estimation uncertainty — *given that history, how much could this number wobble by luck of the sample* — and says nothing about regimes the sample never contained.

## Serial dependence and the block bootstrap

The iid bootstrap has a hidden assumption in plain sight: drawing days independently destroys their order, and lesson three established that the order carries real structure — volatility clusters with a five-week half-life. Resampled histories made of independently drawn days are smoother than real ones. The repair is to resample *blocks* of consecutive days, preserving the local dependence inside each block:

```python
import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
strat = (np.sign(r.rolling(252).sum()).shift(1) * r).dropna().values

def sharpe(x):
    return np.sqrt(252) * x.mean() / x.std()

bs = StationaryBootstrap(18, strat, seed=42)     # mean block length ~ n^(1/3)
dist = bs.apply(lambda x: sharpe(x), 10_000).ravel()
lo, hi = np.percentile(dist, [2.5, 97.5])
print(f"block bootstrap 95% CI [{lo:.2f}, {hi:.2f}]  (SE {dist.std():.2f})")
# => block bootstrap 95% CI [-0.07, 0.68]  (SE 0.19)
```

The standard block-length rule of thumb is $n^{1/3}$ — about 18 days here — and the *stationary* variant draws geometrically distributed block lengths around that mean, which avoids the seam artifacts of fixed blocks. The result deserves an honest reading: the block interval is a hair *narrower* than the iid one (SE 0.19 vs 0.20), not wider. That is the same phenomenon the HAC correction surfaced in [lesson four](04-hypothesis-testing-and-multiple-testing.md) — this strategy's daily returns are mildly *negatively* autocorrelated, so respecting the dependence slightly helps it. The deeper reason the direction is not fixed is that the scheme is chosen by the *statistic*, not by the series: on a volatility-clustered process the iid interval for a mean covers 0.952 and the iid interval for a standard deviation covers 0.537 ([Bootstrap Confidence Intervals](../appendix/part-11-parameter-estimation/08-bootstrap-confidence-intervals.md)). For the positively autocorrelated series that dominate real research — overlapping returns, smoothed marks, persistent exposures — the block interval comes out wider, often dramatically, and using the iid bootstrap there overstates your certainty exactly the way the naive t-statistic did. The scheme is not a formality; it is where the method's honesty lives:

| Scheme | Preserves | Choose when |
|---|---|---|
| iid resampling | marginal distribution only | data plausibly independent — cross-sectional stats, shuffled residuals |
| Moving block (fixed length) | dependence within blocks | serial dependence with a known short range |
| Stationary (random length) | dependence, without seam artifacts | the default for financial time series |

## A complete Sharpe pipeline

Everything so far compresses into one self-contained block — load, build, resample, report — which is the template any strategy statistic in Part IV should pass through before it appears in a document:

```python
import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
strat = (np.sign(r.rolling(252).sum()).shift(1) * r).dropna().values

def sharpe(x):
    return np.sqrt(252) * x.mean() / x.std()

bs = StationaryBootstrap(18, strat, seed=42)
dist = bs.apply(lambda x: sharpe(x), 10_000).ravel()
lo, hi = np.percentile(dist, [5, 95])
print(f"Sharpe {sharpe(strat):.2f}, 90% CI [{lo:.2f}, {hi:.2f}]  "
      f"(stationary bootstrap, 10,000 resamples, mean block 18d)")
# => Sharpe 0.30, 90% CI [-0.01, 0.62]  (stationary bootstrap, 10,000 resamples, mean block 18d)
```

The reporting convention is part of the method: the interval travels with the point estimate, and the parenthetical says how it was made, so a reader can reproduce or dispute it. Equally important is the list of things this line does **not** claim. It does not claim the strategy works (the interval brushes zero, as the last lesson's tests foretold). It does not price in the fifty variants tried before this one — a bootstrap cannot launder multiple testing. And it does not claim next decade resembles the sample. It claims one thing: *given this history and this rule, sampling luck alone moves the Sharpe about this much*. That is a modest claim, which is precisely why it is defensible.

## Monte Carlo: the distribution of pain

A backtest hands you one equity curve — a single draw from the distribution of histories the strategy could produce. Sizing and survival decisions need the distribution, especially its left side: how deep do drawdowns get, and how long do they last? Block-resampling full histories answers by brute force:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
strat = (np.sign(r.rolling(252).sum()).shift(1) * r).dropna().values
n = len(strat)

rng = np.random.default_rng(42)
L, B = 21, 5_000                                 # 21-day blocks, 5,000 histories
starts = rng.integers(0, n - L, (B, int(np.ceil(n / L))))
paths = strat[(starts[:, :, None] + np.arange(L)).reshape(B, -1)][:, :n]
equity = np.exp(np.cumsum(paths, axis=1))
peak = np.maximum.accumulate(equity, axis=1)

maxdd = (equity / peak - 1).min(axis=1)
idx = np.arange(n)
last_high = np.maximum.accumulate(np.where(equity >= peak, idx, -1), axis=1)
longest = (idx - last_high).max(axis=1)

print(f"max drawdown: median {np.median(maxdd):.0%}, worst 5% {np.percentile(maxdd, 5):.0%}")
# => max drawdown: median -49%, worst 5% -71%
print(f"longest underwater: median {np.median(longest) / 252:.1f}y, "
      f"95th pct {np.percentile(longest, 95) / 252:.1f}y")
# => longest underwater: median 7.2y, 95th pct 18.7y
```

Sit with these numbers, because they are the emotional core of the lesson. This strategy's *realized* max drawdown was −43% — and the simulation says the median alternative history was worse (−49%), with one history in twenty reaching −71%. The median longest underwater stretch is **seven years**; the 95th percentile is longer than most careers. None of this contradicts the 0.30 Sharpe — it *is* the 0.30 Sharpe, translated from a ratio into lived experience, the same translation the underwater plot in [Plotting for Research](../part-02-python/06-plotting.md) performed for a single history. A strategy approved on its Sharpe and abandoned in year four of a median drawdown was never actually approved; running this block before allocation is how you find out what you are really signing up for (the appendix pages on [Monte Carlo Simulation](../appendix/part-09-monte-carlo-methods/03-monte-carlo-simulation.md) and [Drawdown Probabilities](../appendix/part-18-quant-finance-applications/03-drawdown-probabilities.md) treat the machinery and the theory).

## Permutation tests: does the timing matter?

A permutation test asks a sharper question than "is the mean positive?" — it asks whether your signal's *specific alignment* with returns beats a meaningless alignment. Build the null by breaking that alignment while preserving everything else: circularly shift the position series by a random offset, which keeps its long/short mix, its autocorrelation, its regime structure — everything except the claimed timing skill:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
df = pd.DataFrame({"pos": np.sign(r.rolling(252).sum()).shift(1), "r": r}).dropna()
pos, ret = df["pos"].values, df["r"].values

def sharpe(x):
    return np.sqrt(252) * x.mean() / x.std()

actual = sharpe(pos * ret)
rng = np.random.default_rng(42)
null = np.array([sharpe(np.roll(pos, k) * ret)
                 for k in rng.integers(1, len(ret), 1_000)])
pval = (1 + (null >= actual).sum()) / (1 + len(null))
print(f"actual Sharpe {actual:.2f}, null mean {null.mean():+.2f}, null SD {null.std():.2f}")
# => actual Sharpe 0.30, null mean +0.23, null SD 0.20
print(f"permutation p = {pval:.3f}")  # => permutation p = 0.296
```

The p-value is the actual statistic's rank in the null distribution, $(1 + \#\{SR^* \ge SR\}) / (1 + B)$, and the verdict is blunt: p = 0.296. But the most instructive number is the null *mean* of +0.23. Shifted-at-random momentum positions still earn a 0.23 Sharpe, because the position series is long 73% of the time and the market drifted up — the strategy's headline number is mostly packaged market beta. Against the only fair benchmark — *same exposures, meaningless timing* — the timing adds 0.07 of Sharpe, indistinguishable from noise. This is why the permutation null must be constructed with care (shift, don't shuffle — shuffling would also destroy the position series' own structure and flatter the strategy): the test is only as sharp as the property you hold fixed. The appendix prices the difference — on strategies built to have no edge at all, an iid shuffle produces a null distribution 18% too narrow and rejects 9.7% of the time against a nominal 5%, where a circular shift holds 4.3%. Formal treatment in [Permutation Tests](../appendix/part-12-hypothesis-testing/09-permutation-tests.md) and its bootstrap-flavored sibling [Bootstrap Tests](../appendix/part-12-hypothesis-testing/10-bootstrap-tests.md).

## Where resampling misleads

The bootstrap feels assumption-free, which is exactly what makes it dangerous — its assumptions are just quieter:

| Pitfall | Symptom | Mitigation |
|---|---|---|
| Extreme-value statistics | intervals for max drawdown or worst day centered near the sample's own extreme | resampling cannot invent a worse day than history contains; use parametric tails ([EVT](../appendix/part-18-quant-finance-applications/14-extreme-value-theory.md)) |
| Heavy tails | intervals that jump when one crisis day enters or leaves a resample | df ≈ 2.6 from lesson two strains the [CLT](../appendix/part-07-asymptotic-theory/03-central-limit-theorem.md) machinery beneath the bootstrap; check stability by deleting the worst day |
| Small samples | beautiful intervals from 36 monthly observations | $\hat F$ is a poor stand-in for $F$ at small $n$; the bootstrap amplifies, not fixes, data poverty |
| Nonstationarity | 2008 blocks resampled into a 2019-shaped future | the interval is conditional on the sampled regime mix; report sub-period intervals too |
| Snooped inputs | a tight CI around a cherry-picked variant | the bootstrap quantifies sampling noise, not selection bias — lesson four's corrections still apply first |

Each row is the same disease in different clothing: the bootstrap knows only the history you feed it. It redistributes that history's information honestly, and cannot add information the history lacks.

!!! warning "The bootstrap resamples your history, not your future"
    Ten thousand resamples of 2001–2025 tell you how differently 2001–2025 could have gone — they are silent about futures whose regimes, correlations, or tail events your sample never saw. A resampled confidence interval is a statement about estimation noise, not a forecast range. The moment a bootstrap interval is presented as "the range of outcomes we expect going forward," it has been promoted beyond its competence, and the promotion ceremony is where the money is lost.

!!! abstract "Key takeaways"
    - The plug-in principle turns twenty lines of code into an error bar for any statistic — the bootstrap exists for the drawdowns, hit rates, and ratios no formula covers.
    - The iid bootstrap on the momentum Sharpe reproduces Lo's analytic interval almost exactly, and BCa's refinements barely move it — agreement with theory is the sanity check.
    - Independent-day resampling destroys volatility clustering; the stationary block bootstrap (mean block ≈ n^⅓) preserves it, and the direction of the resulting correction is the data's decision, not a rule.
    - The reporting template is point estimate, interval, and method in one line — plus the discipline of listing what the interval does not claim.
    - Monte Carlo turns a 0.30 Sharpe into lived experience: median max drawdown −49%, one-in-twenty histories reaching −71%, and a median seven-year longest underwater stretch.
    - The circular-shift permutation test isolates timing skill from exposure, and the verdict is sharp: randomly shifted momentum positions earn Sharpe 0.23 of this strategy's 0.30, p = 0.296.
    - The bootstrap's quiet assumptions — extremes, tails, small n, stationarity, snooped inputs — all reduce to one: it cannot know anything your history does not contain.

## Where this goes next

Everything in Part III so far has treated the strategy's edge as a fixed unknown constant — estimate it, test it, wrap it in an interval. But a desk's actual belief about an edge is not a constant; it updates as evidence arrives, shrinks toward skepticism when data is thin, and conditions on which market regime is in force. Making that reasoning formal — priors, posteriors, shrinkage, and hidden regime states — is the closing lesson, [Bayesian Methods and Hidden Markov Models](06-bayesian-methods-and-hmms.md).
