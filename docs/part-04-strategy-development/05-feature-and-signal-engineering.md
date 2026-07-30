# Feature and Signal Engineering

Two of this part's strategies are dead — `xsmom` from [lesson three](03-cross-sectional-and-volatility-strategies.md) never had an edge, `tom` from [lesson four](04-seasonality-and-calendar-effects.md) never had an effect — and both deaths share a procedural cause: a trading *rule* was built and backtested before anyone measured whether the underlying *signal* contained information. This lesson builds the measurement layer that belongs in between. Its object of study is not a strategy but a pipeline: raw prices become *features* (cleaned, transformed, standardized descriptions of each asset), features become *signals* (cross-sectionally comparable forecasts), and signals are scored — for information content, for how fast that content decays, for how much trading they demand — before any of them is allowed near a portfolio. The pipeline's star exhibit is the sector momentum signal itself, because this lesson owes the course an autopsy: exactly *why* did the textbook formation earn nothing across 293 months?

The answer will turn out to be a single formula — the fundamental law of active management — and it is worth the whole lesson, because it converts "the strategy didn't work" into a quantitative statement about habitat that could have been made *before* the backtest ran.

## From raw prices to features

A feature is a number describing one asset at one date, computed only from information available at that date. The sector panel yields the standard starter set — trailing 12−1 momentum, one-month reversal, 63-day volatility — and the first engineering decision is what to do about scale, because raw features arrive in wildly different units:

```python
import numpy as np
import pandas as pd

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
px = pd.read_parquet("data/part4.parquet")[sectors]
mp = px.resample("ME").last()

mom = mp.pct_change(11).shift(1)                       # 12-1 momentum
rev = -mp.pct_change(1)                                # 1-month reversal
vol = np.log(px).diff().rolling(63).std().resample("ME").last() * np.sqrt(252)

pooled = mom.stack().dropna()
print(f"raw momentum, pooled: mean {pooled.mean():+.1%}, std {pooled.std():.1%}, "
      f"min {pooled.min():+.0%}, max {pooled.max():+.0%}")
z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)
clipped = (z.abs() > 2.5).stack().mean()
z = z.clip(-2.5, 2.5)
print("after cross-sectional z-score: each month mean 0, std 1 by construction")
print(f"winsorized at |z| = 2.5: {clipped:.1%} of observations clipped")
# => raw momentum, pooled: mean +9.0%, std 18.7%, min -68%, max +91%
#    after cross-sectional z-score: each month mean 0, std 1 by construction
#    winsorized at |z| = 2.5: 0.1% of observations clipped
```

Raw trailing momentum spans −68% to +91% — a range dominated by *when* the observation happened (2008 shrinks everything, 2021 inflates everything) rather than *which sector* it describes. The cross-sectional z-score removes the when: each month is centered and scaled against its own peers, so a +1.2 means "1.2 standard deviations better than this month's field" in 2008 and 2021 alike. That is the transformation that makes observations comparable across time, and it is done *per date*, never on the pooled panel — pooling would leak each month's context into every other. The winsorization line, clipping at |z| = 2.5, is stated for discipline and does almost nothing here (0.1% of observations), for a reason worth knowing: with nine assets, a cross-sectional z-score is mathematically bounded near 2.67, so the universe's smallness has already tamed the tails. In a three-thousand-stock universe the same line is load-bearing — single-name blowups produce z-scores of ten that would otherwise own the portfolio.

## Risk adjustment reorders the middle

The same twelve-month return means different things from a utility fund and from a tech fund — one is three volatilities of drift, the other is Tuesday. Dividing momentum by each sector's own volatility converts "how much it moved" into "how unusual the move was," and it is worth measuring how much that re-scoring actually changes:

```python
import numpy as np
import pandas as pd
from scipy import stats

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
px = pd.read_parquet("data/part4.parquet")[sectors]
mp = px.resample("ME").last()
mom = mp.pct_change(11).shift(1)
vol = np.log(px).diff().rolling(63).std().resample("ME").last() * np.sqrt(252)

radj = mom / vol                                       # momentum per unit risk
common = mom.dropna().index.intersection(radj.dropna().index)
rc = np.mean([stats.spearmanr(mom.loc[t], radj.loc[t])[0] for t in common])
flips = np.mean([(mom.loc[t].rank() - radj.loc[t].rank()).abs().max()
                 for t in common])
print(f"rank corr, raw vs risk-adjusted momentum: {rc:.2f}")
print(f"avg worst single-sector rank change per month: {flips:.1f} of 9")
# => rank corr, raw vs risk-adjusted momentum: 0.87
#    avg worst single-sector rank change per month: 2.1 of 9
```

The two versions agree at rank correlation 0.87 — risk adjustment is a refinement, not a revolution — but the second line locates where the disagreement lives: every month, some sector moves an average of two ranks, and in a top-three/bottom-three construction a two-rank move at the boundary is the difference between long and flat. This is the general character of feature engineering choices: they rarely change the story, they routinely change the *portfolio*, and each variant is a trial for the ledger (the count is running; the last section collects it). The professional habit is to fix these choices by argument before scoring — here, risk-adjusted is the defensible default, because a signal should claim information, not just variance — and resist the urge to try both and keep the better backtest, which is how [lesson four's](04-seasonality-and-calendar-effects.md) graveyard filled up.

## The information coefficient is the exchange rate

A signal's information coefficient is the cross-sectional rank correlation between the signal today and returns tomorrow — the purest available measure of "does this ranking know anything?" — and its power comes from the fundamental law of active management, which converts IC into achievable performance:

$$
\mathrm{IR} \;\approx\; \mathrm{IC} \times \sqrt{\mathrm{BR}} ,
$$

information ratio equals information coefficient times the square root of breadth, the number of independent bets per year. Score the sector momentum signal:

```python
import numpy as np
import pandas as pd
from scipy import stats

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
mp = pd.read_parquet("data/part4.parquet")[sectors].resample("ME").last()
mr = mp.pct_change()
mom = mp.pct_change(11).shift(1)

ic = pd.Series({t: stats.spearmanr(mom.loc[t], mr.shift(-1).loc[t])[0]
                for t in mom.dropna().index[:-1]})
t_ic = ic.mean() / ic.std() * np.sqrt(len(ic))
print(f"momentum IC: mean {ic.mean():+.3f}, t = {t_ic:+.2f}, "
      f"positive in {(ic > 0).mean():.0%} of {len(ic)} months")
br = 9 * 12
print(f"fundamental law: IR = IC x sqrt(BR) = {ic.mean():.3f} x sqrt({br}) "
      f"= {ic.mean() * np.sqrt(br):.2f}")
print(f"IC needed for IR 0.5 at this breadth: {0.5 / np.sqrt(br):.3f}")
# => momentum IC: mean +0.012, t = +0.46, positive in 51% of 293 months
#    fundamental law: IR = IC x sqrt(BR) = 0.012 x sqrt(108) = 0.13
#    IC needed for IR 0.5 at this breadth: 0.048
```

Here is the autopsy, complete in three lines. The signal's IC is +0.012 — right about even with a coin (51% of months positive), t = 0.46, agreeing with lesson three's alpha regression that there is nothing here to measure. But run the fundamental law forward *as if the IC were real*: at nine assets rebalanced monthly, breadth is at most 108 bets a year — fewer, since sectors correlate at 0.58 — so even a genuine IC of 0.012 buys an information ratio of 0.13, almost exactly the Sharpe 0.08 the strategy realized. And an IR of 0.5 at this breadth demands IC 0.048 — *four times* the information the best-documented equity signal in history displays in this habitat. Now the habitat argument from lesson three becomes arithmetic: the same 0.012 IC applied to a thousand-stock universe, breadth twelve thousand, projects an IR near 1.3. Cross-sectional momentum did not fail because the effect is fake; it failed because nine sector baskets offer an edge measured in hundredths nowhere to compound. The fundamental law prices a strategy from its ingredients — before any backtest, which is exactly where this lesson claims the analysis belonged.

## Decay curves need signal to decay

Textbook: an edge has a shelf life, measured by recomputing IC against returns further and further out — momentum's curve should peak at one to three months, short-term reversal's should be dead within weeks, and the crossing points dictate rebalance schedules. Practice, on this universe:

```python
import numpy as np
import pandas as pd
from scipy import stats

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
mp = pd.read_parquet("data/part4.parquet")[sectors].resample("ME").last()
mr = mp.pct_change()
mom = mp.pct_change(11).shift(1)
rev = -mp.pct_change(1)

for name, sig in [("momentum", mom), ("reversal", rev)]:
    row = []
    for h in [1, 3, 6, 12]:
        ic = pd.Series({t: stats.spearmanr(sig.loc[t], mr.shift(-h).loc[t])[0]
                        for t in sig.dropna().index[:-h]})
        row.append(f"h={h}m {ic.mean():+.3f}")
    print(f"{name:9s} IC by horizon: " + "  ".join(row))
# => momentum  IC by horizon: h=1m +0.012  h=3m +0.008  h=6m +0.018  h=12m +0.005
#    reversal  IC by horizon: h=1m +0.005  h=3m +0.013  h=6m -0.007  h=12m +0.018
```

The curves are garbage, and diagnosing *why* they are garbage is the section's real content. A Spearman correlation across nine assets carries a standard error near 0.35 per month; averaged over 293 months, each printed IC has a 95% error bar of roughly ±0.04 — wider than the entire vertical range of both "curves." Neither the peaks nor the sign flips mean anything; the whole exhibit is one flat band of noise. The methodological moral is not "decay analysis is useless" — on a universe with real breadth it is among the most decision-relevant plots in the pipeline, setting the rebalance clock and exposing signals that die before they can be traded. The moral is that *every* diagnostic in this lesson has a resolution limit set by breadth and sample, and an analyst who reads structure in a curve whose error bars swallow it is doing astrology with better fonts. Measure the instrument before trusting the reading — the same discipline [Part III](../part-03-statistics/05-bootstrap-and-monte-carlo.md) applied to Sharpe ratios, now applied to the tools themselves.

## Turnover is the signal's metabolism

Two signals with identical ICs can differ tenfold in what they cost to follow, and the difference is visible before any trade: it is the signal's own persistence. A ranking that barely changes month to month asks for little trading; a ranking that reshuffles completely demands the whole book be rebuilt:

```python
import numpy as np
import pandas as pd
from scipy import stats

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
mp = pd.read_parquet("data/part4.parquet")[sectors].resample("ME").last()
mom = mp.pct_change(11).shift(1)
rev = -mp.pct_change(1)

for name, sig in [("momentum", mom), ("reversal", rev)]:
    ac = np.mean([stats.spearmanr(sig.loc[t], sig.shift(1).loc[t])[0]
                  for t in sig.dropna().index[12:]])
    ranks = sig.rank(axis=1)
    w = ((ranks >= 7).astype(float) - (ranks <= 3).astype(float)) / 3
    turn = (w - w.shift(1)).abs().sum(axis=1).dropna().mean() / 2
    print(f"{name:9s}: month-to-month rank autocorr {ac:+.2f}, "
          f"top3/bottom3 one-way turnover {turn:.0%}/month")
# => momentum : month-to-month rank autocorr +0.84, top3/bottom3 one-way turnover 42%/month
#    reversal : month-to-month rank autocorr -0.01, top3/bottom3 one-way turnover 134%/month
```

Momentum's ranking persists at 0.84 month over month — eleven of its twelve formation months are shared between consecutive readings, so persistence is built into its definition — and the resulting portfolio turns over 42% a month. Reversal's ranking has *zero* memory (−0.01, again by construction: each month's reading is one fresh month of returns), and following it means replacing 134% of the book monthly — the portfolio does not even survive from one rebalance to the next. Same universe, same IC neighborhood, triple the metabolic rate. This is why turnover belongs in the signal report and not just the backtest: rank autocorrelation is a property of the signal's *formula*, knowable before any portfolio exists, and it fixes the cost hurdle the IC must clear. A slow signal with a small edge can be viable; a fast signal with a small edge is a donation to the market-making industry — a sentence [lesson seven](07-portfolio-construction-and-transaction-costs.md) will convert into basis points.

## Combination works — and cannot resurrect

The composite question: momentum and reversal measure different things (their feature correlation is −0.02 — genuinely independent), so blending them should help. It does, exactly as theory promises, and the result is still worthless — both facts instructive:

```python
import numpy as np
import pandas as pd
from scipy import stats

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
mp = pd.read_parquet("data/part4.parquet")[sectors].resample("ME").last()
mr = mp.pct_change()

def zx(f):
    return f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1), axis=0).clip(-2.5, 2.5)

mom, rev = zx(mp.pct_change(11).shift(1)), zx(-mp.pct_change(1))
fc = np.mean([stats.spearmanr(mom.loc[t], rev.loc[t])[0]
              for t in mom.dropna().index])
print(f"feature correlation momentum vs reversal: {fc:+.2f}")
for name, sig in [("momentum", mom), ("reversal", rev), ("50/50", (mom + rev) / 2)]:
    ic = pd.Series({t: stats.spearmanr(sig.loc[t], mr.shift(-1).loc[t])[0]
                    for t in sig.dropna().index[:-1]})
    print(f"{name:9s} IC {ic.mean():+.3f}")
# => feature correlation momentum vs reversal: -0.02
#    momentum  IC +0.012
#    reversal  IC +0.005
#    50/50     IC +0.016
```

The equal-weight composite's IC of 0.016 beats both parents — combining uncorrelated signals adds information content, the diversification logic of [lesson one](01-momentum-and-trend-following.md) operating one level up the stack ([Model Averaging](../appendix/part-14-model-selection/05-model-averaging.md) is the same theorem in statistical dress). And 0.016 remains a third of the 0.048 this breadth requires: the combination arithmetic is sound and the conclusion is unchanged, because averaging two signals that round to zero produces a slightly better zero. Signal combination refines edges; it cannot create them — the exact lesson the dilution table taught at portfolio level in lesson three, recurring here at signal level, and it will recur once more at book level. One further caution while the machinery is out: IC-*weighted* combination (weighting each signal by its historical IC) was deliberately not used, because estimated ICs this noisy make the weights themselves a fresh overfitting surface — with two signals and a sample this size, equal weights are the defensible choice ([Feature Selection](../appendix/part-14-model-selection/04-feature-selection.md) treats the general problem).

## The paper trail

Everything this lesson measured belongs in one place, produced by one function, filed for every signal the desk ever evaluates — *including the discarded ones*, because the trials count in the report is what makes every other number in it interpretable:

```python
import numpy as np
import pandas as pd
from scipy import stats

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
mp = pd.read_parquet("data/part4.parquet")[sectors].resample("ME").last()
mr = mp.pct_change()

def signal_report(name, sig, fwd, trials):
    ic = pd.Series({t: stats.spearmanr(sig.loc[t], fwd.loc[t])[0]
                    for t in sig.dropna().index.intersection(fwd.dropna().index)})
    ac = np.mean([stats.spearmanr(sig.loc[t], sig.shift(1).loc[t])[0]
                  for t in sig.dropna().index[12:]])
    t_ic = ic.mean() / ic.std() * np.sqrt(len(ic))
    print(f"signal: {name}   sample: {ic.index[0]:%Y-%m} to {ic.index[-1]:%Y-%m} "
          f"({len(ic)} months)")
    print(f"  IC {ic.mean():+.3f} (t = {t_ic:+.2f}), hit {(ic > 0).mean():.0%}, "
          f"rank autocorr {ac:+.2f}")
    print(f"  trials this family: {trials}  ->  Bonferroni t needed: "
          f"{stats.norm.ppf(1 - 0.025 / trials):.1f}")

signal_report("sector 12-1 momentum", mp.pct_change(11).shift(1),
              mr.shift(-1), trials=24)
# => signal: sector 12-1 momentum   sample: 2001-01 to 2025-05 (293 months)
#      IC +0.012 (t = +0.46), hit 51%, rank autocorr +0.84
#      trials this family: 24  ->  Bonferroni t needed: 3.1
```

Twelve lines of function, and a signal can no longer lie about itself: its information content with an honest t-statistic, its hit rate, its metabolic rate, and — the line that separates research from mining — the size of the family it was drawn from, with the significance bar that family size implies. The 24 is this course's own momentum ledger (lesson one's lookback grid), and the report says a member of that family needs t = 3.1 to be believed; this one brings 0.46. The habit being installed is procedural, not technical: the report is written *when the signal is first scored*, before any backtest, and it is never deleted — a desk's collection of dead signal reports is its institutional memory, the only durable defense against rediscovering the same noise every few years with fresh enthusiasm. [Lesson eight](08-validation-and-overfitting.md) will build the formal version of this discipline; the informal version is a folder that nothing is ever removed from.

!!! warning "Every signal you evaluated and discarded is still standing behind the one you kept"
    The IC you report is the maximum of every IC you computed, whether you admit it or not — every feature variant, every risk adjustment, every horizon you glanced at and moved past. The only defenses are structural: write the report before the backtest, count the trials in the report itself, and let the significance bar grow with the count. A signal that cannot clear the bar its own search history sets was not discovered; it was manufactured.

!!! abstract "Key takeaways"
    - Features are standardized per date, never pooled: the cross-sectional z-score turns −68%-to-+91% raw momentum into peer-relative scores, and winsorization — cosmetic here at 0.1% clipped, load-bearing in wide universes — completes the hygiene.
    - Risk adjustment agrees with the raw feature at rank correlation 0.87 yet moves an average worst sector 2.1 ranks a month — feature choices rarely change the story and routinely change the portfolio, so fix them by argument, not by backtest.
    - The fundamental law is the autopsy: IC +0.012 (t = 0.46) at breadth 108 prices sector momentum at IR 0.13 — the realized 0.08, predicted from ingredients — while IR 0.5 demands IC 0.048, and the same IC at stock-universe breadth projects near 1.3: the effect wasn't fake; the habitat was.
    - The IC decay curves span less than their own ±0.04 error bars: every diagnostic has a resolution limit set by breadth and sample, and reading structure inside the error bar is astrology.
    - Turnover is knowable from the signal's formula alone: momentum's 0.84 rank autocorrelation implies 42% monthly turnover, reversal's −0.01 implies 134% — same IC neighborhood, triple the cost hurdle.
    - Combining independent signals works exactly as promised — IC 0.016 from parents at 0.012 and 0.005 — and cannot resurrect the dead: combination refines edges, it does not create them.
    - The signal report — IC, hit rate, autocorrelation, and the trials count with its Bonferroni bar (t = 3.1 for this family; the signal brings 0.46) — is written before the backtest and never deleted.

## Where this goes next

The pipeline now runs from raw prices to scored, honestly-documented signals — and stops exactly where money begins. A signal, however good, says nothing about *how much* to hold: converting scores into positions, deciding how much risk each sleeve and each asset may spend, scaling to a volatility target, and respecting the capacity limits real markets impose is a separate discipline with its own failure modes. [Position Sizing and Risk Budgeting](06-position-sizing-and-risk-budgeting.md) takes it up, and it opens the account this part has left deliberately unsettled since lesson one: the diversification benefit that `tsmom` banked as reduced volatility and never converted into return.
