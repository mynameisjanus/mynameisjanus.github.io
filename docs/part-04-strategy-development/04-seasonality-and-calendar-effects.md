# Seasonality and Calendar Effects

Calendar effects are the most dangerous terrain in quantitative finance, and not because the stakes are high — they are the *lowest* in this part. They are dangerous because the terrain is a factory for false discoveries: the calendar partitions every return series into weekdays, months, half-years, holidays, expiries, and turn-of-anything windows, each partition is a free hypothesis test, and almost none of them come with a mechanism that could survive five minutes of desk scrutiny. The literature obliged for decades — the Monday effect, the January effect, Sell in May — and this lesson replays that literature against modern data with Part III's [multiple-testing machinery](../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) running from the start rather than bolted on after.

Announced in advance, per the house rules: this is the lesson where the part's designated failure gets built. [The index page](index.md) promised that failed strategies stay in the course, and the turn-of-month strategy constructed at the end of this lesson will be carried to lessons seven and eight *as a failure* — because watching a seductive artifact die under correct procedure teaches the procedure better than any success could.

## The literature, read against modern data

The founding classic is the Monday effect: in data from the 1950s through the 1970s, U.S. stocks reliably *fell* on Mondays — a result so strong in-sample it spawned theories about weekend news flow and settlement mechanics. Replicate the weekday table on a quarter century of SPY:

```python
import numpy as np
import pandas as pd
from scipy import stats

r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()

for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
    grp = r[r.index.day_name().str.startswith(day)]
    t, p = stats.ttest_1samp(grp, 0)
    print(f"{day}: mean {1e4 * grp.mean():+.1f} bp/day  (t = {t:+.2f}, n {len(grp)})")
# => Mon: mean +2.1 bp/day  (t = +0.55, n 1201)
#    Tue: mean +6.2 bp/day  (t = +1.85, n 1315)
#    Wed: mean +3.5 bp/day  (t = +1.04, n 1315)
#    Thu: mean +2.1 bp/day  (t = +0.62, n 1292)
#    Fri: mean +0.8 bp/day  (t = +0.24, n 1287)
```

The famous negative Monday is now +2.1 basis points at t = 0.55: not weakened, not attenuated — *gone*, sign and all. This is the standard biography of a published calendar effect: discovered in one sample, theorized after the fact, and dead in the next sample, whether because the anomaly was arbitraged once named or because it was never there. Note what the table would tempt a fresh researcher to do instead: Tuesday, at +6.2 bp and t = 1.85, is practically begging to be discovered — "Turnaround Tuesday," the whitepaper writes itself. Hold that temptation; the fourth section prices it. The discipline this table teaches is that a replication is not a formality — it is the *cheapest* test a claimed effect ever faces, and most of the classic calendar literature already fails here, before any multiple-testing correction is even summoned.

## Turn-of-month, January, and the seduction of shares

Two more famous residents, tested the same way — with one framing trap laid bare on purpose:

```python
import numpy as np
import pandas as pd
from scipy import stats

r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()

pos_in_month = r.groupby(r.index.to_period("M")).cumcount() + 1
rev = r[::-1].groupby(r[::-1].index.to_period("M")).cumcount()[::-1] + 1
tom = (pos_in_month <= 3) | (rev == 1)
t_tom, _ = stats.ttest_ind(r[tom], r[~tom], equal_var=False)
print(f"turn-of-month ({tom.mean():.0%} of days): {1e4 * r[tom].mean():+.1f} bp/day "
      f"vs other days {1e4 * r[~tom].mean():+.1f} bp  (t = {t_tom:+.2f})")
print(f"share of SPY's total 25-year log return earned in the window: "
      f"{r[tom].sum() / r.sum():.0%}")

iwm = np.log(pd.read_parquet("data/part4.parquet")["IWM"]).diff().dropna()
spread = (iwm - r).dropna()
jan = spread[spread.index.month == 1]
rest = spread[spread.index.month != 1]
t_jan, _ = stats.ttest_ind(jan, rest, equal_var=False)
print(f"January effect (IWM-SPY): Jan {1e4 * jan.mean():+.1f} bp/day "
      f"vs rest {1e4 * rest.mean():+.1f} bp  (t = {t_jan:+.2f})")
# => turn-of-month (19% of days): +5.2 bp/day vs other days +2.4 bp  (t = +0.73)
#    share of SPY's total 25-year log return earned in the window: 33%
#    January effect (IWM-SPY): Jan -0.5 bp/day vs rest -0.1 bp  (t = -0.12)
```

The January effect — small caps outrunning large caps in January, once the most robust seasonal in finance — reports in at t = −0.12 with the *wrong sign*: another corpse, dead in print for decades, still cited. Turn-of-month is the interesting one, because it shows how an artifact stays alive: the effect earns 33% of SPY's entire twenty-five-year return in 19% of the days, and that framing — *a third of the market's return in a fifth of the time!* — has sold a thousand newsletters. The t-statistic on the same numbers is 0.73. Both statements are true; only one is evidence. Concentration-of-return framings inherit all the noise of the sample paths they sum, which is why a difference that sounds enormous as a share can be statistically unremarkable as a mean. The rate at which money managers reach for the share framing when the t-statistic disappoints is not an accident; it is the tell.

## Asking the whole calendar at once

Testing effects one headline at a time flatters whichever one the literature made famous. The even-handed procedure asks the entire calendar simultaneously — a regression of returns on the full set of weekday and month dummies,

$$
r_t \;=\; \alpha \;+\; \textstyle\sum_{d} \gamma_d\, D^{\text{wd}}_{d,t} \;+\; \sum_{m} \delta_m\, D^{\text{mo}}_{m,t} \;+\; \varepsilon_t ,
$$

with joint F-tests asking whether *any* partition matters ([Multiple Linear Regression](../appendix/part-13-regression/02-multiple-linear-regression.md)), plus the spectral view: if the calendar drives returns, the periodogram should spike at calendar frequencies:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import signal

r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()

X = pd.get_dummies(r.index.dayofweek, prefix="wd", drop_first=True).astype(float)
X = pd.concat([X, pd.get_dummies(r.index.month, prefix="m",
                                 drop_first=True).astype(float)], axis=1)
X.index = r.index
fit = sm.OLS(r * 1e4, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
f_wd = fit.f_test([f"{c} = 0" for c in X.columns if c.startswith("wd")])
f_m = fit.f_test([f"{c} = 0" for c in X.columns if c.startswith("m")])
print(f"joint F, weekday dummies: p = {f_wd.pvalue:.2f}")
print(f"joint F, month dummies:   p = {f_m.pvalue:.2f}")

freq, power = signal.periodogram(r.values)
weekly = power[np.argmin(np.abs(freq - 1 / 5))]
print(f"periodogram power at the 5-day period: {weekly / np.median(power):.1f}x the median bin")
# => joint F, weekday dummies: p = 0.84
#    joint F, month dummies:   p = 0.39
#    periodogram power at the 5-day period: 0.2x the median bin
```

Asked properly, the calendar answers plainly: no weekday structure (p = 0.84), no month structure (p = 0.39), and the spectral line at the five-day period — where a weekly rhythm in *returns* would have to live — carries a fifth of the median bin's power, which is to say the weekly frequency is quieter than noise. The joint tests matter because they are immune to the cherry-pick: a single dummy fished out of seventeen can look interesting, but the F-test charges admission for all seventeen at once. One honest aside so the point is not overlearned: return *volatility* has genuine calendar structure — expiry cycles, announcement days, the December liquidity drought are all real and visible in squared returns — and desks trade that structure. It is the *mean* that the calendar refuses to predict, and strategies eat means.

## Nineteen hypotheses, one survivor, zero after correction

Now the accounting the whole lesson has been building toward. Assemble the family this lesson implicitly tested — five weekdays, twelve months, turn-of-month, Sell-in-May — and put the nineteen p-values through the corrections from the [appendix](../appendix/part-15-multiple-testing/02-bonferroni-correction.md):

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()

tests = {}
for d in range(5):
    grp = r[r.index.dayofweek == d]
    tests[f"weekday {d}"] = stats.ttest_ind(grp, r[r.index.dayofweek != d],
                                            equal_var=False)[1]
for mth in range(1, 13):
    grp = r[r.index.month == mth]
    tests[f"month {mth}"] = stats.ttest_ind(grp, r[r.index.month != mth],
                                            equal_var=False)[1]
pos_in_month = r.groupby(r.index.to_period("M")).cumcount() + 1
rev = r[::-1].groupby(r[::-1].index.to_period("M")).cumcount()[::-1] + 1
tests["turn-of-month"] = stats.ttest_ind(r[(pos_in_month <= 3) | (rev == 1)],
                                         r[~((pos_in_month <= 3) | (rev == 1))],
                                         equal_var=False)[1]
winter = r.index.month.isin([11, 12, 1, 2, 3, 4])
tests["sell-in-May"] = stats.ttest_ind(r[winter], r[~winter], equal_var=False)[1]

pv = pd.Series(tests)
raw = pv < 0.05
print(f"{len(pv)} calendar effects tested: {raw.sum()} pass raw 5%, "
      f"{multipletests(pv, 0.05, 'fdr_bh')[0].sum()} survive BH-FDR, "
      f"{multipletests(pv, 0.05, 'bonferroni')[0].sum()} survive Bonferroni")
print("raw-significant:", ", ".join(pv[raw].index),
      "| p =", [f"{p:.3f}" for p in pv[raw]])
# => 19 calendar effects tested: 1 pass raw 5%, 0 survive BH-FDR, 0 survive Bonferroni
#    raw-significant: month 9 | p = ['0.043']
```

Nineteen tests, one raw rejection — and the expected number of false rejections in nineteen independent null tests is about one. The lone survivor is September (p = 0.043), and the September effect even has a literature, which is precisely the trap: *every* cell in this family has a literature, because researchers have been running this same family for fifty years and publishing whichever cell cleared 5% in their sample. Under the null, the largest of nineteen |t|-statistics is expected around 2.4 — so Tuesday's 1.85 from the first section is not merely unconvincing, it is *below par for pure noise*. Both corrections return the family verdict: zero survivors ([False Discovery Rate](../appendix/part-15-multiple-testing/03-false-discovery-rate.md) covers why BH is the gentler of the two, and even it finds nothing). This is what a fully mined vein looks like from the inside, and it took thirty lines of code to establish — the cheapness is the point. Any strategy pitch built on a calendar cell now has to explain why *its* cell beats a family analysis this easy to run.

## Event-driven seasonals: the expiry week

One class of calendar claim deserves a fairer hearing, because it comes with an actual mechanism: recurring *events* — index rebalances, earnings clusters, option expiries — move real flows through real hands on known dates. Monthly index option expiry ("OPEX," the third Friday) is the testable one on this data, with a folklore of pinned prices and hedging-flow drift:

```python
import numpy as np
import pandas as pd
from scipy import stats

r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()

fridays = pd.date_range(r.index[0], r.index[-1], freq="W-FRI")
opex = fridays[fridays.day.isin(range(15, 22))]
opex_weeks = set(zip(opex.isocalendar().year, opex.isocalendar().week))
iso = r.index.isocalendar()
in_opex = pd.Series([(y, w) in opex_weeks for y, w in zip(iso.year, iso.week)],
                    index=r.index)
t, _ = stats.ttest_ind(r[in_opex], r[~in_opex], equal_var=False)
print(f"OPEX week: {1e4 * r[in_opex].mean():+.1f} bp/day vs other weeks "
      f"{1e4 * r[~in_opex].mean():+.1f} bp  (t = {t:+.2f})")
# => OPEX week: +1.4 bp/day vs other weeks +3.4 bp  (t = -0.55)
```

Nothing again — t = −0.55 — but the *reason* it is nothing matters more than the number. The expiry mechanism is real; what it moves is intraday liquidity, volatility, and single-name flows around strikes, and what this test measures is the daily close-to-close mean of the entire index — the mechanism's effects are two aggregations away from the measurement. That is the general shape of legitimate event-driven seasonality: it lives in *specific securities* at *specific times* under *measurable flow pressure* (the stock entering the index, the name reporting earnings), and it decays within days of becoming crowded. An index-level daily-mean test is exactly the wrong instrument for finding it — and, symmetrically, the index-level daily mean is exactly where newsletter seasonality claims to live. The absence of the effect *here* and the presence of real event flows *elsewhere* are not in tension; they are the same fact about where mechanisms operate.

## The strategy that was never there

Procedure demands the ending be played out. The turn-of-month window was the family's most photogenic member — 33% of the return, a plausible-sounding story about month-end 401(k) flows — so build it as [the index page's](index.md) rules require any strategy be built, and let it fail in public. Long SPY during the window, flat otherwise:

```python
import numpy as np
import pandas as pd

r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()
pos_in_month = r.groupby(r.index.to_period("M")).cumcount() + 1
rev = r[::-1].groupby(r[::-1].index.to_period("M")).cumcount()[::-1] + 1
tom = r.where((pos_in_month <= 3) | (rev == 1), 0.0)

def sharpe(s):
    return np.sqrt(252) * s.mean() / s.std()

print(f"tom full sample: ann ret {252 * tom.mean():+.1%}, Sharpe {sharpe(tom):.2f}")
print(f"  vs SPY buy-hold Sharpe {sharpe(r):.2f}")
for lo, hi in [("2000", "2012"), ("2013", "2025")]:
    s = tom.loc[lo:hi]
    print(f"{lo}-{hi}: ann ret {252 * s.mean():+.1%}, Sharpe {sharpe(s):.2f}")
# => tom full sample: ann ret +2.5%, Sharpe 0.31
#      vs SPY buy-hold Sharpe 0.38
#    2000-2012: ann ret +3.2%, Sharpe 0.35
#    2013-2025: ann ret +1.8%, Sharpe 0.25
```

The verdict does not even require the advanced machinery: `tom` never beat the index it hides inside — Sharpe 0.31 against buy-and-hold's 0.38 over the same sample — and its edge decays across the split, +3.2% a year in the first half to +1.8% in the second, exactly the post-publication decay path a mined artifact follows as its discovery sample recedes. The underlying "effect" sat at t = 0.73 in a nineteen-member family; there was never a reason to expect anything else. This is the fifth and final strategy of the part, and its role from here is cadaver: [lesson seven](07-portfolio-construction-and-transaction-costs.md) will bill it eight round trips a month against a nonexistent edge, and [lesson eight](08-validation-and-overfitting.md) will process it through the formal gauntlet so the paperwork of its death is complete. Keeping it costs nothing and buys something rare: a worked example, end to end, of what almost-real looks like — because the dangerous strategies are never the obviously fake ones.

!!! warning "A calendar effect with no owner and no mechanism is a p-value with a publicist"
    Before any seasonal claim gets a second minute of your attention, demand three answers: *who* is on the other side of the trade and why; *what* flow or constraint recurs on this date; and *how many* calendar cells were searched to find it. The Monday effect had publicists for forty years; it never had an owner. If the answer to the third question is "the whole calendar" — and for every screen-discovered seasonal, it is — then the p-value must survive the whole calendar's correction, and this lesson just showed what survives: nothing.

!!! abstract "Key takeaways"
    - The Monday effect — the founding calendar anomaly — is dead with its sign flipped: +2.1 bp at t = 0.55 on 25 years of SPY; replication on fresh data is the cheapest test a published effect faces, and most calendar classics fail it.
    - Framing is the artifact's life support: turn-of-month earns "33% of the return in 19% of the days," which sounds enormous, and tests at t = 0.73, which is nothing — a share of a noisy sum is not evidence; the January small-cap effect prints t = −0.12 with the wrong sign.
    - Asked jointly, the calendar has nothing to say about mean returns: weekday dummies p = 0.84, month dummies p = 0.39, and the periodogram's five-day line runs at 0.2× the median bin — though volatility's calendar structure is real, and means are what strategies eat.
    - The family analysis is the method: 19 calendar effects produced one raw rejection (September, p = 0.043) — almost exactly the one false positive nineteen null tests owe — and zero survivors under BH-FDR or Bonferroni; under the null the best of nineteen t-statistics is expected near 2.4, so Tuesday's 1.85 is below par for noise.
    - Event-driven seasonals earn their fair hearing and fail it at the index level — OPEX week at t = −0.55 — because real expiry mechanisms move liquidity and single names intraday, not the index's daily mean; mechanism location, not mechanism existence, is what the test must match.
    - The turn-of-month strategy is the part's designated failure, built to spec and kept on the books: Sharpe 0.31 versus the index's 0.38, decaying from +3.2% to +1.8% across the 2013 split — carried forward as the cadaver lessons seven and eight will formally process.

## Where this goes next

Four of the part's five strategies now exist, and the score is honest: one modest survivor, one conditional survivor, one gross-Sharpe illusion, one dead ranking, one cadaver. What the failures share is instructive — each was a *rule* reached before anyone measured whether the underlying *signal* contained information at the horizon traded. [Feature and Signal Engineering](05-feature-and-signal-engineering.md) builds that measurement layer properly: features cleaned and standardized, signals scored by information coefficient, decay curves that say how fast an edge evaporates, and the paper trail that records how many things were tried — the discipline that, applied earlier, would have priced `xsmom` and `tom` before a single backtest ran.
