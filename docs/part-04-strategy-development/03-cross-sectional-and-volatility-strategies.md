# Cross-Sectional and Volatility Strategies

The first two lessons judged each asset against its own past. This lesson opens two genuinely new fronts. The cross-sectional front changes the question from "will this go up?" to "which of these will beat the others?" — a reformulation with a structural gift attached: whatever the market does to every asset at once cancels out of a long-short book, leaving only the *relative* bet. The volatility front changes what is being traded at all: not direction but insurance, the standing gap between what the market pays for protection and what protection ends up costing. One of this lesson's two strategies will earn the best conditional Sharpe in the course so far. The other will die on contact with the data — and stay in the book anyway, because [the index page](index.md) promised that failures get kept, and this one's autopsy teaches more than most successes.

Both fronts run on the universe [the previous lesson](02-mean-reversion-and-pairs-trading.md) froze into `data/part4.parquet`: nine sector funds for the cross-section, the two volatility indexes for the insurance business.

## Two axes of prediction

Before ranking sectors, measure how much they actually differ — because a cross-sectional strategy can only eat what the common market move leaves behind:

```python
import numpy as np
import pandas as pd

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
px = pd.read_parquet("data/part4.parquet")[sectors]
mr = px.resample("ME").last().pct_change().dropna()

corr = mr.corr().values
print(f"avg pairwise sector correlation {corr[np.triu_indices(9, k=1)].mean():.2f}")
mkt = mr.mean(axis=1)
r2 = np.mean([np.corrcoef(mr[c], mkt)[0, 1] ** 2 for c in sectors])
print(f"share of monthly variance explained by the common move: {r2:.0%}")
demeaned = mr.sub(mr.mean(axis=1), axis=0)
print(f"avg pairwise corr after demeaning each month: "
      f"{demeaned.corr().values[np.triu_indices(9, k=1)].mean():+.2f}")
# => avg pairwise sector correlation 0.58
#    share of monthly variance explained by the common move: 63%
#    avg pairwise corr after demeaning each month: -0.11
```

Sectors co-move at an average correlation of 0.58, and 63% of a typical sector's monthly variance is simply the market's move wearing a sector costume. That is the bad news and the good news at once. Bad: only about a third of what happens to a sector each month is *sector-specific*, so the raw material for relative bets is a minority of the variance on display. Good: subtracting each month's cross-sectional mean flips the average correlation to −0.11 — mechanically negative, because relative winners and relative losers must offset — which means a long-short book built on demeaned returns is hedged against the market's whole first principal component *by construction*, before any risk model is consulted. Time-series momentum needed three assets and got diversification grudgingly; the cross-section manufactures a market-neutral bet out of a single date's ranking. Whether there is any *signal* in that ranking is a separate question, and the next section asks it.

## Cross-sectional momentum in nine sectors

The hypothesis transplants directly from [lesson one](01-momentum-and-trend-following.md): past relative winners keep winning. The classic formation is "12−1" — rank on the trailing twelve months *excluding* the most recent (which lesson two showed belongs to reversal), rebalance monthly, hold the top three sectors against the bottom three:

```python
import numpy as np
import pandas as pd

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
mp = pd.read_parquet("data/part4.parquet")[sectors].resample("ME").last()
mr = mp.pct_change()
mom = mp.pct_change(11).shift(1)                 # 12-1 formation
ranks = mom.rank(axis=1)
w = ((ranks >= 7).astype(float) - (ranks <= 3).astype(float)) / 3
xsmom = (w.shift(1) * mr).sum(axis=1).loc["2001-02":]

yr = xsmom.groupby(xsmom.index.year).sum()
print(f"xsmom: ann ret {12 * xsmom.mean():+.1%}, "
      f"Sharpe {np.sqrt(12) * xsmom.mean() / xsmom.std():.2f}, n {len(xsmom)} months")
print(f"best year {yr.idxmax()} {yr.max():+.1%}, worst year {yr.idxmin()} {yr.min():+.1%}")
# => xsmom: ann ret +1.0%, Sharpe 0.08, n 293 months
#    best year 2022 +26.4%, worst year 2016 -17.4%
```

One percent a year. Sharpe 0.08. Twenty-four years of monthly rebalancing, and the strategy earned nothing distinguishable from zero. Sit with that result rather than past it, because it was produced by the *textbook* formation — the same 12−1 recipe that mints money in academic studies of individual stocks — applied exactly as the literature prescribes, hypothesis-first, no tuning, no second try. This is `xsmom`, the third of the part's five strategies, and it is dead on arrival. The honest question is *why*, and the candidates are two. First, habitat: cross-sectional momentum's documented home is thousands of individual stocks, where the extremes of the ranking are genuinely extreme; nine diversified sector baskets offer the effect almost nowhere to live — a breadth argument the [signal-engineering lesson](05-feature-and-signal-engineering.md) will make devastatingly precise. Second, era: even in stocks, the effect's public decades were its best ones, and this sample starts *after* publication. Note also what the year rows do to the folklore: the strategy's best year was 2022 — an energy-versus-everything dispersion year — and its worst was 2016, not the 2009 momentum crash the literature would have predicted for it. A strategy this weak does not even fail the way its literature says it should.

## Long-short does not mean market-neutral

The construction still has things to teach, dead signal or not. Two properties every long-short book must have measured before anyone calls it "neutral":

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
mp = pd.read_parquet("data/part4.parquet")[sectors].resample("ME").last()
mr = mp.pct_change()
ranks = mp.pct_change(11).shift(1).rank(axis=1)
w = ((ranks >= 7).astype(float) - (ranks <= 3).astype(float)) / 3
xsmom = (w.shift(1) * mr).sum(axis=1).loc["2001-02":]

spy = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff() \
        .resample("ME").sum(min_count=1)
fit = sm.OLS(xsmom, sm.add_constant(spy.reindex(xsmom.index))).fit()
turn = (w - w.shift(1)).abs().sum(axis=1).loc["2001-02":] / 2
print(f"realized beta to SPY {fit.params.iloc[1]:+.2f}  "
      f"(alpha {12 * fit.params.iloc[0]:+.1%}/yr, t = {fit.tvalues.iloc[0]:+.2f})")
print(f"one-way turnover {turn.mean():.0%}/month  "
      f"(full portfolio replaced every {1 / turn.mean():.1f} months)")
# => realized beta to SPY -0.25  (alpha +3.0%/yr, t = +1.15)
#    one-way turnover 44%/month  (full portfolio replaced every 2.3 months)
```

Equal dollars long and short did not deliver beta zero — it delivered −0.25, because the sectors momentum shorts are systematically the high-beta ones (whatever crashed last year has the most violent relationship with the market this year), so the short side carries more market exposure than the long side. Dollar-neutral is an accounting statement; *beta*-neutral is a risk statement, and getting from one to the other requires hedging with an estimated beta — machinery that belongs to [position sizing](06-position-sizing-and-risk-budgeting.md). The regression also performs the strategy's official autopsy: even after crediting the accidental short-beta hedge, the alpha is +3.0% a year with a t-statistic of 1.15 — nothing, measured properly. And the turnover line prices the corpse's upkeep: 44% of the book turns over each month, the entire portfolio every 2.3 months, which means whatever microscopic edge exists would be paying full freight on one of the most expensive trading tempos in this part. Remember 44% — [lesson seven](07-portfolio-construction-and-transaction-costs.md) will multiply it by a cost per trade.

## The volatility risk premium

Front two. The VIX is, by construction, the strike of a 30-day variance swap on the S&P 500 — the annualized volatility the market will lock in today for the coming month, an identity the [options pricing module](../advanced/11-options-pricing.md) derives from Ito's lemma rather than asserting. The volatility risk premium is the claim that this insurance is persistently overpriced:

$$
\mathrm{VRP}_t \;=\; \sigma^{\text{implied}}_t \;-\; \sigma^{\text{realized}}_{t \to t+1} \;>\; 0
\quad \text{on average},
$$

because the natural buyers of protection (levered investors who cannot tolerate gap risk) outnumber the natural sellers, and the sellers must be paid for carrying the tail. Measured on every month in the sample:

```python
import numpy as np
import pandas as pd

vix = pd.read_parquet("data/part4.parquet")["VIX"].resample("ME").last()
r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()
rv = np.sqrt(252 * (r ** 2).resample("ME").mean()) * 100

gap = (vix.shift(1) - rv).dropna()
print(f"implied (month-start VIX) minus subsequent realized: "
      f"mean {gap.mean():+.1f} vol pts, positive in {(gap > 0).mean():.0%} of months")
print(f"worst month {gap.idxmin():%Y-%m}: {gap.min():+.1f} vol pts")
# => implied (month-start VIX) minus subsequent realized: mean +3.6 vol pts, positive in 82% of months
#    worst month 2020-03: -48.7 vol pts
```

The premium is as advertised: implied volatility exceeded the realized volatility that followed it by 3.6 points on average, and did so in 82% of months — a base rate that makes the previous lesson's tethered spread look coy. But the worst-month line is the entire risk profile in one number: March 2020 realized 48.7 points *more* than February's VIX promised. This is not a strategy result yet — vol points are not P&L — but the shape is already legible: an insurance business, collecting small premiums with high frequency and paying rare, enormous claims. The [returns lesson](../part-03-statistics/02-returns-and-distributions.md) taught the vocabulary for this shape; the next section prices it.

## Selling variance, with the record on

To turn the premium into P&L without an options desk, use the variance-swap identity directly: a short variance position with unit notional pays the strike minus realized variance, $K^2 - \sigma^2_{\text{realized}}$, and the VIX *is* the strike. One line of arithmetic per month, and the strategy's entire history — including its confessions — is on the record:

```python
import numpy as np
import pandas as pd

vix = pd.read_parquet("data/part4.parquet")["VIX"].resample("ME").last()
r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()
rv2 = 252 * (r ** 2).resample("ME").mean()
shortvol = ((vix.shift(1) / 100) ** 2 - rv2).dropna() * 100   # variance pts x100

print(f"shortvol: mean {shortvol.mean():+.2f}/month, "
      f"Sharpe {np.sqrt(12) * shortvol.mean() / shortvol.std():.2f}, "
      f"skew {shortvol.skew():+.1f}")
for m in ["2018-02", "2020-03"]:
    print(f"{m}: {shortvol.loc[m].iloc[0]:+.2f}  "
          f"(= {abs(shortvol.loc[m].iloc[0]) / shortvol.mean():.0f} average months)")
# => shortvol: mean +0.82/month, Sharpe 0.47, skew -7.6
#    2018-02: -5.69  (= 7 average months)
#    2020-03: -62.82  (= 76 average months)
```

A Sharpe of 0.47 — respectable — attached to a skew of **−7.6**, the most violently asymmetric return stream this course will produce. The two exhibit months translate the skew into biography: February 2018, the "Volmageddon" that extinguished several short-vol products in an afternoon, cost seven average months; March 2020 cost *seventy-six* — six and a half years of patiently collected premium, repaid in twenty trading days. This is `shortvol`, the fourth of the part's strategies, and its Sharpe ratio is close to meaningless as a summary: a statistic built on the first two moments has no vocabulary for a distribution whose entire character lives in the third and fourth ([Expected Shortfall](../appendix/part-18-quant-finance-applications/12-expected-shortfall.md) and [Extreme Value Theory](../appendix/part-18-quant-finance-applications/14-extreme-value-theory.md) supply the right instruments). The desk description is exact: selling variance is writing earthquake insurance — profitable in every year without an earthquake, and the earthquake is a matter of *when*.

## Term structure: contango pays, backwardation warns

The volatility market publishes a second number: VIX3M, the same strike at the three-month horizon. Its ratio to VIX describes the term structure's shape — upward-sloping (*contango*) when the market is calm and buying cheap near-dated insurance, inverted (*backwardation*) when panic bids the front month above everything behind it. Inversion is the market itself announcing that the earthquake may be underway — which suggests using it as an off-switch:

```python
import numpy as np
import pandas as pd

p4 = pd.read_parquet("data/part4.parquet")
ts = (p4["VIX3M"] / p4["VIX"]).dropna()
print(f"contango (VIX3M > VIX): {(ts > 1).mean():.0%} of days since {ts.index[0]:%Y-%m}")

vix = p4["VIX"].resample("ME").last()
contango = (p4["VIX3M"] > p4["VIX"]).resample("ME").last()
r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()
rv2 = 252 * (r ** 2).resample("ME").mean()
sv = (((vix.shift(1) / 100) ** 2 - rv2).dropna() * 100)
sv = sv[sv.index >= "2006-08"]
filt = sv.where(contango.shift(1).reindex(sv.index), 0.0)
for name, s in [("unconditional", sv), ("contango-only", filt)]:
    print(f"{name:13s}: mean {s.mean():+.2f}, Sharpe {np.sqrt(12) * s.mean() / s.std():.2f}, "
          f"worst {s.min():+.1f}")
# => contango (VIX3M > VIX): 89% of days since 2006-07
#    unconditional: mean +0.74, Sharpe 0.37, worst -62.8
#    contango-only: mean +1.09, Sharpe 1.55, worst -18.9
```

The curve sits in contango 89% of the time — calm is the default state, which is why there is a premium to collect at all. The filter's effect is the sharpest conditional result in this part: selling variance only when the *prior month-end* curve sloped upward raises the mean, cuts the worst month from −62.8 to −18.9, and quadruples the Sharpe to 1.55, because backwardation at the end of February 2020 ordered the strategy out of the market for March. Three honesty clauses before this number is enshrined. The filter is literature-standard and was not tuned here — but it is *one* trial on *one* history in which it happened to dodge the single worst month, and that dodge is doing much of the work; a history where the crash arrives without curve inversion (2018's single-day Volmageddon nearly qualifies) is charged nearly full price. The sample only begins in mid-2006, where VIX3M's record starts. And an off-switch that works 89% of the time still leaves the seller holding every claim that arrives *unannounced*. Conditioning reshapes the tail; it does not repeal it — the tail budget this strategy owes is [lesson six's](06-position-sizing-and-risk-budgeting.md) business.

## One book, two sleeves — a dilution lesson

The lesson's stated destination was a combined book: the cross-sectional sleeve and the volatility sleeve, scaled to equal volatility, correlation near zero, Sharpe rising in the combination. The data has other plans, and they are worth watching in detail:

```python
import numpy as np
import pandas as pd

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
p4 = pd.read_parquet("data/part4.parquet")
mp = p4[sectors].resample("ME").last()
mr = mp.pct_change()
ranks = mp.pct_change(11).shift(1).rank(axis=1)
w = ((ranks >= 7).astype(float) - (ranks <= 3).astype(float)) / 3
xsmom = (w.shift(1) * mr).sum(axis=1).loc["2001-02":]

vix = p4["VIX"].resample("ME").last()
r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()
rv2 = 252 * (r ** 2).resample("ME").mean()
sv = (((vix.shift(1) / 100) ** 2 - rv2).dropna() * 100)
contango = (p4["VIX3M"] > p4["VIX"]).resample("ME").last()
sv = sv[sv.index >= "2006-08"]
sv = sv.where(contango.shift(1).reindex(sv.index), 0.0)

both = pd.concat([xsmom, sv], axis=1, keys=["xsmom", "shortvol"]).dropna()
scaled = both / both.std()                       # full-sample vol scaling
book = scaled.mean(axis=1)
print(f"sleeve correlation {both['xsmom'].corr(both['shortvol']):+.2f}  "
      f"(common sample {both.index[0]:%Y-%m} on, n {len(both)})")
for name, s in [("xsmom", scaled["xsmom"]),
                ("shortvol", scaled["shortvol"]), ("book", book)]:
    print(f"{name:8s}: Sharpe {np.sqrt(12) * s.mean() / s.std():.2f}")
# => sleeve correlation -0.14  (common sample 2006-08 on, n 227)
#    xsmom   : Sharpe 0.02
#    shortvol: Sharpe 1.55
#    book    : Sharpe 1.19
```

Everything the diversification textbook asks for is present — two sleeves, correlation −0.14, equal volatility — and the combined book is *worse* than the better sleeve alone: 1.19 against 1.55. The arithmetic is unforgiving and general: adding an uncorrelated sleeve with zero expected return to a profitable book contributes no return and full variance, so it dilutes Sharpe exactly as surely as adding cash would dilute return. Diversification is a theorem about *edges*, not about return streams — combining helps when every sleeve has one, and mathematically cannot help when one doesn't. That is the real reason dead strategies must be identified and removed rather than "diversified over": a portfolio is not a shelter for strategies that cannot survive alone. (The scaling here also quietly used full-sample volatilities — a mild hindsight subsidy that flatters both sleeves equally; the honest real-time version belongs to lesson six.) The book this part carries forward keeps `shortvol` and sends `xsmom` to the morgue with its paperwork in order — where lesson five will perform the autopsy that explains the cause of death.

!!! warning "The volatility risk premium is an insurance business — priced by the years you collect, defined by the day you pay"
    Any strategy whose returns are small, frequent, and positive is short a tail somewhere, whether its author knows it or not. Before running one, write down the answer to a single question: which month in the historical record is your whole strategy's profit, repaid at once — and can you hold the position, the mandate, and the job through that month? A Sharpe of 1.55 with skew −7.6 is not a better version of a Sharpe of 0.5 with skew zero. It is a different business, and it must be sized like one.

!!! abstract "Key takeaways"
    - Sectors co-move at 0.58 average correlation with 63% of monthly variance in the common move; demeaning each month flips average correlation to −0.11, making a long-short book market-hedged by construction.
    - Textbook 12−1 cross-sectional momentum on nine sectors earned +1.0% a year, Sharpe 0.08, over 293 months: `xsmom` is dead on arrival — nine diversified baskets are the wrong habitat for a ranking effect documented in thousands of stocks.
    - Dollar-neutral is not beta-neutral: the book realized β = −0.25 because momentum systematically shorts high-beta sectors, and its alpha, +3.0% a year at t = 1.15, is statistically nothing — at 44% monthly one-way turnover.
    - The volatility risk premium is real and persistent: month-start implied exceeded subsequent realized by 3.6 vol points in 82% of months — with a −48.7-point worst month as the price of admission.
    - Selling variance monetizes it at Sharpe 0.47 with skew −7.6: March 2020 repaid 76 average months — six and a half years of premium — in twenty trading days.
    - The term-structure off-switch is the sharpest conditioner in the part: trading only in contango (89% of days) lifts the Sharpe to 1.55 and cuts the worst month to −18.9 — largely by dodging one month, a dependence to respect rather than celebrate.
    - Combining a dead sleeve with a live one diluted the book from 1.55 to 1.19 despite −0.14 correlation: diversification is a theorem about edges, and a portfolio is not a shelter for strategies that cannot survive alone.

## Where this goes next

Three strategies are now on the books — one modest survivor, one gross-Sharpe illusion awaiting its cost bill, one corpse with a pending autopsy — and a fourth, `shortvol`, that survives conditionally. [Seasonality and Calendar Effects](04-seasonality-and-calendar-effects.md) hunts the fifth in the most dangerous terrain in quantitative finance: patterns in the calendar itself, where the number of testable hypotheses is enormous, the mechanisms are mostly folklore, and the multiple-testing machinery from [Part III](../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) stops being a safety check and becomes the entire method. It is the part's designated graveyard — and the discipline it teaches is the one that keeps the other four strategies honest.
