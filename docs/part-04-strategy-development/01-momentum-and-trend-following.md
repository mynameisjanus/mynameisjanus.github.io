# Momentum and Trend Following

Part III spent six lessons interrogating a twelve-month momentum rule it never bothered to justify — the strategy was a crash-test dummy, picked up backwards, tested until its Sharpe of 0.30 stopped meaning anything. That backwardness was the point: most self-taught quants meet their first strategy the same way, as a backtest in search of a reason. This lesson rebuilds the same idea the right way around. The hypothesis comes first, with its economic mechanism and its falsification standard written down before any equity curve exists; the rule is then the *least* creative translation of the hypothesis into positions; and every evaluation runs through machinery you now own — error bars from [Hypothesis Testing and Multiple Testing](../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md), trial-counting discipline, and a standing suspicion of anything that looks too good.

The universe is the [Part III cache](../part-03-statistics/01-probability-and-random-variables.md) — SPY, TLT, GLD: an equity index, long bonds, and gold, three assets with distinct economic drivers, chosen because the momentum hypothesis claims to be *universal*. If the effect only shows up in the asset where we first noticed it, that is not a finding, it is a fit.

## A hypothesis, stated before the backtest

The hypothesis: an asset's return over the past year carries information about the sign of its return next month. The claimed mechanism is behavioral — investors underreact to slow-moving information, then herd into what has already moved — and the claim has unusual empirical pedigree, documented across equities, bonds, currencies, and commodities, and across a century of data. Formally, momentum asserts a conditional-expectation inequality,

$$
\mathbb{E}\big[r_{t+1} \mid r_{t-12 \to t} > 0\big] \;>\; \mathbb{E}\big[r_{t+1} \mid r_{t-12 \to t} \le 0\big],
$$

and the falsification standard, committed to now: the regression slope of next-month return on the sign of the trailing twelve-month return should be positive for *all three* assets. A sign that flips across assets kills the universality claim, whatever any single backtest says ([Random Walks](../appendix/part-08-stochastic-processes/11-random-walks.md) and [Martingales](../appendix/part-08-stochastic-processes/10-martingales.md) describe the null world where every such slope is zero):

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

px = pd.read_parquet("data/prices.parquet")
m = np.log(px[["SPY", "TLT", "GLD"]]).diff().resample("ME").sum(min_count=1)

for a in ["SPY", "TLT", "GLD"]:
    x = np.sign(m[a].rolling(12).sum()).dropna()
    y = (m[a].shift(-1) * 100).reindex(x.index).dropna()
    x = x.reindex(y.index)
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    print(f"{a}: slope {fit.params[a]:+.2f}%/mo on sign(12m)  "
          f"(t = {fit.tvalues[a]:+.2f}, n = {len(y)})")
# => SPY: slope +0.50%/mo on sign(12m)  (t = +1.11, n = 294)
#    TLT: slope +0.03%/mo on sign(12m)  (t = +0.11, n = 264)
#    GLD: slope +0.29%/mo on sign(12m)  (t = +0.89, n = 236)
```

Read this the way Part III taught you to. All three slopes are positive — the universality claim survives — and not one of them is individually significant; the strongest, SPY's half a percent per month, carries a t-statistic of 1.11. This is what a real effect of realistic size looks like in twenty-five years of monthly data: present, faint, and utterly incapable of impressing a hypothesis test asset by asset. The honest conclusions are two. First, the evidence is directional agreement, not statistical proof — three of three is suggestive precisely because the falsification bar was set before the regression ran. Second, an edge this thin per asset can only become a tradeable strategy through aggregation — across assets, and across time. That arithmetic of faint edges is the entire architecture of what follows.

## Time-series momentum, the right way around

The translation into a rule adds nothing the hypothesis did not claim: hold each asset long when its trailing 252-day return is positive, short when negative, sized equally, with yesterday's signal trading today's return. The SPY column of this computation is, character for character, Part III's crash-test dummy:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
sleeves = np.sign(rets.rolling(252).sum()).shift(1) * rets
tsmom = sleeves.mean(axis=1).dropna()

for a in ["SPY", "TLT", "GLD"]:
    s = sleeves[a].dropna()
    print(f"{a} sleeve: Sharpe {np.sqrt(252) * s.mean() / s.std():.2f}, "
          f"ann vol {np.sqrt(252) * s.std():.1%}, n {len(s)}")
corr = sleeves.corr()
print(f"sleeve corr: SPY-TLT {corr.loc['SPY','TLT']:+.2f}, "
      f"SPY-GLD {corr.loc['SPY','GLD']:+.2f}, TLT-GLD {corr.loc['TLT','GLD']:+.2f}")
print(f"tsmom: Sharpe {np.sqrt(252) * tsmom.mean() / tsmom.std():.2f}, "
      f"ann vol {np.sqrt(252) * tsmom.std():.1%}, n {len(tsmom)}")
# => SPY sleeve: Sharpe 0.30, ann vol 19.3%, n 6158
#    TLT sleeve: Sharpe 0.05, ann vol 14.6%, n 5514
#    GLD sleeve: Sharpe 0.17, ann vol 17.9%, n 4932
#    sleeve corr: SPY-TLT -0.04, SPY-GLD +0.06, TLT-GLD +0.08
#    tsmom: Sharpe 0.30, ann vol 12.2%, n 6158
```

The familiar 0.30 reappears on schedule, joined by two weaker siblings — TLT at 0.05 is barely distinguishable from noise, GLD at 0.17 not much better. The three sleeves are essentially uncorrelated, between −0.04 and +0.08, which is the diversification jackpot the hypothesis promised: three independent samples of the same faint effect. And yet the combined book's Sharpe is… 0.30, exactly where SPY alone stood. The benefit did not vanish — look at the volatility line. Equal-weighting three uncorrelated sleeves cut the book's volatility from 19.3% to 12.2% while averaging their returns, and returns averaged across a strong sleeve and a near-dead one is a dilution. Diversification has been collected as *risk reduction* and squandered as *return*, because equal dollars is the wrong allocation when sleeves differ this much. Converting that reclaimed risk budget back into return is a sizing decision, and it gets its own lesson — [Position Sizing and Risk Budgeting](06-position-sizing-and-risk-budgeting.md) will restore what this table left on the table. This four-line construction, `tsmom`, is the first of five strategies this part carries through to the validation gauntlet.

## The moving average is a slow sign

The trend-following folk toolkit — price above its moving average, fast average above slow — looks like a different species from the sign rule. It is the same animal. With log prices $p$, being above an $n$-day moving average is a statement about a weighted sum of past returns,

$$
p_t - \frac{1}{n}\sum_{i=0}^{n-1} p_{t-i} \;=\; \sum_{k=0}^{n-2} \frac{n-1-k}{n}\, r_{t-k},
$$

with triangular weights that favor recent returns. Every moving-average rule is a sign-of-weighted-past-returns rule with a particular smoothing; the choice among them is a choice of window shape, nothing deeper:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
spy = px["SPY"]
r = np.log(spy).diff().dropna()

rules = {
    "buy and hold": pd.Series(1.0, index=r.index),
    "10-month MA": (spy > spy.rolling(210).mean()).astype(float).shift(1),
    "50/200 cross": (spy.rolling(50).mean() > spy.rolling(200).mean()).astype(float).shift(1),
}
for name, pos in rules.items():
    s = (pos * r).dropna()
    eq = np.exp(s.cumsum())
    mdd = (eq / eq.cummax() - 1).min()
    print(f"{name:13s}: Sharpe {np.sqrt(252) * s.mean() / s.std():.2f}, "
          f"maxDD {mdd:.0%}, in market {pos.mean():.0%}")
# => buy and hold : Sharpe 0.38, maxDD -55%, in market 100%
#    10-month MA  : Sharpe 0.59, maxDD -23%, in market 70%
#    50/200 cross : Sharpe 0.59, maxDD -34%, in market 69%
```

Two filters with different folklore pedigrees land on the identical Sharpe of 0.59 — as they should, being reweightings of the same signal — and both beat buy-and-hold's 0.38. But the Sharpe is not where a desk's eye goes first. The filters spent only 70% of the time in the market and cut the maximum drawdown from −55% to −23%; a long-only investor who slept through 2008 in cash did most of the winning by *not being there*. That is the general character of trend overlays on equities: modest improvement in average return, dramatic improvement in the worst stretch — and the worst stretch, [Drawdown Probabilities](../appendix/part-18-quant-finance-applications/03-drawdown-probabilities.md) argues, is the number that actually retires strategies and traders.

## Breakouts and the anatomy of a trend trade

Channel breakouts — enter on a new 55-day high, exit on a new 20-day low, the skeleton of the 1980s Turtle system — make the individual *trade* visible in a way a continuously-held sign rule does not. On gold, long-only:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
gld = px["GLD"].dropna()
r = np.log(gld).diff()

raw = pd.Series(np.nan, index=gld.index)
raw[gld > gld.rolling(55).max().shift(1)] = 1.0   # 55-day breakout entry
raw[gld < gld.rolling(20).min().shift(1)] = 0.0   # 20-day channel exit
pos = raw.ffill().fillna(0.0).shift(1).fillna(0.0)
s = (pos * r).dropna()

entries = ((pos == 1) & (pos.shift(1) == 0)).sum()
trade_id = ((pos == 1) & (pos.shift(1) == 0)).cumsum()[pos == 1]
pnl = s[pos == 1].groupby(trade_id).sum()
days = s[pos == 1].groupby(trade_id).size()
print(f"trades {entries}, hit rate {(pnl > 0).mean():.0%}, "
      f"avg hold {days.mean():.0f} days")
print(f"avg win {pnl[pnl > 0].mean():+.1%}, avg loss {pnl[pnl <= 0].mean():+.1%}, "
      f"Sharpe {np.sqrt(252) * s.mean() / s.std():.2f}")
# => trades 50, hit rate 48%, avg hold 38 days
#    avg win +7.1%, avg loss -3.3%, Sharpe 0.37
```

Fifty trades in twenty years, and the system was wrong more often than right — a 48% hit rate — while remaining profitable, because the average winner (+7.1%) is more than twice the average loser (−3.3%). This asymmetry is not luck; it is the design. The exit channel *is* the stop, and its placement is structural: a trend that continues is held for months, a trend that fails is cut in weeks, so the rule mechanically truncates the loss distribution and lets the win distribution keep its tail. Note what this does to the psychology of running such a system: most trades lose, months pass between the wins that matter, and every behavioral instinct screams to take the +2% winner before it becomes a −1% loser — which would surgically remove the only part of the distribution that pays. Trend following is easy to specify and brutal to sit through, and the sitting is the moat.

## The lookback surface: plateaus, not peaks

Why 252 days and not 120, or 480? Part III's [multiple-testing lesson](../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) built a fifty-variant grid to show how easily a search finds phantom winners — its in-sample champion, the 270-day lookback, lost money out of sample. Now run the search from the builder's side, eyes open, on the three-asset book:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()

grid = {}
for lb in range(40, 501, 20):
    combo = (np.sign(rets.rolling(lb).sum()).shift(1) * rets).mean(axis=1).dropna()
    grid[lb] = np.sqrt(252) * combo.mean() / combo.std()
surf = pd.Series(grid)

print("lookback ", "  ".join(f"{lb}" for lb in [40, 120, 240, 360, 480]))
print("Sharpe   ", "  ".join(f"{surf[lb]:.2f}" for lb in [40, 120, 240, 360, 480]))
best = surf.idxmax()
plateau = surf[surf >= surf.max() - 0.10]
print(f"best cell {best} days (Sharpe {surf.max():.2f}), "
      f"within 0.10 of it: {plateau.index.min()}-{plateau.index.max()} days")
print(f"cells within 0.20 of the 240-day cell: "
      f"{(abs(surf - surf[240]) <= 0.20).sum()} of {len(surf)}")
# => lookback  40  120  240  360  480
#    Sharpe    -0.04  0.35  0.31  0.44  0.56
#    best cell 480 days (Sharpe 0.56), within 0.10 of it: 420-500 days
#    cells within 0.20 of the 240-day cell: 21 of 24
```

The surface's shape is the finding. Below about 80 days there is nothing — short-horizon momentum in these assets is dead on arrival. From 120 days up, every cell lives between roughly 0.3 and 0.56, and 21 of the 24 cells sit within 0.20 of the 240-day cell — which is to say, within one Part III error bar, the surface is *flat*. A flat, broadly positive plateau is exactly what a real but faint effect should produce; a single incandescent spike is what a fit to noise produces. The one feature that should raise your hand is that the best cell, 480 days, sits at the edge of the grid — edges are where searches hallucinate, because there is no neighbor on one side to vote the estimate down. We keep 252: it was named by the hypothesis and the literature before this grid existed, and nothing on the plateau is distinguishable from it. And the ledger entry, made now while it is cheap: this section ran 24 trials, the fifty-variant grid ran 50, and [Validation and Overfitting](08-validation-and-overfitting.md) will demand the full count ([Data Snooping Bias](../appendix/part-15-multiple-testing/04-data-snooping-bias.md) is the formal treatment).

## Paid in skew

Two return streams can share a Sharpe ratio and be entirely different products. What trend following actually sells is a reshaping of the return distribution:

```python
import numpy as np
import pandas as pd
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(rets.rolling(252).sum()).shift(1) * rets).mean(axis=1).dropna()
spy = rets["SPY"].dropna()

for name, s in [("SPY buy-hold", spy), ("tsmom", tsmom)]:
    monthly = s.resample("ME").sum()
    eq = np.exp(s.cumsum())
    mdd = (eq / eq.cummax() - 1).min()
    print(f"{name:12s}: daily hit {(s > 0).mean():.1%}, monthly skew "
          f"{stats.skew(monthly):+.2f}, maxDD {mdd:.0%}")
# => SPY buy-hold: daily hit 54.5%, monthly skew -0.63, maxDD -55%
#    tsmom       : daily hit 53.4%, monthly skew -0.06, maxDD -29%
```

The daily hit rates are nearly twins — 54.5% against 53.4% — so whatever separates these products, it is not the frequency of winning, the number every retail pitch leads with. The separation is in the monthly skew: the equity index carries the fat left tail every long-only investor is short (−0.63, the crash asymmetry [Heavy-Tailed Returns](../appendix/part-18-quant-finance-applications/13-heavy-tailed-returns.md) quantifies), while the trend book's skew is a flat −0.06 — the mechanical consequence of cutting losers and holding winners, now visible at portfolio level. The drawdown line prices the difference: −29% against −55%. This is the honest sales pitch for trend following, and note what it does *not* say: nothing about a higher mean. You are not paid more on average; you are paid *differently* — fewer catastrophic months in exchange for a steady bleed of whipsaw losses — and whether that trade is attractive depends on what the rest of your book looks like, which is a portfolio question deferred to [Portfolio Construction and Transaction Costs](07-portfolio-construction-and-transaction-costs.md).

## Why momentum persists

A century-old, publicly documented effect should have been arbitraged to dust — so any momentum book owes an account of why it survives publication. The behavioral account: information diffuses slowly, investors anchor on old prices and underreact, then late money extrapolates and overshoots — a mechanism that predicts momentum should weaken where attention is total and leverage is cheap. The risk-based account: momentum profits are compensation for crash exposure, because trend followers are effectively short the sharp reversal, as 2009's momentum crash demonstrated in equities. The two stories disagree about *when* the effect should fail, and the sample contains three natural experiments:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(rets.rolling(252).sum()).shift(1) * rets).mean(axis=1).dropna()

for yr in ["2008", "2020", "2022"]:
    t, s = tsmom.loc[yr].sum(), rets["SPY"].loc[yr].sum()
    print(f"{yr}: tsmom {t:+.1%}, SPY {s:+.1%}")
print(f"corr with SPY, all days: {tsmom.corr(rets['SPY']):+.2f}")
print(f"corr, SPY's worst decile: "
      f"{tsmom[rets['SPY'] < rets['SPY'].quantile(0.1)].corr(rets['SPY']):+.2f}")
# => 2008: tsmom +10.7%, SPY -45.9%
#    2020: tsmom +5.5%, SPY +16.8%
#    2022: tsmom -8.7%, SPY -20.1%
#    corr with SPY, all days: -0.30
#    corr, SPY's worst decile: -0.27
```

2008 is the exhibit that built the CTA industry: a year the index lost −45.9%, the trend book made +10.7%, riding short equities and long bonds for months — the slow, persistent kind of crisis momentum feeds on. 2020's crash was the opposite type, too fast for a 252-day signal to turn, and the book survived (+5.5%) mostly by being diversified rather than clairvoyant. 2022 is the honest miss: the year every institutional trend fund feasted on short bonds and long commodities, this three-ETF miniature lost −8.7%, chopped by signals flipping across a whipsawing year — breadth, not signal quality, separated the professionals from this toy, a gap [Feature and Signal Engineering](05-feature-and-signal-engineering.md) will make precise. The correlation lines carry the structural point: −0.30 to the index overall, still −0.27 inside the index's worst-decile days ([Regime Detection](../appendix/part-18-quant-finance-applications/16-regime-detection.md) formalizes the state-dependence). Whichever explanation you favor — and the crisis pattern fits both — the practical conclusion is the same: momentum's persistence is bound up with it being *painful to hold*, and pain is not arbitraged away, it is transferred to whoever can bear it.

!!! warning "The backtest was the last step of this lesson, not the first"
    Everything the backtest was allowed to do here was decided before it ran: the hypothesis named its mechanism, the falsification standard was committed in advance, the lookback was fixed by the claim rather than the surface, and the trials were counted as they happened. Reverse that order — surface first, story after — and the same code, the same data, and the same Sharpe ratios become unauditable. The difference between research and curve-fitting is not in the results; it is in the timestamps.

!!! abstract "Key takeaways"
    - The momentum hypothesis was committed before any backtest: positive slope of next-month return on sign of past year, in all three assets. It survived — three positive slopes — with the strongest t-statistic just 1.11: real effects of realistic size look faint asset by asset.
    - Translating the hypothesis literally gives `tsmom`: sleeves of Sharpe 0.30, 0.05, and 0.17 with pairwise correlations inside ±0.08, combining to Sharpe 0.30 at 12.2% vol — diversification banked as risk reduction, awaiting lesson six's sizing to become return.
    - Moving-average rules are triangularly-weighted sign rules: the 10-month MA and 50/200 cross both print Sharpe 0.59 versus buy-and-hold's 0.38, earning it mostly by truncating the −55% drawdown to −23%.
    - The Donchian system on gold won only 48% of its fifty trades and profited anyway: average winner +7.1% versus average loser −3.3%, the asymmetry that defines the trend trade.
    - The lookback surface is a plateau, not a peak: 21 of 24 cells sit within 0.20 — one error bar — of the 240-day cell, and the nominal best cell (480 days) sits suspiciously at the grid's edge. We keep 252 because the hypothesis, not the surface, chose it; the 24 trials go in the ledger.
    - Trend's product is skew, not mean: monthly skew −0.06 against the index's −0.63 at nearly identical daily hit rates, and a −29% versus −55% maximum drawdown.
    - Momentum survives publication because it is painful: +10.7% in 2008 against the index's −45.9%, but a −8.7% loss in whipsawing 2022 that breadth would have repaired — and a −0.27 correlation to the index precisely in its worst days.

## Where this goes next

Momentum bets that what moved keeps moving. [Mean Reversion and Pairs Trading](02-mean-reversion-and-pairs-trading.md) takes the opposite side: that some spreads are tethered, and that stretch predicts snap-back. The machinery transfers almost unchanged — a hypothesis with a named mechanism, a falsification standard, a literal rule, a trial ledger — but the statistical burden inverts: momentum needed evidence that prices *trend*, reversion needs evidence that a spread is *stationary*, and Part III's [time-series lesson](../part-03-statistics/03-time-series.md) already showed how treacherous that certification is. It is also where this part's universe finally grows beyond four tickers, because the first honest lesson of pairs trading is how many candidate pairs a screen must survive.
