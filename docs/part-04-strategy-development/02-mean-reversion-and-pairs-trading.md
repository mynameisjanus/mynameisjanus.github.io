# Mean Reversion and Pairs Trading

The [previous lesson](01-momentum-and-trend-following.md) bet that what has moved keeps moving. This one takes the opposite side — that some prices are tethered, and that stretch predicts snap-back — and the first thing to notice is that both bets cannot be right *about the same thing at the same horizon*. They coexist because markets are not one process: the same index that trends over a year mean-reverts over days, and a spread between two near-identical instruments mean-reverts violently at every horizon. So the hypothesis this time is not "prices revert" — that claim is false as stated — but something sharper: reversion lives where a *mechanism* enforces a fair value, and the strength of the tether can be measured before a single trade is placed.

The mechanism menu is short. At the days horizon in a liquid index, reversion is the footprint of liquidity provision — market makers paid a spread to lean against order flow. Between instruments holding the same assets, it is arbitrage itself: the SPY–IVV spread from Part III's [time-series lesson](../part-03-statistics/03-time-series.md), held inside a few dozen basis points by the creation-redemption machinery, is the cleanest tether in the market and this lesson's laboratory. The falsification standard, committed now: a spread qualifies only if stationarity tests reject at the horizon we intend to trade *and* the fitted half-life is short enough to survive realistic costs. Anything else is two trends coinciding.

## Where mean reversion lives

Before trading reversion, locate it. The lag-1 autocorrelation of SPY returns, measured at three horizons, maps the territory:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

for label, agg in [("daily", r),
                   ("weekly", r.resample("W").sum(min_count=1).dropna()),
                   ("monthly", r.resample("ME").sum(min_count=1).dropna())]:
    rho = agg.autocorr(1)
    se = 1 / np.sqrt(len(agg))
    print(f"{label:8s} lag-1 autocorr {rho:+.3f}  ({rho / se:+.1f} se, n {len(agg)})")
# => daily    lag-1 autocorr -0.086  (-6.9 se, n 6410)
#    weekly   lag-1 autocorr -0.075  (-2.7 se, n 1331)
#    monthly  lag-1 autocorr +0.025  (+0.4 se, n 306)
```

There is the whole market-structure story in three rows. Daily returns anti-correlate at −0.086 — nearly seven standard errors, one of the few effects in this course that clears any multiple-testing bar you like — weekly returns still lean negative at −0.075, and by the monthly horizon the sign has flipped to a statistically-nothing +0.025, the on-ramp to the twelve-month momentum the previous lesson traded. Two cautions before excitement sets in. First, a −0.086 autocorrelation prices an effect measured in basis points per day, exactly the scale where transaction costs live; whether any of it survives the spread is a question [Portfolio Construction and Transaction Costs](07-portfolio-construction-and-transaction-costs.md) will answer with a straight face. Second, index-level short-horizon reversion has been decaying for decades — it was several times stronger in the 1990s — so the mechanism is real but the rent it pays is shrinking. The durable version of this trade lives not in one index against time, but between instruments.

## Three instruments, one verdict

Part III supplied three ways to interrogate memory: the variance ratio, which asks whether multi-day variance grows proportionally; the Hurst exponent, which fits the growth law

$$
\operatorname{Var}(r_\tau) \;\propto\; \tau^{2H},
$$

so that $H = \tfrac12$ is a random walk, $H > \tfrac12$ trends, and $H < \tfrac12$ reverts; and the ADF test, which asks whether the level series has a unit root at all. Aim all three at the index and at the spread:

```python
import numpy as np
import pandas as pd
from arch.unitroot import VarianceRatio
from statsmodels.tsa.stattools import adfuller

px = pd.read_parquet("data/prices.parquet")
spy = np.log(px["SPY"]).dropna()
spread = (np.log(px["SPY"]) - np.log(px["IVV"])).dropna()

def hurst(x, lags=range(2, 100)):
    tau = [x.diff(l).dropna().std() for l in lags]
    return np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]

for name, x in [("log SPY", spy), ("SPY-IVV spread", spread)]:
    vr = VarianceRatio(x, lags=5)
    print(f"{name:15s}: VR(5) {vr.vr:.2f} (p {vr.pvalue:.2f}), "
          f"Hurst {hurst(x):.2f}, ADF p {adfuller(x, maxlag=5)[1]:.3f}")
# => log SPY        : VR(5) 0.83 (p 0.01), Hurst 0.46, ADF p 0.992
#    SPY-IVV spread : VR(5) 0.21 (p 0.00), Hurst 0.04, ADF p 0.000
```

The index row says: a level that is emphatically not stationary (ADF p = 0.992 — SPY does not return to any fixed price) built from increments with mild anti-persistence (VR 0.83, Hurst 0.46) — the same short-horizon reversion the autocorrelations found, now in growth-law form. The spread row is a different organism: five-day variance is 21% of what a random walk would accumulate, the Hurst exponent is 0.04 — barely above the theoretical floor — and the unit root is rejected at any significance you care to name. One reconciliation is owed. Part III showed this very pair *failing* Engle-Granger at p = 0.46, and both results are correct: the EG test estimates the hedge ratio and lets its lag machinery stare at the spread's slow fee-drift component, while here the ratio is imposed (one-for-one, from the instruments' construction) and `maxlag=5` asks about the horizon a trader holds. Tests see the horizon their lag structure selects — that was Part III's resolution, and this table is what it looks like from the practitioner's side.

## Ornstein-Uhlenbeck: the physics of the tether

Once a spread earns "stationary," it deserves a model with parameters worth estimating. The workhorse is the Ornstein-Uhlenbeck process (solved properly, and shown to make this discrete fit *exact* rather than approximate, in the [stochastic calculus module](../advanced/03-stochastic-calculus.md)),

$$
dX_t \;=\; \theta\,(\mu - X_t)\,dt \;+\; \sigma\,dW_t ,
$$

a random walk on a spring: $\theta$ is the stiffness, $\mu$ the resting point, $\sigma$ the noise constantly re-stretching it ([Brownian Motion](../appendix/part-08-stochastic-processes/08-brownian-motion.md) supplies the $dW_t$). Its exact daily discretization is an AR(1) with $\rho = e^{-\theta \Delta t}$, which means the whole continuous-time apparatus is fitted by one lag regression ([Simple Linear Regression](../appendix/part-13-regression/01-simple-linear-regression.md), whose attenuation result is why a noisily measured spread reports a half-life shorter than the truth):

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

px = pd.read_parquet("data/prices.parquet")
s = ((np.log(px["SPY"]) - np.log(px["IVV"])) * 1e4).dropna()  # basis points

lag = s.shift(1).dropna()
fit = sm.OLS(s.reindex(lag.index), sm.add_constant(lag)).fit()
rho, c = fit.params.iloc[1], fit.params.iloc[0]
theta = -252 * np.log(rho)                      # mean-reversion speed, per year
mu = c / (1 - rho)                              # long-run level
print(f"AR(1) rho {rho:.2f}  ->  theta {theta:.0f}/yr, mu {mu:+.0f} bp, "
      f"equilibrium std {s.std():.0f} bp")
# => AR(1) rho 0.82  ->  theta 51/yr, mu -7 bp, equilibrium std 23 bp
```

The $\rho$ of 0.82 is Part III's number reporting for duty in its third lesson, now wearing physical units: a spring stiffness of 51 per year — shocks decay dozens of times over within a trading year — around a resting point of −7 basis points, which is not zero because SPY and IVV differ in fees and dividend timing; the tether has a *bias*, and a trader who assumes symmetry around zero donates that bias to the other side. The one number to memorize is the last: the equilibrium standard deviation is 23 basis points. Every profit this trade will ever produce is a fraction of a 23 bp leash, which is why this lesson keeps deferring, with increasing menace, to the cost accounting of lesson seven.

## The half-life sets the holding period

Stiffness has a more useful currency: time. An OU displacement decays to half its size in $t_{1/2} = \ln 2 / \theta$ — equivalently $\ln(0.5)/\ln \rho$ in daily AR(1) terms — and that single number dictates the trade's entire tempo: how long positions are held, how fast the z-score window may be, how quickly a loss must be recognized as regime change rather than opportunity ([Hitting and First-Passage Times](../appendix/part-18-quant-finance-applications/04-hitting-and-first-passage-times.md) makes the waiting-time math exact). Compute it for the tethered pair and for two impostors:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")

def half_life(spread):
    s = spread.dropna()
    rho = s.autocorr(1)
    return np.log(0.5) / np.log(rho) if 0 < rho < 1 else np.inf

for a, b in [("SPY", "IVV"), ("TLT", "GLD"), ("SPY", "GLD")]:
    sp = np.log(px[a]) - np.log(px[b])
    print(f"{a}-{b}: AR(1) rho {sp.dropna().autocorr(1):.3f}, "
          f"half-life {half_life(sp):,.1f} days")
# => SPY-IVV: AR(1) rho 0.817, half-life 3.4 days
#    TLT-GLD: AR(1) rho 0.999, half-life 631.0 days
#    SPY-GLD: AR(1) rho 0.999, half-life 1,058.4 days
```

The 3.4-day half-life makes its third appearance in the course — Part III derived it as the resolution of the cointegration paradox; here it becomes a design input. A spread that halves its displacement in 3.4 days is tradeable weekly and demands a z-score window of a few dozen days (the 60-day window used below spans about seventeen half-lives — enough to estimate the resting position, short enough to adapt). The impostor rows show what the number looks like when there is no tether: TLT–GLD and SPY–GLD print $\rho = 0.999$ and half-lives of two and four *years* — which is not "slow mean reversion," it is a unit root in costume, the arithmetic artifact of computing $\ln(0.5)/\ln\rho$ on a spread that never rejected nonstationarity in the first place. A half-life is only as meaningful as the stationarity evidence behind it. Order matters: test first, then fit.

## Pairs need a universe

Everything so far ran on one blessed pair that was *known* to be tethered by construction. Real pairs trading starts from the opposite position — many candidates, no guarantees — and the four-ticker cache cannot pose that problem honestly. So Part IV gets its own frozen universe, built once, exactly as Part III built its cache:

```python
# one-time download — requires a network connection
import pandas as pd
import yfinance as yf

tickers = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
           "IWM", "EFA", "EEM", "^VIX", "^VIX3M"]
px = yf.download(tickers, start="2000-01-01", end="2025-07-01",
                 auto_adjust=True)["Close"]
px = px[tickers].rename(columns={"^VIX": "VIX", "^VIX3M": "VIX3M"})
px.to_parquet("data/part4.parquet")

print(px.shape)
print("sectors XLK-XLU complete from", px["XLK"].first_valid_index().date())
print(", ".join(f"{c} {px[c].first_valid_index():%Y-%m}"
                for c in ["IWM", "EFA", "EEM", "VIX", "VIX3M"]))
# => (6411, 14)
#    sectors XLK-XLU complete from 2000-01-03
#    IWM 2000-05, EFA 2001-08, EEM 2003-04, VIX 2000-01, VIX3M 2006-07
```

The nine SPDR sector funds cover the full window; small caps, developed international, and emerging markets arrive raggedly — the same first-act data hygiene Part III insisted on, and every panel computation below must `dropna()` with eyes open. Two columns are not prices at all: VIX and its three-month sibling are volatility *indexes*, uninvestable directly, stored now because the [next lesson](03-cross-sectional-and-volatility-strategies.md) needs them. The doctrine transfers unchanged: this download runs once, the file is frozen, and every number in the rest of Part IV comes from the file — the vendor rewrites history through dividend adjustments; a cache does not.

## The screen and the spurious-pair trap

Thirteen investable series — SPY plus the twelve new equity funds — make 78 possible pairs, and 78 is where pairs trading stops being statistics and becomes multiple testing ([Multiple Comparisons](../appendix/part-15-multiple-testing/01-multiple-comparisons.md)). Screen every pair with Engle-Granger on 2010–2019, then confront the survivors with data they have never seen:

```python
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint

uni = pd.concat([pd.read_parquet("data/prices.parquet")[["SPY"]],
                 pd.read_parquet("data/part4.parquet")
                   .drop(columns=["VIX", "VIX3M"])], axis=1)
logp = np.log(uni)

fit = logp.loc["2010":"2019"]
pairs_all = list(combinations(uni.columns, 2))
pv = {p: coint(fit[p[0]], fit[p[1]])[1] for p in pairs_all}
hits = sorted([p for p, v in pv.items() if v < 0.05], key=pv.get)
print(f"{len(hits)} of {len(pairs_all)} pairs reject at 5% "
      f"(luck alone would find {0.05 * len(pairs_all):.1f})")
survive = [p for p in hits
           if coint(logp.loc["2020":, p[0]], logp.loc["2020":, p[1]])[1] < 0.05]
print("best in-sample:", ", ".join(f"{a}-{b}" for a, b in hits[:4]))
print(f"still cointegrated on 2020-2025: {len(survive)} of {len(hits)}",
      [f"{a}-{b}" for a, b in survive])
# => 12 of 78 pairs reject at 5% (luck alone would find 3.9)
#    best in-sample: XLI-XLB, XLB-IWM, XLI-IWM, IWM-EFA
#    still cointegrated on 2020-2025: 1 of 12 ['SPY-XLI']
```

Twelve rejections against an expected 3.9 under the null: the screen found *something*, and the hits even look sensible — industrials against materials, the index against its own sectors, pairs whose holdings overlap. Then the out-of-sample line lands: one of twelve still qualifies five years later. Read that as the two traps it contains. The statistical trap is the 3.9 — roughly a third of the in-sample list was expected to be luck, and luck does not persist ([Data Snooping Bias](../appendix/part-15-multiple-testing/04-data-snooping-bias.md)). The economic trap is subtler: even the genuine relationships were mostly decade-scale correlations between sector fortunes — tethers with no enforcement mechanism, nothing standing ready to close a gap — and a tether nobody enforces is free to dissolve when the economy rotates, which 2020 promptly demonstrated. The screen's real output is not twelve candidates; it is a reminder that a p-value cannot supply the thing the hypothesis lacks: a *reason* for the spread to close. Fee-differing share classes, dual listings, holding-company discounts, index-versus-basket — the tradeable pairs universe is the short list where someone is paid to enforce the price, and it is assembled by reading prospectuses, not p-values.

## Trading the spread

The rule, finally — and deliberately last. Enter when the z-score stretches past two, exit when the spread crosses its mean, next close execution. Applied to the one pair with a mechanism, and then, as a control, to the sector pairs the screen liked best:

```python
import numpy as np
import pandas as pd

def zpairs(spread):
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    raw = pd.Series(np.nan, index=z.index)
    raw[z > 2], raw[z < -2] = -1.0, 1.0
    raw[np.sign(z) != np.sign(z.shift(1))] = 0.0
    pos = raw.ffill().fillna(0.0).shift(1)
    return (pos * spread.diff()).dropna(), pos

px = pd.read_parquet("data/prices.parquet")
pairs, pos = zpairs((np.log(px["SPY"]) - np.log(px["IVV"])).dropna())
trades = ((pos != 0) & (pos.shift(1) == 0)).sum()
print(f"SPY-IVV: Sharpe {np.sqrt(252) * pairs.mean() / pairs.std():.2f}, "
      f"ann ret {252 * pairs.mean() * 1e4:.0f} bp, {trades} round trips, "
      f"in market {(pos != 0).mean():.0%}")
p4 = pd.read_parquet("data/part4.parquet")
for a, b in [("XLI", "XLB"), ("XLK", "XLY"), ("XLF", "XLV")]:
    s2, _ = zpairs((np.log(p4[a]) - np.log(p4[b])).dropna())
    eq = s2.cumsum()
    print(f"{a}-{b}: Sharpe {np.sqrt(252) * s2.mean() / s2.std():+.2f}, "
          f"maxDD {(eq - eq.cummax()).min() * 1e4:.0f} bp")
# => SPY-IVV: Sharpe 1.23, ann ret 154 bp, 168 round trips, in market 9%
#    XLI-XLB: Sharpe +0.42, maxDD -2689 bp
#    XLK-XLY: Sharpe +0.08, maxDD -9451 bp
#    XLF-XLV: Sharpe -0.13, maxDD -11982 bp
```

A Sharpe of 1.23 — the best headline number in the course so far, and the first thing to do is read its fine print. The strategy is in the market 9% of the time and extracts 154 basis points a year through 168 round trips on a spread whose entire equilibrium width is 23 bp; it books about 23 basis points per round trip — but only under an execution assumption this backtest quietly makes and [lesson seven](07-portfolio-construction-and-transaction-costs.md) will examine ruthlessly: that fills happen at the very closing print that generated the signal. It also trades an arbitrage whose natural owners run creation-redemption at costs no outsider matches. This is `pairs`, the second of the part's five strategies, carried forward *because* its gross-versus-net gap will be instructive: [lesson seven](07-portfolio-construction-and-transaction-costs.md) collects the promissory note. The control rows are the other half of the argument — the identical rule on the screen's favorite sector pairs produces a fraction of the Sharpe with drawdowns ten to fifty times deeper, because without a tether, "stretched" is not a signal, and averaging into a stretching spread is how pairs desks die. Hence the stop policy, stated as doctrine: a position older than a few half-lives (here, two weeks) is evidence the spring broke, not evidence of a better entry; a z-score beyond four is a structural-break alarm, not a stronger signal; and a broken pair is exited whole, never one leg at a time.

!!! warning "A spread that broke was never yours — it was two trends coinciding"
    Every pairs blowup in the folklore has the same autopsy: a spread that passed a stationarity screen, widened past every historical precedent, and was averaged into all the way down, because the trader treated a statistical artifact as a law of nature. The tests in this lesson measure history; only a mechanism — someone paid to enforce the price — constrains the future. If you cannot name the enforcer, the correct size in the pair is zero, whatever the p-value says.

!!! abstract "Key takeaways"
    - Mean reversion is horizon-local: SPY daily autocorrelation is −0.086 at nearly seven standard errors, weekly −0.075, monthly +0.025 — reversion at days and momentum at months coexist in one instrument.
    - Three instruments agree on the tethered spread — variance ratio 0.21, Hurst 0.04, ADF p = 0.000, against the index's 0.83 / 0.46 / 0.992 — and the apparent conflict with Part III's Engle-Granger failure dissolves once the known hedge ratio is imposed and the test asks about the trading horizon.
    - The OU fit prices the tether: ρ = 0.82 maps to a stiffness of 51 per year around a −7 bp resting point with a 23 bp equilibrium width — every profit this trade produces is a fraction of a 23 bp leash.
    - The 3.4-day half-life is a design input, setting weekly tempo and the 60-day z-window; TLT-GLD's "631-day half-life" is a unit root in costume — test stationarity first, or the half-life is fiction.
    - The part's universe arrives frozen: 14 series, nine sector funds complete from 2000, ragged starts for IWM, EFA, and EEM, and two uninvestable volatility indexes stored for the next lesson.
    - The screen certified 12 of 78 pairs against 3.9 expected by luck — and one of twelve survived 2020–2025, because a p-value cannot supply an enforcement mechanism.
    - The z-score rule earns Sharpe 1.23 on SPY-IVV — 154 bp a year, 23 bp per round trip, 9% of days in the market — while the same rule on screen-blessed sector pairs prints drawdowns to −11,982 bp: the rule was never the edge; the tether was.

## Where this goes next

Momentum and reversion have so far been *time-series* bets — each asset judged against its own past. [Cross-Sectional and Volatility Strategies](03-cross-sectional-and-volatility-strategies.md) rotates the question ninety degrees: not "will this asset go up," but "which of these assets will beat the others" — a formulation that cancels the market's common move and manufactures breadth out of a single date. The universe built in this lesson is exactly what that requires, nine sector funds deep. And the two uninvestable columns riding along in the cache — VIX and its three-month cousin — open the second front: strategies on volatility itself, where the thing being bought and sold is the gap between the price of insurance and the disasters that actually arrive.
