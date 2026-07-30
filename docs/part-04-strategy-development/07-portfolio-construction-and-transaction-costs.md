# Portfolio Construction and Transaction Costs

Six lessons of this part have produced five strategies, each priced in a vacuum: its own Sharpe, its own history, its own generous assumptions. This lesson does the two things a real desk does before any of those numbers are believed. First it *assembles the book* — all five sleeves on one balance sheet, where correlations pay dividends and offsetting trades cancel before they cost anything. Then it *presents the bill*: execution assumptions made explicit and stress-tested, a transaction cost model with its parameters stated where they can be attacked, market impact priced by the square-root law, and the rebalancing schedule adjusted so the book stops paying for precision it does not need.

The part's structure has been building to this lesson's final table. Every gross number so far was a claim; net of realistic execution and costs is where claims become strategies. Two of the five will not survive the paragraph in which the accounting is done — and one of the two deaths will come from an assumption most backtests never even write down.

## Correlation assembles the book

Put all five sleeves on a common monthly grid — the longest code block in the course, deliberately, because it *is* the whole part in one place — and measure what they are to each other:

```python
import numpy as np
import pandas as pd

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
px = pd.read_parquet("data/prices.parquet")
p4 = pd.read_parquet("data/part4.parquet")
r3 = np.log(px[["SPY", "TLT", "GLD"]]).diff()
rs = np.log(px["SPY"]).diff().dropna()

tsmom = (np.sign(r3.rolling(252).sum()).shift(1) * r3).mean(axis=1)
spread = (np.log(px["SPY"]) - np.log(px["IVV"])).dropna()
z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
raw = pd.Series(np.nan, index=z.index)
raw[z > 2], raw[z < -2] = -1.0, 1.0
raw[np.sign(z) != np.sign(z.shift(1))] = 0.0
pairs = raw.ffill().fillna(0.0).shift(1) * spread.diff()
mp = p4[sectors].resample("ME").last()
ranks = mp.pct_change(11).shift(1).rank(axis=1)
w = ((ranks >= 7).astype(float) - (ranks <= 3).astype(float)) / 3
xsmom = (w.shift(1) * mp.pct_change()).sum(axis=1).loc["2001-02":]
vix = p4["VIX"].resample("ME").last()
rv2 = 252 * (rs ** 2).resample("ME").mean()
sv = (((vix.shift(1) / 100) ** 2 - rv2) * 100).loc["2006-08":]
sv = sv.where((p4["VIX3M"] > p4["VIX"]).resample("ME").last()
              .shift(1).reindex(sv.index), 0.0)
pim = rs.groupby(rs.index.to_period("M")).cumcount() + 1
rev = rs[::-1].groupby(rs[::-1].index.to_period("M")).cumcount()[::-1] + 1
tom = rs.where((pim <= 3) | (rev == 1), 0.0)

m = pd.concat([tsmom.resample("ME").sum(min_count=1),
               pairs.resample("ME").sum(min_count=1), xsmom, sv,
               tom.resample("ME").sum(min_count=1)],
              axis=1, keys=["tsmom", "pairs", "xsmom", "svol", "tom"]).dropna()
print(m.corr().round(2).to_string())
book = (m / m.std()).mean(axis=1)
print(f"common sample {m.index[0]:%Y-%m} on, n {len(m)} months")
print(f"equal-vol book of all five: Sharpe {np.sqrt(12) * book.mean() / book.std():.2f}")
# =>        tsmom  pairs  xsmom  svol   tom
#    tsmom   1.00  -0.03   0.15 -0.01  0.04
#    pairs  -0.03   1.00   0.04 -0.29 -0.13
#    xsmom   0.15   0.04   1.00 -0.14 -0.04
#    svol   -0.01  -0.29  -0.14  1.00  0.34
#    tom     0.04  -0.13  -0.04  0.34  1.00
#    common sample 2006-08 on, n 227 months
#    equal-vol book of all five: Sharpe 1.51
```

The matrix is the asset a multi-strategy desk actually owns: ten pairwise correlations, all between −0.29 and +0.34, most near zero — five return streams that genuinely do not share a factor ([Covariance](../appendix/part-04-expectation-and-moments/04-covariance.md) and [Portfolio Risk Simulation](../appendix/part-18-quant-finance-applications/10-portfolio-risk-simulation.md) formalize why that is worth money). Two cells reward reading: `pairs`-`svol` at −0.29 (the pair trade earns in turbulence, when spreads stretch; short vol bleeds there) and `tom`-`svol` at +0.34 (both are long-calm bets wearing different costumes — correlation finding a shared exposure the strategy labels hid). The book prints Sharpe 1.51 — below the 1.55 its best sleeve managed alone in [lesson three](03-cross-sectional-and-volatility-strategies.md), the dilution theorem collecting its toll a third time. But treat 1.51 as fiction for a deeper reason: it averages one dead sleeve, one cadaver, and one number — `pairs` at Sharpe 1.20 — resting on an execution assumption that this lesson is about to take apart. This table is the marketing deck. The rest of the lesson is due diligence.

## Netting: one book, not five

Three of the sleeves trade the same instrument — the trend book, the calendar book, and one leg of the pair all transact SPY — and running them as separate accounts means paying for trades a single book would cancel internally:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rs = np.log(px["SPY"]).diff().dropna()
spy_trend = np.sign(rs.rolling(252).sum()) / 3
spread = (np.log(px["SPY"]) - np.log(px["IVV"])).dropna()
z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
raw = pd.Series(np.nan, index=z.index)
raw[z > 2], raw[z < -2] = -1.0, 1.0
raw[np.sign(z) != np.sign(z.shift(1))] = 0.0
pairs_spy = raw.ffill().fillna(0.0)                  # the pair's SPY leg
pim = rs.groupby(rs.index.to_period("M")).cumcount() + 1
rev = rs[::-1].groupby(rs[::-1].index.to_period("M")).cumcount()[::-1] + 1
tom_spy = ((pim <= 3) | (rev == 1)).astype(float)

pos = pd.concat([spy_trend, pairs_spy, tom_spy], axis=1,
                keys=["trend", "pairs", "tom"]).dropna()
gross_sep = pos.abs().sum(axis=1).mean()
gross_net = pos.sum(axis=1).abs().mean()
turn_sep = pos.diff().abs().sum(axis=1).mean() * 252
turn_net = pos.sum(axis=1).diff().abs().mean() * 252
print(f"avg gross SPY exposure: separate books {gross_sep:.2f}, "
      f"one netted book {gross_net:.2f}  ({1 - gross_net / gross_sep:.0%} less)")
print(f"SPY notional traded/yr: separate {turn_sep:.1f}x, "
      f"netted {turn_net:.1f}x  ({1 - turn_net / turn_sep:.0%} less)")
# => avg gross SPY exposure: separate books 0.62, one netted book 0.56  (9% less)
#    SPY notional traded/yr: separate 39.9x, netted 37.4x  (6% less)
```

The mechanism works and the savings are honest but modest — 9% less gross exposure to finance, 6% less notional crossing the market — and the modesty is itself the finding. Netting pays in proportion to how often sleeves hold the *same asset in opposite directions at the same time*, and these three overlap rarely: the pair is in the market 9% of days, the calendar book 19%, and the trend book switches sides only every year or two. Scale the sleeve count and the picture changes completely — a shop running forty strategies over one universe nets constantly, and internal crossing becomes a line item worth basis points of AUM, which is one structural reason multi-strategy firms can run strategies that die as standalone funds. The precondition is organizational, not mathematical: positions must live in *one* book that trades the net. Five separate accounts with five brokers cancel nothing, whatever the correlation matrix says.

## Execution assumptions decide the number

Every daily backtest in this course fills orders at a closing price. The unexamined question is *which* close — the close that generated the signal (achievable with intraday infrastructure that computes the signal minutes before the bell and sends a market-on-close order), or the next day's (the pen-and-paper guarantee, achievable by anyone). For a slow strategy the distinction is trivia. For a fast one, it is the strategy:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
spread = (np.log(px["SPY"]) - np.log(px["IVV"])).dropna()
z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
raw = pd.Series(np.nan, index=z.index)
raw[z > 2], raw[z < -2] = -1.0, 1.0
raw[np.sign(z) != np.sign(z.shift(1))] = 0.0
base = raw.ffill().fillna(0.0)
r3 = np.log(px[["SPY", "TLT", "GLD"]]).diff()
tsig = np.sign(r3.rolling(252).sum())

for name, lag in [("fill at the signal close", 1), ("fill at the next close", 2)]:
    p = (base.shift(lag) * spread.diff()).dropna()
    trips = ((base.shift(lag) != 0) & (base.shift(lag).shift(1) == 0)).sum()
    t = ((tsig.shift(lag) * r3).mean(axis=1)).dropna()
    print(f"{name:24s}: pairs Sharpe {np.sqrt(252) * p.mean() / p.std():+.2f} "
          f"({p.sum() * 1e4 / trips:+.1f} bp/trip)   "
          f"tsmom Sharpe {np.sqrt(252) * t.mean() / t.std():+.2f}")
# => fill at the signal close: pairs Sharpe +1.23 (+23.0 bp/trip)   tsmom Sharpe +0.30
#    fill at the next close  : pairs Sharpe +0.19 (+2.8 bp/trip)   tsmom Sharpe +0.37
```

One row of lag, two opposite verdicts. The trend book does not care — 0.30 becomes 0.37, a difference inside the error bar (leaning positive because [lesson two's](02-mean-reversion-and-pairs-trading.md) short-horizon reversal mildly rewards the later entry). The pairs book is *destroyed*: 23.0 basis points per round trip collapses to 2.8, Sharpe 1.23 to 0.19, because a spread with a 3.4-day half-life does most of its snapping back in the first day, and the fill assumption decides who owns that day. Now [lesson two's](02-mean-reversion-and-pairs-trading.md) promissory note is paid in full: the Sharpe of 1.23 was real arithmetic resting on the quiet assumption of signal-close fills — infrastructure the ETF arbitrage community has and a daily-data backtest does not. The general law: **execution sensitivity scales with signal speed**, so every backtest owes its reader one sentence — *filled when, at what price* — and every fast strategy owes the stress test this table performs. From here on, `pairs` is graded at its honest, next-close number: 18 basis points a year, gross.

## A cost model you can defend

Costs need a model that is *simple, stated, and auditable* — false precision in cost modeling is its own overfitting surface. The defensible minimum: each one-way trade pays the instrument's half-spread plus commission, with parameters on the table where they can be attacked. Charge each sleeve its measured trading at those rates:

```python
import numpy as np
import pandas as pd

HS = {"SPY": 0.5, "IVV": 1.0, "TLT": 1.0, "GLD": 1.0, "SEC": 2.0}  # half-spread bp
COMM = 0.2                                                         # bp per trade

sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
px = pd.read_parquet("data/prices.parquet")
p4 = pd.read_parquet("data/part4.parquet")
r3 = np.log(px[["SPY", "TLT", "GLD"]]).diff()
w3 = (np.sign(r3.rolling(252).sum()) / 3).dropna()
drag_tsmom = sum((w3[a].diff().abs().sum() / (len(w3) / 252)) * (HS[a] + COMM)
                 for a in ["SPY", "TLT", "GLD"])
mp = p4[sectors].resample("ME").last()
ranks = mp.pct_change(11).shift(1).rank(axis=1)
w9 = (((ranks >= 7).astype(float) - (ranks <= 3).astype(float)) / 3).dropna()
drag_xsmom = w9.diff().abs().sum(axis=1).mean() * 12 * (HS["SEC"] + COMM)
drag_pairs = 6.7 * 4 * ((HS["SPY"] + COMM) + (HS["IVV"] + COMM)) / 2
drag_tom = 24 * (HS["SPY"] + COMM)
print(f"tsmom: {drag_tsmom:.0f} bp/yr   xsmom: {drag_xsmom:.0f} bp/yr   "
      f"pairs: {drag_pairs:.0f} bp/yr (unlevered)   tom: {drag_tom:.0f} bp/yr")
# => tsmom: 11 bp/yr   xsmom: 22 bp/yr   pairs: 25 bp/yr (unlevered)   tom: 17 bp/yr
```

Read the table against each sleeve's gross earnings and the verdicts start arriving early. The trend book pays 11 bp a year against 370 gross — costs are a rounding error, the reward for six round trips a year in deep instruments. The sector book pays 22 against a gross of 100 — a fifth of an edge that was already indistinguishable from zero. The calendar book pays 17 of its 250 for one round trip every month. And `pairs` pays 25 bp *unlevered* against the 18 bp its honest execution grade earns — underwater before leverage, and [lesson six's](06-position-sizing-and-risk-budgeting.md) 8x sizing multiplies the trading costs and adds financing on top, so leverage only deepens the grave. Two liabilities are deliberately absent and named: market impact (next section — it depends on size, which spread-plus-commission does not) and shorting/financing costs, which for these ETF sleeves at modest leverage are secondary everywhere except the already-dead pair. A cost model earns trust by declaring what it excludes.

## The square-root law prices size

Spread and commission are the costs of *trading*; impact is the cost of *size*, and it grows with the square root of the fraction of daily volume an order consumes. For the sector book's tempo — 5.3x one-way turnover a year across funds that trade about $1.5bn a day — the law prices two hypothetical books:

```python
import numpy as np

adv, sigd, oneway = 1.5e9, 0.013, 5.3   # sector ETF depth, daily vol, turnover
for q in [10e6, 500e6]:
    impact = sigd * np.sqrt(q / adv) * 1e4
    annual = 2 * oneway * impact
    print(f"book ${q / 1e6:>4.0f}M: impact {impact:.1f} bp/trade "
          f"-> {annual:.0f} bp/yr drag")
# => book $  10M: impact 10.6 bp/trade -> 113 bp/yr drag
#    book $ 500M: impact 75.1 bp/trade -> 796 bp/yr drag
```

At $10M the impact drag alone — 113 bp a year — exceeds the sector strategy's entire gross return; at $500M the drag is nearly 800 bp, a number that would bankrupt an 8%-vol book on costs alone. This is [lesson six's](06-position-sizing-and-risk-budgeting.md) $4M capacity estimate seen from the other side, and it generalizes into the industry's quiet hierarchy: high-turnover strategies in thin instruments are boutique businesses or nothing; the strategies that absorb institutional capital are the slow ones in deep markets, not because their ideas are better but because the square-root law taxes speed multiplicatively with size. When a backtest is pitched without an AUM attached, this section is the question to ask — *at what size was this Sharpe computed, and at what size does it die?*

## No-trade bands: paying only for what matters

The vol-targeted trend book from lesson six recomputes its ideal leverage every day and, run literally, trades every day to track it — buying precision with turnover. A no-trade band rebalances only when the held leverage drifts materially from target, and the sweep says how much precision was actually worth buying:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
r3 = np.log(px[["SPY", "TLT", "GLD"]]).diff()
w3 = np.sign(r3.rolling(252).sum()) / 3
tsmom = (w3.shift(1) * r3).sum(axis=1, min_count=1).dropna()
target = (0.10 / (np.sqrt(252) * tsmom.ewm(span=32).std())).dropna()

for band in [0.0, 0.10, 0.25, 0.50]:
    held = target.to_numpy(copy=True)
    cur = held[0]
    for i in range(len(held)):
        if abs(held[i] - cur) / cur > band:
            cur = held[i]
        held[i] = cur
    helds = pd.Series(held, index=target.index)
    posn = w3.mul(helds, axis=0).dropna()
    book = (posn.shift(1) * r3).sum(axis=1, min_count=1).dropna()
    turn = posn.diff().abs().sum(axis=1).mean() * 252
    print(f"band {band:>4.0%}: leverage resets/yr {(helds.diff() != 0).mean() * 252:>4.0f}, "
          f"notional traded {turn:4.1f}x/yr, "
          f"Sharpe {np.sqrt(252) * book.mean() / book.std():.2f}")
# => band   0%: leverage resets/yr  252, notional traded 22.0x/yr, Sharpe 0.51
#    band  10%: leverage resets/yr   30, notional traded 16.9x/yr, Sharpe 0.51
#    band  25%: leverage resets/yr    8, notional traded 15.1x/yr, Sharpe 0.51
#    band  50%: leverage resets/yr    2, notional traded 14.6x/yr, Sharpe 0.48
```

Daily tracking rebalances 252 times a year and trades 22x the book; a 25% band rebalances *eight* times, trades 31% less notional, and the Sharpe does not move at the second decimal. Only the 50% band — two leverage adjustments a year — finally shows a cost, dropping to 0.48, because a book that slow to de-risk holds stale leverage into volatility regime-shifts; 25% is the defensible resting point. The economics are general: tracking error around an optimal position is a *second-order* loss (the optimum is flat at the top) while trading costs are *first-order*, so precision is almost always overpriced near the optimum. Bands, turnover penalties, and trade-when-it-matters scheduling are one family of solutions to the same inequality — and the same logic, run in reverse, indicts any backtest that rebalances daily by default: it is paying first-order costs for second-order benefits, and its net Sharpe is the one that pays.

## Net of costs: who survives

The bill, assembled. Each sleeve's gross annual earnings — with `pairs` graded at its honest next-close execution — less its cost drag, over its volatility:

```python
import numpy as np

# assembled from this lesson's own measurements; pairs graded at next-close fill,
# shortvol in its own variance-point units (drag: 0.05 pts/month assumed)
ledger = {                 # gross/yr   ann vol   cost drag/yr
    "tsmom":    (370, 1220, 11),       # bp
    "pairs":    (18, 95, 25),          # bp
    "xsmom":    (100, 1330, 22),       # bp
    "shortvol": (13.1, 8.4, 0.6),      # variance points
    "tom":      (250, 810, 17),        # bp
}
print("sleeve      gross Sharpe   net Sharpe")
for name, (g, v, d) in ledger.items():
    print(f"{name:9s} {g / v:>10.2f} {(g - d) / v:>12.2f}")
rho = -0.01
print(f"surviving book, tsmom + shortvol at equal vol (corr {rho:+.2f}): "
      f"Sharpe {(359 / 1220 + 12.5 / 8.4) / np.sqrt(2 * (1 + rho)):.2f}")
# => sleeve      gross Sharpe   net Sharpe
#    tsmom           0.30         0.29
#    pairs           0.19        -0.07
#    xsmom           0.08         0.06
#    shortvol        1.56         1.49
#    tom             0.31         0.29
#    surviving book, tsmom + shortvol at equal vol (corr -0.01): Sharpe 1.27
```

The reckoning [the index page](index.md) promised. `pairs` is dead — net −0.07, killed by an execution assumption and buried by 25 bp of costs against 18 of honest gross, with the 8x leverage its sizing demanded never even applied; the arbitrage is real, and it belongs to whoever owns the creation-redemption infrastructure, which is the most common autopsy in quantitative trading: *the edge existed, and it wasn't yours at your cost structure*. `xsmom` at 0.06 completes the death certificate [lesson five](05-feature-and-signal-engineering.md) drafted. `tom` survives costs arithmetically at 0.29 — and still trails the buy-and-hold index it hides inside, so it persists only as the cadaver [lesson eight](08-validation-and-overfitting.md) will formally dissect. The survivors are instructive precisely because they are opposite: `tsmom`, whose 11 bp of costs barely dent it because slowness is a cost strategy; and `shortvol`, whose edge is wide enough to pay any plausible toll. The surviving two-sleeve book — genuinely uncorrelated, both edges net-positive — prints Sharpe 1.27, and *that*, not the gross 1.51, is the number this part carries into its final lesson. One caveat travels with it: 1.27 still contains `shortvol`'s single-history term-structure filter and every trial this part has run. The last gauntlet remains.

!!! warning "Costs are not a haircut on the backtest; they are a filter on which strategies exist"
    A cost model applied at the end of research selects among finished strategies. A cost model applied at the start changes which strategies get built at all: it rules out the fast signal in the thin instrument before a line of code is written, prices the execution infrastructure a strategy assumes, and asks at what AUM the idea stops working. Every strategy that died in this lesson died of something knowable on day one — a fill assumption, a turnover number, a spread width. The bill was always going to arrive; professional research just reads it first.

!!! abstract "Key takeaways"
    - The five-sleeve correlation matrix spans −0.29 to +0.34 and prices real diversification — while exposing hidden kinships like `tom`-`svol` at +0.34, two long-calm bets in different costumes; the gross book's 1.51 again fails to beat its best sleeve alone.
    - Netting the three SPY-trading sleeves into one book cuts gross exposure 9% and trading 6% — modest here because the sleeves rarely oppose, structural at multi-strategy scale, and available only if positions live in one book.
    - Execution assumptions scale with signal speed: the trend book shrugs at a one-day fill delay (0.30 to 0.37) while `pairs` collapses from 23.0 to 2.8 bp per trip and Sharpe 1.23 to 0.19 — the strategy lived entirely inside the signal-close fill.
    - The defensible cost model is simple, stated, and auditable — half-spread plus commission with parameters on the table — and its drags (11 to 25 bp a year) already decide half the reckoning before impact is even priced.
    - The square-root law taxes size times speed: the sector book's tempo costs 113 bp a year at $10M and 796 at $500M — a backtest without an AUM attached is a Sharpe without a meaning.
    - No-trade bands buy back first-order costs for second-order tracking error: eight leverage resets a year instead of 252, 31% less notional traded, identical Sharpe to the second decimal.
    - Net of honest execution and costs: `pairs` −0.07, `xsmom` 0.06, `tom` 0.29 and still behind the index — while `tsmom` (0.29) and `shortvol` (1.49) survive to form a genuinely diversified book at Sharpe 1.27, the only number in this part that has paid its bills.

## Where this goes next

The book that survives — trend plus filtered short-vol, net Sharpe 1.27 — has passed every test this part knows how to administer except the ones that matter most. Its ingredients were selected from dozens of trials this part has honestly logged but never yet *charged for*: lookback grids, filter choices, formation variants, a term-structure switch that dodged exactly one catastrophic month on exactly one history. [Validation and Overfitting](08-validation-and-overfitting.md) administers the final examinations — walk-forward, purged cross-validation, White's Reality Check, the probability of backtest overfitting, and the deflated Sharpe ratio — the machinery that decides whether a surviving number is a discovery or a curve fit with good manners. It is the part's last lesson because everything before it is inadmissible until it runs.
