# Performance Metrics and Reporting

The engine can now simulate; this lesson teaches it to testify. Everything here consumes exactly one input — the equity curve that [Portfolio Accounting](02-portfolio-accounting.md) reconciled to the penny — and produces the numbers a desk actually reads: returns at the right frequency, ratios that reward different virtues, drawdowns with dates attached, turnover that predicts costs, and trade statistics that reveal a strategy's personality. The metrics need nothing from the broker; that is why this lesson's prerequisites stop at the accounting layer. But its finale needs everything: the last section assembles the full engine — queue, ledger, and [lesson three's](03-order-management-and-fill-simulation.md) fill assumptions — runs `tsmom` through twenty-five years of bars, and prints the part's first complete tearsheet, reconciled line by line against the vectorized number Part IV left as the benchmark.

A warning about familiarity: most of these formulas look like one-liners, and all of them have a folklore version that is quietly wrong — annualizing at the wrong frequency, a Sortino nobody can reproduce, a maximum drawdown quoted without its dates. The machinery of [Returns and Distributions](../part-03-statistics/02-returns-and-distributions.md) already taught the distributions underneath; this lesson is about not fumbling the arithmetic on top of them.

## From equity to returns

Every metric begins by differentiating the equity curve, and the first fork in the road — simple or log returns — is a choice about what should be *additive*:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(rets.rolling(252).sum()).shift(1) * rets).mean(axis=1).dropna()
eq = np.exp(tsmom.cumsum())               # a $1 equity curve

simple = eq.pct_change().dropna()
logr = np.log(eq).diff().dropna()
print(f"final equity {eq.iloc[-1]:.4f} -> total return {eq.iloc[-1] - 1:+.1%}")
print(f"sum of log returns {logr.sum():+.4f} = log({eq.iloc[-1]:.4f})  # additive")
print(f"sum of simple returns {simple.sum():+.4f}      # not a return at all")
print(f"daily mean: simple {simple.mean() * 1e4:+.2f} bp, "
      f"log {logr.mean() * 1e4:+.2f} bp")
# => final equity 2.4687 -> total return +146.9%
#    sum of log returns +0.9506 = log(2.4687)  # additive
#    sum of simple returns +1.1321      # not a return at all
#    daily mean: simple +1.84 bp, log +1.54 bp
```

Log returns sum to exactly the log of final equity — 0.9506 is $\log 2.4687$ to the fourth decimal — which is the property that makes `cumsum`, resampling, and every rolling window in this course legal arithmetic. Simple returns do not: their sum, 1.1321, corresponds to no dollar amount anywhere, yet it is routinely mistaken for "total return" in amateur tearsheets, here overstating the true +146.9% by a comfortable margin. The two daily means differ by 0.30 bp — small-looking, but that gap *is* volatility drag, $\sigma^2/2$, about 0.7% a year at this book's 12.2% volatility, and it explains how a strategy can have a positive average simple return and still lose money compounding. The working rule is one sentence: aggregate in logs, report in simple — do the time-series arithmetic in the additive representation, then convert the endpoints back to the percentages humans and investors actually experience. The engine's curve later in this lesson is in dollars, so its returns come from `pct_change`; the rule tells us to treat those as the reporting layer and reach for logs whenever a window must be summed.

## Annualization without folklore

Sharpe ratios are quoted annualized, and the scaling factor depends on the frequency of the returns being fed in — a detail folklore compresses into "multiply by $\sqrt{252}$", which is only the daily case:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(rets.rolling(252).sum()).shift(1) * rets).mean(axis=1).dropna()
monthly = tsmom.resample("ME").sum(min_count=15)

print(f"daily Sharpe with sqrt(252):   "
      f"{np.sqrt(252) * tsmom.mean() / tsmom.std():.2f}")
print(f"monthly Sharpe with sqrt(12):  "
      f"{np.sqrt(12) * monthly.mean() / monthly.std():.2f}")
print(f"monthly Sharpe with sqrt(252): "
      f"{np.sqrt(252) * monthly.mean() / monthly.std():.2f}  # frequency mismatch")
print(f"lag-1 autocorrelation of daily returns: {tsmom.autocorr(1):+.3f}")
# => daily Sharpe with sqrt(252):   0.30
#    monthly Sharpe with sqrt(12):  0.36
#    monthly Sharpe with sqrt(252): 1.64  # frequency mismatch
#    lag-1 autocorrelation of daily returns: -0.024
```

The third line is the folklore accident: apply $\sqrt{252}$ to monthly returns and a 0.30 strategy prints 1.64 — an error of a factor of $\sqrt{21}$ that has appeared in real fund marketing more than once. The subtler lesson is the *second* line: even done correctly, the monthly Sharpe is 0.36, not 0.30. That is not a bug either. The $\sqrt{n}$ rule assumes returns are independent across time; aggregation to monthly slightly reshuffles how variance accumulates, and any autocorrelation — here a mild −0.024 at lag one — makes volatility scale a little slower or faster than the square root pretends. For this book the effect is modest; for a strategy with strongly autocorrelated returns (anything smoothed, anything illiquid, anything marked with stale prices) the daily-annualized and monthly-annualized Sharpe can diverge embarrassingly, and the divergence is a *diagnostic*, not a nuisance: it says the returns have memory the iid formula cannot see. The engine's tearsheet therefore fixes one convention and states it — daily returns, $\sqrt{252}$ — because a number is only comparable to another number computed the same way.

## Sharpe, Sortino, Calmar

Three ratios, three definitions of "risk" for the same mean return:

$$
\mathrm{Sharpe} = \frac{\sqrt{252}\;\bar r}{\sigma_r}, \qquad
\mathrm{Sortino} = \frac{\sqrt{252}\;\bar r}{\sigma_{r^-}}, \qquad
\mathrm{Calmar} = \frac{\mathrm{CAGR}}{|\mathrm{maxDD}|},
$$

volatility of everything, volatility of the bad days only, and the worst peak-to-trough hole:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(rets.rolling(252).sum()).shift(1) * rets).mean(axis=1).dropna()

def sharpe(r):
    return np.sqrt(252) * r.mean() / r.std()

def sortino(r):
    return np.sqrt(252) * r.mean() / r[r < 0].std()

def calmar(r):
    eq = np.exp(r.cumsum())
    cagr = eq.iloc[-1] ** (252 / len(eq)) - 1
    return cagr / abs((eq / eq.cummax() - 1).min())

for name, s in [("SPY buy-hold", rets["SPY"].dropna()), ("tsmom", tsmom)]:
    print(f"{name:12s}: Sharpe {sharpe(s):.2f}, Sortino {sortino(s):.2f}, "
          f"Calmar {calmar(s):.2f}")
# => SPY buy-hold: Sharpe 0.38, Sortino 0.48, Calmar 0.14
#    tsmom       : Sharpe 0.30, Sortino 0.39, Calmar 0.13
```

The table is deliberately anticlimactic, and the anticlimax teaches. On daily data, Sortino sits about thirty percent above Sharpe for *both* the index and the trend book — yet Part IV showed these two products have utterly different tail personalities, monthly skew of −0.63 against −0.06. Daily downside deviation barely sees what monthly skew sees clearly, because asymmetry lives at the horizon of multi-day episodes, not single days; a ratio is only as informative as the frequency it is computed at. Calmar comes out nearly tied, 0.14 versus 0.13, but through opposite routes — SPY pairs a higher CAGR with a −55% hole, `tsmom` a lower CAGR with a −29% one — identical quotients hiding opposite risk stories, which is why Calmar should never travel without its numerator and denominator. And one definitional confession, stated because vendors will not state theirs: the Sortino here uses the standard deviation of negative days, one of at least three "downside deviation" conventions in circulation. The number is only reproducible because the formula is printed beside it — the tearsheet at the end of this lesson inherits exactly these three functions, character for character.

## The anatomy of a drawdown

A drawdown number without dates is a rumor. The full anatomy of the underwater periods — when they began, how deep they went, when the account made it back — comes from one running maximum:

$$
D_t \;=\; \frac{E_t}{\max_{s \le t} E_s} \;-\; 1
$$

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(rets.rolling(252).sum()).shift(1) * rets).mean(axis=1).dropna()
eq = np.exp(tsmom.cumsum())

dd = eq / eq.cummax() - 1
under = dd < 0
spell = (under != under.shift()).cumsum()[under]
rows = []
for _, seg in dd[under].groupby(spell):
    end = seg.index[-1] if seg.index[-1] != dd.index[-1] else None
    rows.append((seg.idxmin(), seg.min(), seg.index[0], end, len(seg)))
for trough, depth, start, end, days in sorted(rows, key=lambda r: r[1])[:3]:
    rec = f"recovered {end:%Y-%m-%d}" if end is not None else "not yet recovered"
    print(f"{depth:6.1%}  under water {start:%Y-%m-%d} -> trough "
          f"{trough:%Y-%m-%d}, {rec} ({days} days)")
print(f"new equity highs on {(~under).mean():.0%} of days; "
      f"everything else is drawdown")
# => -28.7%  under water 2015-01-26 -> trough 2019-02-05, not yet recovered (2623 days)
#    -24.3%  under water 2002-10-10 -> trough 2003-09-25, recovered 2006-04-18 (886 days)
#    -23.1%  under water 2008-11-21 -> trough 2009-08-07, recovered 2011-07-12 (663 days)
#    new equity highs on 2% of days; everything else is drawdown
```

Read the first row twice, because it is the most important line in this lesson. The trend book's deepest hole — the −28.7% that Part IV rounded to −29% — began in January 2015, troughed four years later, and *has not recovered*: over a decade under water and counting, inside a strategy whose headline Sharpe is a respectable-sounding 0.30. Depth and duration are different pains with different victims — depth is what blows up leveraged accounts, duration is what makes investors (and their committees, and your own conviction) quit — and no single ratio carries both, which is the argument for printing the table rather than the summary. The last line generalizes the discomfort: this equity curve set a new high on only 2% of its days. That is not a defect of `tsmom`; it is near-universal for any positive-drift-plus-noise process, and it means the *lived experience* of running even a good strategy is losing to your own past self on ninety-eight days out of a hundred. A researcher sees an upward-sloping curve; the person running the money sees the distance from the high-water mark. Both are looking at the same series; only one of them has to sit through it.

## Exposure and turnover

Two more columns of the tearsheet describe not how much the book made but *how it stood* — how much market it held, in which direction, and how often it moved:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
sig = np.sign(rets.rolling(252).sum()).shift(1)
w = sig.div(sig.notna().sum(axis=1), axis=0)      # live sleeves split the book

gross = w.abs().sum(axis=1, min_count=1).dropna()
net = w.sum(axis=1, min_count=1).reindex(gross.index)
turns = w.diff().abs().sum(axis=1).reindex(gross.index).fillna(0)
print(f"gross exposure: always {gross.min():.2f} (the book is fully deployed)")
print(f"net exposure: mean {net.mean():+.2f}, range {net.min():+.2f} "
      f"to {net.max():+.2f}, long {(net > 0).mean():.0%} of days")
print(f"one-way turnover: {turns.sum() / (len(gross) / 252):.1f}x per year")
# => gross exposure: always 1.00 (the book is fully deployed)
#    net exposure: mean +0.32, range -1.00 to +1.00, long 71% of days
#    one-way turnover: 10.2x per year
```

The weight matrix here is exactly the one implicit in Part IV's `sleeves.mean(axis=1)` — each live sleeve gets an equal slice of the book — made explicit so it can be measured. Gross exposure is pinned at 1.00 by construction: a sign strategy is always fully invested, which distinguishes it sharply from the trend *filters* of Part IV's momentum lesson that spent 30% of their life in cash. Net exposure tells the direction story: +0.32 on average and long 71% of days, a structural long tilt inherited from the fact that these three assets spent most of twenty-five years rising — worth knowing before calling the book "market-neutral," which it is not and never claimed to be. Turnover is the cost dial: 10.2× one-way per year means the book trades about ten times its own value annually — walking that through [lesson three's](03-order-management-and-fill-simulation.md) model II at roughly one basis point per crossing prices the strategy's cost drag at about 11 bp a year, which is precisely the figure Part IV's cost lesson charged `tsmom` from the outside. Two independent routes — turnover-times-spread and the engine's fill-by-fill cash ledger — should and will converge on the same drag; when they do not, one of the two backtests is lying about its trading.

## The trades themselves

The equity curve is the portfolio's story; the *trips* — one round of entering and holding a sleeve's direction until it flips — are the strategy's. Slicing each sleeve at its sign changes turns 6,158 daily rows into a few hundred decisions:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
sig = np.sign(rets.rolling(252).sum()).shift(1)

trips = []
for a in ["SPY", "TLT", "GLD"]:
    s, r = sig[a].dropna(), rets[a]
    run = (s != s.shift()).cumsum()
    pnl = (s * r).groupby(run).sum()
    trips += list(zip(pnl, s.groupby(run).size()))
pnl = np.array([p for p, _ in trips])
days = np.array([d for _, d in trips])

hit = (pnl > 0).mean()
win, loss = pnl[pnl > 0].mean(), pnl[pnl <= 0].mean()
print(f"trips {len(pnl)}, median hold {np.median(days):.0f} days")
print(f"hit rate {hit:.0%}, avg win {win:+.1%}, avg loss {loss:+.1%}, "
      f"payoff {-win / loss:.1f}")
print(f"expectancy {pnl.mean() * 1e4:+.0f} bp per trip "
      f"(= {hit:.2f} x {win * 1e4:+.0f} + {1 - hit:.2f} x {loss * 1e4:+.0f})")
# => trips 329, median hold 3 days
#    hit rate 34%, avg win +5.7%, avg loss -2.0%, payoff 2.9
#    expectancy +66 bp per trip (= 0.34 x +569 + 0.66 x -196)
```

The median holding period is *three days* — for a strategy built on a 252-day signal. That is not a typo; it is the signature of a sign rule: when the trailing-year sum hovers near zero, the sign flickers, and half of all 329 trips are whipsaws that die within days, while the money is made by the tail of the distribution — multi-month rides the median never sees. The rest of the row confirms the trend-following personality Part IV's Donchian system first displayed on gold (50 trades, 48% hit rate, winners twice losers): here it is more extreme — wrong 66% of the time, right by +5.7% when right, wrong by only −2.0% when wrong, a payoff ratio of 2.9. The last line is the identity every trade-level report should print rather than imply: expectancy is *exactly* hit rate times average win plus miss rate times average loss, +66 bp per trip, and writing it out kills an entire genre of confusion in which hit rate is discussed as if it were the strategy. A 34% hit rate is not a weakness of this system; it is the *shape* of it — and [lesson five](05-trade-logs-and-visualization.md) will re-derive these same statistics from the engine's actual fills, where every trip also pays its costs on the way in and out.

## The tearsheet, from the engine

Everything assembles. The compressed engine below is lesson one's loop with every placeholder replaced — signals at the close, orders on flips and month-ends, integer shares, fills at the next open with lesson three's costs, cash rounded to the penny — followed by the reporting layer this lesson built. It is the longest code block in the course, a title Part IV's lesson seven held until now, and it is deliberate for the same reason: the whole machine, visible at once:

```python
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
SYMS = ["SPY", "TLT", "GLD"]
HS, COMM = {"SPY": 0.5, "TLT": 1.0, "GLD": 1.0}, 0.2
data = {s: bars.xs(s, axis=1, level=1).dropna() for s in SYMS}
close = pd.DataFrame({s: d["Close"] for s, d in data.items()})
open_ = pd.DataFrame({s: d["Open"] for s, d in data.items()})
sig = np.sign(np.log(close).diff().rolling(252).sum())

idx = close.index
cash, pos, last = 1_000_000.0, dict.fromkeys(SYMS, 0), {}
pending, fills, paid, traded = [], 0, 0.0, 0.0
equity = pd.Series(np.nan, index=idx)
for i, t in enumerate(idx):
    for s, tgt in pending:                # yesterday's orders, today's opens
        dq, o = tgt - pos[s], open_.at[t, s]
        if dq and not np.isnan(o):
            fee = round(abs(dq) * o * (HS[s] + COMM) * 1e-4, 2)
            cash = round(cash - dq * o - fee, 2)
            pos[s] += dq
            fills, paid, traded = fills + 1, round(paid + fee, 2), traded + abs(dq) * o
    pending = []
    eq = cash + sum(pos[s] * close.at[t, s] for s in SYMS if pos[s])
    equity[t] = eq
    if i == len(idx) - 1:
        break
    live = [s for s in SYMS if not np.isnan(sig.at[t, s])]
    for s in live:                        # rebalance on sign flip or month-end
        if sig.at[t, s] != last.get(s, 0.0) or t.month != idx[i + 1].month:
            pending.append((s, int(sig.at[t, s] * eq / len(live) / close.at[t, s])))
        last[s] = sig.at[t, s]

lr = np.log(close).diff()
vec = (np.sign(lr.rolling(252).sum()).shift(1) * lr).mean(axis=1).dropna()
er = equity.pct_change().loc[vec.index[0]:].dropna()  # engine, same window
eq = equity.loc[vec.index[0]:]
dd = eq / eq.cummax() - 1
trough = dd.idxmin()
years = len(er) / 252
longest = (~(dd < 0)).cumsum()[dd < 0].value_counts().max()

def sharpe(r):
    return np.sqrt(252) * r.mean() / r.std()

print(f"tsmom through the engine, {eq.index[0]:%Y-%m-%d} -> {eq.index[-1]:%Y-%m-%d}")
print(f"fills {fills}, costs ${paid:,.0f}, final equity ${eq.iloc[-1]:,.2f}")
print(f"CAGR {(eq.iloc[-1] / 1_000_000) ** (1 / years) - 1:.1%}, "
      f"ann vol {er.std() * np.sqrt(252):.1%}, Sharpe {sharpe(er):.2f}, "
      f"Sortino {np.sqrt(252) * er.mean() / er[er < 0].std():.2f}")
print(f"maxDD {dd.min():.1%} (peak {eq.loc[:trough].idxmax():%Y-%m-%d}, "
      f"trough {trough:%Y-%m-%d}), longest underwater {longest} days")
print(f"one-way turnover {traded / eq.mean() / years:.1f}x/yr, "
      f"fill model: next open + half-spread + {COMM} bp commission")
print(f"vectorized on the same bars, costless signal-close: Sharpe {sharpe(vec):.2f}")
# => tsmom through the engine, 2001-01-03 -> 2025-06-30
#    fills 1103, costs $50,793, final equity $2,522,514.08
#    CAGR 3.9%, ann vol 12.0%, Sharpe 0.38, Sortino 0.49
#    maxDD -27.3% (peak 2002-10-09, trough 2003-06-25), longest underwater 2294 days
#    one-way turnover 10.7x/yr, fill model: next open + half-spread + 0.2 bp commission
#    vectorized on the same bars, costless signal-close: Sharpe 0.30
```

The reconciliation line is the reason this block exists: the engine says 0.38 where the vectorized benchmark says 0.30, and every basis point of that gap has already been purchased somewhere in this part. Lesson three's dial isolated the components: moving from signal-close fantasy to next-open honesty added +0.04 by itself — momentum's flips harvest overnight gaps that tend to run in its favor — and spread-plus-commission costs took back almost nothing, the $50,793 of explicit fees amounting to those same ~11 bp a year that 10.7× turnover at these half-spreads predicts. The remainder comes from what the engine does that no weight matrix can: it rebalances monthly instead of daily, letting winners ride inside the month; it holds integer shares of a compounding dollar book instead of fractional weights of a normalized one; and its first trading day — January 3rd, 2001, a 5% surprise-rate-cut rally straight into a fresh short — cost it −5.2% before lunch on day one. Note also that the engine's worst hole is *not even the same hole*: −27.3% peaking in October 2002, when the book was two sleeves and dollar-compounding, versus the vectorized curve's 2015–2019 trench. None of these differences is an error, and that is the entire point of the exercise — both simulations are internally consistent answers to *different questions*, and only an engine whose books reconcile to $0.00 lets you say that with a straight face. Two omissions are stated for the record: no financing on cash balances, and costs charged as explicit fees rather than price adjustments — both chosen to stay comparable with Part IV's accounting.

!!! warning "A tearsheet is a cross-examination, not a scoreboard"
    Every line answers an accusation. Sharpe: is the mean real relative to the noise? Sortino, with its formula shown: is the volatility the bad kind? Drawdown with dates: when, specifically, would you have been fired? Turnover: who is the strategy really working for, you or the market makers? And the fill-model line: which reality does this number live in? A tearsheet that prints only the flattering answers — or worse, prints numbers whose formulas and assumptions are nowhere stated — is not a report; it is an advertisement wearing a report's clothes. The engine's tearsheet prints its assumptions in the same font as its Sharpe, because a result that cannot be cross-examined is not a result.

!!! abstract "Key takeaways"
    - Aggregate in logs, report in simple: log returns sum to exactly log-final-equity (0.9506 = log 2.4687), simple returns sum to nothing meaningful (1.1321 ≠ +146.9%), and the 0.30 bp gap between their daily means is volatility drag, $\sigma^2/2$.
    - Annualization must match frequency: monthly returns with $\sqrt{252}$ print a fantasy 1.64 for a 0.30 strategy — and even the correct monthly figure (0.36) differs from the daily one, a diagnostic of non-iid returns, not an error.
    - The three ratios reward different virtues, and daily Sortino (0.39 vs Sharpe 0.30) barely sees the tail asymmetry that monthly skew shouts — a ratio is only as informative as its frequency and its stated formula.
    - Drawdowns need dates: `tsmom`'s deepest hole (−28.7%) began 2015-01-26 and has not recovered after 2,623 days, and the curve made new highs on only 2% of days — duration is the pain that makes people quit, and no ratio reports it.
    - The book runs 1.00 gross always, +0.32 net on average (long 71% of days), and 10.2× one-way turnover a year — which at Part IV's spreads prices the cost drag at ~11 bp/yr before the engine ever runs.
    - Trade-level statistics expose the personality: 329 trips, median hold 3 days, 34% hit rate, payoff 2.9, expectancy +66 bp per trip — printed as the identity hit × win + miss × loss, which is the whole cure for hit-rate mysticism.
    - The capstone run reconciles: engine 0.38 versus vectorized 0.30 on identical bars, decomposed into next-open timing (+0.04), ~$50,793 of explicit costs (−11 bp/yr), monthly-versus-daily rebalancing and integer shares — 1,103 fills, final equity $2,522,514.08, maxDD −27.3%, every line auditable because the books balance.

## Where this goes next

The tearsheet compresses a quarter-century into nine lines, and compression is loss: somewhere inside the 0.38 are eleven hundred individual decisions, each with an entry, an exit, a holding period, and a cost, none of them visible from the summary. [Trade Logs and Visualization](05-trade-logs-and-visualization.md) decompresses — it teaches the engine to write a structured log of every fill, proves the log is a complete record by replaying the entire backtest from it to the penny, pairs fills into round trips with the FIFO discipline lesson two built, and then draws the pictures — equity, drawdown, exposure — that let a human audit in seconds what the table can only assert.
