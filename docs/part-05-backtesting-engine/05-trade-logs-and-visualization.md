# Trade Logs and Visualization

A backtest that prints a Sharpe ratio and exits has destroyed most of what it computed. The tearsheet of [Performance Metrics and Reporting](04-performance-metrics-and-reporting.md) compressed eleven hundred fills into nine lines, and compression is loss — inside the 0.38 are individual decisions with entries, exits, holding periods, and costs, none of them recoverable from the summary. This closing lesson makes the engine's output an *artifact*: a structured log of every fill, written in the same parquet format as the price cache and held to the same doctrine. The log's defining test is severe — the entire backtest must be reproducible from the log alone, to the penny — and passing it changes the log's status from debugging aid to primary record: the engine becomes merely one way of producing it.

The second half of the lesson turns the record into pictures, because a table can only assert what a chart lets a human *audit* — the plotting idioms are [Part II's](../part-02-python/06-plotting.md), applied to the curves this part built. And the finale is the payoff of the whole architecture: with runs reduced to comparable artifacts, three engines with three fill models run side by side, overlaid, differenced, and stamped with the metadata that makes the comparison an experiment instead of an anecdote.

## The trade log schema

One row per fill, and every column is there to answer a future question — `run_id` for *which configuration produced this*, the fill fields for *what happened*, and a running `cash_after` so the log carries its own audit column:

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
cash, pos, last, pending, log = 1_000_000.0, dict.fromkeys(SYMS, 0), {}, [], []
for i, t in enumerate(idx):
    for s, tgt in pending:                # the engine of lesson four, logging
        dq, o = tgt - pos[s], open_.at[t, s]
        if dq and not np.isnan(o):
            fee = round(abs(dq) * o * (HS[s] + COMM) * 1e-4, 2)
            cash = round(cash - dq * o - fee, 2)
            pos[s] += dq
            log.append((t, s, dq, o, fee, cash))
    pending = []
    if i == len(idx) - 1:
        break
    eq = cash + sum(pos[s] * close.at[t, s] for s in SYMS if pos[s])
    live = [s for s in SYMS if not np.isnan(sig.at[t, s])]
    for s in live:
        if sig.at[t, s] != last.get(s, 0.0) or t.month != idx[i + 1].month:
            pending.append((s, int(sig.at[t, s] * eq / len(live) / close.at[t, s])))
        last[s] = sig.at[t, s]

trades = pd.DataFrame(log, columns=["ts", "symbol", "qty", "px", "fee",
                                    "cash_after"])
trades.insert(0, "run_id", "tsmom-nextopen-v1")
trades.to_parquet("data/part5trades.parquet")
print(f"{len(trades)} fills logged -> data/part5trades.parquet")
print(trades.iloc[[0, 1, -1]].to_string(index=False,
                                        float_format=lambda x: f"{x:,.2f}"))
# => 1103 fills logged -> data/part5trades.parquet
#               run_id         ts symbol    qty    px    fee   cash_after
#    tsmom-nextopen-v1 2001-01-03    SPY -12260 81.25  69.73 1,996,025.94
#    tsmom-nextopen-v1 2001-01-30    SPY  23076 86.31 139.41     4,305.89
#    tsmom-nextopen-v1 2025-06-04    TLT -20739 81.75 203.44 1,685,966.32
```

The three displayed rows already tell a story the tearsheet never could. The engine's very first act, on 2001-01-03, was to *short* 12,260 SPY — the trailing year of the dot-com bust said down — and the proceeds swelled cash to nearly $2M; twenty-seven days later the signal flipped and one 23,076-share buy took the book long and cash down to $4,305.89, a fully deployed account with walking-around money. Three schema decisions carry the engineering weight. Quantities are signed integers, so side is not a separate error-prone column. Prices are stored at full float precision and merely *displayed* at two decimals — the log must reproduce the ledger's arithmetic exactly, and truncating the stored value would poison the replay this lesson is about to demand. And `cash_after` is deliberately redundant: it is a value the other columns imply, which is precisely what makes it an audit column — redundancy you can check is called evidence, the same design [lesson two](02-portfolio-accounting.md) used when it made the books carry their own invariants. The file lands next to the price cache and inherits its doctrine: frozen, versioned by `run_id`, the thing every later analysis reads instead of re-running the engine.

## Replay: the log is the backtest

The claim that the log is a *complete* record has a pass/fail test: reconstruct everything from it — no engine, no signals, no strategy code — and land on the same dollars:

```python
import pandas as pd

trades = pd.read_parquet("data/part5trades.parquet")
close = pd.read_parquet("data/part5.parquet")["Close"]

cash, pos, drift = 1_000_000.0, {}, 0.0
for f in trades.itertuples():             # nothing but the log
    cash = round(cash - f.qty * f.px - f.fee, 2)
    pos[f.symbol] = pos.get(f.symbol, 0) + f.qty
    drift = max(drift, abs(cash - f.cash_after))

t = close.index[-1]
final = cash + sum(q * close[s].loc[t] for s, q in pos.items() if q)
print(f"replayed {len(trades)} fills: worst cash drift ${drift:.2f}")
print(f"positions on {t:%Y-%m-%d}: {pos}")
print(f"final equity ${final:,.2f}  # lesson four printed $2,522,514.08")
# => replayed 1103 fills: worst cash drift $0.00
#    positions on 2025-06-30: {'SPY': 1429, 'TLT': -10358, 'GLD': 2737}
#    final equity $2,522,514.08  # lesson four printed $2,522,514.08
```

Eleven lines of replay, and three exact matches. The recomputed cash agrees with the log's own `cash_after` column at every one of 1,103 fills — worst drift $0.00, lesson two's penny discipline surviving its trip through a file format. The reconstructed positions — long 1,429 SPY, short 10,358 TLT, long 2,737 GLD — are the engine's actual final book. And marking those positions at the cache's last closes lands on $2,522,514.08, the tearsheet's number *to the cent*. Pause on what this means operationally: the backtest now exists independently of the code that produced it. The strategy could be refactored beyond recognition, the engine rewritten in another language, and the claim "this configuration produced this result" remains checkable by anyone holding two parquet files and eleven lines of arithmetic. That is the standard live trading will demand anyway — a brokerage statement is exactly a trade log you replay against your own books — so the backtester adopting it early is not ceremony, it is rehearsal. The doctrine also has a converse worth stating: a backtest whose fills were never logged cannot be audited at all, only re-run and believed.

## From fills to round trips

Fills are accounting events; *trades*, in the trader's sense, are round trips — capital committed, held, and released. The bridge between them is the FIFO pairing [lesson two](02-portfolio-accounting.md) argued for when it chose lot accounting: walk the log, match each closing fill against the oldest open lot in its symbol:

```python
import pandas as pd

trades = pd.read_parquet("data/part5trades.parquet")

rows, book = [], {}                       # symbol -> open lots [qty, px, ts]
for f in trades.itertuples():
    qty, lots = f.qty, book.setdefault(f.symbol, [])
    while qty and lots and lots[0][0] * qty < 0:
        lot = lots[0]                     # FIFO, as in lesson two
        closed = min(abs(qty), abs(lot[0])) * (1 if lot[0] > 0 else -1)
        rows.append((f.symbol, lot[2], f.ts, closed,
                     round(closed * (f.px - lot[1]), 2)))
        lot[0] -= closed
        qty += closed
        if lot[0] == 0:
            lots.pop(0)
    if qty:
        lots.append([qty, f.px, f.ts])

trips = pd.DataFrame(rows, columns=["symbol", "opened", "closed", "qty", "pnl"])
still = {s: sum(l[0] for l in v) for s, v in book.items()}
print(f"{len(trips)} closed round trips from {len(trades)} fills")
print(f"lots still open: {still}")
print(f"gross realized PnL ${trips.pnl.sum():,.0f}")
# => 1088 closed round trips from 1103 fills
#    lots still open: {'SPY': 1429, 'TLT': -10358, 'GLD': 2737}
#    gross realized PnL $1,153,202
```

The counts are the edge cases made visible. Eleven hundred three fills became 1,088 closed trips *plus* three still-open lots — and those open lots are, share for share, the replay's final positions, because the pairing conserves the book by construction. A reversal fill is handled without a special case, exactly as in lesson two's position object: the fill's quantity drains through the opposing lot queue and whatever remains opens a new lot, so one fill can close several trips and still father a new one. The trailing print reconciles the layers: $1,153,202 of gross realized price PnL, minus the $50,793 of fees the tearsheet reported, plus the roughly $420k of unrealized PnL sitting on the three open lots, is the $1,522,514 the equity curve gained — the conservation law from lesson two's accounting section, now holding across an entire quarter-century log. Every row of `trips` carries its own opened and closed timestamps, which is the raw material the next section turns into the statistics a tearsheet cannot show.

## The shape of the trades

With trips in hand, the strategy's personality becomes a distribution — and it turns out to depend, instructively, on what you called a trade:

```python
import pandas as pd

trades = pd.read_parquet("data/part5trades.parquet")

rows, book = [], {}                       # the pairing of the previous section
for f in trades.itertuples():
    qty, lots = f.qty, book.setdefault(f.symbol, [])
    while qty and lots and lots[0][0] * qty < 0:
        lot = lots[0]
        closed = min(abs(qty), abs(lot[0])) * (1 if lot[0] > 0 else -1)
        rows.append((f.symbol, lot[2], f.ts, round(closed * (f.px - lot[1]), 2)))
        lot[0] -= closed
        qty += closed
        if lot[0] == 0:
            lots.pop(0)
    if qty:
        lots.append([qty, f.px, f.ts])

trips = pd.DataFrame(rows, columns=["symbol", "opened", "closed", "pnl"])
hold = (trips.closed - trips.opened).dt.days
hit = (trips.pnl > 0).mean()
print(f"median holding {hold.median():.0f} days, longest {hold.max():,d} days")
print(f"hit rate {hit:.0%}, avg win ${trips.pnl[trips.pnl > 0].mean():,.0f}, "
      f"avg loss ${trips.pnl[trips.pnl <= 0].mean():,.0f}")
best, worst = trips.loc[trips.pnl.idxmax()], trips.loc[trips.pnl.idxmin()]
print(f"best : {best.symbol} {best.opened:%Y-%m-%d} -> {best.closed:%Y-%m-%d} "
      f"${best.pnl:+,.0f}")
print(f"worst: {worst.symbol} {worst.opened:%Y-%m-%d} -> {worst.closed:%Y-%m-%d} "
      f"${worst.pnl:+,.0f}")
# => median holding 110 days, longest 1,660 days
#    hit rate 57%, avg win $7,169, avg loss $-6,973
#    best : SPY 2012-06-05 -> 2015-08-25 $+231,339
#    worst: GLD 2008-10-23 -> 2008-11-26 $-85,484
```

Now hold this against [lesson four's](04-performance-metrics-and-reporting.md) trade table for the *same strategy*: there, 329 signal-run trips with a 3-day median hold, a 34% hit rate, and a 2.9 payoff ratio; here, 1,088 FIFO trips with a 110-day median, a 57% hit rate, and winners barely larger than losers. Neither table is wrong. Lesson four sliced at signal changes and measured *decisions* — every whipsaw its own verdict. This table slices at lot closures and measures *realized cash*: monthly rebalancing trims a winning ride many times on the way up, so one three-year decision becomes dozens of modest banked wins, flattering the hit rate and shrinking the payoff. The pairing convention is part of the statistic, and any trade analysis that does not state its convention is quoting numbers from an unspecified distribution. The extremes, at least, agree on the story: the best trip is a 39-month ride on SPY out of the 2012 bull (+$231,339, the tail that pays for everything), and the worst is gold whipsawing through the 2008 panic (−$85,484 in a month) — trend following's entire bargain, one row each.

## Equity and drawdown, drawn honestly

The pictures now draw themselves *from the log* — positions by cumulative sum, cash by cumulative flows, equity by marking — one more independent path to the same dollars. Two axes, stacked, sharing time; the equity on a log scale, per [Part II's](../part-02-python/06-plotting.md) standing rule, so equal vertical steps mean equal percentages:

```python
import matplotlib.pyplot as plt
import pandas as pd

trades = pd.read_parquet("data/part5trades.parquet")
bars = pd.read_parquet("data/part5.parquet")
SYMS = ["SPY", "TLT", "GLD"]
close = pd.DataFrame({s: bars.xs(s, axis=1, level=1).dropna()["Close"]
                      for s in SYMS})

qty = trades.pivot_table(index="ts", columns="symbol", values="qty",
                         aggfunc="sum").reindex(close.index).fillna(0.0)
pos = qty.cumsum()
flows = (trades.qty * trades.px + trades.fee).groupby(trades.ts).sum()
cash = 1_000_000.0 - flows.reindex(close.index).fillna(0.0).cumsum()
equity = (cash + (pos * close).sum(axis=1)).loc["2001-01-03":]
dd = equity / equity.cummax() - 1

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6),
                               height_ratios=[2, 1])
ax1.plot(equity / 1e6)
ax1.set_yscale("log")
ax1.set_ylabel("equity, $M, log scale")
ax2.fill_between(dd.index, dd, 0, alpha=0.5)
ax2.set_ylabel("drawdown")
print(f"drawn from the log alone: final ${equity.iloc[-1]:,.0f}, "
      f"maxDD {dd.min():.1%}, trough {dd.idxmin():%Y-%m-%d}")
plt.show()
# => drawn from the log alone: final $2,522,514, maxDD -27.3%, trough 2003-06-25
```

The print is the third reconciliation of the lesson — final equity, maximum drawdown, and trough date all agreeing with the tearsheet, this time via a vectorized reconstruction (`pivot_table`, `cumsum`, broadcast marking) that shares no code with either the engine loop or the itertuples replay. Three implementations, one answer: that is what "the books are right" looks like from the outside. The chart itself is a two-panel argument. The top panel's log scale keeps 2001's dollars honest next to 2025's — on a linear axis, early history flattens into a false calm and recent noise inflates into false drama. The bottom panel is the same curve *seen from the high-water mark*, the view [lesson four's](04-performance-metrics-and-reporting.md) drawdown table quantified: the 2002–2003 trench, the long 2015–2019 shelf, each episode's depth and duration visible at a glance where the table could only list them. These blocks end with `plt.show()` and no saved artifact by course convention; in your own work the last line writes `fig.savefig(...)` next to the trade log, because a figure that cannot be regenerated from artifacts is a screenshot, not evidence.

## Exposure over time

Positions are step functions — they change only at fills — and drawing them as steps rather than interpolated lines is the difference between plotting the book and plotting a fiction:

```python
import matplotlib.pyplot as plt
import pandas as pd

trades = pd.read_parquet("data/part5trades.parquet")
bars = pd.read_parquet("data/part5.parquet")
SYMS = ["SPY", "TLT", "GLD"]
close = pd.DataFrame({s: bars.xs(s, axis=1, level=1).dropna()["Close"]
                      for s in SYMS})

qty = trades.pivot_table(index="ts", columns="symbol", values="qty",
                         aggfunc="sum").reindex(close.index).fillna(0.0)
pos = qty.cumsum()
flows = (trades.qty * trades.px + trades.fee).groupby(trades.ts).sum()
cash = 1_000_000.0 - flows.reindex(close.index).fillna(0.0).cumsum()
equity = cash + (pos * close).sum(axis=1)
expo = (pos * close).div(equity, axis=0).loc["2001-01-03":]

fig, ax = plt.subplots(figsize=(9, 4))
for s in SYMS:
    ax.step(expo.index, expo[s], where="post", label=s)
ax.step(expo.index, expo.sum(axis=1), where="post", lw=2, label="net")
ax.legend(ncols=4)
ax.set_ylabel("exposure / equity")
gross = expo.abs().sum(axis=1)
print(f"mean gross {gross.mean():.2f} (range {gross.min():.2f}-{gross.max():.2f})")
print(f"mean net {expo.sum(axis=1).mean():+.2f}, in market {(gross > 0).mean():.0%}")
plt.show()
# => mean gross 1.00 (range 0.68-1.49)
#    mean net +0.33, in market 100%
```

The averages match [lesson four's](04-performance-metrics-and-reporting.md) weight-matrix view — gross 1.00, net about +0.3, always in the market — and the *range* is what that view could never show: realized gross exposure wandered between 0.68 and 1.49. The weight matrix declared thirds and rebalanced daily by assumption; the engine holds integer shares fixed between rebalances while equity moves underneath them, so a losing stretch mechanically levers the book up (positions shrink slower than equity) and a winning stretch delevers it. The 1.49 reading is the engine confessing that a "fully invested, unlevered" strategy spent moments carrying half again its equity in exposure — risk that existed, that the vectorized abstraction structurally cannot represent, and that a risk manager would want throttled by the rebalance rule, not discovered in a postmortem. This is the recurring shape of the whole part in one chart: the intention (weights) and the fact (the log) are different time series, and the difference is itself information. The step plot per asset also reads as a narrative device — SPY's long green stretches, TLT's years-long short after 2021, GLD's late arrival in 2005 — the strategy's biography, drawn from its receipts.

## Comparing runs

The last skill is comparative: runs as artifacts, differing in exactly one assumption, overlaid. The engine collapses to a function of its fill cost, and three economies run side by side — free, spread-plus-commission, and spread-plus-impact at a pretended $100M scale:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
SYMS = ["SPY", "TLT", "GLD"]
HS = {"SPY": 0.5, "TLT": 1.0, "GLD": 1.0}
data = {s: bars.xs(s, axis=1, level=1).dropna() for s in SYMS}
close = pd.DataFrame({s: d["Close"] for s, d in data.items()})
open_ = pd.DataFrame({s: d["Open"] for s, d in data.items()})
sig = np.sign(np.log(close).diff().rolling(252).sum())
idx = close.index

def run(fill_bp):                         # cost per fill, in bp of notional
    cash, pos, last, pending = 1_000_000.0, dict.fromkeys(SYMS, 0), {}, []
    equity = pd.Series(np.nan, index=idx)
    for i, t in enumerate(idx):
        for s, tgt in pending:
            dq, o = tgt - pos[s], open_.at[t, s]
            if dq and not np.isnan(o):
                cash = round(cash - dq * o - round(abs(dq) * o * fill_bp(s) * 1e-4, 2), 2)
                pos[s] += dq
        pending = []
        eq = cash + sum(pos[s] * close.at[t, s] for s in SYMS if pos[s])
        equity[t] = eq
        if i == len(idx) - 1:
            break
        live = [s for s in SYMS if not np.isnan(sig.at[t, s])]
        for s in live:
            if sig.at[t, s] != last.get(s, 0.0) or t.month != idx[i + 1].month:
                pending.append((s, int(sig.at[t, s] * eq / len(live) / close.at[t, s])))
            last[s] = sig.at[t, s]
    return equity

runs = {"free": run(lambda s: 0.0),
        "spread+comm": run(lambda s: HS[s] + 0.2),
        "impact $100M": run(lambda s: HS[s] + 0.2 + 7.5)}

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
for name, eq in runs.items():
    ax1.plot(eq.loc["2001-01-03":] / 1e6, label=name)
    r = eq.pct_change().loc["2001-01-03":].dropna()
    print(f"{name:12s}: final ${eq.iloc[-1]:,.0f}, "
          f"Sharpe {np.sqrt(252) * r.mean() / r.std():.2f}")
ax1.set_yscale("log")
ax1.legend()
ax2.plot((runs["free"] / runs["impact $100M"] - 1).loc["2001-01-03":])
ax2.set_ylabel("free vs impact, cumulative gap")
print({"run_id": "tsmom-nextopen-v1", "cache": "data/part5.parquet",
      "rebalance": "sign flip + month-end", "fills": "next open + bp fee"})
plt.show()
# => free        : final $2,591,150, Sharpe 0.39
#    spread+comm : final $2,522,514, Sharpe 0.38
#    impact $100M: final $2,072,098, Sharpe 0.31
#    {'run_id': 'tsmom-nextopen-v1', 'cache': 'data/part5.parquet', 'rebalance': 'sign flip + month-end', 'fills': 'next open + bp fee'}
```

The middle line is the quiet triumph of the part: `spread+comm` reprints lesson four's run — $2,522,514, Sharpe 0.38 — because it *is* lesson four's run, the same deterministic loop on the same frozen bars, and determinism is what makes the other two lines interpretable as controlled experiments rather than reruns of a dice roll. The comparisons then say exactly what [lesson three's](03-order-management-and-fill-simulation.md) weight-based dial predicted, now in dollars. Costs at real spreads are nearly invisible in the ratio (0.39 to 0.38) yet total $68,636 of missing final equity against the free run — more than the $50,793 actually paid in fees, because money spent in 2003 is money that never compounded for twenty-two years. The pretended $100M impact regime cuts the final stake by half a million and the Sharpe to 0.31, and the difference panel shows the *shape* of that loss: a smooth, patient bleed, never an event — precisely why cost bugs and cost realities alike escape notice in equity-curve squinting and must be measured, not eyeballed. The metadata dict printed with the figure is the habit that scales: a comparison is only an experiment when every key but one matches, and the run that cannot state its keys cannot be compared with anything, including its future self.

!!! warning "If you cannot replay it from the log, it did not happen"
    A backtest result that exists only as a number in a notebook is a rumor with a decimal point. The standard this part closes on is archival: every run writes its fills, its metadata names its configuration and its data, and any claim the run makes is checkable years later by replaying two parquet files through eleven lines of arithmetic. Live trading will not negotiate on this — the broker's statement is a trade log, reconciliation against it is nightly ritual, and regulators call the whole apparatus an audit trail. Build the habit in simulation, where the only thing at stake is your own honesty, because the discipline does not appear on demand the first night real money needs it.

!!! abstract "Key takeaways"
    - The trade log is one row per fill — `run_id, ts, symbol, qty, px, fee, cash_after` — with signed integer quantities, full-precision prices, and a deliberately redundant running cash column, because redundancy you can check is evidence: 1,103 rows to `data/part5trades.parquet`.
    - Replay is the completeness test, and it passes to the penny: cash re-derived at every fill with $0.00 worst drift, final book long 1,429 SPY / short 10,358 TLT / long 2,737 GLD, final equity $2,522,514.08 — identical to the tearsheet, with no engine code involved.
    - FIFO pairing turns 1,103 fills into 1,088 closed round trips plus exactly the three still-open lots the replay holds, and the layers reconcile: $1,153,202 realized, minus $50,793 fees, plus unrealized on the open lots, equals the curve's $1.52M gain.
    - Trade statistics inherit their pairing convention: the same strategy shows a 3-day median hold at 34% hit rate sliced by signal runs, but a 110-day median at 57% sliced by FIFO lots — best trip +$231,339 over 39 months on SPY, worst −$85,484 in one month of 2008 gold.
    - The equity-and-drawdown figure is drawn from the log by a third independent route and agrees again — final $2,522,514, maxDD −27.3%, trough 2003-06-25 — with the log scale keeping 2001 and 2025 honest and the drawdown panel showing what the high-water mark saw.
    - Exposure drawn from fills breathes in a way weight matrices cannot: gross averaged 1.00 but ranged 0.68–1.49, integer shares levering the book as equity moved between rebalances — realized risk the vectorized abstraction structurally hides.
    - Comparing runs that differ in one key only: free $2,591,150 (0.39), spread+comm $2,522,514 (0.38), $100M impact $2,072,098 (0.31) — $50,793 of fees compounds into $68,636 of missing equity, and the cost gap accrues as a smooth bleed only measurement can see.

## Where this goes next

Part V is complete: events and a clock, books that reconcile to the penny, a broker with stated assumptions, a tearsheet that can be cross-examined, and runs that survive as replayable artifacts. [Part VI — Live Infrastructure](../part-06-live-infrastructure/index.md) takes exactly these components across the line that matters: the data handler becomes a scheduled pipeline against real feeds, the simulated broker becomes an API client with retries and failure modes, the reconciliation invariant becomes a nightly check against the broker's own statement, and the trade log stops being good practice and becomes the audit trail the whole operation stands on. The engine was never the destination — it was the rehearsal space where every one of those habits could be built while mistakes were still free.
