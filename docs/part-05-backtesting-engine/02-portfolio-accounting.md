# Portfolio Accounting

The portfolio is the engine's memory, and accounting is where backtests quietly rot. [Lesson one](01-architecture-and-event-driven-design.md) ended with a confession: its skeleton tracked an entire trading history in a single integer that believed every fill and remembered nothing — not the price paid, not which shares came first, not whether any cash remained. This lesson replaces that integer with a book: positions built from lots, a cash ledger that prices fills, commissions, and financing, mark-to-market equity, and the two events every long-lived book eventually meets — corporate actions and foreign currencies. None of it is glamorous, and all of it is load-bearing: every performance number in this part is arithmetic on the equity curve, and the equity curve is nothing but this lesson's cash plus this lesson's marks.

The through-line is a discipline borrowed from the [brokers and clearing firms](../part-01-foundations/04-exchanges-brokers-ecns.md) whose books settle real money: every state change is an event applied to the ledger, and after every event the books must still balance — exactly, to the penny, with the check automated. The lesson ends by torturing the book with ten thousand random fills to show that "to the penny" is not a figure of speech.

## A position is its lots

The skeleton stored `pos = 100` and threw away everything else a book needs: when the shares were bought, at what basis, and in what order. The professional representation keeps the receipts — a position is a FIFO queue of *lots*, and a fill either eats the oldest opposing lots or opens a new one:

```python
from dataclasses import dataclass, field

@dataclass
class Lot:
    qty: int                              # signed: positive long, negative short
    px: float

@dataclass
class Position:
    lots: list[Lot] = field(default_factory=list)
    realized: float = 0.0

    @property
    def qty(self) -> int:
        return sum(lot.qty for lot in self.lots)

    def apply(self, qty: int, px: float) -> None:
        while qty and self.lots and self.lots[0].qty * qty < 0:
            lot = self.lots[0]            # a closing trade eats the oldest lot
            closed = min(abs(qty), abs(lot.qty)) * (1 if lot.qty > 0 else -1)
            self.realized = round(self.realized + closed * (px - lot.px), 2)
            lot.qty -= closed
            qty += closed
            if lot.qty == 0:
                self.lots.pop(0)
        if qty:                           # whatever remains opens a new lot
            self.lots.append(Lot(qty, px))

p = Position()
for q, px in [(+100, 100.0), (+100, 110.0), (-300, 120.0)]:
    p.apply(q, px)
    print(f"trade {q:+d} @ {px:.0f} -> qty {p.qty:+d}, "
          f"lots {[(l.qty, l.px) for l in p.lots]}, realized {p.realized:,.0f}")
# => trade +100 @ 100 -> qty +100, lots [(100, 100.0)], realized 0
#    trade +100 @ 110 -> qty +200, lots [(100, 100.0), (100, 110.0)], realized 0
#    trade -300 @ 120 -> qty -100, lots [(-100, 120.0)], realized 3,000
```

The `while` loop is the entire algorithm, and the third trade is why it must be a loop. Selling 300 shares against a 200-share long is *two* economic acts wearing one fill: it closes both open lots — a hundred bought at 100 and a hundred at 110, sold at 120, realizing 3,000 — and then reverses, opening a fresh short lot of 100 shares whose basis is the same 120. The book flips direction without a special case, because "close oldest opposing lots, then open with the remainder" handles a partial close, a full close, and a reversal identically. Note what the lot queue preserves that the integer destroyed: each closing fill is matched to specific opening fills, with their dates and prices, which is exactly the pairing [Trade Logs and Visualization](05-trade-logs-and-visualization.md) will need to reconstruct round trips from a raw fill stream. And the `round(..., 2)` on every mutation of `realized` is not tidiness — it is a policy with a theorem behind it, proven at the end of this lesson.

## Two answers for what you made

Brokerage statements report average cost; trade analysis demands FIFO. Same fills, different books — buy a hundred at 100, a hundred at 110, sell 150 at 120, then mark the remainder at 120:

```python
# buy 100 @ 100, buy 100 @ 110, sell 150 @ 120, then mark the rest at 120
fills = [(+100, 100.0), (+100, 110.0), (-150, 120.0)]
mark = 120.0

# FIFO: the sale closes the oldest shares first
fifo_realized = 100 * (120 - 100) + 50 * (120 - 110)
fifo_open = [(50, 110.0)]

# average cost: the sale closes 150 shares at the blended basis
avg = (100 * 100.0 + 100 * 110.0) / 200
avg_realized = 150 * (120 - avg)
avg_open = [(50, avg)]

for name, realized, lots in [("FIFO", fifo_realized, fifo_open),
                             ("avg cost", avg_realized, avg_open)]:
    unreal = sum(q * (mark - px) for q, px in lots)
    print(f"{name:8s}: realized {realized:,.0f}, open lot {lots}, "
          f"unrealized {unreal:,.0f}, total {realized + unreal:,.0f}")
# => FIFO    : realized 2,500, open lot [(50, 110.0)], unrealized 500, total 3,000
#    avg cost: realized 2,250, open lot [(50, 105.0)], unrealized 750, total 3,000
```

Both totals print 3,000, and that agreement is the first lesson: the accounting method can never change how much money you made, only how the total splits between *realized* and *unrealized* — FIFO says you banked 2,500 and are owed 500 by the market, average cost says 2,250 and 750. The split is what matters downstream. Tax authorities care which lots you sold; a trader's sense of "banked" versus "at risk" is built on it; and — the reason this course sides with FIFO — trade analysis lives or dies on it. Under FIFO the 150-share sale decomposes into two auditable statements: the hundred shares bought first earned 20 points over their holding period, the next fifty earned 10. Under average cost the sale closed 150 shares of an undifferentiated blend whose basis, 105, was never any trade's price — entry dates gone, holding periods gone, per-trade PnL gone. An engine that accounts by average cost can still draw an equity curve, but it can never again answer "which trades made the money?", and lesson five's round-trip analysis is built entirely on that answer.

## The cash ledger

Positions are half the book; cash is the other half, and it is the half that punishes hand-waving, because *everything* touches it — fills, commissions, and the financing most backtests pretend not to notice:

```python
from datetime import date

CASH_RATE = 0.04                          # annual, charged on negative balances

ledger, cash = [], 100_000.00

def book(ts, desc, amount):
    global cash
    cash = round(cash + amount, 2)
    ledger.append((ts, desc, amount, cash))

book(date(2024, 1, 3), "buy 800 SPY @ 468.79", -800 * 468.79)
book(date(2024, 1, 3), "commission", -7.50)
for day in [4, 5]:                        # negative cash pays the financing desk
    book(date(2024, 1, day), "financing", round(cash * CASH_RATE / 360, 2))
book(date(2024, 1, 5), "sell 800 SPY @ 470.05", 800 * 470.05)
book(date(2024, 1, 5), "commission", -7.52)

for ts, desc, amount, bal in ledger:
    print(f"{ts}  {desc:21s} {amount:+12,.2f}  balance {bal:12,.2f}")
# => 2024-01-03  buy 800 SPY @ 468.79   -375,032.00  balance  -275,032.00
#    2024-01-03  commission                   -7.50  balance  -275,039.50
#    2024-01-04  financing                   -30.56  balance  -275,070.06
#    2024-01-05  financing                   -30.56  balance  -275,100.62
#    2024-01-05  sell 800 SPY @ 470.05  +376,040.00  balance   100,939.38
#    2024-01-05  commission                   -7.52  balance   100,931.86
```

Read the balance column like a story. A $100,000 account bought $375,032 of SPY, and the ledger did not refuse — whether that order *should* have been rejected is a risk-control question that belongs to the broker component of [the next lesson](03-order-management-and-fill-simulation.md) — it simply priced the leverage: a negative balance accrues financing at 4%/360, $30.56 per day, booked as its own ledger line with its own timestamp. Two days later the position is sold and the account lands at 100,931.86. The decomposition is exact: 800 shares times the $1.26 price gain is $1,008.00 gross, minus $15.02 of commissions, minus $61.12 of financing — a fifth of the gross profit gone to two lines most vectorized backtests do not have a row for. That is the ledger's virtue: nothing is a percentage haircut applied at the end; every cost is an event, timestamped, signed, and auditable. When the full engine run in [lesson four](04-performance-metrics-and-reporting.md) omits financing to stay comparable with Part IV's costing, it will say so out loud — a *stated* omission, which only a ledger this explicit makes possible.

## Mark-to-market and the equity curve

Cash plus lots is still not a performance number. The book becomes an equity curve only when someone asserts what the open positions are worth, and that assertion has a formula:

$$
E_t \;=\; C_t \;+\; \sum_i q_{i,t}\, p_{i,t},
$$

cash plus, for every symbol, position times mark price. Marked with real bars — long 700 SPY, short 1,200 TLT at 2024's first opens, held through the quarter:

```python
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
q1 = {s: bars.xs(s, axis=1, level=1).dropna().loc["2024-01":"2024-03"]
      for s in ["SPY", "TLT"]}

cash, pos = 1_000_000.00, {}
for sym, qty in [("SPY", +700), ("TLT", -1200)]:
    px = round(q1[sym]["Open"].iloc[0], 2)  # filled at the year's first open
    cash = round(cash - qty * px, 2)
    pos[sym] = qty
    print(f"fill {sym} {qty:+d} @ {px}")

equity = pd.Series({ts: cash + sum(pos[s] * q1[s]["Close"].at[ts] for s in pos)
                    for ts in q1["SPY"].index})
print(f"equity start {equity.iloc[0]:,.2f}, end {equity.iloc[-1]:,.2f}")
print(f"minimum {equity.min():,.2f} on {equity.idxmin():%Y-%m-%d}")
# => fill SPY +700 @ 458.33
#    fill TLT -1200 @ 88.33
#    equity start 1,000,247.95, end 1,038,944.15
#    minimum 997,182.64 on 2024-01-03
```

Cash moved twice — once per fill, on January 2nd — and never again; everything after that is marking. The short leg's mechanics deserve a slow read: selling 1,200 TLT *added* $105,996 to cash, and the book carries a negative position marked at each day's close, so the position's contribution to equity is negative and shrinks as TLT falls — which it did, handing both legs a profit and the book +3.9% for the quarter. The minimum is instructive too: two days in, the book was briefly under water at 997,182.64, a fact the final number alone would never confess and exactly the kind of path information [drawdown analysis](04-performance-metrics-and-reporting.md) is built from. One policy is hiding in plain sight: *marks are closes from the frozen cache*. Marking at the close is a choice — a mid, a settlement price, or a stale quote would each produce a different curve — and in live trading the mark source becomes a component of its own, which is [Part VI's](../part-06-live-infrastructure/index.md) problem. In this part the policy is fixed and stated: fills price at opens, marks price at closes, both from `data/part5.parquet`.

## Corporate actions

A long-lived book will eventually hold a stock through a split or a dividend, and both arrive as events that touch lots and cash directly. The test of correct handling is an invariant: a split moves *nothing* of value, a dividend moves value from the issuer to your cash:

```python
lots = [(100, 400.0), (50, 440.0)]        # an open book: (qty, basis px)
cash, mark = 10_000.00, 500.0

def equity():
    return round(cash + sum(q * mark for q, px in lots), 2)

print(f"before split : qty {sum(q for q, _ in lots)}, equity {equity():,.2f}")

ratio = 4                                 # 4-for-1: qty times 4, prices over 4
lots = [(q * ratio, round(px / ratio, 4)) for q, px in lots]
mark = round(mark / ratio, 4)
print(f"after split  : qty {sum(q for q, _ in lots)}, equity {equity():,.2f}")

dps = 0.25                                # declared per post-split share
cash = round(cash + dps * sum(q for q, _ in lots), 2)
print(f"after dividend: cash {cash:,.2f}, equity {equity():,.2f}")
# => before split : qty 150, equity 85,000.00
#    after split  : qty 600, equity 85,000.00
#    after dividend: cash 10,150.00, equity 85,150.00
```

The split multiplies every lot's quantity by four and divides every price — basis and mark — by four, and equity prints 85,000.00 on both sides: that unchanged number *is* the unit test, and Apple's shareholders ran it live in 2014 (7-for-1) and again in 2020 (4-for-1). Dividing the basis, not just the mark, is what keeps FIFO honest: each lot's future realized PnL is preserved through the action, so round trips that straddle a split still report the truth. (Ratios like 7-for-1 can leave fractional shares; real brokers pay "cash in lieu" for the fraction — one more small cash event.) The dividend then adds $150 to cash and equity rises to 85,150. Now the doctrine that protects this part from a subtle double-count: **our cache is dividend-adjusted, so our engine must not also credit dividends as cash.** `auto_adjust=True` already folded every SPY, TLT, and GLD distribution into the price history — the adjusted equity curve is a total-return curve. An engine that books dividend cash *on top of* adjusted prices counts every payout twice; an engine on *unadjusted* prices that forgets dividend events undercounts SPY's total return by roughly its 1.5–2% yield, compounding for two decades. Pick one regime, state it, and test it with the invariant above. This part's choice is adjusted-prices-no-dividend-events, chosen so that engine results reconcile against Part IV's vectorized numbers, which made the same choice implicitly the moment [Part III](../part-03-statistics/01-probability-and-random-variables.md) downloaded adjusted closes.

## One book, many currencies

The moment a book holds anything not quoted in its base currency, every mark needs one more factor:

$$
E_t \;=\; C_t \;+\; \sum_i q_{i,t}\, p_{i,t}\, f_{i,t},
$$

where $f_{i,t}$ converts symbol $i$'s quote currency into the base. The EURUSD series frozen into the cache in lesson one now reveals its purpose. Hold a Frankfurt-listed position and pin its *local* price flat — no European market moves at all — through the fourteen months when the euro slid to parity:

```python
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
fx = bars["Close"]["EURUSD"].dropna().loc["2021-07":"2022-09"]

qty, px_eur = 400, 250.0                  # a Frankfurt listing, price held flat
base = qty * px_eur * fx                  # the USD book moves anyway

print(f"EURUSD {fx.iloc[0]:.4f} -> {fx.iloc[-1]:.4f} "
      f"({fx.index[0]:%Y-%m-%d} -> {fx.index[-1]:%Y-%m-%d})")
print(f"local PnL EUR 0.00; base PnL USD {base.iloc[-1] - base.iloc[0]:+,.0f} "
      f"({fx.iloc[-1] / fx.iloc[0] - 1:+.1%})")
print(f"parity first crossed {fx[fx < 1.0].index[0]:%Y-%m-%d}")
# => EURUSD 1.1857 -> 0.9830 (2021-07-01 -> 2022-09-30)
#    local PnL EUR 0.00; base PnL USD -20,272 (-17.1%)
#    parity first crossed 2022-08-23
```

A position that never gained or lost a single euro cost its dollar-based owner $20,272 — 17.1% — because EURUSD fell from 1.1857 through parity (first crossed 2022-08-23) to 0.9830. The accounting lesson is that every foreign holding is secretly *two* positions: the asset, and an unhedged long in its quote currency, opened silently the moment you bought and never shown on any statement as a position. A multi-currency book therefore keeps cash balances per currency, converts marks through $f_{i,t}$ daily, and reports one base-currency equity — the only number a risk system should consume, because it is the only number that moves when either leg does. Whether to *hedge* the currency leg is a portfolio decision, not an accounting one; the book's job is merely to make the exposure visible instead of silent. (The union calendar returns here too: EURUSD prints on days NYSE sleeps, which is why the engine's components read per-symbol frames and convert at each mark's own date — the discipline lesson one established when the cache's 6,611-row shape first appeared.)

## Reconcile after every event

Everything above is arithmetic a first-year accountant could check by hand — for six fills. An engine will process thousands, through reversals, shorts, and commissions, and the only version of "the books balance" that survives that volume is one the engine checks itself, after every single event. Ten thousand random fills, two invariants, seed 42:

```python
import numpy as np

rng = np.random.default_rng(42)
START = 1_000_000.00
cash, lots = START, []                    # FIFO lots: [signed qty, basis px]
realized = fees = 0.0
worst = 0.0

for _ in range(10_000):
    qty = int(rng.integers(1, 501)) * (1 if rng.random() < 0.5 else -1)
    px = round(float(rng.uniform(50, 150)), 2)
    fee = round(abs(qty) * px * 2e-5, 2)
    cash = round(cash - qty * px - fee, 2)
    fees = round(fees + fee, 2)
    while qty and lots and lots[0][0] * qty < 0:
        lot = lots[0]
        closed = min(abs(qty), abs(lot[0])) * (1 if lot[0] > 0 else -1)
        realized = round(realized + closed * (px - lot[1]), 2)
        lot[0] -= closed
        qty += closed
        if lot[0] == 0:
            lots.pop(0)
    if qty:
        lots.append([qty, px])
    qs = [q for q, _ in lots]             # invariant 1: the book is one-sided
    assert not qs or min(qs) > 0 or max(qs) < 0
    pos = sum(qs)                         # invariant 2: the equity identity
    unrealized = sum(q * (px - p) for q, p in lots)
    drift = abs(cash + pos * px - (START + realized + unrealized - fees))
    worst = max(worst, drift)

print(f"10,000 fills: final position {pos:+,d} shares, cash {cash:,.2f}")
print(f"realized {realized:+,.2f}, fees {fees:,.2f}")
print(f"worst equity-identity drift across every fill: ${worst:.2f}")
# => 10,000 fills: final position -26,530 shares, cash 2,415,222.75
#    realized -1,172,253.65, fees 5,008.37
#    worst equity-identity drift across every fill: $0.00
```

The random book is a disaster as a strategy — it realized a $1.17M loss, which is what trading noise at random should do — and that is precisely why it is a good test: the invariants must hold on *any* fill stream, not just flattering ones. Invariant one says a FIFO book is always one-sided — the close-then-open algorithm can never leave a long lot queued behind a short. Invariant two is the equity identity: cash plus position marked at the last fill's price must equal starting equity plus realized plus unrealized minus fees, and across ten thousand events the worst violation was $0.00 — not "small", *zero*. That exactness is engineered, not lucky: integer share counts, prices quoted in whole cents, and a `round(..., 2)` at every mutation mean every quantity in the identity is an exact multiple of a cent, and the sub-penny float residue (order $10^{-10}$) is re-snapped before it can compound. Loosen any of it — fractional shares, unrounded fees — and drift becomes nonzero-but-small, and you have lost the property that makes the check useful: *binary* failure. A book that reconciles to $0.00 turns every future bug into a loud assertion at the exact event that caused it; a book that reconciles to "about a dollar" turns every bug into a debate.

!!! warning "If the books do not reconcile to the penny, nothing downstream is evidence"
    Every number this part will report — Sharpe, drawdown, turnover, the tearsheet entire — is arithmetic on the equity curve, and the equity curve is cash plus marks. A sign error in short proceeds, a dividend double-counted on adjusted prices, a financing charge silently skipped: each produces a curve that is smoothly, plausibly wrong, and every statistic computed from it inherits the lie with error bars attached. Reconciliation is not bookkeeping hygiene; it is the difference between measuring a strategy and measuring your bug. The check costs a microsecond per event and its verdict is binary — spend the microsecond.

!!! abstract "Key takeaways"
    - A position is a FIFO queue of signed lots, and close-oldest-then-open-remainder handles partial closes, full closes, and reversals with one loop: selling 300 against a 200-share long realized 3,000 and opened a fresh −100 lot at 120 in a single fill.
    - Accounting method changes the split, never the sum: the same fills report realized 2,500 (FIFO) versus 2,250 (average cost) inside an identical 3,000 total — and only FIFO preserves the trade pairing that lesson five's round-trip analysis requires.
    - The cash ledger prices what vectorized backtests wave away: a $100,000 account leveraged into $375,032 of SPY netted +931.86 only after $15.02 of commissions and $61.12 of financing at 4%/360 — a fifth of the gross gone to lines without a vectorized row.
    - Equity is cash plus marks, and the path is part of the answer: the long-SPY/short-TLT book went 1,000,247.95 → 1,038,944.15 over Q1-2024, dipping to 997,182.64 two days in — information the final number never confesses.
    - A split is a no-op on value (equity 85,000.00 before and after the 4-for-1, with lot bases divided to keep FIFO honest); a dividend credits cash — but only on unadjusted prices, because this part's adjusted cache already carries every payout in the price and booking it again double-counts.
    - Every foreign holding is two positions: a flat-in-EUR asset lost $20,272 (−17.1%) in base currency as EURUSD fell 1.1857 → 0.9830 through parity (2022-08-23) — exposure the book's job is to make visible.
    - Ten thousand seed-42 random fills, reversals and shorts included, held both invariants with worst equity-identity drift $0.00 — an exactness engineered by integer shares and cent-rounding, and worth engineering because it makes failure binary.

## Where this goes next

This lesson booked every fill it was handed and asked no questions about where fills come from. That is the other half of the simulated broker's job: [Order Management and Fill Simulation](03-order-management-and-fill-simulation.md) builds the order lifecycle state machine, tests market, limit, and stop orders against real bars, and confronts the question this lesson could defer — *what price do you deserve?* — with three fill models of increasing honesty, partial fills when volume runs thin, and the rejection paths that turn a ledger that prices leverage into a broker that polices it.
