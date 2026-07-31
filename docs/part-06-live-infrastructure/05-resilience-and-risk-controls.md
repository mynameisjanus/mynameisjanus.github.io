# Resilience and Risk Controls

[Lesson four](04-monitoring-logging-alerting.md) ended on its own sharpest confession: a process can be alive, ready, and completely wrong about the world, because its memory died with its last crash while the broker kept trading. Observability sees failure; nothing yet *survives* it. This lesson is where the system earns the right to run unattended, and it is also the lesson two old promises have been waiting for. [Part V's order-management lesson](../part-05-backtesting-engine/03-order-management-and-fill-simulation.md) built a state machine that notices when the story being told cannot be true, and said this part would stress that seam hard. And [Part II's async lesson](../part-02-python/04-async-and-apis.md), teaching retries, drew a line it refused to cross: a timed-out *download* can be retried with a clear conscience, but a timed-out *order* cannot — that case, it said, is handled with idempotency keys and order-state reconciliation, never with a bare retry. Both debts are paid below, with the machinery that makes payment possible.

The lesson's arc is a single argument made seven ways: **a live trading system may never trust its own memory**. A restart reads its state from the stores instead of remembering it; a nightly reconciliation treats the broker's records as senior to ours; a resubmission carries the same client order ID so the broker can recognize the repeat; a watchdog distrusts even a beating heart unless work is also advancing; circuit breakers assume the anomaly nobody predicted is coming anyway; the pre-trade gauntlet assumes every upstream component will eventually go insane; and the kill switch assumes all of the above will someday fail at once. None of this is pessimism. It is the price of the claim lesson one made when it said the system must behave sensibly at 3am with no human watching — *sensibly* was always going to mean *suspiciously*.

## A restart must read its state, not remember it

Here is the crash [lesson four's health endpoint](04-monitoring-logging-alerting.md) cannot see. The execution engine sells 100 SPY; the fill reaches PostgreSQL; the process dies *between* writing the ledger and updating its in-memory book. Docker revives it in seconds, every health check passes — and its memory is 100 shares wrong:

```python
import psycopg

memory = {"GLD": 2737, "SPY": 1429, "TLT": -10358}     # the process's recall

with psycopg.connect("dbname=quant") as conn:
    conn.execute("DELETE FROM fills WHERE run_id = 'tsmom-live-v1'")
    conn.execute(
        "INSERT INTO fills (run_id, ts, symbol, qty, px, fee, cash_after) "
        "VALUES ('tsmom-live-v1', '2025-07-01', 'SPY', -100, 611.20, 0.31, "
        "1747086.01)")                    # the ledger heard about the fill...
    # ...and the process dies HERE, before its dict does

    rebuilt = dict(conn.execute(
        "SELECT symbol, qty FROM positions ORDER BY symbol").fetchall())
    cash = conn.execute(
        "SELECT cash_after FROM fills ORDER BY id DESC LIMIT 1").fetchone()[0]

rebuilt = {s: int(q) for s, q in rebuilt.items()}
print("memory said:", memory)
print("stores say :", rebuilt, f"cash ${cash:,}")
print(f"drift      : SPY {rebuilt['SPY'] - memory['SPY']:+d} shares — "
      f"a restart that trusts memory re-sells them")
# => memory said: {'GLD': 2737, 'SPY': 1429, 'TLT': -10358}
#    stores say : {'GLD': 2737, 'SPY': 1329, 'TLT': -10358} cash $1,747,086.01
#    drift      : SPY -100 shares — a restart that trusts memory re-sells them
```

The recovery is almost insultingly simple — one `SELECT` from the positions view, one for the cash — and that simplicity was *purchased*, deliberately, by three earlier decisions. [Lesson one](01-system-architecture.md) classified positions as reconstructable and open orders as durable; [lesson two](02-scheduling-and-data-plumbing.md) made positions a view over fills, so the rebuilt book cannot disagree with the ledger; and the write *order* in the engine puts the store before the memory, so a crash in the gap leaves the durable record ahead of the process's recall — the safe direction, because a restart that reads the stores is then correct, while a restart that trusts its last dict re-executes the sale it already made and ends 200 shares from the truth. That is the startup sequence, in full: rebuild the book from `positions`, take cash from the last fill, reload working orders from `orders`, refetch marks (never trust a mark that survived a crash — [the TTL](02-scheduling-and-data-plumbing.md) enforces this anyway), and recompute signal state from bars, [lesson one's](01-system-architecture.md) reconstructable class earning its keep. Nothing is remembered; everything is read.

## The broker's statement outranks your database

Crash recovery restores agreement between the process and *our* stores. The harder question is whether our stores agree with *the broker's* — because fills can go missing (a fill event lost in a disconnect) or arrive corrupted, and every hour the books are wrong, every downstream number is fiction. The nightly ritual [Part V's closing lesson](../part-05-backtesting-engine/05-trade-logs-and-visualization.md) promised: pull the broker's statement, diff it against our fills, and classify every break. Here, with two injected wounds — sabotage performed inside a transaction so the evidence prints and the repair is a `rollback`:

```python
import pandas as pd
import psycopg

broker = pd.read_parquet("data/part5trades.parquet")   # the broker's copy
b = broker[["ts", "symbol", "qty"]].assign(ts=broker["ts"].dt.date)

with psycopg.connect("dbname=quant") as conn:
    conn.execute("DELETE FROM fills WHERE run_id = 'tsmom-live-v1'")
    conn.commit()
    # sabotage our own books — inside a transaction we will roll back
    conn.execute("DELETE FROM fills "
                 "WHERE ts = '2008-02-01' AND symbol = 'GLD'")
    conn.execute("UPDATE fills SET qty = qty + 10 "
                 "WHERE ts = '2011-02-01' AND symbol = 'TLT'")
    rows = conn.execute("SELECT ts, symbol, qty FROM fills").fetchall()
    conn.rollback()                       # the evidence, then the repair

ours = pd.DataFrame(rows, columns=["ts", "symbol", "qty"])
m = b.merge(ours, on=["ts", "symbol"], how="outer",
            suffixes=("_broker", "_ours"), indicator=True)
missing = m[m["_merge"] == "left_only"]
diff = m[(m["_merge"] == "both") & (m["qty_broker"] != m["qty_ours"])]

print(f"broker statement: {len(b)} fills   our books: {len(ours)} fills")
for r in missing.itertuples():
    print(f"MISSING   {r.ts} {r.symbol} {int(r.qty_broker):+d} "
          f"— on the statement, not in our books")
for r in diff.itertuples():
    print(f"MISMATCH  {r.ts} {r.symbol} broker {int(r.qty_broker):+d} "
          f"vs ours {int(r.qty_ours):+d}")
print(f"verdict: {len(missing) + len(diff)} breaks -> RECON FAILED, "
      f"PAGE, adopt the broker's numbers")
# => broker statement: 1103 fills   our books: 1102 fills
#    MISSING   2008-02-01 GLD -508 — on the statement, not in our books
#    MISMATCH  2011-02-01 TLT broker +73 vs ours +83
#    verdict: 2 breaks -> RECON FAILED, PAGE, adopt the broker's numbers
```

The classifier finds both wounds and names them — a 508-share gold sale our books never heard about, and ten phantom TLT shares from a corrupted quantity. Two policies in the verdict line deserve defending. First, a recon break is a `PAGE`, [lesson four's severest tier](04-monitoring-logging-alerting.md), even at 2am with no order pending: a wrong book contaminates every decision the system will make at the next open, so the clock is already running. Second — and this is the section's title — when the records disagree, *the broker is right*, not because brokers are infallible but because their record is the one that settles: their statement decides what you own, what you owe, and what a court believes, so reconciliation means adopting their numbers into our fills and *then* investigating how ours went wrong, never the reverse. The seniority is the same humility [Part V's replay doctrine](../part-05-backtesting-engine/05-trade-logs-and-visualization.md) taught — the log outranks the engine's memory — extended one level up: the broker's log outranks ours.

## Retry the request, never the order

Now the promised stress test. An order is submitted; two seconds pass; no acknowledgment. The request may have died on the way there (safe to resend) or the ack may have died on the way back (the order is *live* — resending doubles it). The sender cannot distinguish the cases, which is why [Part II](../part-02-python/04-async-and-apis.md) banned the bare retry — and why the fix must live in the message, not in the sender's judgment:

```python
from collections import namedtuple

Req = namedtuple("Req", "coid symbol side qty")

class FlakyBroker:                        # fills happen; acks get lost
    def __init__(self):
        self.seen, self.fills, self.calls = set(), [], 0

    def submit(self, req):
        self.calls += 1
        if req.coid not in self.seen:     # a new coid is a new order
            self.seen.add(req.coid)
            self.fills.append(req.qty)
        if self.calls == 1:               # the first ack dies in transit
            raise TimeoutError("no ack in 2s")
        return "ack"

def naive(broker):                        # retry the ORDER — the bug
    n = 0
    while True:
        n += 1
        try:
            broker.submit(Req(f"order-{n}", "SPY", "buy", 100))
            return
        except TimeoutError:
            pass

def idempotent(broker):                   # retry the REQUEST — same coid
    req = Req("tsmom-live-v1:20250701:SPY:1", "SPY", "buy", 100)
    while True:
        try:
            broker.submit(req)
            return
        except TimeoutError:
            pass

for style in (naive, idempotent):
    b = FlakyBroker()
    style(b)
    print(f"{style.__name__:<10}: bought {sum(b.fills)} shares "
          f"in {len(b.fills)} fill(s)")

LEGAL = {("created", "submitted"), ("submitted", "partial"),
         ("submitted", "filled"), ("partial", "filled"),
         ("submitted", "cancelled"), ("created", "rejected"),
         ("submitted", "rejected")}      # Part V's machine, as a set

updates = ["submitted", "filled", "filled", "partial", "cancelled"]
state, applied, quarantined = "created", [], []
for u in updates:                         # the wire delivers what it wants
    if (state, u) in LEGAL:
        state = u
        applied.append(u)
    else:
        quarantined.append(u)
print(f"applied {applied} -> final state {state!r}")
print(f"quarantined {quarantined} — duplicates and time travel, logged loud")
# => naive     : bought 200 shares in 2 fill(s)
#    idempotent: bought 100 shares in 1 fill(s)
#    applied ['submitted', 'filled'] -> final state 'filled'
#    quarantined ['filled', 'partial', 'cancelled'] — duplicates and time travel, logged loud
```

The naive sender bought 200 shares while believing it bought 100 — the fill happened, the ack died, and the retry with a fresh identifier was, to the broker, simply a second customer order. The idempotent sender made the *identical* number of network calls and bought 100, because the resubmission carried the same `coid` and the broker recognized the repeat: [lesson one](01-system-architecture.md) called the client order ID the most load-bearing string in this part, and this is the moment it bears the load. The second half of the block is the other direction of the same distrust — the broker's *updates* arrive over a network too, so they arrive duplicated and disordered, and [Part V's state machine](../part-05-backtesting-engine/03-order-management-and-fill-simulation.md) is the component that refuses to be gaslit: the duplicate `filled` and the time-traveling `partial` are quarantined, not applied, and quarantined *loudly*, because an illegal transition means either a broker bug or a lost message upstream — both of which are [section two's](#the-brokers-statement-outranks-your-database) business tonight, not silently someone's business never.

## A wedged process passes every health check it can still answer

[Lesson four's](04-monitoring-logging-alerting.md) liveness checks share a failure mode: they are answered by the part of the process that is still working. The classic wedge — a work loop blocked forever on a dead socket while the heartbeat thread, on its own merry schedule, keeps beating — passes every probe and does no work. The watchdog's counter-move costs one integer: alongside *I am alive*, every component must publish *here is how much work I have done*, and the watchdog compares:

```python
import redis

r = redis.Redis(db=15, decode_responses=True)
r.delete("qt:hb:strategy", "qt:progress:strategy",
         "qt:hb:execution", "qt:progress:execution")

r.set("qt:hb:strategy", 1, ex=15)         # heartbeat thread, beating
r.set("qt:progress:strategy", 42)         # work loop: bars processed
r.set("qt:hb:execution", 1, ex=15)        # beating here too...
r.set("qt:progress:execution", 17)        # ...but the work loop is stuck

def check(component, last_progress):
    alive = r.exists(f"qt:hb:{component}") == 1
    progress = int(r.get(f"qt:progress:{component}") or -1)
    verdict = ("healthy" if alive and progress != last_progress else
               "WEDGED - alive but stuck" if alive else "DOWN")
    print(f"{component:<10} alive={str(alive):<5} "
          f"progress {last_progress} -> {progress}  {verdict}")

r.set("qt:progress:strategy", 43)         # strategy worked since last check
check("strategy", 42)
check("execution", 17)                    # execution did not
r.delete("qt:hb:execution")               # then its heartbeat lapses too
check("execution", 17)
# => strategy   alive=True  progress 42 -> 43  healthy
#    execution  alive=True  progress 17 -> 17  WEDGED - alive but stuck
#    execution  alive=False progress 17 -> 17  DOWN
```

Three verdicts, three different plays. `healthy` needs nothing. `DOWN` is the easy one — the heartbeat key expired, and [lesson three's](03-docker-and-cloud-deployment.md) restart policy is already handling it. `WEDGED` is the verdict the whole section exists for: alive by every signal the process can still emit, and stuck by the one signal it cannot fake. The response is *not* an automatic instant kill — a strategy legitimately idles between sessions, so progress is measured against the work that should exist ([lesson two's calendar](02-scheduling-and-data-plumbing.md) knows whether bars should be flowing) and given a grace interval. But once declared, wedged means kill and restart, because a stuck process holds whatever it holds — locks, sockets, half-submitted orders — and the restart path through [section one's recovery](#a-restart-must-read-its-state-not-remember-it) is *designed* to be safe, while the wedge is guaranteed to be nothing. The deeper principle generalizes the lesson's title: never accept a component's self-report when an independent measurement is available — the same suspicion that made positions a view and the broker's statement senior.

## Circuit breakers trip on the anomaly you did not predict

The controls so far respond to failures with known shapes. Circuit breakers are the confession that the *unknown* shape is coming anyway — and the observation that almost every runaway disaster, whatever its cause, must eventually express itself in one of three measurable symptoms: an impossible price, an impossible loss, or an impossible flow of orders. The breaker watches the symptoms and refuses to need the diagnosis:

```python
import numpy as np
import pandas as pd
import redis

r = redis.Redis(db=15, decode_responses=True)
r.delete("qt:halt")

spy = (pd.read_parquet("data/part5.parquet")
         .xs("SPY", axis=1, level=1).dropna())["Close"]
sd = float(np.log(spy).diff().dropna().std())  # 25 years of daily moves

def trip(reason):
    r.set("qt:halt", reason)              # sticky until a human clears it
    print(f"TRIP -> qt:halt = {reason!r}  (PAGE)")

tick = 550.00                             # a print 10% below the last close
z = float(np.log(tick / spy.iloc[-1])) / sd
if abs(z) > 6:                            # no honest daily move is 6 sigma
    trip(f"mark 550.00 is {z:.1f} sigma")

day_pnl = -31_000                         # simulated mark-to-market today
if day_pnl < -0.01 * 2_522_514:           # floor: 1% of equity per day
    trip(f"day pnl ${day_pnl:,}")

orders_last_min = 41                      # a loop is machine-gunning orders
if orders_last_min > 10:
    trip(f"{orders_last_min} orders/min")

print("halt flag now:", r.get("qt:halt"))
# => TRIP -> qt:halt = 'mark 550.00 is -8.6 sigma'  (PAGE)
#    TRIP -> qt:halt = 'day pnl $-31,000'  (PAGE)
#    TRIP -> qt:halt = '41 orders/min'  (PAGE)
#    halt flag now: 41 orders/min
```

Each tripwire is calibrated from evidence, not vibes. The mark filter measures the suspect tick against a quarter century of real SPY volatility from the frozen cache: a 550.00 print against a 611.08 close is −8.6 standard deviations, and nothing that has happened since 2000 — not 2008, not the COVID crash — comes close, so the number is far more likely a feed bug ([lesson one's cents-for-dollars demon](01-system-architecture.md) again) than a market. The PnL floor says a strategy whose [backtested daily volatility is around 12%](../part-05-backtesting-engine/04-performance-metrics-and-reporting.md) annualized losing 1% of equity intraday is off-script *regardless of why*. And the order-rate limit catches the failure mode where the system is confidently, rapidly *doing things* — the runaway rebalance loop that no anomaly detector on prices would ever see. Three properties make a breaker a breaker: it trips on symptoms (so novel causes are covered), it is *sticky* (the flag stays set until a human clears it, because an anomaly that "resolves itself" thirty seconds later is exactly how flapping markets whipsaw an automatic re-arm), and it pages, since by construction a tripped breaker means the machinery has met something it does not understand.

## Every order passes the gauntlet, or no order does

The breaker halts the system; the pre-trade gauntlet disciplines it while it runs. [Lesson one](01-system-architecture.md) placed the risk engine between opinion and intention with a two-check sketch; here it stands fully armed, and the design rule is architectural, not procedural: this is a *chokepoint* — every order from any source passes, or the check is theater. The gauntlet reads the live book from Postgres, the live marks from Redis, and the halt flag before anything else:

```python
import psycopg
import redis

LIMITS = {"max_qty": 25_000, "max_notional": 2_000_000,
          "max_weight": 0.40, "collar_bps": 300}

r = redis.Redis(db=15, decode_responses=True)
r.delete("qt:halt")                       # the incident stands resolved
for s, px in {"SPY": 611.08, "TLT": 84.09, "GLD": 304.83}.items():
    r.set(f"qt:mark:{s}", px, ex=90)

with psycopg.connect("dbname=quant") as conn:
    book = {s: int(q) for s, q in conn.execute(
        "SELECT symbol, qty FROM positions").fetchall()}
    cash = float(conn.execute(
        "SELECT cash_after FROM fills ORDER BY id DESC LIMIT 1").fetchone()[0])
equity = cash + sum(q * float(r.get(f"qt:mark:{s}")) for s, q in book.items())

def gauntlet(symbol, qty, limit_px):
    if r.get("qt:halt"):
        return "reject: halted"
    mark = r.get(f"qt:mark:{symbol}")
    if mark is None:
        return "reject: no live mark"
    mark = float(mark)
    if abs(qty) > LIMITS["max_qty"]:
        return f"reject: |qty| > {LIMITS['max_qty']:,}"
    if abs(qty) * mark > LIMITS["max_notional"]:
        return "reject: notional"
    if abs(limit_px / mark - 1) * 1e4 > LIMITS["collar_bps"]:
        return f"reject: {abs(limit_px / mark - 1) * 1e4:.0f}bps off mark"
    w = abs((book.get(symbol, 0) + qty) * mark) / equity
    if w > LIMITS["max_weight"]:
        return f"reject: weight {w:.2f} > {LIMITS['max_weight']}"
    return "pass"

for sym, qty, px in [("SPY", -100, 611.00),    # a sane trim
                     ("SPY", 40_000, 611.00),  # a fat-fingered zero
                     ("GLD", 2_000, 304.90),   # concentrates the book
                     ("TLT", 500, 92.00),      # limit far from the mark
                     ("EURUSD", 100, 1.17)]:   # nothing prices it
    print(f"{sym:<7} {qty:+8,d} @ {px:<7.2f} -> {gauntlet(sym, qty, px)}")
r.set("qt:halt", "kill switch drill")
print(f"SPY        -100 @ 611.00  -> {gauntlet('SPY', -100, 611.00)}")
# => SPY         -100 @ 611.00  -> pass
#    SPY      +40,000 @ 611.00  -> reject: |qty| > 25,000
#    GLD       +2,000 @ 304.90  -> reject: weight 0.57 > 0.4
#    TLT         +500 @ 92.00   -> reject: 941bps off mark
#    EURUSD      +100 @ 1.17    -> reject: no live mark
#    SPY        -100 @ 611.00  -> reject: halted
```

Six orders, five distinct refusals, and each check answers a specific historical disaster. The size and notional caps are the fat-finger defense — no single order may be large enough that one bug is one catastrophe. The price collar (941 basis points off the mark is an order that would *fill*, terribly) is the defense against trading on a stale limit against a moved market. The concentration check prices the *resulting* book, not the order: 2,000 more GLD would put 57% of equity in gold, and a sizing bug that arrives in polite, individually-legal increments is still a sizing bug. The missing-mark rejection is [lesson two's TTL](02-scheduling-and-data-plumbing.md) paying off — a system that cannot price an order does not guess. And the final line closes the loop with the breaker: the flag set by section five is honored here, at the moment of submission, every single time. Note what the gauntlet never does, holding [lesson one's rule](01-system-architecture.md): it refuses; it does not repair. A clamped, "helpfully" resized order is a trade nobody designed, sized by an error handler.

## The kill switch must work when everything else does not

Last, the control that assumes the rest of this lesson has failed. The kill switch's design constraints are unusual: it must be triggerable by a human who is panicking, work when the strategy process is wedged and the dashboard is down, and depend on as little of the system as possible. Which is why it is not a feature of any component — it is one Redis key, writable from any shell on the box, checked at the top of every submission path:

```python
import psycopg
import redis

r = redis.Redis(db=15, decode_responses=True)

def submit(order):                        # every submit starts with the flag
    if r.get("qt:halt"):
        return f"BLOCKED  {order}"
    return f"sent     {order}"

r.delete("qt:halt")
print(submit("SPY -100"))
r.set("qt:halt", "breaker: 41 orders/min")
print(submit("SPY -100"))

with psycopg.connect("dbname=quant") as conn:
    book = sorted(conn.execute("SELECT symbol, qty FROM positions").fetchall())

print("freeze : the book stands; nothing new goes out")
print("flatten: the one privileged path — closing orders only:")
for n, (sym, qty) in enumerate(book, 1):
    side = "SELL" if qty > 0 else "BUY"
    print(f"  {side:<4} {abs(int(qty)):>6,} {sym:<4} "
          f"coid tsmom-live-v1:20250701:{sym}:{n}")
print("either way, qt:halt stays set until a human clears it")
# => sent     SPY -100
#    BLOCKED  SPY -100
#    freeze : the book stands; nothing new goes out
#    flatten: the one privileged path — closing orders only:
#      SELL  2,737 GLD  coid tsmom-live-v1:20250701:GLD:1
#      SELL  1,429 SPY  coid tsmom-live-v1:20250701:SPY:2
#      BUY  10,358 TLT  coid tsmom-live-v1:20250701:TLT:3
#    either way, qt:halt stays set until a human clears it
```

The human path is one line at any terminal, memorizable before the emergency: `redis-cli -n 15 SET qt:halt manual` — no dashboard, no deploy, no dependence on the process that is misbehaving. The automatic path is [the breaker](#circuit-breakers-trip-on-the-anomaly-you-did-not-predict), writing the same key. And the flag has two grades of response. **Freeze** — block every new submission, keep the book — is the default, correct when the *machinery* is suspect but the positions are presumably fine: a reject storm, a wedged process, a recon break under investigation. **Flatten** goes further: emit closing orders for the real book — sell the 2,737 GLD and 1,429 SPY longs, buy back the 10,358 TLT short — through the one path allowed to bypass the halt, restricted to trades that strictly reduce exposure, each carrying a `coid` so even panic is idempotent. Flatten is for when the *positions themselves* are the emergency, and the classic operator error is reaching for it when freeze suffices — dumping a fundamentally sound book into a dislocated market converts a scare into a realized loss. The drill discipline settles which is which calmly: both paths tested on paper, criteria written down in advance — and [lesson six's go-live checklist](06-secrets-paper-live-compliance.md) will refuse to promote a system whose kill switch has never been pulled.

!!! warning "When your records and the broker's disagree, the broker is right — especially when yours look fine"
    Every control in this lesson is a refusal to let the system be the judge of its own testimony: the restart reads instead of remembering, the watchdog demands progress instead of pulses, the resubmission lets the broker recognize repeats instead of trusting the sender's certainty. Reconciliation is where the refusal becomes doctrine. Books that pass every internal invariant can still be wrong — the invariants verify the fills you recorded, not the fill you never heard about — and the internally-consistent wrong book is the dangerous one, because nothing about it invites a second look. The broker's statement settles; treat it as senior, adopt its numbers first, investigate your own defect second.

!!! abstract "Key takeaways"
    - A restart reads state instead of remembering it: after a crash between ledger write and memory update, the stores said SPY 1,329 and cash $1,747,086.01 while memory said 1,429 — and the write order (store before memory) makes the readable answer the correct one.
    - Nightly reconciliation diffs the broker's statement against our fills and classifies breaks: 1,103 broker fills vs 1,102 ours surfaced a missing −508 GLD fill and a +10-share TLT corruption — 2 breaks, an immediate page, and the broker's numbers adopted first.
    - Retry the request, never the order: a lost ack turned a naive retry into 200 bought shares, while the same-`coid` resubmission bought exactly 100 — and Part V's state machine, stressed as promised, applied 2 of 5 wire updates and loudly quarantined the duplicate and time-traveling 3.
    - A wedged process passes every check it can still answer: heartbeat plus a progress counter separates healthy (42→43), WEDGED (alive, 17→17), and DOWN — and wedged means kill and restart, because recovery is designed to be safe and the wedge is guaranteed to be nothing.
    - Circuit breakers watch symptoms, not causes: a 550.00 print is −8.6σ against 25 years of real volatility, a −$31,000 day breaches the 1%-of-equity floor, 41 orders/min is a runaway loop — any trip sets a sticky `qt:halt` and pages.
    - The pre-trade gauntlet is a chokepoint with named refusals — qty, notional, a 941bps collar breach, a 0.57 concentration, a missing mark, the halt flag — and it refuses rather than repairs, because a resized order is a trade nobody designed.
    - The kill switch is one Redis key writable from any shell: freeze blocks all new orders, flatten emits closing-only coid-carrying orders (SELL 2,737 GLD, SELL 1,429 SPY, BUY 10,358 TLT), and the flag outlives the emergency until a human clears it.

## Where this goes next

The system now survives its own death: it recovers by reading, reconciles nightly against the senior record, retries without doubling, catches its own wedges, halts on the unforeseen, and can be stopped by one command from any terminal. What remains is not machinery — it is governance. The broker credentials that authenticate every one of those API calls are sitting in an environment variable a `ps` away from any process on the box; nothing yet defines what a system must *prove* in paper trading before the `.env.live` file exists at all; and while [lesson two's fills table](02-scheduling-and-data-plumbing.md) records what happened, nothing yet guarantees the record could not be quietly edited after the fact — a distinction regulators, counterparties, and your own future self all care about intensely. [Secrets, Paper/Live, and Compliance](06-secrets-paper-live-compliance.md) closes the part with the promotion gate, the go-live checklist, and the audit trail — the apparatus that decides whether this system, having learned to survive, has earned the right to touch real money.
