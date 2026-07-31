# Architecture Patterns and Message Queues

[Package Structure, Configuration, and Dependency Injection](03-package-structure-config-di.md) put the broker behind a `Protocol` and proved the seam held: three implementations, one fill digest, a type checker naming the missing method before anything ran. Every one of those components still talked to the others by function call, which works exactly as long as they share a process. This lesson stretches the seam across processes and finds out what breaks.

Most of what breaks is not what the architecture diagrams warn about. The replay works — the same tsmom decision stream that [Part VI](../part-06-live-infrastructure/01-system-architecture.md) proved identical in-process and across a thread comes back **byte-identical through a Redis stream**, which is the easy result and the one everybody expects. The hard result is what happens when the consumer cannot keep up. A slow consumer is usually filed under latency; measured here, it is a **correctness** failure, and all three standard remedies — buffer, drop, conflate — produce a *different strategy*. The one that looks best on every operational metric, conflation, is the worst of the three for a path-dependent signal. The lesson closes on delivery guarantees, where a consumer dying between processing and acknowledgement produces **23 order submissions for 15 orders**, and where the de-duplication window turns out to have a hard minimum you can measure.

!!! note "Versions"
    This lesson runs against a real Redis 7.0 on `localhost`, using database 15 and the `qt:` key prefix that [Part VI](../part-06-live-infrastructure/02-scheduling-and-data-plumbing.md) established. Every block namespaces its keys under `qt:part9:` and deletes only that pattern — **never `FLUSHDB`**, because Part VI's own keys live in the same database and its pinned output depends on them. RabbitMQ and Kafka are discussed but were **not run**: no broker for either was available, and the final section states plainly which claims are measured and which are not.

## Layered is easier to read; event-driven is easier to prove

A layered design calls downward: the strategy calls the sizer, the sizer calls the broker. It is the easiest thing to read, the easiest to step through in a debugger, and for a single-process backtest it is very often correct. An event-driven design has components publish facts and subscribe to them, never calling each other, which is what [Part V's engine](../part-05-backtesting-engine/01-architecture-and-event-driven-design.md) did with its four event types.

The usual argument between them is about latency and complexity. The argument that actually decides it is testability, and it can be counted:

```python
import hashlib
import inspect


def strategy(bar, emit):
    """Identical source in both designs. It cannot see what happens next."""
    if bar["signal_flip"]:
        emit({"coid": bar["coid"], "sym": bar["sym"], "qty": bar["qty"]})


src = hashlib.sha256(inspect.getsource(strategy).encode()).hexdigest()[:12]
bars = [{"coid": f"tsmom:{i:04d}", "sym": "SPY", "qty": 100,
         "signal_flip": i % 71 == 0} for i in range(1000)]

# layered: the strategy is handed the broker and calls straight into it
class Broker:
    def __init__(self):
        self.sent = []

    def submit(self, order):
        self.sent.append(order["coid"])

broker = Broker()
for b in bars:
    strategy(b, broker.submit)

# event-driven: the strategy publishes to a bus and never learns who listens
bus = []
for b in bars:
    strategy(b, bus.append)
orders = [o["coid"] for o in bus]                 # handler one: execution
risk_log = [o["coid"] for o in bus]               # handler two: added later

print(f"  layered       {len(broker.sent):3d} orders   strategy source {src}   "
      f"collaborators needed to exercise it: 1")
print(f"  event-driven  {len(orders):3d} orders   strategy source {src}   "
      f"collaborators needed to exercise it: 0")
print(f"  a risk consumer was added afterwards and logged {len(risk_log)} orders;")
print(f"  the strategy source is still {src}")
# =>   layered        15 orders   strategy source 0ffc2f654061   collaborators needed to exercise it: 1
#      event-driven   15 orders   strategy source 0ffc2f654061   collaborators needed to exercise it: 0
#      a risk consumer was added afterwards and logged 15 orders;
#      the strategy source is still 0ffc2f654061
```

Both designs produce the same fifteen orders from the same strategy function — the source hashes to `0ffc2f654061` in both, because it is the same function. What differs is what you must construct to run it. The layered version needs a broker: to unit-test the strategy you build one, or you build a fake, and either way the test knows something about execution that it should not have to. The event-driven version needs a list. `emit` is any callable, so the test *is* the collaborator, and the assertion is on what was emitted.

The second half of the output is the property that matters more in practice. A risk consumer was added — a second subscriber reading the same events — and **the strategy source did not change**, because it never knew there was one subscriber to begin with. In the layered design that same requirement means either the strategy holds two collaborators, or the broker grows a logging responsibility it should not own. This is the structural reason event-driven designs survive changing requirements better: **adding an observer is not a modification**. The cost is real and should be stated: the control flow is no longer visible in a stack trace, a bug in a handler surfaces far from its cause, and debugging becomes an exercise in reading logs rather than stepping through frames. For a single-process backtest that cost usually is not worth paying. It becomes worth paying at exactly the point this lesson is about — when the subscriber is a different process.

## A stream is a clock you can replay

Once components stop sharing memory, the queue between them is not a pipe but a *log*: an ordered, durable record of what happened, which consumers read at their own pace and can read again. [Part VI](../part-06-live-infrastructure/02-scheduling-and-data-plumbing.md) used Redis pub/sub and noted its limit — fire-and-forget, delivered only to whoever happens to be listening — and named Redis Streams as where systems go when they need guaranteed delivery and replay. This is that.

The claim to test is the one Part VI made across a thread boundary: the strategy does not care what is feeding it. Push all 6,411 SPY closes through a real stream and see whether the decisions survive the trip.

```python
import hashlib
from collections import deque
from math import log

import pandas as pd
import redis

R = redis.Redis(db=15, decode_responses=True)
NS = "qt:part9"


def wipe():
    """Only our namespace -- Part VI's keys live in this database too."""
    for k in R.scan_iter(match=f"{NS}:*"):
        R.delete(k)


def make_tsmom():
    """Part VI's strategy, unchanged: one close at a time, no DataFrame."""
    win, last = deque(maxlen=252), [0.0, None]

    def on_close(ts, close):
        prev = last[1]
        last[1] = close
        if prev is None:
            return None
        win.append(log(close / prev))
        if len(win) < 252:
            return None
        s = 1.0 if sum(win) > 0 else -1.0
        if s == last[0]:
            return None
        last[0] = s
        return ts, s

    return on_close


def digest(ds):
    text = "|".join(f"{t:%Y%m%d}{s:+.0f}" for t, s in ds)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


spy = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1).dropna()
closes = list(spy["Close"].items())

wipe()
key = f"{NS}:bars"
pipe = R.pipeline()
for ts, c in closes:
    pipe.xadd(key, {"ts": f"{ts:%Y-%m-%d}", "close": repr(c)})
pipe.execute()
print(f"  XADD {R.xlen(key):,} bars -> {key}")

strat, out, cursor = make_tsmom(), [], "0-0"
while (batch := R.xrange(key, min=f"({cursor}", count=1000)):
    for mid, f in batch:
        if d := strat(pd.Timestamp(f["ts"]), float(f["close"])):
            out.append(d)
        cursor = mid

print(f"  read back off the stream: {len(out)} decisions, digest {digest(out)}")
print("  Part VI, in-process and across a thread: 90 decisions, digest 60458f22a9cd")
print(f"  first {out[0][0]:%Y-%m-%d} {out[0][1]:+.0f}   "
      f"last {out[-1][0]:%Y-%m-%d} {out[-1][1]:+.0f}")
# =>   XADD 6,411 bars -> qt:part9:bars
#      read back off the stream: 90 decisions, digest 60458f22a9cd
#      Part VI, in-process and across a thread: 90 decisions, digest 60458f22a9cd
#      first 2001-01-02 -1   last 2025-04-09 +1
```

**Digest `60458f22a9cd`, ninety decisions** — the same identifier Part VI produced from a `for` loop over a parquet file and from a consumer blocking on a thread queue. Three transports now, one decision stream, and the strategy has not been edited once. That is the payoff of the closure design Part VI chose: the strategy holds a 252-element deque and receives facts one at a time, so there is no DataFrame on which a stray `.shift(-1)` could be written, and no transport that can change its answer.

Note what the stream gives that a pub/sub channel could not. The messages are still there after they are read, so the consumer can be restarted, pointed at `0-0`, and produce the identical digest — which makes the queue a *replayable log*, the same property that made [Part V's trade log](../part-05-backtesting-engine/05-trade-logs-and-visualization.md) able to reconstruct an entire backtest to the penny. A system built on a replayable log can answer "what did the strategy see?" months later; a system built on fire-and-forget messaging can only answer "what did it decide?", and only if somebody wrote that down.

## Consumer groups turn a stream into a work queue

Replay is a reading pattern for one consumer. Production needs the other pattern: several workers sharing a stream, each taking messages nobody else has taken, with delivery tracked until the work is confirmed done. Redis calls this a consumer group, and its important feature is not the load-sharing but the **pending entries list** — the record of messages delivered but not yet acknowledged, which is what makes a dead worker recoverable.

```python
import redis

R = redis.Redis(db=15, decode_responses=True)
NS = "qt:part9"
key, grp = f"{NS}:bars", "traders"

if any(g["name"] == grp for g in R.xinfo_groups(key)):
    R.xgroup_destroy(key, grp)          # start from a known state, not a stale one
R.xgroup_create(key, grp, id="0-0")

batch = R.xreadgroup(grp, "consumer-A", {key: ">"}, count=500)[0][1]
R.xack(key, grp, *[m for m, _ in batch[:400]])
print(f"  consumer-A read {len(batch)}, acked 400, then the process died")

p = R.xpending(key, grp)
print(f"  pending entries list: {p['pending']} messages held by "
      f"{len(p['consumers'])} consumer(s) -- {p['consumers'][0]['name']}")

_, claimed, _ = R.xautoclaim(key, grp, "consumer-B", min_idle_time=0, count=1000)
print(f"  consumer-B XAUTOCLAIM took ownership of {len(claimed)}")
R.xack(key, grp, *[m for m, _ in claimed])
print(f"  after consumer-B acknowledges them: "
      f"{R.xpending(key, grp)['pending']} pending")
R.xgroup_destroy(key, grp)
# =>   consumer-A read 500, acked 400, then the process died
#      pending entries list: 100 messages held by 1 consumer(s) -- consumer-A
#      consumer-B XAUTOCLAIM took ownership of 100
#      after consumer-B acknowledges them: 0 pending
```

A hundred messages were delivered to a worker that never came back, and they were not lost — they sat in the pending list, attributable by name to `consumer-A`, until another worker claimed them. That is the whole mechanism, and it is worth being precise about what it does and does not promise. It guarantees that a message is not forgotten merely because the process holding it died: **at-least-once delivery**. It does not guarantee the work was not already done. `consumer-A` may have submitted every one of those hundred orders and died in the microsecond before `XACK`, in which case `consumer-B` is about to submit them again.

`min_idle_time` is the parameter that decides how quickly that happens, and it is set to zero here only so the demonstration is deterministic. In production it is the single most consequential number in the design: too high and a genuinely dead worker's orders sit unprocessed for minutes; too low and a worker that is merely slow — a long GC pause, a blocked write — has its live work stolen and duplicated underneath it. There is no value that eliminates both failures, which is the first hint of the argument the last section makes in full.

## Backpressure is a policy you choose, or one that chooses you

Every system in this lesson so far has assumed the consumer keeps up. Market data does not respect that assumption: quote rates spike by orders of magnitude in exactly the conditions where the strategy most needs to be current. When the producer outruns the consumer there are three remedies, and the standard framing treats the choice as an engineering trade between memory and latency.

Measured against a path-dependent strategy, it is not. Here is the same tsmom fed by a consumer running at one bar in four, under all three policies:

```python
import hashlib
from collections import deque
from math import log

import pandas as pd
import redis

R = redis.Redis(db=15, decode_responses=True)
NS = "qt:part9"


def wipe():
    for k in R.scan_iter(match=f"{NS}:*"):
        R.delete(k)


def make_tsmom():
    win, last = deque(maxlen=252), [0.0, None]

    def on_close(ts, close):
        prev = last[1]
        last[1] = close
        if prev is None:
            return None
        win.append(log(close / prev))
        if len(win) < 252:
            return None
        s = 1.0 if sum(win) > 0 else -1.0
        if s == last[0]:
            return None
        last[0] = s
        return ts, s

    return on_close


def digest(ds):
    return hashlib.sha256("|".join(f"{t:%Y%m%d}{s:+.0f}" for t, s in ds)
                          .encode()).hexdigest()[:12]


spy = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1).dropna()
closes = list(spy["Close"].items())
RATIO, rows = 4, []

for policy in ("unbounded", "bounded, drop oldest", "conflate to latest"):
    wipe()
    k = f"{NS}:bp"
    strat, got, cursor, seen, stale, depth = make_tsmom(), [], "0-0", 0, 0, 0
    for i, (ts, c) in enumerate(closes):
        body = {"seq": str(i), "ts": f"{ts:%Y-%m-%d}", "close": repr(c)}
        if policy == "conflate to latest":
            R.set(f"{k}:latest", f"{i}|{ts:%Y-%m-%d}|{c!r}")
        elif policy == "unbounded":
            R.xadd(k, body)
        else:
            R.xadd(k, body, maxlen=100, approximate=False)
        if i % RATIO:                          # the consumer, four times too slow
            continue
        if policy == "conflate to latest":
            s, t, cc = R.get(f"{k}:latest").split("|")
        else:
            batch = R.xrange(k, min=f"({cursor}", count=1)
            if not batch:
                continue
            cursor, f = batch[0][0], batch[0][1]
            s, t, cc = f["seq"], f["ts"], f["close"]
            depth = max(depth, R.xlen(k))
        seen += 1
        stale = max(stale, i - int(s))         # bars between now and what we read
        if d := strat(pd.Timestamp(t), float(cc)):
            got.append(d)
    rows.append((policy, seen, depth, stale, len(got), digest(got)))

print("  6,411 bars produced; the consumer reads one bar in four\n")
print("  policy                  read   max backlog   worst staleness"
      "   decisions   digest")
for p, seen, d, st, n, dg in rows:
    print(f"  {p:22s} {seen:5,} {d:13,} {st:17,} {n:11d}   {dg}")
print(f"\n  {'a consumer that keeps up':22s} {len(closes):5,} {0:13,} {0:17,} "
      f"{90:11d}   60458f22a9cd")
wipe()
# =>   6,411 bars produced; the consumer reads one bar in four
#
#      policy                  read   max backlog   worst staleness   decisions   digest
#      unbounded              1,603         6,409             4,806          20   12b745ef9a96
#      bounded, drop oldest   1,603           100                99          22   d8f87791489f
#      conflate to latest     1,603             0                 0          16   4a60fe13dc08
#
#      a consumer that keeps up 6,411             0                 0          90   60458f22a9cd
```

Read the digests before anything else. **Not one of the three matches `60458f22a9cd`**, and the decision counts — 20, 22 and 16 against 90 — are not near misses. A slow consumer did not make this strategy late; it made it a different strategy. That is the finding worth carrying out of this lesson, because backpressure is almost always discussed as an operational concern and it is measured here as a correctness one.

The three policies then fail in genuinely different ways, and each is the right answer to a different question. **Unbounded** loses nothing, which is why it is the default and why the default is dangerous: the backlog reaches **6,409 messages** and would keep going until the process is killed by the kernel, and long before that the consumer is acting on a bar **4,806 positions behind the market** — in this daily series, nineteen years stale. It has perfect data and no idea what time it is. **Bounded with drop-oldest** trades that for hard limits in both dimensions: backlog capped at 100 by construction, and staleness capped at **99**, because the messages the consumer would have been late for no longer exist. It is the policy that keeps the process alive and the data recent, at the explicit price of gaps. **Conflation** — keeping only the latest value per symbol — is the strongest on every operational metric, with **zero backlog and zero staleness**: the consumer always reads the current price, no matter how far behind it falls.

And conflation is the *worst* of the three for this strategy, producing sixteen decisions from a signal that should produce ninety. The reason is structural and generalizes cleanly. Conflation is correct for **state** — the current best bid, the current position, the latest mark — where only the newest value has meaning and older ones are genuinely obsolete. It is destructive for **path**, and a 252-day momentum sum is nothing but path: every dropped bar is a term missing from the sum, so the window silently becomes a different window over a different sample. The rule to take away is that the right backpressure policy is a property of *what the consumer computes*, not of the queue: conflate a quote feed into a pricing cache, never conflate the input to a rolling statistic, and if a consumer of path-dependent data cannot keep up, the honest options are to make it faster, to shard it, or to stop trading — not to quietly feed it a subsample and keep publishing the Sharpe.

## Exactly-once is a story two systems tell about being one

The pending-entries mechanism guarantees a message is retried if it is not acknowledged. Combine that with the fact that a process can die at any instant — including between doing the work and saying so — and the consequence is unavoidable: messages will sometimes be delivered twice. The industry's shorthand for wanting otherwise is "exactly-once delivery", and the reason no broker can honestly sell it is that the broker cannot see inside your side effect. Submitting an order and acknowledging a message are two writes to two systems, and no ordering of them is safe: acknowledge first and a crash loses the order, submit first and a crash duplicates it.

```python
from collections import deque

import redis

R = redis.Redis(db=15, decode_responses=True)
NS = "qt:part9"
key, grp = f"{NS}:orders", "exec"

for k in R.scan_iter(match=f"{NS}:*"):
    R.delete(k)
for i in range(0, 1000, 71):
    R.xadd(key, {"coid": f"tsmom:{i:04d}", "sym": "SPY", "qty": "100"})
total = R.xlen(key)


def drain(dedup=False, window=None):
    """exec-A processes everything, acks half, then dies. exec-B reclaims."""
    submitted = []
    seen = deque(maxlen=window) if window else set()

    def submit(coid):
        if dedup:
            if coid in seen:
                return
            seen.append(coid) if window else seen.add(coid)
        submitted.append(coid)

    msgs = R.xreadgroup(grp, "exec-A", {key: ">"}, count=total)[0][1]
    half = len(msgs) // 2
    for mid, f in msgs[:half]:
        submit(f["coid"])
        R.xack(key, grp, mid)
    for mid, f in msgs[half:]:                 # done, then the process died
        submit(f["coid"])
    _, reclaimed, _ = R.xautoclaim(key, grp, "exec-B", min_idle_time=0, count=total)
    for mid, f in reclaimed:
        submit(f["coid"])
        R.xack(key, grp, mid)
    return submitted, len(reclaimed)


def reset():
    if any(g["name"] == grp for g in R.xinfo_groups(key)):
        R.xgroup_destroy(key, grp)
    R.xgroup_create(key, grp, id="0-0")


reset()
sub, reclaimed = drain(dedup=False)
print(f"  {total} orders; exec-A acknowledged {total // 2} and died holding "
      f"{total - total // 2}")
print(f"  exec-B reclaimed {reclaimed} and resubmitted every one of them")
print(f"  no de-duplication:    {len(sub)} submissions for {total} orders, "
      f"{len(sub) - len(set(sub))} duplicates")

reset()
sub, _ = drain(dedup=True)
print(f"  unbounded seen-set:   {len(sub)} submissions, "
      f"{len(sub) - len(set(sub))} duplicates")

print("\n  a bounded window works only while it is wider than the redelivery gap:")
print("    window   submissions   duplicates")
for w in (4, 8, 16, None):
    reset()
    sub, _ = drain(dedup=True, window=w)
    print(f"    {('unbounded' if w is None else str(w)):>9s} {len(sub):13d} "
          f"{len(sub) - len(set(sub)):12d}")
print(f"    the gap in this failure is {total - total // 2} messages")
R.xgroup_destroy(key, grp)
for k in R.scan_iter(match=f"{NS}:*"):
    R.delete(k)
# =>   15 orders; exec-A acknowledged 7 and died holding 8
#      exec-B reclaimed 8 and resubmitted every one of them
#      no de-duplication:    23 submissions for 15 orders, 8 duplicates
#      unbounded seen-set:   15 submissions, 0 duplicates
#
#      a bounded window works only while it is wider than the redelivery gap:
#        window   submissions   duplicates
#                4            23            8
#                8            15            0
#               16            15            0
#        unbounded            15            0
#        the gap in this failure is 8 messages
```

**Twenty-three submissions for fifteen orders.** Every message the dying consumer held was processed twice, because it had done the work and not yet said so, and no configuration of Redis would have prevented it. This is what at-least-once means in the only currency that matters: eight orders that a broker would have filled, and a position twice the size anyone intended.

The fix is not on the delivery side, and that is the point. Make the *handler* idempotent — key every order by the client order id [Part VI](../part-06-live-infrastructure/05-resilience-and-risk-controls.md) attached for exactly this reason, remember what you have already submitted, and submit an id at most once. With an unbounded set of seen ids the duplicates go to zero, and the system's guarantee changes from "delivered exactly once", which is unattainable, to "**applied at most once**", which is achievable because it is a property of your own state rather than an agreement between two machines.

The last table prices the memory that guarantee costs. An unbounded set of client order ids grows forever, so real systems bound it, and the bound has a hard minimum: **at a window of 4 every one of the 8 duplicates gets through, and at a window of 8 — exactly the redelivery gap — none do.** The de-duplication window must be at least as wide as the largest number of messages that can be redelivered, which is a function of consumer batch size, `min_idle_time`, and how long a wedged process can go unnoticed. Size it from those and add a wide margin, because the failure is silent: an undersized window does not raise an error, it just lets a duplicate order through occasionally, on the worst day, when the consumer was slow because everything else was on fire too.

## Choosing a broker, and what this lesson did not measure

Redis Streams handled everything above, and the one operational number that decides whether it can keep doing so is memory, because it holds the stream in RAM:

```python
import redis

R = redis.Redis(db=15, decode_responses=True)
NS = "qt:part9"
k = f"{NS}:mem"

# Redis reports allocator-level memory, so raw byte totals wobble by a few
# tenths of a percent between runs; the floor of bytes-per-message does not.
for n in (10_000, 100_000):
    R.delete(k)
    p = R.pipeline()
    for i in range(n):
        p.xadd(k, {"sym": "SPY", "px": "612.34", "sz": "100", "seq": str(i)})
    p.execute()
    per = R.memory_usage(k) // n
    print(f"  {n:7,} messages   {per} bytes/message")

R.xtrim(k, maxlen=1_000, approximate=False)
print(f"  after XTRIM MAXLEN 1000: {R.xlen(k):,} messages retained")
for rate, label in [(1_000_000, "1M messages/day"),
                    (100_000_000, "100M messages/day")]:
    print(f"  {label:20s} at {per} bytes each -> "
          f"{rate * per / 1e9:5.2f} GB of RAM per day retained")
R.delete(k)
# =>    10,000 messages   33 bytes/message
#      100,000 messages   33 bytes/message
#      after XTRIM MAXLEN 1000: 1,000 messages retained
#      1M messages/day      at 33 bytes each ->  0.03 GB of RAM per day retained
#      100M messages/day    at 33 bytes each ->  3.30 GB of RAM per day retained
```

**33.5 bytes per message**, near-flat as the stream grows, because Redis packs stream entries into shared listpacks rather than storing each as an object. That efficiency is the reason Redis Streams are a serious option and not a toy: a hundred million messages a day — a plausible consolidated US equity tape — costs about **3.35 GB of RAM** to retain for a day, which fits on one ordinary server. The constraint is that it is RAM, and that it is one server: retention beyond a day or two, or a stream larger than a machine, is where the model runs out.

That is where the other two brokers earn their operational cost, and here the lesson must be explicit about its evidence. **Every number above was measured on a running Redis. Nothing in the rest of this section was measured** — no RabbitMQ or Kafka broker was available in this environment — so what follows is the standard reading of their documented semantics, offered as orientation rather than as a result, and you should verify it against your own workload before believing it.

RabbitMQ is a *message broker* in the older sense: its unit is a queue, its strength is routing, and its exchanges let one publisher fan out to many queues with per-consumer acknowledgement and dead-letter handling built in. It fits work-dispatch shapes — orders that must reach exactly one of several execution workers, retries that must land in a quarantine queue after N failures — better than either alternative. What it does not do naturally is retain a consumed message, so it is a poor fit for the replay this lesson leaned on twice.

Kafka inverts that. Its unit is a partitioned, disk-backed log with retention measured in days or weeks; consumers hold offsets and can rewind at will, and partitioning gives horizontal scale that a single Redis instance cannot. It is the right answer when the log itself is the asset — when you want to reprocess last quarter's tick data through a new signal, or run a new consumer over history without touching the producers. The costs are real: a cluster to operate, ordering guaranteed only within a partition (so the partition key must be chosen so that a symbol's events never overtake each other), and latency that is generally higher than an in-memory hop.

The honest summary for a small trading system is that these are ordered by operational burden and you should take the least of them that works. Redis Streams if the working set fits in memory and one machine is enough — which for a daily or minute-bar strategy it comfortably is, and which is why [Part VI](../part-06-live-infrastructure/01-system-architecture.md) put five processes on one box and called it done. RabbitMQ when routing topology is the hard part. Kafka when replayable history is a product requirement rather than a debugging convenience. **The delivery semantics do not differ in the way the marketing implies**: all three are at-least-once in practice, all three require the idempotent consumer the previous section built, and the "exactly-once" configurations that exist are scoped to a single system's own state and do not extend to your broker's order book.

!!! warning "A queue changes what your strategy computes, not just when it computes it"
    The instinct to treat messaging as plumbing is what makes these failures expensive. A slow consumer under three different, entirely reasonable backpressure policies produced three digests, none of them the right one, and the policy with the best operational profile produced the worst strategy. A consumer that died a microsecond before acknowledging produced eight duplicate orders that no broker setting could have prevented. A de-duplication window sized by intuition rather than by the redelivery gap lets duplicates through silently, on the busiest day. None of this is visible in a throughput chart or a latency histogram, and none of it will be caught by a monitoring system watching the queue. Test the decisions, not the transport: run the strategy through the real queue and compare the digest against the in-process run, because that is the only assertion that notices any of it.

!!! abstract "Key takeaways"
    - The same strategy source, `0ffc2f654061`, runs in both a layered and an event-driven design, but the layered one needs **one collaborator to exercise and the event-driven one needs zero** — and a second consumer was added without the strategy changing at all, because adding an observer is not a modification.
    - Pushed through a real Redis stream, tsmom returns **90 decisions and digest `60458f22a9cd`** — identical to Part VI's in-process and cross-thread runs. Three transports, one decision stream, and the strategy never edited.
    - A consumer group's pending list held **100 messages** for a worker that never returned, and `XAUTOCLAIM` moved every one to a live consumer. That is at-least-once delivery: the message is not forgotten, but nothing says the work was not already done.
    - Backpressure is a correctness failure, not a latency one. All three policies produced digests **unequal to `60458f22a9cd`**, with 20, 22 and 16 decisions against 90.
    - The three policies fail differently and each suits a different consumer: unbounded reached a **6,409-message backlog and 4,806 bars of staleness**; bounded capped both at **100 and 99**; conflation achieved **zero and zero** and was the worst of the three here, because conflation is correct for state and destructive for path — and a 252-day momentum sum is nothing but path.
    - A consumer dying between doing the work and acknowledging it produced **23 submissions for 15 orders**. Idempotency in the handler, keyed by client order id, took that to **15 and zero**: the achievable guarantee is *applied at most once*, not *delivered exactly once*.
    - A bounded de-duplication window has a hard minimum: at a window of **4 all 8 duplicates got through**, at a window of **8 — exactly the redelivery gap — none did**. Undersizing it fails silently.
    - Redis Streams cost **33.5 bytes per message**, so 100M messages a day is about **3.35 GB of RAM** retained — enough to make one box viable, and the RAM-and-one-machine limit is precisely what RabbitMQ and Kafka charge operational cost to remove.

## Where this goes next

Part IX has spent four lessons making the platform trustworthy — a history you can interrogate, tests that know what the answer should be, boundaries a program enforces, and now transports that provably do not change the decisions. Every one of those has been about *correctness*, and each has quietly added work: a protocol call per submission, a serialization per message, a stream write per bar. The engine that took about a second in lesson two is doing more than it was.

[Profiling, Refactoring, and Versioning](05-profiling-refactoring-versioning.md) asks what that costs and where, and insists the question be answered with a profiler rather than intuition — which turns out to matter, because the hot path is not where the arithmetic is. It then takes the hardest case in the course, a genuinely gnarly function from [Part VIII](../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md), and refactors it under characterization tests, where one obviously-equivalent step turns out not to be. And it closes Part IX where a platform's life actually gets decided: on what a version number promises, and why semantic versioning is not quite enough when a patch release can move the P&L without changing a single signature.
