# Monitoring, Logging, Alerting

[Lesson three](03-docker-and-cloud-deployment.md) closed on a system that deploys from files and restarts on policy — and cannot be *seen*. That is not a cosmetic gap. A process that crashes and cleanly revives at 3:00, 3:20, and 3:40 has, by breakfast, handled every individual failure and told nobody about the pattern; a deployment inspectable only by SSH is inspected only during incidents, which is the one time you need the history you weren't collecting. This lesson is also a debt coming due: [Part II's logging lesson](../part-02-python/07-logging-and-config.md) built structured JSON logs for research pipelines and promised that its closing thread — alert fatigue, and the way unread warnings become unexplained positions — would be picked up here, where the positions are real.

The lesson builds observability in the order evidence flows: metrics that measure the two different things that can be wrong (the box, and the book), health endpoints that let machines interrogate each other, structured logs threaded by correlation ID so one order's story survives four processes, log aggregation that turns incidents into queries, and then the hard, human part — an alerting policy honest enough to wake someone only when a human decision is actually required, and a dashboard that answers the only question worth a glance: is the book what we think it is? Throughout, the tooling is the standard library plus the services already running — the real systems these patterns scale into (Prometheus, Grafana, Loki) are named as each hand-rolled piece earns its introduction, because the formats and the discipline, not the products, are the lesson.

## System metrics say the box is fine; trading metrics say you are broke

Monitoring a trading system means watching two unrelated failure surfaces. *System* metrics — heartbeats, job latencies, queue depths — say whether the machinery runs. *Trading* metrics — positions, cash, equity, fees, rejects — say whether the machinery is doing the right thing with money. The classic monitoring failure is a wall of green system dashboards over a book quietly going wrong, because every process was healthy while they collectively did the wrong trade. Both families, pulled from the services [lesson two](02-scheduling-and-data-plumbing.md) stood up, and emitted in the text format every metrics scraper on earth understands:

```python
import pandas as pd
import psycopg
import redis

r = redis.Redis(db=15, decode_responses=True)
close = pd.read_parquet("data/part5.parquet")["Close"]
for s in ("SPY", "TLT", "GLD"):           # the mark feed, seeded from cache
    r.set(f"qt:mark:{s}", float(close[s].dropna().iloc[-1]), ex=90)
r.set("qt:hb:strategy", 1, ex=15)

with psycopg.connect("dbname=quant") as conn:
    pos = sorted(conn.execute("SELECT symbol, qty FROM positions").fetchall())
    cash = float(conn.execute(
        "SELECT cash_after FROM fills ORDER BY id DESC LIMIT 1").fetchone()[0])
    fees = float(conn.execute("SELECT sum(fee) FROM fills").fetchone()[0])
    fills = conn.execute("SELECT count(*) FROM fills").fetchone()[0]

equity = cash + sum(int(q) * float(r.get(f"qt:mark:{s}")) for s, q in pos)

metrics = [('qt_heartbeat_ttl_seconds{component="strategy"}',
            r.ttl("qt:hb:strategy"))]
metrics += [(f'qt_position_shares{{symbol="{s}"}}', int(q)) for s, q in pos]
metrics += [("qt_fills_total", fills),
            ("qt_fees_dollars_total", f"{fees:.2f}"),
            ("qt_cash_dollars", f"{cash:.2f}"),
            ("qt_equity_dollars", f"{equity:.2f}")]
for name, val in metrics:                 # the Prometheus exposition format
    print(name, val)
# => qt_heartbeat_ttl_seconds{component="strategy"} 15
#    qt_position_shares{symbol="GLD"} 2737
#    qt_position_shares{symbol="SPY"} 1429
#    qt_position_shares{symbol="TLT"} -10358
#    qt_fills_total 1103
#    qt_fees_dollars_total 50792.98
#    qt_cash_dollars 1685966.32
#    qt_equity_dollars 2522514.08
```

The last line is the one to sit with: equity $2,522,514.08, assembled live from three sources — cash from the last fill's ledger column in Postgres, positions from the `SUM(qty)` view, marks from Redis — and it lands *to the cent* on the number [Part V's tearsheet](../part-05-backtesting-engine/04-performance-metrics-and-reporting.md) pinned and [its replay](../part-05-backtesting-engine/05-trade-logs-and-visualization.md) confirmed. That agreement is the entire philosophy of this lesson: a monitoring number is only trustworthy if it is *derived from the system of record by an independent path*, so that agreement means something and disagreement means investigation, not shrugging. The format is deliberately humble — `name{labels} value`, one per line — because that text *is* the [Prometheus](https://prometheus.io) exposition format: hang these lines off an HTTP endpoint and a real scraper collects, stores, and graphs them with no changes. The naming conventions are load-bearing too: `_total` marks counters that only rise (fees, fills), everything else is a gauge, and the heartbeat is exported as its *TTL* — a number that counts down to a page.

## Liveness, readiness, and staleness are three different questions

Metrics are for humans and scrapers; the components also need to interrogate *each other* — compose healthchecks, load balancers, and [lesson five's watchdog](05-resilience-and-risk-controls.md) all need a machine-readable answer to "how are you?" The trap is treating that as one question. It is three: is the process *alive* (it can answer HTTP at all), is it *ready* (its dependencies work), and is its data *fresh* (the world it sees is current)? One server, two endpoints, and the distinction demonstrated by killing a mark:

```python
import http.server
import json
import threading

import psycopg
import redis
import requests

r = redis.Redis(db=15, decode_responses=True)
r.set("qt:mark:SPY", 611.08, ex=90)

def db_ok():
    try:
        with psycopg.connect("dbname=quant", connect_timeout=1) as c:
            return c.execute("SELECT 1").fetchone()[0] == 1
    except psycopg.Error:
        return False

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            code, body = 200, {"alive": True}
        else:                             # /readyz asks the harder question
            body = {"redis": r.ping(), "postgres": db_ok(),
                    "mark_fresh": r.get("qt:mark:SPY") is not None}
            code = 200 if all(body.values()) else 503
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args):         # quiet — the probes print instead
        pass

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"

for path in ("/healthz", "/readyz"):
    resp = requests.get(base + path)
    print(path, resp.status_code, resp.json())
r.delete("qt:mark:SPY")                   # the mark feed goes quiet...
resp = requests.get(base + "/readyz")
print("/readyz", resp.status_code, resp.json())
resp = requests.get(base + "/healthz")
print("/healthz", resp.status_code, "alive is not the same as ready")
srv.shutdown()
# => /healthz 200 {'alive': True}
#    /readyz 200 {'redis': True, 'postgres': True, 'mark_fresh': True}
#    /readyz 503 {'redis': True, 'postgres': True, 'mark_fresh': False}
#    /healthz 200 alive is not the same as ready
```

The last two lines are the lesson. After the mark expired, `/healthz` still said 200 — the process is fine — while `/readyz` flipped to 503 with the failing check *named* in the body: the process is fine and must not trade, because [lesson two built the marks](02-scheduling-and-data-plumbing.md) to expire rather than go stale, and readiness is where that absence becomes an enforceable signal. The distinction drives different reactions: dead liveness means restart the process ([lesson three's](03-docker-and-cloud-deployment.md) `restart:` policy consumes exactly this), while failed readiness means *stop routing work to it and leave it alone* — restarting a healthy process because its data feed went quiet fixes nothing and destroys the evidence. Staleness deserves its named check because it is the failure mode the other two miss: every process up, every dependency answering, and the whole system confidently pricing a book on numbers from twenty minutes ago.

## One event, one JSON object, one correlation ID

When something does go wrong, the reconstruction has to cross four processes — strategy, risk, execution, broker client — each with its own log stream. Text logs make that a night of `grep` and guesswork; the fix costs two fields. [Part II's `JsonFormatter`](../part-02-python/07-logging-and-config.md) returns verbatim, and the `coid` that [lesson one](01-system-architecture.md) minted into every order becomes the correlation ID stitching the streams together:

```python
import json
import logging
import sys

class JsonFormatter(logging.Formatter):   # Part II's formatter, unchanged
    def format(self, record):
        entry = {"level": record.levelname, "msg": record.getMessage()}
        entry.update(getattr(record, "ctx", {}))
        return json.dumps(entry)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
root = logging.getLogger("qt")
root.addHandler(handler)
root.setLevel(logging.INFO)

COID = "tsmom-live-v1:20250630:SPY:1"

logging.getLogger("qt.strategy").info("target computed", extra={"ctx": {
    "coid": COID, "component": "strategy", "symbol": "SPY", "target": 1429}})
logging.getLogger("qt.risk").info("gauntlet pass", extra={"ctx": {
    "coid": COID, "component": "risk", "checks": 6}})
logging.getLogger("qt.execution").info("submitted", extra={"ctx": {
    "coid": COID, "component": "execution", "qty": 1429}})
logging.getLogger("qt.broker").info("filled", extra={"ctx": {
    "coid": COID, "component": "broker", "px": 611.08}})
# => {"level": "INFO", "msg": "target computed", "coid": "tsmom-live-v1:20250630:SPY:1", "component": "strategy", "symbol": "SPY", "target": 1429}
#    {"level": "INFO", "msg": "gauntlet pass", "coid": "tsmom-live-v1:20250630:SPY:1", "component": "risk", "checks": 6}
#    {"level": "INFO", "msg": "submitted", "coid": "tsmom-live-v1:20250630:SPY:1", "component": "execution", "qty": 1429}
#    {"level": "INFO", "msg": "filled", "coid": "tsmom-live-v1:20250630:SPY:1", "component": "broker", "px": 611.08}
```

Four lines, four components, one `coid` — filter any log store on that value and the order's complete biography assembles itself in causal order, which is exactly the query [lesson six's audit trail](06-secrets-paper-live-compliance.md) will formalize. (Timestamps are omitted here for the same reason Part II omitted them — deterministic output on the page; in production every entry carries a UTC timestamp, and the `coid` is what makes entries *joinable* while timestamps only make them *sortable*, a distinction that matters the first time two processes' clocks disagree.) The levels keep the meanings [Part II assigned](../part-02-python/07-logging-and-config.md), with the stakes raised: INFO narrates the pipeline with numbers attached, WARNING still means a human should look eventually — and the rule that a WARNING firing on every run is a configuration bug stops being hygiene advice here and becomes the foundation the alerting policy two sections down is built on.

## An incident is a query, or it is archaeology

Aggregate those JSON streams — `pd.read_json(lines=True)` at course scale, Loki or an ELK stack when the volume grows, same discipline either way — and post-mortems change genre: from archaeology (scrolling terminals, reconstructing from memory) to *query*. The exam question for the whole apparatus: why did the system buy 23,076 SPY on 2001-01-30 — its largest single fill, a quarter century in the past? Everything needed is in the stores this part built:

```python
import numpy as np
import pandas as pd
import psycopg

spy = (pd.read_parquet("data/part5.parquet")
         .xs("SPY", axis=1, level=1).dropna())
momentum = np.log(spy["Close"]).diff().rolling(252).sum()

with psycopg.connect("dbname=quant") as conn:
    before, fill = conn.execute(
        "SELECT ts, qty, px, cash_after FROM fills "
        "WHERE symbol = 'SPY' ORDER BY id LIMIT 2").fetchall()

print(f"the fill  : {fill[0]} SPY {fill[1]:+,d} @ {float(fill[2]):.4f}")
print(f"the signal: 252d momentum {momentum.loc['2001-01-26']:+.4f} "
      f"on 01-26 -> {momentum.loc['2001-01-29']:+.4f} on 01-29")
print(f"the intent: close the {before[1]:+,d} short opened "
      f"{before[0]}, flip long")
print(f"the money : cash ${float(before[3]):,.2f} -> "
      f"${float(fill[3]):,.2f} after the buy")
print(f"the sequel: momentum {momentum.loc['2001-01-30']:+.4f} "
      f"on 01-30 — it flipped straight back")
# => the fill  : 2001-01-30 SPY +23,076 @ 86.3053
#    the signal: 252d momentum -0.0212 on 01-26 -> +0.0158 on 01-29
#    the intent: close the -12,260 short opened 2001-01-03, flip long
#    the money : cash $1,996,025.94 -> $4,305.89 after the buy
#    the sequel: momentum -0.0022 on 01-30 — it flipped straight back
```

Five lines answer the auditor completely: the fill happened (Postgres), because the year-long momentum sum crossed zero at the January 29th close (recomputed from the frozen bars, not from memory), which under sign-flip rules meant buying back a 12,260-share short *and* establishing the long — 23,076 shares that drained cash to $4,305.89 — and the very next day the signal flipped back, marking this as one of the whipsaws [Part IV's trend lesson](../part-04-strategy-development/01-momentum-and-trend-following.md) priced as the cost of doing momentum business. Note what made the reconstruction possible: no log file even existed for 2001 — the *durable stores were sufficient*, because the fills table is a complete causal record and the signal is reconstructable from frozen bars. That is [Part V's replay doctrine](../part-05-backtesting-engine/05-trade-logs-and-visualization.md) working exactly as promised, and it sets the standard the logs extend to finer grain: anything the stores cannot answer (which retry fired? what did the risk gauntlet reject?) must be a log line, or the next post-mortem is archaeology after all.

## Every alert is a page, a ticket, or a lie

Observability's output is a decision: *does a human need to act, and when?* An alerting policy is that decision written down as data — every alertable condition assigned a severity with an unambiguous contract, plus deduplication so one incident produces one notification instead of a night-long siren:

```python
from collections import defaultdict
from enum import StrEnum, auto

class Sev(StrEnum):
    PAGE = auto()                         # wake a human now
    TICKET = auto()                       # a human, in the morning
    LOG = auto()                          # no human at all

RULES = {"heartbeat_lost": Sev.PAGE, "reject_storm": Sev.PAGE,
         "recon_break": Sev.PAGE, "vendor_retry": Sev.TICKET,
         "disk_80pct": Sev.TICKET, "mark_refreshed": Sev.LOG}

DAY = ["mark_refreshed", "vendor_retry", "heartbeat_lost", "heartbeat_lost",
       "heartbeat_lost", "vendor_retry", "disk_80pct", "heartbeat_lost"]

seen, routed = set(), defaultdict(list)
for event in DAY:
    sev = RULES[event]
    if sev is not Sev.LOG and event in seen:
        continue                          # one incident, one notification
    seen.add(event)
    routed[sev].append(event)

for sev in Sev:
    print(f"{sev:<7} {len(routed[sev])}  {', '.join(routed[sev])}")
n = sum(len(v) for v in routed.values())
print(f"{len(DAY)} raw events -> {n} routed, {len(DAY) - n} deduplicated")
# => page    1  heartbeat_lost
#    ticket  2  vendor_retry, disk_80pct
#    log     1  mark_refreshed
#    8 raw events -> 4 routed, 4 deduplicated
```

Eight raw events became one page, two tickets, and a log line — the four duplicates suppressed because a heartbeat that is lost stays lost until the incident resolves, and paging a human four times about it teaches the human to silence the pager. Real routers add *hysteresis* to the dedup: a flapping check (lost, recovered, lost, recovered) must stay healthy for a cooldown period before it re-arms, or flapping becomes the siren. The third severity is the subtle one. `LOG` is not "unimportant" — it is the honest name for *no human action exists*: the mark expired and the next fetch refreshed it, the system healed, and routing that to a human is how [Part II's alert-fatigue warning](../part-02-python/07-logging-and-config.md) comes true operationally — every unactionable notification trains the reader to skim, and the skimming is indistinguishable from not reading precisely on the night it matters. The title's word "lie" is literal: an alert that pages a human who, on reading it, can do nothing, lied about its own severity — and each lie devalues every future page.

## Page only on what a human must do now

Which conditions deserve `PAGE` is a policy question with a sharp test: *is there an action only a human can take, and does it matter tonight?* The policy, written as the code it should be rather than the wiki page it usually is:

```python
SCENARIOS = [
    ("execution heartbeat lost, market open",    "PAGE",
     "orders may be dangling unwatched"),
    ("5 rejects in 60s",                         "PAGE",
     "our risk model disagrees with the broker"),
    ("recon break: broker 1429, books 1439",     "PAGE",
     "the book is not what we think it is"),
    ("vendor fetch failed, retries remain",      "TICKET",
     "the DAG is doing its job"),
    ("disk at 82% on the log volume",            "TICKET",
     "days of headroom remain"),
    ("strategy crash after a flat close",        "TICKET",
     "no positions, nothing dangles overnight"),
    ("mark TTL expired, next fetch refreshed",   "LOG",
     "the system healed itself"),
]
for what, sev, why in SCENARIOS:
    print(f"{sev:<7} {what:<42} {why}")
# => PAGE    execution heartbeat lost, market open      orders may be dangling unwatched
#    PAGE    5 rejects in 60s                           our risk model disagrees with the broker
#    PAGE    recon break: broker 1429, books 1439       the book is not what we think it is
#    TICKET  vendor fetch failed, retries remain        the DAG is doing its job
#    TICKET  disk at 82% on the log volume              days of headroom remain
#    TICKET  strategy crash after a flat close          no positions, nothing dangles overnight
#    LOG     mark TTL expired, next fetch refreshed     the system healed itself
```

Read the pages for what they share: in each one, *money is exposed and the machinery cannot resolve the exposure itself*. A dead execution engine during the session may have live orders working with nobody watching them; a reject storm means our model of what the broker will accept has diverged from the broker's — [lesson five's gauntlet](05-resilience-and-risk-controls.md) fires it; a reconciliation break means the books are wrong *right now* and every downstream number is contaminated. Now read the tickets for what they share: each has either no exposure (flat book at the crash) or a machine already executing the correct response (retries remaining, headroom remaining). The same event moves categories as context changes — the strategy crash *is* a page during the session with positions on, and the vendor failure *becomes* one when the retries run out, which is exactly the escalation [lesson two's trading gate](02-scheduling-and-data-plumbing.md) performs when it skips the day and pages. The context-dependence is why the policy must live in code where it can branch on market state, not in a wiki where it can only branch on hope.

## The dashboard answers one question: is the book what we think it is

Last, the glance. Metrics feed scrapers, logs feed queries; a human walking past with coffee needs one page that answers the standing question of any live book. Grafana is the grown-up tool, but the essential dashboard is fifty lines of standard library — and building it by hand makes plain that a dashboard is just the monitoring queries, rendered:

```python
import http.server
import re
import threading

import pandas as pd
import psycopg
import redis
import requests

r = redis.Redis(db=15, decode_responses=True)
close = pd.read_parquet("data/part5.parquet")["Close"]
for s in ("SPY", "TLT", "GLD"):
    r.set(f"qt:mark:{s}", float(close[s].dropna().iloc[-1]), ex=90)

def render():
    with psycopg.connect("dbname=quant") as conn:
        pos = conn.execute(
            "SELECT symbol, qty FROM positions ORDER BY symbol").fetchall()
        cash = float(conn.execute(
            "SELECT cash_after FROM fills ORDER BY id DESC LIMIT 1"
        ).fetchone()[0])
    halt = r.get("qt:halt") or "clear"
    equity, rows = cash, ""
    for sym, qty in pos:
        mark = float(r.get(f"qt:mark:{sym}") or "nan")
        mv = int(qty) * mark
        equity += mv
        rows += (f"<tr><td>{sym}</td><td>{int(qty):+,}</td>"
                 f"<td>{mark:.2f}</td><td>{mv:,.0f}</td></tr>")
    return (f"<html><body><h1>tsmom-live-v1 &mdash; halt: {halt}</h1>"
            f"<table>{rows}</table>"
            f"<p>cash ${cash:,.2f} | equity ${equity:,.2f}</p>"
            f"</body></html>")

class Dash(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = render().encode()          # re-queried on every refresh
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Dash)
threading.Thread(target=srv.serve_forever, daemon=True).start()

html = requests.get(f"http://127.0.0.1:{srv.server_address[1]}/").text
srv.shutdown()
print(re.search(r"<h1>(.*?)</h1>", html).group(1).replace("&mdash;", "—"))
for row in re.findall(r"<tr>(.*?)</tr>", html):
    print("  " + " ".join(re.findall(r"<td>(.*?)</td>", row)))
print(re.search(r"<p>(cash.*?)</p>", html).group(1))
# => tsmom-live-v1 — halt: clear
#      GLD +2,737 304.83 834,320
#      SPY +1,429 611.08 873,232
#      TLT -10,358 84.09 -871,004
#    cash $1,685,966.32 | equity $2,522,514.08
```

Everything on the page is pulled fresh on every request from the systems of record — positions from the view that cannot drift from fills, marks from the store where staleness is impossible by construction, and the halt flag, front and center in the title, because the first thing a human should know about a live system is whether it is allowed to trade ([lesson five](05-resilience-and-risk-controls.md) gives that flag its teeth). Nothing is cached, nothing is remembered from the last render, and so the page cannot disagree with the system — it can only reveal it. The equity line closes the loop this lesson opened: $2,522,514.08, the same number the metrics endpoint exported, the same number Part V pinned, now one SSH tunnel away ([lesson three's](03-docker-and-cloud-deployment.md) loopback-only port) at any hour. This page is deliberately ugly. A dashboard's job is to be *correct and current*; the first hour spent styling it is an hour not spent on the reconciliation that makes it worth reading.

!!! warning "Unread warnings become unexplained positions"
    Part II coined this sentence about research pipelines, and live trading is where it collects. Every unactionable page, every duplicate alert, every WARNING that fires on schedule trains the on-call human to read a little less carefully — and that decay is invisible until the night a genuinely novel alert arrives and receives the skim it taught everyone to give. Alert fatigue is not a tooling problem; it is a credibility budget, spent by every notification that wasn't worth its interruption. Guard it the way this lesson does: three severities with contracts, deduplication and hysteresis by default, and a standing rule that anything routed to a human names the action only a human can take.

!!! abstract "Key takeaways"
    - Monitoring watches two failure surfaces: system metrics (a heartbeat exported as its 15s TTL) and trading metrics (the book, fees $50,792.98, cash $1,685,966.32) — and equity $2,522,514.08, assembled live from Postgres cash, the positions view, and Redis marks, lands to the cent on Part V's pinned number because it is derived from the systems of record by an independent path.
    - The hand-rolled `name{labels} value` lines are the real Prometheus exposition format — counters end in `_total` and only rise; everything else is a gauge — so the toy scales into the tool without changing shape.
    - Liveness, readiness, and staleness are different questions with different reactions: after the mark expired, `/healthz` stayed 200 while `/readyz` flipped 503 with the failing check named — restart a dead process, but *stop routing to* an unready one, and never confuse the two.
    - One order's story crosses four processes joined by its `coid`: four JSON log lines, one correlation ID, and any log store reassembles the biography — timestamps make entries sortable, correlation IDs make them joinable.
    - The 2001-01-30 post-mortem needed no log file: fills in Postgres plus signals recomputed from frozen bars answered why the system bought 23,076 SPY — momentum crossed zero on 01-29, closing a −12,260 short, draining cash to $4,305.89, and flipping straight back the next day — replay doctrine as incident response.
    - An alerting policy is data: three severities with contracts (page/ticket/log), dedup so 8 raw events became 4 notifications and one page, hysteresis against flapping — and "log" is the honest name for events where no human action exists.
    - Page only when money is exposed *and* the machinery cannot resolve it alone: dead execution mid-session, reject storms, recon breaks; ticket what has no exposure or an automated response still running — and the dashboard answers the standing question with fresh queries, the halt flag first.

## Where this goes next

The system can now be seen: measured, probed, reconstructed, and honest about what deserves to wake you. Seeing is not surviving. Every capability in this lesson *observes* failure — none of it makes failure safe, and the sharpest confession in the lesson is hiding in the health endpoint: a process can be alive, ready, and completely wrong about the world, because its memory of open orders died with its last crash while the broker kept trading. Restarts are still not safe: [lesson three's](03-docker-and-cloud-deployment.md) supervisor revives a process, but nothing yet tells the revived process what happened while it was dead — which orders are dangling, which fills it missed, whether the book it remembers is the book the broker holds. [Resilience and Risk Controls](05-resilience-and-risk-controls.md) closes that gap: crash recovery that rebuilds state from the stores instead of trusting memory, reconciliation that treats the broker's records as senior to ours, idempotent orders that make retries safe, circuit breakers for the anomalies nobody predicted, and the kill switch that must work precisely when everything else does not.
