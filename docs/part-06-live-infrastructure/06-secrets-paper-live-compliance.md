# Secrets, Paper/Live, and Compliance

[Lesson five](05-resilience-and-risk-controls.md) left a system that survives its own death — and closed by observing that what remains is not machinery but governance. The remaining risks are human and procedural: credentials that authenticate every broker call sitting one careless commit away from the public internet, a promotion decision — pointing the system at real money — that is bigger than any single order and currently governed by nothing, and a trade record that, while accurate, could not *prove* it had never been edited. This closing lesson of Part VI builds that governance, and it pays the part's oldest outstanding note: [Part V's final lesson](../part-05-backtesting-engine/05-trade-logs-and-visualization.md) promised that the trade log would stop being good practice and become the audit trail the whole operation stands on. Here is where it stands on it.

The through-line is a single idea in three costumes: **evidence beats assertion**. A secret is safe not because everyone promises to be careful but because a scanner finds it when they are not, and a canary key screams when the vault is read. A system is ready for money not because its author feels ready but because twenty sessions of paper trading produced measured numbers against written criteria. And a trading record is trustworthy not because nobody would ever edit it but because a hash chain makes editing *detectable* — which converts "trust me" into "check me," the only conversion compliance has ever been about.

## A secret in code is already leaked

The injection pattern is settled doctrine by now — [Part II](../part-02-python/07-logging-and-config.md) put configuration in the environment, [lesson three](03-docker-and-cloud-deployment.md) injected it at container start and kept it out of image layers. What that discipline cannot prevent is the oldest mistake in the industry: a key pasted into a scratch file "temporarily," committed, and published to every clone of the repository forever. The defense is not vigilance; it is a scanner that runs before every commit, hunting the two signatures a credential cannot shed — the naming convention around it, and the entropy inside it:

```python
import math
import re
import tempfile
from pathlib import Path

def entropy(s):                           # bits per character
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

repo = Path(tempfile.mkdtemp())           # a fake repo with a classic mistake
(repo / "strategy.py").write_text(
    'LOOKBACK = 252\nBROKER_URL = "https://paper.broker.example"\n')
(repo / "run.sh").write_text("export QT_ENV=paper\npython -m strategy\n")
(repo / "scratch_nb.py").write_text(
    'api_key = "qk_live_9fK2mQ8xLpR3vN7jW4tY6bZ1"  # temp, remove later\n')

PATTERN = r'(?i)(api_key|secret|token|passw)\w*\s*=\s*["\']([^"\']{16,})["\']'

hits = 0
for path in sorted(repo.iterdir()):
    for line in path.read_text().splitlines():
        if m := re.search(PATTERN, line):
            hits += 1
            print(f"{path.name}: {m.group(1)} = \"{m.group(2)[:12]}...\" "
                  f"entropy {entropy(m.group(2)):.1f} b/char -> LEAK")
print(f"{len(list(repo.iterdir()))} files scanned, {hits} leak found")
# => scratch_nb.py: api_key = "qk_live_9fK2..." entropy 4.9 b/char -> LEAK
#    3 files scanned, 1 leak found
```

The clean files pass — `BROKER_URL` is an assignment of a string, but not one named like a credential, and `QT_ENV=paper` has neither the name nor the randomness. The planted key trips both wires: the keyword `api_key`, and 4.9 bits per character of entropy, which is what 32 characters drawn from a real key-generation process look like and what no human-written identifier ever does. The `# temp, remove later` comment is included because that is *always* the comment. Two operational notes graduate this toy into practice. First, use the real tools — `gitleaks` or `trufflehog` wired into a pre-commit hook and CI — which know hundreds of provider-specific key formats; the twelve lines above are their principle, not their replacement. Second, the response to a committed secret is rotation, *never deletion*: a force-pushed history rewrite does not un-publish anything (clones, forks, and caches all remember), so the moment a key touches version control it is compromised by definition, and the only fix is making the compromised value worthless.

## Scope keys to the minimum that still trades

The scanner defends against publishing a key; scoping defends against what a stolen key can *do*. Broker API keys carry permissions, and the principle is the same least-privilege that keeps [lesson three's containers](03-docker-and-cloud-deployment.md) running as non-root — grant what the system needs, deny what it cannot justify, and make the denial structural:

```python
from datetime import date

SCOPES = {"market data: read": "ON", "orders: place/cancel": "ON",
          "withdrawals": "NEVER", "account admin": "NEVER"}
for scope, state in SCOPES.items():
    print(f"  {scope:<22} {state}")

old, new = "key-2026Q2", "key-2026Q3"
for d, event in [
    (date(2026, 7, 1), f"mint {new}; {old} stays valid — overlap opens"),
    (date(2026, 7, 2), f"deploy {new}; watch it authenticate in the logs"),
    (date(2026, 7, 8), f"revoke {old} — overlap closes, one key again"),
]:
    print(f"{d}  {event}")

CANARY = "key-canary"                     # stored beside the real keys,
auth_log = ["key-2026Q3", "key-2026Q3",   # ...wired to nothing, used never
            "key-canary", "key-2026Q3"]
for k in auth_log:
    if k == CANARY:
        print(f"auth by {k!r} -> someone read the vault -> "
              f"PAGE, revoke everything, rotate now")
# =>   market data: read      ON
#      orders: place/cancel   ON
#      withdrawals            NEVER
#      account admin          NEVER
#    2026-07-01  mint key-2026Q3; key-2026Q2 stays valid — overlap opens
#    2026-07-02  deploy key-2026Q3; watch it authenticate in the logs
#    2026-07-08  revoke key-2026Q2 — overlap closes, one key again
#    auth by 'key-canary' -> someone read the vault -> PAGE, revoke everything, rotate now
```

The scope matrix's two `NEVER`s are the entire threat model in four words: a thief holding a data-and-trading key can lose you money, painfully but boundedly, inside [lesson five's risk limits](05-resilience-and-risk-controls.md); a thief holding a withdrawal-enabled key can simply *take* the money, no trading required — so withdrawal permission does not belong on any key a machine holds, ever, at any convenience. The rotation timeline shows why rotation is painless when designed and skipped when not: the overlap window means the new key is deployed and *observed working* before the old one dies, so rotation never risks an outage — remove the overlap and every rotation is a small deployment gamble, which is precisely how quarterly rotation becomes annual becomes never. And the canary is the cheapest intrusion detector ever built: a credential that exists only to be stolen, stored beside the real ones, wired to nothing. Legitimate code cannot touch it, so the single authentication attempt in the log is a fact with only one explanation — the vault has been read, and everything beside the canary must be presumed taken.

## Paper trading validates the plumbing, not the profits

With secrets governed, the question becomes promotion — and the industry's answer is a paper phase: run the full system against the broker's simulated endpoint, real data in, real decisions out, fake fills back. What paper trading is *for* is the part this course can now state precisely, because [Part V measured it](../part-05-backtesting-engine/05-trade-logs-and-visualization.md). A paper fill is an optimistic fill — no queue, no spread paid, no impact — which makes the paper account a live rerun of Part V's frictionless experiment, and that experiment's price tag is already pinned:

```python
import psycopg

FREE_RUN = 2_591_150                      # Part V L5: the frictionless run
COSTED = 2_522_514                        # same engine, spread + commission

with psycopg.connect("dbname=quant") as conn:
    fees = float(conn.execute("SELECT sum(fee) FROM fills").fetchone()[0])

gap = FREE_RUN - COSTED
print(f"fees the costed run paid   : ${fees:,.2f}")
print(f"final equity, frictionless : ${FREE_RUN:,}")
print(f"final equity, real costs   : ${COSTED:,}")
print(f"paper's flattery           : ${gap:,} over the sample "
      f"({gap / fees:.2f}x the fees — compounding, not arithmetic)")
# => fees the costed run paid   : $50,792.98
#    final equity, frictionless : $2,591,150
#    final equity, real costs   : $2,522,514
#    paper's flattery           : $68,636 over the sample (1.35x the fees — compounding, not arithmetic)
```

Read the flattery line the way Part V taught: the costed run paid $50,793 in fees, but the frictionless account finishes $68,636 ahead, because money not spent in 2003 compounded for twenty-two years — paper's optimism is not the fee bill, it is the fee bill *with interest*. So a paper account that beats its backtest is not good news; it is the frictionless gap arriving on schedule, and treating it as alpha is self-deception with a login screen. What paper *does* validate — and validates irreplaceably — is everything this part built: that [the scheduler](02-scheduling-and-data-plumbing.md) fires on real half-days, that fills flow through [the state machine](05-resilience-and-risk-controls.md) without quarantines, that reconciliation passes against a statement the system did not write, that [alerts route](04-monitoring-logging-alerting.md) and humans respond, that recovery works when a process is killed mid-session on purpose. Paper trading is the dress rehearsal of the *theater*, not the play: it proves the machinery under real time pressure with fake money, and its economics are read only for *tracking* — does the paper account do what the backtest predicts, gap included — never for profit.

## Go-live is a measurement, not a feeling

Which raises the gate itself. The moment a system is allowed to touch real money should be governed the way every claim in this course is governed — by written criteria and measured numbers, because the author's confidence at the end of a paper run is the least reliable instrument in the building. Twenty real XNYS sessions, tallies collected by [lesson four's monitoring](04-monitoring-logging-alerting.md), criteria written before the run began:

```python
import exchange_calendars as xcals

xnys = xcals.get_calendar("XNYS", start="2026-01-01", end="2026-12-31")
paper = xnys.sessions_in_range("2026-06-29", "2026-07-27")

# tallied from the paper run: job_runs, the alert log, the recon reports
M = {"paper sessions": len(paper), "uptime": 0.9991, "recon breaks": 0,
     "own orders risk-rejected": 1, "median page response (min)": 22,
     "tracking error vs backtest (bps/day)": 11}

CRITERIA = [
    ("paper sessions", ">= 20", lambda v: v >= 20),
    ("uptime", ">= 99.5%", lambda v: v >= 0.995),
    ("recon breaks", "== 0", lambda v: v == 0),
    ("own orders risk-rejected", "<= 3", lambda v: v <= 3),
    ("median page response (min)", "<= 15", lambda v: v <= 15),
    ("tracking error vs backtest (bps/day)", "<= 20", lambda v: v <= 20),
]
failed = []
for name, rule, ok in CRITERIA:
    verdict = "pass" if ok(M[name]) else "FAIL"
    if verdict == "FAIL":
        failed.append(name)
    print(f"{verdict}  {name:<38} {M[name]}  (need {rule})")
print("verdict:", "PROMOTE" if not failed else
      f"HOLD — fix {failed[0]!r}, then restart the 20-session clock")
# => pass  paper sessions                         20  (need >= 20)
#    pass  uptime                                 0.9991  (need >= 99.5%)
#    pass  recon breaks                           0  (need == 0)
#    pass  own orders risk-rejected               1  (need <= 3)
#    FAIL  median page response (min)             22  (need <= 15)
#    pass  tracking error vs backtest (bps/day)   11  (need <= 20)
#    verdict: HOLD — fix 'median page response (min)', then restart the 20-session clock
```

The system just failed its promotion, and the failure is the scorecard working. Five criteria passed — the machinery ran, reconciled, tracked its backtest inside 11 basis points a day, and even the one risk rejection is within tolerance (a gauntlet that never fires during a twenty-session shakeout is itself suspicious). What failed is the *human*: pages took a median of 22 minutes to acknowledge, and a system whose alerts are answered in 22 minutes is a system that can be wrong for 22 minutes with real money on. Three properties make the gate a gate. The criteria predate the run — written afterward, they would be fitted to whatever happened, [the same sin](../part-04-strategy-development/08-validation-and-overfitting.md) Part IV spent a lesson prosecuting in backtests. The verdict names its blocker, so "not yet" comes with a work order — fix the on-call rotation, not the code. And a HOLD restarts the clock, because the fix changes the system being evaluated, and evidence about the old system does not transfer to the new one.

## The checklist is the interview your incident will conduct

A PROMOTE verdict earns the go-live checklist — the final inventory conducted the morning real money starts. Its wiki-page form is famous and famously skipped; its honest form is a program, because every item worth checking is a fact some store can be asked about, and [lesson five's doctrine](05-resilience-and-risk-controls.md) — never accept a self-report where a measurement exists — applies to the operators too:

```python
import psycopg
import redis

r = redis.Redis(db=15, decode_responses=True)
r.delete("qt:halt")                       # lesson five's drill, cleared
r.set("qt:mark:SPY", 611.08, ex=90)

with psycopg.connect("dbname=quant") as conn:
    conn.execute("DELETE FROM job_runs")  # seed today's operational record
    conn.execute(
        "INSERT INTO job_runs (job, run_date, status, detail) VALUES "
        "('ingest', '2026-07-27', 'ok', '4 symbols landed'), "
        "('recon', '2026-07-27', 'ok', '0 breaks'), "
        "('kill_drill', '2026-07-27', 'ok', 'freeze + clear rehearsed')")
    tables = {t for (t,) in conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
    fills = conn.execute("SELECT count(*) FROM fills").fetchone()[0]
    ok_jobs = {j for (j,) in conn.execute(
        "SELECT job FROM job_runs WHERE status = 'ok'")}

checks = [
    ("durable stores present", {"fills", "orders", "job_runs"} <= tables),
    ("ledger populated", fills > 0),
    ("last ingest landed and validated", "ingest" in ok_jobs),
    ("last reconciliation clean", "recon" in ok_jobs),
    ("kill-switch drill on record", "kill_drill" in ok_jobs),
    ("halt flag clear", r.get("qt:halt") is None),
    ("live mark flowing", r.get("qt:mark:SPY") is not None),
]
for name, ok in checks:
    print(f"{'pass' if ok else 'FAIL'}  {name}")
print("sign-off:", "GO" if all(ok for _, ok in checks) else "NO-GO")
# => pass  durable stores present
#    pass  ledger populated
#    pass  last ingest landed and validated
#    pass  last reconciliation clean
#    pass  kill-switch drill on record
#    pass  halt flag clear
#    pass  live mark flowing
#    sign-off: GO
```

Every check queries a record, not a memory. The kill-switch drill counts only because [lesson five's rehearsal](05-resilience-and-risk-controls.md) wrote a `job_runs` row — an undrilled kill switch is a hypothesis, and the checklist refuses hypotheses. The reconciliation check reads the recon job's own testimony; the mark check catches the go-live morning where the data vendor, of course, is having an outage. The section title is the design principle: every item is an answer to a question the post-incident review will ask — *was the kill switch tested? when did reconciliation last pass? was the halt flag clear at open?* — and the checklist is simply that interview, conducted in advance, while every answer can still be fixed. Items that pass silently today are testimony tomorrow: the sign-off line, logged, is itself part of the audit trail the next section makes tamper-evident.

## Append-only, or it did not happen

Everything so far governs the system going forward; compliance also governs the *past* — specifically, the guarantee that the record of what happened cannot be quietly rewritten. [Lesson two's](02-scheduling-and-data-plumbing.md) fills table is accurate, but accuracy is not integrity: any process with write access could `UPDATE` a quantity and leave no trace. The fix is centuries older than computers — a ledger where each entry seals the one before it — implemented with an HMAC chain: every audit row's hash covers its own contents *plus the previous row's hash*, so editing any historical row breaks every seal after it:

```python
import hashlib
import hmac

import pandas as pd
import psycopg

KEY = b"course-demo-key"                  # in production: from the vault

def link(prev, ts, symbol, qty, px, fee):
    msg = f"{prev}|{ts}|{symbol}|{qty}|{float(px):.6f}|{float(fee):.2f}"
    return hmac.new(KEY, msg.encode(), hashlib.sha256).hexdigest()

trades = pd.read_parquet("data/part5trades.parquet")

with psycopg.connect("dbname=quant") as conn:
    conn.execute("DROP TABLE IF EXISTS audit")
    conn.execute("CREATE TABLE audit (id BIGSERIAL PRIMARY KEY, ts DATE, "
                 "symbol TEXT, qty INTEGER, px NUMERIC(18,10), "
                 "fee NUMERIC(12,2), prev_hash TEXT, hash TEXT)")
    prev = "genesis"
    with conn.cursor() as cur:
        for t in trades.itertuples(index=False):
            h = link(prev, t.ts.date(), t.symbol, t.qty, t.px, t.fee)
            cur.execute("INSERT INTO audit (ts, symbol, qty, px, fee, "
                        "prev_hash, hash) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (t.ts.date(), t.symbol, t.qty, t.px, t.fee, prev, h))
            prev = h
print(f"chained {len(trades)} fills, head {prev[:16]}...")

def verify(rows):
    prev = "genesis"
    for i, (ts, symbol, qty, px, fee, stored) in enumerate(rows, 1):
        if link(prev, ts, symbol, qty, px, fee) != stored:
            return f"chain BREAKS at row {i} ({ts} {symbol})"
        prev = stored
    return "chain intact"

with psycopg.connect("dbname=quant") as conn:
    q = "SELECT ts, symbol, qty, px, fee, hash FROM audit ORDER BY id"
    print("fresh table :", verify(conn.execute(q).fetchall()))
    conn.execute("UPDATE audit SET qty = qty + 1 WHERE id = 500")
    print("after tamper:", verify(conn.execute(q).fetchall()))
    conn.rollback()
# => chained 1103 fills, head cceeb2472d873704...
#    fresh table : chain intact
#    after tamper: chain BREAKS at row 500 (2014-04-01 SPY)
```

One share added to one fill from 2014, and verification names the exact row — not because row 500 looks wrong (it looks fine) but because its stored hash no longer matches its edited contents, and *recomputing* it would break row 501, and so on to the head: a forger must re-seal every subsequent row, and without the HMAC key ([vault-kept](#a-secret-in-code-is-already-leaked), naturally) cannot re-seal any. That head hash — sixteen hex characters — is the entire table's integrity compressed to one line; publish it daily to somewhere outside the attacker's reach (an email to yourself is a legitimate poor-man's notary) and even a key thief cannot rewrite yesterday. The doctrine completing [Part II's point-in-time principle](../part-02-python/05-sql-and-data-storage.md) and Part V's log-first discipline: the fills table stays append-only by policy, the audit chain makes violations of the policy *detectable*, and retention is measured in years — regulators ask for five to seven, and your own future self, debugging a strategy's decade-old behavior, asks for all of them.

## "Why did the system do that?" has a deadline

The part ends with its final exam. The question every compliance apparatus exists to answer arrives — from a regulator, a counterparty, or you in five years — pointing at the largest single fill in the book's history: *why did the system buy 23,076 SPY on January 30th, 2001?* The answer must assemble authority, trigger, execution, accounting, and integrity, and it must come from records, not recollection:

```python
import numpy as np
import pandas as pd
import psycopg

spy = (pd.read_parquet("data/part5.parquet")
         .xs("SPY", axis=1, level=1).dropna())
momentum = np.log(spy["Close"]).diff().rolling(252).sum()

with psycopg.connect("dbname=quant") as conn:
    fill = conn.execute(
        "SELECT ts, qty, px, fee, cash_after FROM fills "
        "WHERE ts = '2001-01-30' AND symbol = 'SPY'").fetchone()
    chain = conn.execute(
        "SELECT id, prev_hash, hash FROM audit "
        "WHERE ts = '2001-01-30' AND symbol = 'SPY'").fetchone()

print("DECISION RECORD — SPY +23,076 on 2001-01-30")
print("  authority : run tsmom-nextopen-v1, sign-flip rule, 252d lookback")
print(f"  trigger   : momentum {momentum.loc['2001-01-26']:+.4f} -> "
      f"{momentum.loc['2001-01-29']:+.4f} at the 01-29 close")
print(f"  execution : {fill[1]:+,d} @ {float(fill[2]):.4f}, "
      f"fee ${float(fill[3]):.2f}")
print(f"  accounting: cash after ${float(fill[4]):,.2f}")
print(f"  integrity : audit row {chain[0]}, "
      f"link {chain[1][:10]}.. -> {chain[2][:10]}..")
print("  replay    : data/part5.parquet + data/part5trades.parquet")
# => DECISION RECORD — SPY +23,076 on 2001-01-30
#      authority : run tsmom-nextopen-v1, sign-flip rule, 252d lookback
#      trigger   : momentum -0.0212 -> +0.0158 at the 01-29 close
#      execution : +23,076 @ 86.3053, fee $139.41
#      accounting: cash after $4,305.89
#      integrity : audit row 2, link 318d0bfff6.. -> b1d84c74d8..
#      replay    : data/part5.parquet + data/part5trades.parquet
```

Six lines, and read what each one stands on: the authority is the run's recorded configuration ([lesson three](03-docker-and-cloud-deployment.md) froze config into restartable, nameable state precisely so this line could be written); the trigger is *recomputed from frozen bars*, not quoted from a log — the strongest possible answer, since it re-derives the decision rather than remembering it; the execution and accounting come from the fills ledger; the integrity line places the fill in the tamper-evident chain; and the replay line names the two files from which every number above can be regenerated by anyone, indefinitely. The section title's "deadline" is literal — regulators expect answers in days, not quarters — but the deeper deadline is architectural: this record was *cheap* tonight because every lesson in this part contributed a line, and it would be impossible to construct after the fact for a system that had not. An unexplainable trade is not a record-keeping embarrassment; in any serious review it is indistinguishable from an unauthorized one, which is the standard — [lesson two's](02-scheduling-and-data-plumbing.md) durable stores, [lesson four's](04-monitoring-logging-alerting.md) correlation IDs, this lesson's chain — the whole part has been quietly building toward.

!!! warning "An unexplained trade is indistinguishable from an unauthorized one"
    To a compliance review, a counterparty dispute, or your own audit of a strategy gone wrong, a trade you cannot reconstruct — what triggered it, what checked it, what configuration governed it — carries exactly the evidentiary weight of a trade your system was never supposed to make: none. The apparatus of this lesson is not bureaucracy layered onto engineering; it is the engineering that makes every other claim checkable. Secrets management proves who could act, the promotion gate proves the system earned the right to act, and the audit chain proves the record of its actions is the record. Build it before the first real order, because the first question about that order may arrive years after the person who could answer it from memory has moved on.

!!! abstract "Key takeaways"
    - A secret in code is already leaked: the scanner's two tripwires — credential-shaped names and 4.9 bits/char of entropy — caught the planted key in 3 files, the real tools are gitleaks/trufflehog in pre-commit and CI, and the only response to a committed key is rotation, never history-rewriting.
    - Keys are scoped to the minimum that still trades — withdrawals NEVER, on any machine-held key — rotated through an overlap window so rotation never gambles an outage, and guarded by a canary credential whose single use means the vault has been read.
    - Paper fills are frictionless fills: Part V priced that optimism at $68,636 over the sample — 1.35× the $50,792.98 fee bill, the difference being lost compounding — so paper economics are read for tracking, never for profit; what paper genuinely validates is the plumbing under real time pressure.
    - Go-live is measured against criteria written before the run: five of six passed over exactly 20 XNYS sessions, but a 22-minute median page response failed the gate — HOLD, with the blocker named, the fix procedural, and the 20-session clock restarted.
    - The go-live checklist is a program that queries records, not memories — stores present, ledger populated, recon clean, kill-switch drill *on record*, halt clear, marks flowing — because every item is an answer the post-incident interview will demand.
    - The audit chain seals 1,103 fills under an HMAC per row covering the previous row's hash (head `cceeb2472d873704...`): a one-share edit to a 2014 row breaks the chain at exactly row 500, and re-sealing requires the vault-kept key.
    - "Why did the system do that?" now closes in six lines — authority, trigger recomputed from frozen bars, execution, accounting, integrity link, replay files — cheap tonight only because every lesson of this part contributed one.

## Where this goes next

Part VI is complete, and the four notes [Part V's handoff](../part-05-backtesting-engine/05-trade-logs-and-visualization.md) signed are paid in full: the data handler became a scheduled, calendar-aware, validating pipeline; the simulated broker became a client with acknowledgments, idempotent retries, and failure modes rehearsed on purpose; the reconciliation invariant became a nightly diff against the broker's own statement, with the broker senior; and the trade log became a hash-chained audit trail that can answer for any decision, years later, from files anyone can replay. The system runs overnight with no human watching, and — the part's real graduation — what happens when it breaks is a design, not a discovery. The weakest assumption left standing is no longer infrastructure at all. It is the strategy: a sign of a 252-day sum, chosen in Part IV for its honesty rather than its edge, now running on machinery that deserves better signals. [Part VII — Machine Learning](../part-07-machine-learning/index.md) goes hunting for them, and it will need this part sooner than it expects: models decay in production the way processes crash, and [its closing lesson](../part-07-machine-learning/05-production-ml.md) monitors them with exactly the apparatus lesson four built.
