# Docker and Cloud Deployment

[Lesson two](02-scheduling-and-data-plumbing.md) left the system running — scheduled, calendar-aware, its state split correctly between Redis and PostgreSQL — on a machine that is itself the system's biggest unstated risk. That box was configured by hand: packages installed one `pip` at a time, two services set up interactively, environment variables exported from a shell history nobody saved. It works, which is the trap. A hand-built machine is a machine whose configuration exists nowhere but in the machine, and the day it dies — disk failure, botched upgrade, cloud provider retiring the instance type — the system does not have an outage, it has an archaeology project. This course has one doctrine it applies to everything, from [Part III's first frozen cache](../part-03-statistics/01-probability-and-random-variables.md) to Part V's replayable trade log: anything you depend on gets frozen into a versioned artifact and rebuilt from that artifact on demand. This lesson applies the doctrine to the machine itself. A Docker image is to your environment exactly what `data/part5.parquet` is to your data.

One honesty note before the tooling: Docker is the single tool in this course whose console output is shown rather than pinned — the two transcripts below are labeled *illustrative* and reproduce the shape of what the commands print, not a captured run. Every Python block in this lesson remains pinned from a real execution, and they carry the lesson's actual arguments: what an image freezes, what restart policies really decide, how configuration reaches a container, what the network exposes, and why the speed of light — the only latency bound nobody can negotiate — says your deployment region matters less than you think for a daily-rebalancing system.

## The box you built by hand cannot be rebuilt

Start by measuring the thing that needs freezing. Everything below is an input to every pinned number in this part — and every line of it can drift when a machine is rebuilt by hand and memory:

```python
import importlib.metadata as md
import platform
import sys

print("python :", sys.version.split()[0], platform.machine())
print("kernel :", platform.system(), platform.release())
for pkg in ["pandas", "numpy", "apscheduler", "redis", "psycopg",
            "exchange-calendars"]:
    print(f"{pkg:<18}: {md.version(pkg)}")
# => python : 3.12.3 x86_64
#    kernel : Linux 7.0.0-28-generic
#    pandas            : 3.0.5
#    numpy             : 2.5.1
#    apscheduler       : 3.11.3
#    redis             : 8.1.0
#    psycopg           : 3.3.4
#    exchange-calendars: 4.13.2
```

This printout is the *drift surface*: the set of versions that must all agree between the machine where the strategy was validated and the machine where it trades. [Part V certified](../part-05-backtesting-engine/04-performance-metrics-and-reporting.md) a strategy, and [lesson one](01-system-architecture.md) proved the certificate transfers by running identical code in both harnesses — but that proof quietly assumed *identical interpreters and identical libraries*, and a hand-rebuilt box honors that assumption only by luck. Rebuild next year and `pip install pandas` fetches a different pandas; a subtly different `rolling` implementation is a subtly different strategy, and no reconciliation report will name the culprit. The fix is not a better runbook — runbooks are hand-configuration with better handwriting. The fix is making the environment an artifact: built once, hashed, tagged, and started identically every time, which is all a container image is.

## An image is a frozen environment

A `Dockerfile` is the recipe; the image it builds is the frozen result — a stack of immutable layers holding the OS userland, the interpreter, the packages, and the code, tagged and content-addressed. The recipe for the strategy container, with its three deliberate decisions in the comments:

```dockerfile
FROM python:3.12-slim                # small, current, patched

WORKDIR /app
RUN useradd --create-home runner     # the process will not run as root

COPY requirements.txt .              # dependencies first: this layer is
RUN pip install --no-cache-dir -r requirements.txt   # cached until reqs change

COPY strategy/ ./strategy/           # code last — it changes daily
USER runner                          # no root, and no secrets in any layer
CMD ["python", "-m", "strategy"]
```

The layer *order* is the part beginners miss: each instruction is a cached layer, invalidated only when its inputs change, so dependencies go before code — a one-line strategy edit rebuilds in seconds instead of re-resolving every package. The `USER runner` line is the security floor (a compromised strategy process should not own the box), and the "no secrets in any layer" comment is a rule with teeth that [lesson six](06-secrets-paper-live-compliance.md) will enforce: layers are immutable and shipped everywhere the image goes, so a credential COPY'd in during a build is published, permanently, to anyone who can pull the image. Building and tagging looks like this:

```text
# illustrative — requires Docker
$ docker build -t qt-strategy:2026-07-31 .
 => [1/5] FROM docker.io/library/python:3.12-slim
 => [2/5] WORKDIR /app
 => [3/5] RUN useradd --create-home runner
 => [4/5] COPY requirements.txt .
 => [5/5] RUN pip install --no-cache-dir -r requirements.txt
 => exporting to image
 => naming to docker.io/library/qt-strategy:2026-07-31
```

The tag `qt-strategy:2026-07-31` is the environment's version string — the analogue of a cache's filename. The rule that keeps it meaningful: tags are dates or version numbers, never `latest`, because "deploy `latest`" is a sentence with no fixed referent, and the whole point of the exercise is that the image running in production can be named, pulled, and rerun three years from now — the same standard the frozen parquet files already meet.

## Compose is the architecture diagram, executable

[Lesson one's](01-system-architecture.md) pipeline had five boxes; docker-compose makes the boxes real. One file declares every service, how they find each other, what survives a restart, and what order they are allowed to wake up in:

```yaml
services:
  strategy:
    image: qt-strategy:2026-07-31   # the tag is the frozen environment
    env_file: [.env.paper]
    depends_on:
      redis: {condition: service_healthy}
      postgres: {condition: service_healthy}
    restart: on-failure
  risk:
    image: qt-risk:2026-07-31
    env_file: [.env.paper]
    depends_on:
      redis: {condition: service_healthy}
    restart: on-failure
  execution:
    image: qt-execution:2026-07-31
    env_file: [.env.paper]
    depends_on:
      redis: {condition: service_healthy}
      postgres: {condition: service_healthy}
    restart: on-failure
  redis:
    image: redis:7.2
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
    restart: always                 # hot state is rebuildable — revive freely
  postgres:
    image: postgres:17
    environment: {POSTGRES_DB: quant}
    volumes: [pgdata:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
    restart: always
  dashboard:
    image: qt-dashboard:2026-07-31
    ports: ["127.0.0.1:8080:8080"]  # loopback only — reach it by SSH tunnel
    restart: on-failure

volumes:
  pgdata:                           # the one thing that must outlive any box
```

Read it as the architecture diagram it is. The three pipeline processes depend on the stores *with a condition*: `service_healthy` means the strategy does not start until Redis answers `PING` and Postgres answers `pg_isready` — lesson two's "no job runs until the job it believes in has proven itself," enforced at boot. The named volume `pgdata` is the durable/hot distinction drawn in infrastructure: containers are disposable and their filesystems die with them, so the one directory holding fills and orders lives in a volume that outlives every container, while Redis gets no volume at all because [lesson two](02-scheduling-and-data-plumbing.md) designed its contents to be affordable to lose. And the dashboard's port binding names the entire external surface of the system — one port, loopback only. Starting the stack is one command:

```text
# illustrative — requires Docker
$ docker compose up -d
 ✔ Container qt-postgres   Healthy
 ✔ Container qt-redis      Healthy
 ✔ Container qt-strategy   Started
 ✔ Container qt-risk       Started
 ✔ Container qt-execution  Started
 ✔ Container qt-dashboard  Started
```

## Restart is a policy, not heroics

Every service above declares a `restart:` policy, and the word is doing more work than it looks. A restart policy is a standing decision, made calmly at design time, about what should happen at 3am when a process dies — the exact situation [lesson one's warning](01-system-architecture.md) said gets discovered rather than designed. Here is the machinery a supervisor actually runs, with the child process's death codes scripted so both timelines pin:

```python
import random
import subprocess
import sys

CHILD = "import sys; sys.exit(int(sys.argv[1]))"
MAX_RESTARTS = 4

def supervise(name, exit_codes):
    random.seed(7)                        # reproducible jitter
    print(f"{name}:")
    for attempt, code in enumerate(exit_codes, 1):
        r = subprocess.run([sys.executable, "-c", CHILD, str(code)])
        if r.returncode == 0:
            print(f"  attempt {attempt}: exit 0 -> up")
            return
        if attempt >= MAX_RESTARTS:
            print(f"  attempt {attempt}: exit {r.returncode} "
                  f"-> crash loop, stop and page")
            return
        delay = min(2 ** attempt, 30) * random.uniform(0.5, 1.0)
        print(f"  attempt {attempt}: exit {r.returncode} "
              f"-> restart in {delay:.1f}s")   # timeline only — no sleep here

supervise("transient bug ", [13, 13, 0])
supervise("permanent bug ", [13, 13, 13, 13, 13])
# => transient bug :
#      attempt 1: exit 13 -> restart in 1.3s
#      attempt 2: exit 13 -> restart in 2.3s
#      attempt 3: exit 0 -> up
#    permanent bug :
#      attempt 1: exit 13 -> restart in 1.3s
#      attempt 2: exit 13 -> restart in 2.3s
#      attempt 3: exit 13 -> restart in 6.6s
#      attempt 4: exit 13 -> crash loop, stop and page
```

The transient timeline is why restart policies exist: a wedged connection or an OOM kill clears on retry, backoff spacing the attempts so a struggling service is not hammered — the same [exponential-backoff-with-jitter discipline](../part-02-python/04-async-and-apis.md) Part II applied to vendors, now applied to ourselves. The permanent timeline is why the *cutoff* exists: a bug that survives four restarts will survive four hundred, and a supervisor that keeps trying converts one crash into an infinite loop of crashes — each one, for an execution process, potentially re-running startup logic that touches orders. That is the map to the compose policies above: the stores get `always` (their startup is idempotent by construction, and hot state that died is supposed to be rebuilt), while the pipeline processes get `on-failure` — and the execution engine in particular must *never* be blind-revived past its cutoff, because a process that submits orders may only rejoin the market after [lesson five's crash recovery](05-resilience-and-risk-controls.md) has reconciled what happened while it was dead. Docker enforces exactly this shape with `restart: on-failure` plus backoff; what Docker cannot know is what the process must *do* on the way back up, and that gap is lesson five's whole subject.

## Configuration is injected, never baked

The image is frozen; the deployment is not. The same `qt-strategy:2026-07-31` image must trade paper against the paper broker today and, after [lesson six's promotion gate](06-secrets-paper-live-compliance.md), live against the real one — with different endpoints, different risk limits, different database DSNs. If those differences were baked into the image there would be two images, and the one that was tested would not be the one that trades. They are injected instead, from the environment, exactly as [Part II's config lesson](../part-02-python/07-logging-and-config.md) prescribed:

```python
import os
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Settings:
    env: str
    broker_url: str
    max_qty: int
    db_dsn: str

    @classmethod
    def from_env(cls, e=os.environ):
        return cls(
            env=e.get("QT_ENV", "paper"),  # paper unless told otherwise
            broker_url=e.get("QT_BROKER_URL",
                             "https://paper.broker.example"),
            max_qty=int(e.get("QT_MAX_QTY", "25000")),
            db_dsn=e.get("QT_DB_DSN", "dbname=quant"))

paper = Settings.from_env({})             # an empty env -> safe defaults
live = Settings.from_env({"QT_ENV": "live",
                          "QT_BROKER_URL": "https://api.broker.example",
                          "QT_MAX_QTY": "10000"})
print(paper)
print(live)
# => Settings(env='paper', broker_url='https://paper.broker.example', max_qty=25000, db_dsn='dbname=quant')
#    Settings(env='live', broker_url='https://api.broker.example', max_qty=10000, db_dsn='dbname=quant')
```

Two design choices carry the block. The defaults are the *paper* profile — an empty environment produces a configuration that cannot touch real money, so the failure mode of forgotten configuration is embarrassment, not bankruptcy. And the result is a frozen dataclass: configuration is read once, at startup, into an immutable object, so no code path can discover mid-session that someone edited a limit — a changed configuration is a restart, which makes "what config was it running?" answerable, a question [lesson six's audit trail](06-secrets-paper-live-compliance.md) will demand answered. The values arrive through compose's `env_file:` — one small file per deployment profile:

```text
# .env.paper — injected at start, never COPY'd into any image
QT_ENV=paper
QT_BROKER_URL=https://paper.broker.example
QT_MAX_QTY=25000
QT_DB_DSN=host=postgres dbname=quant
```

Swapping `.env.paper` for `.env.live` in one line of the compose file is the *entire* difference between the two deployments — same images, same topology, same code, which is precisely what makes the paper run evidence about the live one. Note what the file does not contain: no API keys, no passwords. Connection to Postgres rides on trust inside a private network for now; the honest handling of real credentials is [lesson six's](06-secrets-paper-live-compliance.md) opening act.

## Nothing listens on the internet but the dashboard

Containers on a compose network reach each other by service name — `host=postgres` in the DSN above resolves inside the private network Docker creates — and none of it is reachable from outside unless a port is deliberately published. That boundary is checkable from this very machine:

```python
import socket

def probe(host, port, timeout=1.0):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return "open"
    except OSError as e:
        return type(e).__name__

for host, port, label in [
    ("127.0.0.1", 6379, "redis, from this box"),
    ("127.0.0.1", 5432, "postgres, from this box"),
    ("192.0.2.7", 6379, "redis, from the internet"),
]:
    print(f"{label:<25} -> {probe(host, port)}")
# => redis, from this box      -> open
#    postgres, from this box   -> open
#    redis, from the internet  -> TimeoutError
```

The stores answer on loopback — the processes that need them are on the box — and the probe against an outside address (192.0.2.7, a reserved test network standing in for "anywhere else") times out, which is what a firewalled port looks like from the attacker's side: not a refusal, silence. Keeping it that way on a real VPS takes four lines, plus the tunnel that replaces every port you did not open:

```text
$ sudo ufw default deny incoming
$ sudo ufw allow OpenSSH
$ sudo ufw enable
# the dashboard is never published to the internet — tunnel to it:
$ ssh -L 8080:127.0.0.1:8080 trader@your-vps
```

Default-deny is the whole philosophy: the system's attack surface is SSH and nothing else, the dashboard rides the tunnel, and Redis and Postgres — which between them hold the marks and the money — are never network-reachable from anywhere but the box itself. An unfirewalled Redis on a public IP is compromised in *minutes* by automated scanners; this is among the most common self-inflicted wounds in amateur trading infrastructure, and the defense is these four lines applied before the first deploy, not after the first incident.

## Rent the box or rent the ops

Where should this stack live? For a system of this size the choice is one rented VPS running compose, or a managed cloud where the stores are someone else's services. The arithmetic, at typical mid-2026 list prices (estimates age fast — redo this table before believing it):

```python
VPS = [("4 vCPU / 8 GB VPS", 48.00),
       ("80 GB SSD, included", 0.00),
       ("automated backups", 9.60)]
MANAGED = [("EC2 t3.medium", 30.37),
           ("RDS db.t4g.micro", 24.82),
           ("ElastiCache t4g.micro", 11.52),
           ("EBS + snapshots", 8.00),
           ("NAT + egress, est.", 12.00)]

totals = {}
for name, items in [("one VPS", VPS), ("managed cloud", MANAGED)]:
    totals[name] = sum(c for _, c in items)
    print(f"{name}: ${totals[name]:,.2f}/mo")
    for label, cost in items:
        print(f"  {label:<24} ${cost:6.2f}")
print(f"difference: ${totals['managed cloud'] - totals['one VPS']:.2f}/mo")
# => one VPS: $57.60/mo
#      4 vCPU / 8 GB VPS        $ 48.00
#      80 GB SSD, included      $  0.00
#      automated backups        $  9.60
#    managed cloud: $86.71/mo
#      EC2 t3.medium            $ 30.37
#      RDS db.t4g.micro         $ 24.82
#      ElastiCache t4g.micro    $ 11.52
#      EBS + snapshots          $  8.00
#      NAT + egress, est.       $ 12.00
#    difference: $29.11/mo
```

Twenty-nine dollars a month is not a decision; it is noise against a single operational mistake. The real trade is responsibility. On the VPS, *you* are the database administrator: Postgres upgrades, backup verification, disk monitoring, kernel patches — roughly a competent hour a month, forever, with the failure modes yours to own. Managed services buy back that hour and add guarantees that are genuinely hard to self-host — point-in-time database recovery, automated failover — at the price of less control, another dashboard of knobs, and bills that grow with every convenience enabled. The honest recommendation for this course's system: one VPS, compose, `pgdata` backed up nightly off the box, because the operational skills it forces you to practice are the same ones the rest of this part teaches, and at five processes the managed cloud's advantages are still mostly latent. The decision *reverses* when the stakes rise — more capital, more strategies, or anyone else's money, at which point point-in-time recovery of the fills table stops being a luxury — and knowing where the flip point sits is worth more than either default.

## The speed of light is not your bottleneck

Last, geography. Brokers and exchanges for US equities live effectively in the New Jersey and Virginia data-center corridors, and physics sets a floor under every round trip: light in fiber travels at about two-thirds of $c$, so a message cannot beat

$$
\text{RTT} \;\ge\; \frac{2d}{\tfrac{2}{3}c}
$$

for path distance $d$. Compute the floors, then measure reality from this very machine — which sits in Manila, about as far from New Jersey as a trading box can be:

```python
import time
from math import asin, cos, radians, sin, sqrt

import requests

C_FIBER = 299_792.458 * 2 / 3             # km/s — light, slowed by glass

CITIES = {"Secaucus NJ": (40.79, -74.06), "Ashburn VA": (39.04, -77.49),
          "London": (51.51, -0.13), "Tokyo": (35.68, 139.69),
          "Manila": (14.60, 120.98)}

def rtt_floor(a, b):
    la1, lo1 = map(radians, CITIES[a])
    la2, lo2 = map(radians, CITIES[b])
    d = 6371 * 2 * asin(sqrt(sin((la2 - la1) / 2) ** 2 +
                             cos(la1) * cos(la2) *
                             sin((lo2 - lo1) / 2) ** 2))
    return d, 2 * d / C_FIBER * 1000

for city in ("Ashburn VA", "London", "Tokyo", "Manila"):
    d, ms = rtt_floor("Secaucus NJ", city)
    print(f"NY metro <-> {city:<10} {d:7,.0f} km  RTT floor {ms:5.1f} ms")

for url in ("https://api.github.com", "https://www.google.com"):
    times = []                            # a live measurement — will vary
    for _ in range(3):
        t0 = time.perf_counter()
        requests.get(url, timeout=10)
        times.append(time.perf_counter() - t0)
    print(f"{url:<28} best of 3: ~{min(times)*1000:.0f} ms")
# => NY metro <-> Ashburn VA     351 km  RTT floor   3.5 ms
#    NY metro <-> London       5,568 km  RTT floor  55.7 ms
#    NY metro <-> Tokyo       10,840 km  RTT floor 108.5 ms
#    NY metro <-> Manila      13,664 km  RTT floor 136.7 ms
#    https://api.github.com       best of 3: ~144 ms
#    https://www.google.com       best of 3: ~118 ms
```

The measured numbers land close to the geometric floors — real HTTPS round trips from Manila to US infrastructure come in near the ~137 ms physics allows, which says the internet's routing overhead is modest and the floor is what governs. Now put the numbers against the strategy. The tsmom system decides once a day and fills at the next open; its edge, [Part IV established](../part-04-strategy-development/01-momentum-and-trend-following.md), plays out over *months*. Whether its order arrives 4 ms or 140 ms after submission is invisible next to the overnight gap it already accepts by design — so the box belongs near the broker (a US-East VPS turns every conversation into a ~4 ms local call and, more importantly, keeps trading *unaffected by your home internet, your laptop's sleep schedule, and your own timezone*), and then latency stops being worth another minute of optimization. The systems for which this paragraph is false — market makers and arbitrageurs paying for co-location and microwave links, where the floor itself is the battlefield — are [Part I's microstructure territory](../part-01-foundations/03-market-microstructure.md) and the crypto variant in the [advanced modules](../advanced/13-crypto-microstructure.md); knowing which side of that line your strategy lives on is the entire regional-latency decision.

!!! warning "If you cannot rebuild the machine from a file, the machine is a hostage"
    Everything else in this course is reproducible: data from frozen caches, results from replayable logs, strategies from versioned code. A hand-configured server is the one component that breaks the chain — its configuration lives nowhere but inside itself, it cannot be tested without touching production, and it converts every hardware failure into a reconstruction project conducted from memory, under pressure, with positions on. Freezing the environment into images and the topology into a compose file is not devops ceremony; it is the same standard the parquet files already meet, applied to the last component that was still exempt.

!!! abstract "Key takeaways"
    - The drift surface is real and printable: python 3.12.3, pandas 3.0.5, and four infra libraries whose versions must agree between the machine that validated the strategy and the machine that trades it — an agreement a hand-rebuilt box honors only by luck.
    - An image freezes the environment the way parquet froze the data: dependencies-first layer ordering for fast rebuilds, a non-root user, no secrets in any layer, and date tags — never `latest` — so the running environment can be named and re-pulled years later.
    - The compose file is the executable architecture diagram: pipeline services gated on `service_healthy` stores, the `pgdata` volume as the one thing that outlives any container, Redis unvolumed because its contents are designed to be lost, and one loopback-only port as the system's entire external surface.
    - Restart is a policy: seeded-jitter backoff timelines pin at 1.3s/2.3s/6.6s, a transient bug is up on attempt 3, and a permanent one hits the crash-loop cutoff at attempt 4 — stores get `always`, order-touching processes get `on-failure` plus lesson five's recovery, never blind revival.
    - Configuration is injected at start into a frozen dataclass — empty environment yields the paper profile, so forgotten config cannot touch money — and swapping `.env.paper` for `.env.live` is the entire difference between deployments, which is what makes paper evidence about live.
    - The network stance is default-deny: stores open on loopback, silent (TimeoutError) from outside, SSH as the only inbound service, and the dashboard reached by tunnel — because an unfirewalled Redis on a public IP is found by scanners in minutes.
    - A VPS runs this stack for ~$57.60/mo against ~$86.71 managed; the $29 is noise and the real currency is operational responsibility — while physics prices geography: RTT floors of 3.5 ms (Virginia) to 136.7 ms (Manila) against measured ~118–144 ms round trips, all invisible to a strategy that decides once a day.

## Where this goes next

The stack now deploys from files: the environment is an image, the topology is a compose file, the machine is disposable, and a crashed process comes back on policy instead of heroics. What none of that provides is *sight*. A system that restarts cleanly at 3am is a system that can fail at 3am, recover, fail again at 3:20, and greet you at breakfast with a book you do not recognize — every individual event handled, the pattern invisible, nothing anywhere obligated to tell a human. A deployment you can only inspect by SSH-ing in and reading logs is a deployment nobody inspects on a good day. [Monitoring, Logging, Alerting](04-monitoring-logging-alerting.md) gives the system its eyes: metrics that distinguish "the box is fine" from "the book is wrong," health endpoints that answer liveness and readiness as different questions, structured logs that reconstruct any incident after the fact, and an alerting policy honest about the only question that matters at 3am — what is worth waking a human for.
