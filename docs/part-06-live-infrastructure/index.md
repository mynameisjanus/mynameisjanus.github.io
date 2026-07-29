# Part VI — Live Trading Infrastructure

This is the part most courses skip entirely. Research and backtesting get all the attention, but the gap between a promising backtest and a system that trades real money unattended is where most aspiring quants stall. Live trading is an engineering discipline: processes crash, data arrives late, brokers reject orders, and the system must behave sensibly through all of it — often at 3am with no human watching.

In this part you study a complete trading system with the same architecture professionals use: Strategy → Risk Engine → Execution Engine → Broker API → Exchange. Each stage has explicit interfaces and its own failure domain, so a bug in signal generation cannot bypass risk checks, and an execution problem cannot corrupt strategy state. The treatment is concrete throughout — schedulers and market calendars, Redis and PostgreSQL, Docker and cloud deployment, monitoring and alerting — not architecture diagrams left as an exercise.

The final lessons cover what separates hobby deployments from professional ones: crash recovery and state reconciliation, idempotent order handling, circuit breakers and kill switches, secrets management, paper trading as a promotion gate, and audit trails that would satisfy a compliance review. By the end you understand what it takes for a system to run overnight with no human watching — and exactly what should happen when it breaks.

## Modules

| Module | Focus |
|---|---|
| [System Architecture](01-system-architecture.md) | The Strategy → Risk Engine → Execution Engine → Broker API → Exchange pipeline, process boundaries, state, failure domains, and backtest/live parity |
| [Scheduling and Data Plumbing](02-scheduling-and-data-plumbing.md) | Schedulers and market calendars, Redis for hot state and pub/sub, PostgreSQL for durable storage, and data pipelines into the live system |
| [Docker and Cloud Deployment](03-docker-and-cloud-deployment.md) | Containerizing the stack, docker-compose, VPS vs managed cloud, networking, and regions/latency basics |
| [Monitoring, Logging, Alerting](04-monitoring-logging-alerting.md) | Metrics and health endpoints, structured log aggregation, alerting policies, and dashboards for positions and PnL |
| [Resilience and Risk Controls](05-resilience-and-risk-controls.md) | Crash recovery, state reconciliation, idempotent orders, circuit breakers, pre-trade limits, and kill switches |
| [Secrets, Paper/Live, and Compliance](06-secrets-paper-live-compliance.md) | Secrets management, API key hygiene, paper trading as a gate, go-live checklists, and audit trails |
