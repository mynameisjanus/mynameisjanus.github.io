# Capstone 7 — Live Deployment with Monitoring and Risk Controls

!!! warning "Under development"
    Full project specifications, datasets, and grading rubrics are being written.
    The brief below defines the project's scope.

You will deploy the full stack — data, strategy, portfolio optimizer, execution — to a cloud server and operate it as a production system: pre-trade risk limits, circuit breakers, alerting, operational dashboards, and audit-grade logs. This is the integration of everything built in Capstones 1–6, and the grading emphasis shifts from building to operating: the system must enforce its own limits without human intervention, page you when something is wrong, and leave a log trail from which any trade can be reconstructed after the fact.

## Objectives

- Deploy the complete stack to a cloud server with reproducible provisioning and documented recovery procedures
- Implement pre-trade risk limits and circuit breakers (position caps, loss limits, kill switch) with tests proving each one triggers
- Build alerting and dashboards covering system health, positions, PnL, and risk-limit utilization
- Maintain compliance-style logs and an operational runbook, and operate the system over a sustained multi-week period

## Deliverables

- A deployed, running system with infrastructure defined as code or a fully documented provisioning procedure
- A risk-control module with an automated test for every limit and circuit breaker
- Alerting rules and an operational dashboard in working order
- An operational runbook plus an incident log from the multi-week run, including at least one deliberately induced failure and its handling
- Audit logs sufficient to reconstruct any order from signal to fill

## You will use

- [Part VI](../part-06-live-infrastructure/index.md)
- [Part IX](../part-09-software-engineering/index.md)
- [Part X](../part-10-trading-business/index.md)
