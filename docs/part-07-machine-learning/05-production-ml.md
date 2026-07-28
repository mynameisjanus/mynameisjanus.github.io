# Production ML

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Implement online learning updates and decide, with evidence, when incremental updates beat scheduled full retrains
- Detect concept drift with statistical monitors on features and predictions, and define the action each drift signal triggers
- Run champion/challenger retraining with explicit promotion criteria and a rollback path
- Version models in a registry so any live prediction can be traced to an exact model artifact, training window, and feature set

## Outline

1. Online learning — incremental updates vs full retrains
2. Concept drift — detection on features, predictions, and outcomes
3. Model monitoring — performance, stability, and staleness in production
4. Champion/challenger retraining — promotion criteria and rollback
5. Model registries and versioning — artifacts, lineage, and reproducibility
6. Wiring model health into the live monitoring and alerting stack

## Prerequisites

- [Tree Ensembles](02-tree-ensembles.md)
- [Part VI — Monitoring, Logging, Alerting](../part-06-live-infrastructure/04-monitoring-logging-alerting.md)
