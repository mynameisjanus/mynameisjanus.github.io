# Secrets, Paper/Live, and Compliance

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Manage broker credentials with a secrets store and injection pattern so keys never appear in code, images, logs, or version control
- Scope and rotate API keys, and detect a leaked credential before it is abused
- Run a paper trading phase as a formal promotion gate with explicit, measurable criteria for going live
- Produce position and order audit trails, backed by compliance-grade logging, that can reconstruct any trading decision after the fact

## Outline

1. Secrets management — stores, injection at runtime, and rotation
2. API key hygiene — least-privilege scoping, rotation schedules, and leak detection
3. Paper trading as a gate — what it can and cannot validate
4. Promotion criteria — measurable thresholds for leaving paper
5. The go-live checklist — infrastructure, limits, and sign-off
6. Position and order audit trails — immutable records of every decision
7. Compliance logging — retention, integrity, and answering "why did the system do that?"

## Prerequisites

- [Resilience and Risk Controls](05-resilience-and-risk-controls.md)
