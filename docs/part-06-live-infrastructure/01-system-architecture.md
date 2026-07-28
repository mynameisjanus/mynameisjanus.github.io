# System Architecture

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Design a live trading system as the Strategy → Risk Engine → Execution Engine → Broker API → Exchange pipeline with an explicit message contract at each boundary
- Decide which components share a process and which are isolated, justifying each choice in terms of failure domains and blast radius
- Classify every piece of system state as hot, durable, or reconstructable and assign it a storage location and recovery path
- Structure strategy code so that the identical module runs under the backtest harness and the live harness without modification

## Outline

1. The professional pipeline — Strategy → Risk Engine → Execution Engine → Broker API → Exchange
2. Message contracts — the interfaces between pipeline stages
3. Process boundaries — monolith vs separate processes, and what actually matters
4. State management — hot, durable, and reconstructable state
5. Failure domains — containing the blast radius of each component
6. Backtest/live code parity — one strategy, two harnesses

## Prerequisites

- [Part V — Architecture and Event-Driven Design](../part-05-backtesting-engine/01-architecture-and-event-driven-design.md)
