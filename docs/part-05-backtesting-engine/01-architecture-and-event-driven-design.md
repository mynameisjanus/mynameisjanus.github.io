# Architecture and Event-Driven Design

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Explain when vectorized backtesting produces wrong answers and an event-driven simulation is required
- Implement an event queue and dispatch loop that processes market data, signal, order, and fill events in strict timestamp order
- Design event dataclasses whose fields and flow make look-ahead bias structurally impossible rather than a matter of discipline
- Trace a single bar of market data through the full loop — from data handler to signal to order to fill — and account for every state change it causes

## Outline

1. Vectorized vs event-driven backtesting — speed, fidelity, and failure modes
2. The four core event types — market data, signal, order, fill
3. Event dataclasses — fields, immutability, and the event contract
4. The event queue — FIFO semantics and timestamp ordering
5. The main event loop — dispatch, component handlers, and termination
6. Look-ahead bias by construction — why the architecture, not vigilance, prevents it
7. The engine skeleton — the components you will extend for the rest of the course

## Prerequisites

- [Part II — Typing, Dataclasses, and Code Structure](../part-02-python/03-typing-dataclasses-structure.md)
