# Async and Market Data APIs

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Write async/await code with correct task creation, cancellation, and timeout handling.
- Consume a REST market data API with aiohttp, handling pagination, authentication, and vendor rate limits.
- Implement retry logic with exponential backoff and jitter, and identify which requests are safe to retry.
- Maintain a resilient websocket subscription to streaming market data with reconnect and gap-detection logic, and justify the choice between REST polling and streaming for a given data need.

## Outline

1. The event loop — coroutines, tasks, await semantics, blocking hazards
2. Concurrency patterns — gather, semaphores, cancellation, timeouts
3. REST market data — aiohttp sessions, pagination, authentication
4. Rate limits — token buckets, concurrency caps, respecting vendor terms
5. Retries — exponential backoff, jitter, idempotency, giving up cleanly
6. Streaming with websockets — subscriptions, heartbeats, reconnects, sequence gaps
7. REST vs streaming — latency, completeness, and cost trade-offs

## Prerequisites

- [Typing, Dataclasses, and Code Structure](03-typing-dataclasses-structure.md)
