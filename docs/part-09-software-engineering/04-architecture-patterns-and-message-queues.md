# Architecture Patterns and Message Queues

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Choose between layered and event-driven architectures for a given trading system and defend the choice in terms of latency, complexity, and testability
- Select the appropriate message queue — Redis Streams, RabbitMQ, or Kafka — for a stated workload, justified by delivery semantics, throughput, and operational burden
- Implement backpressure handling so a slow consumer degrades gracefully (bounded queues, conflation, load shedding) rather than silently dropping market data
- Explain why exactly-once delivery is unattainable end to end, and design idempotent consumers that make at-least-once delivery safe

## Outline

1. Layered architectures — where they work in trading systems, where they creak
2. Event-driven architectures — events, handlers, and replayable logs
3. Message queues in a trading stack — decoupling feeds, strategies, and execution
4. Redis Streams — lightweight streaming and consumer groups
5. RabbitMQ — routing, acknowledgements, and work queues
6. Kafka — partitions, offsets, retention, and replay
7. Backpressure — bounded buffers, conflating market data, shedding load
8. Exactly-once myths — delivery guarantees, deduplication, idempotent order handling

## Prerequisites

- [Package Structure, Configuration, and Dependency Injection](03-package-structure-config-di.md)
- [Part VI — System Architecture](../part-06-live-infrastructure/01-system-architecture.md)
