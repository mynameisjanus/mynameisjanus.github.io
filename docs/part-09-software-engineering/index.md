# Part IX — Professional Software Engineering

A major weakness of most trading courses is that they ignore software engineering entirely. Research code that cannot be trusted, tested, or deployed is not an asset — it is a liability, and often a silent one: the lookahead bug in an unversioned notebook, the "temporary" config edit that changed live sizing, the backtest nobody can reproduce six months later. Professional desks treat their codebase as production infrastructure, because it is.

This part professionalizes the codebase built in Parts II through VI. The material is not generic software engineering with trading examples bolted on; every practice is motivated by a failure mode specific to quantitative work — reviewing for lookahead bias, testing stochastic code without flaky suites, golden-file backtests that catch unintended P&L changes, message-queue semantics under market-data bursts, and refactoring legacy research code without changing its behavior.

The standard to internalize: any experiment reproducible from a commit hash, any deployment gated by CI, any hot path justified by a profile. That is the difference between a pile of scripts and a trading platform.

## Modules

| Module | Focus |
|---|---|
| [Git and Code Review](01-git-and-code-review.md) | Workflows for research and production code, reviewing quant code, reproducible experiments |
| [Testing and CI/CD](02-testing-and-cicd.md) | Unit, property, and regression tests, testing stochastic code, golden-file backtests, pipelines that gate deployment |
| [Package Structure, Configuration, and Dependency Injection](03-package-structure-config-di.md) | Laying out a trading platform, configuration strategy, swapping brokers and feeds, plugin architectures |
| [Architecture Patterns and Message Queues](04-architecture-patterns-and-message-queues.md) | Layered versus event-driven designs, Redis Streams / RabbitMQ / Kafka, backpressure, exactly-once myths |
| [Profiling, Refactoring, and Versioning](05-profiling-refactoring-versioning.md) | Finding and fixing hot paths, safely refactoring legacy research code, semantic versioning and releases |
