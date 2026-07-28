# Profiling, Refactoring, and Versioning

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Profile a Python trading pipeline with cProfile and py-spy and identify the true hot path from measured data rather than intuition
- Optimize a hot path through vectorization, caching, or algorithmic change, and quantify the speedup with before/after benchmarks
- Refactor legacy research code safely using characterization tests and incremental extraction, demonstrating unchanged backtest output at each step
- Apply semantic versioning and a release process — changelog, tags, deprecation policy — to the trading platform

## Outline

1. Measure before optimizing — benchmarks, baselines, and representative workloads
2. cProfile — call-graph profiling and reading the output honestly
3. py-spy — sampling a live process without stopping it
4. Optimizing hot paths — vectorization, caching, better algorithms, when to drop to native code
5. Refactoring legacy research code — characterization tests and finding seams
6. Incremental refactors — the strangler pattern applied to research pipelines
7. Semantic versioning — what counts as a breaking change in a trading platform
8. Releases — changelogs, tagging, deprecation windows, and rollbacks

## Prerequisites

- [Testing and CI/CD](02-testing-and-cicd.md)
