# Distributed Backtesting

!!! warning "Under development"
    This optional advanced module is part of the course scaffold.

This module covers scaling backtests across many cores and machines: parallelizing parameter sweeps with Ray and Dask, aggregating results reliably, and confronting the statistical hazard that comes with the speed — a cluster that evaluates ten thousand variants overnight is a machine for manufacturing overfit strategies unless the multiple-testing accounting scales with the compute. It is for learners who have built the Part V engine and outgrown a single machine.

## Topics

- Backtesting as an embarrassingly parallel workload: sweep structure, task granularity, and shared data
- Ray and Dask in practice: task graphs, cluster setup, and choosing between them
- Data locality: distributing market data to workers without making I/O the bottleneck
- Result aggregation: schema design, durable storage of sweep results, and resumable sweeps after partial failure
- Reproducibility across workers: seeding, environment pinning, and deterministic re-runs
- Distributed overfitting: why more compute means more multiple testing, and scaling the statistical corrections to match
- Cost management: spot instances, checkpointing, and knowing when a sweep is not worth running

## Recommended background

- [Part V — the backtesting engine](../part-05-backtesting-engine/index.md)
