# Deep Learning

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Implement feed-forward networks, LSTMs, temporal convolutional networks, and transformers for financial sequence problems
- Diagnose why deep models underperform gradient boosting on daily-bar problems in terms of sample size, signal-to-noise, and non-stationarity
- Identify the settings — intraday data, cross-sectional scale, alternative data — where deep learning plausibly earns its keep
- Benchmark every deep model against the tree-ensemble and linear baselines from the previous lesson and report the comparison honestly

## Outline

1. Feed-forward networks on tabular features — the honest starting point
2. LSTMs — sequence modeling and its cost in data
3. Temporal convolutional networks
4. Transformers for financial sequences — attention, positional encoding, and hype control
5. Why deep learning underwhelms on daily bars — samples, noise, and regime shift
6. Where deep learning earns its keep — intraday, cross-sectional, and alternative data
7. Benchmarking discipline — deep models vs the baselines that came before

## Prerequisites

- [Tree Ensembles](02-tree-ensembles.md)
