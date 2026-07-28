# Feature Engineering for ML

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Construct candidate features from price, volume, and microstructure data and test each for stationarity before it enters a model
- Implement fixed-horizon and triple-barrier labeling and justify which is appropriate for a given strategy's holding period and exits
- Compute sample weights that correct for overlapping labels and show their effect on model training
- Audit a feature pipeline for leakage — temporal, cross-sectional, and preprocessing-induced — and demonstrate the fix

## Outline

1. Feature sources — prices, volume, and microstructure
2. Stationarity of features — tests, transformations, and fractional differencing
3. Labeling I — fixed-horizon returns and their pathologies
4. Labeling II — the triple-barrier method
5. Sample weights — correcting for overlapping labels
6. Leakage — how it enters pipelines and how to audit for it
7. The leak-free pipeline — the dataset used for the rest of this part

## Prerequisites

- [Part IV — Feature and Signal Engineering](../part-04-strategy-development/05-feature-and-signal-engineering.md)
