# Bayesian Optimization for Hyperparameters

!!! warning "Under development"
    This optional advanced module is part of the course scaffold.

This module covers Bayesian optimization as a disciplined replacement for grid and random search when tuning strategy and model hyperparameters: Gaussian-process surrogates, acquisition functions, and — most importantly — how to use a sample-efficient optimizer without turning it into a more efficient way to overfit a backtest. It is for learners who already understand the validation and overfitting material from Part IV and routinely face expensive objective functions, whether backtest runs or ML training jobs.

## Topics

- Gaussian process regression as a surrogate model: kernels, hyperpriors, and what the posterior actually tells you
- Acquisition functions: expected improvement, upper confidence bound, and the exploration–exploitation trade-off
- Handling noisy objectives, which is the normal case when the objective is a backtest metric
- Practical tooling: Optuna, scikit-optimize, and BoTorch/Ax, and when the library choice matters
- Tuning strategy parameters without overfitting: nested validation, search-history accounting, and deflating reported performance
- Diagnosing a converged-but-wrong search: surrogate misfit and pathological parameter surfaces

## Recommended background

- [Part IV — validation and overfitting](../part-04-strategy-development/08-validation-and-overfitting.md)
