# Bayesian Methods and Hidden Markov Models

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Compute Bayesian posteriors for trading parameters such as hit rate and mean return, and interpret credible intervals against their frequentist counterparts.
- Apply shrinkage estimators to noisy expected returns and covariance matrices and measure the improvement out of sample.
- Fit a Hidden Markov Model to real return data, select the number of states, and label the resulting volatility regimes.
- Evaluate a strategy's performance conditional on inferred regime and incorporate regime probabilities into a sizing decision.

## Outline

1. The Bayesian framework for trading parameters — priors, posteriors, sequential updating
2. Choosing priors — informative vs weak priors, sensitivity analysis
3. Shrinkage — James-Stein intuition, shrinking means and covariance matrices
4. Hidden Markov Models — states, transition matrices, emission distributions
5. Fitting HMMs — Baum-Welch, state-count selection, convergence diagnostics
6. Regime detection on real data — labeling volatility and trend regimes
7. Using regimes — conditional performance analysis and regime-aware sizing

## Prerequisites

- [The Bayesian Framework](../appendix/part-16-bayesian-statistics/01-bayesian-framework.md)
- [Markov Chains](../appendix/part-08-stochastic-processes/05-markov-chains.md)
- [Hidden Markov Models](../appendix/part-08-stochastic-processes/07-hidden-markov-models.md)
