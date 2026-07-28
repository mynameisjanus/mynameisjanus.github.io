# Bootstrap and Monte Carlo Methods

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Construct iid bootstrap confidence intervals for a strategy's Sharpe ratio and explain what the interval does and does not claim.
- Apply the block bootstrap to serially dependent returns, select a defensible block length, and compare the resulting intervals to the iid case.
- Simulate strategy return paths by Monte Carlo to estimate distributions of maximum drawdown and time-to-recovery.
- Design and run permutation tests that assess whether a signal's performance is distinguishable from noise.

## Outline

1. Why resampling — where analytic standard errors fail for strategy statistics
2. The iid bootstrap — mechanics, percentile and BCa intervals
3. The block bootstrap — serial dependence and block-length selection
4. Sharpe ratio confidence intervals — a complete worked pipeline on real strategy returns
5. Monte Carlo simulation of strategies — path generation, drawdown distributions
6. Permutation tests — constructing the null, signal-vs-noise verdicts
7. Pitfalls — heavy tails, small samples, and when the bootstrap misleads

## Prerequisites

- [The Weak Law of Large Numbers](../appendix/probability/36-weak-law-of-large-numbers.md)
- [The Central Limit Theorem](../appendix/probability/37-central-limit-theorem.md)
