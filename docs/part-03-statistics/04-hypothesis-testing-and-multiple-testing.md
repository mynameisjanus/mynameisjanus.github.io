# Hypothesis Testing and Multiple Testing

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Test whether a strategy's mean return differs from zero and state explicitly which assumptions the test relies on.
- Correct t-statistics for autocorrelated returns using HAC (Newey-West) standard errors and quantify how much the naive t-statistic overstates significance.
- Quantify the multiple-testing problem for a grid of strategy variants, including the expected number of false positives at a given significance level.
- Apply Bonferroni and Benjamini-Hochberg (FDR) corrections to a set of backtest p-values and defend which surviving strategies merit further work.

## Outline

1. Hypothesis tests on strategy returns — nulls, alternatives, test statistics
2. The t-test and its assumptions — independence, normality, sample size
3. Autocorrelation breaks the t-test — effective sample size, HAC/Newey-West corrections
4. Sharpe ratio inference — standard errors and testing Sharpe differences
5. The multiple-testing problem — how trying many variants manufactures significance
6. Familywise error control — Bonferroni and its conservatism
7. False discovery rate — Benjamini-Hochberg applied to a strategy grid

## Prerequisites

- [Introduction to Hypothesis Testing](../appendix/statistics/07-intro-to-hypothesis-testing.md)
- [Levels and p-Values](../appendix/statistics/08-levels-and-pvalues.md)
