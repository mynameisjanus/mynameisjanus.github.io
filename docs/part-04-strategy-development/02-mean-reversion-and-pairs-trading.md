# Mean Reversion and Pairs Trading

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Test a price series or spread for mean reversion using variance ratios, the Hurst exponent, and ADF tests, and interpret the results jointly.
- Fit an Ornstein-Uhlenbeck model to a spread and compute its mean-reversion half-life as an input to holding-period design.
- Screen a universe for tradable pairs, estimate hedge ratios, and construct cointegration-based spreads while avoiding spurious relationships.
- Build z-score entry and exit rules for a spread strategy and evaluate its performance including the failure mode of a broken pair.

## Outline

1. The mean-reversion hypothesis — where and why prices revert
2. Testing for mean reversion — variance ratios, Hurst exponent, ADF on spreads
3. Ornstein-Uhlenbeck intuition — the workhorse model of a reverting spread
4. Half-life estimation — from OU parameters to holding-period design
5. Pairs selection — universe screens, filters, and spurious-pair traps
6. Cointegration-based spreads — hedge ratios and spread construction
7. Trading the spread — z-score entries, exits, and stop policy for broken pairs

## Prerequisites

- [Part III — Time Series](../part-03-statistics/03-time-series.md)
