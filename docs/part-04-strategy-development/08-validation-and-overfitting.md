# Validation and Overfitting

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Run a walk-forward optimization over a strategy's parameters and quantify the degradation from in-sample to out-of-sample performance.
- Implement purged and embargoed cross-validation for financial data with overlapping labels and explain the leakage each mechanism removes.
- Apply White's Reality Check to a family of strategy variants and interpret its data-snooping-robust p-value.
- Compute the Probability of Backtest Overfitting and the Deflated Sharpe Ratio for a research campaign and use them as explicit go/no-go gates.

## Outline

1. The overfitting problem — why this is the capstone of strategy development
2. Walk-forward optimization — windows, refitting, degradation analysis
3. Purged cross-validation — leakage from overlapping labels and how purging removes it
4. Embargoes — construction, parameter choice, and what they protect against
5. White's Reality Check — data-snooping-robust testing of a strategy family
6. Probability of Backtest Overfitting — combinatorially symmetric cross-validation
7. Deflated Sharpe Ratio — adjusting for trials, non-normality, and track length
8. The validation gauntlet — assembling the gates every course strategy must pass

## Prerequisites

- [Part III — Hypothesis Testing](../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md)
- [Part III — Bootstrap and Monte Carlo](../part-03-statistics/05-bootstrap-and-monte-carlo.md)
