# Capstone 3 — Pairs Trading via Cointegration

!!! warning "Under development"
    Full project specifications, datasets, and grading rubrics are being written.
    The brief below defines the project's scope.

You will screen an equity universe for cointegrated pairs, build a spread-trading strategy on the pairs that survive, and analyze how sensitive the strategy is to regime change. Pairs trading is the classic application of cointegration, and it fails in a classic way: relationships that test as significant in-sample break down out-of-sample. Your job is to select pairs with statistical discipline — including honest accounting for the multiple testing inherent in screening thousands of candidates — and to characterize exactly how the strategy behaves when a relationship breaks.

## Objectives

- Build a pair-selection pipeline using cointegration tests with strict in-sample/out-of-sample separation and multiple-testing control across the candidate universe
- Estimate hedge ratios and construct spread-trading rules with explicit entry, exit, and stop logic
- Quantify regime sensitivity: track rolling cointegration statistics and measure strategy behavior when relationships decay
- Evaluate aggregate and per-pair performance net of costs, including the short-borrow assumptions

## Deliverables

- A pair-selection pipeline with documented test statistics, selection thresholds, and the multiple-testing adjustment applied
- A spread strategy backtest with per-pair and portfolio-level results
- A regime analysis report including at least two case studies of pairs that broke down, with rolling diagnostics
- A reproducible code repository covering screening, backtesting, and analysis

## You will use

- [Part III — time series](../part-03-statistics/03-time-series.md)
- [Part IV — mean reversion and pairs](../part-04-strategy-development/index.md)
