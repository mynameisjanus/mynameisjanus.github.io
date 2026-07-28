# Capstone 1 — Momentum Strategy on Historical Equities

!!! warning "Under development"
    Full project specifications, datasets, and grading rubrics are being written.
    The brief below defines the project's scope.

You will research and document a time-series momentum strategy on historical equity data, taking it through a complete research cycle: data acquisition and cleaning, signal construction, position sizing, cost-aware evaluation, and a written research note. Momentum is deliberately unoriginal — the point of this project is not the alpha, it is proving you can execute the full research loop with clean data handling and honest evaluation, because every later capstone assumes you can.

## Objectives

- Construct a time-series momentum signal across multiple lookback horizons and translate it into positions with an explicit sizing rule
- Handle data correctly: adjust for corporate actions, avoid survivorship bias, and document every cleaning decision
- Evaluate performance net of realistic transaction cost assumptions, including sensitivity to those assumptions
- Communicate the research in a note a portfolio manager could act on, including negative results

## Deliverables

- A reproducible research repository: data loaders, signal and backtest code, and a script that regenerates every figure from raw data
- Backtest results: equity curve, drawdown profile, rolling Sharpe ratio, turnover, and cost-sensitivity table
- A parameter sensitivity analysis showing how results vary across lookback and holding-period choices
- A written research note (5–10 pages) documenting methodology, assumptions, results, and what did not work

## You will use

- [Part II](../part-02-python/index.md)
- [Part III](../part-03-statistics/index.md)
- [Part IV](../part-04-strategy-development/index.md)
