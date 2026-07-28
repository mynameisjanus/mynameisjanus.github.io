# Time Series Analysis

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Test price and return series for stationarity using ADF and KPSS tests and interpret cases where the two disagree.
- Compute and interpret ACF and PACF of returns, absolute returns, and squared returns, including rolling autocorrelation over time.
- Fit ARIMA and GARCH-family models to return series, run residual diagnostics, and produce out-of-sample volatility forecasts.
- Test asset pairs and baskets for cointegration using the Engle-Granger and Johansen procedures and extract the cointegrating relationship.

## Outline

1. Stationarity — definitions, why it matters for every downstream model
2. Unit-root tests — ADF, KPSS, and how to read their disagreement
3. ACF and PACF — estimation, significance bands, returns vs absolute returns
4. ARIMA models — identification, fitting, residual diagnostics
5. The GARCH family — GARCH, EGARCH, GJR; volatility forecasting and evaluation
6. Cointegration with Engle-Granger — the two-step procedure on real pairs
7. The Johansen framework — multivariate cointegration and rank tests

## Prerequisites

- [Statistical Modeling](../appendix/statistics/03-statistical-modeling.md)
- [Markov Processes I](../appendix/probability/42-markov-processes-1.md)
