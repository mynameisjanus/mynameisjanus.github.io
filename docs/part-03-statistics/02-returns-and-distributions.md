# Returns and Their Distributions

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Convert between simple and log returns and aggregate each correctly across time and across portfolio positions.
- Document the stylized facts of asset returns — fat tails, skew, and volatility clustering — with statistics computed on real data.
- Fit normal, Student-t, normal inverse Gaussian, and stable distributions to real return series by maximum likelihood and compare their fit formally.
- Construct and read QQ plots to diagnose tail behavior and identify which distributional family a return series resembles.

## Outline

1. Simple vs log returns — definitions, and when each is the correct choice
2. Aggregation — compounding over time vs summing across a portfolio
3. Stylized facts — fat tails, skewness, volatility clustering in real data
4. The normal benchmark — where the Gaussian fails and by how much
5. Heavier-tailed families — Student-t, normal inverse Gaussian, stable distributions
6. Fitting and model comparison — maximum likelihood, information criteria
7. QQ plots — construction, reading the tails, common patterns

## Prerequisites

- [The Gaussian Distribution](../appendix/part-05-common-distributions/16-gaussian-distribution.md)
- [Functions of Random Variables](../appendix/part-03-random-variables/08-functions-of-random-variables.md)
