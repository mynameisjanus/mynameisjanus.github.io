# Drawdowns, Tail Risk, and Stress Testing

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Compute drawdown statistics — maximum drawdown, duration, conditional drawdown — and their sampling distributions for a strategy with a given Sharpe ratio and horizon
- Estimate tail risk with fat-tail-aware methods, including extreme value theory, and quantify how far Gaussian assumptions understate it
- Design and run a stress-testing program that combines historical crisis replays with hypothetical shocks and correlation stress
- Compare the cost and effectiveness of left-tail hedges (index puts, trend overlays, cash buffers) against simply reducing gross exposure

## Outline

1. Drawdown statistics — depth, duration, and path dependence
2. The psychology of drawdowns — why books get cut at the bottom
3. Expected drawdown as a function of Sharpe, volatility, and horizon
4. Tail risk measurement — fat tails, extreme value theory, tail dependence
5. Historical scenario analysis — replaying past crises against today's book
6. Hypothetical stress tests — factor shocks, liquidity stress, correlation stress
7. Hedging the left tail — options, trend overlays, cash, and their carry costs
8. The stress-test review — turning results into position and policy changes

## Prerequisites

- [Kelly, Volatility Targeting, and Leverage](02-kelly-vol-targeting-leverage.md)
