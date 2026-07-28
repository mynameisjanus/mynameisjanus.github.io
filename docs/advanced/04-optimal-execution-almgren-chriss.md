# Optimal Execution: Almgren–Chriss

!!! warning "Under development"
    This optional advanced module is part of the course scaffold.

This module develops the Almgren–Chriss framework for optimal trade scheduling: how to split a parent order over time to balance market impact against timing risk, measured through implementation shortfall. It is the foundational model of execution research — the baseline every practical scheduler and every academic extension is compared against — and is intended for learners who understood the microstructure material in Part I and are targeting execution or trading-infrastructure roles.

## Topics

- Implementation shortfall: definition, decomposition into impact, timing, and opportunity components
- The Almgren–Chriss model: temporary and permanent impact terms, risk aversion, and the objective function
- Deriving the optimal trajectory and the efficient frontier of execution strategies
- Comparing scheduled strategies: TWAP, VWAP, and implementation-shortfall schedules, and when each is appropriate
- Calibrating the model's impact parameters from execution data, and the noise involved
- Extensions: adaptive schedules that respond to price and liquidity, and the limits of the static model

## Recommended background

- [Part I — market microstructure](../part-01-foundations/03-market-microstructure.md)
