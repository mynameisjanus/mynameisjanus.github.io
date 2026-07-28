# Market Impact Models

!!! warning "Under development"
    This optional advanced module is part of the course scaffold.

This module covers how trading moves prices and how to model it: the distinction between temporary and permanent impact, the empirically robust square-root law, and the practical craft of estimating impact from your own fills. Impact is the dominant cost for any strategy at scale and the binding constraint on capacity, so this module is for learners who intend to run size — or to answer honestly whether a backtested strategy survives its own footprint.

## Topics

- Temporary versus permanent impact: definitions, mechanisms, and why the distinction matters for scheduling
- The square-root law: the empirical evidence, its surprising universality across markets, and its limits
- Propagator and transient-impact models: impact decay and the autocorrelation of order flow
- Estimating impact from your own fills: markout curves, sample-size realities, and the noise floor
- Impact-aware backtesting: replacing fixed-bps cost assumptions with size-dependent models
- Capacity estimation: translating an impact model into a maximum deployable capital figure

## Recommended background

- [Part I — market microstructure](../part-01-foundations/03-market-microstructure.md)
