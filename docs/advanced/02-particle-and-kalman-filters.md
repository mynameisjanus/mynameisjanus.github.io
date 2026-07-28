# Particle and Kalman Filters

!!! warning "Under development"
    This optional advanced module is part of the course scaffold.

This module treats filtering as the general problem of estimating an unobserved state from noisy observations: the Kalman filter for linear-Gaussian state-space models, its workhorse application to dynamic hedge ratios, and particle filters for the nonlinear, non-Gaussian cases where the Kalman assumptions fail. It is for learners comfortable with the time-series material from Part III who want estimators that adapt through time rather than assuming fixed parameters.

## Topics

- State-space model formulation: latent states, observation equations, and how to cast trading problems in this form
- The Kalman filter: predict–update recursion, derivation, and a from-scratch implementation
- Dynamic hedge ratios for pairs and spread trading as time-varying regression via the Kalman filter
- Extended and unscented Kalman filters for mildly nonlinear systems, and their failure modes
- Particle filters: sequential importance sampling, resampling schemes, and degeneracy
- Applications to regime and volatility-state inference
- Numerical practice: covariance conditioning, initialization, and tuning process/observation noise

## Recommended background

- [Part III — time series](../part-03-statistics/03-time-series.md)
