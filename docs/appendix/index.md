# Mathematical Prerequisites

This appendix is a self-contained reference for the mathematics, probability, and statistics that the course builds on. It is not a detour — quantitative trading *is* applied probability and statistics, and [Part III — Statistics for Trading](../part-03-statistics/index.md) links directly into these pages wherever a lesson leans on a formal result.

## How to use this appendix

- **If you have a quantitative background**, skim the part titles and move on. Come back when a course lesson cites a specific result.
- **If you are newer to probability**, work through Parts I–VIII in order — they build from counting principles up to the Central Limit Theorem and Markov processes, which is exactly the toolkit regime models and Monte Carlo methods assume later.
- **The statistics parts (X–XVII)** cover estimation, hypothesis testing, regression, model selection, multiple testing, and Bayesian methods — the machinery behind every "is this strategy actually profitable?" question the course asks.
- **Part XVIII** connects the formal results back to trading: sizing, risk, microstructure, and regime detection.

## Contents

| Part | Focus |
|---|---|
| [Part I — Mathematical Foundations](part-01-mathematical-foundations/index.md) | Sets, counting, notation, series, and linear algebra |
| [Part II — Foundations of Probability](part-02-probability-foundations/index.md) | Sample spaces, axioms, conditioning, Bayes' rule, and independence |
| [Part III — Random Variables](part-03-random-variables/index.md) | CDFs, PMFs, PDFs, joint and conditional distributions, transformations |
| [Part IV — Expectation and Moments](part-04-expectation-and-moments/index.md) | Expectation, variance, covariance, correlation, and the total laws |
| [Part V — Common Probability Distributions](part-05-common-distributions/index.md) | The standard discrete and continuous families |
| [Part VI — Multivariate Probability](part-06-multivariate-probability/index.md) | Random vectors, covariance matrices, and the multivariate Gaussian |
| [Part VII — Asymptotic Theory](part-07-asymptotic-theory/index.md) | Laws of large numbers, the CLT, and the mapping theorems |
| [Part VIII — Stochastic Processes](part-08-stochastic-processes/index.md) | Arrival processes, Markov chains, HMMs, and Brownian motion |
| [Part IX — Monte Carlo Methods](part-09-monte-carlo-methods/index.md) | Simulation, sampling, variance reduction, and resampling |
| [Part X — Foundations of Statistics](part-10-statistics-foundations/index.md) | Samples, sampling distributions, and statistical models |
| [Part XI — Parameter Estimation](part-11-parameter-estimation/index.md) | Point and interval estimation, ML, moments, and Bayesian estimates |
| [Part XII — Hypothesis Testing](part-12-hypothesis-testing/index.md) | Tests, p-values, errors, power, and the main test families |
| [Part XIII — Regression and Statistical Models](part-13-regression/index.md) | Linear and generalized linear models, regularization, diagnostics |
| [Part XIV — Model Selection](part-14-model-selection/index.md) | Bias–variance, cross-validation, information criteria |
| [Part XV — Multiple Testing](part-15-multiple-testing/index.md) | Corrections for many comparisons and data snooping |
| [Part XVI — Bayesian Statistics](part-16-bayesian-statistics/index.md) | Priors, posteriors, conjugacy, updating, and prediction |
| [Part XVII — Statistical Computing](part-17-statistical-computing/index.md) | Optimization, EM, and MCMC |
| [Part XVIII — Applications to Quantitative Finance](part-18-quant-finance-applications/index.md) | Sizing, risk, microstructure, and regime detection |

!!! note
    Much of this material was migrated from an earlier site and is kept mathematically rigorous, with proofs in collapsible blocks and R-based Monte Carlo simulations. Pages marked **Draft** are placeholders for planned content.
