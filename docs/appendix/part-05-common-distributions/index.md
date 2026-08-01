# Part V — Common Probability Distributions

A reference catalog of the standard families. Each page gives the distribution's PMF or density, expectation, and variance, and — where written — a Monte Carlo simulation in R. Discrete families first, then continuous.

The two summary numbers every page reports are defined and developed in [Expected Value](../part-04-expectation-and-moments/01-expected-value.md) and [Variance](../part-04-expectation-and-moments/02-variance.md); this part supplies their values family by family rather than re-deriving what they are. Two results from there do most of the work below: linearity of expectation, which is why a binomial mean falls out of a sum of Bernoullis with no independence argument, and the computational formula $\mathrm{var}(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2$, which is the route almost every variance on these pages takes. Where a family has moments only up to a finite order — the Student's $t$ and the Cauchy among them — the reason is [Higher-Order Moments](../part-04-expectation-and-moments/03-higher-order-moments.md).

## Topics

### Discrete

| Topic | Focus |
|---|---|
| [Bernoulli Distribution](01-bernoulli-distribution.md) | Indicator random variables, expectation, and variance |
| [Binomial Distribution](02-binomial-distribution.md) | PMF, moments, and Monte Carlo simulation of coin-toss counts |
| [Geometric Distribution](03-geometric-distribution.md) | PMF, memorylessness, and moments |
| [Negative Binomial Distribution](04-negative-binomial-distribution.md) | PMF and moments of the trials-until-k-successes distribution |
| [Hypergeometric Distribution](05-hypergeometric-distribution.md) | Sampling without replacement, and the contrast with the binomial |
| [The Poisson Distribution](06-poisson-distribution.md) | PMF, moments, and the Poisson limit of the binomial |
| [Multinomial Distribution](07-multinomial-distribution.md) | Partitions and counts across more than two categories |
| [Discrete Uniform Distribution](08-discrete-uniform-distribution.md) | Equally likely outcomes on a finite range |

### Continuous

| Topic | Focus |
|---|---|
| [Continuous Uniform Distribution](09-continuous-uniform-distribution.md) | The flat density on an interval |
| [Exponential Distribution](10-exponential-distribution.md) | Density, moments, and the continuous analog of geometric waiting times |
| [Gamma Distribution](11-gamma-distribution.md) | Sums of exponential waiting times and the gamma family |
| [Beta Distribution](12-beta-distribution.md) | The conjugate family for a Bernoulli parameter on the unit interval |
| [Chi-Square Distribution](13-chi-square-distribution.md) | Sums of squared standard normals and their role in variance tests |
| [Student's t Distribution](14-students-t-distribution.md) | Heavy-tailed sampling distribution of the standardized mean |
| [F Distribution](15-f-distribution.md) | Ratios of scaled chi-square variables, used to compare variances |
| [The Gaussian Distribution](16-gaussian-distribution.md) | The normal family, standardization, and linear transformations |
| [Lognormal Distribution](17-lognormal-distribution.md) | Multiplicative growth and the distribution of exponentiated normals |
| [Weibull Distribution](18-weibull-distribution.md) | Flexible failure-time distribution with shape-dependent hazard |
