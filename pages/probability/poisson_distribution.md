---
title: The Poisson Distribution
sidebar: probability_sidebar
permalink: poisson_distribution.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 20
---

<p align="center">
  <img src="images/prob/poisson.png" style="width:200px;height:auto;"/>
</p>

The **Poisson random variable** describes the probability that a given number of events occurs within a fixed time interval. It is a discrete random variable with a single parameter $$\lambda$$.

## Probability Mass Function

The Poisson distribution has the PMF

$$p_X(k)=\dfrac{\lambda^k}{k!}e^{-\lambda}$$

where $$k=0,1,2,\ldots$$

## Expectation

The mean of a Poisson random variable is

$$\mathbb{E}[X]=\lambda.$$

{{site.data.alerts.proof}}
<p>$$\begin{align}
\mathbb{E}[X]&=\sum_{k=0}^{\infty}k\frac{\lambda^k}{k!}e^{-\lambda}=e^{-\lambda}\sum_{k=1}^{\infty}\frac{\lambda^k}{(k-1)!}\\
&=e^{-\lambda}\sum_{k=0}^{\infty}\frac{\lambda^{k+1}}{k!}\\
&=\lambda e^{-\lambda}\sum_{k=0}^{\infty}\frac{\lambda^k}{k!}\\
&=\lambda e^{-\lambda}e^{\lambda}\\
&=\lambda
\end{align}$$
</p>
{{site.data.alerts.end}}

## Variance

The variance of a Poisson distribution with parameter $$\lambda$$ is

$$\mathrm{var}(X)=\lambda.$$

{{site.data.alerts.proof}}
<p>
To calculate the variance, we use

$$\mathrm{var}(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2=\mathbb{E}[X(X-1)]+\mathbb{E}[X]-(\mathbb{E}[X])^2.$$

The first term is

$$\begin{align}
\mathbb{E}[X(X-1)]&=\sum_{k=0}^{\infty}k(k-1)\frac{\lambda^k}{k!}e^{-\lambda}\\
&=e^{-\lambda}\sum_{k=2}^{\infty}\frac{\lambda^k}{(k-2)!}\\
&=\lambda^2e^{-\lambda}\sum_{k=2}^{\infty}\frac{\lambda^{k-2}}{(k-2)!}\\
&=\lambda^2e^{-\lambda}\sum_{k=0}^{\infty}\frac{\lambda^k}{k!}\\
&=\lambda^2e^{-\lambda}e^{\lambda}\\
&=\lambda^2.
\end{align}$$
Then,
$$\mathrm{var}(X)=\lambda^2+\lambda-\lambda^2=\lambda.$$
</p>
{{site.data.alerts.end}}

## Binomial versus Poisson Distributions

## Monte Carlo Simulation

```r
library(tidyverse)

# Set the parameters
p <- 0.01
n <- 300
# lambda = p*n

B <- 10000  # Number of replications

num_arrivals <- replicate(B, {
  arrival <- sample(c(0,1), size = n, replace = TRUE, prob = c(1-p,p))
  sum(arrival)
})

# Plot the distribution
data.frame(num_arrivals) %>%
  ggplot(aes(num_arrivals, y = ..prop..)) +
  geom_bar(width = 0.5, color = "dodgerblue", fill = "dodgerblue") +
  labs(x="Number of Arrivals", y="Probability")
```

<p align="center">
  <img src="images/prob/poisson_dist.png" style="width:600px;height:auto;"/>
</p>

<br>

{% include custom/series_prob_next.html %}
