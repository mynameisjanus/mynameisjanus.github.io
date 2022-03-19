---
title: Geometric Distribution
sidebar: probability_sidebar
permalink: geometric_distribution.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 17
---

<p align="center">
  <img src="images/prob/geom_dist.png" style="width:200px;height:auto;"/>
</p>

The **geometric distribution** represents the number of independent Bernoulli trials until the first success. For example, it describes the number of coin tosses needed to get a head, where head is considered as a success. It is a discrete probability distribution with one parameter $$p$$ and denoted by $$\mathrm{Geom}(p)$$.

## Probability Mass Function

The probability mass function of a geometric distribution is given by

$$p_X(k)=(1-p)^{k-1}p$$

where $$k=1,2,\ldots$$

## Expectation

The mean of a geometric distribution with parameter $$p$$ is

$$\mathbb{E}[X]=\dfrac{1}{p}.$$

{{site.data.alerts.proof}}
<p>
$$\begin{align}
\mathbb{E}[X]&=\sum_{k=1}^{\infty}kp(1-p)^{k-1}\\
&=-p\,\dfrac{\mathrm{d}}{\mathrm{d}p}\sum_{k=1}^{\infty}(1-p)^k\\
&=-p\,\dfrac{\mathrm{d}}{\mathrm{d}p}\left[-1+\sum_{k=0}^{\infty}(1-p)^k\right]\\
&=-p\,\dfrac{\mathrm{d}}{\mathrm{d}p}\left(-1+\dfrac{1}{p}\right)\\
&=\frac{1}{p}
\end{align}$$
</p>
{{site.data.alerts.end}}

## Variance

$$\mathrm{var}(X)=\dfrac{1-p}{p^2}$$

## Conditioning

Suppose now that the first trial was a failure. What is the distribution of the remaining trials until the first success? Because we are dealing with independent trials, the remaining trials will still have a geometric distribution, i.e., conditioned on $$X>1$$, $$X-1$$ is geometric with parameter $$p$$.

### Memorylessness

In general, if the first $$n$$ trials are failures, then the remaining trials will still be geometric, i.e., conditioned on $$X>n$$, $$X-n$$ is geometric with parameter $$p$$. This property is called **memorylessness**.

## Monte Carlo Simulation

```r
library(tidyverse)

reps <- 10000 # number of replications
p <- 0.5      # probability of success

# Initialize vector for the replications
tosses_to_success <- 0

for (i in 1:reps) {
  success <- 0
  counter <- 0
  while (success == 0) {
    counter <- counter + 1
    toss <- sample(c(0, 1), size = 1, replace = TRUE, prob = c(1-p, p))
    success <- success + toss
  }
  tosses_to_success[i] <- counter
}

# Plot the distribution
data.frame(tosses_to_success) %>%
  ggplot(aes(tosses_to_success, y = ..prop..)) +
  geom_bar() +
  labs(x = "Number of Tosses Until First Success", y = "Probability")
```

<p align="center">
  <img src="images/prob/geom_plot.png" style="width:600px;height:auto;"/>
</p>

<br>

{% include custom/series_prob_next.html %}
