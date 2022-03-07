---
title: Binomial Distribution
sidebar: probability_sidebar
permalink: binomial_distribution.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 16
---


<p align="center">
  <img src="images/prob/binomial.png" style="width:200px;height:auto;"/>
</p>

Number of successes in $$n$$ independent trials.
Parameters $$n$$ and $$p$$

## Probability Mass Function

$$p_X(k)=\binom{n}{k}p^k(1-p)^{n-k}$$

## Expectation

$$\mathbb{E}[X]=np$$

## Variance

$$\mathrm{var}(X)=np(1-p)$$

## Monte Carlo Simulation

```r
library(tidyverse)

n <- 25        # number of trials
p <- 0.5       # probability of success
reps <- 10000  # number of replications

num_success <- replicate(reps, {
  exp <- sample(c(0, 1), size = n, replace = TRUE, prob = c(1-p, p))
  sum(x)
})

# Plot the number of success versus the percentage of replications
data.frame(num_success) %>%
  ggplot(aes(num_success)) +
  geom_bar(aes(y = ..prop..)) +
  labs(x = "Number of Successes", y = "Probability")
```
<br>

{% include custom/series_prob_next.html %}
