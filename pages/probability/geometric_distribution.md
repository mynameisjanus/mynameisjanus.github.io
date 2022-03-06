---
title: Geometric Distribution
sidebar: probability_sidebar
permalink: geometric_distribution.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 16
---

Number of tosses until the first Head

## Probability Mass Function

$$p_X(k)=\mathbf{P}(X=k)=(1-p)^{k-1}p$$

## Expectation

## Variance

$$\mathrm{var}(X)=\dfrac{1-p}{p^2}$$

## Conditioning

### Memorylessness



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
  labs(x = "Number of Tosses Until Success", y = "Probability")
```

<br>

{% include custom/series_prob_next.html %}
