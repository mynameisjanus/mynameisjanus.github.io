---
title: Hypergeometric Distribution
sidebar: probability_sidebar
permalink: hypergeometric_distribution.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 19
---

parameters: population size, event count, sample size

## Probability Mass Function

The probability mass function for a hypergeometric distribution is

$$p_X(k)=\dfrac{\displaystyle\binom{r}{k}\binom{n-r}{m-k}}{\displaystyle\binom{n}{m}}.$$

## Expectation

$$\mathbb{E}[X]=\frac{mr}{n}$$

## Variance

$$\mathrm{var}(X)=\frac{mr(n-r)(n-m)}{n^2(n-1)}$$

## Monte Carlo Simulation

<br>

{% include custom/series_prob_next.html %}
