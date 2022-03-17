---
title: Negative Binomial Distribution
sidebar: probability_sidebar
permalink: negative_binomial_distribution.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 18
---

generalization of the geometric distribution

## Probability Mass Function

$$p_X(k)=\binom{k-1}{r-1}p^r(1-p)^{k-r}$$

$$k=r,r+1,\ldots$$

sum of $$r$$ independent geometric random variables

## Expectation

$$\mathbb{E}[X]=\frac{r}{p}$$

## Variance

$$\mathrm{var}(X)=\dfrac{r(1-p)}{p^2}$$

<br>

{% include custom/series_prob_next.html %}
