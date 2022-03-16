---
title: The Gaussian Distribution
sidebar: probability_sidebar
permalink: gaussian_distribution.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 28
---

## The Standard Normal Distribution

$$\mathcal{N}(0,1)$$

$$f_X(x)=\dfrac{1}{\sqrt{2\pi}}\,\exp\left(-\dfrac{x^2}{2}\right)$$

### Expectation and Variance

$$\mathbb{E}[X]=0$$

$$\mathrm{var}[X]=1$$


## General Normal Random Variables

$$f_X(x)=\dfrac{1}{\sigma\sqrt{2\pi}}\,\exp\left[-\dfrac{(x-\mu)^2}{2\sigma^2}\right]$$

### Expectation and Variance

$$\mathbb{E}[X]=\mu$$

$$\mathrm{var}[X]=\sigma^2$$

## Linear Functions of a Normal Random Variable

Let $$Y=aX+b$$ where $$X\sim\mathcal{N}(\mu,\sigma^2)$$. Then,

$$\mathbb{E}[Y]=a\mu+b\qquad\text{and}\qquad\mathrm{var}[Y]=a^2\sigma^2.$$

$$Y$$ is also a normal random variable, i.e.,

$$Y\sim\mathcal{N}(a\mu,a^2\sigma^2).$$

## Standard Normal Tables


## Standardizing a Random Variable

<br>

{% include custom/series_prob_next.html %}
