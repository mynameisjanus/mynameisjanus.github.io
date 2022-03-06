---
title: Probability Density Function
sidebar: probability_sidebar
permalink: continuouspdf.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 18
---


## Probability Density Function

$$\mathbf{P}(a\leq X\leq b)=\int_{a}^{b}f_X(x)\,\mathrm{d}x$$


### Normalization

$$\int_{-\infty}^{\infty}f_X(x)\,\mathrm{d}x=1$$


## Expectation

$$\mathbb{E}[X]=\int_{-\infty}^{\infty}xf_X(x)\,\mathrm{d}x$$

Assume that $$\int_{-\infty}^{\infty}\lvert x\rvert f_X(x)\mathrm{d}x<\infty$$

## Variance

$$\mathrm{var}(X)=\int_{-\infty}^{\infty}(x-\mu)^2f_X(x)\,\mathrm{d}x$$


## Cumulative Distribution Function

$$F_X(x)=\mathbf{P}(X\leq x)=\int_{-\infty}^{x}f_X(t)\,\mathrm{d}t$$


$$f_X(x)=\dfrac{\mathrm{d}F_X(x)}{\mathrm{d}x}$$

<br>

{% include custom/series_prob_next.html %}
