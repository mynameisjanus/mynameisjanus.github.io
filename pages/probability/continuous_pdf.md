---
title: Continuous Random Variables
sidebar: probability_sidebar
permalink: continuous_pdf.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 19
---

## Probability Density Function

A continuous random variable $$X$$ is described by a **probability density function** $$f_X(x)$$. In this case, however, the probability that it takes a single value, say $$a$$, is zero, i.e.,

$$\mathbf{P}(X=a)=0.$$

Instead, we can take the probability that it is within a certain ***range***, say between $$a$$ and $$b$$. This is defined by

$$\mathbf{P}(a\leq X\leq b)=\int_{a}^{b}f_X(x)\,\mathrm{d}x.$$

### Non-negativity and Normalization

Like in the discrete case, the probability density function has to be **non-negative** and **normalized**, i.e.,

$$f_X(x)\geq 0\qquad\text{and}\qquad\int_{-\infty}^{\infty}f_X(x)\,\mathrm{d}x=1.$$


## Expectation

Instead of taking sums, we use integration to calculate the expectation of a continuous random variable, i.e.,

$$\mathbb{E}[X]=\int_{-\infty}^{\infty}xf_X(x)\,\mathrm{d}x.$$

### Properties of the Expectation

* If $$X\geq 0$$, then $$\mathbb{E}[X]\geq 0$$.
* If $$a\leq X\leq b$$, then $$a\leq\mathbb{E}[X]\leq b$$.
* Given a function $$g(X)$$,

  $$\mathbb{E}[g(X)]=\int_{-\infty}^{\infty}g(x)f_X(x)\,\mathrm{d}x.$$

* Linearity

  $$\mathbb{E}[aX+b]=a\,\mathbb{E}[X]+b$$

## Variance

Just like in the discrete case, the variance is defined as

$$\mathrm{var}(X)=\mathbb{E}[(x-\mu)^2]$$

where $$\mu=\mathbb{E}[X]$$. However, the explicit formula for calculating it is

$$\mathrm{var}(X)=\int_{-\infty}^{\infty}(x-\mu)^2f_X(x)\,\mathrm{d}x.$$

## Cumulative Distribution Function

The **cumulative distribution function** is defined as the probability that a random variable $$X$$ is less than $$x$$, i.e.,

$$F_X(x)=\mathbf{P}(X\leq x)=\int_{-\infty}^{x}f_X(t)\,\mathrm{d}t.$$

Using the fundamental theorem of calculus, we can calculate the probability density function from the cumulative distribution function, i.e.,

$$f_X(x)=\dfrac{\mathrm{d}F_X(x)}{\mathrm{d}x}.$$

### Properties of the CDF

* It is **non-decreasing**, i.e., if $$y\geq x$$, then $$F_X(y)\geq F_X(x)$$.
* As $$x\rightarrow -\infty$$, $$F_X(x)\rightarrow 0$$.
* As $$x\rightarrow +\infty$$, $$F_X(x)\rightarrow 1$$.

<br>

{% include custom/series_prob_next.html %}
