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

The **negative binomial distribution** is a generalization of the geometric distribution. It represents the number of independent Bernoulli trials until $$r$$ successes is achieved. It is described by two parameters, the stopping parameter $$k$$ and the probability of success $$p$$.

## Probability Mass Function

$$p_X(k)=\binom{k-1}{r-1}p^r(1-p)^{k-r}$$

$$k=r,r+1,\ldots$$

sum of $$r$$ independent geometric random variables

## Expectation

To calculate the expectation of a negative binomial distribution, we use the fact that it is the sum of $$r$$ independent geometric random variables, i.e.,

$$X=X_1+\ldots+X_r$$

where $$X_i\sim\mathrm{Geom}(p)$$. Then,

$$\begin{align}
\mathbb{E}[X]&=\mathbb{E}[X_1]+\ldots+\mathbb{E}[X_r]\\
&=\frac{1}{p}+\ldots+\frac{1}{p}\\
&=\frac{r}{p}.
\end{align}$$

## Variance

The variance of a negative binomial distribution is

$$\begin{align}
\mathrm{var}(X)&=\mathrm{var}(X_1)+\ldots+\mathrm{var}(X_r)\\
&=\frac{1-p}{p^2}+\ldots+\frac{1-p}{p^2}\\
&=\dfrac{r(1-p)}{p^2}.
\end{align}$$

<br>

{% include custom/series_prob_next.html %}
