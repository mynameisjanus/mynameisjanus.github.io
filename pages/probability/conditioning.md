---
title: Conditioning
sidebar: probability_sidebar
permalink: conditioning.html
folder: probability
usemathjax: true
toc: false
series: "Probability series"
weight: 11
---

## Conditional PMF and Expectation

Assuming that $$\mathbf{P}(A)\geq 0$$,

$$p_{X\lvert A}(x)=\mathbf{P}(X=x\lvert A).$$

$$\sum_{x}p_{X\lvert A}(x)=1$$

$$\mathbb{E}[X\lvert A]=\sum_{x}xp_{X\lvert A}(x)$$

$$\mathbb{E}[g(X)\lvert A]=\sum_{x}g(x)p_{X\lvert A}(x)$$

## Total Expectation Theorem

$$\mathbb{E}[X]=\mathbf{P}(A_1)\mathbb{E}[X\lvert A_1]+\ldots+\mathbf{P}(A_n)\mathbb{E}[X\lvert A_n]$$

<br>

{% include custom/series_prob_next.html %}
