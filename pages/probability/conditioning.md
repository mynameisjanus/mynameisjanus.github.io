---
title: Conditioning
sidebar: probability_sidebar
permalink: conditioning.html
folder: probability
usemathjax: true
toc: false
series: "Probability series"
weight: 13
---

## Conditioning on an Event

### Conditional PMF and Expectation

Assuming that $$\mathbf{P}(A)>0$$,

$$p_{X\lvert A}(x)=\mathbf{P}(X=x\lvert A).$$

$$\sum_{x}p_{X\lvert A}(x)=1$$

$$\mathbb{E}[X\lvert A]=\sum_{x}xp_{X\lvert A}(x)$$

$$\mathbb{E}[g(X)\lvert A]=\sum_{x}g(x)p_{X\lvert A}(x)$$

### Total Expectation Theorem

$$\mathbb{E}[X]=\mathbf{P}(A_1)\mathbb{E}[X\lvert A_1]+\ldots+\mathbf{P}(A_n)\mathbb{E}[X\lvert A_n]$$


## Conditioning on another Random Variable

$$p_{X\lvert Y}(x\lvert y)=\dfrac{p_{X,Y}(x,y)}{p_Y(y)}$$

## Independence

If $$X$$ and $$Y$$ are independent, then

$$\mathbb{E}[XY]=\mathbb{E}[X]\,\mathbb{E}[Y]$$

If $$X$$ and $$Y$$ are independent, then $$g(X)$$ and $$h(Y)$$ are also independent.

$$\mathbb{E}[g(X)h(Y)]=\mathbb{E}[g(x)]\,\mathbb{E}[h(Y)].$$

$$\mathrm{var}(X+Y)=\mathrm{var}(X)+\mathrm{var}(Y).$$

<br>

{% include custom/series_prob_next.html %}
