---
title: Conditioning
sidebar: probability_sidebar
permalink: conditioning_cont.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 21
---

## Conditioning on an Event

$$\mathbb{E}[X\lvert A]=\int xf_{X\lvert A}(x)\,\mathrm{d}x$$

$$\mathbb{E}[g(X)\lvert A]=\int g(x)f_{X\lvert A}\,\mathrm{d}x$$

### Total Probability Theorem

$$f_X(x)=\mathbf{P}(A_1)f_{X\lvert A_1}(x)+\ldots+\mathbf{P}(A_n)f_{X\lvert A_n}(x)$$


## Conditioning on another Random Variable

If $$f_Y(y)>0,$$

$$f_{X\lvert Y}(x\lvert y)=\dfrac{f_{X,Y}(x,y)}{f_Y(y)}$$

## Independence

$$f_{X,Y}(x,y)=f_X(x)f_Y(y)$$

If $$X$$ and $$Y$$ are independent,

$$\mathbb{E}[XY]=\mathbb{E}[X]\,\mathbb{E}[Y]$$



<br>

{% include custom/series_prob_next.html %}
