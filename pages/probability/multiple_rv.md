---
title: Multiple Random Variables
sidebar: probability_sidebar
permalink: multiple_rv.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 20
---

## Joint Probability Distribution Function

$$\mathbf{P}(a\leq X\leq b,c\leq Y\leq d)=\int_{c}^{d}\int_{a}^{b}f_{X,Y}(x,y)\,\mathrm{d}x\,\mathrm{d}y$$


### Marginal Probability Density Function

$$f_X(x)=\int f_{X,Y}(x,y)\,\mathrm{d}y$$

$$f_Y(y)=\int f_{X,Y}(x,y)\,\mathrm{d}x$$


## Joint Cumulative Distribution Function

$$F_{X,Y}(x,y)=\mathbf{P}(X\leq x,Y\leq y)=\int_{-\infty}^{y}\int_{-\infty}^{x}f_{X,Y}(s,t)\,\mathrm{d}s\,\mathrm{d}t$$

$$f_{X,Y}(x,y)=\dfrac{\partial^2F_{X,Y}(x,y)}{\partial x\,\partial y}$$

<br>

{% include custom/series_prob_next.html %}
