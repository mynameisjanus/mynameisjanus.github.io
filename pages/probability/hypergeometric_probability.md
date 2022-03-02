---
title: Hypergeometric Probability
sidebar: probability_sidebar
permalink: hypergeometric_probability.html
folder: probability
usemathjax: true
toc: false
series: "Probability series"
weight: 7
---

<p align="center">
  <img src="images/prob/hypergeom.png" style="width:100px;height:auto;"/>
</p>

<br>

Consider a bowl with $$n$$ items of which $$m$$ are *red*. We randomly pick $$r$$ items from this bowl **without replacement**, i.e., we don't return to the bowl the items that we've picked. What is the probability that $$k$$ of the items we picked are red?

{% include warning.html content="This might seem to be exactly like the multinomial probability but it is not. In this problem, we are only interested in having $$k$$ red items out of the $$r$$ items that we've picked. Moreover, we are sampling without replacement. In the multinomial case, we can either sample with replacement or think of the bowl as having an infinite number of items so that, if we pick an item, the probability of getting the next item with the same color is unchanged." %}

To solve this problem, we first calculate the size of our sample space. This is simply the number of ways to pick $$r$$ items from $$n$$ items and it is

$$\text{size}(\Omega)=\lvert\Omega\rvert=\binom{n}{r}.$$

Next, we pick $$k$$ items from the $$m$$ red ones. The number of ways to do this is $$\binom{m}{k}$$. Once we have the red items, we pick the remaining $$(r-k)$$ items from the $$(n-m)$$ non-red items. There are $$\binom{n-m}{r-k}$$ ways to do this. We multiply these two to get the number of ways to get $$k$$ red items out of the $$r$$ items we picked. Therefore, the probability of getting $$k$$ red items, if we picked $$r$$ items, is

$$\dfrac{\displaystyle\binom{m}{k}\binom{n-m}{r-k}}{\displaystyle\binom{n}{r}}.$$

<br>

{% include custom/series_prob_next.html %}
