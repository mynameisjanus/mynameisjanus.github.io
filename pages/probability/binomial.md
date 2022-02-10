---
title: Binomial Probability
sidebar: probability_sidebar
permalink: binomial.html
folder: probability
usemathjax: true
toc: false
---

<br>

<p align="center">
  <img src="images/prob/coins.png" style="width:400px;height:auto;"/>
</p>

<br>

Suppose we toss a coin $$n$$ times with the probability of getting a head $$\mathbf{P}(\text{H})=p$$ and, hence, the probability of getting a tail $$\mathbf{P}(\text{T})=1-p$$. We assume that the tosses are *independent*, i.e., the outcome of previous tosses does not affect subsequent tosses. What is the probability of getting $$k$$ heads?

Before we proceed, we consider the specific case $$n=6$$ and compute the probability of getting the sequence $$\text{HTHHTH}$$. We use the multiplication principle to get

$$\mathbf{P}(\text{HTHHTH})=p\cdot(1-p)\cdot p\cdot p\cdot(1-p)\cdot p=p^4(1-p)^2.$$

To compute for the probability of getting **any** sequence with $$4$$ heads, we have to multiply the above expression with the number of $$4$$-head sequences. This is just the same as number of $$4$$-element subsets taken from a set with $$6$$ elements, i.e., $$\binom{6}{4}$$. Therefore, the probability of getting $$4$$ heads in $$6$$ coin tosses is

$$\mathbf{P}(4\;\text{heads})=\binom{6}{4}p^4(1-p)^{6-4}.$$

Generalizing this result, the probability of getting $$k$$ heads in $$n$$ coin tosses is

$$\mathbf{P}(k\;\text{heads})=\binom{n}{k}p^k(1-p)^{n-k}.$$
