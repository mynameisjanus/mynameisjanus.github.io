---
title: Revisiting Conditioning
sidebar: probability_sidebar
permalink: revisiting_conditioning.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 30
---

### Conditional Expectation as a Random Variable

$$\mathbb{E}[X\lvert Y]=g(Y)$$

## Law of Iterated Expectations

$$\mathbb{E}\left[\mathbb{E}[X\lvert Y]\right]=\mathbb{E}[X]$$

## Law of Total Variance

$$\mathrm{var}(X)=\mathbb{E}[\mathrm{var}(X\lvert Y)]+\mathrm{var}\!\left(\mathbb{E}[X\lvert Y]\right)$$

### Sum of a Random Number of Independent Random Variables

$$\mathrm{var}(Y)=\mathbb{E}[N]\,\mathrm{var}(X)+(\mathbb{E}[X])^2\,\mathrm{var}(N)$$

<br>

{% include custom/series_prob_next.html %}
