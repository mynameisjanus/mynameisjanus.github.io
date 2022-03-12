---
title: The Unknown Bias of a Coin
sidebar: probability_sidebar
permalink: bias_of_a_coin.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 32
---

### Continuous $$\Theta$$, Discrete $$K$$

coin with bias $$\Theta$$; prior $$f_{\Theta}(\cdot)$$

$$f_{\Theta\lvert K}=\dfrac{f_{\Theta}(\theta)\,p_{K\lvert\Theta}(k\lvert\theta)}{p_K(k)}$$

$$p_K(k)=\int f_{\Theta}(\theta')\,p_{K\lvert\Theta}(k\lvert\theta')\,\mathrm{d}\theta'$$

## The Beta Distribution

$$\theta^k(1-\theta)^{n-k}$$

## MAP Estimate

## LMS Estimate

{{site.data.alerts.note}}
<p>
$$\int_{0}^{1}\theta^\alpha(1-\theta)^\beta\mathrm{d}\theta=\dfrac{\alpha!\,\beta!}{(\alpha+\beta+1)!}\qquad\text{for}\quad\alpha,\beta\geq0$$
</p>
{{site.data.alerts.end}}


<br>

{% include custom/series_prob_next.html %}
