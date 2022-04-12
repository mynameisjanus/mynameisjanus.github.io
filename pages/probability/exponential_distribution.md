---
title: Exponential Distribution
sidebar: probability_sidebar
permalink: exponential_distribution.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 26
---


## Probability Density Function

$$f_X(x)=\begin{cases}
\lambda e^{-\lambda x},&\quad x\geq0\\
0,&\quad x<0
\end{cases}$$

## Expectation

$$\mathbb{E}[X]=\frac{1}{\lambda}$$

{{site.data.alerts.proof}}
<p>$$\begin{align}
\mathbb{E}[X]&=\int_{0}^{\infty}x\lambda e^{-\lambda x}\mathrm{d}x\\
&=
\end{align}$$
</p>
{{site.data.alerts.end}}

## Variance

$$\mathrm{var}(X)=\frac{1}{\lambda^2}$$

{{site.data.alerts.proof}}
<p>$$\begin{align}
\mathbb{E}[X]&=\int_{0}^{\infty}x^2\lambda e^{-\lambda x}\mathrm{d}x\\
&=
\end{align}$$
</p>
{{site.data.alerts.end}}


<br>

{% include custom/series_prob_next.html %}
