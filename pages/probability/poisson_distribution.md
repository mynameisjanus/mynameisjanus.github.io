---
title: The Poisson Distribution
sidebar: probability_sidebar
permalink: poisson_distribution.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 20
---

parameter $$\lambda$$

## Probability Mass Function

$$p_X(k)=\dfrac{\lambda^k}{k!}e^{-\lambda}$$

where $$k=0,1,2,\ldots$$

## Expectation

The mean of a Poisson random variable is

$$\mathbb{E}[X]=\lambda.$$

{{site.data.alerts.proof}}
<p>$$\begin{align}
\mathbb{E}[X]&=\sum_{k=0}^{\infty}k\frac{\lambda^k}{k!}e^{-\lambda}=e^{-\lambda}\sum_{k=1}^{\infty}\frac{\lambda^k}{(k-1)!}\\
&=e^{-\lambda}\sum_{k=0}^{\infty}\frac{\lambda^{k+1}}{k!}\\
&=\lambda e^{-\lambda}\sum_{k=0}^{\infty}\frac{\lambda^k}{k!}\\
&=\lambda e^{-\lambda}e^{\lambda}\\
&=\lambda
\end{align}$$
</p>
{{site.data.alerts.end}}

## Variance

The variance of a Poisson distribution with parameter $$\lambda$$ is

$$\mathrm{var}(X)=\lambda.$$

{{site.data.alerts.proof}}
<p>
To calculate the variance, we use

$$\mathrm{var}(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2=\mathbb{E}[X(X-1)]+\mathbb{E}[X]-(\mathbb{E}[X])^2.$$

The first term is

$$\begin{align}
\mathbb{E}[X(X-1)]&=\sum_{k=0}^{\infty}k(k-1)\frac{\lambda^k}{k!}e^{-\lambda}\\
&=e^{-\lambda}\sum_{k=2}^{\infty}\frac{\lambda^k}{(k-2)!}\\
&=\lambda^2e^{-\lambda}\sum_{k=2}^{\infty}\frac{\lambda^{k-2}}{(k-2)!}\\
&=\lambda^2e^{-\lambda}\sum_{k=0}^{\infty}\frac{\lambda^k}{k!}\\
&=\lambda^2e^{-\lambda}e^{\lambda}\\
&=\lambda^2.
\end{align}$$
Then,
$$\mathrm{var}(X)=\lambda^2+\lambda-\lambda^2=\lambda.$$
</p>
{{site.data.alerts.end}}

<br>

{% include custom/series_prob_next.html %}
