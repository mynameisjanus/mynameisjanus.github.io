---
title: The Bayesian Inference Framework
sidebar: probability_sidebar
permalink: bayesian_inference_framework.html
folder: probability
toc: false
usemathjax: true
series: "Probability series"
weight: 32
---

## The Output of Bayesian Inference


## Point Estimates

### Maximum a Posteriori Probability (MAP)

$$p_{\Theta\lvert X}(\theta^*\lvert x)=\max_{\theta}p_{\Theta\lvert X}(\theta\lvert x)$$

$$f_{\Theta\lvert X}(\theta\lvert x)=\max_{\theta}f_{\Theta\lvert X}(\theta\lvert x)$$

### Least Mean Squares (LMS)

estimate: $$\hat{\theta}=g(x)$$

estimator: $$\hat{\Theta}=g(X)$$

### Discrete $$\Theta$$, Discrete $$X$$

$$p_{\Theta\lvert X}(\theta\lvert x)=\dfrac{p_{\Theta}(\theta)\,p_{X\lvert\Theta}(x\lvert\theta)}{p_X(x)}$$

$$p_X(x)=\sum_{\theta'}p_{\Theta}(\theta')\,p_{X\lvert\Theta}(x\lvert\theta')$$

### Discrete $$\Theta$$, Continuous $$X$$

$$p_{\Theta\lvert X}(\theta\lvert x)=\dfrac{p_{\Theta}(\theta)\,f_{X\lvert\Theta}(x\lvert\theta)}{f_X(x)}$$

$$f_X(x)=\sum_{\theta'}p_{\Theta}(\theta')\,f_{X\lvert\Theta}(x\lvert\theta')$$

### Continuous $$\Theta$$, Continuous $$X$$

$$f_{\Theta\lvert X}(\theta\lvert x)=\dfrac{f_{\Theta}(\theta)\,f_{X\lvert\Theta}(x\lvert\theta)}{f_X(x)}$$

$$f_X(x)=\int f_{\Theta}(\theta')\,f_{X\lvert\Theta}(x\lvert\theta')\,\mathrm{d}\theta'$$

<br>

{% include custom/series_prob_next.html %}
