---
title: Multiple Random Variables
sidebar: probability_sidebar
permalink: multiple_random_variables.html
folder: probability
usemathjax: true
toc: false
series: "Probability series"
weight: 12

---

## Joint Probability Mass Function

When we have two random variables, $$X$$ and $$Y$$, their **joint probability mass function** is defined as the probability that $$X=x$$ **and** $$Y=y$$, i.e.,

$$p_{X,Y}(x,y)=\mathbf{P}(X=x, Y=y).$$

Like ordinary probabilities, it is normalized, i.e.,

$$\sum_{x}\sum_{y}p_{X,Y}(x,y)=1.$$

### Marginal Probability Mass Function

The probabilities $$p_X(x)$$ and $$p_Y(y)$$ are now called the **marginal PMFs** and they are calculated as

$$p_X(x)=\sum_{x}p_{X,Y}(x,y)\qquad p_Y(y)=\sum_{x}p_{X,Y}(x,y)$$

### More than 2 Random Variables

We can general this concept to three or more random variables. If we have three random variables, $$X$$, $$Y$$ and $$Z$$, the PMF is

$$p_{X,Y,Z}(x,y,z)=\mathbf{P}(X=x,Y=y,Z=z)$$

with normalization

$$\sum_{x}\sum_{y}\sum_{z}p_{X,Y,Z}(x,y,z)=1.$$

The marginal PMFs are

$$\begin{align}
p_{X,Y}(x,y)&=\sum_{z}p_{X,Y,Z}(x,y,z)\\
p_{X}(x)&=\sum_{y}\sum_{z}p_{X,Y,Z}(x,y,z)
\end{align}$$

and similarly for $$p_{Y,Z}(y,z)$$, $$p_{X,Z}(x,z)$$, $$p_{Y}(y)$$ and $$p_{Z}(z)$$.

### Functions of Multiple Random Variables

Now, consider a function of two random variables $$g(X,Y)$$. It is also a random variable, i.e., $$g(X,Y)=Z$$ with PMF

$$p_Z(z)=\mathbf{P}(g(X,Y)=z)=\sum_{(x,y):\,g(x,y)=z}p_{X,Y}(x,y).$$

The latter states that we have to sum the joint probability mass function over all pairs $$(x,y)$$ such that $$g(x,y)=z$$. Its expectation is

$$\mathbb{E}[g(X,Y)]=\sum_{x}\sum_{y}g(x,y)p_{X,Y}(x,y).$$

## Linearity of Expectations

Consider now the sum of $$n$$ random variables $$X_1, X_2,\ldots,X_n$$. Because expectations are linear, we have

$$\mathbb{E}[X_1+\ldots+X_n]=\mathbb{E}[X_1]+\ldots+\mathbb{E}[X_n].$$


<br>

{% include custom/series_prob_next.html %}
