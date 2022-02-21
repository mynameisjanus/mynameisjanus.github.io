---
title: Discrete Random Variables
sidebar: probability_sidebar
permalink: randomv.html
folder: probability
usemathjax: true
toc: false
---

## Introduction to Random Variables

<p align="center">
  <img src="images/prob/rvmap.png" style="width:150px;height:auto;"/>
</p>

<br>

A **random variable** $$X$$ is a function or a mapping from the sample space $$\Omega$$ to the real numbers, i.e., it associates a real number to every possible outcome. Mathematically, we say that

$$X:\Omega\rightarrow\mathbb{R}.$$

We denote a random variable by a capital letter, e.g., $$X$$, and its numerical values by a small letter, e.g., $$x$$. A function of one or several random variables is also a random variable. For example, $$X+Y$$ is a random variable whose value is $$x+y$$ when $$X=x$$ and $$Y=y$$. There can be several random variables defined on a single sample space.

A random variable can be discrete or continuous. Consider tossing a coin 5 times. We can define a random variable as the number of heads that comes up. In this case, we have a discrete random variable. We might also consider the height of students in a university, in which case we have a continuous random variable. In this section, we will be dealing with discrete random variables.

## Discrete Random Variables

A **discrete random variable** takes ***countably*** many values. We say that a set is **countable** if we can put the elements into a one-to-one correspondence with the set of integers.

### Probability Mass Function (PMF)

The **probability mass function** $$p_X(x)$$ of a discrete random variable $$X$$ is its ***probability law*** or ***probability distribution***. It is the probability that $$X$$ will take the value $$x$$, i.e.,

$$p_X(x)=\mathbf{P}(X=x)=\mathbf{P}(\{\omega\in\Omega:X(\omega)=x\}).$$

#### Example

Consider tossing a fair coin 3 times and define the random variable $$X$$ as the number of heads. We summarize all the possible outcomes of the experiment below.

| $$\omega$$       | $$\mathbf{P}(\{\omega\})$$ | $$X(\omega)$$    |
| :--------------: |     :--------------:       | :--------------: |
| $$HHH$$ | $$\frac{1}{8}$$ | $$3$$ |
| $$HHT$$ | $$\frac{1}{8}$$ | $$2$$ |
| $$HTH$$ | $$\frac{1}{8}$$ | $$2$$ |
| $$THH$$ | $$\frac{1}{8}$$ | $$2$$ |
| $$HTT$$ | $$\frac{1}{8}$$ | $$1$$ |
| $$THT$$ | $$\frac{1}{8}$$ | $$1$$ |
| $$TTH$$ | $$\frac{1}{8}$$ | $$1$$ |
| $$TTT$$ | $$\frac{1}{8}$$ | $$0$$ |

The PMF of the random variable $$X$$ follows from the table above.

| $$x$$ | $$\mathbf{P}(X=x)$$ |
|:-----:|:-------------------:|
| $$0$$ |   $$\frac{1}{8}$$   |
| $$1$$ |   $$\frac{3}{8}$$   |
| $$2$$ |   $$\frac{3}{8}$$   |
| $$3$$ |   $$\frac{1}{8}$$   |
