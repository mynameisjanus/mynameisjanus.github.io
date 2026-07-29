# Expected Value

The **expectation** or **mean** of a random variable summarizes its distribution by a single number — its center of mass.

## Discrete Random Variables

The expectation of a discrete random variable is defined as

$$\mathbb{E}[X]=\sum_{x}xp_X(x).$$

### Properties of Expectations

* If $X\geq0$, then $\mathbb{E}[X]\geq0$.
* If $a\leq X\leq b$, then $a\leq\mathbb{E}[X]\leq b$.
* If $c$ is a constant, then $\mathbb{E}[c]=c$.
* Expectation of $g(X)$

  $$\mathbb{E}[g(X)]=\sum_{x}g(x)p_X(x)$$

!!! warning
    In general, $\mathbb{E}\!\left[g(X)\right]\neq g\left(\mathbb{E}[X]\right)$.

* Linearity

  $$\mathbb{E}[aX+b]=a\,\mathbb{E}[X]+b$$

## Continuous Random Variables

Instead of taking sums, we use integration to calculate the expectation of a continuous random variable, i.e.,

$$\mathbb{E}[X]=\int_{-\infty}^{\infty}xf_X(x)\,\mathrm{d}x.$$

### Properties of the Expectation

* If $X\geq 0$, then $\mathbb{E}[X]\geq 0$.
* If $a\leq X\leq b$, then $a\leq\mathbb{E}[X]\leq b$.
* Given a function $g(X)$,

  $$\mathbb{E}[g(X)]=\int_{-\infty}^{\infty}g(x)f_X(x)\,\mathrm{d}x.$$

* Linearity

  $$\mathbb{E}[aX+b]=a\,\mathbb{E}[X]+b$$
