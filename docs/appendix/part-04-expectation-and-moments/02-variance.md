# Variance

The **variance** is a measure of the spread of a probability distribution around its mean.

## Discrete Random Variables

If $X$ is a random variable with expectation $\mathbb{E}[X]=\mu$, then its variance is defined by

$$\mathrm{var}(X)=\mathbb{E}\!\left[\left(X-\mu\right)^2\right].$$

The variance is always positive or zero, i.e., $\mathrm{var}(X)\geq0$.

From the variance, we can calculate the **standard deviation** $\sigma_X$ of a random variable:

$$\sigma_X=\sqrt{\mathrm{var}(X)}.$$

### Properties of the Variance

* The variance of a linear function of $X$ is

  $$\mathrm{var}(aX+b)=a^2\mathrm{var}(X).$$

??? note "Proof"
       $$\begin{align}
      \mathrm{var}(aX+b)&=\mathbb{E}\!\left[\left(aX+b-\mathbb{E}[aX+b]\right)^2\right]\\
      &=\mathbb{E}\!\left[(aX+b-a\,\mathbb{E}[X]-b)^2\right]\\
      &=\mathbb{E}\!\left[a^2(X-\mathbb{E}[X])^2\right]\\
      &=a^2\,\mathbb{E}[(X-\mathbb{E}[X])^2]\\
      &=a^2\mathrm{var}(X)
      \end{align}$$

* A useful formula for variance is

  $$\mathrm{var}(X)=\mathbb{E}[X^2]-\left(\mathbb{E}[X]\right)^2.$$

??? note "Proof"
      $$\begin{align}
      \mathrm{var}(X)&=\mathbb{E}[(X-\mu)^2]\\
      &=\mathbb{E}[X^2-2\mu X+\mu^2]\\
      &=\mathbb{E}[X^2]-2\mu\,\mathbb{E}[X]+\mu^2\\
      &=\mathbb{E}[X^2]-2(\mathbb{E}[X])^2+(\mathbb{E}[X])^2\\
      &=\mathbb{E}[X^2]-(\mathbb{E}[X])^2
      \end{align}$$

## Continuous Random Variables

Just like in the discrete case, the variance is defined as

$$\mathrm{var}(X)=\mathbb{E}[(x-\mu)^2]$$

where $\mu=\mathbb{E}[X]$. However, the explicit formula for calculating it is

$$\mathrm{var}(X)=\int_{-\infty}^{\infty}(x-\mu)^2f_X(x)\,\mathrm{d}x.$$

## Standardized Random Variables

Let $X$ have mean $\mu$ and variance $\sigma^2$. Also, let

$$Y=\frac{X-\mu}{\sigma}.$$

Then,

$$\mathbb{E}[Y]=0\qquad\text{and}\qquad\mathrm{var}(Y)=1,$$

i.e., $Y$ is a standard random variable. If $X$ is a normal random variable, then $Y\sim\mathcal{N}(0,1)$, i.e., $Y$ is the **standard normal random variable**.
