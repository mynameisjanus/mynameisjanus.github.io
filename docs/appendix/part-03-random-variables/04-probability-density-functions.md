# Probability Density Functions

A continuous random variable $X$ is described by a **probability density function** $f_X(x)$. In this case, however, the probability that it takes a single value, say $a$, is zero, i.e.,

$$\mathbf{P}(X=a)=0.$$

Instead, we can take the probability that it is within a certain ***range***, say between $a$ and $b$. This is defined by

$$\mathbf{P}(a\leq X\leq b)=\int_{a}^{b}f_X(x)\,\mathrm{d}x.$$

## Non-negativity and Normalization

Like in the discrete case, the probability density function has to be **non-negative** and **normalized**, i.e.,

$$f_X(x)\geq 0\qquad\text{and}\qquad\int_{-\infty}^{\infty}f_X(x)\,\mathrm{d}x=1.$$
