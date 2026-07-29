# Cumulative Distribution Functions

The **cumulative distribution function** is defined as the probability that a random variable $X$ is less than $x$, i.e.,

$$F_X(x)=\mathbf{P}(X\leq x)=\int_{-\infty}^{x}f_X(t)\,\mathrm{d}t.$$

Using the fundamental theorem of calculus, we can calculate the probability density function from the cumulative distribution function, i.e.,

$$f_X(x)=\dfrac{\mathrm{d}F_X(x)}{\mathrm{d}x}.$$

## Properties of the CDF

* It is **non-decreasing**, i.e., if $y\geq x$, then $F_X(y)\geq F_X(x)$.
* As $x\rightarrow -\infty$, $F_X(x)\rightarrow 0$.
* As $x\rightarrow +\infty$, $F_X(x)\rightarrow 1$.
