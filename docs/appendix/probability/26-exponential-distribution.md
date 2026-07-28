# Exponential Distribution

## Probability Density Function

$$f_X(x)=\begin{cases}
\lambda e^{-\lambda x},&\quad x\geq0\\
0,&\quad x<0
\end{cases}$$

## Expectation

$$\mathbb{E}[X]=\frac{1}{\lambda}$$

??? note "Proof"
    $$\begin{align}
    \mathbb{E}[X]&=\int_{0}^{\infty}x\lambda e^{-\lambda x}\mathrm{d}x\\
    &=
    \end{align}$$

## Variance

$$\mathrm{var}(X)=\frac{1}{\lambda^2}$$

??? note "Proof"
    $$\begin{align}
    \mathbb{E}[X]&=\int_{0}^{\infty}x^2\lambda e^{-\lambda x}\mathrm{d}x\\
    &=
    \end{align}$$
