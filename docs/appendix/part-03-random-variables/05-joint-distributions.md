# Joint Distributions

## Discrete Random Variables

### Joint Probability Mass Function

When we have two random variables, $X$ and $Y$, their **joint probability mass function** is defined as the probability that $X=x$ **and** $Y=y$, i.e.,

$$p_{X,Y}(x,y)=\mathbf{P}(X=x, Y=y).$$

Like ordinary probabilities, it is normalized, i.e.,

$$\sum_{x}\sum_{y}p_{X,Y}(x,y)=1.$$

#### Marginal Probability Mass Function

The probabilities $p_X(x)$ and $p_Y(y)$ are now called the **marginal PMFs** and they are calculated as

$$p_X(x)=\sum_{y}p_{X,Y}(x,y)\qquad p_Y(y)=\sum_{x}p_{X,Y}(x,y)$$

#### More than 2 Random Variables

We can generalize this concept to three or more random variables. If we have three random variables, $X$, $Y$ and $Z$, the PMF is

$$p_{X,Y,Z}(x,y,z)=\mathbf{P}(X=x,Y=y,Z=z)$$

with normalization

$$\sum_{x}\sum_{y}\sum_{z}p_{X,Y,Z}(x,y,z)=1.$$

The marginal PMFs are

$$\begin{align}
p_{X,Y}(x,y)&=\sum_{z}p_{X,Y,Z}(x,y,z)\\
p_{X}(x)&=\sum_{y}\sum_{z}p_{X,Y,Z}(x,y,z)
\end{align}$$

and similarly for $p_{Y,Z}(y,z)$, $p_{X,Z}(x,z)$, $p_{Y}(y)$ and $p_{Z}(z)$.

### Linearity of Expectations

Consider now the sum of $n$ random variables $X_1, X_2,\ldots,X_n$. Because expectations are linear, we have

$$\mathbb{E}[X_1+\ldots+X_n]=\mathbb{E}[X_1]+\ldots+\mathbb{E}[X_n].$$

## Continuous Random Variables

### Joint Probability Density Function

$$\mathbf{P}(a\leq X\leq b,c\leq Y\leq d)=\int_{c}^{d}\int_{a}^{b}f_{X,Y}(x,y)\,\mathrm{d}x\,\mathrm{d}y$$

#### Marginal Probability Density Function

$$f_X(x)=\int f_{X,Y}(x,y)\,\mathrm{d}y$$

$$f_Y(y)=\int f_{X,Y}(x,y)\,\mathrm{d}x$$

### Joint Cumulative Distribution Function

$$F_{X,Y}(x,y)=\mathbf{P}(X\leq x,Y\leq y)=\int_{-\infty}^{y}\int_{-\infty}^{x}f_{X,Y}(s,t)\,\mathrm{d}s\,\mathrm{d}t$$

$$f_{X,Y}(x,y)=\dfrac{\partial^2F_{X,Y}(x,y)}{\partial x\,\partial y}$$
